// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU byte oracle of the gfx1201 BF16 lane-row PM twins, module
//! `lane_rows_bf16_pm_gfx1201`: ordinary `x4` and HC `k4` narrow schedules, plus fixed-eight-row
//! physical tiles at every total logical row count 1..=64.
//!
//! Every launch runs on the peacemaker CPU interpreter (emit -> `native::assemble` -> `lift_object(Builder)` ->
//! `Emulator::run_workgroup` per workgroup, real barriers, LDS, `ds_swizzle`/`ds_bpermute`). Every output word of
//! every entry, at every live `C` in its range, is compared with BOTH
//!
//! 1. the word of the **independent singleton reference**: this file emits, with the checked builder, a straight-line
//!    transcription of the hipcc gfx1201 singleton DAG (`gemv_bf16_xf32` row body / `gemv_bf16_xf32_k4` quarters and
//!    fold; see `emit_ref_dot` and `emit_ref_fold` below) and runs it on the same
//!    interpreter, one launch per activation row (`C = 1`, fully unrolled over the compile-time `K`, no shared code
//!    with the PM emitter): per-lane chain assignment, single-rounded `v_fmac_f32 acc, x, w` order, the
//!    `(a3 + a2) + (a1 + a0)` fold, the `ds_swizzle` `lane | 16` add and the `ds_bpermute` `+8/+4/+2/+1` own+other
//!    tree, the HC quarter fold `(q0 + q1) + (q2 + q3)` and, for `hc_act`, the VALU epilogue (owner ruling; the
//!    object's SALU float unit versus this VALU expansion on exceptional inputs is H0's claim, not this oracle's);
//! 2. the word the PM `c1` entry produces at `C = 1` for the same row alone.
//!
//! There is no host-side numerical model anywhere in this file.
//!
//! The lifted hipcc object is a third reference, gated independently per family: the ordinary singleton
//! (`gemv_bf16_xf32`) is compared in every launch check whenever the interpreter can execute it, the HC singleton
//! (`gemv_bf16_xf32_k4`) likewise; a family that cannot execute is SKIPPED with its exact reason (logged by every
//! test and by `reference_identity_and_hipcc_object_status`) and never disables the other. That test also proves
//! emitted-DAG == hipcc object word for word for each executable family. `LANE_ROWS_BF16_REQUIRE_HIPCC=1` turns an
//! ordinary skip into a failure, `=all` also an HC skip.
//!
//! Also checked per launch: the same row in every slot, rotations/reversals of the slot order, zero- and
//! odd-height x4 slots in every position (null pointers for zero-height slots on odd cases), out-of-range `C`
//! writes nothing, guard/padding words are untouched, the exact written byte set equals the logical output extent,
//! weights and activations are byte-identical after every launch. Structural tests: frozen record, ABI and
//! resources, no atomics / scratch / spills, widening independent of capacity, pinned singleton objects, M7.
//!
//! Environment: `LANE_ROWS_BF16_FULL=1` widens the shape matrix; `LANE_ROWS_BF16_LOG=<file>` appends every log line
//! to a file (all lines also go to stdout; run with `--nocapture`).
//!
//! Missing actual singleton objects are reported as missing evidence, never replaced by the emitted DAG.
//! `LANE_ROWS_BF16_FIXTURE_DIR` may supply the pinned objects without carrying generated images or provenance docs.
//! HARDWARE GAP: CPU comparisons do not qualify HC SALU/VALU exceptional-domain equivalence. Hardware HC
//! extreme-input byte checks remain mandatory, including the explicitly counted interpreter refusal domains.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::{Abi, Frontend};
use peacemaker_ir::Program;
use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::kernels::common::{
    add64, bload, lit, mem as mem_op, op, s_add_u32, s_addc_u32, smem, sop, sr, v, SRD_WORD3,
};
use peacemaker_isa::kernels::lane_rows_bf16_gfx1201::{
    self as lr, GRID_RULE, KERNARGS, KERNARG_BYTES, LDS_BYTES, MAX_ROWS, MODULE, RECIPE, SYMBOLS,
    WORKGROUP_SIZE,
};
use peacemaker_isa::ledger::Counter;
use peacemaker_isa::plan::Access;
use peacemaker_isa::reg::{Kind, Live};
use peacemaker_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const ARCH: Arch = Arch::Gfx1201;

const EXPECTED_SYMBOLS: [&str; 10] = [
    "gemv_bf16_xf32_x4_c1_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201",
    "gemv_bf16_xf32_k4_c1_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201",
];
const EXPECTED_MAX_ROWS: [usize; 10] = [1, 4, 8, 1, 4, 8, 4, 8, 4, 8];
const RECIPE_X4: &str = "bf16.gfx1201.x4.singleton-dag.v1";
const RECIPE_K4: &str = "bf16.gfx1201.k4.singleton-dag.v1";

/// Pinned identities of the original uncompressed hipcc singleton bundles.
const FIXTURE_QWEN4: (&str, &str) = (
    "qwen4_gemv_bf16_xf32.gfx1201.hsaco",
    "a610695fa6df3aaaf2771d98416fcfe1e229948b8ccae868bf03eed001850325",
);
const FIXTURE_SINGLE: (&str, &str) = (
    "gemv_bf16_xf32.gfx1201.hsaco",
    "a94c1c43f95811c81213a42a4d8f4489fbbc3ec0639e4a5837b3298eaf219601",
);
const SINGLE_SYMBOL: &str = "gemv_bf16_xf32";
const K4_SYMBOL: &str = "gemv_bf16_xf32_k4";

type R = Result<(), String>;

// ------------------------------------------------------------------------------------------------ logging

fn log(line: impl AsRef<str>) {
    use std::io::Write;
    println!("{}", line.as_ref());
    if let Ok(path) = std::env::var("LANE_ROWS_BF16_LOG") {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", line.as_ref());
        }
    }
}

fn full() -> bool {
    std::env::var("LANE_ROWS_BF16_FULL").is_ok_and(|v| v == "1")
}

// ------------------------------------------------------------------------------------------------ entries

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fam {
    X4,
    K4,
}

#[derive(Clone, Debug)]
struct Entry {
    idx: usize,
    symbol: &'static str,
    fam: Fam,
    cap: usize,
    wg8: bool,
    tiled: bool,
}

fn entries() -> Vec<Entry> {
    SYMBOLS
        .iter()
        .enumerate()
        .map(|(idx, &symbol)| {
            let fam = if symbol.contains("_x4_") {
                Fam::X4
            } else if symbol.contains("_k4_") {
                Fam::K4
            } else {
                panic!("{symbol}: neither x4 nor k4")
            };
            let cap = if symbol.contains("_c1_") {
                1
            } else if symbol.contains("_r4_") {
                4
            } else if symbol.contains("_r8_") {
                8
            } else {
                panic!("{symbol}: no capacity class")
            };
            Entry {
                idx,
                symbol,
                fam,
                cap,
                wg8: symbol.contains("_wg8_"),
                tiled: false,
            }
        })
        .collect()
}

impl Entry {
    /// Workgroup size derived from the contract (not read from the record under test).
    fn block(&self) -> u32 {
        match (self.fam, self.wg8) {
            (Fam::X4, false) => 32,
            (Fam::K4, false) => 128,
            (_, true) => 256,
        }
    }
    fn base(&self) -> &'static str {
        match self.fam {
            Fam::X4 => RECIPE_X4,
            Fam::K4 => RECIPE_K4,
        }
    }
}

fn rowtile_entries() -> Vec<Entry> {
    let mut es: Vec<Entry> = entries().into_iter().filter(|e| e.cap == 1).collect();
    for (i, fam) in [Fam::X4, Fam::K4].into_iter().enumerate() {
        es.push(Entry {
            idx: 0,
            symbol: lr::ROWTILE_SYMBOLS[i],
            fam,
            cap: 64,
            wg8: false,
            tiled: true,
        });
    }
    for (idx, e) in es.iter_mut().enumerate() {
        e.idx = idx;
    }
    es
}

// ------------------------------------------------------------------------------------------------ rng / data

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// BF16 weights: wide random normals, signed zeros, rounding-tie mantissas, +-1/2/0.5, bf16 subnormals.
fn gen_w(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let sign = ((rng.next() & 1) as u16) << 15;
            match rng.below(16) {
                0 => sign,
                1 => {
                    let e = 120 + rng.below(14) as u16;
                    let m = [0u16, 0x40, 0x20, 0x60, 0x10, 0x01][rng.below(6)];
                    sign | (e << 7) | m
                }
                2 => sign | ([0x3f80u16, 0x4000, 0x3f00][rng.below(3)] & 0x7fff),
                3 => sign | (1 + rng.below(0x7f) as u16),
                _ => sign | ((110 + rng.below(36) as u16) << 7) | (rng.next() as u16 & 0x7f),
            }
        })
        .collect()
}

fn rnd_wide(rng: &mut Rng, shift: i32) -> f32 {
    let e = 127 + shift + rng.below(41) as i32 - 20;
    f32::from_bits(
        (((rng.next() & 1) as u32) << 31) | ((e as u32) << 23) | (rng.next() as u32 & 0x7f_ffff),
    )
}

#[derive(Clone, Copy)]
enum Kind_ {
    Wide(i32),
    Cancel,
    Tie,
    Zeros,
    Mixed,
}

/// Full-mantissa random F32 over a wide exponent band; cancellation-heavy; RNE-tie-heavy; zero-heavy; mixed.
fn gen_row(rng: &mut Rng, k: usize, kind: Kind_) -> Vec<f32> {
    const TIES: [f32; 8] = [16777216.0, 12582912.0, 1.0, 0.5, 3.0, 5.0, 0.25, 8388608.0];
    (0..k)
        .map(|i| match kind {
            Kind_::Wide(shift) => {
                if rng.below(16) == 0 {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    rnd_wide(rng, shift)
                }
            }
            Kind_::Cancel => {
                if i % 7 == 6 {
                    rnd_wide(rng, -20)
                } else {
                    let big = f32::from_bits((149 << 23) | (rng.next() as u32 & 0x7f_ffff));
                    if i % 2 == 0 {
                        big
                    } else {
                        -big
                    }
                }
            }
            Kind_::Tie => {
                let t = TIES[rng.below(8)];
                if rng.below(2) == 0 {
                    t
                } else {
                    -t
                }
            }
            Kind_::Zeros => match rng.below(12) {
                0 => rnd_wide(rng, 0),
                n if n % 2 == 0 => 0.0,
                _ => -0.0,
            },
            Kind_::Mixed => match rng.below(5) {
                0 => rnd_wide(rng, 10),
                1 => rnd_wide(rng, -40),
                2 => TIES[rng.below(8)],
                3 => -0.0,
                _ => f32::from_bits(rng.next() as u32 & 0x007f_ffff),
            },
        })
        .collect()
}

const ROW_KINDS: [Kind_; 8] = [
    Kind_::Wide(0),
    Kind_::Wide(-50),
    Kind_::Wide(55),
    Kind_::Cancel,
    Kind_::Tie,
    Kind_::Zeros,
    Kind_::Mixed,
    Kind_::Wide(20),
];

/// One problem: up to four matrices sharing `k`, eight activation rows, an HC epilogue setting.
#[derive(Clone)]
struct Problem {
    label: String,
    fam: Fam,
    ms: [usize; 4],
    k: usize,
    w: [Vec<u16>; 4],
    rows: Vec<Vec<f32>>,
    scale: f32,
    act: bool,
}

impl Problem {
    fn matrices(&self) -> usize {
        match self.fam {
            Fam::X4 => 4,
            Fam::K4 => 1,
        }
    }
    /// 1-D grid derived from the contract rule of the entry (see `GRID_RULE`).
    fn grid(&self, e: &Entry) -> u32 {
        (match (e.fam, e.wg8) {
            (Fam::X4, false) => self.ms.iter().sum::<usize>(),
            (Fam::X4, true) => self.ms.iter().map(|m| m.div_ceil(8)).sum(),
            (Fam::K4, false) => self.ms[0],
            (Fam::K4, true) => self.ms[0].div_ceil(2),
        }) as u32
    }
}

fn make_x4(seed: u64, ms: [usize; 4], k: usize, nonfinite: bool) -> Problem {
    let mut rng = Rng::new(seed);
    let w: [Vec<u16>; 4] = std::array::from_fn(|i| {
        let mut w = gen_w(&mut rng, ms[i] * k);
        if nonfinite && !w.is_empty() {
            for _ in 0..1 + w.len() / 61 {
                let at = rng.below(w.len());
                w[at] = [0x7fc0u16, 0x7f80, 0xff80, 0xffc1][rng.below(4)];
            }
        }
        w
    });
    let rows = make_rows(&mut rng, k, seed, nonfinite);
    Problem {
        label: format!("x4 ms={ms:?} K={k} seed={seed}"),
        fam: Fam::X4,
        ms,
        k,
        w,
        rows,
        scale: 1.0,
        act: false,
    }
}

fn make_hc(seed: u64, m: usize, k: usize, act: bool, scale: f32, nonfinite: bool) -> Problem {
    let mut rng = Rng::new(seed);
    let mut w0 = gen_w(&mut rng, m * k);
    if nonfinite && !w0.is_empty() {
        for _ in 0..1 + w0.len() / 97 {
            let at = rng.below(w0.len());
            w0[at] = [0x7fc0u16, 0x7f80, 0xff80][rng.below(3)];
        }
    }
    let rows = make_rows(&mut rng, k, seed, nonfinite);
    Problem {
        label: format!(
            "k4 M={m} K={k} act={} scale={scale:e} seed={seed}",
            if act { "Some" } else { "None" }
        ),
        fam: Fam::K4,
        ms: [m, 0, 0, 0],
        k,
        w: [w0, Vec::new(), Vec::new(), Vec::new()],
        rows,
        scale,
        act,
    }
}

fn make_rows(rng: &mut Rng, k: usize, seed: u64, nonfinite: bool) -> Vec<Vec<f32>> {
    (0..8)
        .map(|r| {
            let mut row = gen_row(rng, k, ROW_KINDS[(r + seed as usize) % 8]);
            if nonfinite && k > 0 {
                for j in 0..1 + rng.below(3) {
                    let at = rng.below(k);
                    row[at] = [
                        f32::from_bits(0x7fc0_1234),
                        f32::from_bits(0xffc0_0001),
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                    ][(rng.below(4) + j) % 4];
                }
            }
            row
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------ independent singleton reference
//
// Checked-builder transcription of the pinned hipcc gfx1201 object (`qwen4_gemv_bf16_xf32.gfx1201.hsaco`),
// fully unrolled over the compile-time K (so no loop, label or divergent branch), one workgroup = one wave = one
// output. Out-of-range elements are masked by the raw buffer descriptor (records = K*2 / K*4 bytes: both the weight
// and the activation word read 0 and `fma(0, 0, acc) == acc` exactly because an accumulator that starts at +0 can never
// be -0), so no exec arithmetic is needed. Only instructions the interpreter has semantics rows for are used.

const MASK0: u8 = 48;
const MASK1: u8 = 50;

fn ref_spec(symbol: &str, variant: &str, kernargs: KernargLayout) -> KernelSpec {
    KernelSpec {
        kernel_id: "bf16_singleton_ref".into(),
        variant: variant.into(),
        arch: ARCH,
        symbol: symbol.into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

fn ref_plan(vgprs: u8, sgpr_end: u8) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(vgprs), 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    for base in (8u8..sgpr_end).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..vgprs).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

fn alu_wait(b: &mut Builder, which: &str) -> R {
    b.push(Instruction::new(
        format!("s_wait_alu {which}"),
        vec![],
        vec![],
    ))
}

/// `gemv_bf16_xf32_row` for one output row of `k` elements: kernarg `W@0 x@8 y@16 row_bytes@24`, grid `[M]`, block 32.
/// Row `wg` of `W` starts `wg * row_bytes` after `W`; `y[wg]` receives the lane-0 sum.
///
/// `K % 8 == 0`: lane `l` owns elements `256 g + 8 l + j`, chain `j & 3` (`v_fmac acc, x_j, w_j`, j ascending, so chain
/// `a0` gets `j0` then `j4`, ...), groups ascending; else all-scalar: element `32 t + l` into chain `a0`. Then
/// `(a3 + a2) + (a1 + a0)`, `ds_swizzle` partner `l | 16` own+other, `ds_bpermute` `+8 +4 +2 +1` own+other
/// (a lane reads itself when `l + d` would leave the wave); lane 0 stores.
fn emit_ref_dot(k: usize) -> Result<Emitted, String> {
    let kernargs = KernargLayout::new(32)
        .pointer_access("W", 0, Access::ReadOnly)
        .pointer_access("x", 8, Access::ReadOnly)
        .pointer_access("y", 16, Access::WriteOnly)
        .hidden("row_bytes", 24, 4, "by_value");
    let mut bld = Builder::new(
        ref_spec(&format!("ref_dot_k{k}"), &format!("k{k}"), kernargs),
        ref_plan(32, 40)?,
    );
    bld.enable_delay_alu();
    let b = &mut bld;
    smem(b, 8, 4, 0, 0)?; // W -> s8:9, x -> s10:11
    smem(b, 12, 2, 0, 16)?; // y -> s12:13
    smem(b, 14, 1, 0, 24)?; // row_bytes -> s14
    b.wait_all()?;
    sop(b, "s_mov_b32 s24, ttmp9", &[24], &[])?;
    sop(b, "s_mul_i32 s26, s24, s14", &[26], &[24, 14])?;
    sop(b, "s_mul_hi_u32 s27, s24, s14", &[27], &[24, 14])?;
    sop(
        b,
        format!("{} s8, s8, s26", s_add_u32(ARCH)),
        &[8],
        &[8, 26],
    )?;
    sop(
        b,
        format!("{} s9, s9, s27", s_addc_u32(ARCH)),
        &[9],
        &[9, 27],
    )?;
    sop(b, "s_lshl_b32 s25, s24, 2", &[25], &[24])?;
    add64(b, 12, 12, 25)?;
    // Raw descriptors: W (records = 2K bytes) at s16:19, x (records = 4K bytes) at s20:23.
    sop(b, "s_mov_b32 s16, s8", &[16], &[8])?;
    sop(b, "s_mov_b32 s17, s9", &[17], &[9])?;
    sop(
        b,
        format!("s_mov_b32 s18, {}", lit(2 * k as u32)),
        &[18],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s19, {}", lit(SRD_WORD3)), &[19], &[])?;
    sop(b, "s_mov_b32 s20, s10", &[20], &[10])?;
    sop(b, "s_mov_b32 s21, s11", &[21], &[11])?;
    sop(
        b,
        format!("s_mov_b32 s22, {}", lit(4 * k as u32)),
        &[22],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s23, {}", lit(SRD_WORD3)), &[23], &[])?;
    // Lane-mask pairs of the tree (the high words are never written in wave32).
    sop(b, "s_mov_b32 s33, 0", &[33], &[])?;
    // Accumulators a0..a3 = v8..v11.
    for a in 8..12u8 {
        op(b, format!("v_mov_b32_e32 v{a}, 0"), &[v(a)], &[])?;
    }
    let vector = k.is_multiple_of(8);
    if vector {
        op(b, "v_lshlrev_b32_e32 v1, 4, v0", &[v(1)], &[v(0)])?; // W byte offset of the lane's 8 elements
        op(b, "v_lshlrev_b32_e32 v2, 5, v0", &[v(2)], &[v(0)])?; // x byte offset
        for g in 0..k.div_ceil(256) {
            let (wv, xv) = if g == 0 {
                (1u8, 2u8)
            } else {
                op(
                    b,
                    format!("v_add_nc_u32_e32 v3, {}, v1", lit(512 * g as u32)),
                    &[v(3)],
                    &[v(1)],
                )?;
                op(
                    b,
                    format!("v_add_nc_u32_e32 v4, {}, v2", lit(1024 * g as u32)),
                    &[v(4)],
                    &[v(2)],
                )?;
                (3u8, 4u8)
            };
            bload(b, 4, 16, wv, 16, 0)?; // raw bf16 pairs -> v16..v19
            bload(b, 4, 24, xv, 20, 0)?; // x[0..4) -> v24..v27
            bload(b, 4, 28, xv, 20, 16)?; // x[4..8) -> v28..v31
            b.wait(Counter::Load, 0)?;
            // bf16 -> f32 is bitwise: even element `u << 16`, odd element `u & 0xffff0000`; dword 3 first, in place.
            for d in (0..4u8).rev() {
                op(
                    b,
                    format!("v_and_b32_e32 v{}, 0xffff0000, v{}", 16 + 2 * d + 1, 16 + d),
                    &[v(16 + 2 * d + 1)],
                    &[v(16 + d)],
                )?;
                op(
                    b,
                    format!("v_lshlrev_b32_e32 v{}, 16, v{}", 16 + 2 * d, 16 + d),
                    &[v(16 + 2 * d)],
                    &[v(16 + d)],
                )?;
            }
            for j in 0..8u8 {
                let acc = 8 + (j & 3);
                op(
                    b,
                    format!("v_fmac_f32_e32 v{acc}, v{}, v{}", 24 + j, 16 + j),
                    &[v(acc)],
                    &[v(acc), v(24 + j), v(16 + j)],
                )?;
            }
        }
    } else {
        op(b, "v_lshlrev_b32_e32 v1, 1, v0", &[v(1)], &[v(0)])?;
        op(b, "v_lshlrev_b32_e32 v2, 2, v0", &[v(2)], &[v(0)])?;
        for t in 0..k.div_ceil(32) {
            let (wv, xv) = if t == 0 {
                (1u8, 2u8)
            } else {
                op(
                    b,
                    format!("v_add_nc_u32_e32 v3, {}, v1", lit(64 * t as u32)),
                    &[v(3)],
                    &[v(1)],
                )?;
                op(
                    b,
                    format!("v_add_nc_u32_e32 v4, {}, v2", lit(128 * t as u32)),
                    &[v(4)],
                    &[v(2)],
                )?;
                (3u8, 4u8)
            };
            mem_op(
                b,
                format!("buffer_load_u16 v16, v{wv}, s[16:19], null offen"),
                &[v(16)],
                &[v(wv), sr(16, 4)],
                MemoryClass::VmemLoad,
            )?;
            bload(b, 1, 24, xv, 20, 0)?;
            b.wait(Counter::Load, 0)?;
            op(b, "v_lshlrev_b32_e32 v16, 16, v16", &[v(16)], &[v(16)])?;
            op(
                b,
                "v_fmac_f32_e32 v8, v24, v16",
                &[v(8)],
                &[v(8), v(24), v(16)],
            )?;
        }
    }
    // (a3 + a2) + (a1 + a0)
    op(b, "v_add_f32_e32 v5, v11, v10", &[v(5)], &[v(11), v(10)])?;
    op(b, "v_add_f32_e32 v6, v9, v8", &[v(6)], &[v(9), v(8)])?;
    op(b, "v_add_f32_e32 v7, v5, v6", &[v(7)], &[v(5), v(6)])?;
    // ds_swizzle partner `l | 16` (and_mask 0x0f, or_mask 0x10), own + other.
    b.ds_crosslane(
        Instruction::new(
            "ds_swizzle_b32 v12, v7 offset:swizzle(BITMASK_PERM,\"1pppp\")",
            vec![v(12)],
            vec![v(7)],
        )
        .memory(MemoryClass::DsLoad),
    )?;
    b.wait(Counter::Ds, 0)?;
    op(b, "v_add_f32_e32 v7, v7, v12", &[v(7)], &[v(7), v(12)])?;
    for d in [8u32, 4, 2, 1] {
        op(
            b,
            format!("v_cmp_gt_u32_e64 s32, {}, v0", 32 - d),
            &[sr(32, 2)],
            &[v(0)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v13, 0, {d}, s32"),
            &[v(13)],
            &[sr(32, 2)],
        )?;
        op(
            b,
            "v_add_lshl_u32 v13, v13, v0, 2",
            &[v(13)],
            &[v(13), v(0)],
        )?;
        b.ds_crosslane(
            Instruction::new(
                "ds_bpermute_b32 v12, v13, v7",
                vec![v(12)],
                vec![v(13), v(7)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.wait(Counter::Ds, 0)?;
        op(b, "v_add_f32_e32 v7, v7, v12", &[v(7)], &[v(7), v(12)])?;
    }
    // Lane 0 stores y[wg] (s12:13 already advanced by 4 * wg).
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    op(b, "v_mov_b32_e32 v14, 0", &[v(14)], &[])?;
    mem_op(
        b,
        "global_store_b32 v14, v7, s[12:13]",
        &[],
        &[v(14), v(7), sr(12, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(b, "s_endpgm", &[], &[])?;
    bld.finish()
}

/// BF16 round trip of the wave-uniform f32 in `v{val}` on three scratch VGPRs (finite: `(u + ((u >> 16) & 1) + 0x7fff) &
/// 0xffff0000`; exponent field all ones: `low16(u) == 0 ? u : u | 0x10000`, then the mask) on the VALU.
fn ref_round_trip(b: &mut Builder, val: u8, t: [u8; 3]) -> R {
    let [t0, t1, t2] = t;
    op(
        b,
        format!("v_bfe_u32 v{t0}, v{val}, 16, 1"),
        &[v(t0)],
        &[v(val)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{t0}, v{val}, v{t0}"),
        &[v(t0)],
        &[v(val), v(t0)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{t0}, 0x7fff, v{t0}"),
        &[v(t0)],
        &[v(t0)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{t1}, 0xffff, v{val}"),
        &[v(t1)],
        &[v(val)],
    )?;
    op(
        b,
        format!("v_or_b32_e32 v{t2}, 0x10000, v{val}"),
        &[v(t2)],
        &[v(val)],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MASK0}, 0, v{t1}"),
        &[sr(MASK0, 2)],
        &[v(t1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t1}, v{t2}, v{val}, s{MASK0}"),
        &[v(t1)],
        &[v(t2), v(val), sr(MASK0, 2)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{t2}, 0x7f800000, v{val}"),
        &[v(t2)],
        &[v(val)],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MASK1}, 0x7f800000, v{t2}"),
        &[sr(MASK1, 2)],
        &[v(t2)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t0}, v{t0}, v{t1}, s{MASK1}"),
        &[v(t0)],
        &[v(t0), v(t1), sr(MASK1, 2)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{val}, 0xffff0000, v{t0}"),
        &[v(val)],
        &[v(t0)],
    )
}

/// Quarter fold `(q0 + q1) + (q2 + q3)` of `gemv_bf16_xf32_k4` and, for `act`, its epilogue on the VALU:
/// `x = rt(sum); x = rt(scale * x); y = x / (1 + exp(-x)); rt(y)`; `exp(-x)` with the object's range reduction
/// (`s1 = x * -log2e; s4 = fma(x, -log2e, -s1); s4 = fma(x, 0xb2a5705f, s4); s5 = rndne(s1); s1 = (s1 - s5) + s4;
/// ldexp(v_exp_f32(s1), cvt_i32(s5))`), the `ngt`/`nlt` clamps (unordered true) and the full
/// `v_div_scale / v_rcp / v_fma / v_div_fmas / v_div_fixup` expansion.
///
/// Kernarg `q0..q3@0,8,16,24` (pointers to `[M]` quarter sums), `y@32`, `scale@40`; grid `[M]`, block 32.
fn emit_ref_fold(act: bool) -> Result<Emitted, String> {
    let kernargs = KernargLayout::new(48)
        .pointer_access("q0", 0, Access::ReadOnly)
        .pointer_access("q1", 8, Access::ReadOnly)
        .pointer_access("q2", 16, Access::ReadOnly)
        .pointer_access("q3", 24, Access::ReadOnly)
        .pointer_access("y", 32, Access::WriteOnly)
        .hidden("scale", 40, 4, "by_value");
    let name = if act { "act" } else { "none" };
    let mut bld = Builder::new(
        ref_spec(
            &format!("ref_fold_{name}"),
            &format!("fold_{name}"),
            kernargs,
        ),
        ref_plan(24, 56)?,
    );
    bld.enable_delay_alu();
    let b = &mut bld;
    smem(b, 8, 8, 0, 0)?; // q0..q3 -> s8..s15
    smem(b, 16, 2, 0, 32)?; // y -> s16:17
    smem(b, 18, 1, 0, 40)?; // scale -> s18
    b.wait_all()?;
    sop(b, "s_mov_b32 s24, ttmp9", &[24], &[])?;
    sop(b, "s_lshl_b32 s25, s24, 2", &[25], &[24])?;
    for i in 0..4u8 {
        add64(b, 32 + 2 * i, 8 + 2 * i, 25)?;
        smem(b, 40 + i, 1, 32 + 2 * i, 0)?;
    }
    add64(b, 16, 16, 25)?;
    b.wait_all()?;
    sop(b, "s_mov_b32 s49, 0", &[49], &[])?;
    sop(b, "s_mov_b32 s51, 0", &[51], &[])?;
    for i in 0..4u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, s{}", 1 + i, 40 + i),
            &[v(1 + i)],
            &[sr(40 + i, 1)],
        )?;
    }
    const OUT: u8 = 7;
    const T: [u8; 6] = [8, 9, 10, 11, 12, 13];
    op(b, "v_add_f32_e32 v5, v1, v2", &[v(5)], &[v(1), v(2)])?;
    op(b, "v_add_f32_e32 v6, v3, v4", &[v(6)], &[v(3), v(4)])?;
    op(
        b,
        format!("v_add_f32_e32 v{OUT}, v5, v6"),
        &[v(OUT)],
        &[v(5), v(6)],
    )?;
    if act {
        let [t0, t1, t2, t3, t4, t5] = T;
        ref_round_trip(b, OUT, [t0, t1, t2])?;
        op(
            b,
            format!("v_mul_f32_e32 v{OUT}, s18, v{OUT}"),
            &[v(OUT)],
            &[v(OUT), sr(18, 1)],
        )?;
        ref_round_trip(b, OUT, [t0, t1, t2])?;
        op(
            b,
            format!("v_mul_f32_e32 v{t0}, 0xbfb8aa3b, v{OUT}"),
            &[v(t0)],
            &[v(OUT)],
        )?;
        op(
            b,
            format!("v_xor_b32_e32 v{t1}, 0x80000000, v{t0}"),
            &[v(t1)],
            &[v(t0)],
        )?;
        op(
            b,
            format!("v_rndne_f32_e32 v{t2}, v{t0}"),
            &[v(t2)],
            &[v(t0)],
        )?;
        op(
            b,
            format!("v_fma_f32 v{t1}, v{OUT}, 0xbfb8aa3b, v{t1}"),
            &[v(t1)],
            &[v(OUT), v(t1)],
        )?;
        op(
            b,
            format!("v_sub_f32_e32 v{t0}, v{t0}, v{t2}"),
            &[v(t0)],
            &[v(t0), v(t2)],
        )?;
        op(
            b,
            format!("v_fma_f32 v{t1}, v{OUT}, 0xb2a5705f, v{t1}"),
            &[v(t1)],
            &[v(OUT), v(t1)],
        )?;
        op(
            b,
            format!("v_add_f32_e32 v{t0}, v{t0}, v{t1}"),
            &[v(t0)],
            &[v(t0), v(t1)],
        )?;
        op(
            b,
            format!("v_cvt_i32_f32_e32 v{t3}, v{t2}"),
            &[v(t3)],
            &[v(t2)],
        )?;
        op(b, format!("v_exp_f32_e32 v{t0}, v{t0}"), &[v(t0)], &[v(t0)])?;
        op(
            b,
            format!("v_ldexp_f32 v{t0}, v{t0}, v{t3}"),
            &[v(t0)],
            &[v(t0), v(t3)],
        )?;
        // keep e when !(x > 0x42ce8ed0); +Inf when x < 0xc2b17218 (both unordered-true).
        op(
            b,
            format!("v_cmp_nlt_f32_e64 s{MASK0}, 0x42ce8ed0, v{OUT}"),
            &[sr(MASK0, 2)],
            &[v(OUT)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{t0}, 0, v{t0}, s{MASK0}"),
            &[v(t0)],
            &[v(t0), sr(MASK0, 2)],
        )?;
        op(
            b,
            format!("v_cmp_ngt_f32_e64 s{MASK1}, 0xc2b17218, v{OUT}"),
            &[sr(MASK1, 2)],
            &[v(OUT)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{t0}, 0x7f800000, v{t0}, s{MASK1}"),
            &[v(t0)],
            &[v(t0), sr(MASK1, 2)],
        )?;
        // y = x / (1 + exp(-x)).
        let [w0, w1, w2, w3, w4, w5] = [t0, t1, t2, t3, t4, t5];
        op(
            b,
            format!("v_add_f32_e32 v{w0}, 1.0, v{w0}"),
            &[v(w0)],
            &[v(w0)],
        )?;
        op(
            b,
            format!("v_div_scale_f32 v{w1}, null, v{w0}, v{w0}, v{OUT}"),
            &[v(w1)],
            &[v(w0), v(OUT)],
        )?;
        op(b, format!("v_rcp_f32_e32 v{w2}, v{w1}"), &[v(w2)], &[v(w1)])?;
        op(
            b,
            format!("v_xor_b32_e32 v{w1}, 0x80000000, v{w1}"),
            &[v(w1)],
            &[v(w1)],
        )?;
        op(
            b,
            format!("v_fma_f32 v{w3}, v{w1}, v{w2}, 1.0"),
            &[v(w3)],
            &[v(w1), v(w2)],
        )?;
        op(
            b,
            format!("v_fmac_f32_e32 v{w2}, v{w3}, v{w2}"),
            &[v(w2)],
            &[v(w2), v(w3)],
        )?;
        op(
            b,
            format!("v_div_scale_f32 v{w3}, vcc_lo, v{OUT}, v{w0}, v{OUT}"),
            &[v(w3)],
            &[v(OUT), v(w0)],
        )?;
        op(
            b,
            format!("v_mul_f32_e32 v{w4}, v{w3}, v{w2}"),
            &[v(w4)],
            &[v(w3), v(w2)],
        )?;
        op(
            b,
            format!("v_fma_f32 v{w5}, v{w1}, v{w4}, v{w3}"),
            &[v(w5)],
            &[v(w1), v(w4), v(w3)],
        )?;
        op(
            b,
            format!("v_fmac_f32_e32 v{w4}, v{w5}, v{w2}"),
            &[v(w4)],
            &[v(w4), v(w5), v(w2)],
        )?;
        op(
            b,
            format!("v_fmac_f32_e32 v{w3}, v{w1}, v{w4}"),
            &[v(w3)],
            &[v(w3), v(w1), v(w4)],
        )?;
        alu_wait(b, "depctr_va_vcc(0)")?;
        op(
            b,
            format!("v_div_fmas_f32 v{w1}, v{w3}, v{w2}, v{w4}"),
            &[v(w1)],
            &[v(w3), v(w2), v(w4)],
        )?;
        op(
            b,
            format!("v_div_fixup_f32 v{w0}, v{w1}, v{w0}, v{OUT}"),
            &[v(w0)],
            &[v(w1), v(w0), v(OUT)],
        )?;
        op(
            b,
            format!("v_mov_b32_e32 v{OUT}, v{w0}"),
            &[v(OUT)],
            &[v(w0)],
        )?;
        ref_round_trip(b, OUT, [t0, t1, t2])?;
    }
    op(b, "v_mov_b32_e32 v14, 0", &[v(14)], &[])?;
    mem_op(
        b,
        format!("global_store_b32 v14, v{OUT}, s[16:17]"),
        &[],
        &[v(14), v(OUT), sr(16, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(b, "s_endpgm", &[], &[])?;
    bld.finish()
}

/// The emitted reference module for every compile-time `K` in `ks`: `ref_dot_k{K}` plus the two folds.
fn build_reference_program(ks: &BTreeSet<usize>) -> Program {
    let mut emitted = Vec::new();
    for &k in ks {
        emitted.push(emit_ref_dot(k).unwrap_or_else(|e| panic!("reference ref_dot_k{k}: {e}")));
    }
    emitted.push(emit_ref_fold(false).unwrap_or_else(|e| panic!("reference ref_fold_none: {e}")));
    emitted.push(emit_ref_fold(true).unwrap_or_else(|e| panic!("reference ref_fold_act: {e}")));
    let (text, _) =
        peacemaker_isa::kernels::iu4_gemm::module(&emitted, "lane_rows_bf16_reference").unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

// ------------------------------------------------------------------------------------------------ memory model

const GUARD: usize = 128;
const KERNARG: u64 = 0x1_0000;
const XBASE: u64 = 0x1_0000_0000;
const SENT: u32 = 0x7fc0_a5a5;
const XGUARD: u32 = 0x7fc0_dead;
const WGUARD: u32 = 0x7fc1_7fc1;

fn wbase(i: usize) -> u64 {
    0x2_0000_0000 + i as u64 * 0x400_0000
}
fn ybase(i: usize) -> u64 {
    0x3_0000_0000 + i as u64 * 0x400_0000
}
fn qbase(i: usize) -> u64 {
    0x4_0000_0000 + i as u64 * 0x400_0000
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// Maps `data` between two `GUARD`-byte poison guards; returns the data pointer.
fn map_guarded(mem: &mut Memory, name: &str, base: u64, data: Vec<u8>, pat: u32) -> u64 {
    let guard: Vec<u8> = (0..GUARD / 4).flat_map(|_| pat.to_le_bytes()).collect();
    let mut bytes = guard.clone();
    bytes.extend(data);
    bytes.extend(guard);
    mem.map(name.to_string(), base, bytes).unwrap();
    base + GUARD as u64
}

fn region<'a>(mem: &'a Memory, name: &str) -> &'a peacemaker_ir::emu::Region {
    mem.regions
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("region {name}"))
}

fn launch(emu: &Emulator, mem: &mut Memory, grid: u32, block: u32) -> Result<(), String> {
    launch_grid(emu, mem, grid, 0..1, block)
}

fn launch_grid(
    emu: &Emulator,
    mem: &mut Memory,
    grid: u32,
    ys: std::ops::Range<u32>,
    block: u32,
) -> Result<(), String> {
    for gy in ys {
        for gx in 0..grid {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [gx, gy, 0],
                    block: [block, 1, 1],
                    dynamic_lds: 0,
                },
                mem,
            )
            .map_err(|e| format!("wg({gx},{gy}): {e}"))?;
        }
    }
    Ok(())
}

struct Run {
    /// `[slot][matrix] -> words`; empty when the launch failed or `C` was out of range.
    out: Vec<Vec<Vec<u32>>>,
    problems: Vec<String>,
    launched: bool,
    /// The emulator error of a failed launch (callers classify it: only the exact `REFUSAL` is ever allowed).
    error: Option<String>,
}

/// One launch of PM entry `e` over `rows` with kernarg `C = c_arg`. All structural invariants are checked here and
/// returned as `problems`.
fn run(
    emu: &Emulator,
    e: &Entry,
    p: &Problem,
    rows: &[&[f32]],
    c_arg: i32,
    null_unused: bool,
    tag: &str,
) -> Run {
    run_y(emu, e, p, rows, c_arg, null_unused, tag, None)
}

/// `only_y` launches an isolated, wholly dead y tile against fresh poisoned outputs.
fn run_y(
    emu: &Emulator,
    e: &Entry,
    p: &Problem,
    rows: &[&[f32]],
    c_arg: i32,
    null_unused: bool,
    tag: &str,
    only_y: Option<u32>,
) -> Run {
    let valid = (1..=e.cap as i32).contains(&c_arg);
    let live = if valid && only_y.is_none() {
        c_arg as usize
    } else {
        0
    };
    if valid {
        assert_eq!(rows.len(), c_arg as usize, "{tag}: rows");
    }
    let mut problems = Vec::new();
    let mut mem = Memory::default();
    let nm = p.matrices();
    let mut wptr = [0u64; 4];
    let mut yptr = [0u64; 4];
    let mut sizes = [0usize; 4];
    for i in 0..nm {
        let m = p.ms[i];
        sizes[i] = rows.len() * m;
        if m == 0 && null_unused {
            continue;
        }
        wptr[i] = map_guarded(
            &mut mem,
            &format!("w{i}"),
            wbase(i),
            u16_bytes(&p.w[i]),
            WGUARD,
        );
        let ysent: Vec<u8> = (0..sizes[i]).flat_map(|_| SENT.to_le_bytes()).collect();
        yptr[i] = map_guarded(&mut mem, &format!("y{i}"), ybase(i), ysent, SENT);
    }
    let xdata: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let xptr = map_guarded(&mut mem, "x", XBASE, f32_bytes(&xdata), XGUARD);
    let ksize: usize = match e.fam {
        Fam::X4 => 96,
        Fam::K4 => 48,
    };
    let mut ka = vec![0u8; ksize.max(128)];
    let put64 =
        |ka: &mut Vec<u8>, off: usize, v: u64| ka[off..off + 8].copy_from_slice(&v.to_le_bytes());
    let put32 =
        |ka: &mut Vec<u8>, off: usize, v: u32| ka[off..off + 4].copy_from_slice(&v.to_le_bytes());
    match e.fam {
        Fam::X4 => {
            for i in 0..4 {
                put64(&mut ka, 8 * i, wptr[i]);
                put64(&mut ka, 40 + 8 * i, yptr[i]);
                put32(&mut ka, 72 + 4 * i, p.ms[i] as u32);
            }
            put64(&mut ka, 32, xptr);
            put32(&mut ka, 88, p.k as u32);
            put32(&mut ka, 92, c_arg as u32);
        }
        Fam::K4 => {
            put64(&mut ka, 0, wptr[0]);
            put64(&mut ka, 8, xptr);
            put64(&mut ka, 16, yptr[0]);
            put32(&mut ka, 24, p.ms[0] as u32);
            put32(&mut ka, 28, p.k as u32);
            put32(&mut ka, 32, p.scale.to_bits());
            put32(&mut ka, 36, u32::from(p.act));
            put32(&mut ka, 40, c_arg as u32);
        }
    }
    mem.map("kernarg".into(), KERNARG, ka).unwrap();
    let before_in: Vec<(String, Vec<u8>)> = mem
        .regions
        .iter()
        .filter(|r| r.name.starts_with('w') || r.name == "x")
        .map(|r| (r.name.clone(), r.bytes.clone()))
        .collect();

    let ys = if let Some(y) = only_y {
        y..y + 1
    } else if e.tiled {
        0..if valid { (c_arg as u32).div_ceil(8) } else { 9 }
    } else {
        0..1
    };
    if let Err(err) = launch_grid(emu, &mut mem, p.grid(e), ys, e.block()) {
        return Run {
            out: Vec::new(),
            problems,
            launched: false,
            error: Some(format!("{tag}: emulator: {err}")),
        };
    }

    for (name, bytes) in &before_in {
        let r = region(&mem, name);
        if &r.bytes != bytes || r.written.count() != 0 {
            problems.push(format!(
                "{tag}: input region {name} modified (written bytes {})",
                r.written.count()
            ));
        }
    }
    let mut out = vec![vec![Vec::new(); nm]; live];
    for i in 0..nm {
        if p.ms[i] == 0 && null_unused {
            continue;
        }
        let r = region(&mem, &format!("y{i}"));
        let m = p.ms[i];
        let data_end = GUARD + sizes[i] * 4;
        let guard_ok = |lo: usize, hi: usize| words(&r.bytes[lo..hi]).iter().all(|&w| w == SENT);
        if !guard_ok(0, GUARD) || !guard_ok(data_end, data_end + GUARD) {
            problems.push(format!("{tag}: y{i} guard words overwritten"));
        }
        let want_written = live * m * 4;
        let mut exact = r.written.count() == want_written;
        for b in 0..r.bytes.len() {
            let inside = b >= GUARD && b < GUARD + want_written;
            if r.written.contains(b) != inside {
                exact = false;
                break;
            }
        }
        if !exact {
            problems.push(format!(
                "{tag}: y{i} written byte set != logical extent ({} live words), written bytes {}",
                live * m,
                r.written.count()
            ));
        }
        let all = words(&r.bytes[GUARD..data_end]);
        if all[live * m..].iter().any(|&w| w != SENT) {
            problems.push(format!(
                "{tag}: y{i} words beyond the live rows were written"
            ));
        }
        for (c, slot) in out.iter_mut().enumerate() {
            slot[i] = all[c * m..(c + 1) * m].to_vec();
        }
    }
    Run {
        out,
        problems,
        launched: true,
        error: None,
    }
}

// ------------------------------------------------------------------------------------------------ reference execution

fn fixture_dir() -> String {
    std::env::var("LANE_ROWS_BF16_FIXTURE_DIR").unwrap_or_else(|_| {
        format!(
            "{}/tests/fixtures/gfx1201_singletons/bf16",
            env!("CARGO_MANIFEST_DIR")
        )
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn pinned_fixture((name, sha): (&str, &str)) -> Result<Vec<u8>, String> {
    let path = format!("{}/{name}", fixture_dir());
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("actual-object evidence unavailable: {path}: {e}"))?;
    assert_eq!(
        sha256_hex(&bytes),
        sha,
        "{path}: actual singleton fixture sha256"
    );
    Ok(bytes)
}

/// The lifted hipcc fixture (`Frontend::Hipcc`). A genuine lift failure is reported, not forced.
fn reference_program() -> Result<Program, String> {
    let bytes = pinned_fixture(FIXTURE_QWEN4)?;
    peacemaker_lift::lift_object(
        &bytes,
        peacemaker_lift::Options {
            frontend: Frontend::Hipcc,
        },
    )
    .map(|l| l.program)
    .map_err(|e| format!("lift_object(Hipcc) of {}: {e}", FIXTURE_QWEN4.0))
}

/// Opcodes of `symbol` that have no interpreter semantics row (`Emulator::new` hard-errors on the first of them).
fn unsupported_ops(program: &Program, symbol: &str) -> Result<BTreeSet<String>, String> {
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("no kernel {symbol} in the lifted program"))?;
    let mut set = BTreeSet::new();
    for id in &kernel.body.layout {
        let inst = kernel
            .body
            .insts
            .get(*id)
            .ok_or("layout names a missing instruction")?;
        match peacemaker_ir::isa::lookup(program.target.arch, inst.op, inst.form) {
            None => {
                set.insert("<unknown decoded opcode>".to_string());
            }
            Some(row) => match peacemaker_ir::emu::semantic_class(row.name) {
                Some("wait" | "control" | "memory" | "alu") => {}
                Some(other) => {
                    set.insert(format!("{} (class {other})", row.name));
                }
                None => {
                    set.insert(row.name.to_string());
                }
            },
        }
    }
    Ok(set)
}

fn make_emulator<'a>(program: &'a Program, symbol: &str, what: &str) -> Emulator<'a> {
    Emulator::new(program, symbol).unwrap_or_else(|err| {
        panic!(
            "{what} {symbol}: Emulator::new: {err}; opcodes without an interpreter semantics row: {:?}",
            unsupported_ops(program, symbol)
        )
    })
}

/// The singleton reference: the emitted DAG executed on the interpreter (one `C = 1` launch per row and quarter), plus
/// the lifted hipcc object of each family where the interpreter can run it. The two families are gated
/// independently: an HC object blocked on scalar float ops never disables the ordinary comparison.
struct Reference<'a> {
    dots: BTreeMap<usize, Emulator<'a>>,
    fold: [Emulator<'a>; 2],
    hip_single: Result<Emulator<'a>, String>,
    hip_k4: Result<Emulator<'a>, String>,
}

impl<'a> Reference<'a> {
    fn new(emitted: &'a Program, hip: &'a Result<Program, String>) -> Self {
        let mut dots = BTreeMap::new();
        for kernel in &emitted.kernels {
            if let Some(k) = kernel.symbol.0.strip_prefix("ref_dot_k") {
                dots.insert(
                    k.parse::<usize>().unwrap(),
                    make_emulator(emitted, &kernel.symbol.0, "reference"),
                );
            }
        }
        let fold = [
            make_emulator(emitted, "ref_fold_none", "reference"),
            make_emulator(emitted, "ref_fold_act", "reference"),
        ];
        let hip_single = hipcc_gate(hip, SINGLE_SYMBOL);
        let hip_k4 = hipcc_gate(hip, K4_SYMBOL);
        // `1` requires the ordinary singleton, `all` both families.
        let require = std::env::var("LANE_ROWS_BF16_REQUIRE_HIPCC").unwrap_or_default();
        if require == "1" || require == "all" {
            if let Err(why) = &hip_single {
                panic!("LANE_ROWS_BF16_REQUIRE_HIPCC={require} but the ordinary hipcc singleton is not executable: {why}");
            }
        }
        if require == "all" {
            if let Err(why) = &hip_k4 {
                panic!("LANE_ROWS_BF16_REQUIRE_HIPCC=all but the HC hipcc singleton is not executable: {why}");
            }
        }
        Reference {
            dots,
            fold,
            hip_single,
            hip_k4,
        }
    }

    fn hipcc_available(&self, fam: Fam) -> bool {
        match fam {
            Fam::X4 => self.hip_single.is_ok(),
            Fam::K4 => self.hip_k4.is_ok(),
        }
    }

    fn describe(&self) -> String {
        let gate = |r: &Result<Emulator, String>| match r {
            Ok(_) => "EXECUTABLE (third reference, compared in every launch check)".to_string(),
            Err(why) => format!("SKIPPED, exact reason: {why}"),
        };
        format!(
            "reference: INDEPENDENT checked-builder emitted singleton DAG (ref_dot_k* / ref_fold_*) executed on the CPU interpreter; hipcc object ordinary x4 ({SINGLE_SYMBOL}): {}; hipcc object HC k4 ({K4_SYMBOL}): {}; HARDWARE HC EXCEPTIONAL-DOMAIN GAP: SALU/VALU equivalence and refused division domains are NOT qualified by this CPU oracle",
            gate(&self.hip_single),
            gate(&self.hip_k4)
        )
    }

    /// `ref_dot_k{k_eff}` over the `m` rows of `w` (row stride `k_total`), starting `k_off` elements into each row and
    /// into `x`.
    fn exec_dot(
        &self,
        w: &[u16],
        m: usize,
        k_total: usize,
        x: &[f32],
        k_off: usize,
        k_eff: usize,
    ) -> Result<Vec<u32>, String> {
        if m == 0 {
            return Ok(Vec::new());
        }
        let emu = self
            .dots
            .get(&k_eff)
            .ok_or_else(|| format!("no emitted reference kernel for K = {k_eff}"))?;
        let mut mem = Memory::default();
        let wptr = map_guarded(&mut mem, "w", wbase(0), u16_bytes(w), WGUARD);
        let xptr = map_guarded(&mut mem, "x", XBASE, f32_bytes(x), XGUARD);
        let ysent: Vec<u8> = (0..m).flat_map(|_| SENT.to_le_bytes()).collect();
        let yptr = map_guarded(&mut mem, "y", ybase(0), ysent, SENT);
        let mut ka = vec![0u8; 64];
        ka[0..8].copy_from_slice(&(wptr + 2 * k_off as u64).to_le_bytes());
        ka[8..16].copy_from_slice(&(xptr + 4 * k_off as u64).to_le_bytes());
        ka[16..24].copy_from_slice(&yptr.to_le_bytes());
        ka[24..28].copy_from_slice(&((2 * k_total) as u32).to_le_bytes());
        mem.map("kernarg".into(), KERNARG, ka).unwrap();
        launch(emu, &mut mem, m as u32, 32).map_err(|e| format!("ref_dot_k{k_eff}: {e}"))?;
        Ok(words(&region(&mem, "y").bytes[GUARD..GUARD + 4 * m]))
    }

    /// `ref_fold_{none,act}` over four `[m]` quarter-sum arrays.
    fn exec_fold(
        &self,
        q: &[Vec<u32>],
        m: usize,
        scale: f32,
        act: bool,
    ) -> Result<Vec<u32>, String> {
        let mut mem = Memory::default();
        let mut ka = vec![0u8; 64];
        for (i, quarter) in q.iter().enumerate() {
            let ptr = map_guarded(
                &mut mem,
                &format!("q{i}"),
                qbase(i),
                u32_bytes(quarter),
                WGUARD,
            );
            ka[8 * i..8 * i + 8].copy_from_slice(&ptr.to_le_bytes());
        }
        let ysent: Vec<u8> = (0..m).flat_map(|_| SENT.to_le_bytes()).collect();
        let yptr = map_guarded(&mut mem, "y", ybase(0), ysent, SENT);
        ka[32..40].copy_from_slice(&yptr.to_le_bytes());
        ka[40..44].copy_from_slice(&scale.to_bits().to_le_bytes());
        mem.map("kernarg".into(), KERNARG, ka).unwrap();
        launch(&self.fold[usize::from(act)], &mut mem, m as u32, 32)
            .map_err(|e| format!("ref_fold: {e}"))?;
        Ok(words(&region(&mem, "y").bytes[GUARD..GUARD + 4 * m]))
    }

    /// `[matrix] -> words` of the singleton reference for activation row `x` alone.
    fn row(&self, p: &Problem, x: &[f32]) -> Result<Vec<Vec<u32>>, String> {
        match p.fam {
            Fam::X4 => (0..4)
                .map(|i| self.exec_dot(&p.w[i], p.ms[i], p.k, x, 0, p.k))
                .collect(),
            Fam::K4 => {
                let m = p.ms[0];
                if m == 0 {
                    return Ok(vec![Vec::new()]);
                }
                if !p.k.is_multiple_of(32) {
                    return Err(format!("HC K = {} is not a multiple of 32", p.k));
                }
                let part = p.k / 4;
                let q: Vec<Vec<u32>> = (0..4)
                    .map(|qi| self.exec_dot(&p.w[0], m, p.k, x, qi * part, part))
                    .collect::<Result<_, _>>()?;
                Ok(vec![self.exec_fold(&q, m, p.scale, p.act)?])
            }
        }
    }

    /// The same through the lifted hipcc object of the problem's family: `None` when that family's object is not
    /// executable (see [`Reference::describe`]), `Some(Err)` when it errors at runtime on this input.
    fn hipcc_row(&self, p: &Problem, x: &[f32]) -> Option<Result<Vec<Vec<u32>>, String>> {
        match p.fam {
            Fam::X4 => {
                let single = self.hip_single.as_ref().ok()?;
                Some(
                    (0..4)
                        .map(|i| hipcc_single(single, &p.w[i], p.ms[i], p.k, x))
                        .collect(),
                )
            }
            Fam::K4 => {
                let k4 = self.hip_k4.as_ref().ok()?;
                Some(hipcc_k4(k4, &p.w[0], p.ms[0], p.k, x, p.scale, p.act).map(|y| vec![y]))
            }
        }
    }
}

fn hipcc_launch(
    emu: &Emulator,
    w: &[u16],
    m: usize,
    k: usize,
    x: &[f32],
    extra: &[(usize, u32)],
    block: u32,
) -> Result<Vec<u32>, String> {
    let mut mem = Memory::default();
    let wptr = map_guarded(&mut mem, "w", wbase(0), u16_bytes(w), WGUARD);
    let xptr = map_guarded(&mut mem, "x", XBASE, f32_bytes(x), XGUARD);
    let ysent: Vec<u8> = (0..m).flat_map(|_| SENT.to_le_bytes()).collect();
    let yptr = map_guarded(&mut mem, "y", ybase(0), ysent, SENT);
    let mut ka = vec![0u8; 256];
    ka[0..8].copy_from_slice(&wptr.to_le_bytes());
    ka[8..16].copy_from_slice(&xptr.to_le_bytes());
    ka[16..24].copy_from_slice(&yptr.to_le_bytes());
    ka[24..28].copy_from_slice(&(m as u32).to_le_bytes());
    ka[28..32].copy_from_slice(&(k as u32).to_le_bytes());
    for &(off, val) in extra {
        ka[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
    mem.map("kernarg".into(), KERNARG, ka).unwrap();
    launch(emu, &mut mem, m as u32, block).map_err(|e| format!("hipcc singleton: {e}"))?;
    Ok(words(&region(&mem, "y").bytes[GUARD..GUARD + 4 * m]))
}

/// The CLOSED list of float domains the interpreter refuses that these tests reach (H0's claim, not the CPU oracle's):
/// the four `v_div_scale_f32` source errors of `peacemaker-ir/src/emu/alu.rs` (`div_scale_chain`), matched by exact
/// message and nothing else. Any other emulator error is a defect and fails.
const REFUSALS: [&str; 4] = [
    "v_div_scale_f32: exponent difference -96 is measured only for denominator exponent 254",
    "v_div_scale_f32: S0 * 2^-64 with exponent field 64..=90 is not measured",
    "v_div_scale_f32: denominator exponent 228..=253 is not measured",
    "v_div_scale_f32: numerator exponent 24..=26 is not measured",
];

fn refusal_kind(err: &str) -> Option<usize> {
    REFUSALS.iter().position(|m| err.contains(m))
}

fn is_refusal(err: &str) -> bool {
    refusal_kind(err).is_some()
}

/// Generator-reproducible description of a row: length, its first four words and its first six non-zero words.
fn row_bits(row: &[f32]) -> String {
    let head: Vec<String> = row
        .iter()
        .take(4)
        .map(|v| format!("{:#010x}", v.to_bits()))
        .collect();
    let nz: Vec<String> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| v.to_bits() & 0x7fff_ffff != 0)
        .take(6)
        .map(|(i, v)| format!("{i}:{:#010x}", v.to_bits()))
        .collect();
    format!(
        "row_len={} head=[{}] first_nonzero=[{}]",
        row.len(),
        head.join(" "),
        nz.join(" ")
    )
}

/// A runtime error of the lifted hipcc object that is the interpreter's precisely documented refusal of an unmeasured
/// float domain / mode / NaN selection (`NaN operand: the ISA document does not specify ...`, `... is not measured`,
/// `... not qualified`, ...) and therefore allowed to skip that single row. Anything else (unmapped or out-of-bounds
/// memory, LDS bounds, barrier deadlock, instruction limit, PC overrun, ABI/descriptor errors, launch failures) is a
/// real defect and must fail.
fn is_semantic_gap(err: &str) -> bool {
    const HARD: [&str; 8] = [
        "unmapped memory",
        "out of bounds",
        "deadlock",
        "instruction limit",
        "PC ran past",
        "descriptor",
        "overflow",
        "SRD",
    ];
    const GAP: [&str; 7] = [
        "does not specify",
        "not specified",
        "not measured",
        "unmeasured",
        "not qualified",
        "unqualified",
        "not modeled",
    ];
    is_refusal(err)
        || (!HARD.iter().any(|h| err.contains(h)) && GAP.iter().any(|g| err.contains(g)))
}

/// One family's gate on the lifted hipcc object: lifted, free of opcodes without interpreter semantics, accepted by
/// `Emulator::new`. The error is the exact reason.
fn hipcc_gate<'a>(hip: &'a Result<Program, String>, sym: &str) -> Result<Emulator<'a>, String> {
    let p = hip
        .as_ref()
        .map_err(|e| format!("hipcc fixture not lifted: {e}"))?;
    let census = unsupported_ops(p, sym).map_err(|e| format!("{sym}: {e}"))?;
    if !census.is_empty() {
        return Err(format!(
            "{sym}: opcodes without interpreter semantics {census:?}"
        ));
    }
    Emulator::new(p, sym).map_err(|e| format!("{sym}: Emulator::new: {e}"))
}

/// `gemv_bf16_xf32`: `W@0 x@8 y@16 M@24 K@28`, grid `[M]`, block 32.
fn hipcc_single(
    emu: &Emulator,
    w: &[u16],
    m: usize,
    k: usize,
    x: &[f32],
) -> Result<Vec<u32>, String> {
    if m == 0 {
        return Ok(Vec::new());
    }
    hipcc_launch(emu, w, m, k, x, &[], 32)
}

/// `gemv_bf16_xf32_k4`: `.. M@24 K@28 scale@32 hc_act@36`, grid `[M]`, block 128.
fn hipcc_k4(
    emu: &Emulator,
    w: &[u16],
    m: usize,
    k: usize,
    x: &[f32],
    scale: f32,
    act: bool,
) -> Result<Vec<u32>, String> {
    if m == 0 {
        return Ok(Vec::new());
    }
    hipcc_launch(
        emu,
        w,
        m,
        k,
        x,
        &[(32, scale.to_bits()), (36, u32::from(act))],
        128,
    )
}

// ------------------------------------------------------------------------------------------------ tally / driver

#[derive(Default)]
struct Tally {
    per: BTreeMap<&'static str, (u64, u64, u64)>,
    first: Vec<String>,
    bad: u64,
    hip_rows: u64,
    hip_skips: u64,
    hip_first_skip: Option<String>,
    refused_rows: u64,
    refusal_kinds: [u64; 4],
    refused_launches: u64,
    refusal_records: Vec<String>,
}

impl Tally {
    fn record_refusal(&mut self, rec: String) {
        if self.refusal_records.len() < 400 {
            self.refusal_records.push(rec);
        }
    }

    fn launch(&mut self, e: &Entry, words: u64) {
        let t = self.per.entry(e.symbol).or_default();
        t.0 += 1;
        t.1 += words;
    }
    fn fail(&mut self, e: &Entry, what: String) {
        self.per.entry(e.symbol).or_default().2 += 1;
        self.bad += 1;
        if self.first.len() < 32 {
            self.first.push(format!("{}: {what}", e.symbol));
        }
    }
    fn finish(&self, test: &str, reference: &Reference) {
        log(format!("[{test}] {}", reference.describe()));
        for (sym, (launches, words, bad)) in &self.per {
            log(format!("[{test}] {sym}: launches={launches} words_compared={words} (each vs singleton AND vs PM c1) mismatches={bad}"));
        }
        log(format!(
            "[{test}] hipcc object as third reference: rows compared={} rows skipped at runtime={} first skip={}",
            self.hip_rows,
            self.hip_skips,
            self.hip_first_skip.as_deref().unwrap_or("none")
        ));
        log(format!(
            "[{test}] H0-ONLY refusals (closed list of 4 exact emulator messages, see REFUSALS), counted separately from word mismatches: rows refused by reference AND PM c1 = {} (per message #0..#3 = {:?}), PM entry launches refused = {}",
            self.refused_rows, self.refusal_kinds, self.refused_launches
        ));
        for rec in &self.refusal_records {
            log(format!("[{test}] H0-ONLY ROW: {rec}"));
        }
        log(format!("[{test}] total mismatches/violations={}", self.bad));
        for f in &self.first {
            log(format!("[{test}] FIRST DIFFERENCE: {f}"));
        }
        assert_eq!(
            self.bad, 0,
            "[{test}] {} mismatches/violations; first: {:#?}",
            self.bad, self.first
        );
        assert!(!self.per.is_empty(), "[{test}] nothing ran");
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// Every `C` in `1..=cap`, plus rotations, reversal and the same row in every slot.
    Core,
    /// `C in {1, cap}` only.
    Light,
    /// `C = cap` windows over all rows (and `C = 1`): every row of the problem reaches every entry.
    Windows,
}

struct Rig<'a> {
    entries: &'a [Entry],
    emus: &'a [Emulator<'a>],
    refs: &'a Reference<'a>,
}

impl Rig<'_> {
    fn c1(&self, fam: Fam) -> &Entry {
        self.entries
            .iter()
            .find(|e| e.fam == fam && e.cap == 1 && !e.wg8)
            .unwrap()
    }
}

/// Everything one test needs: the PM module, the emitted singleton reference for the compile-time `K`s the test
/// uses, the lifted hipcc fixture and the entry table.
struct Env {
    pm: Pm,
    refprog: Program,
    hip: Result<Program, String>,
    entries: Vec<Entry>,
}

impl Env {
    fn new(ks: &BTreeSet<usize>) -> Self {
        Env {
            pm: Pm::new(),
            refprog: build_reference_program(ks),
            hip: reference_program(),
            entries: entries(),
        }
    }
    fn rowtiles(ks: &BTreeSet<usize>) -> Self {
        let entries = rowtile_entries();
        let emitted: Vec<Emitted> = entries
            .iter()
            .map(|e| {
                let spec = lr::Entry {
                    hc: e.fam == Fam::K4,
                    wg8: false,
                    cap: if e.tiled { 8 } else { 1 },
                };
                (if e.tiled {
                    lr::build_rowtiles(spec)
                } else {
                    lr::build(spec)
                })
                .unwrap_or_else(|err| panic!("{}: {err}", e.symbol))
            })
            .collect();
        let (text, _) =
            peacemaker_isa::kernels::iu4_gemm::module(&emitted, "bf16_rowtile_test").unwrap();
        let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Env {
            pm: Pm {
                elf,
                emitted,
                program,
            },
            refprog: build_reference_program(ks),
            hip: reference_program(),
            entries,
        }
    }
}

/// Keep the original eight stress rows and generate distinct later rows so a missed tile-base offset cannot pass.
fn widen_problem(p: &mut Problem, seed: u64) {
    let mut rng = Rng::new(seed);
    let k = p.k;
    p.rows
        .extend((8..64).map(|r| gen_row(&mut rng, k, ROW_KINDS[r % ROW_KINDS.len()])));
    assert_eq!(p.rows.len(), 64);
}

fn rowtile_emulators<'a>(env: &'a Env) -> Vec<Emulator<'a>> {
    env.entries
        .iter()
        .map(|e| make_emulator(&env.pm.program, e.symbol, "rowtile / PM c1"))
        .collect()
}

#[test]
fn rowtiles_ordinary_every_r_1_through_64_matches_all_singleton_oracles() {
    let ks: BTreeSet<usize> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 40, 250, 257, 264]
        .into_iter()
        .collect();
    let env = Env::rowtiles(&ks);
    let emus = rowtile_emulators(&env);
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    // Every scalar K%8 residue, packed paths and long scalar tails. Each shape runs every R, not just boundaries.
    for (id, &k) in ks.iter().enumerate() {
        let mut p = make_x4(0xa000 + id as u64, MS[id % MS.len()], k, false);
        widen_problem(&mut p, 0xa100 + id as u64);
        check_problem(&rig, &mut tally, &p, id, Shape::Core, id % 2 == 1);
    }
    // Every empty/odd-height matrix placement, with null pointers for empty slots.
    for (id, ms) in MS.into_iter().chain([[0, 0, 0, 0]]).enumerate() {
        let k = if id % 2 == 0 { 40 } else { 33 };
        let mut p = make_x4(0xa200 + id as u64, ms, k, false);
        widen_problem(&mut p, 0xa300 + id as u64);
        check_problem(&rig, &mut tally, &p, id, Shape::Core, true);
    }
    assert_eq!(tally.refused_rows, 0, "ordinary rows must all be compared");
    tally.finish("rowtiles_ordinary_R1..64", &refs);
}

#[test]
fn rowtiles_hc_raw_and_activated_every_r_1_through_64_matches_all_singleton_oracles() {
    // K/4 has every legal packed remainder in its 32-element lane sweep, plus multiple full sweeps.
    let totals = [32usize, 64, 96, 128, 160, 288, 1056];
    let ks = totals.into_iter().map(|k| k / 4).collect();
    let env = Env::rowtiles(&ks);
    let emus = rowtile_emulators(&env);
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for (id, k) in totals.into_iter().enumerate() {
        for act in [false, true] {
            let mut p = make_hc(0xb000 + id as u64, 1 + id % 3, k, act, 0.75, false);
            widen_problem(&mut p, 0xb100 + id as u64);
            // Bounded finite cases must compare every row, not replace exceptional rows with qualified-fill.
            let mut rng = Rng::new(0xb200 + id as u64);
            for w in &mut p.w[0] {
                *w = [0x3f80, 0xbf80, 0x3e80, 0xbe80, 0][rng.below(5)];
            }
            for row in &mut p.rows {
                for x in row {
                    *x = f32::from_bits(
                        0x3d00_0000
                            | (rng.next() as u32 & 0x007f_ffff)
                            | ((rng.next() as u32 & 1) << 31),
                    );
                }
            }
            check_problem(&rig, &mut tally, &p, id, Shape::Core, false);
        }
    }
    assert_eq!(
        tally.refused_rows, 0,
        "bounded HC cases must compare every live row"
    );
    assert_eq!(tally.refused_launches, 0);
    tally.finish("rowtiles_hc_R1..64", &refs);
}

#[test]
fn rowtiles_nonfinite_rows_remain_isolated_with_explicit_hardware_hc_gap() {
    let ks = [8usize, 33, 264].into_iter().collect();
    let env = Env::rowtiles(&ks);
    let emus = rowtile_emulators(&env);
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for (id, k) in [33usize, 264].into_iter().enumerate() {
        let mut p = make_x4(0xc000 + id as u64, [3, 0, 1, 2], k, true);
        widen_problem(&mut p, 0xc100 + id as u64);
        check_problem(&rig, &mut tally, &p, id, Shape::Core, true);
    }
    for act in [false, true] {
        let mut p = make_hc(0xc200, 3, 32, act, 0.75, true);
        widen_problem(&mut p, 0xc300);
        check_problem(&rig, &mut tally, &p, 0, Shape::Core, false);
    }
    tally.finish("rowtiles_nonfinite_HC_hardware_gap", &refs);
}

#[test]
fn rowtiles_invalid_0_and_65_rows_uniformly_write_nothing() {
    let env = Env::rowtiles(&BTreeSet::new());
    let emus = rowtile_emulators(&env);
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for p in [
        make_x4(0xd000, [3, 0, 1, 2], 33, false),
        make_hc(0xd100, 3, 32, false, 0.75, false),
        make_hc(0xd200, 3, 32, true, 0.75, false),
    ] {
        // Includes 0, 65, negative and extreme signed counts; all nine y tiles must terminate without writes.
        check_invalid_c(&rig, &mut tally, &p);
    }
    tally.finish("rowtiles_invalid_R", &refs);
}

fn orders(c: usize, rows: usize, shape: Shape, case: usize) -> Vec<(String, Vec<usize>)> {
    let mut o = vec![("identity".to_string(), (0..c).collect::<Vec<_>>())];
    match shape {
        Shape::Core if c >= 2 => {
            o.push(("rotate1".into(), (0..c).map(|j| (j + 1) % c).collect()));
            o.push(("reverse".into(), (0..c).rev().collect()));
            let r = case % c;
            o.push((format!("row{r}-in-every-slot"), vec![r; c]));
        }
        Shape::Windows if c >= 1 => {
            o.clear();
            let mut start = 0;
            while start + c <= rows {
                o.push((
                    format!("rows{start}..{}", start + c),
                    (start..start + c).collect(),
                ));
                start += c;
            }
        }
        _ => {}
    }
    o
}

fn c_list(e: &Entry, shape: Shape) -> Vec<usize> {
    let mut cs: Vec<usize> = match shape {
        Shape::Core => (1..=e.cap).collect(),
        Shape::Light | Shape::Windows => vec![1, e.cap],
    };
    if full() {
        cs = (1..=e.cap).collect();
    }
    cs.dedup();
    cs
}

fn check_problem(
    rig: &Rig,
    tally: &mut Tally,
    p: &Problem,
    case: usize,
    shape: Shape,
    null_unused: bool,
) {
    let c1 = rig.c1(p.fam);
    let nrows = p.rows.len();
    // The independent singleton reference per row (emitted DAG on the interpreter). The exact semantic-domain refusal
    // is H0-only; every other error fails.
    let mut reference: Vec<Option<Vec<Vec<u32>>>> = Vec::new();
    let mut ref_refused = vec![false; nrows];
    let mut ref_kind: Vec<Option<usize>> = vec![None; nrows];
    let mut ref_failed = vec![false; nrows];
    for (r, row) in p.rows.iter().enumerate() {
        match rig.refs.row(p, row) {
            Ok(v) => reference.push(Some(v)),
            Err(e) if is_refusal(&e) => {
                ref_refused[r] = true;
                ref_kind[r] = refusal_kind(&e);
                reference.push(None);
            }
            Err(e) => {
                ref_failed[r] = true;
                tally.fail(
                    c1,
                    format!(
                        "{}: reference row {r} failed on the interpreter: {e}",
                        p.label
                    ),
                );
                reference.push(None);
            }
        }
    }
    // The PM c1 entry at C = 1 for every row alone.
    let mut baseline: Vec<Option<Vec<Vec<u32>>>> = Vec::new();
    let mut c1_refused = vec![false; nrows];
    let mut c1_kind: Vec<Option<usize>> = vec![None; nrows];
    let mut c1_failed = vec![false; nrows];
    for (r, row) in p.rows.iter().enumerate() {
        let res = run(
            &rig.emus[c1.idx],
            c1,
            p,
            &[row.as_slice()],
            1,
            null_unused,
            &format!("{} PM-c1 baseline row {r}", p.label),
        );
        for problem in &res.problems {
            tally.fail(c1, problem.clone());
        }
        if !res.launched {
            match &res.error {
                Some(e) if is_refusal(e) => {
                    c1_refused[r] = true;
                    c1_kind[r] = refusal_kind(e);
                }
                Some(e) => {
                    c1_failed[r] = true;
                    tally.fail(c1, e.clone());
                }
                None => {
                    c1_failed[r] = true;
                    tally.fail(c1, format!("{} PM-c1 row {r}: launch failed", p.label));
                }
            }
        }
        baseline.push(if res.launched {
            res.out.into_iter().next()
        } else {
            None
        });
    }
    // A row is blocked (H0-only) iff the reference AND PM c1 refuse it with the exact message. One refusing while the
    // other succeeds is a failure; if neither refuses, bytes are compared normally.
    let mut blocked = vec![false; nrows];
    for r in 0..nrows {
        if ref_failed[r] || c1_failed[r] {
            continue;
        }
        if ref_refused[r] && c1_refused[r] {
            blocked[r] = true;
            tally.refused_rows += 1;
            if let Some(k) = ref_kind[r] {
                tally.refusal_kinds[k] += 1;
            }
            tally.record_refusal(format!(
                "{} row {r} scale_bits={:#010x} {} refusal ref=#{:?} c1=#{:?} ({})",
                p.label,
                p.scale.to_bits(),
                row_bits(&p.rows[r]),
                ref_kind[r],
                c1_kind[r],
                ref_kind[r].map_or("?", |k| REFUSALS[k])
            ));
        } else if ref_refused[r] != c1_refused[r] {
            tally.fail(
                c1,
                format!(
                    "{} row {r}: refusal disagreement: reference {} / PM c1 {}",
                    p.label,
                    if ref_refused[r] {
                        "REFUSES"
                    } else {
                        "succeeds"
                    },
                    if c1_refused[r] { "REFUSES" } else { "succeeds" }
                ),
            );
        }
    }
    // Third reference: the lifted hipcc object, per family, wherever the interpreter can execute it. The emitted DAG
    // must also equal it (so a divergence of the independent reference is caught here, not only in the identity test).
    let mut hip: Vec<Option<Vec<Vec<u32>>>> = Vec::new();
    for (r, row) in p.rows.iter().enumerate() {
        match rig.refs.hipcc_row(p, row) {
            None => hip.push(None),
            Some(Ok(h)) => {
                if let Some(rf) = &reference[r] {
                    for i in 0..p.matrices() {
                        for n in 0..p.ms[i] {
                            if rf[i][n] != h[i][n] {
                                tally.fail(
                                    c1,
                                    format!(
                                        "{} row {r} matrix {i} output {n}: emitted DAG {:#010x} != hipcc object {:#010x}",
                                        p.label, rf[i][n], h[i][n]
                                    ),
                                );
                            }
                        }
                    }
                }
                tally.hip_rows += 1;
                hip.push(Some(h));
            }
            Some(Err(e)) if is_semantic_gap(&e) => {
                tally.hip_skips += 1;
                tally
                    .hip_first_skip
                    .get_or_insert_with(|| format!("{}: {e}", p.label));
                hip.push(None);
            }
            Some(Err(e)) => {
                tally.fail(c1, format!("{} row {r}: hipcc object failed on the interpreter (not a documented semantic gap): {e}", p.label));
                hip.push(None);
            }
        }
    }
    // Rows the whole stack accepts: they get every entry at every C in `+qualified-fill` batches where a blocked row
    // would otherwise abort the launch.
    let qual: Vec<usize> = (0..nrows)
        .filter(|&r| !blocked[r] && baseline[r].is_some())
        .collect();
    for e in rig.entries.iter().filter(|e| e.fam == p.fam) {
        for c in c_list(e, shape) {
            let mut variants: Vec<(String, Vec<usize>)> = Vec::new();
            for (oname, order) in orders(c, nrows, shape, case) {
                if order.iter().any(|&r| blocked[r]) && !qual.is_empty() {
                    let fill: Vec<usize> = order
                        .iter()
                        .enumerate()
                        .map(|(j, &r)| if blocked[r] { qual[j % qual.len()] } else { r })
                        .collect();
                    variants.push((format!("{oname}+qualified-fill"), fill));
                }
                variants.push((oname, order));
            }
            for (oname, order) in variants {
                let blocked_in = order.iter().any(|&r| blocked[r]);
                let rows: Vec<&[f32]> = order.iter().map(|&r| p.rows[r].as_slice()).collect();
                let tag = format!("{} C={c} order={oname} null_unused={null_unused}", p.label);
                let result = run(&rig.emus[e.idx], e, p, &rows, c as i32, null_unused, &tag);
                if e.tiled && oname == "identity" {
                    let dead_y = (c as u32).div_ceil(8);
                    let dead = run_y(
                        &rig.emus[e.idx],
                        e,
                        p,
                        &rows,
                        c as i32,
                        null_unused,
                        &format!("{tag} dead y={dead_y}"),
                        Some(dead_y),
                    );
                    for problem in dead.problems {
                        tally.fail(e, problem);
                    }
                    if !dead.launched {
                        tally.fail(
                            e,
                            format!("{tag}: dead y tile did not terminate: {:?}", dead.error),
                        );
                    }
                }
                let mut words_compared = 0u64;
                for problem in &result.problems {
                    tally.fail(e, problem.clone());
                }
                if result.launched && blocked_in {
                    tally.fail(
                        e,
                        format!("{tag}: refusal not reproduced: the entry computed a batch containing a row PM c1 and the reference refuse"),
                    );
                } else if result.launched {
                    for (slot, &r) in order.iter().enumerate() {
                        for i in 0..p.matrices() {
                            for n in 0..p.ms[i] {
                                let got = result.out[slot][i][n];
                                words_compared += 1;
                                if let Some(rf) = &reference[r] {
                                    let want = rf[i][n];
                                    if got != want {
                                        tally.fail(
                                            e,
                                            format!(
                                                "{tag} slot {slot} (row {r}) matrix {i} output {n}: got {got:#010x} singleton {want:#010x}"
                                            ),
                                        );
                                    }
                                }
                                if let Some(h) = &hip[r] {
                                    let want = h[i][n];
                                    if got != want {
                                        tally.fail(
                                            e,
                                            format!(
                                                "{tag} slot {slot} (row {r}) matrix {i} output {n}: got {got:#010x} hipcc-object {want:#010x}"
                                            ),
                                        );
                                    }
                                }
                                if let Some(bs) = &baseline[r] {
                                    let want = bs[i][n];
                                    if got != want {
                                        tally.fail(
                                            e,
                                            format!(
                                                "{tag} slot {slot} (row {r}) matrix {i} output {n}: got {got:#010x} PM-c1(C=1) {want:#010x}"
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }
                } else {
                    match &result.error {
                        Some(err) if is_refusal(err) && blocked_in => tally.refused_launches += 1,
                        Some(err) if is_refusal(err) => tally.fail(
                            e,
                            format!("{tag}: exact refusal on a batch whose rows PM c1 and the reference both accept: {err}"),
                        ),
                        Some(err) => tally.fail(e, err.clone()),
                        None => tally.fail(e, format!("{tag}: launch failed")),
                    }
                }
                tally.launch(e, words_compared);
            }
        }
    }
}

/// Out-of-range `C` returns before any store (and before any barrier: the launch must also terminate).
fn check_invalid_c(rig: &Rig, tally: &mut Tally, p: &Problem) {
    let rows: Vec<&[f32]> = p.rows.iter().take(3).map(|r| r.as_slice()).collect();
    for e in rig.entries.iter().filter(|e| e.fam == p.fam) {
        let mut cs = vec![0i32, e.cap as i32 + 1, -1, i32::MAX, i32::MIN];
        if e.cap == 1 {
            cs.push(2);
        }
        for c in cs {
            let tag = format!("{} INVALID C={c}", p.label);
            let result = run(&rig.emus[e.idx], e, p, &rows, c, false, &tag);
            for problem in &result.problems {
                tally.fail(e, problem.clone());
            }
            if !result.launched {
                tally.fail(
                    e,
                    format!(
                        "{tag}: launch failed: {}",
                        result.error.as_deref().unwrap_or("no error")
                    ),
                );
            }
            tally.launch(e, 0);
        }
    }
}

struct Pm {
    elf: Vec<u8>,
    emitted: Vec<Emitted>,
    program: Program,
}

impl Pm {
    fn new() -> Self {
        let (emitted, text, _) = lr::emit_module().unwrap();
        let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Pm {
            elf,
            emitted,
            program,
        }
    }
    fn emulators(&self) -> Vec<Emulator<'_>> {
        SYMBOLS
            .iter()
            .map(|sym| make_emulator(&self.program, sym, "PM entry"))
            .collect()
    }
}

// ------------------------------------------------------------------------------------------------ shape matrices

const MS: [[usize; 4]; 11] = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 4],
    [5, 0, 7, 0],
    [0, 9, 0, 1],
    [8, 3, 0, 2],
    [4, 5, 6, 7],
    [9, 8, 1, 0],
    [0, 0, 17, 0],
    [7, 0, 0, 9],
];

fn x4_cases() -> Vec<(usize, [usize; 4], usize)> {
    let mut ks = vec![8usize, 40, 256, 264, 520, 2560, 1, 7, 9, 33, 250, 257, 1003];
    if full() {
        ks.extend([96, 1024, 1032, 6144, 3, 15, 63, 511, 1537]);
    }
    let mut cases = Vec::new();
    for (ki, &k) in ks.iter().enumerate() {
        for (mi, ms) in MS.iter().enumerate() {
            let heavy = k >= 2048;
            if heavy && ms.iter().sum::<usize>() > 10 {
                continue;
            }
            if full() || (ki + mi) % 3 == 0 {
                cases.push((cases.len(), *ms, k));
            }
        }
    }
    for ms in [[3, 0, 2, 0], [0, 9, 0, 1], [1, 0, 0, 0]] {
        cases.push((cases.len(), ms, 2560));
    }
    cases
}

const SCALES: [f32; 6] = [1.0, 0.5, 3.7, -2.25, 1e-3, 123456.0];

fn hc_cases() -> Vec<(usize, usize, usize)> {
    let mut ks = vec![32usize, 64, 96, 160, 288, 320, 1024, 1056, 2560, 10240];
    if full() {
        ks.extend([128, 192, 224, 6144]);
    }
    let ms = [1usize, 2, 3, 4, 5, 7, 8, 9];
    let mut cases = Vec::new();
    for (ki, &k) in ks.iter().enumerate() {
        for (mi, &m) in ms.iter().enumerate() {
            if k >= 2560 && ![2usize, 3, 5].contains(&m) {
                continue;
            }
            if k == 10240 && m == 5 {
                continue;
            }
            if full() || k >= 2560 || (ki + mi) % 3 == 0 {
                cases.push((cases.len(), m, k));
            }
        }
    }
    cases
}

// ------------------------------------------------------------------------------------------------ numerical tests

#[test]
fn ordinary_x4_every_entry_matches_the_singleton_at_every_c() {
    let cases = x4_cases();
    assert!(cases.len() >= 30, "{} cases", cases.len());
    let ks: BTreeSet<usize> = cases.iter().map(|c| c.2).collect();
    let env = Env::new(&ks);
    let emus = env.pm.emulators();
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for (id, ms, k) in cases {
        let p = make_x4(0x1000 + id as u64, ms, k, false);
        let shape = if id % 4 == 0 {
            Shape::Core
        } else {
            Shape::Light
        };
        check_problem(&rig, &mut tally, &p, id, shape, id % 2 == 1);
    }
    tally.finish("ordinary_x4", &refs);
}

#[test]
fn hc_k4_every_entry_matches_the_singleton_at_every_c() {
    let cases = hc_cases();
    assert!(
        cases.iter().any(|&(_, _, k)| k == 10240),
        "actual K = 10240 must be exercised"
    );
    let ks: BTreeSet<usize> = cases.iter().map(|c| c.2 / 4).collect();
    let env = Env::new(&ks);
    let emus = env.pm.emulators();
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for (id, m, k) in cases {
        for (v, act) in [false, true].into_iter().enumerate() {
            let scale = SCALES[(id + v) % SCALES.len()];
            let p = make_hc(0x2000 + id as u64 * 2 + v as u64, m, k, act, scale, false);
            let shape = if id % 4 == 0 && k < 2560 {
                Shape::Core
            } else {
                Shape::Light
            };
            check_problem(&rig, &mut tally, &p, id, shape, false);
        }
    }
    tally.finish("hc_k4", &refs);
}

/// `hc_act` extremes and the exact semantic-domain refusal. Large +-, near zero, values that round across BF16 ties,
/// SiLU saturation both sides, exp under/overflow, overflow of the scaled value. Row `c` carries the target sum `T_c`
/// in element 0 (everything else is +-0), output `n` scales it by an exactly representable BF16 weight.
///
/// The interpreter refuses a CLOSED list of four `v_div_scale_f32` float domains these inputs reach (`REFUSALS`: exponent
/// difference -96 with a denominator exponent other than 254; `S0 * 2^-64` with exponent field 64..=90; denominator
/// exponent 228..=253; numerator exponent 24..=26; all H0's claims, not the CPU oracle's). Nothing is
/// filtered: every generated input runs. A row is H0-only when the emitted reference AND PM `c1` refuse it with one of
/// those exact messages; every PM entry launch whose batch contains such a row must refuse with one of them
/// (and a batch without one must not); the qualified rows of the same problem are then run through every entry at every
/// `C` in refusal-free `+qualified-fill` batches with full byte comparison against the reference and PM `c1`. One side
/// refusing while the other succeeds is a failure. When a future emulator model qualifies the domain nothing refuses and
/// all rows are byte-compared. Refusals are counted separately from output-word mismatches and logged with the exact
/// generator label, scale bits and row bits (the reproducible H0 case list).
#[test]
fn hc_unqualified_division_inputs_are_refused_or_byte_compared() {
    let ks: BTreeSet<usize> = [8usize, 40].into_iter().collect();
    let env = Env::new(&ks);
    let emus = env.pm.emulators();
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    // The known case (first reported by the full run): M1 K32 scale 0.5 seed 8193, row 5.
    let known = make_hc(8193, 1, 32, true, 0.5, false);
    {
        let c1 = rig.c1(Fam::K4);
        let row = known.rows[5].as_slice();
        let want = refs.row(&known, row);
        let got = run(
            &emus[c1.idx],
            c1,
            &known,
            &[row],
            1,
            false,
            "known case row 5 (PM c1)",
        );
        log(format!(
            "[hc_unqualified] known case {} row 5 scale_bits={:#010x} {}",
            known.label,
            known.scale.to_bits(),
            row_bits(row)
        ));
        assert!(
            got.problems.is_empty(),
            "known case: PM c1 structural problems {:?}",
            got.problems
        );
        match (&want, got.launched, &got.error) {
            (Err(re), false, Some(ce)) if re.contains(REFUSALS[0]) && ce.contains(REFUSALS[0]) => {
                log("[hc_unqualified] known case row 5: reference and PM c1 refuse with the exact message (H0-only)")
            }
            (Ok(w), true, _) => {
                assert_eq!(got.out[0][0], w[0], "known case row 5: both succeed but bytes differ");
                log("[hc_unqualified] known case row 5: both succeed, bytes equal (the model now qualifies this domain)")
            }
            (w, launched, err) => panic!(
                "known case row 5: reference {:?} / PM c1 launched={launched} error={err:?}: neither the exact refusal on both sides nor a byte comparison",
                w.as_ref().map(|_| ())
            ),
        }
    }
    check_problem(&rig, &mut tally, &known, 0, Shape::Windows, false);
    let t: Vec<f32> = [
        0.0f32,
        -0.0,
        1.0e-45,
        1.0e-38,
        3.0e-30,
        0.001,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        50.0,
        80.0,
        88.0,
        88.72,
        89.5,
        100.0,
        1.0e10,
        1.0e30,
        3.3e38,
        f32::MAX,
        // BF16 tie neighbours: below / at / above the halfway point of the f32 low half.
        f32::from_bits(0x3f80_7fff),
        f32::from_bits(0x3f80_8000),
        f32::from_bits(0x3f80_8001),
        f32::from_bits(0x3f81_8000),
        f32::from_bits(0x7f7f_8000),
        f32::from_bits(0x7f7f_7fff),
    ]
    .into_iter()
    .flat_map(|x| [x, -x])
    .collect();
    let wscales = [1.0f32, -1.0, 0.5, 2.0, 0.75, 3.0, -0.125, 1.5];
    let escales = [1.0f32, 0.5, 3.7, -2.25, 1e-3, 123456.0, 1e30, 1e-30, 1e-40];
    let mut id = 0usize;
    for &k in &[32usize, 160] {
        for chunk in t.chunks(8) {
            for &sc in &escales {
                let seed = 0x3000 + id as u64;
                let mut rng = Rng::new(seed);
                let mut w0 = gen_w(&mut rng, 8 * k);
                for (n, ws) in wscales.iter().enumerate() {
                    w0[n * k] = (ws.to_bits() >> 16) as u16;
                }
                let rows: Vec<Vec<f32>> = (0..8)
                    .map(|c| {
                        let mut row: Vec<f32> = (0..k)
                            .map(|j| if j % 2 == 0 { 0.0 } else { -0.0 })
                            .collect();
                        row[0] = chunk.get(c).copied().unwrap_or(0.0);
                        row
                    })
                    .collect();
                let p = Problem {
                    label: format!(
                        "epilogue seed={seed} K={k} T={:?} scale={sc:e}",
                        chunk.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    ),
                    fam: Fam::K4,
                    ms: [8, 0, 0, 0],
                    k,
                    w: [w0, Vec::new(), Vec::new(), Vec::new()],
                    rows,
                    scale: sc,
                    act: true,
                };
                check_problem(&rig, &mut tally, &p, id, Shape::Windows, false);
                id += 1;
            }
        }
    }
    tally.finish("hc_unqualified_division", &refs);
    // Qualified rows must have reached every HC entry despite the mixed-batch refusals.
    for e in rig.entries.iter().filter(|e| e.fam == Fam::K4) {
        let (_, words, _) = tally.per.get(&e.symbol).copied().unwrap_or_default();
        assert!(
            words > 0,
            "{}: no qualified output word reached this entry",
            e.symbol
        );
    }
}

/// NaN / +-Inf activations and weights: neighbour rows stay bitwise the singleton's, every NaN output word is the
/// emitted reference's (the interpreter's modelled hardware NaN selection) and the PM c1 entry's, and the written
/// extent stays exact. Includes the HC epilogue with non-finite sums (SiLU of NaN / Inf, scale 0 against Inf).
#[test]
fn nonfinite_rows_stay_isolated_and_every_entry_agrees() {
    let ks: BTreeSet<usize> = [264usize, 250, 520, 33, 16, 72, 32, 8]
        .into_iter()
        .collect();
    let env = Env::new(&ks);
    let emus = env.pm.emulators();
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for (id, (ms, k)) in [
        ([3, 0, 5, 1], 264usize),
        ([0, 9, 0, 2], 250),
        ([4, 4, 0, 0], 520),
        ([1, 0, 0, 8], 33),
    ]
    .into_iter()
    .enumerate()
    {
        let p = make_x4(0x4000 + id as u64, ms, k, true);
        check_problem(&rig, &mut tally, &p, id, Shape::Core, id % 2 == 1);
    }
    for (id, (m, k, act)) in [
        (3usize, 64usize, false),
        (5, 288, true),
        (2, 128, true),
        (9, 128, false),
    ]
    .into_iter()
    .enumerate()
    {
        let p = make_hc(
            0x5000 + id as u64,
            m,
            k,
            act,
            [1.0f32, 0.5, 0.0, 3.7][id],
            true,
        );
        check_problem(&rig, &mut tally, &p, id, Shape::Core, false);
    }
    // Non-finite epilogue sums on dedicated rows.
    let specials = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_1234),
        f32::MAX,
        -f32::MAX,
        1.0,
        -1.0,
    ];
    for (id, &sc) in [1.0f32, 0.0, 1e30, -3.0].iter().enumerate() {
        let k = 32;
        let mut rng = Rng::new(0x6000 + id as u64);
        let mut w0 = gen_w(&mut rng, 4 * k);
        for n in 0..4 {
            w0[n * k] = 0x3f80;
        }
        let rows: Vec<Vec<f32>> = specials
            .iter()
            .map(|&t| {
                let mut row = vec![0.0f32; k];
                row[0] = t;
                row
            })
            .collect();
        let p = Problem {
            label: format!("epilogue-nonfinite scale={sc:e}"),
            fam: Fam::K4,
            ms: [4, 0, 0, 0],
            k,
            w: [w0, Vec::new(), Vec::new(), Vec::new()],
            rows,
            scale: sc,
            act: true,
        };
        check_problem(&rig, &mut tally, &p, id, Shape::Windows, false);
    }
    tally.finish("nonfinite", &refs);
}

#[test]
fn out_of_range_c_writes_nothing_and_terminates() {
    let ks: BTreeSet<usize> = [8usize].into_iter().collect();
    let env = Env::new(&ks);
    let emus = env.pm.emulators();
    let refs = Reference::new(&env.refprog, &env.hip);
    let rig = Rig {
        entries: &env.entries,
        emus: &emus,
        refs: &refs,
    };
    let mut tally = Tally::default();
    for (id, (ms, k)) in [
        ([2, 0, 3, 1], 264usize),
        ([9, 8, 0, 3], 33),
        ([0, 0, 0, 5], 8),
    ]
    .into_iter()
    .enumerate()
    {
        check_invalid_c(&rig, &mut tally, &make_x4(0x7000 + id as u64, ms, k, false));
    }
    for (id, (m, k, act)) in [(3usize, 64usize, false), (5, 320, true), (1, 32, true)]
        .into_iter()
        .enumerate()
    {
        check_invalid_c(
            &rig,
            &mut tally,
            &make_hc(0x7100 + id as u64, m, k, act, 0.5, false),
        );
    }
    tally.finish("invalid_c", &refs);
}

// ------------------------------------------------------------------------------------------------ reference census

/// Identity of the checked-in hipcc fixtures, the independent emitted reference and the status of the lifted hipcc
/// object on the interpreter. The emitted reference must itself be executable and, whenever the hipcc singleton also
/// is, must equal it word for word (the emitted DAG is then proven against the real object, not assumed).
#[test]
fn reference_identity_and_hipcc_object_status() {
    for fixture in [FIXTURE_QWEN4, FIXTURE_SINGLE] {
        match pinned_fixture(fixture) {
            Ok(bytes) => log(format!(
                "[reference] fixture {} sha256 {} ({} bytes)",
                fixture.0,
                fixture.1,
                bytes.len()
            )),
            Err(why) => log(format!(
                "[reference] ACTUAL OBJECT MISSING; no object-equivalence claim: {why}"
            )),
        }
    }
    let provenance = format!("{}/PROVENANCE.md", fixture_dir());
    match std::fs::read_to_string(&provenance) {
        Ok(text) => {
            for fixture in [FIXTURE_QWEN4, FIXTURE_SINGLE] {
                assert!(
                    text.contains(fixture.1),
                    "{provenance}: missing fixture SHA {}",
                    fixture.1
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            log(format!("[reference] optional provenance document unavailable: {provenance}; object SHA pins remain enforced"));
        }
        Err(e) => panic!("{provenance}: {e}"),
    }
    let ks: BTreeSet<usize> = [8, 33, 256, 264, 1003, 2560, 16, 640].into_iter().collect();
    let refprog = build_reference_program(&ks);
    for k in &refprog.kernels {
        let census = unsupported_ops(&refprog, &k.symbol.0).unwrap();
        assert!(
            census.is_empty(),
            "{}: the emitted reference uses opcodes without interpreter semantics: {census:?}",
            k.symbol.0
        );
    }
    let hip = reference_program();
    match &hip {
        Err(e) => log(format!(
            "[reference] hipcc object NOT LIFTED (exact reason): {e}"
        )),
        Ok(p) => {
            for sym in [SINGLE_SYMBOL, K4_SYMBOL, "gemv_bf16_xf32_x4"] {
                match unsupported_ops(p, sym) {
                    Ok(census) => log(format!(
                        "[reference] hipcc {sym}: {} opcode(s) without interpreter semantics: {census:?}",
                        census.len()
                    )),
                    Err(e) => log(format!("[reference] hipcc {sym}: {e}")),
                }
            }
        }
    }
    let refs = Reference::new(&refprog, &hip);
    log(format!("[reference] {}", refs.describe()));
    let mut bad = 0;
    let mut compared = [0u64; 2];
    let mut skipped: Vec<String> = Vec::new();
    let mut check = |p: &Problem| {
        let fi = usize::from(p.fam == Fam::K4);
        if !refs.hipcc_available(p.fam) {
            return;
        }
        for row in &p.rows {
            let emitted = refs
                .row(p, row)
                .unwrap_or_else(|e| panic!("{}: emitted reference: {e}", p.label));
            let hipcc = match refs.hipcc_row(p, row) {
                Some(Ok(h)) => h,
                Some(Err(e)) if is_semantic_gap(&e) => {
                    skipped.push(format!("{}: {e}", p.label));
                    continue;
                }
                Some(Err(e)) => {
                    bad += 1;
                    log(format!(
                        "[reference] hipcc object FAILED (not a documented semantic gap): {}: {e}",
                        p.label
                    ));
                    continue;
                }
                None => return,
            };
            for i in 0..p.matrices() {
                for n in 0..p.ms[i] {
                    compared[fi] += 1;
                    if emitted[i][n] != hipcc[i][n] {
                        bad += 1;
                        log(format!(
                            "[reference] emitted DAG != hipcc object: {} matrix {i} output {n}: emitted {:#010x} object {:#010x}",
                            p.label, emitted[i][n], hipcc[i][n]
                        ));
                    }
                }
            }
        }
    };
    for (id, k) in [8usize, 33, 256, 264, 1003, 2560].into_iter().enumerate() {
        check(&make_x4(0x8000 + id as u64, [5, 0, 3, 0], k, false));
    }
    for (id, (m, k)) in [(3usize, 64usize), (2, 1056), (4, 2560)]
        .into_iter()
        .enumerate()
    {
        for act in [false, true] {
            check(&make_hc(0x8100 + id as u64, m, k, act, 0.75, false));
        }
    }
    log(format!(
        "[reference] emitted DAG vs hipcc object: ordinary x4 compared {} words, HC k4 compared {} words (a family whose object is not executable is SKIPPED above with its exact reason), differences {bad}, per-row hipcc runtime skips {}",
        compared[0],
        compared[1],
        skipped.len()
    ));
    for s in skipped.iter().take(8) {
        log(format!("[reference] hipcc runtime skip: {s}"));
    }
    assert_eq!(
        bad, 0,
        "the emitted singleton DAG disagrees with the executable hipcc object"
    );
    if refs.hipcc_available(Fam::X4) {
        assert!(compared[0] > 0, "the ordinary hipcc singleton is executable but nothing was compared (all rows skipped: {skipped:?})");
    }
}

/// The PM entries themselves must be executable on the interpreter; any unsupported opcode is the exact blocker.
#[test]
fn pm_entries_are_executable_on_the_cpu_interpreter() {
    let pm = Pm::new();
    let mut blockers = Vec::new();
    for sym in SYMBOLS {
        let census = unsupported_ops(&pm.program, sym).unwrap();
        log(format!("[pm-census] {sym}: unsupported opcodes {census:?}"));
        if !census.is_empty() {
            blockers.push(format!("{sym}: {census:?}"));
        } else {
            Emulator::new(&pm.program, sym).unwrap_or_else(|e| panic!("{sym}: Emulator::new: {e}"));
        }
    }
    assert!(
        blockers.is_empty(),
        "PM entries use opcodes without interpreter semantics: {blockers:#?}"
    );
}

// ------------------------------------------------------------------------------------------------ incumbents (informational)

/// `gemv_bf16_xf32_x4_rows_r{R}` / `gemv_bf16_xf32_k4_rows_r{R}` (the HIP read-once rows incumbents compiled for
/// gfx1201, `R = C = 2..=8`) against the singleton reference: equal / mismatch counts and the first difference per
/// `R`. A mismatch is a finding for MultiCardLand, not a failure; an interpreter blocker is reported with the exact
/// opcodes. The lift of the hipcc fixture may itself fail: that is logged, not asserted.
#[test]
fn hip_rows_incumbents_vs_singleton_report() {
    let ks: BTreeSet<usize> = [250, 264, 2560].into_iter().collect();
    let refprog = build_reference_program(&ks);
    let hip = reference_program();
    let refs = Reference::new(&refprog, &hip);
    log(format!("[incumbent] {}", refs.describe()));
    let program = match &hip {
        Ok(p) => p,
        Err(e) => {
            log(format!(
                "[incumbent] hipcc fixture not lifted, no incumbent rows to analyse: {e}"
            ));
            return;
        }
    };
    let mut executable = 0;
    for r in 2..=8usize {
        for (fam, sym) in [
            (Fam::X4, format!("gemv_bf16_xf32_x4_rows_r{r}")),
            (Fam::K4, format!("gemv_bf16_xf32_k4_rows_r{r}")),
        ] {
            let census = match unsupported_ops(program, &sym) {
                Ok(c) => c,
                Err(e) => {
                    log(format!("[incumbent] {sym}: {e}"));
                    continue;
                }
            };
            if !census.is_empty() {
                log(format!("[incumbent] {sym}: NOT EXECUTABLE on the interpreter; unsupported opcodes {census:?}"));
                continue;
            }
            let emu = match Emulator::new(program, &sym) {
                Ok(e) => e,
                Err(e) => {
                    log(format!("[incumbent] {sym}: Emulator::new: {e}"));
                    continue;
                }
            };
            executable += 1;
            let cases: Vec<Problem> = match fam {
                Fam::X4 => [264usize, 250, 2560]
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| make_x4(0x9000 + (r * 8 + i) as u64, [5, 0, 3, 2], k, false))
                    .collect(),
                Fam::K4 => [
                    (3usize, 1056usize, false),
                    (3, 1056, true),
                    (2, 10240, true),
                ]
                .iter()
                .enumerate()
                .map(|(i, &(m, k, act))| {
                    make_hc(0x9100 + (r * 8 + i) as u64, m, k, act, 0.75, false)
                })
                .collect(),
            };
            for p in cases {
                let rows: Vec<&[f32]> = p.rows.iter().take(r).map(|x| x.as_slice()).collect();
                let (words, equal, first, kind) = run_incumbent(&emu, &refs, fam, &p, &rows);
                log(format!(
                    "[incumbent] {sym} C={r} {}: reference={kind} words={words} equal={equal} mismatch={} first={}",
                    p.label,
                    words - equal,
                    first.unwrap_or_else(|| "none".into())
                ));
            }
        }
    }
    log(format!(
        "[incumbent] {executable}/14 incumbent rows entries executable on the interpreter"
    ));
}

fn run_incumbent(
    emu: &Emulator,
    refs: &Reference,
    fam: Fam,
    p: &Problem,
    rows: &[&[f32]],
) -> (u64, u64, Option<String>, String) {
    let mut mem = Memory::default();
    let nm = p.matrices();
    let mut wptr = [0u64; 4];
    let mut yptr = [0u64; 4];
    for i in 0..nm {
        wptr[i] = map_guarded(
            &mut mem,
            &format!("w{i}"),
            wbase(i),
            u16_bytes(&p.w[i]),
            WGUARD,
        );
        let y: Vec<u8> = (0..rows.len() * p.ms[i])
            .flat_map(|_| SENT.to_le_bytes())
            .collect();
        yptr[i] = map_guarded(&mut mem, &format!("y{i}"), ybase(i), y, SENT);
    }
    let x: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let xptr = map_guarded(&mut mem, "x", XBASE, f32_bytes(&x), XGUARD);
    let mut ka = vec![0u8; 256];
    match fam {
        Fam::X4 => {
            for i in 0..4 {
                ka[8 * i..8 * i + 8].copy_from_slice(&wptr[i].to_le_bytes());
                ka[40 + 8 * i..48 + 8 * i].copy_from_slice(&yptr[i].to_le_bytes());
                ka[72 + 4 * i..76 + 4 * i].copy_from_slice(&(p.ms[i] as u32).to_le_bytes());
            }
            ka[32..40].copy_from_slice(&xptr.to_le_bytes());
            ka[88..92].copy_from_slice(&(p.k as u32).to_le_bytes());
        }
        Fam::K4 => {
            ka[0..8].copy_from_slice(&wptr[0].to_le_bytes());
            ka[8..16].copy_from_slice(&xptr.to_le_bytes());
            ka[16..24].copy_from_slice(&yptr[0].to_le_bytes());
            ka[24..28].copy_from_slice(&(p.ms[0] as u32).to_le_bytes());
            ka[28..32].copy_from_slice(&(p.k as u32).to_le_bytes());
            ka[32..36].copy_from_slice(&p.scale.to_bits().to_le_bytes());
            ka[36..40].copy_from_slice(&u32::from(p.act).to_le_bytes());
        }
    }
    mem.map("kernarg".into(), KERNARG, ka).unwrap();
    let (grid, block) = match fam {
        Fam::X4 => (p.ms.iter().sum::<usize>() as u32, 32),
        Fam::K4 => (p.ms[0] as u32, 128),
    };
    if let Err(e) = launch(emu, &mut mem, grid, block) {
        return (
            0,
            0,
            Some(format!("launch failed: {e}")),
            "none".to_string(),
        );
    }
    let (mut words_n, mut equal, mut first) = (0u64, 0u64, None);
    let mut kinds: BTreeSet<&str> = BTreeSet::new();
    for (c, row) in rows.iter().enumerate() {
        // Direct comparison with the actual hipcc singleton wherever the interpreter executes it; otherwise the
        // independent emitted VALU DAG. The kind is reported with every result.
        let (want, kind) = match refs.hipcc_row(p, row) {
            Some(Ok(w)) => (w, "hipcc-object (actual singleton)"),
            Some(Err(e)) => {
                return (
                    words_n,
                    equal,
                    Some(format!("reference failed: {e}")),
                    "hipcc-object (actual singleton)".to_string(),
                )
            }
            None => match refs.row(p, row) {
                Ok(w) => (
                    w,
                    "emitted independent VALU DAG (hipcc object not executable for this family)",
                ),
                Err(e) => return (
                    words_n,
                    equal,
                    Some(format!("reference failed: {e}")),
                    "emitted independent VALU DAG (hipcc object not executable for this family)"
                        .to_string(),
                ),
            },
        };
        kinds.insert(kind);
        for (i, want_i) in want[..nm].iter().enumerate() {
            let m = p.ms[i];
            let y = words(&region(&mem, &format!("y{i}")).bytes[GUARD..GUARD + 4 * rows.len() * m]);
            for n in 0..m {
                words_n += 1;
                let got = y[c * m + n];
                if got == want_i[n] {
                    equal += 1;
                } else if first.is_none() {
                    first = Some(format!(
                        "row {c} matrix {i} output {n}: incumbent {got:#010x} singleton {:#010x}",
                        want_i[n]
                    ));
                }
            }
        }
    }
    let kind = if kinds.is_empty() {
        "none".to_string()
    } else {
        kinds.into_iter().collect::<Vec<_>>().join(", ")
    };
    (words_n, equal, first, kind)
}

// ------------------------------------------------------------------------------------------------ structural tests

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

/// Instruction lines of one emitted entry (directives, labels, comments removed).
fn instructions(text: &str) -> Vec<String> {
    let head = text.split(".amdgpu_metadata").next().unwrap();
    let body = head.split_once(".text").map_or(head, |(_, b)| b);
    body.lines()
        .map(|l| {
            l.split(';')
                .next()
                .unwrap()
                .split("//")
                .next()
                .unwrap()
                .trim()
                .to_string()
        })
        .filter(|l| !l.is_empty() && !l.starts_with('.') && !l.ends_with(':'))
        .collect()
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

#[test]
fn module_record_is_frozen() {
    assert_eq!(MODULE, "lane_rows_bf16_pm_gfx1201");
    assert_eq!(
        SYMBOLS.to_vec(),
        EXPECTED_SYMBOLS,
        "exports and their order"
    );
    assert_eq!(lr::bundle_name(), "lane_rows_bf16_pm_gfx1201.hxaco");
    let n = EXPECTED_SYMBOLS.len();
    for (what, len) in [
        ("KERNARG_BYTES", KERNARG_BYTES.len()),
        ("KERNARGS", KERNARGS.len()),
        ("WORKGROUP_SIZE", WORKGROUP_SIZE.len()),
        ("LDS_BYTES", LDS_BYTES.len()),
        ("MAX_ROWS", MAX_ROWS.len()),
        ("GRID_RULE", GRID_RULE.len()),
        ("RECIPE", RECIPE.len()),
    ] {
        assert_eq!(len, n, "{what} must have one record per export");
    }
    let es = entries();
    for e in &es {
        let i = e.idx;
        assert_eq!(
            MAX_ROWS[i] as usize, EXPECTED_MAX_ROWS[i],
            "{}: MAX_ROWS",
            e.symbol
        );
        assert_eq!(
            e.cap, EXPECTED_MAX_ROWS[i],
            "{}: capacity class in the name",
            e.symbol
        );
        assert_eq!(
            RECIPE[i],
            e.base(),
            "{}: equal recipe string means same per-output DAG as the singleton",
            e.symbol
        );
        assert_eq!(WORKGROUP_SIZE[i], e.block(), "{}: workgroup size", e.symbol);
        assert!(!GRID_RULE[i].is_empty(), "{}: GRID_RULE", e.symbol);
        let args: Vec<(u32, u32)> = KERNARGS[i].to_vec();
        match e.fam {
            Fam::X4 => {
                let mut want: Vec<(u32, u32)> = (0..9).map(|j| (8 * j, 8)).collect();
                want.extend((0..6).map(|j| (72 + 4 * j, 4)));
                assert_eq!(args, want, "{}: ABI96", e.symbol);
                assert_eq!(KERNARG_BYTES[i], 96, "{}", e.symbol);
            }
            Fam::K4 => {
                let want = vec![
                    (0, 8),
                    (8, 8),
                    (16, 8),
                    (24, 4),
                    (28, 4),
                    (32, 4),
                    (36, 4),
                    (40, 4),
                ];
                assert_eq!(args, want, "{}: HIP k4 layout with C appended", e.symbol);
                assert!(
                    [44, 48].contains(&KERNARG_BYTES[i]),
                    "{}: kernarg bytes {}",
                    e.symbol,
                    KERNARG_BYTES[i]
                );
            }
        }
    }
    // All x4 entries (and all k4 entries) are the same recipe: equal strings, only the two frozen ones.
    let recipes: BTreeSet<&str> = RECIPE.iter().copied().collect();
    assert_eq!(recipes, BTreeSet::from([RECIPE_K4, RECIPE_X4]));
    // REFERENCE names the singleton objects it was read from by sha256 and symbol.
    let dbg = format!("{:?}", lr::REFERENCE);
    assert!(
        dbg.contains(FIXTURE_QWEN4.1),
        "REFERENCE must record the fixture sha256 {}: {dbg}",
        FIXTURE_QWEN4.1
    );
    for sym in ["gemv_bf16_xf32_x4", K4_SYMBOL] {
        assert!(
            dbg.contains(sym),
            "REFERENCE must name the singleton symbol {sym}: {dbg}"
        );
    }
    for i in 0..n {
        log(format!(
            "[record] {} block={} lds={} max_rows={} kernarg={} grid_rule={:?} recipe={}",
            SYMBOLS[i],
            WORKGROUP_SIZE[i],
            LDS_BYTES[i],
            MAX_ROWS[i],
            KERNARG_BYTES[i],
            GRID_RULE[i],
            RECIPE[i]
        ));
    }
}

#[test]
fn emission_is_deterministic_exact_gfx1201_and_resource_clean() {
    let (ea, ta, pa) = lr::emit_module().unwrap();
    let (_, tb, pb) = lr::emit_module().unwrap();
    assert_eq!(
        (&ta, &pa.s_text_sha256),
        (&tb, &pb.s_text_sha256),
        "emission must be deterministic"
    );
    assert_eq!(ea.len(), SYMBOLS.len());
    assert_eq!((pa.module.as_str(), pa.arch), (MODULE, ARCH));
    assert_eq!(pa.kernels.len(), SYMBOLS.len());
    assert!(ta.contains("amdgcn-amd-amdhsa--gfx1201"));
    assert_eq!(
        ta.matches(".fill 96, 4, 3214868480").count(),
        1,
        "one module s_code_end guard"
    );
    let forbidden_prefix = [
        "scratch_",
        "buffer_atomic",
        "global_atomic",
        "flat_",
        "v_wmma",
        "v_swmmac",
    ];
    let ds_ok = [
        "ds_load",
        "ds_store",
        "ds_bpermute",
        "ds_swizzle",
        "ds_permute",
        "ds_read",
        "ds_write",
    ];
    let es = entries();
    for (e, em) in es.iter().zip(&ea) {
        let sym = e.symbol;
        let t = &em.s_text;
        assert_eq!(em.proof.arch, ARCH, "{sym}");
        assert!(
            t.contains(&format!(".globl {sym}\n")) && t.contains(&format!(".name: {sym}\n")),
            "{sym}: symbol"
        );
        assert!(t.contains(".amdhsa_wavefront_size32 1"), "{sym}: wave32");
        assert!(
            t.contains(".private_segment_fixed_size: 0\n"),
            "{sym}: private segment"
        );
        assert!(
            t.contains(".sgpr_spill_count: 0\n") && t.contains(".vgpr_spill_count: 0\n"),
            "{sym}: spills"
        );
        let size = KERNARG_BYTES[e.idx];
        assert!(
            t.contains(&format!(".kernarg_segment_size: {size}\n")),
            "{sym}: kernarg size {size}"
        );
        assert_eq!(
            field(t, ".offset:"),
            KERNARGS[e.idx].iter().map(|a| a.0).collect::<Vec<_>>(),
            "{sym}: offsets"
        );
        assert_eq!(
            field(t, ".size:"),
            KERNARGS[e.idx].iter().map(|a| a.1).collect::<Vec<_>>(),
            "{sym}: sizes"
        );
        assert_eq!(
            field(t, ".max_flat_workgroup_size:"),
            [e.block()],
            "{sym}: block"
        );
        let lds = field(t, ".group_segment_fixed_size:");
        assert_eq!(lds, [LDS_BYTES[e.idx]], "{sym}: LDS record vs descriptor");
        let lds = lds[0] as usize;
        let cap = e.cap;
        let min_lds = match (e.fam, e.wg8) {
            (Fam::X4, false) => 0,
            (Fam::K4, false) => cap * 4 * 4,
            (Fam::X4, true) => cap * 256 * 4,
            (Fam::K4, true) => cap * 2 * 4 * 4,
        };
        assert!(
            lds >= min_lds && lds <= 65536,
            "{sym}: LDS {lds} outside [{min_lds}, 65536]"
        );
        let lines = instructions(t);
        assert!(!lines.is_empty(), "{sym}: no instructions found");
        for l in &lines {
            let m = mnemonic(l);
            assert!(
                !forbidden_prefix.iter().any(|f| m.starts_with(f)),
                "{sym}: forbidden `{l}`"
            );
            if m.starts_with("ds_") {
                assert!(
                    ds_ok.iter().any(|ok| m.starts_with(ok)),
                    "{sym}: LDS atomic or exotic op `{l}`"
                );
            }
            assert!(
                !(m.contains("atomic") || m.contains("_cmpswap") || m.contains("_pk_add_f")),
                "{sym}: atomic `{l}`"
            );
        }
        let barriers = lines
            .iter()
            .filter(|l| mnemonic(l).starts_with("s_barrier"))
            .count();
        if e.fam == Fam::X4 && !e.wg8 {
            assert_eq!(barriers, 0, "{sym}: a one-wave workgroup has no barrier");
        } else {
            assert!(
                barriers > 0,
                "{sym}: cross-wave data (X tile / quarter fold) needs a barrier"
            );
        }
        let stores = lines
            .iter()
            .filter(|l| {
                mnemonic(l).starts_with("global_store") || mnemonic(l).starts_with("buffer_store")
            })
            .count();
        assert!(stores > 0, "{sym}: no global store");
        log(format!(
            "[resources] {sym}: vgpr={:?} sgpr={:?} lds={lds} instructions={} barriers={barriers} stores={stores} loads_b128={} fma={} waits={}",
            field(t, ".vgpr_count:"),
            field(t, ".sgpr_count:"),
            lines.len(),
            lines.iter().filter(|l| mnemonic(l) == "global_load_b128").count(),
            lines.iter().filter(|l| l.contains("fmac_f32") || l.contains("fma_f32")).count(),
            em.shape.waits,
        ));
    }
    // c1 VGPR budget against the singleton it replaces: reported (not asserted, "where feasible").
    match reference_program() {
        Err(e) => log(format!(
            "[resources] singleton descriptor budget not available (hipcc fixture not lifted: {e})"
        )),
        Ok(prog) => {
            let vgpr_of = |sym: &str| -> Option<u32> {
                let k = prog.kernels.iter().find(|k| k.symbol.0 == sym)?;
                match &k.abi {
                    Abi::Hsa { descriptor, .. } => {
                        Some((descriptor.compute_pgm_rsrc1.0 & 0x3f) * 8 + 8)
                    }
                    _ => None,
                }
            };
            for (e, em) in es.iter().zip(&ea) {
                if e.cap != 1 {
                    continue;
                }
                let sym = if e.fam == Fam::X4 {
                    "gemv_bf16_xf32_x4"
                } else {
                    K4_SYMBOL
                };
                let got = field(&em.s_text, ".vgpr_count:");
                log(format!(
                    "[resources] c1 {} vgpr {:?} vs singleton {sym} descriptor-allocated vgpr {:?} => {}",
                    e.symbol,
                    got,
                    vgpr_of(sym),
                    match (got.first(), vgpr_of(sym)) {
                        (Some(a), Some(b)) if *a <= b => "within the singleton budget",
                        (Some(_), Some(_)) => "ABOVE the singleton budget (report)",
                        _ => "not comparable",
                    }
                ));
            }
        }
    }
}

/// First VGPR index of the first operand of `line` (`v16`, `v[16:19]`), if it is a VGPR.
fn first_vgpr(operands: &[&str]) -> Option<u32> {
    let r = operands.first()?.trim();
    let r = r.strip_prefix("v[").or_else(|| r.strip_prefix('v'))?;
    r.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

fn operands_of(line: &str) -> Vec<&str> {
    line.strip_prefix(mnemonic(line))
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .collect()
}

/// The weight decoder: the weight tile of every entry lives in `v16..v23` (`WS`); one `global_load_b128 v[16:19]`
/// (vector domain) or one 16-bit load into `v16` (scalar domain) fetches it, and the bf16 -> f32 widening is
/// `v_lshlrev_b32 v{16..23}, 16, v{16..19}` (even element) / `v_and_b32 v{16..23}, 0xffff0000, v{16..19}` (odd
/// element). Epilogue BF16 round-trip masks (`and 0xffff0000` on other registers) and X loads are not decoder ops.
fn decoder_census(text: &str) -> (usize, usize) {
    let mut widen = 0;
    let mut weight_loads = 0;
    for l in instructions(text) {
        let m = mnemonic(&l);
        let ops = operands_of(&l);
        if m.starts_with("global_load") || m.starts_with("buffer_load") {
            if first_vgpr(&ops) == Some(16) {
                weight_loads += 1;
            }
            continue;
        }
        let imm_ok = if m.starts_with("v_lshlrev_b32") {
            ops.get(1) == Some(&"16")
        } else if m.starts_with("v_and_b32") {
            ops.get(1) == Some(&"0xffff0000")
        } else {
            continue;
        };
        let dst = first_vgpr(&ops);
        let src = ops.get(2).and_then(|r| first_vgpr(std::slice::from_ref(r)));
        if imm_ok && matches!(dst, Some(16..=23)) && matches!(src, Some(16..=19)) {
            widen += 1;
        }
    }
    (widen, weight_loads)
}

/// The weight decoder (loads of the weight tile and the bf16 widening of it) is a property of the owned weight tile,
/// never of the capacity: every capacity class of one (family, schedule) owns the same tile and must emit the same
/// number of decoder widening ops AND the same number of weight loads. Only the loaded-weight decoder is counted:
/// not the epilogue round trips, not the activation loads (which scale with `C`).
#[test]
fn weight_widening_is_independent_of_the_row_capacity() {
    let (emitted, _, _) = lr::emit_module().unwrap();
    let es = entries();
    for fam in [Fam::X4, Fam::K4] {
        for wg8 in [false, true] {
            let group: Vec<(&Entry, usize, usize)> = es
                .iter()
                .filter(|e| e.fam == fam && e.wg8 == wg8)
                .map(|e| {
                    let (w, l) = decoder_census(&emitted[e.idx].s_text);
                    (e, w, l)
                })
                .collect();
            for (e, w, l) in &group {
                log(format!(
                    "[census] {}: decoder widening ops {w}, weight loads {l}",
                    e.symbol
                ));
                assert!(
                    *w > 0,
                    "{}: no decoder widening instruction recognised in the emitted text",
                    e.symbol
                );
                assert!(
                    *l > 0,
                    "{}: no weight load into v16 recognised in the emitted text",
                    e.symbol
                );
            }
            let (base_w, base_l) = (group[0].1, group[0].2);
            for (e, w, l) in &group {
                assert_eq!(
                    *w, base_w,
                    "{}: decoder widening count differs between capacity classes of one tile",
                    e.symbol
                );
                assert_eq!(
                    *l, base_l,
                    "{}: weight load count differs between capacity classes of one tile",
                    e.symbol
                );
            }
        }
    }
}

#[test]
fn rowtile_structural_abi_fixed_s1_resources_and_singleton_dag() {
    assert_eq!(
        lr::ROWTILE_SYMBOLS,
        [
            "gemv_bf16_xf32_rows_x4_t8_pm_gfx1201",
            "gemv_bf16_xf32_k4_rows_t8_pm_gfx1201",
        ]
    );
    for (i, hc) in [false, true].into_iter().enumerate() {
        let spec = lr::Entry {
            hc,
            wg8: false,
            cap: 8,
        };
        let tile = lr::build_rowtiles(spec).unwrap();
        let again = lr::build_rowtiles(spec).unwrap();
        let narrow = lr::build(spec).unwrap();
        let sym = lr::ROWTILE_SYMBOLS[i];
        let t = &tile.s_text;
        assert_eq!(t, &again.s_text, "{sym}: deterministic emission");
        assert_eq!(tile.proof.arch, ARCH);
        assert!(t.contains(&format!(".globl {sym}\n")));
        assert!(t.contains(&format!(".name: {sym}\n")));
        assert!(
            t.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
            "{sym}: y tile ABI"
        );
        assert!(t.contains(".amdhsa_wavefront_size32 1"));
        assert_eq!(tile.shape.next_free_vgpr, 120);
        assert_eq!(field(t, ".vgpr_count:"), [120]);
        // highest SGPR is TILE_R0 = s99 -> next_free 100, + VCC (2) = 102
        // (committed .s: both t8 kernels report `.sgpr_count: 102`).
        assert_eq!(field(t, ".sgpr_count:"), [102]);
        assert_eq!(
            field(t, ".max_flat_workgroup_size:"),
            [if hc { 128 } else { 32 }]
        );
        assert_eq!(
            field(t, ".group_segment_fixed_size:"),
            [if hc { 128 } else { 0 }]
        );
        assert_eq!(
            field(t, ".kernarg_segment_size:"),
            [if hc { 48 } else { 96 }]
        );
        assert_eq!(field(t, ".private_segment_fixed_size:"), [0]);
        assert_eq!(field(t, ".sgpr_spill_count:"), [0]);
        assert_eq!(field(t, ".vgpr_spill_count:"), [0]);
        let narrow_idx = if hc { 5 } else { 2 };
        assert_eq!(
            field(t, ".offset:"),
            KERNARGS[narrow_idx].iter().map(|a| a.0).collect::<Vec<_>>()
        );
        assert_eq!(
            field(t, ".size:"),
            KERNARGS[narrow_idx].iter().map(|a| a.1).collect::<Vec<_>>()
        );
        assert_eq!(
            decoder_census(t),
            decoder_census(&narrow.s_text),
            "{sym}: physical tile retains cap8 weight widening"
        );
        let dag = |text: &str| {
            instructions(text)
                .into_iter()
                .filter(|l| {
                    let m = mnemonic(l);
                    m.contains("_f32")
                        || m.contains("_bf16")
                        || m.starts_with("ds_swizzle")
                        || m.starts_with("ds_bpermute")
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(dag(t), dag(&narrow.s_text), "{sym}: float operands, chain/fold order, reduction and HC expansion must remain cap8-identical");
        let lines = instructions(t);
        for line in &lines {
            let m = mnemonic(line);
            assert!(
                !["scratch_", "flat_", "v_wmma", "v_swmmac"]
                    .iter()
                    .any(|prefix| m.starts_with(prefix))
                    && !m.contains("atomic")
                    && !m.contains("_cmpswap")
                    && !m.contains("_pk_add_f"),
                "{sym}: forbidden {line}"
            );
        }
        let barriers = lines
            .iter()
            .filter(|l| mnemonic(l).starts_with("s_barrier"))
            .count();
        assert_eq!(
            barriers > 0,
            hc,
            "{sym}: HC quarters alone require cross-wave barriers"
        );
    }
}

/// Rowtile producer discipline: the live-row register `s33` (SC) has exactly
/// ONE encoded definition per rowtile kernel --- the `s_min_u32 s33, s55, 8`
/// in `rowtile_domain` --- dominating every `s_cmp_le_u32 s33, r` user. The
/// validated raw total stays in its kernarg word (`s31` x4 / `s18` HC, never
/// copied into SC), the domain subtracts through the dead-at-entry `s55`
/// scratch, and non-rowtile kernels keep the copied `s_mov_b32 s33` prologue.
#[test]
fn rowtile_sc_has_a_single_dominating_definition() {
    // First-operand `s33` of a defining SALU op (compares/branches never define).
    let defines_s33 = |line: &str| {
        let m = mnemonic(line);
        if m.starts_with("s_cmp") || m.starts_with("s_cbranch") || m.starts_with("s_branch") {
            return false;
        }
        matches!(operands_of(line).first().map(|s| s.trim()), Some("s33"))
    };
    for (i, hc) in [false, true].into_iter().enumerate() {
        let spec = lr::Entry {
            hc,
            wg8: false,
            cap: 8,
        };
        let raw = if hc { 18 } else { 31 };
        let tile = lr::build_rowtiles(spec).unwrap().s_text;
        let narrow = lr::build(spec).unwrap().s_text;
        let sym = lr::ROWTILE_SYMBOLS[i];
        let lines = instructions(&tile);
        let defs: Vec<&String> = lines.iter().filter(|l| defines_s33(l.as_str())).collect();
        assert_eq!(
            defs.len(),
            1,
            "{sym}: s33 must have exactly one definition, got {defs:?}"
        );
        assert!(
            defs[0].starts_with("s_min_u32 s33, s55, 8"),
            "{sym}: single def must be the domain min, got {}",
            defs[0]
        );
        let count = |needle: &str| lines.iter().filter(|l| l.contains(needle)).count();
        assert_eq!(
            count(&format!("s_mov_b32 s33, s{raw}")),
            0,
            "{sym}: raw total must stay in s{raw}, never copied into s33"
        );
        assert_eq!(
            count("s33, s33"),
            0,
            "{sym}: no read-modify-write rebase of s33"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains(&format!("s_cmp_ge_u32 s99, s{raw}"))),
            "{sym}: dead-tile exit must compare the raw count"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains(&format!("s_sub_co_i32 s55, s{raw}, s99"))
                    || l.contains(&format!("s_sub_i32 s55, s{raw}, s99"))),
            "{sym}: domain subtraction must run through s55"
        );
        assert!(
            lines.iter().any(|l| l.contains("s_cmp_le_u32 s33,")),
            "{sym}: skip_dead_row users still read s33"
        );
        let def_pos = lines.iter().position(|l| defines_s33(l)).unwrap();
        for (j, l) in lines.iter().enumerate() {
            if j != def_pos && l.contains("s33") {
                assert!(
                    j > def_pos,
                    "{sym}: s33 use precedes its single definition: {l}"
                );
            }
        }
        let narrow_lines = instructions(&narrow);
        assert!(
            narrow_lines
                .iter()
                .any(|l| l == &format!("s_mov_b32 s33, s{raw}")),
            "{sym}: non-rowtile keeps the copied s33 prologue"
        );
    }
}

/// M7 (ungated half): the linked object lifts byte-exactly (`emit::module` reproduces it) and every export's
/// analysis has no obligations, no ambiguous delay windows and well-paired barriers.
#[test]
fn m7_lift_is_byte_exact_with_no_obligations_on_every_export() {
    assert_m7(&Pm::new(), &SYMBOLS);
}

#[test]
fn rowtile_m7_lift_is_byte_exact_with_no_obligations() {
    let env = Env::rowtiles(&BTreeSet::new());
    let symbols: Vec<&str> = env.entries.iter().map(|e| e.symbol).collect();
    assert_m7(&env.pm, &symbols);
}

fn assert_m7(pm: &Pm, symbols: &[&str]) {
    use peacemaker_ir::passes::{barriers, windows};
    let lifted = peacemaker_lift::lift_object(
        &pm.elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap();
    assert_eq!(
        peacemaker_lift::emit::module(&lifted.program).unwrap(),
        pm.elf,
        "emit::module must reproduce the linked object"
    );
    assert_eq!(pm.emitted.len(), symbols.len());
    for &sym in symbols {
        let kernel = lifted
            .program
            .kernels
            .iter()
            .find(|k| k.symbol.0 == sym)
            .unwrap_or_else(|| panic!("{sym}: not in the lifted object"));
        let analysis = peacemaker_ir::edit::analyze(lifted.program.clone(), &kernel.symbol)
            .unwrap_or_else(|e| panic!("{sym}: analyze: {e}"));
        let pairings = barriers::analyze(&kernel.body, lifted.program.target.arch)
            .unwrap_or_else(|e| panic!("{sym}: barriers: {e}"));
        let win =
            windows::check_windows(&kernel.body).unwrap_or_else(|e| panic!("{sym}: windows: {e}"));
        let mut obligations: Vec<String> = analysis
            .obligations
            .iter()
            .chain(&pairings.obligations)
            .map(|o| format!("{:?}/{}: {}", o.kind, o.rule_id, o.text))
            .collect();
        let ambiguous = win.delays.iter().filter(|d| d.is_ambiguous()).count();
        if ambiguous != 0 {
            obligations.push(format!("{ambiguous} ambiguous delay windows"));
        }
        log(format!(
            "[m7] {sym}: obligations={} barrier_pairs={} delays={}",
            obligations.len(),
            pairings.pairs.len(),
            win.delays.len()
        ));
        assert!(
            obligations.is_empty(),
            "{sym}: M7 obligations: {obligations:#?}"
        );
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check};

    /// Narrow and rowtile exports compose safely; every wait replays and every M7 lift is byte-exact.
    #[test]
    fn every_export_passes_wait_replay_and_pm_check_m7() {
        let (mut emitted, _, _) = lr::emit_module().unwrap();
        for hc in [false, true] {
            emitted.push(
                lr::build_rowtiles(lr::Entry {
                    hc,
                    wg8: false,
                    cap: 8,
                })
                .unwrap(),
            );
        }
        let (text, _) =
            peacemaker_isa::kernels::iu4_gemm::module(&emitted, "bf16_narrow_and_rowtiles")
                .unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-lane-rows-bf16-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        for symbol in SYMBOLS.into_iter().chain(lr::ROWTILE_SYMBOLS) {
            let m7 =
                pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            log(format!("[m7-toolchain] {symbol}: {m7}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
    }

    /// Actual `lr::emit_hc_quarter_diagnostic()` regression: the emitted helper exports the
    /// quarter symbol, materializes the fused-quarter plane strides 64/128/192/256, stores one
    /// payload register per quarter, its waits replay, and the natively assembled diagnostic ELF
    /// is M7 byte-exact with `{}` obligations and zero ambiguous delays.
    #[test]
    fn hc_quarter_diagnostic_passes_wait_replay_and_pm_check_m7() {
        const SYMBOL: &str = "gemv_bf16_xf32_k4_rows_t8_quarters_pm_gfx1201";
        let diag = lr::emit_hc_quarter_diagnostic().unwrap();
        assert_eq!(diag.proof.arch, ARCH, "{SYMBOL}: arch");
        assert!(
            diag.s_text.contains(&format!(".amdhsa_kernel {SYMBOL}")),
            "diagnostic must export {SYMBOL}"
        );
        for (q, lit) in ["64", "0x80", "0xc0", "0x100"].iter().enumerate() {
            assert!(
                diag.s_text.contains(&format!("s_mov_b32 s45, {lit}")),
                "quarter {q}: plane stride 64*{} must be materialized",
                q + 1
            );
        }
        for q in 0..4u8 {
            let reg = 12 + q;
            assert!(
                diag.s_text
                    .lines()
                    .any(|l| l.contains("global_store_b32") && l.contains(&format!(", v{reg},"))),
                "quarter {q}: fused payload v{reg} must be stored"
            );
        }
        ledger_replay::replay_waits(&diag.s_text, ARCH).unwrap();
        let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(
            std::slice::from_ref(&diag),
            "bf16_hc_quarter_diagnostic",
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-lane-rows-bf16-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("hc-quarter-diagnostic.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        let m7 = pm_check::m7(&co, ARCH.name(), SYMBOL).unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        log(format!("[m7-toolchain] {SYMBOL}: {m7}"));
        assert_eq!(
            (
                m7["lift"].as_str(),
                &m7["obligations"],
                &m7["ambiguous_delays"]
            ),
            (
                Some("byte-exact"),
                &serde_json::json!({}),
                &serde_json::json!(0)
            ),
            "{SYMBOL}"
        );
    }
}
