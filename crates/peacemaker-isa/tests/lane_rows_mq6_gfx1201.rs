// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase A (A1) MQ6G256V2 gfx1201 lane-row twins, module `lane_rows_mq6_pm_gfx1201`
//! (`CONTRACT-A.md` §2, §4): the frozen module record, ABI, resources and committed image, and the CPU byte
//! oracle on the peacemaker interpreter.
//!
//! **Oracle.** Every output word `(c, n)` of every entry (six S1 + four S2 `wg8`), at every live `C` of its
//! capacity class, must equal bit for bit the gfx1201 singleton's word for activation row `c` launched alone
//! (`C = 1`). The singleton is the hipcc object checked in under `tests/fixtures/gfx1201_singletons/mq6/`
//! (`gemv_mq6g256v2.elf`; the x4 module's vector code is identical, see its `PROVENANCE.md`). The reference path is
//!
//! 1. `lift_object(.., Frontend::Hipcc)` + `Emulator` when the interpreter can execute that object, else
//! 2. the **DAG transcription** of that object's disassembly (`true-cb/DAG-GFX1201-MQ6.md`; fixed-order `f32`
//!    `mul`/`mul_add` plus the swizzle/bpermute lane tree) — never a host dot product. The interpreter's refusal
//!    (verbatim `Emulator::new` / first failing `run_workgroup` error from real tiny launches) is printed as `REFERENCE-MODE ...` under `--nocapture`, and the transcription
//!    is cross-checked against the object whenever the interpreter becomes able to run it.
//!
//! Inputs are full-mantissa random F32 over a wide exponent band (random signs, rows of very different magnitude,
//! cancellation-heavy rows, signed zeros), random packed 6-bit codes and finite f16 scales/zeros including subnormals
//! and zeros. Every weight/x byte, every y guard word (NaN sentinel) and the exact written extent of `y` (byte
//! granular `WriteSet`) are checked after every launch. `LANE_ROWS_MQ6_FULL=1` widens the shape matrix.
//!
//! Informational (never fails): the HIP rows incumbent (`gemm_mq6g256v2_f32_rows_r{2..8}`, `_x4_r{2..8}`) against the
//! same reference, printed as `INCUMBENT ...`; and per-entry resources as `CENSUS ...`.
//!
//! Fixed-eight-row `rows*_t8` tests below are a separate, strict gate: total R is 1..=64, grid y is ceil(R/8),
//! and both lifted production singleton objects MUST execute. Their reference path never calls `ref_mode` or
//! substitutes a transcription. The transcription and PM C1 launches are additional, mandatory expectations.
//! These tests use 64 independently generated rows (not a repeated eight-row pool), exact/unmapped and poisoned
//! tails, and isolated workgroups to prove absolute row addressing and nonoverlapping write ownership.
use peacemaker_ir::emu::{semantic_class, Emulator, Launch, Memory, Region};
use peacemaker_ir::inst::{Abi, Frontend};
use peacemaker_ir::Program;
use peacemaker_isa::kernels::lane_rows_mq6_gfx1201::{
    self as module, GRID_RULE, KERNARGS, KERNARG_BYTES, LDS_BYTES, MAX_ROWS, MODULE, RECIPE,
    SYMBOLS, WORKGROUP_SIZE,
};
use peacemaker_isa::Arch;
use peacemaker_lift::{lift_object, Options};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

const ARCH: Arch = Arch::Gfx1201;
const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gfx1201_singletons/mq6"
);

// ------------------------------------------------------------------------------------------------ entry table

/// The ten frozen entries in module order: S1 `c1`, `xbatch_r4`, `xbatch_r8`, x4 `c1`, `r4`, `r8`, then S2 `wg8`
/// (single r4, single r8, x4 r4, x4 r8).
#[derive(Clone, Copy, Debug)]
struct Entry {
    idx: usize,
    x4: bool,
    cap: usize,
    wg8: bool,
}
const ENTRIES: [Entry; 10] = [
    Entry {
        idx: 0,
        x4: false,
        cap: 1,
        wg8: false,
    },
    Entry {
        idx: 1,
        x4: false,
        cap: 4,
        wg8: false,
    },
    Entry {
        idx: 2,
        x4: false,
        cap: 8,
        wg8: false,
    },
    Entry {
        idx: 3,
        x4: true,
        cap: 1,
        wg8: false,
    },
    Entry {
        idx: 4,
        x4: true,
        cap: 4,
        wg8: false,
    },
    Entry {
        idx: 5,
        x4: true,
        cap: 8,
        wg8: false,
    },
    Entry {
        idx: 6,
        x4: false,
        cap: 4,
        wg8: true,
    },
    Entry {
        idx: 7,
        x4: false,
        cap: 8,
        wg8: true,
    },
    Entry {
        idx: 8,
        x4: true,
        cap: 4,
        wg8: true,
    },
    Entry {
        idx: 9,
        x4: true,
        cap: 8,
        wg8: true,
    },
];
const NAMES: [&str; 10] = [
    "gemv_mq6g256v2_c1_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r4_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r8_pm_gfx1201",
    "gemv_mq6g256v2_x4_c1_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r4_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r8_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r4_wg8_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r8_wg8_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r4_wg8_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r8_wg8_pm_gfx1201",
];
const SINGLE_ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)];
const X4_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 8),
    (32, 8),
    (40, 8),
    (48, 8),
    (56, 8),
    (64, 8),
    (72, 4),
    (76, 4),
    (80, 4),
    (84, 4),
    (88, 4),
    (92, 4),
];

impl Entry {
    fn name(&self) -> &'static str {
        NAMES[self.idx]
    }
    fn block(&self) -> u32 {
        if self.wg8 {
            256
        } else {
            32
        }
    }
    /// `[C][K256]` F32 tile: `cap * 256 * 4` bytes for the wg8 schedule, none for S1.
    fn lds(&self) -> u32 {
        if self.wg8 {
            self.cap as u32 * 1024
        } else {
            0
        }
    }
    fn kernarg_bytes(&self) -> u32 {
        if self.x4 {
            96
        } else {
            36
        }
    }
    fn grid(&self, ms: &[usize; 4]) -> [u32; 3] {
        let per = |m: usize| if self.wg8 { m.div_ceil(8) } else { m };
        let n: usize = if self.x4 {
            ms.iter().map(|&m| per(m)).sum()
        } else {
            per(ms[0])
        };
        [n as u32, 1, 1]
    }
}

// ------------------------------------------------------------------------------------------------ data

const SENTINEL: u32 = 0x7fc0_a5a5;
const GUARD: usize = 16; // words on each side of every y extent
const KERNARG: u64 = 0x1_0000;
const A_BASE: u64 = 0x1000_0000;
const A_STRIDE: u64 = 0x1000_0000;
const X_BASE: u64 = 0x6000_0000;
const Y_BASE: u64 = 0x7000_0000;
const Y_STRIDE: u64 = 0x0100_0000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

fn words_to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// A finite f16 (never inf/NaN): mostly normals, some subnormals, some signed zeros.
fn f16_bits(rng: &mut Rng, hi_exp: usize) -> u16 {
    let sign = (rng.below(2) as u16) << 15;
    let mant = (rng.next() & 0x3ff) as u16;
    match rng.below(20) {
        0 => sign,                   // signed zero
        1..=3 => sign | mant.max(1), // subnormal
        _ => sign | (((1 + rng.below(hi_exp)) as u16) << 10) | mant,
    }
}

/// `m` rows of `k / 256` 200-byte groups: `[s0 z0 s1 z1]` f16 then 192 random payload bytes.
fn mq6_matrix(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let mut out = Vec::with_capacity(m * k / 256 * 200);
    for _ in 0..m * (k / 256) {
        for h in 0..4 {
            let bits = if h % 2 == 0 {
                f16_bits(rng, 22)
            } else {
                f16_bits(rng, 24)
            };
            out.extend_from_slice(&bits.to_le_bytes());
        }
        for _ in 0..192 {
            out.push(rng.next() as u8);
        }
    }
    out
}

/// Full-mantissa F32 with a random sign and unbiased exponent in `[lo, hi]`.
fn f32_in(rng: &mut Rng, lo: i32, hi: i32) -> u32 {
    let e = lo + rng.below((hi - lo + 1) as usize) as i32;
    ((rng.below(2) as u32) << 31) | (((127 + e) as u32) << 23) | (rng.next() as u32 & 0x7f_ffff)
}

/// One activation row of the pool: plain, magnitude-shifted, cancellation-heavy, signed-zero sprinkled, all zeros.
fn x_row(rng: &mut Rng, slot: usize, k: usize) -> Vec<u32> {
    let shift = (slot as i32 - 4) * 8;
    (0..k)
        .map(|_| match slot {
            0 | 4 => f32_in(rng, -24, 24),
            1 | 5 => f32_in(rng, shift - 12, shift + 12),
            // Cancellation: rare huge terms among small ones, random signs.
            2 | 7 => {
                if rng.below(10) == 0 {
                    f32_in(rng, 28, 34)
                } else {
                    f32_in(rng, -12, 0)
                }
            }
            // Signed zeros among normals.
            3 => match rng.below(10) {
                0..=1 => (rng.below(2) as u32) << 31,
                _ => f32_in(rng, -20, 20),
            },
            // All signed zeros: the sign of every product and sum is what is compared.
            _ => (rng.below(2) as u32) << 31,
        })
        .collect()
}
const POOL_SLOTS: [usize; 8] = [0, 1, 2, 3, 6, 5, 4, 7];

// ------------------------------------------------------------------------------------------------ reference

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 0x3ff) as u32;
    match e {
        0 => {
            let v = m as f32 * f32::from_bits(0x3380_0000); // m * 2^-24, exact
            if s != 0 {
                -v
            } else {
                v
            }
        }
        31 => f32::from_bits(s | 0x7f80_0000 | (m << 13)),
        _ => f32::from_bits(s | ((e + 112) << 23) | (m << 13)),
    }
}

/// The DAG of `gemv_mq6g256v2.elf` (hipcc gfx1201), one output row, transcribed from its disassembly: lane `tid`,
/// group `g` covers weights `8*tid..8*tid+7` of the group; `w_i = fma_mix(sc_f16, f32(q_i), zp_f16)` (one rounding);
/// `t = x1*w1`, then `fma(x0,w0,t)`, `fma(x2,w2,t)`..`fma(x7,w7,t)`; `acc[g % 4] = acc[g % 4] + t` from `+0`;
/// lane value `(a0+a1)+(a3+a2)`; own+other tree `shfl 16`, then `8, 4, 2, 1`, lane 0 stores.
fn ref_row(a: &[u8], row: usize, k: usize, x: &[u32]) -> u32 {
    let groups = k / 256;
    let rp = &a[row * groups * 200..(row + 1) * groups * 200];
    let mut lane = [0f32; 32];
    for (tid, out) in lane.iter_mut().enumerate() {
        let mut acc = [0f32; 4];
        for g in 0..groups {
            let gp = &rp[g * 200..g * 200 + 200];
            let h = if tid < 16 { 0 } else { 4 };
            let hdr = u32::from_le_bytes([gp[h], gp[h + 1], gp[h + 2], gp[h + 3]]);
            let sc = f16_to_f32(hdr as u16);
            let zp = f16_to_f32((hdr >> 16) as u16);
            let mut word = 0u64;
            for (i, b) in gp[8 + 6 * tid..8 + 6 * tid + 6].iter().enumerate() {
                word |= (*b as u64) << (8 * i);
            }
            let mut w = [0f32; 8];
            for (i, wi) in w.iter_mut().enumerate() {
                *wi = sc.mul_add(((word >> (6 * i)) & 63) as f32, zp);
            }
            let base = g * 256 + tid * 8;
            let xv = |i: usize| f32::from_bits(x[base + i]);
            let mut t = xv(1) * w[1];
            for i in [0usize, 2, 3, 4, 5, 6, 7] {
                t = xv(i).mul_add(w[i], t);
            }
            acc[g % 4] += t;
        }
        *out = (acc[0] + acc[1]) + (acc[3] + acc[2]);
    }
    let mut v = lane;
    for off in [16usize, 8, 4, 2, 1] {
        let old = v;
        for i in 0..32 {
            let src = if i + off < 32 { i + off } else { i };
            v[i] = old[i] + old[src];
        }
    }
    v[0].to_bits()
}

/// Every mnemonic of `symbol` the interpreter has no semantics for (empty = runnable).
fn unsupported_ops(program: &Program, symbol: &str) -> Vec<String> {
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .unwrap_or_else(|| panic!("{symbol}: not in lifted program"));
    let mut out = BTreeSet::new();
    for id in &kernel.body.layout {
        let inst = kernel.body.insts.get(*id).expect("layout instruction");
        match peacemaker_ir::isa::lookup(program.target.arch, inst.op, inst.form) {
            None => {
                out.insert(format!("<undecodable {:?}/{:?}>", inst.op, inst.form));
            }
            Some(row) => match semantic_class(row.name) {
                Some("wait" | "control" | "memory" | "alu") => {}
                Some(class) => {
                    out.insert(format!("{} (class {class})", row.name));
                }
                None => {
                    out.insert(row.name.to_string());
                }
            },
        }
    }
    out.into_iter().collect()
}

fn lift_fixture(file: &str) -> Result<Program, String> {
    let path = format!("{FIXTURES}/{file}");
    let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .map(|l| l.program)
    .map_err(|e| format!("lift {file}: {e}"))
}

/// How `ref` outputs are produced for this run.
struct RefMode {
    /// `Ok`: the lifted hipcc singleton, executed by the interpreter. `Err`: the DAG transcription is used instead;
    /// the string is the interpreter's refusal, recorded verbatim.
    reference: Result<Program, String>,
}
const SINGLETON: &str = "gemv_mq6g256v2";

/// Executes the lifted hipcc singleton on the interpreter: `Err` is the interpreter's verbatim refusal
/// (`Emulator::new` or the first instruction `run_workgroup` cannot execute), never a static prediction.
fn hipcc_try(p: &Program, a: &[u8], m: usize, k: usize, x: &[u32]) -> Result<Vec<u32>, String> {
    let emu = Emulator::new(p, SINGLETON).map_err(|e| format!("Emulator::new: {e}"))?;
    let mut mem = Memory::default();
    let mut ka = vec![0u8; 96];
    ka[0..8].copy_from_slice(&A_BASE.to_le_bytes());
    ka[8..16].copy_from_slice(&X_BASE.to_le_bytes());
    ka[16..24].copy_from_slice(&Y_BASE.to_le_bytes());
    ka[24..28].copy_from_slice(&(m as u32).to_le_bytes());
    ka[28..32].copy_from_slice(&(k as u32).to_le_bytes());
    mem.map("kernarg".into(), KERNARG, ka)?;
    mem.map("a".into(), A_BASE, a.to_vec())?;
    mem.map("x".into(), X_BASE, words_to_bytes(x))?;
    mem.map("y".into(), Y_BASE, vec![0xa5; m * 4])?;
    for gx in 0..m as u32 {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KERNARG,
                workgroup: [gx, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .map_err(|e| format!("launch wg {gx}: {e}"))?;
    }
    Ok(bytes_to_words(&region(&mem, "y").bytes))
}

/// Lift, `Emulator::new`, then REAL tiny launches (K = 256 tail-only, K = 768 tail 3, K = 1024 quad loop): the
/// hipcc object is the reference only if all of them execute. Any refusal is recorded verbatim.
fn ref_mode() -> RefMode {
    let p = match lift_fixture("gemv_mq6g256v2.elf") {
        Err(e) => {
            return RefMode {
                reference: Err(format!("lift refused: {e}")),
            }
        }
        Ok(p) => p,
    };
    let mut rng = Rng(0x9B0BE);
    for (k, m) in [(256usize, 2usize), (768, 1), (1024, 1)] {
        let a = mq6_matrix(&mut rng, m, k);
        let x = x_row(&mut rng, 0, k);
        if let Err(why) = hipcc_try(&p, &a, m, k, &x) {
            return RefMode {
                reference: Err(format!("{why} (probe K={k} M={m})")),
            };
        }
    }
    RefMode { reference: Ok(p) }
}
impl RefMode {
    fn describe(&self) -> String {
        match &self.reference {
            Ok(_) => "hipcc-object (lifted gemv_mq6g256v2.elf executed by the interpreter)".into(),
            Err(why) => {
                format!("DAG-transcription (interpreter cannot run the hipcc object: {why})")
            }
        }
    }
    /// Singleton outputs of every row of `a` against activation row `x` (`C = 1`).
    fn outputs(&self, a: &[u8], m: usize, k: usize, x: &[u32]) -> Vec<u32> {
        match &self.reference {
            Err(_) => (0..m).map(|n| ref_row(a, n, k, x)).collect(),
            Ok(p) => hipcc_try(p, a, m, k, x).unwrap_or_else(|e| panic!("hipcc singleton: {e}")),
        }
    }
}

// ------------------------------------------------------------------------------------------------ problem + launch

struct Problem {
    k: usize,
    ms: [usize; 4],
    mats: [Vec<u8>; 4],
}

struct Case {
    p: Problem,
    pool: Vec<Vec<u32>>,
    /// `c1[row][matrix]` = PM `c1` entry's words for pool row `row` alone; empty until `attach_c1`.
    c1: Vec<[Vec<u32>; 4]>,
    /// `refy[row][matrix]` = the singleton's `M_i` words for pool row `row` alone (hipcc object when executable).
    refy: Vec<[Vec<u32>; 4]>,
    /// `dag[row][matrix]` = the independent DAG transcription's words (always computed).
    dag: Vec<[Vec<u32>; 4]>,
    /// `x4ref[row][matrix]` = the production x4 singleton's words; empty unless attached (x4 cases only).
    x4ref: Vec<[Vec<u32>; 4]>,
}

fn make_case(seed: u64, k: usize, ms: [usize; 4], mode: &RefMode) -> Case {
    let mut rng = Rng(seed | 1);
    for _ in 0..4 {
        rng.next();
    }
    let mats: [Vec<u8>; 4] = std::array::from_fn(|i| {
        if ms[i] > 0 {
            mq6_matrix(&mut rng, ms[i], k)
        } else {
            Vec::new()
        }
    });
    let pool: Vec<Vec<u32>> = POOL_SLOTS.iter().map(|&s| x_row(&mut rng, s, k)).collect();
    // The DAG transcription is computed for every pool row regardless of mode; with the hipcc object as `refy` it
    // is a second, independent expectation (and equals `refy` itself in fallback mode).
    let dag: Vec<[Vec<u32>; 4]> = pool
        .iter()
        .map(|x| {
            std::array::from_fn(|i| {
                if ms[i] > 0 {
                    (0..ms[i]).map(|n| ref_row(&mats[i], n, k, x)).collect()
                } else {
                    Vec::new()
                }
            })
        })
        .collect();
    let refy = match &mode.reference {
        Err(_) => dag.clone(),
        Ok(_) => pool
            .iter()
            .map(|x| {
                std::array::from_fn(|i| {
                    if ms[i] > 0 {
                        mode.outputs(&mats[i], ms[i], k, x)
                    } else {
                        Vec::new()
                    }
                })
            })
            .collect(),
    };
    Case {
        p: Problem { k, ms, mats },
        pool,
        refy,
        dag,
        x4ref: Vec::new(),
        c1: Vec::new(),
    }
}

fn region<'a>(mem: &'a Memory, name: &str) -> &'a Region {
    mem.regions
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("region {name}"))
}

fn a_ptr(i: usize) -> u64 {
    A_BASE + i as u64 * A_STRIDE
}
fn y_ptr(i: usize) -> u64 {
    Y_BASE + i as u64 * Y_STRIDE + (GUARD * 4) as u64
}

fn kernarg_single(p: &Problem, c: Option<i32>) -> Vec<u8> {
    let mut ka = vec![0u8; if c.is_some() { 36 } else { 32 }];
    ka[0..8].copy_from_slice(&a_ptr(0).to_le_bytes());
    ka[8..16].copy_from_slice(&X_BASE.to_le_bytes());
    ka[16..24].copy_from_slice(&y_ptr(0).to_le_bytes());
    ka[24..28].copy_from_slice(&(p.ms[0] as u32).to_le_bytes());
    ka[28..32].copy_from_slice(&(p.k as u32).to_le_bytes());
    if let Some(c) = c {
        ka[32..36].copy_from_slice(&c.to_le_bytes());
    }
    ka
}

fn kernarg_x4(p: &Problem, c: Option<i32>) -> Vec<u8> {
    let mut ka = vec![0u8; if c.is_some() { 96 } else { 92 }];
    for i in 0..4 {
        ka[8 * i..8 * i + 8].copy_from_slice(&a_ptr(i).to_le_bytes());
        ka[40 + 8 * i..48 + 8 * i].copy_from_slice(&y_ptr(i).to_le_bytes());
        ka[72 + 4 * i..76 + 4 * i].copy_from_slice(&(p.ms[i] as u32).to_le_bytes());
    }
    ka[32..40].copy_from_slice(&X_BASE.to_le_bytes());
    ka[88..92].copy_from_slice(&(p.k as u32).to_le_bytes());
    if let Some(c) = c {
        ka[92..96].copy_from_slice(&c.to_le_bytes());
    }
    ka
}

/// Weights exactly sized (a read past a matrix is an unmapped access), x exactly sized, y sentinel with guards.
/// A zero-height matrix maps an EMPTY region, so any access to it faults.
fn build_mem(p: &Problem, x: &[u32], c_alloc: usize, kernarg: Vec<u8>) -> Memory {
    let mut mem = Memory::default();
    mem.map("kernarg".into(), KERNARG, kernarg).unwrap();
    for i in 0..4 {
        mem.map(format!("a{i}"), a_ptr(i), p.mats[i].clone())
            .unwrap();
    }
    mem.map("x".into(), X_BASE, words_to_bytes(x)).unwrap();
    for i in 0..4 {
        let n = GUARD + c_alloc * p.ms[i] + GUARD;
        mem.map(
            format!("y{i}"),
            Y_BASE + i as u64 * Y_STRIDE,
            words_to_bytes(&vec![SENTINEL; n]),
        )
        .unwrap();
    }
    mem
}

fn launch_grid_res(
    emu: &Emulator,
    mem: &mut Memory,
    grid: [u32; 3],
    block: u32,
) -> Result<(), String> {
    for gy in 0..grid[1] {
        for gx in 0..grid[0] {
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

fn launch_grid(emu: &Emulator, mem: &mut Memory, tag: &str, grid: [u32; 3], block: u32) {
    launch_grid_res(emu, mem, grid, block).unwrap_or_else(|e| panic!("{tag} {e}"));
}

static COMPARED: AtomicUsize = AtomicUsize::new(0);

/// Weights/x unchanged and unwritten; per y: guards intact, exactly the live `C * M_i` words written, nothing else.
fn verify(mem: &Memory, p: &Problem, x: &[u32], cap: usize, c_alloc: usize, c_arg: i32, tag: &str) {
    for i in 0..4 {
        let r = region(mem, &format!("a{i}"));
        assert!(
            r.bytes == p.mats[i] && r.written.count() == 0,
            "{tag}: matrix {i} modified"
        );
    }
    let rx = region(mem, "x");
    assert!(
        rx.bytes == words_to_bytes(x) && rx.written.count() == 0,
        "{tag}: x modified"
    );
    let live = if c_arg >= 1 && c_arg as usize <= cap {
        c_arg as usize
    } else {
        0
    };
    for i in 0..4 {
        let (r, m) = (region(mem, &format!("y{i}")), p.ms[i]);
        let w = bytes_to_words(&r.bytes);
        assert!(
            w[..GUARD].iter().all(|v| *v == SENTINEL),
            "{tag}: y{i} low guard clobbered"
        );
        assert!(
            w[GUARD + c_alloc * m..].iter().all(|v| *v == SENTINEL),
            "{tag}: y{i} high guard clobbered"
        );
        assert!(
            w[GUARD + live * m..GUARD + c_alloc * m]
                .iter()
                .all(|v| *v == SENTINEL),
            "{tag}: y{i} wrote past the live rows"
        );
        let extent = live * m * 4;
        assert_eq!(
            r.written.count(),
            extent,
            "{tag}: y{i} written bytes != live extent {extent}"
        );
        assert!(
            (0..extent).all(|b| r.written.contains(GUARD * 4 + b)),
            "{tag}: y{i} live extent not fully written"
        );
    }
}

/// One PM launch at runtime `C = c_arg`, `c_alloc` rows allocated in x/y.
fn run_pm(
    emus: &[Emulator],
    e: &Entry,
    p: &Problem,
    x: &[u32],
    c_alloc: usize,
    c_arg: i32,
    tag: &str,
) -> Memory {
    let ka = if e.x4 {
        kernarg_x4(p, Some(c_arg))
    } else {
        kernarg_single(p, Some(c_arg))
    };
    let mut mem = build_mem(p, x, c_alloc, ka);
    launch_grid(&emus[e.idx], &mut mem, tag, e.grid(&p.ms), e.block());
    verify(&mem, p, x, e.cap, c_alloc, c_arg, tag);
    mem
}

fn y_word(mem: &Memory, i: usize, c: usize, n: usize, m: usize) -> u32 {
    let r = region(mem, &format!("y{i}"));
    let o = (GUARD + c * m + n) * 4;
    u32::from_le_bytes([r.bytes[o], r.bytes[o + 1], r.bytes[o + 2], r.bytes[o + 3]])
}

/// PM `c1` (`C = 1`) per pool row, independent launches: `case.c1[row][matrix]`. Mandatory second oracle: every
/// entry at every `C` must equal these AND the singleton reference. The `c1` entry itself must equal the reference.
fn attach_c1(emus: &[Emulator], case: &mut Case, x4: bool) {
    let e = ENTRIES.iter().find(|e| e.cap == 1 && e.x4 == x4).unwrap();
    let mut c1: Vec<[Vec<u32>; 4]> = Vec::with_capacity(case.pool.len());
    for (r, x) in case.pool.iter().enumerate() {
        let tag = format!(
            "{} K={} M={:?} PM c1 row {r}",
            e.name(),
            case.p.k,
            case.p.ms
        );
        let mem = run_pm(emus, e, &case.p, x, 1, 1, &tag);
        let words: [Vec<u32>; 4] = std::array::from_fn(|i| {
            (0..case.p.ms[i])
                .map(|col| y_word(&mem, i, 0, col, case.p.ms[i]))
                .collect()
        });
        for (i, w) in words.iter().enumerate() {
            assert!(
                *w == case.refy[r][i],
                "{tag}: PM c1 matrix {i} differs from the singleton reference"
            );
            assert!(
                *w == case.dag[r][i],
                "{tag}: PM c1 matrix {i} differs from the DAG transcription"
            );
        }
        c1.push(words);
    }
    case.c1 = c1;
}

/// The production x4 singleton (`qwen4_gemv_mq6g256v2.elf`, symbol `gemv_mq6g256v2_x4`).
const X4_SINGLETON: &str = "gemv_mq6g256v2_x4";

/// One launch of the production x4 singleton on activation row `x` (92 B kernarg, `C` absent, grid `sum Mi`, block 32):
/// per matrix the `M_i` output words.
fn run_x4_ref(emu: &Emulator, p: &Problem, x: &[u32]) -> Result<[Vec<u32>; 4], String> {
    let mut mem = build_mem(p, x, 1, kernarg_x4(p, None));
    let height: usize = p.ms.iter().sum();
    launch_grid_res(emu, &mut mem, [height as u32, 1, 1], 32)?;
    Ok(std::array::from_fn(|i| {
        (0..p.ms[i])
            .map(|col| y_word(&mem, i, 0, col, p.ms[i]))
            .collect()
    }))
}

/// Lift the x4 fixture, build the emulator and run REAL probe launches (zero-height and odd slots, quad and tail
/// groups). `Err` is the verbatim refusal; the oracle then states the skip instead of silently dropping the reference.
fn x4_ref_program() -> Result<Program, String> {
    let prog = lift_fixture("qwen4_gemv_mq6g256v2.elf")?;
    let emu = Emulator::new(&prog, X4_SINGLETON)
        .map_err(|e| format!("Emulator::new({X4_SINGLETON}): {e}"))?;
    let mut rng = Rng(0x4E4);
    for (k, ms) in [(256usize, [2usize, 1, 0, 1]), (1280, [0, 3, 1, 0])] {
        let mats: [Vec<u8>; 4] = std::array::from_fn(|i| {
            if ms[i] > 0 {
                mq6_matrix(&mut rng, ms[i], k)
            } else {
                Vec::new()
            }
        });
        let p = Problem { k, ms, mats };
        let x = x_row(&mut rng, 0, k);
        run_x4_ref(&emu, &p, &x).map_err(|e| format!("{e} (x4 probe K={k} M={ms:?})"))?;
    }
    Ok(prog)
}

/// Execute the production x4 singleton per pool row for this case; it must equal the single-matrix reference and the
/// DAG transcription, and is then a third expectation `compare` asserts for every PM output.
fn attach_x4_ref(emu: &Emulator, case: &mut Case) {
    let mut out = Vec::with_capacity(case.pool.len());
    for (r, x) in case.pool.iter().enumerate() {
        let words = run_x4_ref(emu, &case.p, x).unwrap_or_else(|e| {
            panic!(
                "production x4 singleton K={} M={:?} row {r}: {e}",
                case.p.k, case.p.ms
            )
        });
        for (i, w) in words.iter().enumerate() {
            assert!(*w == case.refy[r][i], "x4 K={} M={:?} row {r} matrix {i}: production x4 singleton != single-matrix reference", case.p.k, case.p.ms);
            assert!(
                *w == case.dag[r][i],
                "x4 K={} M={:?} row {r} matrix {i}: production x4 singleton != DAG transcription",
                case.p.k,
                case.p.ms
            );
        }
        out.push(words);
    }
    case.x4ref = out;
}

/// Slot `s` of `mem` must hold pool row `rows[s]`'s singleton words for every matrix and column.
fn compare(mem: &Memory, case: &Case, rows: &[usize], tag: &str) -> usize {
    let mut n = 0;
    for (s, &r) in rows.iter().enumerate() {
        for i in 0..4 {
            let m = case.p.ms[i];
            for col in 0..m {
                let (got, want) = (y_word(mem, i, s, col, m), case.refy[r][i][col]);
                assert_eq!(got, want, "{tag}: slot {s} (row {r}) matrix {i} col {col}: {got:#010x} != singleton {want:#010x}");
                let dagw = case.dag[r][i][col];
                assert_eq!(got, dagw, "{tag}: slot {s} (row {r}) matrix {i} col {col}: {got:#010x} != DAG transcription {dagw:#010x}");
                if !case.x4ref.is_empty() {
                    let xw = case.x4ref[r][i][col];
                    assert_eq!(got, xw, "{tag}: slot {s} (row {r}) matrix {i} col {col}: {got:#010x} != production x4 singleton {xw:#010x}");
                }
                if !case.c1.is_empty() {
                    let c1w = case.c1[r][i][col];
                    assert_eq!(got, c1w, "{tag}: slot {s} (row {r}) matrix {i} col {col}: {got:#010x} != PM c1 per-row launch {c1w:#010x}");
                }
                n += 1;
            }
        }
    }
    n
}

/// Rows for slots of a launch with `c` live rows: distinct pool rows, different per `c`.
fn rows_for(c: usize) -> Vec<usize> {
    (0..c).map(|s| (s * 3 + c) % 8).collect()
}

fn run_rows(emus: &[Emulator], e: &Entry, case: &Case, rows: &[usize], tag: &str) -> usize {
    let x: Vec<u32> = rows
        .iter()
        .flat_map(|&r| case.pool[r].iter().copied())
        .collect();
    let mem = run_pm(emus, e, &case.p, &x, rows.len(), rows.len() as i32, tag);
    compare(&mem, case, rows, tag)
}

/// Every live `C` of the entry; with `deep`: lane permutations, the same row in every slot, all slots one row.
fn check_entry(emus: &[Emulator], e: &Entry, case: &Case, deep: bool, rng: &mut Rng) -> usize {
    let tag = |what: &str| format!("{} K={} M={:?} {what}", e.name(), case.p.k, case.p.ms);
    let mut n = 0;
    for c in 1..=e.cap {
        n += run_rows(emus, e, case, &rows_for(c), &tag(&format!("C={c}")));
    }
    if deep {
        let mut perm: Vec<usize> = (0..8).collect();
        for t in 0..2 {
            rng.shuffle(&mut perm);
            n += run_rows(emus, e, case, &perm[..e.cap], &tag(&format!("perm{t}")));
        }
        if e.cap >= 2 {
            for s in 0..e.cap {
                let star = (s * 5 + 2) % 8;
                let mut rows: Vec<usize> = (0..e.cap).map(|j| (star + 1 + j) % 8).collect();
                rows[s] = star;
                // `star` may also appear elsewhere in `rows`; every slot is still checked against its own row.
                n += run_rows(
                    emus,
                    e,
                    case,
                    &rows,
                    &tag(&format!("row {star} in slot {s}")),
                );
            }
            let star = rng.below(8);
            n += run_rows(
                emus,
                e,
                case,
                &vec![star; e.cap],
                &tag(&format!("all slots row {star}")),
            );
        }
    }
    n
}

// ------------------------------------------------------------------------------------------------ the PM rig

struct Rig {
    program: Program,
}
impl Rig {
    fn new() -> Rig {
        let (_, text, _) = module::emit_module().unwrap();
        let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
        let program = lift_object(
            &elf,
            Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Rig { program }
    }
    fn emus(&self) -> Vec<Emulator<'_>> {
        SYMBOLS
            .iter()
            .map(|s| Emulator::new(&self.program, s).unwrap_or_else(|e| panic!("{s}: {e}")))
            .collect()
    }
}

fn full() -> bool {
    std::env::var("LANE_ROWS_MQ6_FULL").is_ok_and(|v| v == "1")
}

/// `(K, [M..])` single-matrix matrix: K group counts 1, 2, 3, 4, 5, 7, 10 (0..3 mod 4, with and without a quad loop),
/// M around the wave / eight-output tiles; FULL crosses everything and adds K = 6144.
fn single_shapes() -> Vec<(usize, Vec<usize>)> {
    let all = vec![1usize, 2, 3, 4, 5, 7, 8, 9];
    let mut v = vec![
        (256, all.clone()),
        (512, vec![1, 5, 8]),
        (768, vec![2, 4, 9]),
        (1024, vec![3, 7, 8]),
        (1280, vec![1, 4, 9]),
        (1792, vec![2, 5]),
        (2560, vec![3, 8]),
    ];
    if full() {
        v = [256, 512, 768, 1024, 1280, 1792, 2560]
            .iter()
            .map(|&k| (k, all.clone()))
            .collect();
        v.push((6144, vec![1, 4, 9]));
    }
    v
}

/// x4 height tuples: zero-height and odd-height slots in every position.
fn x4_tuples() -> Vec<[usize; 4]> {
    let mut v = vec![
        [5, 3, 1, 9],
        [9, 0, 7, 2],
        [1, 1, 1, 1],
        [4, 4, 4, 4],
        [8, 9, 0, 3],
        [0, 0, 0, 1],
        [0, 0, 7, 0],
        [0, 2, 0, 0],
        [3, 0, 0, 0],
        [0, 8, 5, 0],
        [7, 0, 0, 9],
        [2, 1, 0, 5],
    ];
    if full() {
        for m in [1usize, 3, 5, 8, 9] {
            for slot in 0..4 {
                let mut t = [0usize; 4];
                t[slot] = m;
                v.push(t);
            }
        }
    }
    v
}
fn x4_k(i: usize) -> usize {
    const KS: [usize; 7] = [256, 512, 768, 1024, 1280, 1792, 2560];
    KS[i % KS.len()]
}

/// One `ORACLE` line per entry of the selected family, after the sweeps passed: every assertion in `compare` /
/// `verify` aborts the test on the first mismatch, so reaching here means `mismatches=0` for the counted words.
fn print_oracle_table(per_entry: &[usize; 10], pick: impl Fn(&Entry) -> bool) {
    for e in ENTRIES.iter().filter(|e| pick(e)) {
        println!(
            "ORACLE {} C=1..{} words={} mismatches=0",
            e.name(),
            e.cap,
            per_entry[e.idx]
        );
    }
}

// ------------------------------------------------------------------------------------------------ oracle tests

#[test]
fn single_matrix_entries_equal_the_singleton_at_every_c() {
    let mode = ref_mode();
    println!("REFERENCE-MODE {}", mode.describe());
    let rig = Rig::new();
    let emus = rig.emus();
    let mut words = 0usize;
    let mut sweeps = 0usize;
    let mut per_entry = [0usize; 10];
    let mut idx = 0u64;
    for (k, ms) in single_shapes() {
        for m in ms {
            idx += 1;
            let mut case = make_case(
                0x51_0001 + (k as u64) * 131 + m as u64 * 7,
                k,
                [m, 0, 0, 0],
                &mode,
            );
            attach_c1(&emus, &mut case, false);
            let mut rng = Rng(0xA1 + idx);
            for e in ENTRIES.iter().filter(|e| !e.x4) {
                let n = check_entry(&emus, e, &case, idx.is_multiple_of(3), &mut rng);
                per_entry[e.idx] += n;
                words += n;
                sweeps += 1;
            }
        }
    }
    COMPARED.fetch_add(words, Ordering::Relaxed);
    println!("MQ6 single entries: {words} output words equal BOTH the singleton reference and the PM c1 per-row launches ({sweeps} entry-case sweeps, reference {})", mode.describe());
    print_oracle_table(&per_entry, |e| !e.x4);
}

#[test]
fn x4_entries_equal_the_singleton_at_every_c() {
    let mode = ref_mode();
    println!("REFERENCE-MODE {}", mode.describe());
    let x4prog = x4_ref_program();
    match &x4prog {
        Ok(_) => println!("REFERENCE-MODE-X4 production x4 singleton (lifted qwen4_gemv_mq6g256v2.elf `{X4_SINGLETON}`, 92 B ABI without C) executed per pool row for every x4 case"),
        Err(why) => println!("REFERENCE-MODE-X4 SKIPPED: {why}"),
    }
    println!("ORACLE-INPUTS every PM output is asserted == DAG transcription AND == {} AND == PM c1 per-row launches{}", mode.describe(), if x4prog.is_ok() { " AND == production x4 singleton" } else { "" });
    let x4emu = x4prog
        .as_ref()
        .ok()
        .map(|p| Emulator::new(p, X4_SINGLETON).unwrap());
    let rig = Rig::new();
    let emus = rig.emus();
    let mut words = 0usize;
    let mut sweeps = 0usize;
    let mut per_entry = [0usize; 10];
    for (i, ms) in x4_tuples().into_iter().enumerate() {
        let k = x4_k(i);
        let mut case = make_case(0x54_0001 + i as u64 * 977, k, ms, &mode);
        if let Some(emu) = &x4emu {
            attach_x4_ref(emu, &mut case);
        }
        attach_c1(&emus, &mut case, true);
        let mut rng = Rng(0xA4 + i as u64);
        for e in ENTRIES.iter().filter(|e| e.x4) {
            let n = check_entry(&emus, e, &case, i % 3 == 0, &mut rng);
            per_entry[e.idx] += n;
            words += n;
            sweeps += 1;
        }
    }
    COMPARED.fetch_add(words, Ordering::Relaxed);
    println!("MQ6 x4 entries: {words} output words equal BOTH the singleton reference and the PM c1 per-row launches ({sweeps} entry-case sweeps, reference {})", mode.describe());
    print_oracle_table(&per_entry, |e| e.x4);
}

/// A runtime `C` outside the entry's range (0, negative, above the capacity) returns before any store, any x read
/// and, for wg8, before any barrier: not one byte of y is written and nothing deadlocks.
#[test]
fn out_of_range_c_returns_before_any_store() {
    let mode = ref_mode();
    let rig = Rig::new();
    let emus = rig.emus();
    let mut checked = 0;
    for e in &ENTRIES {
        let ms = if e.x4 { [3, 0, 9, 1] } else { [9, 0, 0, 0] };
        let case = make_case(0x0C_0001 + e.idx as u64, 512, ms, &mode);
        let bad = [0i32, -1, e.cap as i32 + 1, 9, i32::MAX, i32::MIN];
        for c_arg in bad {
            if (1..=e.cap as i32).contains(&c_arg) {
                continue;
            }
            // One row allocated in x: a kernel that reads rows it does not own faults.
            let x = case.pool[0].clone();
            let tag = format!("{} C={c_arg}", e.name());
            run_pm(&emus, e, &case.p, &x, 1, c_arg, &tag);
            checked += 1;
        }
    }
    println!("MQ6 out-of-range C: {checked} launches wrote nothing");
}

/// The transcription is validated two ways: independent decode known answers (integer-exact, below), and the hipcc
/// object itself whenever the interpreter can run it. The interpreter's exact refusal is recorded here.
#[test]
fn reference_status_and_transcription_cross_check() {
    let mode = ref_mode();
    println!("REFERENCE-MODE {}", mode.describe());
    match &mode.reference {
        Ok(_) => {
            let t = RefMode {
                reference: Err(String::new()),
            };
            let mut rng = Rng(0xC0FFEE);
            let mut n = 0;
            for (k, m) in [(256usize, 3usize), (768, 4), (1280, 2), (2560, 5)] {
                let a = mq6_matrix(&mut rng, m, k);
                for slot in POOL_SLOTS {
                    let x = x_row(&mut rng, slot, k);
                    assert_eq!(
                        mode.outputs(&a, m, k, &x),
                        t.outputs(&a, m, k, &x),
                        "transcription != hipcc object K={k} M={m} slot {slot}"
                    );
                    n += m;
                }
            }
            println!("TRANSCRIPTION == hipcc object on {n} words");
        }
        Err(why) => {
            // Documented fallback (CONTRACT-A §4.1): the interpreter names the instruction(s) it lacks (or the lift
            // refusal is the finding). Fail only if the refusal is not an instruction/lift-level one.
            assert!(
                why.starts_with("Emulator::new:")
                    || why.starts_with("launch wg")
                    || why.starts_with("lift refused:"),
                "unexpected reference refusal: {why}"
            );
            // Informational only: what the static opcode table lacks (never decides the mode).
            if let Ok(p) = lift_fixture("gemv_mq6g256v2.elf") {
                println!(
                    "STATIC-SEMANTICS-TBL lacks (informational, not authoritative): {}",
                    unsupported_ops(&p, SINGLETON).join(", ")
                );
            }
        }
    }
}

/// Independent bit-stream decode of the 6-bit codes and the half-split headers, integer-exact so any summation order
/// gives the same answer: validates the transcription's decoder and lane assignment.
#[test]
fn transcription_decoder_known_answers() {
    let mut rng = Rng(0xDEC0DE);
    for k in [256usize, 512, 1280] {
        let m = 2;
        let groups = k / 256;
        let mut a = Vec::new();
        for _ in 0..m * groups {
            // half 0: w = 1*q + 0.5, half 1: w = 2*q - 0.25
            for h in [0x3c00u16, 0x3800, 0x4000, 0xb400] {
                a.extend_from_slice(&h.to_le_bytes());
            }
            a.extend((0..192).map(|_| rng.next() as u8));
        }
        let x: Vec<u32> = (0..k).map(|j| ((j % 7 + 1) as f32).to_bits()).collect();
        for row in 0..m {
            let mut want = 0f64;
            for g in 0..groups {
                let gp = &a[(row * groups + g) * 200..][..200];
                for j in 0..256 {
                    let bit = 6 * j;
                    let mut q = 0u32;
                    for b in 0..6 {
                        let p = bit + b;
                        q |= (((gp[8 + p / 8] >> (p % 8)) & 1) as u32) << b;
                    }
                    let w = if j < 128 {
                        q as f64 + 0.5
                    } else {
                        2.0 * q as f64 - 0.25
                    };
                    want += w * ((g * 256 + j) % 7 + 1) as f64;
                }
            }
            assert_eq!(
                f32::from_bits(ref_row(&a, row, k, &x)) as f64,
                want,
                "K={k} row {row}"
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------ HIP rows incumbent (informational)

/// Runs a rows-incumbent symbol (`x` is `[C][K]`, `y` is `[C][M]` exactly sized, kernargs without `C`) and
/// returns per-matrix words `[C][M_i]`.
fn run_incumbent(
    emu: &Emulator,
    x4: bool,
    case: &Case,
    rows: &[usize],
) -> Result<Vec<Vec<u32>>, String> {
    let c = rows.len();
    let p = &case.p;
    let ka = if x4 {
        kernarg_x4(p, None)
    } else {
        kernarg_single(p, None)
    };
    let x: Vec<u32> = rows
        .iter()
        .flat_map(|&r| case.pool[r].iter().copied())
        .collect();
    let mut mem = build_mem(p, &x, c, ka);
    let height: usize = if x4 { p.ms.iter().sum() } else { p.ms[0] };
    launch_grid_res(emu, &mut mem, [height as u32, 1, 1], 32)?;
    Ok((0..4)
        .map(|i| {
            let m = p.ms[i];
            (0..c * m)
                .map(|j| y_word(&mem, i, j / m, j % m, m))
                .collect()
        })
        .collect())
}

#[test]
fn hip_rows_incumbent_vs_singleton_report() {
    let mode = ref_mode();
    println!("INCUMBENT reference: {}", mode.describe());
    let program = match lift_fixture("gemm_mq6g256v2_f32_rows.elf") {
        Ok(p) => p,
        Err(e) => {
            println!("INCUMBENT mq6 rows: not liftable ({e}); verdict left to H0");
            return;
        }
    };
    let cases = [
        (512usize, [5usize, 0, 0, 0]),
        (1280, [9, 0, 0, 0]),
        (768, [3, 0, 0, 0]),
    ];
    let x4_cases = [(768usize, [5usize, 3, 0, 9]), (512, [0, 2, 7, 1])];
    for c in 2..=8usize {
        for (x4, name) in [
            (false, format!("gemm_mq6g256v2_f32_rows_r{c}")),
            (true, format!("gemm_mq6g256v2_f32_rows_x4_r{c}")),
        ] {
            // No static prefilter: actually construct the emulator and launch; the refusal is whatever it reports.
            let emu = match Emulator::new(&program, &name) {
                Ok(e) => e,
                Err(e) => {
                    println!("INCUMBENT {name}: not emulable (Emulator::new: {e})");
                    continue;
                }
            };
            let mut refused: Option<String> = None;
            let shapes: Vec<(usize, [usize; 4])> = if x4 {
                x4_cases.to_vec()
            } else {
                cases.to_vec()
            };
            let (mut equal, mut total) = (0usize, 0usize);
            for (si, (k, ms)) in shapes.into_iter().enumerate() {
                let case = make_case(0x1C_0000 + si as u64 * 31 + c as u64, k, ms, &mode);
                let rows = rows_for(c);
                let got = match run_incumbent(&emu, x4, &case, &rows) {
                    Ok(g) => g,
                    Err(e) => {
                        refused = Some(e);
                        break;
                    }
                };
                for (s, &r) in rows.iter().enumerate() {
                    for i in 0..4 {
                        let m = ms[i];
                        for col in 0..m {
                            total += 1;
                            equal += usize::from(got[i][s * m + col] == case.refy[r][i][col]);
                        }
                    }
                }
            }
            if let Some(e) = refused {
                println!("INCUMBENT {name} C={c}: not emulable (launch {e})");
                continue;
            }
            let verdict = if equal == total { "EQUAL" } else { "MISMATCH" };
            println!(
                "INCUMBENT {name} C={c}: {verdict} ({equal}/{total} words equal the singleton)"
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------ structural

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

#[test]
fn module_record_is_the_frozen_contract() {
    assert_eq!(MODULE, "lane_rows_mq6_pm_gfx1201");
    assert_eq!(SYMBOLS.len(), 10);
    assert_eq!(SYMBOLS[..], NAMES[..], "symbols and order");
    assert_eq!(module::bundle_name(), "lane_rows_mq6_pm_gfx1201.hxaco");
    assert_eq!(KERNARG_BYTES.len(), 10);
    for e in &ENTRIES {
        let i = e.idx;
        assert_eq!(
            KERNARG_BYTES[i],
            e.kernarg_bytes(),
            "{}: kernarg bytes",
            e.name()
        );
        let want = if e.x4 { X4_ARGS } else { SINGLE_ARGS };
        assert_eq!(
            KERNARGS[i],
            want,
            "{}: kernarg (offset, size) list",
            e.name()
        );
        assert_eq!(WORKGROUP_SIZE[i], e.block(), "{}: workgroup size", e.name());
        assert_eq!(LDS_BYTES[i], e.lds(), "{}: LDS", e.name());
        assert_eq!(MAX_ROWS[i] as usize, e.cap, "{}: max rows", e.name());
        // `max(Mi, 0)` is the emitter's spelling of a non-negative height.
        let rule: String = GRID_RULE[i]
            .to_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .replace("max(mi,0)", "mi");
        if e.wg8 {
            let want = if e.x4 { "sumceil(mi/8)" } else { "ceil(m/8)" };
            assert!(
                rule.contains(want),
                "{}: grid rule {:?} lacks {want}",
                e.name(),
                GRID_RULE[i]
            );
        } else if e.x4 {
            assert!(
                rule.contains("sum") && !rule.contains('/'),
                "{}: grid rule {:?}",
                e.name(),
                GRID_RULE[i]
            );
        } else {
            assert!(
                rule.contains('m') && !rule.contains('/'),
                "{}: grid rule {:?}",
                e.name(),
                GRID_RULE[i]
            );
        }
    }
    // Equal recipe strings mean "same per-output DAG as the singleton": all ten are one DAG.
    assert!(
        RECIPE.iter().all(|r| *r == RECIPE[0]),
        "recipes differ: {RECIPE:?}"
    );
    assert!(
        RECIPE[0].contains("mq6g256v2") && RECIPE[0].contains("gfx1201"),
        "recipe {:?}",
        RECIPE[0]
    );
    let (emitted, text, proof) = module::emit_module().unwrap();
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!((emitted.len(), proof.kernels.len()), (10, 10));
    for symbol in SYMBOLS {
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
    // Frozen order in the object itself.
    let order: Vec<usize> = SYMBOLS
        .iter()
        .map(|s| {
            text.find(&format!(".amdhsa_kernel {s}\n"))
                .unwrap_or_else(|| panic!("{s}: no descriptor"))
        })
        .collect();
    assert!(
        order.windows(2).all(|w| w[0] < w[1]),
        "descriptors out of frozen order"
    );
}

#[test]
fn emission_is_deterministic_exact_gfx1201_wave32_without_scratch() {
    let (ea, ta, pa) = module::emit_module().unwrap();
    let (_, tb, pb) = module::emit_module().unwrap();
    assert_eq!((&ta, &pa.s_text_sha256), (&tb, &pb.s_text_sha256));
    assert!(ta.contains("amdgcn-amd-amdhsa--gfx1201"));
    assert!(
        !ta.contains("gfx1151") && !ta.contains("gfx1100"),
        "module mentions another ISA"
    );
    for (e, symbol) in ea.iter().zip(SYMBOLS) {
        assert_eq!(e.proof.arch, ARCH, "{symbol}");
        let t = &e.s_text;
        assert!(t.contains(".amdhsa_wavefront_size32 1"), "{symbol}: wave32");
        assert!(
            t.contains(".private_segment_fixed_size: 0\n"),
            "{symbol}: private segment"
        );
        assert!(
            !t.contains(".amdhsa_private_segment_fixed_size ")
                || t.contains(".amdhsa_private_segment_fixed_size 0"),
            "{symbol}: private segment descriptor"
        );
        assert!(
            t.contains(".sgpr_spill_count: 0\n") && t.contains(".vgpr_spill_count: 0\n"),
            "{symbol}: spills"
        );
        assert!(
            !t.lines()
                .any(|l| l.trim().starts_with("scratch_") || l.trim().starts_with("flat_")),
            "{symbol}: scratch/flat access"
        );
        assert!(
            !t.lines().any(|l| l.contains("atomic"))
                && !t.contains("ds_add_f")
                && !t.contains("pk_add"),
            "{symbol}: atomics"
        );
    }
}

#[test]
fn kernarg_abi_and_launch_bounds_match_the_contract() {
    let (emitted, _, _) = module::emit_module().unwrap();
    for (e, ent) in emitted.iter().zip(&ENTRIES) {
        let (t, name) = (&e.s_text, ent.name());
        assert!(
            t.contains(&format!(".kernarg_segment_size: {}\n", ent.kernarg_bytes())),
            "{name}: kernarg size"
        );
        let args = if ent.x4 { X4_ARGS } else { SINGLE_ARGS };
        assert_eq!(
            field(t, ".offset:"),
            args.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{name}: kernarg offsets"
        );
        assert_eq!(
            field(t, ".size:"),
            args.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{name}: kernarg sizes"
        );
        let (last_off, last_size) = *args.last().unwrap();
        assert_eq!(
            last_off + last_size,
            ent.kernarg_bytes(),
            "{name}: arguments cover the block"
        );
        assert_eq!(
            field(t, ".max_flat_workgroup_size:"),
            [ent.block()],
            "{name}: workgroup size"
        );
        assert_eq!(
            field(t, ".group_segment_fixed_size:"),
            [ent.lds()],
            "{name}: LDS"
        );
    }
}

/// Barriers only at uniform points. Hard failures: a barrier while exec is masked, a vcc/exec-conditional forward
/// branch over a barrier. Scalar-conditional forward branches over barriers (uniform tail groups, early exits) are
/// listed as NOTE lines; the ragged-M oracle cases prove the invalid waves still stage X and arrive.
fn check_barriers(name: &str, text: &str) -> usize {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let is_barrier = |l: &str| l.starts_with("s_barrier");
    let barriers: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| is_barrier(l))
        .map(|(i, _)| i)
        .collect();
    let mut masked = false;
    for (i, l) in lines.iter().enumerate() {
        if l.starts_with("v_cmpx_")
            || l.contains("saveexec")
            || l.starts_with("s_and_b32 exec_lo,")
            || l.starts_with("s_and_b64 exec,")
        {
            masked = true;
        }
        if l.starts_with("s_mov_b32 exec_lo,")
            || l.starts_with("s_or_b32 exec_lo,")
            || l.starts_with("s_mov_b64 exec,")
            || l.starts_with("s_or_b64 exec,")
        {
            masked = false;
        }
        if is_barrier(l) {
            assert!(
                !masked,
                "{name}: barrier at text line {i} while exec is masked"
            );
        }
    }
    for (i, l) in lines.iter().enumerate() {
        let Some(rest) = l.strip_prefix("s_cbranch_") else {
            continue;
        };
        let target = l.rsplit(' ').next().unwrap();
        let Some(t) = lines.iter().position(|x| *x == format!("{target}:")) else {
            continue;
        };
        if t <= i {
            continue; // loop back edge
        }
        if !barriers.iter().any(|&b| b > i && b < t) {
            continue;
        }
        assert!(
            rest.starts_with("scc"),
            "{name}: divergent branch `{l}` jumps over a barrier"
        );
        println!("NOTE {name}: scalar branch `{l}` skips a barrier (must be workgroup-uniform by construction)");
    }
    barriers.len()
}

#[test]
fn resource_census_and_barrier_contract() {
    let (emitted, _, _) = module::emit_module().unwrap();
    let reference = lift_fixture("gemv_mq6g256v2.elf").ok().and_then(|p| {
        p.kernels
            .iter()
            .find(|k| k.symbol.0 == SINGLETON)
            .and_then(|k| match &k.abi {
                Abi::Hsa { metadata, .. } => {
                    Some((metadata.parsed.vgpr_count, metadata.parsed.sgpr_count))
                }
                Abi::Raw(_) => None,
            })
    });
    let mut census: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    for (e, ent) in emitted.iter().zip(&ENTRIES) {
        let (t, name) = (&e.s_text, ent.name());
        let vgpr = field(t, ".vgpr_count:");
        let sgpr = field(t, ".sgpr_count:");
        assert_eq!((vgpr.len(), sgpr.len()), (1, 1), "{name}: resource counts");
        let nbar = check_barriers(name, t);
        println!(
            "CENSUS {name}: vgpr={} sgpr={} lds={} block={} barriers={nbar} fma_mix={} cvt_ubyte0={}",
            vgpr[0], sgpr[0], ent.lds(), ent.block(), t.matches("v_fma_mix").count(), t.matches("v_cvt_f32_ubyte0").count()
        );
        if ent.wg8 {
            assert!(
                nbar >= 2,
                "{name}: cooperative X staging needs publish/consume barriers (found {nbar})"
            );
        } else {
            assert_eq!(
                field(t, ".group_segment_fixed_size:"),
                [0],
                "{name}: S1 has no LDS"
            );
        }
        if ent.cap == 1 {
            match reference {
                Some((rv, rs)) => {
                    println!("CENSUS {name}: reference singleton vgpr={rv} sgpr={rs}");
                    if vgpr[0] > rv {
                        println!("REPORT {name}: c1 uses {} VGPR > reference {rv} (CONTRACT-A 4.6 asks for <= where feasible)", vgpr[0]);
                    }
                }
                None => println!(
                    "CENSUS {name}: reference VGPR count unavailable (fixture did not lift)"
                ),
            }
        }
        census.push(if ent.wg8 {
            // S2 has no per-C dispatch: the whole export is one fixed owned tile (4 stream bodies).
            vec![tally(t)]
        } else {
            let secs = s1_sections(t);
            assert_eq!(
                secs.len(),
                ent.cap,
                "{name}: expected one unrolled section per live row count 1..={}",
                ent.cap
            );
            secs.iter().map(|s| tally(s)).collect()
        });
    }
    // Weight loads and dequant of one owned tile are independent of C: every per-C section of every S1 export (single
    // and x4 alike) has the same (fma_mix, cvt_ubyte0, buffer_load) census, equal across the capacity siblings; the
    // wg8 exports (no dispatch) have equal whole-export census across r4/r8.
    println!("CENSUS per-section (fma_mix, cvt_ubyte0, weight buffer_loads): {census:?}");
    for sib in [&[0usize, 1, 2][..], &[3, 4, 5], &[6, 7], &[8, 9]] {
        let want = census[sib[0]][0];
        assert!(
            want.0 > 0 && want.1 > 0 && want.2 > 0,
            "{}: no dequant/weight loads counted: {want:?}",
            NAMES[sib[0]]
        );
        for &i in sib {
            for (c, sec) in census[i].iter().enumerate() {
                assert_eq!(*sec, want, "{}: section for C={} has census {sec:?}, sibling baseline {want:?} (weight loads/dequant must not depend on C)", NAMES[i], c + 1);
            }
        }
    }
}

/// `(v_fma_mix, v_cvt_f32_ubyte0, buffer_load)` instruction counts of an assembly slice.
fn tally(text: &str) -> (usize, usize, usize) {
    let n = |p: &str| {
        text.lines()
            .filter(|l| l.trim_start().starts_with(p))
            .count()
    };
    (n("v_fma_mix"), n("v_cvt_f32_ubyte0"), n("buffer_load"))
}

/// The S1 per-live-row-count sections: each ends at its `.L..._skip{nb}:` label (the dispatch compare, branch and the
/// unrolled body precede it); the prologue belongs to no section because the first slice is cut from the start but
/// holds no dequant or weight loads.
fn s1_sections(text: &str) -> Vec<String> {
    let (mut out, mut cur) = (Vec::new(), String::new());
    for l in text.lines() {
        let t = l.trim();
        if t.starts_with(".L") && t.contains("_skip") && t.ends_with(':') {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push_str(l);
            cur.push('\n');
        }
    }
    out
}

/// Every entry's FP mode (rounding, denormals, IEEE, DX10 clamp, FP16 overflow) equals the hipcc singleton's.
#[test]
fn fp_mode_matches_the_reference_descriptor() {
    let reference = match lift_fixture("gemv_mq6g256v2.elf") {
        Ok(p) => p,
        Err(e) => {
            println!("FPMODE reference not liftable ({e}); the interpreter's RNE/denormal-preserve gate still applies to every entry");
            return;
        }
    };
    let bits = |p: &Program, sym: &str| -> u32 {
        let k = p
            .kernels
            .iter()
            .find(|k| k.symbol.0 == sym)
            .unwrap_or_else(|| panic!("{sym}"));
        match &k.abi {
            Abi::Hsa { descriptor, .. } => descriptor.compute_pgm_rsrc1.0,
            Abi::Raw(_) => panic!("{sym}: raw abi"),
        }
    };
    // float_round_mode_32/64, float_denorm_mode_32/64 (12..19), DX10_CLAMP (21), IEEE_MODE (23), FP16_OVFL (26).
    let mask: u32 = (0xff << 12) | (1 << 21) | (1 << 23) | (1 << 26);
    let want = bits(&reference, SINGLETON) & mask;
    let rig = Rig::new();
    for symbol in SYMBOLS {
        assert_eq!(
            bits(&rig.program, symbol) & mask,
            want,
            "{symbol}: RSRC1 FP mode differs from the reference ({want:#x})"
        );
    }
}

// ------------------------------------------------------------------------------------------------ image

fn kernels_rs() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rdna-compute/src/kernels.rs"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn fresh_bundle() -> Vec<u8> {
    let mut emitted = module::emit_module().unwrap().0;
    emitted.extend(rowtile_emitted());
    let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(&emitted, MODULE).unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET)
}

/// Every instruction of every PM export has interpreter semantics (the same `semantic_class` preflight as
/// `Emulator::new`). Collects ALL missing mnemonics across all ten symbols at once (with the exports that use each),
/// instead of the serial first-failure the oracle tests would give.
#[test]
fn every_pm_instruction_has_emulator_semantics() {
    let rig = Rig::new();
    let mut missing: std::collections::BTreeMap<String, Vec<&str>> =
        std::collections::BTreeMap::new();
    for symbol in SYMBOLS {
        for name in unsupported_ops(&rig.program, symbol) {
            missing.entry(name).or_default().push(symbol);
        }
    }
    let list: Vec<String> = missing
        .iter()
        .map(|(n, s)| format!("{n} [{} export(s): {}]", s.len(), s.join(",")))
        .collect();
    println!(
        "PM-EMU-GAPS {} missing mnemonic(s): {}",
        missing.len(),
        missing.keys().cloned().collect::<Vec<_>>().join(", ")
    );
    assert!(
        missing.is_empty(),
        "interpreter lacks semantics for {} mnemonic(s) used by the PM exports:\n  {}",
        missing.len(),
        list.join("\n  ")
    );
}

#[test]
fn module_assembles_natively() {
    let (_, text, _) = module::emit_module().unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let bundle =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(bundle.len() > elf.len());
}

/// The runtime image contains the ten unchanged narrow exports followed by single and x4 rowtiles. Its bytes must
/// equal that exact composition, and its sha256 must be the runtime pin in `rdna-compute/src/kernels.rs`.
#[test]
fn committed_image_equals_fresh_emission_and_the_runtime_sha_pin() {
    let path = format!(
        "{}/../../kernels/{}",
        env!("CARGO_MANIFEST_DIR"),
        module::bundle_name()
    );
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel lane_rows_mq6_gfx1201 --arch gfx1201 --epi all ...)"));
    let fresh = fresh_bundle();
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
    let sha: String = Sha256::digest(&committed)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("IMAGE {} sha256={sha}", module::bundle_name());
    let src = kernels_rs();
    assert!(
        src.contains("pub const LANE_ROWS_MQ6_PM_GFX1201: &[u8]"),
        "R/kernels.rs lacks LANE_ROWS_MQ6_PM_GFX1201"
    );
    assert!(
        src.contains("kernels/lane_rows_mq6_pm_gfx1201.hxaco"),
        "R/kernels.rs does not include the image"
    );
    let at = src
        .find("pub const LANE_ROWS_MQ6_PM_GFX1201_SHA256")
        .expect("R/kernels.rs lacks LANE_ROWS_MQ6_PM_GFX1201_SHA256");
    // The runtime pin is `include_str!` of the committed `<image>.sha256` sidecar (bare hex,
    // no newline), so resolve the literal as that path and compare the sidecar's contents.
    let rest = &src[at..];
    let stmt = &rest[..rest.find(';').expect("pin statement end")];
    let sidecar = format!("kernels/{}.sha256", module::bundle_name());
    assert!(
        stmt.contains("include_str!") && stmt.contains(&sidecar),
        "R/kernels.rs sha pin must be include_str! of {sidecar}: {stmt}"
    );
    let pin = std::fs::read_to_string(format!("{}/../../{sidecar}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("{sidecar}: {e}"));
    assert_eq!(
        pin, sha,
        "sha256 sidecar (the R/kernels.rs runtime pin) != sha256 of the committed image"
    );
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check};

    /// Every export's waits replay independently, and M7 lifts the linked object byte-exactly with no obligations.
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = module::emit_module().unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-lane-rows-mq6-gfx1201-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 =
                pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------ fixed-eight-row tiles

const ROWTILE_NAMES: [&str; 2] = [
    "gemv_mq6g256v2_rows_t8_pm_gfx1201",
    "gemv_mq6g256v2_rows_x4_t8_pm_gfx1201",
];

fn rowtile_emitted() -> Vec<peacemaker_isa::plan::Emitted> {
    [false, true]
        .into_iter()
        .map(|x4| {
            module::emit_rowtiles(module::Spec {
                x4,
                cap: 8,
                wg8: true,
            })
            .unwrap()
        })
        .collect()
}

fn lift_pm_text(text: &str) -> Program {
    let elf = peacemaker_isa::native::assemble(text, ARCH).unwrap();
    lift_object(
        &elf,
        Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn rowtile_program() -> Program {
    let (text, _) =
        peacemaker_isa::kernels::iu4_gemm::module(&rowtile_emitted(), "mq6_rowtile_test").unwrap();
    lift_pm_text(&text)
}

fn rowtile_grid(ms: &[usize; 4], rows: usize, x4: bool) -> [u32; 3] {
    let gx = if x4 {
        ms.iter().map(|m| m.div_ceil(8)).sum()
    } else {
        ms[0].div_ceil(8)
    };
    [gx as u32, rows.div_ceil(8) as u32, 1]
}

fn rowtile_kernarg(p: &Problem, rows: i32, x4: bool) -> Vec<u8> {
    if x4 {
        kernarg_x4(p, Some(rows))
    } else {
        kernarg_single(p, Some(rows))
    }
}

/// No optional reference mode: fixture/lift/instruction/launch failures are fatal to this gate.
fn strict_singleton_program(file: &str, symbol: &str) -> Program {
    let path = format!("{FIXTURES}/{file}");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("required actual gfx1201 singleton {path}: {e}"));
    let (reference, elf_sha) = match symbol {
        SINGLETON => (
            module::SINGLE_REFERENCE,
            "d141a32fa1f2d7ad1aaeec63d63afa48ad6d53c3f512e9237f9e307f85094023",
        ),
        X4_SINGLETON => (
            module::X4_REFERENCE,
            "9f5ec178ea5618b3f6bd0552a9d8d3836519ec4dee9bc55a25f58257aa842959",
        ),
        _ => panic!("unbound singleton fixture {symbol}"),
    };
    assert_eq!(file, format!("{}.elf", reference.fixture));
    let sha = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(sha, elf_sha, "{symbol}: actual singleton ELF changed");
    let bundle_path = format!("{FIXTURES}/{}.hsaco", reference.fixture);
    let bundle = std::fs::read(&bundle_path).unwrap_or_else(|e| panic!("{bundle_path}: {e}"));
    assert_eq!(
        format!("{:x}", Sha256::digest(&bundle)),
        reference.sha256,
        "{symbol}: production bundle changed"
    );
    let provenance = std::fs::read_to_string(format!("{FIXTURES}/PROVENANCE.md")).unwrap();
    assert!(provenance.contains(elf_sha) && provenance.contains(reference.sha256));
    assert!(
        provenance.contains("HIP version: 7.16.26385")
            && provenance.contains("d6f6cb691863e9258034ab48c1dfce21689d5606"),
        "{symbol}: missing pinned producer toolchain"
    );
    println!(
        "ROWTILE-REFERENCE {symbol} file={path} sha256={sha} bundle_sha256={}",
        reference.sha256
    );
    let program = lift_object(
        &bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .unwrap_or_else(|e| panic!("required actual gfx1201 singleton {path}: {e}"))
    .program;
    Emulator::new(&program, symbol)
        .unwrap_or_else(|e| panic!("required actual gfx1201 singleton {symbol}: {e}"));
    program
}

fn verify_readonly(mem: &Memory, before: &Memory, tag: &str) {
    for r in before.regions.iter().filter(|r| !r.name.starts_with('y')) {
        let after = region(mem, &r.name);
        assert_eq!(after.bytes, r.bytes, "{tag}: {} bytes changed", r.name);
        assert_eq!(after.written.count(), 0, "{tag}: {} was written", r.name);
    }
}

/// Guarded actual-object execution, including the object's own exact output extent.
fn strict_object_outputs(emu: &Emulator, p: &Problem, x: &[u32], x4: bool) -> [Vec<u32>; 4] {
    let ka = if x4 {
        kernarg_x4(p, None)
    } else {
        kernarg_single(p, None)
    };
    let mut mem = build_mem(p, x, 1, ka);
    let before = mem.clone();
    let gx = if x4 { p.ms.iter().sum() } else { p.ms[0] };
    launch_grid(
        emu,
        &mut mem,
        "required actual singleton",
        [gx as u32, 1, 1],
        32,
    );
    verify(&mem, p, x, 1, 1, 1, "required actual singleton");
    verify_readonly(&mem, &before, "required actual singleton");
    std::array::from_fn(|i| {
        (0..p.ms[i])
            .map(|n| y_word(&mem, i, 0, n, p.ms[i]))
            .collect()
    })
}

/// Independent absolute rows, including cancellation and signed-zero data, with a nonperiodic row fingerprint.
/// In particular, gy=1 cannot accidentally pass by reusing gy=0's eight activations.
fn strict_rowtile_case(
    seed: u64,
    k: usize,
    ms: [usize; 4],
    singleton: &Emulator,
    x4_singleton: Option<&Emulator>,
    c1: &[Emulator],
) -> Case {
    let mut rng = Rng(seed | 1);
    let mats = std::array::from_fn(|i| mq6_matrix(&mut rng, ms[i], k));
    let pool: Vec<Vec<u32>> = (0..64)
        .map(|r| {
            let mut x = x_row(&mut rng, POOL_SLOTS[r % 8], k);
            x[0] = ((r as f32 + 1.125) * if r % 2 == 0 { 1.0 } else { -1.0 }).to_bits();
            x
        })
        .collect();
    let p = Problem { k, ms, mats };
    let mut refy: Vec<[Vec<u32>; 4]> = (0..64)
        .map(|_| std::array::from_fn(|_| Vec::new()))
        .collect();
    for i in 0..4 {
        if ms[i] == 0 {
            continue;
        }
        let single = Problem {
            k,
            ms: [ms[i], 0, 0, 0],
            mats: std::array::from_fn(|j| {
                if j == 0 {
                    p.mats[i].clone()
                } else {
                    Vec::new()
                }
            }),
        };
        for (r, x) in pool.iter().enumerate() {
            let [words, _, _, _] = strict_object_outputs(singleton, &single, x, false);
            refy[r][i] = words;
        }
    }
    let dag = pool
        .iter()
        .map(|x| {
            std::array::from_fn(|i| (0..ms[i]).map(|n| ref_row(&p.mats[i], n, k, x)).collect())
        })
        .collect();
    let mut case = Case {
        p,
        pool,
        c1: Vec::new(),
        refy,
        dag,
        x4ref: Vec::new(),
    };
    if let Some(emu) = x4_singleton {
        case.x4ref = case
            .pool
            .iter()
            .map(|x| strict_object_outputs(emu, &case.p, x, true))
            .collect();
        for r in 0..64 {
            assert_eq!(
                case.x4ref[r], case.refy[r],
                "actual x4 != actual single K={k} M={ms:?} row={r}"
            );
        }
    }
    attach_c1(c1, &mut case, x4_singleton.is_some());
    case
}

/// Poison is separately mapped after the exact live allocations. The tight variant leaves those bytes unmapped;
/// the poison variant also allocates a full extra output tile, whose WriteSet must remain empty.
fn rowtile_mem(p: &Problem, x: &[u32], rows: usize, x4: bool, poison: bool) -> (Memory, usize) {
    let allocated = if poison {
        rows.div_ceil(8) * 8 + 8
    } else {
        rows
    };
    let mut mem = build_mem(p, x, allocated, rowtile_kernarg(p, rows as i32, x4));
    if poison {
        mem.map(
            "x_tail".into(),
            X_BASE + (x.len() * 4) as u64,
            words_to_bytes(&vec![0x7fc1_dead; (allocated - rows) * p.k]),
        )
        .unwrap();
        for i in 0..4 {
            mem.map(
                format!("a_tail{i}"),
                a_ptr(i) + p.mats[i].len() as u64,
                vec![0xff; 8 * p.k / 256 * 200],
            )
            .unwrap();
        }
    }
    (mem, allocated)
}

fn run_rowtile_width(emu: &Emulator, case: &Case, rows: usize, x4: bool, poison: bool) {
    let order: Vec<usize> = (0..rows).map(|r| (17 * r + 7 * rows) % 64).collect();
    let x: Vec<u32> = order
        .iter()
        .flat_map(|&r| case.pool[r].iter().copied())
        .collect();
    let (mut mem, allocated) = rowtile_mem(&case.p, &x, rows, x4, poison);
    let before = mem.clone();
    let tag = format!(
        "{} R={rows} K={} M={:?} poison={poison}",
        ROWTILE_NAMES[usize::from(x4)],
        case.p.k,
        case.p.ms
    );
    let grid = rowtile_grid(&case.p.ms, rows, x4);
    // Reverse both axes: independent tiles cannot rely on prior workgroups having filled scratch or output.
    for gy in (0..grid[1]).rev() {
        for gx in (0..grid[0]).rev() {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [gx, gy, 0],
                    block: [256, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{tag} wg({gx},{gy}): {e}"));
        }
    }
    verify(&mem, &case.p, &x, 64, allocated, rows as i32, &tag);
    verify_readonly(&mem, &before, &tag);
    compare(&mem, case, &order, &tag);
}

#[test]
fn rowtiles_single_every_width_1_to_64_equals_actual_object_and_pm_c1() {
    let reference = strict_singleton_program("gemv_mq6g256v2.elf", SINGLETON);
    let singleton = Emulator::new(&reference, SINGLETON).unwrap();
    let program = rowtile_program();
    let emu = Emulator::new(&program, ROWTILE_NAMES[0]).unwrap();
    let narrow = Rig::new();
    let c1 = narrow.emus();
    // Tail-only 1/2/3 groups; a quad with each 0/1/2/3 remainder; repeated quads and model-sized K.
    // M covers full/partial output tiles, including a second ragged tile and seven inactive output waves.
    for (k, m) in [
        (256, 1),
        (512, 2),
        (768, 3),
        (1024, 7),
        (1280, 8),
        (1536, 9),
        (1792, 17),
        (2048, 2),
        (2560, 9),
        (6144, 3),
    ] {
        let case = strict_rowtile_case(0x7438 + k as u64, k, [m, 0, 0, 0], &singleton, None, &c1);
        // Widths are independent: run them concurrently; any panic fails the scope.
        std::thread::scope(|s| {
            for rows in 1..=64 {
                let (emu, case) = (&emu, &case);
                s.spawn(move || {
                    for poison in [false, true] {
                        run_rowtile_width(emu, case, rows, false, poison);
                    }
                });
            }
        });
    }
}

#[test]
fn rowtiles_x4_every_width_and_all_empty_live_masks_equal_actual_objects_and_pm_c1() {
    let reference = strict_singleton_program("gemv_mq6g256v2.elf", SINGLETON);
    let x4_reference = strict_singleton_program("qwen4_gemv_mq6g256v2.elf", X4_SINGLETON);
    let singleton = Emulator::new(&reference, SINGLETON).unwrap();
    let x4_singleton = Emulator::new(&x4_reference, X4_SINGLETON).unwrap();
    let program = rowtile_program();
    let emu = Emulator::new(&program, ROWTILE_NAMES[1]).unwrap();
    let narrow = Rig::new();
    let c1 = narrow.emus();
    for mask in 0..16usize {
        let heights = if mask % 2 == 0 {
            [1, 3, 8, 9]
        } else {
            [2, 7, 9, 17]
        };
        let ms = std::array::from_fn(|i| if mask & (1 << i) != 0 { heights[i] } else { 0 });
        let k = (mask % 8 + 1) * 256;
        let case = strict_rowtile_case(
            0x74_4001 + mask as u64,
            k,
            ms,
            &singleton,
            Some(&x4_singleton),
            &c1,
        );
        // Widths are independent: run them concurrently; any panic fails the scope.
        std::thread::scope(|s| {
            for rows in 1..=64 {
                let (emu, case) = (&emu, &case);
                s.spawn(move || {
                    for poison in [false, true] {
                        run_rowtile_width(emu, case, rows, true, poison);
                    }
                });
            }
        });
    }
}

/// Launch each owner in fresh memory: a byte cannot be hidden by an earlier/later workgroup overwriting it.
#[test]
fn rowtiles_each_workgroup_writes_only_its_absolute_row_and_output_tile_once() {
    let reference = strict_singleton_program("gemv_mq6g256v2.elf", SINGLETON);
    let x4_reference = strict_singleton_program("qwen4_gemv_mq6g256v2.elf", X4_SINGLETON);
    let singleton = Emulator::new(&reference, SINGLETON).unwrap();
    let x4_singleton = Emulator::new(&x4_reference, X4_SINGLETON).unwrap();
    let program = rowtile_program();
    let narrow = Rig::new();
    let c1 = narrow.emus();
    for x4 in [false, true] {
        let ms = if x4 { [1, 9, 0, 17] } else { [17, 0, 0, 0] };
        let case = strict_rowtile_case(
            0x0a11_0c8,
            768,
            ms,
            &singleton,
            if x4 { Some(&x4_singleton) } else { None },
            &c1,
        );
        let emu = Emulator::new(&program, ROWTILE_NAMES[usize::from(x4)]).unwrap();
        for rows in [1usize, 7, 8, 9, 15, 17, 33, 63, 64] {
            let x: Vec<u32> = case.pool[..rows].iter().flatten().copied().collect();
            let grid = rowtile_grid(&ms, rows, x4);
            let mut seen: [BTreeSet<usize>; 4] = std::array::from_fn(|_| BTreeSet::new());
            for gy in 0..grid[1] {
                for gx in 0..grid[0] {
                    let (mut mem, _) = rowtile_mem(&case.p, &x, rows, x4, true);
                    let before = mem.clone();
                    emu.run_workgroup(
                        &Launch {
                            kernarg_ptr: KERNARG,
                            workgroup: [gx, gy, 0],
                            block: [256, 1, 1],
                            dynamic_lds: 0,
                        },
                        &mut mem,
                    )
                    .unwrap();
                    verify_readonly(&mem, &before, "isolated rowtile workgroup");
                    let mut local_gx = gx as usize;
                    let matrix = (0..4)
                        .find(|&i| {
                            let blocks = ms[i].div_ceil(8);
                            if local_gx < blocks {
                                true
                            } else {
                                local_gx -= blocks;
                                false
                            }
                        })
                        .unwrap();
                    let first_col = local_gx * 8;
                    let first_row = gy as usize * 8;
                    for i in 0..4 {
                        let y = region(&mem, &format!("y{i}"));
                        let expected: BTreeSet<usize> = if i == matrix {
                            (first_row..(first_row + 8).min(rows))
                                .flat_map(|r| {
                                    (first_col..(first_col + 8).min(ms[i])).flat_map(move |n| {
                                        (GUARD + r * ms[i] + n) * 4..(GUARD + r * ms[i] + n + 1) * 4
                                    })
                                })
                                .collect()
                        } else {
                            BTreeSet::new()
                        };
                        assert_eq!(y.written.count(), expected.len());
                        for b in 0..y.bytes.len() {
                            assert_eq!(
                                y.written.contains(b),
                                expected.contains(&b),
                                "x4={x4} R={rows} wg({gx},{gy}) y{i} byte={b}"
                            );
                            if expected.contains(&b) {
                                assert!(seen[i].insert(b), "overlapping workgroup ownership");
                            } else {
                                assert_eq!(
                                    y.bytes[b],
                                    before
                                        .regions
                                        .iter()
                                        .find(|r| r.name == y.name)
                                        .unwrap()
                                        .bytes[b],
                                    "unowned output byte changed"
                                );
                            }
                        }
                    }
                    for r in first_row..(first_row + 8).min(rows) {
                        for n in first_col..(first_col + 8).min(ms[matrix]) {
                            assert_eq!(
                                y_word(&mem, matrix, r, n, ms[matrix]),
                                case.refy[r][matrix][n]
                            );
                            assert_eq!(
                                y_word(&mem, matrix, r, n, ms[matrix]),
                                case.c1[r][matrix][n]
                            );
                        }
                    }
                }
            }
            for i in 0..4 {
                assert_eq!(
                    seen[i].len(),
                    rows * ms[i] * 4,
                    "absolute row coverage y{i} R={rows}"
                );
            }
        }
    }
}

#[test]
fn rowtiles_invalid_total_r_and_dead_grid_workgroups_touch_no_tensor() {
    let program = rowtile_program();
    for x4 in [false, true] {
        let emu = Emulator::new(&program, ROWTILE_NAMES[usize::from(x4)]).unwrap();
        for ms in [if x4 { [1, 9, 0, 2] } else { [9, 0, 0, 0] }, [0; 4]] {
            let p = Problem {
                k: 256,
                ms,
                mats: std::array::from_fn(|_| Vec::new()),
            };
            let mut launches = Vec::new();
            for invalid in [i32::MIN, -1, 0, 65, 128, i32::MAX] {
                launches.push((invalid, [0, 0, 0]));
                launches.push((invalid, [0, 7, 0]));
            }
            for rows in 1..=64usize {
                let grid = rowtile_grid(&ms, rows, x4);
                launches.push((rows as i32, [0, grid[1], 0]));
                launches.push((rows as i32, [grid[0], 0, 0]));
                launches.push((rows as i32, [grid[0] + 1, grid[1] + 1, 0]));
            }
            for (rows, wg) in launches {
                // Only kernargs and guarded outputs are mapped; ANY tensor read is a fatal unmapped access.
                let mut mem = build_mem(&p, &[], 64, rowtile_kernarg(&p, rows, x4));
                mem.regions
                    .retain(|r| r.name == "kernarg" || r.name.starts_with('y'));
                let before = mem.clone();
                emu.run_workgroup(
                    &Launch {
                        kernarg_ptr: KERNARG,
                        workgroup: wg,
                        block: [256, 1, 1],
                        dynamic_lds: 0,
                    },
                    &mut mem,
                )
                .unwrap_or_else(|e| panic!("x4={x4} R={rows} M={ms:?} wg={wg:?}: {e}"));
                for r in &mem.regions {
                    assert_eq!(r.bytes, region(&before, &r.name).bytes);
                    assert_eq!(
                        r.written.count(),
                        0,
                        "x4={x4} R={rows} wg={wg:?} wrote {}",
                        r.name
                    );
                }
            }
        }
    }
}

#[test]
fn rowtiles_reject_every_spec_except_cap8_wg8() {
    for x4 in [false, true] {
        for cap in 0..=u8::MAX {
            for wg8 in [false, true] {
                let result = module::emit_rowtiles(module::Spec { x4, cap, wg8 });
                assert_eq!(
                    result.is_ok(),
                    cap == 8 && wg8,
                    "x4={x4} cap={cap} wg8={wg8}"
                );
            }
        }
    }
}

#[test]
fn rowtiles_resources_abi_orientation_and_fp_mode_match_the_frozen_contract() {
    let emitted = rowtile_emitted();
    let program = rowtile_program();
    let reference = strict_singleton_program("gemv_mq6g256v2.elf", SINGLETON);
    let x4_reference = strict_singleton_program("qwen4_gemv_mq6g256v2.elf", X4_SINGLETON);
    let fp_mode = |p: &Program, symbol: &str| {
        let k = p.kernels.iter().find(|k| k.symbol.0 == symbol).unwrap();
        match &k.abi {
            Abi::Hsa { descriptor, .. } => {
                descriptor.compute_pgm_rsrc1.0 & ((0xff << 12) | (1 << 21) | (1 << 23) | (1 << 26))
            }
            Abi::Raw(_) => panic!("{symbol}: expected HSA descriptor"),
        }
    };
    for (i, e) in emitted.iter().enumerate() {
        let (name, t) = (ROWTILE_NAMES[i], &e.s_text);
        let args = if i == 0 { SINGLE_ARGS } else { X4_ARGS };
        assert!(t.contains(&format!(".amdhsa_kernel {name}\n")));
        assert!(t.contains(&format!(".name: {name}\n")));
        assert_eq!(e.proof.arch, ARCH);
        assert_eq!((e.proof.next_free_vgpr, e.proof.next_free_sgpr), (72, 58));
        assert_eq!(field(t, ".vgpr_count:"), [72]);
        // Metadata counts the reserved VCC pair in addition to the 58 allocated ordinary SGPRs.
        assert_eq!(field(t, ".sgpr_count:"), [60]);
        assert_eq!(
            field(t, ".offset:"),
            args.iter().map(|a| a.0).collect::<Vec<_>>()
        );
        assert_eq!(
            field(t, ".size:"),
            args.iter().map(|a| a.1).collect::<Vec<_>>()
        );
        assert_eq!(
            field(t, ".kernarg_segment_size:"),
            [if i == 0 { 36 } else { 96 }]
        );
        assert_eq!(field(t, ".max_flat_workgroup_size:"), [256]);
        assert_eq!(field(t, ".group_segment_fixed_size:"), [8192]);
        assert_eq!(field(t, ".private_segment_fixed_size:"), [0]);
        assert_eq!(field(t, ".sgpr_spill_count:"), [0]);
        assert_eq!(field(t, ".vgpr_spill_count:"), [0]);
        assert!(t.contains(".amdhsa_wavefront_size32 1"));
        assert!(t.contains(".amdhsa_system_sgpr_workgroup_id_y 1"));
        assert!(!t.lines().any(|l| {
            let l = l.trim();
            l.starts_with("scratch_") || l.starts_with("flat_") || l.contains("atomic")
        }));
        assert!(check_barriers(name, t) >= 2);
        assert!(
            unsupported_ops(&program, name).is_empty(),
            "{name}: unsupported interpreter instructions"
        );
        let (oracle, oracle_symbol) = if i == 0 {
            (&reference, SINGLETON)
        } else {
            (&x4_reference, X4_SINGLETON)
        };
        assert_eq!(fp_mode(&program, name), fp_mode(oracle, oracle_symbol));
        // S2's four group bodies each have eight row dots. Freeze the object's activation-src0 / weight-src1
        // orientation, x1-first initialization and x0,x2..x7 FMA order, not merely a commutative finite dot.
        let dots: Vec<&str> = t
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("v_mul_f32_e32 v9,") || l.starts_with("v_fmac_f32_e32 v9,"))
            .collect();
        let expected = [
            "v_mul_f32_e32 v9, v25, v17",
            "v_fmac_f32_e32 v9, v24, v16",
            "v_fmac_f32_e32 v9, v26, v18",
            "v_fmac_f32_e32 v9, v27, v19",
            "v_fmac_f32_e32 v9, v28, v20",
            "v_fmac_f32_e32 v9, v29, v21",
            "v_fmac_f32_e32 v9, v30, v22",
            "v_fmac_f32_e32 v9, v31, v23",
        ];
        assert_eq!(
            dots.len(),
            4 * 8 * expected.len(),
            "{name}: fixed physical tile"
        );
        for dot in dots.chunks_exact(expected.len()) {
            assert_eq!(
                dot, expected,
                "{name}: singleton instruction DAG/orientation"
            );
        }
    }
}

#[test]
fn rowtile_image_composition_preserves_every_narrow_instruction_byte() {
    let (mut emitted, narrow_text, _) = module::emit_module().unwrap();
    let narrow = lift_pm_text(&narrow_text);
    emitted.extend(rowtile_emitted());
    let (full_text, _) = peacemaker_isa::kernels::iu4_gemm::module(&emitted, MODULE).unwrap();
    let full = lift_pm_text(&full_text);
    assert_eq!(full.kernels.len(), SYMBOLS.len() + 2);
    for symbol in SYMBOLS {
        let instructions = |p: &Program| {
            let k = p.kernels.iter().find(|k| k.symbol.0 == symbol).unwrap();
            k.body
                .layout
                .iter()
                .map(|id| {
                    k.body
                        .insts
                        .get(*id)
                        .unwrap()
                        .prov
                        .bytes
                        .unwrap_or_else(|| panic!("{symbol}: lifted instruction lacks raw bytes"))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            instructions(&full),
            instructions(&narrow),
            "{symbol}: narrow instruction bytes changed"
        );
    }
}

#[cfg(feature = "toolchain")]
#[test]
fn rowtiles_pass_wait_replay_and_m7() {
    use peacemaker_isa::{ledger_replay, pm_check};
    let emitted = rowtile_emitted();
    for e in &emitted {
        ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
    }
    let (text, _) =
        peacemaker_isa::kernels::iu4_gemm::module(&emitted, "mq6_rowtile_test").unwrap();
    let dir = std::env::temp_dir().join(format!(
        "peacemaker-isa-mq6-rowtiles-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let co = dir.join("rowtiles.co");
    std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
    for symbol in ROWTILE_NAMES {
        let m7 = pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{symbol}"
        );
    }
}
