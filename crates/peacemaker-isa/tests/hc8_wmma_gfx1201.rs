// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU byte oracle of the gfx1201 compressed-weight BF16 WMMA PM GEMM, module `hc8_wmma_pm_gfx1201`
//! (`peacemaker_isa::kernels::hc8_wmma_gfx1201`): 8-bit weights (OCP E4M3 / int8), f32 BF16-exact activations,
//! per-output-row f32 scale, `set` and `act` (hc_act) epilogues, rows 1..=64, `Geo{fmt,mw,kw,nt,act}` entries.
//!
//! Every launch runs on the peacemaker CPU interpreter exactly like `lane_rows_bf16_gfx1201.rs`: PM `emit_module` ->
//! `native::assemble` -> `lift_object(Builder)` -> `Emulator::run_workgroup` per workgroup (real LDS and barriers for
//! the KW fold). All tests execute the emitted PM source; nothing is read from a committed artifact.
//!
//! References (none shares an instruction, tile, LDS fold or decode with the compressed generator):
//!
//! 1. **Independent WMMA reference** (`tests/support/hc8_reference.rs`, also used by the hardware byte oracle): a
//!    separately emitted Builder kernel that takes HOST-decoded BF16 weights in plain row-major order, builds its own
//!    A/B fragments from `f16_loc`, runs `v_wmma_f32_16x16x16_bf16` with the SAME frozen K split and fold
//!    (per slice an ascending K16 chain from +0, partials folded `f = p0; f += p1; ...`, then `f * s[m]`, and for
//!    `act` the pinned hipcc `hc_act` VALU transcription). Every output word of every kernel launch is compared with
//!    it bit for bit, including NaN payloads and signed zeros.
//! 2. **Singleton row invariance**: output row `r` of a batched launch equals, bit for bit, the `rows = 1` launch of
//!    the same input row (batched with distinct rows, one row replicated at every position, rotated slot order).
//! 3. **Host model** (`host_model_set`): pure Rust, the emulator's hardware-qualified `mma::wmma_f32_bf16` dot per
//!    K16 step in logical K order with the same chain/fold; compared on finite words (NaN by NaN-ness, subnormal
//!    words skipped because host f32 add/mul are not the interpreter's FTZ-aware ones).
//! 4. **f64 reference** (decoded weights x BF16 x in f64, exact products): rel 1e-5 when the sum does not cancel,
//!    otherwise an absolute bound `(K/16 + KW + 2) * 2^-22 * sum|w x| * |s|` (cancellation aware; the f32 WMMA chain
//!    cannot do better), for both formats and every geometry class. `act` is checked loosely (two BF16 ulps) against
//!    an f64 silu chain; its exact bytes are pinned by (1).
//! 5. **Layout**: the swizzle's lane/byte map is checked against an expected map derived independently from the
//!    gfx1201 `f16_loc` fragment layout (not a round trip only), plus round trips both ways.
//! 6. **Decode sweep**: every finite E4M3 code and every int8 value, observed through one-hot activations with unit
//!    scale (the product is the decoded weight itself), against the host decode.
//!
//! Also checked on every launch: weights, scale and activations are byte-identical and never written; the written
//! byte set of `y` is exactly `rows * M * 4` (dead columns and capacity padding never stored, sentinel words intact,
//! guard words intact); `rows == 0` and `rows > 16*NT` write nothing and terminate. Structural test: 80 symbols,
//! ABI 48 B, offsets, no scratch/spills/atomics/flat, deterministic emission, native assemble + lift + `Emulator::new`
//! of every entry.
//!
//! SEMANTICS PROVENANCE. Independently hardware-qualified (float_edges.md, checked by
//! `qualification_notes_are_recorded`): `v_wmma_f32_16x16x16_bf16` (0 mismatches over 7,437,056 gfx1201 words),
//! the f32 `v_add`/`v_mul`/bit ops of the fold, FP8 WMMA/convert tables that the PM does not use. EMULATOR-MODEL ONLY
//! (the hardware byte oracle `hc8_byte_oracle` is their receipt, this file is not): `v_cvt_pk_f32_fp8` decode incl.
//! NaN sign (`sign | 0x7fc00000`) and the exceptional float domains of `act` (the `v_div_scale_f32` domains the
//! interpreter refuses are counted and reported, never silently passed; `HC8_ORACLE_REQUIRE_NO_REFUSAL=1` fails on
//! any). A pass of this file is NOT a hardware receipt.
//!
//! Runtime: [INFERENCE, never run: no builds allowed here] about 1-2 minutes of CPU, spread over 18 tests that cargo runs in parallel (about 25-40 s wall on 8+
//! cores). `HC8_ORACLE_LOG=<file>` appends every log line to a file.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_ir::Program;
use peacemaker_isa::kernels::hc8_wmma_gfx1201::{
    emit_module, swizzle_hc8, symbols, unswizzle_hc8, Fmt, Geo, MODULE,
};
use peacemaker_isa::Arch;
use std::collections::{BTreeMap, BTreeSet};

#[path = "support/hc8_reference.rs"]
mod hc8_reference;
use hc8_reference::{
    decode_weights, host_model_set, reference_grid, reference_module, reference_symbol,
    REFERENCE_BLOCK,
};

const ARCH: Arch = Arch::Gfx1201;

// ------------------------------------------------------------------------------------------------ logging

fn log(line: impl AsRef<str>) {
    use std::io::Write;
    println!("{}", line.as_ref());
    if let Ok(path) = std::env::var("HC8_ORACLE_LOG") {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", line.as_ref());
        }
    }
}

fn require_no_refusal() -> bool {
    std::env::var("HC8_ORACLE_REQUIRE_NO_REFUSAL").is_ok_and(|v| v == "1")
}

// ------------------------------------------------------------------------------------------------ geometry

const CLASSES: [(u32, u32); 5] = [(1, 8), (1, 16), (4, 1), (8, 1), (2, 4)];
const FMTS: [Fmt; 2] = [Fmt::E4M3, Fmt::I8];

fn fmt_name(f: Fmt) -> &'static str {
    match f {
        Fmt::E4M3 => "e4m3",
        Fmt::I8 => "i8",
    }
}

fn geo(fmt: Fmt, mw: u32, kw: u32, nt: u32, act: bool) -> Geo {
    Geo {
        fmt,
        mw,
        kw,
        nt,
        act,
    }
}

/// Entry name rule of the spec, spelled independently of `Geo::symbol`.
fn expected_symbol(g: Geo) -> String {
    format!(
        "hc8_gemm_{}_mw{}_kw{}_n{}_{}",
        fmt_name(g.fmt),
        g.mw,
        g.kw,
        g.nt,
        if g.act { "act" } else { "set" }
    )
}

/// Shapes `(M, K)` of the main matrix: several workgroups where cheap, `KT >= 2` K tiles per wave where possible.
fn main_shape(mw: u32, kw: u32) -> (usize, usize) {
    match (mw, kw) {
        (1, 8) => (32, 512),
        (1, 16) => (32, 1024),
        (4, 1) => (64, 320),
        (8, 1) => (128, 320),
        (2, 4) => (64, 512),
        _ => panic!("unknown class"),
    }
}

/// Smallest legal shape (one workgroup, `K = 32 * KW` or two K tiles), used by the every-rows sweep.
fn small_shape(mw: u32, kw: u32) -> (usize, usize) {
    match (mw, kw) {
        (1, 8) => (16, 256),
        (1, 16) => (16, 512),
        (4, 1) => (64, 64),
        (8, 1) => (128, 32),
        (2, 4) => (32, 128),
        _ => panic!("unknown class"),
    }
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum QKind {
    /// Every finite code.
    Full,
    /// Small magnitudes (E4M3 exponent field 4..=9, int8 in -7..=7).
    Small,
    /// Mostly zero.
    Sparse,
}

/// Raw 8-bit weights, never an E4M3 NaN code.
fn gen_q(rng: &mut Rng, n: usize, fmt: Fmt, kind: QKind) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let sign = (rng.next() & 1) as u8;
            match (fmt, kind) {
                (_, QKind::Sparse) if rng.below(8) != 0 => 0,
                (Fmt::E4M3, QKind::Small) => {
                    (sign << 7) | ((4 + rng.below(6) as u8) << 3) | rng.below(8) as u8
                }
                (Fmt::E4M3, _) => {
                    let b = rng.next() as u8;
                    if b & 0x7f == 0x7f {
                        b ^ 1
                    } else {
                        b
                    }
                }
                (Fmt::I8, QKind::Small) => (rng.below(15) as i8 - 7) as u8,
                (Fmt::I8, _) => match rng.below(16) {
                    0 => 0x80,
                    1 => 0x7f,
                    _ => rng.next() as u8,
                },
            }
        })
        .collect()
}

fn bf(sign: u32, e: u32, m: u32) -> f32 {
    f32::from_bits((sign << 31) | (e << 23) | (m << 16))
}

const EDGE_X: [u16; 16] = [
    0x0000, 0x8000, 0x3f80, 0xbf80, 0x0080, 0x8080, 0x0001, 0x807f, 0x7f7f, 0xff7f, 0x4000, 0x3f00,
    0x4348, 0x2300, 0x0000, 0x3f81,
];

/// A BF16-exact f32 activation row. Kinds: 0 moderate, 1 cancelling pairs, 2 wide exponent, 3 finite edges
/// (signed zeros, min normal, subnormals, max finite), 4 sparse, 5 tiny / subnormal heavy.
fn gen_x_row(rng: &mut Rng, k: usize, kind: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    let mut i = 0;
    while i < k {
        let sign = (rng.next() & 1) as u32;
        let v = match kind {
            0 => {
                if rng.below(16) == 0 {
                    bf(sign, 0, 0)
                } else {
                    bf(sign, 118 + rng.below(17) as u32, rng.below(128) as u32)
                }
            }
            1 => {
                let mag = bf(0, 119 + rng.below(8) as u32, rng.below(128) as u32);
                out.push(mag);
                i += 1;
                if i < k {
                    // opposite sign, a one-BF16-ulp perturbation about a third of the time
                    let bump = if rng.below(3) == 0 { 1 } else { 0 };
                    out.push(-f32::from_bits(mag.to_bits() + (bump << 16)));
                    i += 1;
                }
                continue;
            }
            2 => bf(sign, 100 + rng.below(51) as u32, rng.below(128) as u32),
            3 => f32::from_bits(u32::from(EDGE_X[rng.below(EDGE_X.len())]) << 16),
            4 => {
                if rng.below(8) == 0 {
                    bf(sign, 120 + rng.below(10) as u32, rng.below(128) as u32)
                } else {
                    bf(0, 0, 0)
                }
            }
            _ => bf(sign, rng.below(24) as u32, rng.below(128) as u32),
        };
        out.push(v);
        i += 1;
    }
    out
}

fn gen_pool(rng: &mut Rng, k: usize, n: usize, kinds: &[usize]) -> Vec<Vec<f32>> {
    (0..n)
        .map(|r| {
            let kind = kinds[(r + rng.below(2)) % kinds.len()];
            gen_x_row(rng, k, kind)
        })
        .collect()
}

const ALL_KINDS: [usize; 6] = [0, 1, 2, 3, 4, 5];
const MODERATE_KINDS: [usize; 3] = [0, 1, 4];

fn gen_scale(rng: &mut Rng, m: usize, act: bool) -> Vec<f32> {
    (0..m)
        .map(|_| {
            let mant = (rng.next() as u32) & 0x7f_ffff;
            let e = if act {
                118 - rng.below(5) as u32
            } else {
                124 + rng.below(4) as u32
            };
            let v = f32::from_bits((e << 23) | mant);
            match rng.below(16) {
                0 => 1.0,
                1 | 2 => -v,
                _ => v,
            }
        })
        .collect()
}

/// A weight matrix with its scale: raw bytes, swizzled bytes, host-decoded BF16.
struct Case {
    fmt: Fmt,
    m: usize,
    k: usize,
    raw: Vec<u8>,
    swz: Vec<u8>,
    wbf: Vec<u16>,
    sc: Vec<f32>,
    act_scale: f32,
}

fn make_case(fmt: Fmt, m: usize, k: usize, raw: Vec<u8>, sc: Vec<f32>, act_scale: f32) -> Case {
    assert_eq!(raw.len(), m * k);
    let swz = swizzle_hc8(&raw, m, k);
    let wbf = decode_weights(fmt, &raw);
    Case {
        fmt,
        m,
        k,
        raw,
        swz,
        wbf,
        sc,
        act_scale,
    }
}

fn rand_case(seed: u64, fmt: Fmt, m: usize, k: usize, act: bool, kind: QKind) -> Case {
    let mut rng = Rng::new(seed);
    let raw = gen_q(&mut rng, m * k, fmt, kind);
    let sc = gen_scale(&mut rng, m, act);
    let act_scale = [0.5f32, 1.0, 1.25, 2.0][rng.below(4)];
    make_case(fmt, m, k, raw, sc, act_scale)
}

// ------------------------------------------------------------------------------------------------ memory model

const GUARD: usize = 128;
const KERNARG: u64 = 0x1_0000;
const XBASE: u64 = 0x1_0000_0000;
const QBASE: u64 = 0x2_0000_0000;
const SBASE: u64 = 0x3_0000_0000;
const YBASE: u64 = 0x4_0000_0000;
const SENT: u32 = 0x7fc0_a5a5;
const XGUARD: u32 = 0x7fc0_dead;
const WGUARD: u32 = 0x7fc1_7fc1;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

enum Bad {
    /// A documented interpreter refusal of an unmeasured float domain (never a pass: counted and reported).
    Refused(String),
    /// Everything else: a defect.
    Defect(String),
}

/// The CLOSED list of float domains the interpreter refuses that these tests can reach (`div_scale_chain` of
/// `peacemaker-ir/src/emu/alu.rs`), exact messages.
const REFUSALS: [&str; 4] = [
    "v_div_scale_f32: exponent difference -96 is measured only for denominator exponent 254",
    "v_div_scale_f32: S0 * 2^-64 with exponent field 64..=90 is not measured",
    "v_div_scale_f32: denominator exponent 228..=253 is not measured",
    "v_div_scale_f32: numerator exponent 24..=26 is not measured",
];

fn classify(err: String) -> Bad {
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
    let refusal = REFUSALS.iter().any(|m| err.contains(m))
        || (!HARD.iter().any(|h| err.contains(h)) && GAP.iter().any(|g| err.contains(g)));
    if refusal {
        Bad::Refused(err)
    } else {
        Bad::Defect(err)
    }
}

fn launch_grid(
    emu: &Emulator,
    mem: &mut Memory,
    gx: u32,
    gy: u32,
    block: u32,
) -> Result<(), String> {
    for y in 0..gy {
        for x in 0..gx {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [x, y, 0],
                    block: [block, 1, 1],
                    dynamic_lds: 0,
                },
                mem,
            )
            .map_err(|e| format!("wg({x},{y}): {e}"))?;
        }
    }
    Ok(())
}

fn kernarg_bytes(q: u64, s: u64, x: u64, y: u64, c: &Case, rows: usize) -> Vec<u8> {
    let mut ka = vec![0u8; 128];
    ka[0..8].copy_from_slice(&q.to_le_bytes());
    ka[8..16].copy_from_slice(&s.to_le_bytes());
    ka[16..24].copy_from_slice(&x.to_le_bytes());
    ka[24..32].copy_from_slice(&y.to_le_bytes());
    ka[32..36].copy_from_slice(&(c.m as u32).to_le_bytes());
    ka[36..40].copy_from_slice(&(c.k as u32).to_le_bytes());
    ka[40..44].copy_from_slice(&(rows as u32).to_le_bytes());
    ka[44..48].copy_from_slice(&c.act_scale.to_bits().to_le_bytes());
    ka
}

/// Post-launch invariants shared by kernel and reference runs. Returns the problems and the live output words.
fn check_regions(
    mem: &Memory,
    tag: &str,
    before_in: &[(String, Vec<u8>)],
    rows: usize,
    m: usize,
    live: usize,
) -> (Vec<String>, Vec<u32>) {
    let mut problems = Vec::new();
    for (name, bytes) in before_in {
        let r = region(mem, name);
        if &r.bytes != bytes || r.written.count() != 0 {
            problems.push(format!(
                "{tag}: input region {name} modified (written bytes {})",
                r.written.count()
            ));
        }
    }
    let r = region(mem, "y");
    let data_end = GUARD + rows * m * 4;
    let guard_ok = |lo: usize, hi: usize| words(&r.bytes[lo..hi]).iter().all(|&w| w == SENT);
    if !guard_ok(0, GUARD) || !guard_ok(data_end, data_end + GUARD) {
        problems.push(format!("{tag}: y guard words overwritten"));
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
            "{tag}: y written byte set != logical extent ({} live words), written bytes {}",
            live * m,
            r.written.count()
        ));
    }
    let all = words(&r.bytes[GUARD..data_end]);
    if all[live * m..].iter().any(|&w| w != SENT) {
        problems.push(format!("{tag}: y words beyond the live rows were written"));
    }
    (problems, all[..live * m].to_vec())
}

/// One launch of PM entry `g` over `rows` activation rows. `rows` outside `1..=16*NT` must write nothing.
fn run_kernel(
    emu: &Emulator,
    g: Geo,
    c: &Case,
    x: &[f32],
    rows: usize,
    tag: &str,
) -> Result<Vec<u32>, Bad> {
    assert_eq!(x.len(), rows * c.k);
    let valid = rows >= 1 && rows <= g.max_rows() as usize;
    let live = if valid { rows } else { 0 };
    let mut mem = Memory::default();
    let q = map_guarded(&mut mem, "q", QBASE, c.swz.clone(), WGUARD);
    let sp = map_guarded(&mut mem, "s", SBASE, f32_bytes(&c.sc), WGUARD);
    let xp = map_guarded(&mut mem, "x", XBASE, f32_bytes(x), XGUARD);
    let ysent: Vec<u8> = (0..rows * c.m).flat_map(|_| SENT.to_le_bytes()).collect();
    let yp = map_guarded(&mut mem, "y", YBASE, ysent, SENT);
    mem.map(
        "kernarg".into(),
        KERNARG,
        kernarg_bytes(q, sp, xp, yp, c, rows),
    )
    .unwrap();
    let before_in: Vec<(String, Vec<u8>)> = ["q", "s", "x"]
        .iter()
        .map(|n| (n.to_string(), region(&mem, n).bytes.clone()))
        .collect();
    launch_grid(emu, &mut mem, g.grid(c.m as u32), 1, g.block())
        .map_err(|e| classify(format!("{tag}: emulator: {e}")))?;
    let (problems, out) = check_regions(&mem, tag, &before_in, rows, c.m, live);
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(Bad::Defect(problems.join("; ")))
    }
}

/// One launch of the independent reference (host-decoded row-major BF16 weights).
fn run_reference(
    emu: &Emulator,
    g: Geo,
    c: &Case,
    x: &[f32],
    rows: usize,
    tag: &str,
) -> Result<Vec<u32>, Bad> {
    assert_eq!(x.len(), rows * c.k);
    assert!(rows >= 1 && rows <= 64);
    let mut mem = Memory::default();
    let q = map_guarded(&mut mem, "q", QBASE, u16_bytes(&c.wbf), WGUARD);
    let sp = map_guarded(&mut mem, "s", SBASE, f32_bytes(&c.sc), WGUARD);
    let xp = map_guarded(&mut mem, "x", XBASE, f32_bytes(x), XGUARD);
    let ysent: Vec<u8> = (0..rows * c.m).flat_map(|_| SENT.to_le_bytes()).collect();
    let yp = map_guarded(&mut mem, "y", YBASE, ysent, SENT);
    mem.map(
        "kernarg".into(),
        KERNARG,
        kernarg_bytes(q, sp, xp, yp, c, rows),
    )
    .unwrap();
    let before_in: Vec<(String, Vec<u8>)> = ["q", "s", "x"]
        .iter()
        .map(|n| (n.to_string(), region(&mem, n).bytes.clone()))
        .collect();
    let (gx, gy) = reference_grid(c.m as u32, rows as u32);
    launch_grid(emu, &mut mem, gx, gy, REFERENCE_BLOCK)
        .map_err(|e| classify(format!("{tag}: reference emulator: {e}")))?;
    let _ = g;
    let (problems, out) = check_regions(&mem, tag, &before_in, rows, c.m, rows);
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(Bad::Defect(format!("reference {}", problems.join("; "))))
    }
}

// ------------------------------------------------------------------------------------------------ module / emulators

struct Pm {
    program: Program,
}

impl Pm {
    fn new() -> Self {
        let (_, text, _) = emit_module().unwrap_or_else(|e| panic!("emit_module: {e}"));
        Pm {
            program: lift_text(&text),
        }
    }
}

fn lift_text(text: &str) -> Program {
    let elf = peacemaker_isa::native::assemble(text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn emulators<'a>(
    program: &'a Program,
    syms: &BTreeSet<String>,
    what: &str,
) -> BTreeMap<String, Emulator<'a>> {
    syms.iter()
        .map(|s| {
            let e = Emulator::new(program, s)
                .unwrap_or_else(|err| panic!("{what} {s}: Emulator::new: {err}"));
            (s.clone(), e)
        })
        .collect()
}

/// The reference module for `(geo, k)` pairs, lifted.
fn reference_program(keys: &[(Geo, u32)]) -> Program {
    let (_, text) = reference_module(keys).unwrap_or_else(|e| panic!("reference module: {e}"));
    lift_text(&text)
}

// ------------------------------------------------------------------------------------------------ tally / compare

#[derive(Default)]
struct Tally {
    launches: u64,
    executed: u64,
    words_vs_reference: u64,
    words_vs_host: u64,
    host_skipped: u64,
    words_vs_f64: u64,
    words_vs_singleton: u64,
    refused: Vec<String>,
    fails: Vec<String>,
    nfail: u64,
}

impl Tally {
    fn fail(&mut self, msg: String) {
        self.nfail += 1;
        if self.fails.len() < 24 {
            self.fails.push(msg);
        }
    }
    fn defect(&mut self, label: &str, what: &str, b: Bad) -> bool {
        match b {
            Bad::Refused(m) => {
                self.refused.push(format!("{label} [{what}]: {m}"));
                false
            }
            Bad::Defect(m) => {
                self.fail(format!("{label} [{what}]: {m}"));
                false
            }
        }
    }
    /// `min_exec_pct`: 100 = every launch must execute (no interpreter refusal tolerated); otherwise the share of
    /// launches that must execute (the rest are reported refusals).
    fn finish(self, name: &str, min_exec_pct: u64) {
        log(format!(
            "[{name}] launches={} executed={} words_vs_reference={} words_vs_singleton={} words_vs_host={} (skipped {}) words_vs_f64={} refused={} failures={}",
            self.launches,
            self.executed,
            self.words_vs_reference,
            self.words_vs_singleton,
            self.words_vs_host,
            self.host_skipped,
            self.words_vs_f64,
            self.refused.len(),
            self.nfail
        ));
        for r in self.refused.iter().take(6) {
            log(format!("[{name}] REFUSED (interpreter, not a pass): {r}"));
        }
        log(format!(
            "[{name}] semantics: BF16 WMMA hardware-qualified; E4M3 decode NaN sign and act exceptional domains emulator-model only -> hardware byte oracle is the receipt"
        ));
        assert!(
            self.nfail == 0,
            "{name}: {} failures:\n{}",
            self.nfail,
            self.fails.join("\n")
        );
        assert!(self.executed > 0, "{name}: nothing executed");
        if require_no_refusal() {
            assert!(self.refused.is_empty(), "{name}: interpreter refusals");
        } else if min_exec_pct >= 100 {
            assert!(
                self.refused.is_empty(),
                "{name}: finite set-only inputs must never be refused: {:?}",
                self.refused
            );
        } else {
            assert!(
                self.executed * 100 >= self.launches * min_exec_pct,
                "{name}: fewer than {min_exec_pct}% of the launches executed ({} of {}): {:?}",
                self.executed,
                self.launches,
                self.refused.iter().take(4).collect::<Vec<_>>()
            );
        }
    }
}

fn compare_words(t: &mut Tally, label: &str, what: &str, got: &[u32], want: &[u32], m: usize) {
    if got.len() != want.len() {
        t.fail(format!(
            "{label} [{what}]: {} words != {}",
            got.len(),
            want.len()
        ));
        return;
    }
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if !bad.is_empty() {
        let shown: Vec<String> = bad
            .iter()
            .take(4)
            .map(|&i| {
                format!(
                    "y[{}][{}] got {:#010x} want {:#010x}",
                    i / m,
                    i % m,
                    got[i],
                    want[i]
                )
            })
            .collect();
        t.fail(format!(
            "{label} [{what}]: {} of {} words differ: {}",
            bad.len(),
            got.len(),
            shown.join(", ")
        ));
    }
}

/// Kernel launch vs independent reference launch, bit for bit. Returns the kernel words when both executed.
#[allow(clippy::too_many_arguments)]
fn kernel_vs_reference(
    t: &mut Tally,
    label: &str,
    pm: &BTreeMap<String, Emulator>,
    rf: &BTreeMap<String, Emulator>,
    g: Geo,
    c: &Case,
    x: &[f32],
    rows: usize,
) -> Option<Vec<u32>> {
    t.launches += 1;
    let kr = run_kernel(&pm[&g.symbol()], g, c, x, rows, label);
    let rr = run_reference(&rf[&reference_symbol(g, c.k as u32)], g, c, x, rows, label);
    let (kw, rw) = match (kr, rr) {
        (Ok(k), Ok(r)) => (k, r),
        (Err(e), _) => {
            t.defect(label, "kernel", e);
            return None;
        }
        (_, Err(e)) => {
            t.defect(label, "reference", e);
            return None;
        }
    };
    t.executed += 1;
    t.words_vs_reference += kw.len() as u64;
    compare_words(
        t,
        label,
        "kernel vs independent WMMA reference",
        &kw,
        &rw,
        c.m,
    );
    Some(kw)
}

fn is_subnormal(w: u32) -> bool {
    (w >> 23) & 0xff == 0 && w & 0x7f_ffff != 0
}

/// Host model of the `set` epilogue vs the kernel words (finite words exact, NaN by NaN-ness, subnormals skipped).
fn host_check(t: &mut Tally, label: &str, g: Geo, c: &Case, x: &[f32], rows: usize, y: &[u32]) {
    if g.act {
        return;
    }
    let want = host_model_set(&c.wbf, c.m, c.k, x, rows, g.kw as usize, &c.sc);
    for i in 0..y.len() {
        let (a, b) = (y[i], want[i]);
        if f32::from_bits(a).is_nan() || f32::from_bits(b).is_nan() {
            if f32::from_bits(a).is_nan() != f32::from_bits(b).is_nan() {
                t.fail(format!(
                    "{label} [host model]: y[{}][{}] NaN-ness differs: got {a:#010x} want {b:#010x}",
                    i / c.m,
                    i % c.m
                ));
                return;
            }
            t.words_vs_host += 1;
        } else if is_subnormal(a) || is_subnormal(b) {
            t.host_skipped += 1;
        } else {
            t.words_vs_host += 1;
            if a != b {
                t.fail(format!(
                    "{label} [host model]: y[{}][{}] got {a:#010x} want {b:#010x}",
                    i / c.m,
                    i % c.m
                ));
                return;
            }
        }
    }
}

fn rt32(x: f32) -> f32 {
    let u = x.to_bits();
    if x.is_nan() || (u >> 23) & 0xff == 0xff {
        return x;
    }
    f32::from_bits(u.wrapping_add(((u >> 16) & 1) + 0x7fff) & 0xffff_0000)
}

/// f64 reference: decoded weights x BF16 x, rel 1e-5 when the sum does not cancel, else the cancellation-aware
/// absolute bound; `act` loosely against an f64 silu chain (two BF16 ulps; the exact bytes are pinned by the
/// independent reference). Finite inputs only.
fn f64_check(t: &mut Tally, label: &str, g: Geo, c: &Case, x: &[f32], rows: usize, y: &[u32]) {
    let wf: Vec<f64> = c
        .wbf
        .iter()
        .map(|&b| f64::from(f32::from_bits(u32::from(b) << 16)))
        .collect();
    let steps = (c.k / 16) as f64 + g.kw as f64 + 2.0;
    let unit = steps * 2f64.powi(-22);
    for n in 0..rows {
        let xr = &x[n * c.k..(n + 1) * c.k];
        for m in 0..c.m {
            let w = &wf[m * c.k..(m + 1) * c.k];
            let (mut dot, mut sabs) = (0f64, 0f64);
            for k in 0..c.k {
                let p = w[k] * f64::from(xr[k]);
                dot += p;
                sabs += p.abs();
            }
            let sc = f64::from(c.sc[m]);
            let got = f64::from(f32::from_bits(y[n * c.m + m]));
            let abs_tol = unit * sabs * sc.abs() + 1e-37;
            let (want, tol) = if g.act {
                let pre = (dot * sc) as f32;
                let u = rt32(pre);
                let v = rt32(c.act_scale * u);
                let e = (-f64::from(v)).exp();
                let yv = rt32((f64::from(v) / (1.0 + e)) as f32);
                let yv = f64::from(yv);
                (
                    yv,
                    yv.abs() * 2f64.powi(-6) + 2.0 * abs_tol * f64::from(c.act_scale.abs()) + 1e-30,
                )
            } else {
                let want = dot * sc;
                let strict = sabs * sc.abs() <= 4.0 * want.abs();
                (
                    want,
                    want.abs() * 1e-5 + if strict { 0.0 } else { abs_tol } + 1e-37,
                )
            };
            t.words_vs_f64 += 1;
            if !got.is_finite() || (got - want).abs() > tol {
                t.fail(format!(
                    "{label} [f64]: y[{n}][{m}] got {got:e} want {want:e} (tol {tol:e}, sum|wx|*|s| {:e})",
                    sabs * sc.abs()
                ));
                return;
            }
        }
    }
}

fn nt_for(rows: usize) -> u32 {
    rows.div_ceil(16) as u32
}

fn syms_of(geos: &[Geo]) -> BTreeSet<String> {
    geos.iter().map(|g| g.symbol()).collect()
}

// ------------------------------------------------------------------------------------------------ swizzle layout

/// Expected byte of the swizzled tensor derived from the gfx1201 `f16_loc` fragment layout of
/// `peacemaker-ir/src/emu/wmma.rs` (NOT from the generator): tile `(mt, kt)` is 512 B at `(mt*(K/32) + kt)*512`, lane
/// `l` holds 16 contiguous bytes, the first 8 for K16 step 0 and the next 8 for step 1; inside a step the lane's 8
/// elements are `A[r, kk]` with `lane = ((kk>>2)&1)*16 + r`, `vgpr = (kk>>3)*2 + ((kk>>1)&1)`, `half = kk&1`, so the
/// byte index inside the step is `vgpr*2 + half`.
fn expected_swizzle(w: &[u8], m: usize, k: usize) -> Vec<u8> {
    let mut q = vec![0u8; m * k];
    let ktiles = k / 32;
    for mt in 0..m / 16 {
        for kt in 0..ktiles {
            for step in 0..2 {
                for r in 0..16 {
                    for kk in 0..16 {
                        let lane = ((kk >> 2) & 1) * 16 + r;
                        let vgpr = (kk >> 3) * 2 + ((kk >> 1) & 1);
                        let half = kk & 1;
                        let byte =
                            (mt * ktiles + kt) * 512 + lane * 16 + step * 8 + vgpr * 2 + half;
                        q[byte] = w[(mt * 16 + r) * k + kt * 32 + step * 16 + kk];
                    }
                }
            }
        }
    }
    q
}

#[test]
fn swizzle_matches_the_independently_derived_wmma_fragment_layout() {
    for (seed, (m, k)) in [
        (16usize, 32usize),
        (16, 64),
        (32, 32),
        (48, 96),
        (64, 320),
        (128, 256),
    ]
    .into_iter()
    .enumerate()
    {
        let mut rng = Rng::new(0x5a17 + seed as u64);
        let w: Vec<u8> = (0..m * k).map(|_| rng.next() as u8).collect();
        let q = swizzle_hc8(&w, m, k);
        assert_eq!(q.len(), m * k, "{m}x{k}: swizzled length");
        assert_eq!(q, expected_swizzle(&w, m, k), "{m}x{k}: lane/byte map");
        // The lane map is a bijection: every input byte appears exactly once.
        let mut a = w.clone();
        let mut b = q.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "{m}x{k}: permutation");
    }
    // One-hot probes: the byte of A[row, col] lands exactly where the fragment layout says.
    let (m, k) = (32usize, 64usize);
    for row in [0usize, 1, 15, 16, 31] {
        for col in [0usize, 1, 3, 4, 7, 8, 12, 15, 16, 17, 31, 32, 47, 63] {
            let mut w = vec![0u8; m * k];
            w[row * k + col] = 0xA5;
            let q = swizzle_hc8(&w, m, k);
            let hits: Vec<usize> = (0..q.len()).filter(|&i| q[i] != 0).collect();
            let (mt, r, kt, kin) = (row / 16, row % 16, col / 32, col % 32);
            let (step, kk) = (kin / 16, kin % 16);
            let lane = ((kk >> 2) & 1) * 16 + r;
            let byte = (mt * (k / 32) + kt) * 512
                + lane * 16
                + step * 8
                + (kk >> 3) * 4
                + ((kk >> 1) & 1) * 2
                + (kk & 1);
            assert_eq!(hits, vec![byte], "one-hot A[{row}][{col}]");
        }
    }
}

#[test]
fn swizzle_round_trips_both_ways() {
    for (seed, (m, k)) in [(16usize, 32usize), (32, 64), (64, 320), (128, 1024)]
        .into_iter()
        .enumerate()
    {
        let mut rng = Rng::new(0x77 + seed as u64);
        let w: Vec<u8> = (0..m * k).map(|_| rng.next() as u8).collect();
        let q = swizzle_hc8(&w, m, k);
        assert_eq!(unswizzle_hc8(&q, m, k), w, "{m}x{k}: unswizzle(swizzle(w))");
        let q2: Vec<u8> = (0..m * k).map(|_| rng.next() as u8).collect();
        assert_eq!(
            swizzle_hc8(&unswizzle_hc8(&q2, m, k), m, k),
            q2,
            "{m}x{k}: swizzle(unswizzle(q))"
        );
    }
}

// ------------------------------------------------------------------------------------------------ module structure

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

/// Mnemonics of the instruction lines of one emitted entry (directives, labels and comments removed).
fn mnemonics(text: &str) -> Vec<String> {
    let head = text.split(".amdgpu_metadata").next().unwrap();
    let body = head.split_once(".text").map_or(head, |(_, b)| b);
    body.lines()
        .filter_map(|l| {
            let l = l
                .split(';')
                .next()
                .unwrap()
                .split("//")
                .next()
                .unwrap()
                .trim();
            if l.is_empty() || l.starts_with('.') || l.ends_with(':') {
                None
            } else {
                l.split_whitespace().next().map(str::to_string)
            }
        })
        .collect()
}

fn expected_geos() -> Vec<Geo> {
    let mut v = Vec::new();
    for fmt in FMTS {
        for (mw, kw) in CLASSES {
            for nt in 1..=4 {
                for act in [false, true] {
                    v.push(geo(fmt, mw, kw, nt, act));
                }
            }
        }
    }
    v
}

#[test]
fn module_record_abi_resources_and_native_roundtrip() {
    assert_eq!(MODULE, "hc8_wmma_pm_gfx1201");
    let (emitted, text, proof) = emit_module().unwrap();
    let (_, text2, proof2) = emit_module().unwrap();
    assert_eq!(text, text2, "emission must be deterministic");
    assert_eq!(proof.s_text_sha256, proof2.s_text_sha256);
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1201"));
    let want: Vec<String> = expected_geos().into_iter().map(expected_symbol).collect();
    let want_set: BTreeSet<String> = want.iter().cloned().collect();
    assert_eq!(want.len(), 80, "2 fmt x 5 classes x 4 NT x 2 epilogues");
    assert_eq!(want_set.len(), 80);
    let got: Vec<String> = symbols();
    assert_eq!(
        got.iter().cloned().collect::<BTreeSet<_>>(),
        want_set,
        "exported symbols (spec naming rule)"
    );
    assert_eq!(got.len(), 80, "no duplicate or missing export");
    for g in expected_geos() {
        assert_eq!(g.symbol(), expected_symbol(g), "Geo::symbol");
        assert_eq!(g.block(), 32 * g.mw * g.kw, "{}: block", expected_symbol(g));
        assert_eq!(g.max_rows(), 16 * g.nt, "{}: max_rows", expected_symbol(g));
        assert!([4, 8, 16].contains(&(g.mw * g.kw)));
        for m in [16 * g.mw, 32 * g.mw, 160 * g.mw] {
            assert_eq!(g.grid(m), m / (16 * g.mw), "{}: grid", expected_symbol(g));
        }
    }
    assert_eq!(emitted.len(), 80);
    assert_eq!(proof.kernels.len(), 80);
    let forbidden = [
        "scratch_",
        "buffer_atomic",
        "global_atomic",
        "flat_",
        "v_swmmac",
    ];
    let mut vgprs = BTreeMap::new();
    for sym in &want {
        let owners: Vec<_> = emitted
            .iter()
            .filter(|e| e.s_text.contains(&format!(".globl {sym}\n")))
            .collect();
        assert_eq!(owners.len(), 1, "{sym}: exactly one emitted entry");
        let em = owners[0];
        let t = &em.s_text;
        assert_eq!(em.proof.arch, ARCH, "{sym}");
        assert!(
            t.contains(&format!(".name: {sym}\n")),
            "{sym}: metadata name"
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
        assert!(t.contains(".kernarg_segment_size: 48\n"), "{sym}: ABI 48 B");
        assert_eq!(
            field(t, ".offset:"),
            [0, 8, 16, 24, 32, 36, 40, 44],
            "{sym}: kernarg offsets q s x y M K rows act_scale"
        );
        assert_eq!(
            field(t, ".size:"),
            [8, 8, 8, 8, 4, 4, 4, 4],
            "{sym}: kernarg sizes"
        );
        let g = expected_geos()
            .into_iter()
            .find(|g| expected_symbol(*g) == *sym)
            .unwrap();
        assert_eq!(
            field(t, ".max_flat_workgroup_size:"),
            [32 * g.mw * g.kw],
            "{sym}: workgroup size"
        );
        let lds = field(t, ".group_segment_fixed_size:");
        assert_eq!(lds.len(), 1, "{sym}: LDS descriptor");
        assert!(
            lds[0] <= 65536,
            "{sym}: LDS {} B over the 64 KiB limit",
            lds[0]
        );
        let vg = field(t, ".vgpr_count:");
        let sg = field(t, ".sgpr_count:");
        assert!(vg.len() == 1 && vg[0] <= 256, "{sym}: vgpr_count {vg:?}");
        let ms = mnemonics(t);
        for m in &ms {
            assert!(
                !forbidden.iter().any(|f| m.starts_with(f))
                    && !m.contains("atomic")
                    && !m.contains("cmpswap"),
                "{sym}: forbidden instruction {m}"
            );
        }
        assert!(
            ms.iter().any(|m| m == "v_wmma_f32_16x16x16_bf16"),
            "{sym}: BF16 WMMA"
        );
        let cvt = ms.iter().any(|m| m.starts_with("v_cvt_pk_f32_fp8"));
        assert_eq!(cvt, g.fmt == Fmt::E4M3, "{sym}: v_cvt_pk_f32_fp8 iff E4M3");
        let act_ops = ms.iter().any(|m| m.starts_with("v_div_fixup"));
        assert_eq!(act_ops, g.act, "{sym}: hc_act division sequence iff act");
        vgprs.insert(
            sym.clone(),
            (vg[0], sg.first().copied().unwrap_or(0), lds[0]),
        );
    }
    for (sym, (v, s, l)) in &vgprs {
        log(format!("[resources] {sym} vgprs={v} sgprs={s} lds={l}"));
    }
    // Native assemble + lift is exact: every symbol lifts and constructs an emulator.
    let program = lift_text(&text);
    let syms: BTreeSet<String> = want_set;
    let _ = emulators(&program, &syms, "PM entry");
}

#[test]
fn qualification_notes_are_recorded() {
    // The only independently hardware-qualified numeric the oracle leans on is the BF16 WMMA dot; the notes it cites
    // live in float_edges.md. This test keeps that provenance claim honest.
    let path = format!(
        "{}/../peacemaker-ir/isa/emu/float_edges.md",
        env!("CARGO_MANIFEST_DIR")
    );
    let doc = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(doc.contains("`v_wmma_f32_16x16x16_bf16`"));
    assert!(
        doc.contains("0 mismatches over 7,437,056 words"),
        "float_edges.md no longer records the gfx1201 BF16 WMMA receipt"
    );
}

// ------------------------------------------------------------------------------------------------ decode / reference sanity

#[test]
fn host_decode_matches_the_emulator_fp8_model_and_int8_is_exact() {
    for b in 0..=255u8 {
        let d = decode_weights(Fmt::E4M3, &[b])[0];
        let model = peacemaker_ir::emu::convert::fp8_to_f32(b);
        assert_eq!(
            u32::from(d) << 16,
            model,
            "E4M3 code {b:#04x}: host decode {d:#06x} vs emulator model {model:#010x}"
        );
        let i = decode_weights(Fmt::I8, &[b])[0];
        assert_eq!(
            f32::from_bits(u32::from(i) << 16),
            f32::from(b as i8),
            "int8 {}",
            b as i8
        );
    }
    // pinned spot values (OCP E4M3)
    for (code, want) in [
        (0x00u8, 0.0f32),
        (0x01, 2f32.powi(-9)),
        (0x08, 2f32.powi(-6)),
        (0x38, 1.0),
        (0x7e, 448.0),
        (0xfe, -448.0),
        (0x30, 0.5),
    ] {
        assert_eq!(
            f32::from_bits(u32::from(decode_weights(Fmt::E4M3, &[code])[0]) << 16),
            want,
            "E4M3 {code:#04x}"
        );
    }
    assert_eq!(
        decode_weights(Fmt::E4M3, &[0x80])[0],
        0x8000,
        "-0 keeps its sign"
    );
    assert_eq!(decode_weights(Fmt::E4M3, &[0x7f])[0], 0x7fc0, "NaN +");
    assert_eq!(decode_weights(Fmt::E4M3, &[0xff])[0], 0xffc0, "NaN -");
    assert_eq!(decode_weights(Fmt::I8, &[0x80])[0], 0xc300, "int8 -128");
    assert_eq!(decode_weights(Fmt::I8, &[0x7f])[0], 0x42fe, "int8 127");
    assert_eq!(decode_weights(Fmt::I8, &[0x00])[0], 0x0000, "int8 0 is +0");
}

#[test]
fn reference_kernel_matches_host_wmma_model_and_f64_for_every_kw() {
    // Validates the independent reference itself (no PM kernel involved): its emulated words equal the pure-Rust model
    // built from the hardware-qualified `mma::wmma_f32_bf16`, and the f64 reference.
    let mut t = Tally::default();
    let cases: [(u32, usize, usize); 4] = [(1, 16, 64), (4, 32, 512), (8, 16, 512), (16, 16, 1024)];
    let mut keys = Vec::new();
    for &(kw, _, k) in &cases {
        for act in [false, true] {
            keys.push((geo(Fmt::E4M3, 1, kw, 4, act), k as u32));
        }
    }
    let rprog = reference_program(&keys);
    let rsyms: BTreeSet<String> = keys.iter().map(|&(g, k)| reference_symbol(g, k)).collect();
    let rf = emulators(&rprog, &rsyms, "reference");
    for (ci, &(kw, m, k)) in cases.iter().enumerate() {
        for fmt in FMTS {
            for act in [false, true] {
                let g = geo(fmt, 1, kw, 4, act);
                let c = rand_case(
                    0x7e0 + ci as u64 * 17 + act as u64,
                    fmt,
                    m,
                    k,
                    act,
                    QKind::Full,
                );
                let mut rng = Rng::new(0xabc + ci as u64);
                let kinds: &[usize] = if act { &MODERATE_KINDS } else { &[0, 1, 2, 4] };
                let pool = gen_pool(&mut rng, k, 33, kinds);
                for rows in [1usize, 16, 17, 33] {
                    let label = format!("reference {} kw{kw} M{m} K{k} rows{rows}", fmt_name(fmt));
                    let x: Vec<f32> = pool[..rows].iter().flatten().copied().collect();
                    t.launches += 1;
                    match run_reference(
                        &rf[&reference_symbol(g, k as u32)],
                        g,
                        &c,
                        &x,
                        rows,
                        &label,
                    ) {
                        Ok(y) => {
                            t.executed += 1;
                            host_check(&mut t, &label, g, &c, &x, rows, &y);
                            f64_check(&mut t, &label, g, &c, &x, rows, &y);
                        }
                        Err(e) => {
                            t.defect(&label, "reference", e);
                        }
                    }
                }
            }
        }
    }
    t.finish("reference_vs_host_model_and_f64", 50);
}

// ------------------------------------------------------------------------------------------------ every rows 1..=64

/// Every logical row count 1..=64 on the entry `NT = ceil(rows/16)` of one class (both formats, both epilogues, the
/// smallest shape): kernel == independent reference bit for bit; every output row == the `rows = 1` launch of the same
/// input row; one row replicated at every position and rotated slot order at the NT boundaries.
fn sweep_class(mw: u32, kw: u32) {
    let (m, k) = small_shape(mw, kw);
    let pm = Pm::new();
    let name = format!("rows_1_to_64_mw{mw}_kw{kw}");
    let mut t = Tally::default();
    for (fi, fmt) in FMTS.into_iter().enumerate() {
        for act in [false, true] {
            let seed = 0x1000 + (mw * 100 + kw) as u64 * 10 + (fi * 2 + act as usize) as u64;
            let c = rand_case(seed, fmt, m, k, act, QKind::Full);
            let mut rng = Rng::new(seed ^ 0xfeed);
            let kinds: &[usize] = if act { &MODERATE_KINDS } else { &ALL_KINDS };
            let pool = gen_pool(&mut rng, k, 64, kinds);
            let geos: Vec<Geo> = (1..=4).map(|nt| geo(fmt, mw, kw, nt, act)).collect();
            let pm_e = emulators(&pm.program, &syms_of(&geos), "PM entry");
            let rprog = reference_program(&[(geos[0], k as u32)]);
            let rsyms: BTreeSet<String> =
                [reference_symbol(geos[0], k as u32)].into_iter().collect();
            let rf = emulators(&rprog, &rsyms, "reference");
            let tag = format!("{} act={act}", fmt_name(fmt));
            // Singleton outputs of every pool row (NT = 1 entry, rows = 1).
            let singles: Vec<Option<Vec<u32>>> = (0..64)
                .map(|j| {
                    t.launches += 1;
                    let label = format!("{name} {tag} singleton row {j}");
                    match run_kernel(&pm_e[&geos[0].symbol()], geos[0], &c, &pool[j], 1, &label) {
                        Ok(y) => {
                            t.executed += 1;
                            Some(y)
                        }
                        Err(e) => {
                            t.defect(&label, "singleton", e);
                            None
                        }
                    }
                })
                .collect();
            for rows in 1..=64usize {
                let g = geos[nt_for(rows) as usize - 1];
                let label = format!("{name} {tag} rows={rows}");
                let x: Vec<f32> = pool[..rows].iter().flatten().copied().collect();
                if let Some(y) = kernel_vs_reference(&mut t, &label, &pm_e, &rf, g, &c, &x, rows) {
                    for r in 0..rows {
                        if let Some(s) = &singles[r] {
                            t.words_vs_singleton += c.m as u64;
                            compare_words(
                                &mut t,
                                &label,
                                &format!("row {r} vs rows=1 singleton"),
                                &y[r * c.m..(r + 1) * c.m],
                                s,
                                c.m,
                            );
                        }
                    }
                }
            }
            // Same input row at every position, and rotated slot order.
            for (rows, j) in [
                (2usize, 0usize),
                (16, 3),
                (17, 9),
                (32, 31),
                (33, 40),
                (48, 47),
                (64, 63),
            ] {
                let g = geos[nt_for(rows) as usize - 1];
                let label = format!("{name} {tag} replicate row {j} x{rows}");
                let x: Vec<f32> = (0..rows).flat_map(|_| pool[j].iter().copied()).collect();
                t.launches += 1;
                match run_kernel(&pm_e[&g.symbol()], g, &c, &x, rows, &label) {
                    Ok(y) => {
                        t.executed += 1;
                        if let Some(s) = &singles[j] {
                            for r in 0..rows {
                                t.words_vs_singleton += c.m as u64;
                                compare_words(
                                    &mut t,
                                    &label,
                                    &format!("position {r}"),
                                    &y[r * c.m..(r + 1) * c.m],
                                    s,
                                    c.m,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        t.defect(&label, "kernel", e);
                    }
                }
            }
            for (rows, rot) in [(17usize, 5usize), (48, 17), (64, 33)] {
                let g = geos[nt_for(rows) as usize - 1];
                let label = format!("{name} {tag} rotated by {rot} rows={rows}");
                let order: Vec<usize> = (0..rows).map(|i| (i + rot) % 64).collect();
                let x: Vec<f32> = order
                    .iter()
                    .flat_map(|&j| pool[j].iter().copied())
                    .collect();
                t.launches += 1;
                match run_kernel(&pm_e[&g.symbol()], g, &c, &x, rows, &label) {
                    Ok(y) => {
                        t.executed += 1;
                        for (r, &j) in order.iter().enumerate() {
                            if let Some(s) = &singles[j] {
                                t.words_vs_singleton += c.m as u64;
                                compare_words(
                                    &mut t,
                                    &label,
                                    &format!("slot {r} = pool row {j}"),
                                    &y[r * c.m..(r + 1) * c.m],
                                    s,
                                    c.m,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        t.defect(&label, "kernel", e);
                    }
                }
            }
        }
    }
    t.finish(&name, 50);
}

#[test]
fn rows_1_to_64_mw1_kw8() {
    sweep_class(1, 8);
}
#[test]
fn rows_1_to_64_mw1_kw16() {
    sweep_class(1, 16);
}
#[test]
fn rows_1_to_64_mw4_kw1() {
    sweep_class(4, 1);
}
#[test]
fn rows_1_to_64_mw8_kw1() {
    sweep_class(8, 1);
}
#[test]
fn rows_1_to_64_mw2_kw4() {
    sweep_class(2, 4);
}

// ------------------------------------------------------------------------------------------------ entry matrix

/// All 40 entries of one format (class x NT x epilogue) on the main shapes: capacity padding (rows below the NT
/// capacity, including `rows = 1` on `NT = 4`), the NT boundaries `16(NT-1)+1` and `16 NT`: kernel == independent
/// reference; host model; f64 reference.
fn entry_matrix(fmt: Fmt) {
    let pm = Pm::new();
    let name = format!("entry_matrix_{}", fmt_name(fmt));
    let mut t = Tally::default();
    for (ci, (mw, kw)) in CLASSES.into_iter().enumerate() {
        let (m, k) = main_shape(mw, kw);
        for act in [false, true] {
            let geos: Vec<Geo> = (1..=4).map(|nt| geo(fmt, mw, kw, nt, act)).collect();
            let pm_e = emulators(&pm.program, &syms_of(&geos), "PM entry");
            let rprog = reference_program(&[(geos[0], k as u32)]);
            let rsyms: BTreeSet<String> =
                [reference_symbol(geos[0], k as u32)].into_iter().collect();
            let rf = emulators(&rprog, &rsyms, "reference");
            let seed =
                0x2000 + (ci * 2 + act as usize) as u64 + if fmt == Fmt::I8 { 0x100 } else { 0 };
            let c = rand_case(seed, fmt, m, k, act, QKind::Full);
            let mut rng = Rng::new(seed ^ 0xbeef);
            let kinds: &[usize] = if act { &MODERATE_KINDS } else { &[0, 1, 2, 4] };
            let pool = gen_pool(&mut rng, k, 64, kinds);
            for g in &geos {
                let nt = g.nt as usize;
                let mut rows_list = vec![1, 16 * (nt - 1) + 1, 16 * nt];
                rows_list.sort_unstable();
                rows_list.dedup();
                for rows in rows_list {
                    let label = format!("{name} {} rows={rows}", g.symbol());
                    let x: Vec<f32> = pool[..rows].iter().flatten().copied().collect();
                    if let Some(y) =
                        kernel_vs_reference(&mut t, &label, &pm_e, &rf, *g, &c, &x, rows)
                    {
                        host_check(&mut t, &label, *g, &c, &x, rows, &y);
                        f64_check(&mut t, &label, *g, &c, &x, rows, &y);
                    }
                }
            }
        }
    }
    t.finish(&name, 50);
}

#[test]
fn entry_matrix_e4m3() {
    entry_matrix(Fmt::E4M3);
}
#[test]
fn entry_matrix_i8() {
    entry_matrix(Fmt::I8);
}

#[test]
fn invalid_row_counts_write_nothing_in_every_entry() {
    let pm = Pm::new();
    let mut t = Tally::default();
    let all = expected_geos();
    let pm_e = emulators(&pm.program, &syms_of(&all), "PM entry");
    for g in all {
        let (m, k) = small_shape(g.mw, g.kw);
        let c = rand_case(0x3000 + g.nt as u64, g.fmt, m, k, g.act, QKind::Small);
        let mut rng = Rng::new(0x3333);
        let cap = g.max_rows() as usize;
        for rows in [0usize, cap + 1, 65] {
            if rows <= cap {
                continue;
            }
            let pool = gen_pool(&mut rng, k, rows, &MODERATE_KINDS);
            let x: Vec<f32> = pool.iter().flatten().copied().collect();
            let label = format!("invalid rows {rows} {}", g.symbol());
            t.launches += 1;
            match run_kernel(&pm_e[&g.symbol()], g, &c, &x, rows, &label) {
                Ok(_) => t.executed += 1,
                Err(e) => {
                    t.defect(&label, "kernel", e);
                }
            }
        }
        let label = format!("invalid rows 0 {}", g.symbol());
        t.launches += 1;
        match run_kernel(&pm_e[&g.symbol()], g, &c, &[], 0, &label) {
            Ok(_) => t.executed += 1,
            Err(e) => {
                t.defect(&label, "kernel", e);
            }
        }
    }
    t.finish("invalid_row_counts", 100);
}

// ------------------------------------------------------------------------------------------------ decode sweep / exceptional inputs

/// Every finite E4M3 code (254 + the two zeros) and every int8 value through one-hot activations with unit scale: the
/// kernel output must be the decoded weight itself. `W[m][k] = code((m*K + k) % 256)`; activation row `n` of launch
/// `base` is one-hot at `k = base + n`.
fn decode_sweep(fmt: Fmt) {
    let (mw, kw) = (4u32, 1u32);
    let (m, k) = (64usize, 128usize);
    let pm = Pm::new();
    let name = format!("decode_sweep_{}", fmt_name(fmt));
    let mut t = Tally::default();
    let g = geo(fmt, mw, kw, 4, false);
    let ga = geo(fmt, mw, kw, 4, true);
    let pm_e = emulators(&pm.program, &syms_of(&[g, ga]), "PM entry");
    let rprog = reference_program(&[(g, k as u32), (ga, k as u32)]);
    let rsyms: BTreeSet<String> = [
        reference_symbol(g, k as u32),
        reference_symbol(ga, k as u32),
    ]
    .into_iter()
    .collect();
    let rf = emulators(&rprog, &rsyms, "reference");
    let raw: Vec<u8> = (0..m * k)
        .map(|i| {
            let b = (i % 256) as u8;
            // E4M3 NaN codes are covered by the NaN test (they poison the whole row of any dot product).
            if fmt == Fmt::E4M3 && b & 0x7f == 0x7f {
                b ^ 1
            } else {
                b
            }
        })
        .collect();
    let mut seen = BTreeSet::new();
    raw.iter().for_each(|&b| {
        seen.insert(b);
    });
    assert_eq!(
        seen.len(),
        if fmt == Fmt::E4M3 { 254 } else { 256 },
        "every finite code present"
    );
    for act in [false, true] {
        let c = make_case(fmt, m, k, raw.clone(), vec![1.0; m], 1.0);
        let gg = if act { ga } else { g };
        for base in (0..k).step_by(64) {
            let x: Vec<f32> = (0..64)
                .flat_map(|n| (0..k).map(move |kk| if kk == base + n { 1.0f32 } else { 0.0 }))
                .collect();
            let label = format!("{name} act={act} base={base}");
            let Some(y) = kernel_vs_reference(&mut t, &label, &pm_e, &rf, gg, &c, &x, 64) else {
                continue;
            };
            if act {
                continue;
            }
            for n in 0..64usize {
                for mm in 0..m {
                    let want = decode_weights(fmt, &[raw[mm * k + base + n]])[0];
                    let want = u32::from(want) << 16;
                    let got = y[n * m + mm];
                    t.words_vs_host += 1;
                    // Mixed-sign zero terms sum to +0 on RNE hardware: a zero weight is compared by magnitude.
                    let ok = if want & 0x7fff_ffff == 0 {
                        got & 0x7fff_ffff == 0
                    } else {
                        got == want
                    };
                    if !ok {
                        t.fail(format!(
                            "{label}: code {:#04x} (m={mm}, k={}) decoded {want:#010x}, kernel {got:#010x}",
                            raw[mm * k + base + n],
                            base + n
                        ));
                    }
                }
            }
        }
    }
    t.finish(&name, 50);
}

#[test]
fn decode_sweep_e4m3() {
    decode_sweep(Fmt::E4M3);
}
#[test]
fn decode_sweep_i8() {
    decode_sweep(Fmt::I8);
}

/// NaN / Inf / signed-zero / extreme inputs through every geometry class (`NT = 2`, `set` and `act`): E4M3 NaN codes
/// `0x7f`/`0xff` as weights, int8 extremes, and activation rows with qNaN, sNaN, negative NaN, +-Inf, signed zeros,
/// max finite and subnormals. The kernel must equal the independent reference bit for bit (NaN payloads and signs
/// included); interpreter refusals of unmeasured domains are counted and reported.
fn exceptional(fmt: Fmt) {
    let pm = Pm::new();
    let name = format!("exceptional_{}", fmt_name(fmt));
    let mut t = Tally::default();
    for (ci, (mw, kw)) in CLASSES.into_iter().enumerate() {
        let (m, k) = small_shape(mw, kw);
        for act in [false, true] {
            let g = geo(fmt, mw, kw, 2, act);
            let pm_e = emulators(&pm.program, &syms_of(&[g]), "PM entry");
            let rprog = reference_program(&[(g, k as u32)]);
            let rsyms: BTreeSet<String> = [reference_symbol(g, k as u32)].into_iter().collect();
            let rf = emulators(&rprog, &rsyms, "reference");
            let seed = 0x4000 + ci as u64 * 8 + act as u64;
            let mut rng = Rng::new(seed);
            // Weights: finite random with extremes; for E4M3 a matrix with NaN codes at known positions.
            let mut raw = gen_q(&mut rng, m * k, fmt, QKind::Full);
            let mut nan_rows: Vec<usize> = Vec::new();
            if fmt == Fmt::E4M3 {
                for (i, code) in [0x7fu8, 0xff].into_iter().enumerate() {
                    let row = (3 + 5 * i) % m;
                    raw[row * k + (7 * i + 1) % k] = code;
                    nan_rows.push(row);
                }
            } else {
                raw[0] = 0x80;
                raw[k + 1] = 0x7f;
            }
            let c = make_case(fmt, m, k, raw, gen_scale(&mut rng, m, act), 1.0);
            // Activation rows.
            let special = |at: usize, v: f32| {
                let mut r = gen_x_row(&mut Rng::new(seed ^ at as u64), k, if act { 0 } else { 4 });
                r[at % k] = v;
                r
            };
            let mut pool: Vec<Vec<f32>> = vec![
                special(1, f32::from_bits(0x7fc0_0000)),
                special(2, f32::from_bits(0x7fa0_0000)),
                special(3, f32::from_bits(0xffc1_0000)),
                special(4, f32::INFINITY),
                special(5, f32::NEG_INFINITY),
                vec![f32::from_bits(0x8000_0000); k],
                vec![0.0; k],
                gen_x_row(&mut rng, k, 3),
                gen_x_row(&mut rng, k, 5),
                (0..k)
                    .map(|i| f32::from_bits(0x7f7f_0000 | if i % 2 == 0 { 0 } else { 0x8000_0000 }))
                    .collect(),
            ];
            // Clean rows alongside, to prove NaN/Inf rows do not leak into their neighbours.
            pool.push(gen_x_row(&mut rng, k, 0));
            pool.push(gen_x_row(&mut rng, k, 0));
            for rows in [pool.len(), 1usize, 3] {
                let sel: Vec<&Vec<f32>> = if rows == 1 {
                    vec![&pool[0]]
                } else if rows == 3 {
                    vec![&pool[10], &pool[3], &pool[11]]
                } else {
                    pool.iter().collect()
                };
                let x: Vec<f32> = sel.iter().flat_map(|r| r.iter().copied()).collect();
                let label = format!("{name} {} rows={rows}", g.symbol());
                let Some(y) = kernel_vs_reference(&mut t, &label, &pm_e, &rf, g, &c, &x, rows)
                else {
                    continue;
                };
                // A NaN activation poisons the whole output row; clean rows stay finite in `set`.
                for (r, row) in sel.iter().enumerate() {
                    let has_nan = row.iter().any(|v| v.is_nan());
                    let out = &y[r * m..(r + 1) * m];
                    if has_nan && !out.iter().all(|&w| f32::from_bits(w).is_nan()) {
                        t.fail(format!(
                            "{label}: row {r} has a NaN activation but a non-NaN output"
                        ));
                    }
                }
                if !act && fmt == Fmt::E4M3 {
                    for (r, row) in sel.iter().enumerate() {
                        if row.iter().all(|v| v.is_finite()) {
                            for &nr in &nan_rows {
                                if !f32::from_bits(y[r * m + nr]).is_nan() {
                                    t.fail(format!("{label}: E4M3 NaN weight in row {nr} did not produce NaN at y[{r}][{nr}]"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    t.finish(&name, 10);
}

#[test]
fn exceptional_inputs_e4m3() {
    exceptional(Fmt::E4M3);
}
#[test]
fn exceptional_inputs_i8() {
    exceptional(Fmt::I8);
}
