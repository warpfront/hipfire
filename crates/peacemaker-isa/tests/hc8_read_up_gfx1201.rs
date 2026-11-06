// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-emulator differential tests of the four fused HC8 read-up entries `hc8_read_up_e4m3_n{1..4}` of module
//! `hc8_wmma_pm_gfx1201` (`peacemaker_isa::kernels::hc8_wmma_gfx1201::{emit_read_up, emit_read_ups, read_up_symbol}`):
//! one launch computes, for `rows` in `1..=16*NT`, the E4M3 up projection `u[t][m] = acc * s[m]` (`m = b*hidden + h`,
//! `M = 4*hidden`, `K = low_rank`) and the product `hyper_read_projected_f32` read mix of `branches = 4` without the
//! projected `[rows, 4*hidden]` activations ever reaching memory: `mixed[t][h] = rt(rt(0 + term_0 + .. + term_3) / 4)`,
//! `term_b = rt(gate_b * rt(normalized[t][b*hidden + h]))`, `gate_b = rt(1 / (1 + expf(-rt(u[t][b*hidden + h]))))`.
//!
//! Every launch executes the emitted PM source on the peacemaker CPU interpreter (`emit_module` -> `native::assemble`
//! -> `lift_object(Builder)` -> `Emulator::run_workgroup` per workgroup, real LDS and barriers); nothing is read from a
//! committed artifact. This file is the sibling of `tests/hc8_wmma_gfx1201.rs` (same harness, same data generators, the
//! same independent WMMA reference) and does not change any GEMM semantic case.
//!
//! References (none shares an instruction, tile, LDS fold or decode with the fused generator):
//!
//! 1. **Decomposed pipeline** (the primary oracle, bit for bit, every output word):
//!    (a) the product `hc8_gemm_e4m3_mw4_kw1_n{NT}_set` entry (the entry the decomposed route launches) on the same
//!    `q`/`s`/`low`, giving `u [rows][4*hidden]`; (b) the independent `hc8_ref_kw1_k{K}_set` WMMA reference of
//!    `support/hc8_reference.rs` (host-decoded row-major weights, own fragment loads), which must reproduce (a) word
//!    for word before it is trusted; then (c) the separately emitted projected-read kernel of
//!    `support/hc8_read_up_reference.rs`, a straight-line transcription of the disassembled gfx1201
//!    `hyper_read_projected_f32` object (`hc8-read-anchor.md`: BF16 round with the NaN payload rule, `exp(-x)` with
//!    `v_ldexp_f32` and unordered clamps, the full `v_div_scale/v_rcp/v_fma/v_div_fmas/v_div_fixup` for the gate and for
//!    `value / 4.0`, `value` from `+0` summed over branches `0..3` in order) fed with (a)'s words. The fused kernel's
//!    output must equal (c)'s output bit for bit, NaN payloads, signs and signed zeros included.
//! 2. **Singleton row invariance**: output row `r` of a batched launch equals, bit for bit, the `rows = 1` launch of the
//!    same `(low row, normalized row)` (batched with distinct rows, one row replicated at every position, rotated slot
//!    order, every row count `1..=64` on its `NT = ceil(rows/16)` entry), and a launch is independent of the NT
//!    capacity of the entry that runs it.
//! 3. **Host model** (`support::host_read_word`): the same DAG in pure Rust on the emulator's raw-bit `v_exp_f32` /
//!    `v_ldexp_f32` models (never the host `exp`), host IEEE `+ * fma` and correctly rounded division; compared on the
//!    finite normal domain only (`safe()`: non-finite, huge `|x|`, subnormal and scaled-division words are skipped and
//!    counted). The decomposed reference kernel itself is cross-checked against the same model in
//!    `projected_reference_matches_host_model_and_known_answers`.
//! 4. **Known answers**: all-zero E4M3 weights give `u = +-0`, `gate = 0.5`, so `mixed = (n0 + n1 + n2 + n3) / 8` for
//!    small-integer normalized words (exact in BF16): computed on the host without any DAG.
//!
//! Also checked on every launch: `q`, `s`, `low`, `normalized` are byte-identical and never written; the written byte
//! set of `mixed` is exactly `rows * hidden * 4` (dead columns `n >= rows` and the capacity padding of the NT tile
//! are never stored, sentinel words and guard words intact); `rows == 0` and `rows > 16*NT` write nothing and
//! terminate. Structural test: symbols, 56 B ABI + offsets (`pad@52` declared), block 128, static LDS `4096*NT`, no
//! scratch/spills/atomics/flat, deterministic emission, entries 82..86 of `emit_module`, native assemble + lift +
//! `Emulator::new`.
//!
//! LIMITS (stated, not hidden): the emulator is the oracle of the instruction semantics. `v_exp_f32` and `v_rcp_f32`
//! are its documented candidate models (not exhaustively hardware-qualified, `peacemaker-ir/src/emu/trans.rs`);
//! `v_div_scale_f32` / `v_div_fmas_f32` / `v_div_fixup_f32` on gfx1201 are its hardware-measured models; the decomposed
//! reference transcribes the observed object and has no equivalence receipt against it beyond that disassembly; the
//! fused kernel and the reference run on the SAME instruction models, so a pass proves identical DAG, order and
//! constants, not the models themselves. `v_cvt_pk_f32_fp8` NaN decode and every exceptional-domain result are
//! emulator-model only. Domains the interpreter refuses (counted, printed `REFUSED`, never a pass;
//! `HC8_ORACLE_REQUIRE_NO_REFUSAL=1` fails on any) are excluded from the comparison. A pass of this file is NOT a
//! hardware receipt: the hardware byte oracle is.
//!
//! Runtime: [INFERENCE, never run: no builds allowed here] under about 1-2 minutes of CPU spread over 7 tests that
//! cargo runs in parallel. `HC8_ORACLE_LOG=<file>` appends every log line to a file.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_ir::Program;
use peacemaker_isa::kernels::hc8_wmma_gfx1201::{
    emit_module, emit_read_up, emit_read_ups, read_up_symbol, swizzle_hc8, symbols, Fmt, Geo,
    MODULE, READ_UP_BLOCK, READ_UP_KERNARG_BYTES,
};
use peacemaker_isa::Arch;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

#[path = "support/hc8_read_up_reference.rs"]
mod hc8_read_up_reference;
#[path = "support/hc8_reference.rs"]
mod hc8_reference;
use hc8_read_up_reference::{
    bf16_rt_bits, host_read_word, projected_grid, projected_module, PROJECTED_BLOCK,
    PROJECTED_SYMBOL,
};
use hc8_reference::{
    decode_weights, reference_grid, reference_module, reference_symbol, REFERENCE_BLOCK,
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

/// Small `(hidden, low_rank)` shapes: `hidden % 16 == 0`, `low_rank % 32 == 0`; several workgroups (grid = hidden/16),
/// the product `low_rank = 320` at the smallest hidden, and `K` tiles beyond the unroll depth.
const SHAPES: [(usize, usize); 5] = [(16, 32), (32, 64), (48, 96), (16, 320), (64, 160)];

/// The decomposed route's GEMM entry for `nt` column tiles.
fn set_geo(nt: u32) -> Geo {
    Geo {
        fmt: Fmt::E4M3,
        mw: 4,
        kw: 1,
        nt,
        act: false,
    }
}

fn nt_for(rows: usize) -> u32 {
    rows.div_ceil(16) as u32
}

/// Row counts of one entry: the spec's `1,3,8,16,17,48,64` that fit, plus the NT boundaries `16(NT-1)+1` and `16 NT`.
fn rows_for(nt: usize) -> Vec<usize> {
    let cap = 16 * nt;
    let mut v: Vec<usize> = [1usize, 3, 8, 16, 17, 48, 64]
        .into_iter()
        .filter(|&r| r <= cap)
        .collect();
    v.push(16 * (nt - 1) + 1);
    v.push(cap);
    v.sort_unstable();
    v.dedup();
    v
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
    /// Every finite E4M3 code.
    Full,
    /// Small magnitudes (exponent field 4..=9): the projected logits stay mostly inside the sigmoid's live range.
    Small,
    /// Mostly zero.
    Sparse,
}

/// Raw E4M3 weights, never a NaN code.
fn gen_q(rng: &mut Rng, n: usize, kind: QKind) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let sign = (rng.next() & 1) as u8;
            match kind {
                QKind::Sparse if rng.below(8) != 0 => 0,
                QKind::Small => (sign << 7) | ((4 + rng.below(6) as u8) << 3) | rng.below(8) as u8,
                _ => {
                    let b = rng.next() as u8;
                    if b & 0x7f == 0x7f {
                        b ^ 1
                    } else {
                        b
                    }
                }
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

/// A BF16-exact f32 `low` row. Kinds: 0 moderate-to-wide, 1 cancelling pairs, 2 very wide exponent, 3 finite edges
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

/// A `normalized` row of `n` f32 words (NOT BF16-exact unless stated: the kernel rounds them). Kinds: 0 full 23-bit
/// mantissas, moderate exponents; 1 BF16-exact; 2 a quarter signed zeros; 3 any normal exponent (overflowing products
/// and sums reachable).
fn gen_norm_row(rng: &mut Rng, n: usize, kind: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let sign = (rng.next() & 1) as u32;
            let mant = (rng.next() as u32) & 0x7f_ffff;
            match kind {
                0 => f32::from_bits((sign << 31) | ((110 + rng.below(31) as u32) << 23) | mant),
                1 => bf(sign, 112 + rng.below(25) as u32, rng.below(128) as u32),
                2 => {
                    if rng.below(4) == 0 {
                        f32::from_bits(sign << 31)
                    } else {
                        f32::from_bits((sign << 31) | ((118 + rng.below(12) as u32) << 23) | mant)
                    }
                }
                _ => f32::from_bits((sign << 31) | ((1 + rng.below(254) as u32) << 23) | mant),
            }
        })
        .collect()
}

fn gen_low_pool(rng: &mut Rng, k: usize, n: usize, kinds: &[usize]) -> Vec<Vec<f32>> {
    (0..n)
        .map(|r| {
            let kind = kinds[(r + rng.below(2)) % kinds.len()];
            gen_x_row(rng, k, kind)
        })
        .collect()
}

fn gen_norm_pool(rng: &mut Rng, width: usize, n: usize, kinds: &[usize]) -> Vec<Vec<f32>> {
    (0..n)
        .map(|r| {
            let kind = kinds[(r + rng.below(2)) % kinds.len()];
            gen_norm_row(rng, width, kind)
        })
        .collect()
}

/// Finite-input kinds of the matrix tests (kind 3, the max-finite / min-normal edge rows that overflow the GEMM to
/// Inf/NaN, belongs to `exceptional_inputs`).
const LOW_KINDS: [usize; 5] = [0, 1, 2, 4, 5];
const NORM_KINDS: [usize; 4] = [0, 1, 2, 3];

/// Per-row scales `2^-10..2^-2`, some negative and some exactly `1.0`, so logits `u` spread over the sigmoid.
fn gen_scale(rng: &mut Rng, m: usize) -> Vec<f32> {
    (0..m)
        .map(|_| {
            let mant = (rng.next() as u32) & 0x7f_ffff;
            let v = f32::from_bits(((117 + rng.below(9) as u32) << 23) | mant);
            match rng.below(16) {
                0 => 1.0,
                1 | 2 => -v,
                _ => v,
            }
        })
        .collect()
}

/// One weight matrix `[M = 4*hidden, K]` with its scales: raw bytes, swizzled bytes, host-decoded BF16.
struct Case {
    hidden: usize,
    k: usize,
    m: usize,
    swz: Vec<u8>,
    wbf: Vec<u16>,
    sc: Vec<f32>,
}

fn make_case(hidden: usize, k: usize, raw: Vec<u8>, sc: Vec<f32>) -> Case {
    let m = 4 * hidden;
    assert_eq!(raw.len(), m * k);
    assert_eq!(sc.len(), m);
    Case {
        hidden,
        k,
        m,
        swz: swizzle_hc8(&raw, m, k),
        wbf: decode_weights(Fmt::E4M3, &raw),
        sc,
    }
}

fn rand_case(seed: u64, hidden: usize, k: usize, kind: QKind) -> Case {
    let mut rng = Rng::new(seed);
    let raw = gen_q(&mut rng, 4 * hidden * k, kind);
    let sc = gen_scale(&mut rng, 4 * hidden);
    make_case(hidden, k, raw, sc)
}

fn flat(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

// ------------------------------------------------------------------------------------------------ memory model

const GUARD: usize = 128;
const KERNARG: u64 = 0x1_0000;
const XBASE: u64 = 0x1_0000_0000;
const QBASE: u64 = 0x2_0000_0000;
const SBASE: u64 = 0x3_0000_0000;
const YBASE: u64 = 0x4_0000_0000;
const NBASE: u64 = 0x5_0000_0000;
const UBASE: u64 = 0x6_0000_0000;
const SENT: u32 = 0x7fc0_a5a5;
const XGUARD: u32 = 0x7fc0_dead;
const NGUARD: u32 = 0x7fc0_beef;
const WGUARD: u32 = 0x7fc1_7fc1;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
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
fn sentinel(n_words: usize) -> Vec<u8> {
    (0..n_words).flat_map(|_| SENT.to_le_bytes()).collect()
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
    /// A documented interpreter refusal of an unmeasured domain (never a pass: counted and reported).
    Refused(String),
    /// Everything else: a defect.
    Defect(String),
}

/// The closed list of `v_div_scale_f32` refusals of unqualified exponent domains (`div_scale_chain` of
/// `peacemaker-ir/src/emu/alu.rs`), exact messages; gfx1201 itself is fully measured there.
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

fn put_u64(ka: &mut [u8], at: usize, v: u64) {
    ka[at..at + 8].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(ka: &mut [u8], at: usize, v: u32) {
    ka[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Fused read-up kernarg: `q@0 s@8 low@16 normalized@24 mixed@32` (u64), `hidden@40 low_rank@44 rows@48`, `pad@52`.
fn fused_kernarg(ptrs: [u64; 5], c: &Case, rows: usize) -> Vec<u8> {
    let mut ka = vec![0u8; 128];
    for (i, p) in ptrs.into_iter().enumerate() {
        put_u64(&mut ka, 8 * i, p);
    }
    put_u32(&mut ka, 40, c.hidden as u32);
    put_u32(&mut ka, 44, c.k as u32);
    put_u32(&mut ka, 48, rows as u32);
    ka
}

/// GEMM (`set` entry and WMMA reference) kernarg: `q@0 s@8 x@16 y@24`, `M@32 K@36 rows@40 act_scale@44`.
fn gemm_kernarg(ptrs: [u64; 4], c: &Case, rows: usize) -> Vec<u8> {
    let mut ka = vec![0u8; 128];
    for (i, p) in ptrs.into_iter().enumerate() {
        put_u64(&mut ka, 8 * i, p);
    }
    put_u32(&mut ka, 32, c.m as u32);
    put_u32(&mut ka, 36, c.k as u32);
    put_u32(&mut ka, 40, rows as u32);
    put_u32(&mut ka, 44, 1.0f32.to_bits());
    ka
}

/// Projected reference kernarg: `up@0 normalized@8 mixed@16`, `hidden@24 rows@28`.
fn projected_kernarg(ptrs: [u64; 3], hidden: usize, rows: usize) -> Vec<u8> {
    let mut ka = vec![0u8; 128];
    for (i, p) in ptrs.into_iter().enumerate() {
        put_u64(&mut ka, 8 * i, p);
    }
    put_u32(&mut ka, 24, hidden as u32);
    put_u32(&mut ka, 28, rows as u32);
    ka
}

/// Post-launch invariants shared by every kernel and reference run. `out` is `[rows][cols]` words in region `y`;
/// exactly `live_rows * cols` of them may have been written, the rest (and the guards) must hold the sentinel.
fn check_regions(
    mem: &Memory,
    tag: &str,
    before_in: &[(String, Vec<u8>)],
    rows: usize,
    cols: usize,
    live_rows: usize,
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
    let data_end = GUARD + rows * cols * 4;
    let guard_ok = |lo: usize, hi: usize| words(&r.bytes[lo..hi]).iter().all(|&w| w == SENT);
    if !guard_ok(0, GUARD) || !guard_ok(data_end, data_end + GUARD) {
        problems.push(format!("{tag}: y guard words overwritten"));
    }
    let want_written = live_rows * cols * 4;
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
            live_rows * cols,
            r.written.count()
        ));
    }
    let all = words(&r.bytes[GUARD..data_end]);
    if all[live_rows * cols..].iter().any(|&w| w != SENT) {
        problems.push(format!("{tag}: y words beyond the live rows were written"));
    }
    (problems, all[..live_rows * cols].to_vec())
}

fn input_snapshot(mem: &Memory, names: &[&str]) -> Vec<(String, Vec<u8>)> {
    names
        .iter()
        .map(|n| (n.to_string(), region(mem, n).bytes.clone()))
        .collect()
}

fn finish_run(
    mem: Memory,
    tag: &str,
    before_in: &[(String, Vec<u8>)],
    rows: usize,
    cols: usize,
    live: usize,
) -> Result<Vec<u32>, Bad> {
    let (problems, out) = check_regions(&mem, tag, before_in, rows, cols, live);
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(Bad::Defect(problems.join("; ")))
    }
}

// ------------------------------------------------------------------------------------------------ runs

/// One launch of fused entry `hc8_read_up_e4m3_n{nt}` over `rows` rows. `rows` outside `1..=16*NT` must write nothing.
fn run_fused(
    emu: &Emulator,
    nt: u32,
    c: &Case,
    low: &[f32],
    norm: &[f32],
    rows: usize,
    tag: &str,
) -> Result<Vec<u32>, Bad> {
    assert_eq!(low.len(), rows * c.k);
    assert_eq!(norm.len(), rows * c.m);
    let live = if rows >= 1 && rows <= 16 * nt as usize {
        rows
    } else {
        0
    };
    let mut mem = Memory::default();
    let q = map_guarded(&mut mem, "q", QBASE, c.swz.clone(), WGUARD);
    let sp = map_guarded(&mut mem, "s", SBASE, f32_bytes(&c.sc), WGUARD);
    let xp = map_guarded(&mut mem, "x", XBASE, f32_bytes(low), XGUARD);
    let np = map_guarded(&mut mem, "n", NBASE, f32_bytes(norm), NGUARD);
    let yp = map_guarded(&mut mem, "y", YBASE, sentinel(rows * c.hidden), SENT);
    mem.map(
        "kernarg".into(),
        KERNARG,
        fused_kernarg([q, sp, xp, np, yp], c, rows),
    )
    .unwrap();
    let before_in = input_snapshot(&mem, &["q", "s", "x", "n"]);
    launch_grid(
        emu,
        &mut mem,
        (c.hidden / 16) as u32,
        1,
        READ_UP_BLOCK as u32,
    )
    .map_err(|e| classify(format!("{tag}: emulator: {e}")))?;
    finish_run(mem, tag, &before_in, rows, c.hidden, live)
}

/// The decomposed route's `hc8_gemm_e4m3_mw4_kw1_n{nt}_set` entry: `u [rows][4*hidden]` words.
fn run_set(
    emu: &Emulator,
    nt: u32,
    c: &Case,
    low: &[f32],
    rows: usize,
    tag: &str,
) -> Result<Vec<u32>, Bad> {
    assert_eq!(low.len(), rows * c.k);
    let g = set_geo(nt);
    let mut mem = Memory::default();
    let q = map_guarded(&mut mem, "q", QBASE, c.swz.clone(), WGUARD);
    let sp = map_guarded(&mut mem, "s", SBASE, f32_bytes(&c.sc), WGUARD);
    let xp = map_guarded(&mut mem, "x", XBASE, f32_bytes(low), XGUARD);
    let yp = map_guarded(&mut mem, "y", YBASE, sentinel(rows * c.m), SENT);
    mem.map(
        "kernarg".into(),
        KERNARG,
        gemm_kernarg([q, sp, xp, yp], c, rows),
    )
    .unwrap();
    let before_in = input_snapshot(&mem, &["q", "s", "x"]);
    launch_grid(emu, &mut mem, g.grid(c.m as u32), 1, g.block())
        .map_err(|e| classify(format!("{tag}: set entry emulator: {e}")))?;
    finish_run(mem, tag, &before_in, rows, c.m, rows)
}

/// The independent WMMA reference `hc8_ref_kw1_k{K}_set` (host-decoded row-major BF16 weights): `u` words.
fn run_wmma_ref(
    emu: &Emulator,
    c: &Case,
    low: &[f32],
    rows: usize,
    tag: &str,
) -> Result<Vec<u32>, Bad> {
    assert_eq!(low.len(), rows * c.k);
    assert!((1..=64).contains(&rows));
    let mut mem = Memory::default();
    let q = map_guarded(&mut mem, "q", QBASE, u16_bytes(&c.wbf), WGUARD);
    let sp = map_guarded(&mut mem, "s", SBASE, f32_bytes(&c.sc), WGUARD);
    let xp = map_guarded(&mut mem, "x", XBASE, f32_bytes(low), XGUARD);
    let yp = map_guarded(&mut mem, "y", YBASE, sentinel(rows * c.m), SENT);
    mem.map(
        "kernarg".into(),
        KERNARG,
        gemm_kernarg([q, sp, xp, yp], c, rows),
    )
    .unwrap();
    let before_in = input_snapshot(&mem, &["q", "s", "x"]);
    let (gx, gy) = reference_grid(c.m as u32, rows as u32);
    launch_grid(emu, &mut mem, gx, gy, REFERENCE_BLOCK)
        .map_err(|e| classify(format!("{tag}: WMMA reference emulator: {e}")))?;
    finish_run(mem, tag, &before_in, rows, c.m, rows)
}

/// The projected-read reference (`support/hc8_read_up_reference.rs`) on the GEMM words `up [rows][4*hidden]`.
fn run_projected(
    emu: &Emulator,
    up: &[u32],
    norm: &[f32],
    hidden: usize,
    rows: usize,
    tag: &str,
) -> Result<Vec<u32>, Bad> {
    assert_eq!(up.len(), rows * 4 * hidden);
    assert_eq!(norm.len(), rows * 4 * hidden);
    let mut mem = Memory::default();
    let u = map_guarded(&mut mem, "u", UBASE, u32_bytes(up), WGUARD);
    let np = map_guarded(&mut mem, "n", NBASE, f32_bytes(norm), NGUARD);
    let yp = map_guarded(&mut mem, "y", YBASE, sentinel(rows * hidden), SENT);
    mem.map(
        "kernarg".into(),
        KERNARG,
        projected_kernarg([u, np, yp], hidden, rows),
    )
    .unwrap();
    let before_in = input_snapshot(&mem, &["u", "n"]);
    let (gx, gy) = projected_grid(hidden as u32, rows as u32);
    launch_grid(emu, &mut mem, gx, gy, PROJECTED_BLOCK)
        .map_err(|e| classify(format!("{tag}: projected reference emulator: {e}")))?;
    finish_run(mem, tag, &before_in, rows, hidden, rows)
}

// ------------------------------------------------------------------------------------------------ module / emulators

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

/// The emitted full module (GEMMs, dequants, fused read-ups), lifted once.
fn pm_program() -> &'static Program {
    static PROGRAM: LazyLock<Program> = LazyLock::new(|| {
        let (_, text, _) = emit_module().unwrap_or_else(|e| panic!("emit_module: {e}"));
        lift_text(&text)
    });
    &PROGRAM
}

fn proj_program() -> &'static Program {
    static PROGRAM: LazyLock<Program> = LazyLock::new(|| {
        let (_, text) =
            projected_module().unwrap_or_else(|e| panic!("projected reference module: {e}"));
        lift_text(&text)
    });
    &PROGRAM
}

/// The independent WMMA reference module (`hc8_ref_kw1_k{K}_set` for every `K` in `ks`), lifted. The caller owns it
/// and lends it to [`Env::new`].
fn reference_program(ks: &[usize]) -> Program {
    let keys: Vec<(Geo, u32)> = ks.iter().map(|&k| (set_geo(1), k as u32)).collect();
    let (_, text) = reference_module(&keys).unwrap_or_else(|e| panic!("reference module: {e}"));
    lift_text(&text)
}

/// Every emulator a differential launch needs.
struct Env<'a> {
    fused: BTreeMap<String, Emulator<'a>>,
    set: BTreeMap<String, Emulator<'a>>,
    wref: BTreeMap<String, Emulator<'a>>,
    proj: Emulator<'a>,
}

impl<'a> Env<'a> {
    /// `rprog`: [`reference_program`] of the same `ks` (the WMMA reference has `K` compile-time).
    fn new(rprog: &'a Program, ks: &[usize]) -> Env<'a> {
        let prog: &'a Program = pm_program();
        let mut fused = BTreeMap::new();
        let mut set = BTreeMap::new();
        for nt in 1..=4u32 {
            let sym = read_up_symbol(nt);
            let e = Emulator::new(prog, &sym)
                .unwrap_or_else(|err| panic!("fused {sym}: Emulator::new: {err}"));
            fused.insert(sym, e);
            let sym = set_geo(nt).symbol();
            let e = Emulator::new(prog, &sym)
                .unwrap_or_else(|err| panic!("set {sym}: Emulator::new: {err}"));
            set.insert(sym, e);
        }
        let mut wref = BTreeMap::new();
        for &k in ks {
            let sym = reference_symbol(set_geo(1), k as u32);
            let e = Emulator::new(rprog, &sym)
                .unwrap_or_else(|err| panic!("reference {sym}: Emulator::new: {err}"));
            wref.insert(sym, e);
        }
        let pprog: &'a Program = proj_program();
        let proj = Emulator::new(pprog, PROJECTED_SYMBOL)
            .unwrap_or_else(|err| panic!("projected reference: Emulator::new: {err}"));
        Env {
            fused,
            set,
            wref,
            proj,
        }
    }
    fn fused(&self, nt: u32) -> &Emulator<'a> {
        &self.fused[&read_up_symbol(nt)]
    }
    fn set(&self, nt: u32) -> &Emulator<'a> {
        &self.set[&set_geo(nt).symbol()]
    }
    fn wref(&self, k: usize) -> &Emulator<'a> {
        &self.wref[&reference_symbol(set_geo(1), k as u32)]
    }
}

// ------------------------------------------------------------------------------------------------ tally / compare

#[derive(Default)]
struct Tally {
    launches: u64,
    executed: u64,
    words_vs_decomposed: u64,
    words_vs_gemm_ref: u64,
    words_vs_host: u64,
    host_skipped: u64,
    words_vs_singleton: u64,
    words_vs_entry: u64,
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
    fn defect(&mut self, label: &str, what: &str, b: Bad) {
        match b {
            Bad::Refused(m) => self.refused.push(format!("{label} [{what}]: {m}")),
            Bad::Defect(m) => self.fail(format!("{label} [{what}]: {m}")),
        }
    }
    /// `min_exec_pct`: the share of launches that must execute (the rest are reported interpreter refusals; 100 =
    /// none tolerated). `need_host`: at least one word must have been checked against the host model.
    fn finish(self, name: &str, min_exec_pct: u64, need_host: bool) {
        log(format!(
            "[{name}] launches={} executed={} words_vs_decomposed={} words_vs_gemm_ref={} words_vs_singleton={} words_vs_entry={} words_vs_host={} (skipped {}) refused={} failures={}",
            self.launches,
            self.executed,
            self.words_vs_decomposed,
            self.words_vs_gemm_ref,
            self.words_vs_singleton,
            self.words_vs_entry,
            self.words_vs_host,
            self.host_skipped,
            self.refused.len(),
            self.nfail
        ));
        for r in self.refused.iter().take(6) {
            log(format!("[{name}] REFUSED (interpreter, not a pass): {r}"));
        }
        log(format!(
            "[{name}] semantics: emulator models of v_exp_f32/v_rcp_f32 (candidate), v_div_scale/fmas/fixup (gfx1201 measured), E4M3 NaN decode; fused kernel and decomposed reference share those models -> hardware byte oracle is the receipt"
        ));
        assert!(
            self.nfail == 0,
            "{name}: {} failures:\n{}",
            self.nfail,
            self.fails.join("\n")
        );
        assert!(self.executed > 0, "{name}: nothing executed");
        if need_host {
            assert!(
                self.words_vs_host > 0,
                "{name}: no word was checked against the host model"
            );
        }
        if require_no_refusal() {
            assert!(self.refused.is_empty(), "{name}: interpreter refusals");
        } else if min_exec_pct >= 100 {
            assert!(
                self.refused.is_empty(),
                "{name}: inputs must never be refused: {:?}",
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

fn compare_words(t: &mut Tally, label: &str, what: &str, got: &[u32], want: &[u32], width: usize) {
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
                    "[{}][{}] got {:#010x} want {:#010x}",
                    i / width,
                    i % width,
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

/// Host model vs the kernel words, finite normal domain only (see `host_read_word`).
fn host_check(
    t: &mut Tally,
    label: &str,
    c: &Case,
    up: &[u32],
    norm: &[f32],
    rows: usize,
    got: &[u32],
) {
    let (h, m) = (c.hidden, c.m);
    for token in 0..rows {
        for col in 0..h {
            let mut ups = [0u32; 4];
            let mut ns = [0u32; 4];
            for b in 0..4 {
                ups[b] = up[token * m + b * h + col];
                ns[b] = norm[token * m + b * h + col].to_bits();
            }
            match host_read_word(ups, ns) {
                Some(want) => {
                    t.words_vs_host += 1;
                    let g = got[token * h + col];
                    if g != want {
                        t.fail(format!(
                            "{label} [host model]: mixed[{token}][{col}] got {g:#010x} want {want:#010x} (up {ups:08x?} normalized {ns:08x?})"
                        ));
                        return;
                    }
                }
                None => t.host_skipped += 1,
            }
        }
    }
}

/// Fused launch vs the decomposed pipeline (set GEMM entry -> projected DAG), bit for bit; the independent WMMA
/// reference must agree with the set entry first. Returns the fused words when everything executed.
#[allow(clippy::too_many_arguments)]
fn differential(
    t: &mut Tally,
    label: &str,
    env: &Env,
    c: &Case,
    nt: u32,
    low: &[f32],
    norm: &[f32],
    rows: usize,
) -> Option<Vec<u32>> {
    t.launches += 1;
    let fr = run_fused(env.fused(nt), nt, c, low, norm, rows, label);
    let sr = run_set(env.set(nt), nt, c, low, rows, label);
    let wr = run_wmma_ref(env.wref(c.k), c, low, rows, label);
    let (fused, up, up_ref) = match (fr, sr, wr) {
        (Ok(f), Ok(s), Ok(w)) => (f, s, w),
        (Err(e), _, _) => {
            t.defect(label, "fused kernel", e);
            return None;
        }
        (_, Err(e), _) => {
            t.defect(label, "decomposed set entry", e);
            return None;
        }
        (_, _, Err(e)) => {
            t.defect(label, "independent WMMA reference", e);
            return None;
        }
    };
    t.words_vs_gemm_ref += up.len() as u64;
    compare_words(
        t,
        label,
        "set entry vs independent WMMA reference",
        &up,
        &up_ref,
        c.m,
    );
    let want = match run_projected(&env.proj, &up, norm, c.hidden, rows, label) {
        Ok(w) => w,
        Err(e) => {
            t.defect(label, "projected reference", e);
            return None;
        }
    };
    t.executed += 1;
    t.words_vs_decomposed += fused.len() as u64;
    compare_words(
        t,
        label,
        "fused vs set GEMM + projected DAG",
        &fused,
        &want,
        c.hidden,
    );
    host_check(t, label, c, &up, norm, rows, &fused);
    Some(fused)
}

/// Singleton fused outputs (`rows = 1` on the NT = 1 entry) of every `(low, norm)` pool row.
fn singletons(
    t: &mut Tally,
    name: &str,
    env: &Env,
    c: &Case,
    lows: &[Vec<f32>],
    norms: &[Vec<f32>],
) -> Vec<Option<Vec<u32>>> {
    (0..lows.len())
        .map(|j| {
            t.launches += 1;
            let label = format!("{name} singleton row {j}");
            match run_fused(env.fused(1), 1, c, &lows[j], &norms[j], 1, &label) {
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
        .collect()
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

#[test]
fn read_up_module_abi_resources_and_native_roundtrip() {
    assert_eq!(MODULE, "hc8_wmma_pm_gfx1201");
    assert_eq!(READ_UP_BLOCK, 128, "four waves, one per HC branch");
    assert_eq!(READ_UP_KERNARG_BYTES, 56);
    let want: Vec<String> = (1..=4)
        .map(|nt| format!("hc8_read_up_e4m3_n{nt}"))
        .collect();
    for (i, w) in want.iter().enumerate() {
        assert_eq!(&read_up_symbol(i as u32 + 1), w, "read_up_symbol rule");
    }
    let ups = emit_read_ups().unwrap();
    let ups2 = emit_read_ups().unwrap();
    assert_eq!(ups.len(), 4);
    for (a, b) in ups.iter().zip(&ups2) {
        assert_eq!(a.s_text, b.s_text, "emission must be deterministic");
    }
    let forbidden = [
        "scratch_",
        "buffer_atomic",
        "global_atomic",
        "flat_",
        "v_swmmac",
    ];
    for (i, e) in ups.iter().enumerate() {
        let nt = i as u32 + 1;
        let sym = &want[i];
        let t = &e.s_text;
        assert_eq!(e.proof.arch, ARCH, "{sym}");
        assert_eq!(
            *t,
            emit_read_up(nt).unwrap().s_text,
            "{sym}: emit_read_up == emit_read_ups"
        );
        assert!(t.contains(&format!(".globl {sym}\n")), "{sym}: globl");
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
        assert!(
            t.contains(&format!(".kernarg_segment_size: {READ_UP_KERNARG_BYTES}\n")),
            "{sym}: ABI 56 B"
        );
        assert_eq!(
            field(t, ".offset:"),
            [0, 8, 16, 24, 32, 40, 44, 48, 52],
            "{sym}: kernarg offsets q s low normalized mixed hidden low_rank rows pad"
        );
        assert_eq!(
            field(t, ".size:"),
            [8, 8, 8, 8, 8, 4, 4, 4, 4],
            "{sym}: kernarg sizes"
        );
        assert_eq!(
            field(t, ".max_flat_workgroup_size:"),
            [READ_UP_BLOCK],
            "{sym}: workgroup size"
        );
        assert_eq!(
            field(t, ".group_segment_fixed_size:"),
            [4096 * nt],
            "{sym}: static LDS of the four waves' terms"
        );
        let vg = field(t, ".vgpr_count:");
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
        let has = |p: &str| ms.iter().any(|m| m.starts_with(p));
        for p in [
            "v_wmma_f32_16x16x16_bf16",
            "v_cvt_pk_f32_fp8",
            "v_exp_f32",
            "v_ldexp_f32",
            "v_rcp_f32",
            "v_div_scale_f32",
            "v_div_fmas_f32",
            "v_div_fixup_f32",
            "ds_store_b32",
            "ds_load_b32",
            "s_barrier",
        ] {
            assert!(has(p), "{sym}: missing {p}");
        }
        assert!(
            ms.iter()
                .filter(|m| m.starts_with("v_div_fixup_f32"))
                .count()
                >= 2,
            "{sym}: gate division and value / 4.0 division"
        );
    }
    // Entries 82..86 of the module are exactly these four, after the 80 GEMMs and the 2 dequants.
    let (emitted, text, proof) = emit_module().unwrap();
    assert_eq!(emitted.len(), 86);
    assert_eq!(proof.kernels.len(), 86);
    let syms = symbols();
    assert_eq!(syms.len(), 86);
    assert_eq!(&syms[82..], &want[..]);
    for (i, e) in emitted[82..].iter().enumerate() {
        assert_eq!(e.s_text, ups[i].s_text, "{}: module entry", want[i]);
    }
    let program = lift_text(&text);
    for sym in &want {
        Emulator::new(&program, sym).unwrap_or_else(|e| panic!("{sym}: Emulator::new: {e}"));
    }
    // The projected reference is itself assemblable, liftable and has no entry in common with the product module.
    let (_, rtext) = projected_module().unwrap();
    let rprog = lift_text(&rtext);
    Emulator::new(&rprog, PROJECTED_SYMBOL)
        .unwrap_or_else(|e| panic!("{PROJECTED_SYMBOL}: Emulator::new: {e}"));
    assert!(!syms.iter().any(|s| s == PROJECTED_SYMBOL));
}

// ------------------------------------------------------------------------------------------------ reference sanity

/// The decomposed projected-read reference against the pure-Rust host model on a dense finite sample, and against
/// hand-derived known answers (`up = 0` gives `gate = 0.5`; saturated `up` gives `gate = 1`; `up` far below the
/// `0xc2b17218` clamp gives `gate = +0` through `1 / +Inf`).
#[test]
fn projected_reference_matches_host_model_and_known_answers() {
    let env_proj = Emulator::new(proj_program(), PROJECTED_SYMBOL)
        .unwrap_or_else(|e| panic!("{PROJECTED_SYMBOL}: Emulator::new: {e}"));
    let mut t = Tally::default();
    let hidden = 16usize;
    let rows = 64usize;
    let width = 4 * hidden;
    // Dense sample: logits 2^-12..2^5 either sign with 23-bit mantissas (not BF16-exact), normalized words moderate.
    let mut rng = Rng::new(0x7001);
    let mut up = Vec::with_capacity(rows * width);
    for _ in 0..rows * width {
        let sign = (rng.next() & 1) as u32;
        let mant = (rng.next() as u32) & 0x7f_ffff;
        let e = 115 + rng.below(17) as u32;
        up.push((sign << 31) | (e << 23) | mant);
    }
    let norm_pool = gen_norm_pool(&mut rng, width, rows, &[0, 1, 2]);
    let norm = flat(&norm_pool);
    t.launches += 1;
    match run_projected(&env_proj, &up, &norm, hidden, rows, "projected sample") {
        Ok(y) => {
            t.executed += 1;
            let mut compared = 0u64;
            for token in 0..rows {
                for col in 0..hidden {
                    let mut ups = [0u32; 4];
                    let mut ns = [0u32; 4];
                    for b in 0..4 {
                        ups[b] = up[token * width + b * hidden + col];
                        ns[b] = norm[token * width + b * hidden + col].to_bits();
                    }
                    match host_read_word(ups, ns) {
                        Some(w) => {
                            compared += 1;
                            t.words_vs_host += 1;
                            let g = y[token * hidden + col];
                            if g != w {
                                t.fail(format!(
                                    "projected reference vs host model: mixed[{token}][{col}] got {g:#010x} want {w:#010x} (up {ups:08x?} normalized {ns:08x?})"
                                ));
                            }
                        }
                        None => t.host_skipped += 1,
                    }
                }
            }
            if compared * 10 < (rows * hidden) as u64 * 9 {
                t.fail(format!(
                    "projected sample: only {compared} of {} words inside the host model domain",
                    rows * hidden
                ));
            }
        }
        Err(e) => t.defect("projected sample", "projected reference", e),
    }
    // Known answers on constant planes. normalized = 2.0 everywhere, rows 0..3 = (up, mixed expected).
    let cases: [(f32, u32); 4] = [
        (0.0, 0x3f80_0000),    // gate .5 -> term 1.0 each -> value 4.0 -> 1.0
        (80.0, 0x4000_0000),   // gate 1.0 (exp(-80) vanishes) -> term 2.0 -> value 8.0 -> 2.0
        (-200.0, 0x0000_0000), // exp(+200) clamps to +Inf -> gate 1/Inf = +0 -> +0
        (-0.0, 0x3f80_0000),   // x = -0 behaves as 0
    ];
    let rows_k = cases.len();
    let mut up_k = Vec::new();
    for (u, _) in cases {
        up_k.extend(std::iter::repeat(u.to_bits()).take(width));
    }
    let norm_k: Vec<f32> = vec![2.0; rows_k * width];
    t.launches += 1;
    match run_projected(
        &env_proj,
        &up_k,
        &norm_k,
        hidden,
        rows_k,
        "projected known answers",
    ) {
        Ok(y) => {
            t.executed += 1;
            for (r, (u, want)) in cases.iter().enumerate() {
                for col in 0..hidden {
                    let g = y[r * hidden + col];
                    if g != *want {
                        t.fail(format!(
                            "projected known answer up={u}: mixed[{r}][{col}] got {g:#010x} want {want:#010x}"
                        ));
                        break;
                    }
                }
            }
        }
        Err(e) => t.defect("projected known answers", "projected reference", e),
    }
    // The BF16 round of the host model is the object's: payload NaN OR 0x10000, Inf kept, ties to even.
    for (w, want) in [
        (0x7f80_0001u32, 0x7f81_0000u32),
        (0x7fa0_0000, 0x7fa0_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0xff80_0000),
        (0x3f80_8000, 0x3f80_0000),
        (0x3f81_8000, 0x3f82_0000),
        (0x7f7f_ffff, 0x7f80_0000),
        (0x0000_0001, 0x0000_0000),
    ] {
        let g = bf16_rt_bits(w);
        if g != want {
            t.fail(format!(
                "bf16_rt_bits({w:#010x}) = {g:#010x}, want {want:#010x}"
            ));
        }
    }
    t.finish("projected_reference_sanity", 100, true);
}

// ------------------------------------------------------------------------------------------------ rows x shapes x entries

/// Every entry `n1..n4` on every shape at the spec's row counts and the NT boundaries (capacity padding included,
/// e.g. `rows = 1` on `n4`): fused == decomposed pipeline bit for bit, host model on the finite domain, the set GEMM
/// == the independent WMMA reference, the same rows give the same words on every NT entry that can run them.
#[test]
fn fused_matches_decomposed_pipeline_every_entry_shape_and_row_count() {
    let ks: Vec<usize> = SHAPES.iter().map(|&(_, k)| k).collect();
    let rprog = reference_program(&ks);
    let env = Env::new(&rprog, &ks);
    let mut t = Tally::default();
    for (si, &(hidden, k)) in SHAPES.iter().enumerate() {
        let kind = match si % 3 {
            0 => QKind::Small,
            1 => QKind::Full,
            _ => QKind::Sparse,
        };
        let c = rand_case(0x5100 + si as u64, hidden, k, kind);
        let mut rng = Rng::new(0x5200 + si as u64);
        let lows = gen_low_pool(&mut rng, k, 64, &LOW_KINDS);
        let norms = gen_norm_pool(&mut rng, 4 * hidden, 64, &NORM_KINDS);
        let mut first: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for nt in 1..=4u32 {
            for rows in rows_for(nt as usize) {
                let label = format!("shape h{hidden}k{k} {} rows={rows}", read_up_symbol(nt));
                let low = flat(&lows[..rows]);
                let norm = flat(&norms[..rows]);
                if let Some(y) = differential(&mut t, &label, &env, &c, nt, &low, &norm, rows) {
                    match first.get(&rows) {
                        Some(f) => {
                            t.words_vs_entry += y.len() as u64;
                            compare_words(
                                &mut t,
                                &label,
                                "NT capacity independence vs the first entry that ran these rows",
                                &y,
                                f,
                                hidden,
                            );
                        }
                        None => {
                            first.insert(rows, y);
                        }
                    }
                }
            }
        }
    }
    t.finish("fused_matches_decomposed_pipeline", 90, true);
}

// ------------------------------------------------------------------------------------------------ rows 1..=64 and singleton invariance

/// Every logical row count `1..=64` on its entry `NT = ceil(rows/16)`: fused == decomposed pipeline, every output row
/// bitwise equal to the `rows = 1` launch of the same `(low, normalized)` row; one row replicated at every position and
/// rotated slot orders at the NT boundaries.
#[test]
fn rows_1_to_64_row_invariance() {
    let shapes = [(16usize, 32usize), (32, 64)];
    let ks: Vec<usize> = shapes.iter().map(|&(_, k)| k).collect();
    let rprog = reference_program(&ks);
    let env = Env::new(&rprog, &ks);
    let mut t = Tally::default();
    for (si, &(hidden, k)) in shapes.iter().enumerate() {
        let name = format!("rows_1_to_64 h{hidden}k{k}");
        let c = rand_case(0x6100 + si as u64, hidden, k, QKind::Small);
        let mut rng = Rng::new(0x6200 + si as u64);
        let lows = gen_low_pool(&mut rng, k, 64, &LOW_KINDS);
        let norms = gen_norm_pool(&mut rng, 4 * hidden, 64, &NORM_KINDS);
        let singles = singletons(&mut t, &name, &env, &c, &lows, &norms);
        for rows in 1..=64usize {
            let nt = nt_for(rows);
            let label = format!("{name} rows={rows}");
            let low = flat(&lows[..rows]);
            let norm = flat(&norms[..rows]);
            if let Some(y) = differential(&mut t, &label, &env, &c, nt, &low, &norm, rows) {
                for r in 0..rows {
                    if let Some(s) = &singles[r] {
                        t.words_vs_singleton += hidden as u64;
                        compare_words(
                            &mut t,
                            &label,
                            &format!("row {r} vs rows=1 singleton"),
                            &y[r * hidden..(r + 1) * hidden],
                            s,
                            hidden,
                        );
                    }
                }
            }
        }
        // The same (low, normalized) row at every position.
        for (rows, j) in [
            (2usize, 0usize),
            (16, 3),
            (17, 9),
            (32, 31),
            (33, 40),
            (48, 47),
            (64, 63),
        ] {
            let nt = nt_for(rows);
            let label = format!("{name} replicate row {j} x{rows}");
            let low: Vec<f32> = (0..rows).flat_map(|_| lows[j].iter().copied()).collect();
            let norm: Vec<f32> = (0..rows).flat_map(|_| norms[j].iter().copied()).collect();
            t.launches += 1;
            match run_fused(env.fused(nt), nt, &c, &low, &norm, rows, &label) {
                Ok(y) => {
                    t.executed += 1;
                    if let Some(s) = &singles[j] {
                        for r in 0..rows {
                            t.words_vs_singleton += hidden as u64;
                            compare_words(
                                &mut t,
                                &label,
                                &format!("position {r}"),
                                &y[r * hidden..(r + 1) * hidden],
                                s,
                                hidden,
                            );
                        }
                    }
                }
                Err(e) => t.defect(&label, "fused kernel", e),
            }
        }
        // Rotated slot order at the NT boundaries.
        for (rows, rot) in [(17usize, 5usize), (48, 17), (64, 33)] {
            let nt = nt_for(rows);
            let label = format!("{name} rotated by {rot} rows={rows}");
            let order: Vec<usize> = (0..rows).map(|i| (i + rot) % 64).collect();
            let low: Vec<f32> = order
                .iter()
                .flat_map(|&j| lows[j].iter().copied())
                .collect();
            let norm: Vec<f32> = order
                .iter()
                .flat_map(|&j| norms[j].iter().copied())
                .collect();
            t.launches += 1;
            match run_fused(env.fused(nt), nt, &c, &low, &norm, rows, &label) {
                Ok(y) => {
                    t.executed += 1;
                    for (r, &j) in order.iter().enumerate() {
                        if let Some(s) = &singles[j] {
                            t.words_vs_singleton += hidden as u64;
                            compare_words(
                                &mut t,
                                &label,
                                &format!("slot {r} = pool row {j}"),
                                &y[r * hidden..(r + 1) * hidden],
                                s,
                                hidden,
                            );
                        }
                    }
                }
                Err(e) => t.defect(&label, "fused kernel", e),
            }
        }
    }
    t.finish("rows_1_to_64_row_invariance", 90, true);
}

// ------------------------------------------------------------------------------------------------ known answers

/// All-zero E4M3 weights: `u = +-0`, `gate = 0.5`, `term = 0.5 * n_b`, so `mixed = (n0 + n1 + n2 + n3) / 8` exactly
/// for small-integer normalized words. No DAG, no reference kernel: host integer arithmetic only. Rows 1, 3, 8, 16, 17,
/// 48, 64 on their entries, with scales of both signs so the zero logit carries both signs.
#[test]
fn zero_weights_give_the_hand_computed_mixed_value() {
    let rprog = reference_program(&[32]);
    let env = Env::new(&rprog, &[32]);
    let mut t = Tally::default();
    let (hidden, k) = (32usize, 32usize);
    let m = 4 * hidden;
    let mut rng = Rng::new(0x7100);
    let raw = vec![0u8; m * k];
    let sc: Vec<f32> = (0..m)
        .map(|i| if i % 3 == 0 { -0.5 } else { 0.25 })
        .collect();
    let c = make_case(hidden, k, raw, sc);
    for rows in [1usize, 3, 8, 16, 17, 48, 64] {
        let nt = nt_for(rows);
        let label = format!("zero weights rows={rows}");
        let lows = gen_low_pool(&mut rng, k, rows, &[0, 1, 4]);
        let low = flat(&lows);
        let ints: Vec<u32> = (0..rows * m).map(|_| 1 + rng.below(8) as u32).collect();
        let norm: Vec<f32> = ints.iter().map(|&n| n as f32).collect();
        t.launches += 1;
        match run_fused(env.fused(nt), nt, &c, &low, &norm, rows, &label) {
            Ok(y) => {
                t.executed += 1;
                let mut want = Vec::with_capacity(rows * hidden);
                for token in 0..rows {
                    for h in 0..hidden {
                        let sum: u32 = (0..4).map(|b| ints[token * m + b * hidden + h]).sum();
                        want.push((sum as f32 / 8.0).to_bits());
                    }
                }
                t.words_vs_decomposed += y.len() as u64;
                compare_words(
                    &mut t,
                    &label,
                    "hand-computed (n0+n1+n2+n3)/8",
                    &y,
                    &want,
                    hidden,
                );
            }
            Err(e) => t.defect(&label, "fused kernel", e),
        }
    }
    t.finish("zero_weights_known_answer", 100, false);
}

// ------------------------------------------------------------------------------------------------ exceptional inputs

const EDGE_N: [u32; 20] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x0000_0001, // smallest subnormal (BF16 round to +0)
    0x007f_ffff, // largest subnormal (BF16 round to min normal)
    0x0080_0000, // min normal
    0x7f7f_ffff, // max finite (BF16 round to +Inf)
    0xff7f_ffff, // -max finite
    0x7f80_0000, // +Inf
    0xff80_0000, // -Inf
    0x7fc0_0000, // qNaN
    0x7fc1_2345, // qNaN with payload (low16 != 0)
    0x7f80_0001, // sNaN, low16 != 0
    0x7fa0_0000, // sNaN with low16 == 0
    0xffc1_0000, // negative qNaN
    0x7f80_ffff, // sNaN, all low bits
    0x4000_0000, // 2.0
    0x3f7f_ffff, // just below 1.0 (BF16 round to 1.0)
    0x4b80_0000, // 2^24
];

const NAN_WEIGHT_CODES: [u8; 2] = [0x7f, 0xff];

/// NaN / Inf / signed-zero / extreme inputs through every entry: E4M3 NaN weight codes, special scales (Inf, NaN with
/// payload, signed zeros, subnormal, max finite), `low` rows with qNaN / sNaN / negative NaN / +-Inf / signed zeros /
/// max finite / subnormals, `normalized` words from every BF16-round class (payload NaN, Inf, subnormal, ties).
/// Fused == decomposed pipeline bit for bit (NaN payloads, signs, signed zeros included); NaN `low` rows poison their
/// whole output row; NaN weight rows poison their hidden column for every token; clean rows stay NaN-free in clean
/// columns; singleton invariance holds on every exceptional row. Interpreter refusals are counted, not passed.
#[test]
fn exceptional_inputs() {
    let (hidden, k) = (32usize, 64usize);
    let m = 4 * hidden;
    let rprog = reference_program(&[k]);
    let env = Env::new(&rprog, &[k]);
    let mut t = Tally::default();
    let mut rng = Rng::new(0x8000);
    let mut raw = gen_q(&mut rng, m * k, QKind::Full);
    // NaN E4M3 weight codes in (branch 0, h 2) and (branch 3, h 13).
    let mut nan_cols: BTreeSet<usize> = BTreeSet::new();
    for (i, (b, h)) in [(0usize, 2usize), (3, 13)].into_iter().enumerate() {
        raw[(b * hidden + h) * k + (7 * b + 1) % k] = NAN_WEIGHT_CODES[i];
        nan_cols.insert(h);
    }
    let mut sc = gen_scale(&mut rng, m);
    // Special scales: columns 5, 9 (Inf / NaN: poison), 21 (signed zero / subnormal), 27 (max finite: overflows to Inf).
    sc[hidden + 5] = f32::INFINITY;
    sc[5] = f32::NEG_INFINITY;
    sc[2 * hidden + 9] = f32::from_bits(0x7fc0_1234);
    sc[21] = -0.0;
    sc[3 * hidden + 21] = f32::from_bits(0x0000_0001);
    sc[hidden + 27] = f32::from_bits(0x7f7f_ffff);
    sc[2 * hidden + 27] = 0.0;
    for h in [5usize, 9, 27] {
        nan_cols.insert(h);
    }
    let c = make_case(hidden, k, raw, sc);
    // `low` rows.
    let special = |at: usize, v: f32, seed: u64| {
        let mut r = gen_x_row(&mut Rng::new(seed), k, 0);
        r[at % k] = v;
        r
    };
    let mut low_pool: Vec<Vec<f32>> = vec![
        special(1, f32::from_bits(0x7fc0_0000), 1),
        special(2, f32::from_bits(0x7fa0_0000), 2),
        special(3, f32::from_bits(0xffc1_0000), 3),
        special(4, f32::INFINITY, 4),
        special(5, f32::NEG_INFINITY, 5),
        vec![f32::from_bits(0x8000_0000); k],
        vec![0.0; k],
        gen_x_row(&mut rng, k, 3),
        gen_x_row(&mut rng, k, 5),
        (0..k)
            .map(|i| f32::from_bits(0x7f7f_0000 | if i % 2 == 0 { 0 } else { 0x8000_0000 }))
            .collect(),
    ];
    let clean_low: Vec<Vec<f32>> = vec![
        gen_x_row(&mut rng, k, 0),
        gen_x_row(&mut rng, k, 1),
        gen_x_row(&mut rng, k, 4),
    ];
    // `normalized` rows: the edge words rotated, random wide rows; and clean moderate rows.
    let mut norm_pool: Vec<Vec<f32>> = (0..4)
        .map(|j| {
            (0..m)
                .map(|i| f32::from_bits(EDGE_N[(i + 5 * j) % EDGE_N.len()]))
                .collect()
        })
        .collect();
    norm_pool.extend(gen_norm_pool(&mut rng, m, 4, &[3, 0]));
    let clean_norm: Vec<Vec<f32>> = (0..3).map(|_| gen_norm_row(&mut rng, m, 1)).collect();
    low_pool.truncate(10);
    // Token `r`: clean when `r % 13 >= 10`, else exceptional.
    let pick = |r: usize| -> (Vec<f32>, Vec<f32>, bool) {
        if r % 13 >= 10 {
            let i = r % 13 - 10;
            (clean_low[i].clone(), clean_norm[i].clone(), true)
        } else {
            (
                low_pool[r % low_pool.len()].clone(),
                norm_pool[r % norm_pool.len()].clone(),
                false,
            )
        }
    };
    for nt in 1..=4u32 {
        let cap = 16 * nt as usize;
        let mut rows_list = vec![cap, 16 * (nt as usize - 1) + 1];
        rows_list.dedup();
        for rows in rows_list {
            let label = format!("exceptional {} rows={rows}", read_up_symbol(nt));
            let toks: Vec<(Vec<f32>, Vec<f32>, bool)> = (0..rows).map(&pick).collect();
            let low: Vec<f32> = toks
                .iter()
                .flat_map(|(l, _, _)| l.iter().copied())
                .collect();
            let norm: Vec<f32> = toks
                .iter()
                .flat_map(|(_, n, _)| n.iter().copied())
                .collect();
            let Some(y) = differential(&mut t, &label, &env, &c, nt, &low, &norm, rows) else {
                continue;
            };
            for (r, (l, _, clean)) in toks.iter().enumerate() {
                let out = &y[r * hidden..(r + 1) * hidden];
                if l.iter().any(|v| v.is_nan()) && !out.iter().all(|&w| f32::from_bits(w).is_nan())
                {
                    t.fail(format!(
                        "{label}: token {r} has a NaN activation but a non-NaN output word"
                    ));
                }
                for &h in &nan_cols {
                    if h == 5 || h == 9 || h == 27 {
                        continue; // scale-driven: only the NaN weight columns are asserted below
                    }
                    if !f32::from_bits(out[h]).is_nan() {
                        t.fail(format!(
                            "{label}: NaN E4M3 weight in column {h} did not give NaN at mixed[{r}][{h}]"
                        ));
                    }
                }
                if *clean {
                    for (h, &w) in out.iter().enumerate() {
                        if !nan_cols.contains(&h) && f32::from_bits(w).is_nan() {
                            t.fail(format!(
                                "{label}: clean token {r} column {h} is NaN ({w:#010x}) with no NaN source"
                            ));
                        }
                    }
                }
            }
            // Singleton invariance on the first exceptional rows.
            for r in 0..rows.min(14) {
                let sl = label.clone() + &format!(" singleton {r}");
                t.launches += 1;
                match run_fused(env.fused(1), 1, &c, &toks[r].0, &toks[r].1, 1, &sl) {
                    Ok(s) => {
                        t.executed += 1;
                        t.words_vs_singleton += hidden as u64;
                        compare_words(
                            &mut t,
                            &label,
                            &format!("row {r} vs rows=1 singleton"),
                            &y[r * hidden..(r + 1) * hidden],
                            &s,
                            hidden,
                        );
                    }
                    Err(e) => t.defect(&sl, "singleton", e),
                }
            }
        }
    }
    t.finish("exceptional_inputs", 10, false);
}

// ------------------------------------------------------------------------------------------------ invalid row counts

/// `rows == 0` and `rows > 16*NT` (including 65) write nothing and terminate in every entry (sentinel `mixed`).
#[test]
fn invalid_row_counts_write_nothing_in_every_entry() {
    let rprog = reference_program(&[32]);
    let env = Env::new(&rprog, &[32]);
    let mut t = Tally::default();
    let (hidden, k) = (16usize, 32usize);
    let c = rand_case(0x9000, hidden, k, QKind::Small);
    let mut rng = Rng::new(0x9100);
    for nt in 1..=4u32 {
        let cap = 16 * nt as usize;
        for rows in [0usize, cap + 1, 65] {
            if rows != 0 && rows <= cap {
                continue;
            }
            let lows = gen_low_pool(&mut rng, k, rows, &[0, 1, 4]);
            let norms = gen_norm_pool(&mut rng, 4 * hidden, rows, &[0, 1]);
            let label = format!("invalid rows {rows} {}", read_up_symbol(nt));
            t.launches += 1;
            match run_fused(
                env.fused(nt),
                nt,
                &c,
                &flat(&lows),
                &flat(&norms),
                rows,
                &label,
            ) {
                Ok(_) => t.executed += 1,
                Err(e) => t.defect(&label, "fused kernel", e),
            }
        }
    }
    t.finish("invalid_row_counts", 100, false);
}
