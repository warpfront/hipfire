// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C for the gfx942 QSA FP8 K/V row codec (`kernels::cdna::qsa_fp8_kv`):
//! the PeaceMaker encode and decode kernels are built with the checked wave64
//! builder, linked into a native code object, loaded by the CPU wave64
//! simulator (`peacemaker_cdna::sim`) and run over Hkv ∈ {2, 4}, D256 rows.
//! What they store/produce must equal, byte for byte, a CPU reference written
//! in this file from the format definition (route `R4-fp8-kv.md` E1, source
//! `kernels/src/tensor_ops.hip:3091-3154`) and the reference shipped with the
//! kernel module (`reference_encode` / `reference_decode`).
//!
//! Format: a row is `Hkv*256` OCP E4M3FN code bytes (head-major) then `Hkv`
//! little-endian F16 scales. `s16 = amax==0 ? 1 : F16(amax/448)`, a positive
//! rounded-to-zero scale becomes bits `0x0001`, and one `if` (not a loop) bumps
//! the F16 bits by one ulp (cap `0x7bff`) when `448*f32(s16) < amax`. The code
//! is the OCP RNE byte of `value / f32(s16)` (IEEE binary32 division). The
//! gfx942 hardware conversion is FNUZ, so the kernel encodes `z*0.5` through
//! it; the only byte the OCP grid has and the kernel cannot produce is `0x80`
//! (OCP -0), canonicalised to `0x00`; this reference does the same.
//!
//! This reference shares nothing with the kernel module or with
//! `peacemaker_cdna::float`: the OCP rounding is an exact nearest-value search
//! over the 127 finite non-negative codes in `f64` (every operand and midpoint
//! is exactly representable), the F16 rounding is the same search over the 31744
//! finite non-negative binary16 values.
//!
//! The HIP twin (`kernels/src/pm_cdna/qsa_fp8_kv.gfx942.hip`) is compiled
//! offline for gfx942 with ROCm (oracle-gated like every PeaceMaker oracle
//! test: `HIPFIRE_TEST_REQUIRE_ROCM=0` opts out) and its machine code is
//! checked for the IEEE division sequence, the FNUZ-half fold and the decode.
//! It is not executed: no GPU.
use peacemaker_cdna::oracle;
use peacemaker_cdna::sim::{self, Config, KernelImage, Launch, Memory};
use peacemaker_isa::cdna::emit::{self, CodeObjectVersion};
use peacemaker_isa::cdna::Kernel;
use peacemaker_isa::kernels::cdna::qsa_fp8_kv as q;
use peacemaker_isa::Arch;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const D: usize = 256;
const HEADS: [usize; 2] = [2, 4];

// ---------------------------------------------------------------------------
// Independent CPU reference

/// Finite OCP E4M3FN value of `code` (not `0x7f`/`0xff`).
fn ocp_value(code: u8) -> f64 {
    assert!(code & 0x7f != 0x7f, "OCP NaN code {code:#x}");
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((code >> 3) & 15);
    let m = f64::from(code & 7);
    let mag = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    sign * mag
}

/// Magnitudes of the finite non-negative codes `0x00..=0x7e`, ascending.
fn ocp_mags() -> Vec<f64> {
    let mags: Vec<f64> = (0u8..=0x7e).map(ocp_value).collect();
    assert!(mags.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(mags[126], 448.0);
    mags
}

/// OCP E4M3FN byte of `z`, round to nearest even, canonical zero (never 0x80).
fn ocp_code(mags: &[f64], z: f32) -> u8 {
    assert!(z.is_finite(), "non-finite quotient {z}");
    let a = f64::from(z).abs();
    assert!(a <= 448.0, "quotient {z} outside the codec's domain");
    let mut code = 126usize;
    for c in 0..126 {
        let mid = (mags[c] + mags[c + 1]) / 2.0;
        if a < mid {
            code = c;
            break;
        }
        if a == mid {
            code = if c % 2 == 0 { c } else { c + 1 };
            break;
        }
    }
    if code != 0 && z.is_sign_negative() {
        code as u8 | 0x80
    } else {
        code as u8
    }
}

/// Finite binary16 value of `bits` (`0..=0x7bff`).
fn f16_value(bits: u16) -> f64 {
    assert!(bits <= 0x7bff);
    let e = i32::from(bits >> 10);
    let m = f64::from(bits & 0x3ff);
    if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

/// binary16 of non-negative finite `x`, round to nearest even (overflow → 0x7c00).
fn f32_to_f16_rne(x: f32) -> u16 {
    assert!(x.is_finite() && x.is_sign_positive() || x == 0.0);
    let a = f64::from(x);
    // Largest c in 0..=0x7bff with value(c) <= a.
    let (mut lo, mut hi) = (0u16, 0x7bffu16);
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if f16_value(mid) <= a {
            lo = mid
        } else {
            hi = mid - 1
        }
    }
    let c = lo;
    if c == 0x7bff {
        return if a >= 65520.0 { 0x7c00 } else { 0x7bff };
    }
    let mid = (f16_value(c) + f16_value(c + 1)) / 2.0;
    if a < mid {
        c
    } else if a > mid {
        c + 1
    } else if c % 2 == 0 {
        c
    } else {
        c + 1
    }
}

/// `(scale bits, bumped)` for a head with absolute maximum `amax`.
fn scale_bits(amax: f32) -> (u16, bool) {
    assert!(
        amax.is_finite() && amax >= 0.0 && amax <= 448.0 * 65504.0,
        "amax {amax} outside the codec's domain"
    );
    let mut s = if amax == 0.0 {
        0x3c00
    } else {
        f32_to_f16_rne(amax / 448.0)
    };
    if amax > 0.0 && s == 0 {
        s = 1
    }
    let sf = f16_value(s) as f32;
    let mut bumped = false;
    if amax > 0.0 && 448.0f32 * sf < amax {
        bumped = true;
        s = if s >= 0x7bff { 0x7bff } else { s + 1 };
    }
    (s, bumped)
}

/// The stored bytes of `rows` rows of `heads*256` f32 (and how many heads bumped).
fn encode_ref(heads: usize, src: &[f32]) -> (Vec<u8>, usize) {
    let mags = ocp_mags();
    let row_len = heads * D;
    assert_eq!(src.len() % row_len, 0);
    let mut out = Vec::new();
    let mut bumps = 0;
    for row in src.chunks(row_len) {
        let mut scales = Vec::new();
        for head in row.chunks(D) {
            let amax = head.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (s, bumped) = scale_bits(amax);
            bumps += usize::from(bumped);
            let sf = f16_value(s) as f32;
            out.extend(head.iter().map(|&v| ocp_code(&mags, v / sf)));
            scales.push(s);
        }
        for s in scales {
            out.extend(s.to_le_bytes())
        }
    }
    (out, bumps)
}

/// The f32 rows a stored buffer decodes to: `ocp_value(code) * f32(scale)`.
fn decode_ref(heads: usize, stored: &[u8]) -> Vec<f32> {
    let row_bytes = heads * D + 2 * heads;
    assert_eq!(stored.len() % row_bytes, 0);
    let mut out = Vec::new();
    for row in stored.chunks(row_bytes) {
        for h in 0..heads {
            let bits = u16::from_le_bytes([row[heads * D + 2 * h], row[heads * D + 2 * h + 1]]);
            let sf = f16_value(bits) as f32;
            out.extend(
                row[h * D..(h + 1) * D]
                    .iter()
                    .map(|&c| ocp_value(c) as f32 * sf),
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Inputs

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in (-1, 1).
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
    /// Roughly normal, std ~1.15.
    fn normal(&mut self) -> f32 {
        self.signed() + self.signed() + self.signed() + self.signed()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Case {
    name: String,
    heads: usize,
    src: Vec<f32>,
}

impl Case {
    fn new(name: &str, heads: usize, rows: usize) -> Self {
        Case {
            name: name.into(),
            heads,
            src: vec![0.0; rows * heads * D],
        }
    }
    fn rows(&self) -> usize {
        self.src.len() / (self.heads * D)
    }
    fn head_mut(&mut self, row: usize, head: usize) -> &mut [f32] {
        let at = (row * self.heads + head) * D;
        &mut self.src[at..at + D]
    }
}

fn next_up(x: f32) -> f32 {
    f32::from_bits(x.to_bits() + 1)
}
fn next_down(x: f32) -> f32 {
    f32::from_bits(x.to_bits() - 1)
}

fn random_case(heads: usize, rng: &mut Rng) -> Case {
    const MAGS: [f32; 8] = [1e-7, 1e-3, 0.02, 1.0, 37.0, 4096.0, 2e5, 3e5];
    let mut c = Case::new("random", heads, 6);
    for row in 0..c.rows() {
        for head in 0..heads {
            let mag = MAGS[rng.below(MAGS.len())];
            let exp = 2f32.powi(rng.below(7) as i32 - 3);
            let h = c.head_mut(row, head);
            for v in h.iter_mut() {
                *v = mag * rng.normal() * if rng.below(5) == 0 { exp } else { 1.0 }
            }
        }
    }
    c
}

fn zero_case(heads: usize, rng: &mut Rng) -> Case {
    let mut c = Case::new("zero", heads, 4);
    // row 0: +0 everywhere. row 1: head 0 zero, the others random. row 2: -0 everywhere.
    for head in 1..heads {
        for v in c.head_mut(1, head) {
            *v = rng.signed()
        }
    }
    for head in 0..heads {
        for v in c.head_mut(2, head) {
            *v = -0.0
        }
    }
    // row 3: alternating signed zeros with one tiny nonzero in the last head.
    for head in 0..heads {
        for (i, v) in c.head_mut(3, head).iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.0 } else { -0.0 }
        }
    }
    let last = heads - 1;
    c.head_mut(3, last)[D - 1] = 1e-30;
    c
}

fn tiny_case(heads: usize, rng: &mut Rng) -> Case {
    let mut c = Case::new("tiny", heads, 5);
    for head in 0..heads {
        // amax ~1e-9: amax/448 rounds to zero in F16, the scale becomes bits 0x0001.
        for v in c.head_mut(0, head) {
            *v = 1e-9 * rng.signed()
        }
        // amax/448 = 2^-26 exactly: still below the F16 subnormal midpoint 2^-25.
        let h = c.head_mut(1, head);
        for v in h.iter_mut() {
            *v = 448.0 * 2f32.powi(-26) * rng.signed()
        }
        h[7] = -448.0 * 2f32.powi(-26);
        // F16 subnormal scales.
        let h = c.head_mut(2, head);
        for v in h.iter_mut() {
            *v = 448.0 * 2f32.powi(-20) * 1.3 * rng.signed()
        }
        // f32 subnormals under a large amax: quotients far below the smallest code.
        let h = c.head_mut(3, head);
        for (i, v) in h.iter_mut().enumerate() {
            *v = f32::from_bits(1 + (i as u32) * 977) * if i % 2 == 0 { 1.0 } else { -1.0 }
        }
        h[0] = 448.0;
        // Mixed: smallest f16-subnormal-scale quotients.
        let h = c.head_mut(4, head);
        for v in h.iter_mut() {
            *v = 2f32.powi(-24) * 448.0 * rng.signed() * rng.unit()
        }
    }
    c
}

/// Rows whose head scales sit on / next to F16 grid points so that the one-ulp
/// bump fires (`next_up(448*g)`), does not (`448*g`, `prev(448*g)`).
fn bump_case(heads: usize, rng: &mut Rng) -> (Case, usize) {
    let mut gs: Vec<u16> = vec![
        0x0001, 0x0002, 0x0003, 0x03ff, 0x0400, 0x0401, 0x1234, 0x3bff, 0x3c00, 0x3c01, 0x5555,
        0x7bfd, 0x7bfe, 0x7bff,
    ];
    for _ in 0..40 {
        gs.push(1 + rng.below(0x7bfd) as u16)
    }
    let mut amaxes = Vec::new(); // (amax, expect bump)
    for &g in &gs {
        let exact = 448.0f32 * f16_value(g) as f32;
        amaxes.push((exact, false));
        amaxes.push((next_down(exact), false));
        if g < 0x7bff {
            amaxes.push((next_up(exact), true))
        }
    }
    let rows = amaxes.len().div_ceil(heads);
    let mut c = Case::new("bump", heads, rows);
    let mut bumps = 0;
    for (i, &(amax, bump)) in amaxes.iter().enumerate() {
        bumps += usize::from(bump);
        let h = c.head_mut(i / heads, i % heads);
        for v in h.iter_mut() {
            *v = amax * rng.signed()
        }
        h[(i * 37) % D] = if i % 2 == 0 { amax } else { -amax };
        h[(i * 37 + 101) % D] = amax / 2.0;
    }
    (c, bumps)
}

/// Heads whose largest value is exactly ±448*s, with every adjacent-code RNE
/// midpoint (both signs) and both zeros: ties, subnormals and extremes.
fn extremes_case(heads: usize, rng: &mut Rng) -> Case {
    const SCALES: [u16; 8] = [
        0x3c00, 0x3800, 0x0400, 0x4200, 0x7bff, 0x0001, 0x2e66, 0x5a3c,
    ];
    let mags = ocp_mags();
    let mut c = Case::new("pm448", heads, SCALES.len());
    for row in 0..SCALES.len() {
        for head in 0..heads {
            let g = f16_value(SCALES[(row + head) % SCALES.len()]) as f32;
            let mut vals: Vec<f32> = vec![448.0 * g, -448.0 * g, 0.0, -0.0];
            for w in mags.windows(2) {
                let mid = ((w[0] + w[1]) / 2.0) as f32 * g;
                vals.push(mid);
                vals.push(-mid);
            }
            assert_eq!(vals.len(), D);
            for i in (1..D).rev() {
                vals.swap(i, rng.below(i + 1))
            } // random channel placement
            c.head_mut(row, head).copy_from_slice(&vals);
        }
    }
    c
}

/// Negative zero and negative values that round to zero or sit on a tie next to it.
fn neg_zero_case(heads: usize) -> Case {
    let mut c = Case::new("neg0", heads, 2);
    let half_sub = 2f32.powi(-10); // half the smallest OCP subnormal: tie 0|1 → 0
    let vals = [
        -0.0,
        0.0,
        -1e-30,
        1e-30,
        -half_sub,
        half_sub,
        -3.0 * half_sub,
        3.0 * half_sub, // 3/2 subnormals: tie 1|2 → 2
        -next_up(half_sub),
        next_up(half_sub),
        -next_down(half_sub),
        next_down(half_sub),
        -2f32.powi(-9),
        2f32.powi(-9),
        -f32::from_bits(1),
        f32::from_bits(1),
    ];
    for row in 0..2 {
        for head in 0..heads {
            let h = c.head_mut(row, head);
            for (i, v) in h.iter_mut().enumerate() {
                *v = vals[(i + row + head) % vals.len()]
            }
            // amax = 448 → scale exactly 1, so the values above are the quotients.
            h[(head * 61 + row * 17) % D] = if row == 0 { 448.0 } else { -448.0 };
        }
    }
    c
}

/// One large value sweeping through every channel (hence every logical 32-lane
/// group and physical wave) of every head: the max reduction must see it.
fn sweep_case(heads: usize, rng: &mut Rng) -> Case {
    let mut c = Case::new("sweep", heads, D);
    for row in 0..D {
        for head in 0..heads {
            let h = c.head_mut(row, head);
            for v in h.iter_mut() {
                *v = 0.003 * rng.normal()
            }
        }
        let hot = row % heads;
        c.head_mut(row, hot)[row] =
            if row % 3 == 0 { -100.0 } else { 100.0 } * (1.0 + 0.01 * (row % 7) as f32);
    }
    c
}

fn encode_cases(heads: usize) -> (Vec<Case>, usize) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ heads as u64);
    let random = random_case(heads, &mut rng);
    let zero = zero_case(heads, &mut rng);
    let tiny = tiny_case(heads, &mut rng);
    let (bump, expected_bumps) = bump_case(heads, &mut rng);
    let extremes = extremes_case(heads, &mut rng);
    let neg0 = neg_zero_case(heads);
    let sweep = sweep_case(heads, &mut rng);
    (
        vec![random, zero, tiny, bump, extremes, neg0, sweep],
        expected_bumps,
    )
}

// ---------------------------------------------------------------------------
// Simulator harness

const KERNARG: u64 = 0x1000;
const DST: u64 = 0x2000_0000;
const SRC: u64 = 0x4000_0000;
const POISON: u8 = 0xee;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn image(kernel: &Kernel, symbol: &str) -> KernelImage {
    assert_eq!(kernel.name, symbol);
    assert_eq!(kernel.kernargs.len(), 2, "kernargs are (dst, src)");
    assert_eq!(
        (
            kernel.kernargs[0].offset,
            kernel.kernargs[0].size,
            kernel.kernargs[1].offset,
            kernel.kernargs[1].size
        ),
        (0, 8, 8, 8)
    );
    let elf = emit::code_object(&[kernel], CodeObjectVersion::V6)
        .unwrap_or_else(|e| panic!("{symbol}: code object: {e}"));
    let image =
        sim::load_code_object(&elf, symbol).unwrap_or_else(|e| panic!("{symbol}: load: {e}"));
    assert_eq!(image.arch, peacemaker_cdna::Arch::Gfx942);
    assert!(
        image.words.starts_with(&kernel.words),
        "{symbol}: the loaded code is the built code"
    );
    image
}

/// Run one dispatch (`grid = (heads, rows)`, block 256) with `src` mapped read-only and
/// a poisoned `dst_len`-byte destination; returns the destination bytes.
fn run(image: &KernelImage, heads: usize, rows: usize, src: Vec<u8>, dst_len: usize) -> Vec<u8> {
    let mut mem = Memory::default();
    mem.map("dst", DST, vec![POISON; dst_len]).unwrap();
    mem.map_readonly("src", SRC, src).unwrap();
    let mut kernarg = Vec::new();
    kernarg.extend(DST.to_le_bytes());
    kernarg.extend(SRC.to_le_bytes());
    let launch = Launch {
        grid: [heads as u32, rows as u32, 1],
        block: [256, 1, 1],
        kernarg,
        kernarg_addr: KERNARG,
        dynamic_lds: 0,
    };
    let stats = sim::run(image, &launch, &mut mem, &Config::default())
        .unwrap_or_else(|e| panic!("{}: {e}", image.name));
    assert_eq!(
        stats.workgroups,
        (heads * rows) as u64,
        "{}: workgroups",
        image.name
    );
    assert_eq!(
        stats.waves,
        (heads * rows * 4) as u64,
        "{}: waves (block 256 = 4 wave64)",
        image.name
    );
    mem.region("dst").unwrap().bytes.clone()
}

fn first_byte_mismatch(heads: usize, got: &[u8], want: &[u8]) -> Option<String> {
    assert_eq!(got.len(), want.len());
    let row_bytes = heads * D + 2 * heads;
    let i = (0..got.len()).find(|&i| got[i] != want[i])?;
    let (row, at) = (i / row_bytes, i % row_bytes);
    let what = if at < heads * D {
        format!("head {} channel {} code", at / D, at % D)
    } else {
        format!("scale byte {}", at - heads * D)
    };
    Some(format!(
        "row {row} {what}: got {:#04x}, want {:#04x}",
        got[i], want[i]
    ))
}

// ---------------------------------------------------------------------------
// Tests

#[test]
fn reference_is_the_format_definition() {
    let mags = ocp_mags();
    // A few hand-derived points independent of the search code.
    assert_eq!(ocp_code(&mags, 448.0), 0x7e);
    assert_eq!(ocp_code(&mags, -448.0), 0xfe);
    assert_eq!(ocp_code(&mags, 1.0), 0x38);
    assert_eq!(ocp_code(&mags, 2f32.powi(-9)), 0x01);
    assert_eq!(ocp_code(&mags, -(2f32.powi(-9))), 0x81);
    assert_eq!(
        ocp_code(&mags, 2f32.powi(-10)),
        0x00,
        "tie 0|1 rounds to the even code"
    );
    assert_eq!(
        ocp_code(&mags, -(2f32.powi(-10))),
        0x00,
        "negative underflow is canonical zero, never 0x80"
    );
    assert_eq!(ocp_code(&mags, -0.0), 0x00);
    assert_eq!(
        ocp_code(&mags, 3.0 * 2f32.powi(-10)),
        0x02,
        "tie 1|2 rounds to the even code"
    );
    assert_eq!(
        ocp_code(&mags, 432.0),
        0x7e,
        "tie 416|448 rounds to the even code"
    );
    assert_eq!(
        ocp_code(&mags, 0.0625 * 1.0625),
        0x18,
        "tie 1.0|1.0625 (m=0|1) rounds to the even mantissa"
    );
    assert_eq!(f32_to_f16_rne(1.0), 0x3c00);
    assert_eq!(f32_to_f16_rne(2f32.powi(-24)), 0x0001);
    assert_eq!(
        f32_to_f16_rne(2f32.powi(-25)),
        0x0000,
        "tie 0|1 rounds to even"
    );
    assert_eq!(f32_to_f16_rne(65504.0), 0x7bff);
    assert_eq!(f32_to_f16_rne(65520.0), 0x7c00);
    assert_eq!(scale_bits(0.0), (0x3c00, false));
    assert_eq!(scale_bits(1e-9), (0x0001, false));
    assert_eq!(scale_bits(448.0), (0x3c00, false));
    assert_eq!(scale_bits(next_up(448.0)), (0x3c01, true));
}

#[test]
fn reference_matches_the_module_reference() {
    // The module's CPU reference and this file's independent one agree on every case.
    let mut rng = Rng(7);
    for heads in HEADS {
        let (cases, expected_bumps) = encode_cases(heads);
        for c in &cases {
            let (want, bumps) = encode_ref(heads, &c.src);
            let got = q::reference_encode(heads, &c.src)
                .unwrap_or_else(|e| panic!("H{heads} {}: reference_encode: {e}", c.name));
            if let Some(m) = first_byte_mismatch(heads, &got, &want) {
                panic!("H{heads} {}: module reference vs independent: {m}", c.name)
            }
            if c.name == "bump" {
                assert_eq!(bumps, expected_bumps, "H{heads}: one-ulp bump count")
            }
            let dec = q::reference_decode(heads, &got).unwrap();
            let want_dec = decode_ref(heads, &want);
            assert_eq!(dec.len(), want_dec.len());
            let i = (0..dec.len()).find(|&i| dec[i].to_bits() != want_dec[i].to_bits());
            assert!(
                i.is_none(),
                "H{heads} {}: module decode differs at {:?}",
                c.name,
                i
            );
        }
    }
    // Scale rule on its own, including amax exactly on grid points.
    for _ in 0..2000 {
        let amax = 2f32.powi(rng.below(60) as i32 - 30) * rng.unit();
        if amax > 448.0 * 65504.0 {
            continue;
        }
        assert_eq!(
            q::reference_scale_bits(amax),
            scale_bits(amax).0,
            "amax {amax:e}"
        );
    }
    assert_eq!(q::HEAD_DIM, D);
    for heads in HEADS {
        assert_eq!(q::row_bytes(heads), heads * D + 2 * heads)
    }
}

#[test]
fn rejects_unadmitted_head_counts() {
    // The admitted geometry is D256 with Hkv ∈ {2, 4}; anything else is an error, not a guess.
    assert!(q::build_encode(Arch::Gfx942, 3).is_err());
    assert!(q::build_decode(Arch::Gfx942, 3).is_err());
}

#[test]
fn encode_in_simulator_equals_reference_bytes() {
    for heads in HEADS {
        let kernel = q::build_encode(Arch::Gfx942, heads)
            .unwrap_or_else(|e| panic!("H{heads}: build_encode: {e}"));
        let image = image(&kernel, &q::encode_symbol(heads));
        let cases = encode_cases(heads).0;
        for c in &cases {
            let (want, _) = encode_ref(heads, &c.src);
            // -0 and negative underflow must have been canonicalised by the reference itself.
            assert!(
                want.chunks(heads * D + 2 * heads)
                    .all(|r| r[..heads * D].iter().all(|&b| b != 0x80)),
                "{}: 0x80 in reference",
                c.name
            );
            let got = run(&image, heads, c.rows(), f32_bytes(&c.src), want.len());
            if let Some(m) = first_byte_mismatch(heads, &got, &want) {
                panic!(
                    "H{heads} case {}: simulator PM encode differs from the CPU reference: {m}",
                    c.name
                );
            }
        }
    }
}

/// All finite OCP codes under adversarial scales: every byte `0x00..=0xff` except
/// the three the codec never emits (`0x80` OCP -0, `0x7f`/`0xff` OCP NaN).
fn decode_stored(heads: usize) -> Vec<u8> {
    const SCALES: [u16; 12] = [
        0x0001, 0x0003, 0x03ff, 0x0400, 0x3800, 0x3c00, 0x4a3d, 0x5a3c, 0x7800, 0x7bfe, 0x7bff,
        0x0155,
    ];
    let codes: Vec<u8> = (0..=255u8)
        .filter(|&c| !matches!(c, 0x80 | 0x7f | 0xff))
        .collect();
    assert_eq!(codes.len(), 253);
    let rows = 2 * SCALES.len();
    let mut rng = Rng(0xabcdef ^ heads as u64);
    let mut out = Vec::new();
    for row in 0..rows {
        let mut scales = Vec::new();
        for head in 0..heads {
            let start = rng.below(codes.len());
            out.extend((0..D).map(|i| codes[(start + i) % codes.len()]));
            scales.push(SCALES[(row + head * 5) % SCALES.len()]);
        }
        for s in scales {
            out.extend(s.to_le_bytes())
        }
    }
    out
}

#[test]
fn decode_in_simulator_equals_reference_exactly() {
    for heads in HEADS {
        let kernel = q::build_decode(Arch::Gfx942, heads)
            .unwrap_or_else(|e| panic!("H{heads}: build_decode: {e}"));
        let image = image(&kernel, &q::decode_symbol(heads));
        let mut inputs: Vec<(String, Vec<u8>)> = vec![("all-codes".into(), decode_stored(heads))];
        // Everything the encoder produces decodes through the same kernel.
        for c in &encode_cases(heads).0 {
            inputs.push((format!("encoded-{}", c.name), encode_ref(heads, &c.src).0))
        }
        for (name, stored) in inputs {
            let row_bytes = heads * D + 2 * heads;
            let rows = stored.len() / row_bytes;
            let want = decode_ref(heads, &stored);
            let module = q::reference_decode(heads, &stored).unwrap();
            let got = f32s(&run(&image, heads, rows, stored.clone(), want.len() * 4));
            assert_eq!(got.len(), want.len());
            for i in 0..got.len() {
                assert_eq!(got[i].to_bits(), want[i].to_bits(),
                    "H{heads} {name}: element {i} (row {}, head {}, channel {}): simulator {:e} ({:#010x}) vs reference {:e} ({:#010x}); stored code {:#04x}",
                    i / (heads * D), i % (heads * D) / D, i % D, got[i], got[i].to_bits(), want[i], want[i].to_bits(),
                    stored[(i / (heads * D)) * row_bytes + (i % (heads * D))]);
                assert_eq!(
                    module[i].to_bits(),
                    want[i].to_bits(),
                    "H{heads} {name}: module decode reference, element {i}"
                );
            }
        }
    }
}

#[test]
fn round_trip_in_simulator_is_the_reference_round_trip() {
    // encode → decode in the simulator reproduces decode_ref(encode_ref(x)) bit for bit.
    let heads = 2;
    let enc = q::build_encode(Arch::Gfx942, heads).unwrap();
    let dec = q::build_decode(Arch::Gfx942, heads).unwrap();
    let (enc_image, dec_image) = (
        image(&enc, &q::encode_symbol(heads)),
        image(&dec, &q::decode_symbol(heads)),
    );
    let mut rng = Rng(42);
    let c = random_case(heads, &mut rng);
    let stored = run(
        &enc_image,
        heads,
        c.rows(),
        f32_bytes(&c.src),
        c.rows() * (heads * D + 2 * heads),
    );
    let back = f32s(&run(&dec_image, heads, c.rows(), stored, c.src.len() * 4));
    let want = decode_ref(heads, &encode_ref(heads, &c.src).0);
    for i in 0..want.len() {
        assert_eq!(back[i].to_bits(), want[i].to_bits(), "element {i}")
    }
}

// ---------------------------------------------------------------------------
// HIP twin: offline gfx942 compile + machine-code checks (oracle-gated)

fn rocm_root() -> PathBuf {
    std::env::var_os("ROCM_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
}

fn hip_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src/pm_cdna/qsa_fp8_kv.gfx942.hip")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("cdna-qsa-fp8-kv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(tag)
}

/// `hipcc --offload-arch=gfx942` of the twin to a device code object; `Err` carries the diagnostics.
fn hipcc(tag: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
    let out = scratch(&format!("{tag}.co"));
    let result = Command::new(rocm_root().join("bin/hipcc"))
        .args([
            "--offload-arch=gfx942",
            "-O3",
            "--offload-device-only",
            "--no-gpu-bundle-output",
            "-c",
        ])
        .args(extra)
        .arg("-o")
        .arg(&out)
        .arg(hip_source())
        .output()
        .map_err(|e| format!("hipcc: {e}"))?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).into_owned());
    }
    let bytes = std::fs::read(&out).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&out);
    Ok(bytes)
}

/// `llvm-objdump -d` of a code object, split per function: mnemonic list and full lines.
fn disassemble(co: &[u8], tag: &str) -> BTreeMap<String, Vec<String>> {
    let path = scratch(&format!("{tag}.dis.co"));
    std::fs::write(&path, co).unwrap();
    let out = Command::new(oracle::tool("llvm-objdump"))
        .args(["-d", "--mcpu=gfx942", "--no-show-raw-insn"])
        .arg(&path)
        .output()
        .expect("llvm-objdump");
    let _ = std::fs::remove_file(&path);
    assert!(
        out.status.success(),
        "llvm-objdump: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut functions: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Some(head) = line.strip_suffix(">:") {
            if let Some((_, name)) = head.split_once(" <") {
                if name.starts_with('.') {
                    continue;
                } // local labels stay inside their function
                current = Some(name.to_string());
                functions.entry(name.to_string()).or_default();
                continue;
            }
        }
        if let (Some(f), true) = (
            &current,
            line.starts_with(char::is_whitespace) && !line.trim().is_empty(),
        ) {
            functions.get_mut(f).unwrap().push(oracle::normalize(line));
        }
    }
    functions
}

fn count(lines: &[String], mnemonic: &str) -> usize {
    lines
        .iter()
        .filter(|l| {
            let m = l.split_whitespace().next().unwrap_or("");
            m == mnemonic
                || m.strip_prefix(mnemonic)
                    .is_some_and(|r| matches!(r, "_e32" | "_e64" | "_sdwa" | "_dpp"))
        })
        .count()
}

fn skip_without_rocm(test: &str) -> bool {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping {test}");
        return true;
    }
    false
}

#[test]
fn hip_twin_compiles_for_gfx942_with_ieee_division() {
    if skip_without_rocm("hip_twin_compiles_for_gfx942_with_ieee_division") {
        return;
    }
    let co = hipcc("ieee", &[]).unwrap_or_else(|e| panic!("hipcc failed on the HIP twin:\n{e}"));
    let functions = disassemble(&co, "ieee");
    // Exactly the PM kernels' symbols without `_pm`, pointer-only ABI (kernargs at offsets 0 and 8).
    let mut names: Vec<&str> = functions.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "qsa_kv_decode_fp8_h2_gfx942",
            "qsa_kv_decode_fp8_h4_gfx942",
            "qsa_kv_encode_fp8_h2_gfx942",
            "qsa_kv_encode_fp8_h4_gfx942"
        ]
    );
    for heads in HEADS {
        assert_eq!(
            q::encode_symbol(heads).replace("_pm_gfx942", "_gfx942"),
            format!("qsa_kv_encode_fp8_h{heads}_gfx942")
        );
        assert_eq!(
            q::decode_symbol(heads).replace("_pm_gfx942", "_gfx942"),
            format!("qsa_kv_decode_fp8_h{heads}_gfx942")
        );
        let enc = &functions[&format!("qsa_kv_encode_fp8_h{heads}_gfx942")];
        let dec = &functions[&format!("qsa_kv_decode_fp8_h{heads}_gfx942")];
        // No kernarg beyond the two pointers (bytes 0..16): nothing is loaded from offset 0x10 or later.
        for (name, body) in [("encode", enc), ("decode", dec)] {
            assert!(
                !body
                    .iter()
                    .any(|l| l.starts_with("s_load") && l.contains("s[0:1], 0x1")),
                "H{heads} {name}: kernarg beyond the two pointers is read:\n{}",
                body.join("\n")
            );
        }

        // Two IEEE binary32 divisions in the encoder (amax/448 and value/sf), each the full
        // v_div_scale → (rcp, fma refinement) → v_div_fmas → v_div_fixup sequence — not a
        // reciprocal multiply: value*(1/sf) would change stored bytes.
        assert_eq!(
            count(enc, "v_div_scale_f32"),
            4,
            "H{heads}: two divisions, two v_div_scale each:\n{}",
            enc.join("\n")
        );
        assert_eq!(count(enc, "v_div_fmas_f32"), 2);
        assert_eq!(count(enc, "v_div_fixup_f32"), 2);
        // The FNUZ-half fold: z*0.5 into the FNUZ converter, F16 scale by the RNE converter.
        assert!(
            enc.iter()
                .any(|l| l.starts_with("v_mul_f32") && l.contains(", 0.5,")),
            "no z*0.5 before the conversion"
        );
        assert_eq!(count(enc, "v_cvt_pk_fp8_f32"), 1);
        assert!(count(enc, "v_cvt_f16_f32") >= 1);
        assert!(
            enc.iter().any(|l| l.starts_with("ds_")
                || l.contains("row_")
                || l.starts_with("v_permlane")
                || l.contains("swizzle")),
            "no cross-lane max exchange in the encoder"
        );
        assert!(
            enc.iter().any(|l| l.starts_with("s_barrier")),
            "no workgroup barrier in the encoder"
        );
        assert_eq!(
            count(dec, "v_div_scale_f32"),
            0,
            "the decoder divides nothing"
        );

        // Decode: FNUZ byte read, ×(2·scale) in f32; no divide, no convert-to-FP8.
        assert!(
            count(dec, "v_cvt_f32_fp8") >= 1,
            "decode lacks v_cvt_f32_fp8:\n{}",
            dec.join("\n")
        );
        assert_eq!(count(dec, "v_cvt_pk_fp8_f32"), 0);
        assert_eq!(count(dec, "v_div_fixup_f32"), 0);
        assert!(
            count(dec, "v_cvt_f32_f16") + count(dec, "v_fma_mix_f32") >= 1,
            "decode lacks the f16 scale widening"
        );
    }
}

#[test]
fn hip_twin_refuses_fast_math() {
    if skip_without_rocm("hip_twin_refuses_fast_math") {
        return;
    }
    let err = hipcc("fast", &["-ffast-math"])
        .expect_err("fast-math build of the HIP twin must be refused");
    assert!(
        err.contains("IEEE division"),
        "unexpected diagnostics:\n{err}"
    );
}

#[test]
fn hip_twin_is_gfx942_only() {
    if skip_without_rocm("hip_twin_is_gfx942_only") {
        return;
    }
    let result = Command::new(rocm_root().join("bin/hipcc"))
        .args([
            "--offload-arch=gfx1201",
            "-O3",
            "--offload-device-only",
            "--no-gpu-bundle-output",
            "-c",
            "-o",
        ])
        .arg(scratch("wrong-arch.co"))
        .arg(hip_source())
        .output()
        .expect("hipcc");
    assert!(
        !result.status.success(),
        "the FNUZ-fold twin must not compile for gfx1201 (OCP converter)"
    );
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("gfx942-only"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
