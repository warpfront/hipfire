// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C for the gfx942 plain FP8 activation producer
//! (`pm_cdna_fp8_act_producer`): the PeaceMaker kernel is built, emitted as a
//! code object, loaded into the CPU wave64 emulator and launched; every output
//! (FP8 codes, half sums, row scales) is compared bitwise with
//! `fp8_act_producer::reference`, and the reference itself is pinned to a
//! formula written out here from the original gfx12 source
//! (`pack_f32_to_fp8_mq4v2_gfx12`: `v = x * (1.0f / a_n)`, OCP float8_e4m3fn
//! round-nearest-even, half sums over the decoded emitted bytes) that shares no
//! code with the kernel module.
//!
//! The hardware FP8 conversions are FNUZ (bias 8) and the kernel feeds them
//! `ldexp(x, -e-1)`; the FNUZ byte is then the OCP byte of `x / 2^e` — except
//! OCP `0x80` (-0), which FNUZ cannot encode (it is NaN there) and which is the
//! byte `0x00`. The OCP model below therefore maps `0x80 -> 0x00`.
//!
//! The OCP byte identity above is claimed only inside the OCP finite range after scaling
//! (`|x| / a_n <= 448`): beyond it OCP (`0x7f` NaN) and FNUZ (`0x7f` = 240) disagree. Past 480
//! (with the PeaceMaker default FP16_OVFL = false) the kernel and `reference` are defined —
//! overflow gives FNUZ `0x80` — and are compared with each other only, never with the OCP model;
//! magnitudes in (448, 480] are avoided. Supported `row_scale_shift` is [-1, 3] (anything else is
//! rejected by `reference`); any `scale_mode` other than 1 means unit scale (source-exact `!= 1`).
//!
//! The offline HIP compile of the twin needs ROCm and honours
//! `HIPFIRE_TEST_REQUIRE_ROCM=0` like every other oracle test; no GPU is used.
use peacemaker_cdna::sim::{self, Config, KernelImage, Launch, Memory};
use peacemaker_isa::cdna::emit::{code_object, CodeObjectVersion};
use peacemaker_isa::kernels::cdna::fp8_act_producer::{build, reference, SYMBOL};
use peacemaker_isa::Arch;
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const X_ADDR: u64 = 0x1000_0000;
const FP8_ADDR: u64 = 0x4000_0000;
const HALF_ADDR: u64 = 0x5000_0000;
const SCALE_ADDR: u64 = 0x6000_0000;
const KARG_ADDR: u64 = 0x100;
/// Poison for output buffers: every byte must be overwritten by the kernel.
const POISON: u8 = 0xee;

// ----------------------------------------------------------------------------
// Kernel under test

/// The kernel built, emitted as a v5 code object and loaded once for the whole test binary.
static IMAGE: LazyLock<KernelImage> = LazyLock::new(|| {
    let kernel = build(Arch::Gfx942).unwrap_or_else(|e| panic!("build(gfx942): {e}"));
    let elf = code_object(&[&kernel], CodeObjectVersion::V5)
        .unwrap_or_else(|e| panic!("code_object: {e}"));
    sim::load_code_object(&elf, SYMBOL)
        .unwrap_or_else(|e| panic!("load_code_object({SYMBOL}): {e}"))
});

struct Outputs {
    codes: Vec<u8>,
    half_sums: Vec<f32>,
    row_scales: Vec<f32>,
}

/// Launch grid `[n]` × block 64 over `n = x.len() / k` rows.
fn run_sim(x: &[f32], k: usize, mode: u32, shift: i32) -> Outputs {
    let n = x.len() / k;
    let mut mem = Memory::default();
    mem.map_readonly(
        "x_f32",
        X_ADDR,
        x.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map("x_fp8", FP8_ADDR, vec![POISON; n * k]).unwrap();
    mem.map("half_sums", HALF_ADDR, vec![POISON; n * (k / 256) * 2 * 4])
        .unwrap();
    mem.map("row_scales", SCALE_ADDR, vec![POISON; n * 4])
        .unwrap();
    let mut kernarg = Vec::with_capacity(48);
    for p in [X_ADDR, FP8_ADDR, HALF_ADDR, SCALE_ADDR] {
        kernarg.extend(p.to_le_bytes())
    }
    kernarg.extend((k as u32).to_le_bytes());
    kernarg.extend((n as u32).to_le_bytes());
    kernarg.extend(mode.to_le_bytes());
    kernarg.extend(shift.to_le_bytes());
    assert_eq!(kernarg.len(), 48);
    let launch = Launch {
        grid: [n as u32, 1, 1],
        block: [64, 1, 1],
        kernarg,
        kernarg_addr: KARG_ADDR,
        dynamic_lds: 0,
    };
    sim::run(&IMAGE, &launch, &mut mem, &Config::default())
        .unwrap_or_else(|e| panic!("sim (K={k} N={n} mode={mode} shift={shift}): {e}"));
    let floats = |name: &str| {
        mem.region(name)
            .unwrap()
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect::<Vec<f32>>()
    };
    Outputs {
        codes: mem.region("x_fp8").unwrap().bytes.clone(),
        half_sums: floats("half_sums"),
        row_scales: floats("row_scales"),
    }
}

// ----------------------------------------------------------------------------
// Independent model of the original source

fn pow2_f64(e: i32) -> f64 {
    assert!((-1022..=1023).contains(&e), "{e}");
    f64::from_bits(((1023 + e) as u64) << 52)
}
fn pow2_f32(e: i32) -> f32 {
    assert!((-126..=127).contains(&e), "{e}");
    f32::from_bits(((e + 127) as u32) << 23)
}

/// OCP float8_e4m3fn magnitude of a non-NaN byte (bias 7; max 0x7e = 448).
fn ocp_value(byte: u8) -> f64 {
    let (e, m) = (i32::from((byte >> 3) & 15), f64::from(byte & 7));
    if e == 0 {
        m * pow2_f64(-9)
    } else {
        (1.0 + m / 8.0) * pow2_f64(e - 7)
    }
}

/// Nearest-even OCP E4M3 byte of `x` (`|x| <= 448`): the neighbour pair is
/// found by exact comparison on the f64 grid, a tie goes to the even byte
/// (even mantissa LSB), the sign bit is kept for results that round to zero.
fn ocp_code(x: f64) -> u8 {
    let a = x.abs();
    assert!(a <= 448.0, "{x} is outside the OCP finite range");
    let mut lo = 0u8;
    while lo < 0x7e && ocp_value(lo + 1) <= a {
        lo += 1
    }
    let mag = if lo == 0x7e {
        lo
    } else {
        let (below, above) = (a - ocp_value(lo), ocp_value(lo + 1) - a);
        if below < above {
            lo
        } else if above < below {
            lo + 1
        } else if lo % 2 == 0 {
            lo
        } else {
            lo + 1
        }
    };
    mag | (u8::from(x.is_sign_negative()) << 7)
}

/// `(unclamped e, clamped e)` of `amax > 0`: `e = ceil(log2(RN(amax * RN(1/224))))`
/// read from the f32 bit pattern. A product that underflows to zero reads as
/// exponent field 0: `-127`, clamped.
fn scale_exponents(amax: f32) -> (i32, i32) {
    let t = amax * (1.0f32 / 224.0f32);
    let b = t.to_bits();
    let e = (b >> 23) as i32 - 127 + i32::from(b & 0x7f_ffff != 0);
    (e, e.clamp(-31, 9))
}

/// `(a_n, e_f)` with `a_n = 2^e_f` the final row scale.
fn model_scale(row: &[f32], mode: u32, shift: i32) -> (f32, i32) {
    if mode != 1 {
        return (1.0, 0);
    }
    let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let e = if amax == 0.0 {
        0
    } else {
        scale_exponents(amax).1
    };
    let mut a = pow2_f32(e);
    if shift != 0 {
        a *= pow2_f32(shift)
    }
    (a, (a.to_bits() >> 23) as i32 - 127)
}

/// Every `|x| / a_n` lies in the OCP finite range.
fn fits(row: &[f32], mode: u32, shift: i32) -> bool {
    let (_, ef) = model_scale(row, mode, shift);
    row.iter()
        .all(|v| f64::from(v.abs()) * pow2_f64(-ef) <= 448.0)
}

fn ocp_decode(byte: u8) -> f32 {
    assert!(byte != 0x7f && byte != 0xff, "NaN byte");
    let v = ocp_value(byte & 0x7f) as f32;
    if byte & 0x80 != 0 {
        -v
    } else {
        v
    }
}

struct Model {
    codes: Vec<u8>,
    half_sums: Vec<f32>,
    row_scales: Vec<f32>,
}

fn model(x: &[f32], k: usize, mode: u32, shift: i32) -> Model {
    let mut m = Model {
        codes: Vec::with_capacity(x.len()),
        half_sums: Vec::new(),
        row_scales: Vec::new(),
    };
    for row in x.chunks_exact(k) {
        let (a_n, _) = model_scale(row, mode, shift);
        m.row_scales.push(a_n);
        let inv_a = 1.0f32 / a_n;
        let start = m.codes.len();
        for &v in row {
            let code = ocp_code(f64::from(v * inv_a));
            m.codes.push(if code == 0x80 { 0 } else { code });
        }
        let codes = &m.codes[start..];
        for g in 0..k / 256 {
            for side in 0..2 {
                let base = g * 256 + side * 128;
                // Per logical lane: four consecutive decoded bytes, left-associated.
                let mut lane: Vec<f32> = (0..32)
                    .map(|l| {
                        let d = |i: usize| ocp_decode(codes[base + 4 * l + i]);
                        ((d(0) + d(1)) + d(2)) + d(3)
                    })
                    .collect();
                for off in [16usize, 8, 4, 2, 1] {
                    lane = (0..32).map(|l| lane[l] + lane[l ^ off]).collect();
                }
                m.half_sums.push(lane[0]);
            }
        }
    }
    m
}

// ----------------------------------------------------------------------------
// Comparison helpers

fn assert_bytes(what: &str, ctx: &str, got: &[u8], want: &[u8], k: usize) {
    assert_eq!(got.len(), want.len(), "{ctx}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{ctx}: {what} differs at row {} element {} (byte {i}): got {:#04x}, want {:#04x}",
            i / k,
            i % k,
            got[i],
            want[i]
        );
    }
}

fn assert_bits(what: &str, ctx: &str, got: &[f32], want: &[f32], per_row: usize) {
    assert_eq!(got.len(), want.len(), "{ctx}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!("{ctx}: {what} differs at row {} index {} (flat {i}): got {:?} ({:#010x}), want {:?} ({:#010x})",
            i / per_row, i % per_row, got[i], got[i].to_bits(), want[i], want[i].to_bits());
    }
}

// ----------------------------------------------------------------------------
// Cases

/// The source-route `row_scale_shift` range the contract supports; `reference` rejects
/// anything else, in either scale mode.
const SHIFTS: std::ops::RangeInclusive<i32> = -1..=3;

struct Case {
    name: String,
    k: usize,
    mode: u32,
    shift: i32,
    rows: Vec<Vec<f32>>,
}

impl Case {
    fn new(name: impl Into<String>, k: usize, mode: u32, shift: i32, rows: Vec<Vec<f32>>) -> Self {
        assert!(
            SHIFTS.contains(&shift),
            "shift {shift} is outside the supported range {SHIFTS:?}"
        );
        let name = name.into();
        assert!(!rows.is_empty(), "{name}: no rows");
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), k, "{name}: row {r} length");
            assert!(
                row.iter().all(|v| v.is_finite()),
                "{name}: row {r} not finite"
            );
            assert!(
                fits(row, mode, shift),
                "{name}: row {r} leaves the OCP finite range"
            );
        }
        Self {
            name,
            k,
            mode,
            shift,
            rows,
        }
    }
    /// Keeps only the rows that fit; the case must retain at least one.
    fn filtered(
        name: impl Into<String>,
        k: usize,
        mode: u32,
        shift: i32,
        rows: Vec<Vec<f32>>,
    ) -> Self {
        let rows = rows.into_iter().filter(|r| fits(r, mode, shift)).collect();
        Self::new(name, k, mode, shift, rows)
    }
    /// A case that deliberately leaves the OCP finite range (`|x / a_n| > 480`, FNUZ overflow):
    /// compared with `reference` only, never with the OCP model. No scaled magnitude may sit in
    /// (448, 480], where OCP (`0x7f` NaN) and FNUZ (`0x7f` = 240) are both defined but disagree.
    fn overflow(
        name: impl Into<String>,
        k: usize,
        mode: u32,
        shift: i32,
        rows: Vec<Vec<f32>>,
    ) -> Self {
        let name = name.into();
        assert!(
            SHIFTS.contains(&shift),
            "shift {shift} is outside the supported range {SHIFTS:?}"
        );
        assert!(!rows.is_empty(), "{name}: no rows");
        let mut overflowing = 0;
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), k, "{name}: row {r} length");
            assert!(
                row.iter().all(|v| v.is_finite()),
                "{name}: row {r} not finite"
            );
            let (_, ef) = model_scale(row, mode, shift);
            for v in row {
                let s = f64::from(v.abs()) * pow2_f64(-ef);
                assert!(
                    !(448.0 < s && s <= 480.0),
                    "{name}: row {r} has a scaled magnitude {s} in (448, 480]"
                );
                if s > 480.0 {
                    overflowing += 1
                }
            }
        }
        assert!(overflowing > 0, "{name}: no element overflows");
        Self {
            name,
            k,
            mode,
            shift,
            rows,
        }
    }
    /// No scaled magnitude of `row` in (448, 480].
    fn gray_free(row: &[f32], mode: u32, shift: i32) -> bool {
        let (_, ef) = model_scale(row, mode, shift);
        row.iter().all(|v| {
            let s = f64::from(v.abs()) * pow2_f64(-ef);
            !(448.0 < s && s <= 480.0)
        })
    }
    fn ctx(&self) -> String {
        format!(
            "{} (K={} N={} mode={} shift={})",
            self.name,
            self.k,
            self.rows.len(),
            self.mode,
            self.shift
        )
    }
    fn flat(&self) -> Vec<f32> {
        self.rows.concat()
    }
}

/// The reference equals the independent model of the original formula.
fn check_reference(c: &Case) {
    let x = c.flat();
    let r = reference(&x, c.k, c.mode, c.shift)
        .unwrap_or_else(|e| panic!("{}: reference: {e}", c.ctx()));
    let m = model(&x, c.k, c.mode, c.shift);
    let ctx = c.ctx();
    assert_bytes(
        "reference codes vs original OCP formula (0x80 -> 0)",
        &ctx,
        &r.codes,
        &m.codes,
        c.k,
    );
    assert_bits(
        "reference row_scales vs model",
        &ctx,
        &r.row_scales,
        &m.row_scales,
        1,
    );
    assert_bits(
        "reference half_sums vs model",
        &ctx,
        &r.half_sums,
        &m.half_sums,
        c.k / 128,
    );
}

/// The kernel on the emulator equals the reference (and so the model) bitwise.
fn check_sim(c: &Case) {
    check_reference(c);
    let x = c.flat();
    let r = reference(&x, c.k, c.mode, c.shift).unwrap();
    let got = run_sim(&x, c.k, c.mode, c.shift);
    let ctx = c.ctx();
    assert_bytes("x_fp8", &ctx, &got.codes, &r.codes, c.k);
    assert_bits("row_scales", &ctx, &got.row_scales, &r.row_scales, 1);
    assert_bits("half_sums", &ctx, &got.half_sums, &r.half_sums, c.k / 128);
}

// ----------------------------------------------------------------------------
// Row generators

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn mantissa(&mut self) -> f64 {
        1.0 + (self.next() >> 41) as f64 / (1u64 << 23) as f64
    }
    fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn up(f: f32) -> f32 {
    assert!(f > 0.0);
    f32::from_bits(f.to_bits() + 1)
}
fn down(f: f32) -> f32 {
    assert!(f > 0.0);
    f32::from_bits(f.to_bits() - 1)
}

/// A row whose largest magnitude lies in `[2^b, 2^(b+1))`: values spread over
/// 24 binades below it, some zeros of either sign.
fn random_row(rng: &mut Rng, k: usize, b: i32) -> Vec<f32> {
    let mut row: Vec<f32> = (0..k)
        .map(|_| {
            if rng.below(16) == 0 {
                return if rng.below(2) == 0 { 0.0 } else { -0.0 };
            }
            (rng.sign() * rng.mantissa() * pow2_f64(b - rng.below(24) as i32 - 1)) as f32
        })
        .collect();
    let i = rng.below(k as u64) as usize;
    row[i] = (rng.sign() * rng.mantissa() * pow2_f64(b)) as f32;
    row
}

fn amax_exponent(rng: &mut Rng) -> i32 {
    rng.below(78) as i32 - 60
}

fn random_rows(rng: &mut Rng, n: usize, k: usize, mode: u32, shift: i32) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    let mut tries = 0;
    while rows.len() < n {
        tries += 1;
        assert!(
            tries < 100_000,
            "cannot generate fitting random rows for mode {mode} shift {shift}"
        );
        let b = if mode != 1 {
            rng.below(24) as i32 - 14
        } else {
            amax_exponent(rng)
        };
        let row = random_row(rng, k, b);
        if fits(&row, mode, shift) {
            rows.push(row)
        }
    }
    rows
}

/// Midpoints between adjacent OCP codes (every rounding tie) and the f32
/// neighbours on both sides of each, in units of the final scale.
fn tie_magnitudes() -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..0x7e {
        let mid = ((ocp_value(i) + ocp_value(i + 1)) / 2.0) as f32;
        v.extend([mid, up(mid), down(mid)]);
    }
    v
}

/// One row of rounding ties of both signs scaled by `2^ef`, with
/// `x[0] = cap * 2^ef` and no larger magnitude.
fn ties_row(k: usize, ef: i32, cap: f64) -> Vec<f32> {
    let mags: Vec<f32> = tie_magnitudes()
        .into_iter()
        .filter(|&m| f64::from(m) <= cap)
        .collect();
    let mut row = vec![(cap * pow2_f64(ef)) as f32];
    let mut i = 0usize;
    while row.len() < k {
        let m = f64::from(mags[(i / 2) % mags.len()]);
        row.push(((if i % 2 == 0 { m } else { -m }) * pow2_f64(ef)) as f32);
        i += 1;
    }
    row
}

// ----------------------------------------------------------------------------
// Case sets

const KS: [usize; 4] = [256, 512, 768, 2048];

fn zero_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for k in [256, 768] {
        let mut mixed: Vec<f32> = (0..k)
            .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
            .collect();
        mixed[k - 1] = -0.0;
        let rows = vec![vec![0.0; k], vec![-0.0; k], mixed];
        for (mode, shift) in [(0, 0), (0, 3), (1, 0), (1, 3), (1, -1)] {
            cases.push(Case::new(
                format!("zero rows K{k}"),
                k,
                mode,
                shift,
                rows.clone(),
            ));
        }
        // One nonzero among zeros, positive and negative, at several positions.
        let single: Vec<Vec<f32>> = [0, 1, 127, 128, k / 2, k - 1]
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                let mut r = vec![if j % 2 == 0 { 0.0 } else { -0.0 }; k];
                r[i] = if j % 2 == 0 { 1.0 } else { -3.5 };
                r
            })
            .collect();
        for (mode, shift) in [(0, 0), (1, 0), (1, 2)] {
            cases.push(Case::new(
                format!("single nonzero K{k}"),
                k,
                mode,
                shift,
                single.clone(),
            ));
        }
    }
    cases
}

fn tiny_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let k = 256;
    let min_sub = f32::from_bits(1);
    let rows: Vec<Vec<f32>> = vec![
        // The smallest subnormal alone: RN(amax * RN(1/224)) underflows to zero, exponent field 0.
        {
            let mut r = vec![0.0; k];
            r[7] = min_sub;
            r
        },
        {
            let mut r = vec![-0.0; k];
            r[200] = -min_sub;
            r
        },
        (0..k)
            .map(|i| if i % 2 == 0 { min_sub } else { -min_sub })
            .collect(),
        (0..k)
            .map(|i| f32::from_bits(i as u32 + 1) * if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect(),
        (0..k)
            .map(|i| f32::from_bits(0x7f_ffff - i as u32) * if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect(),
        // Largest subnormal, smallest normal and neighbours.
        (0..k)
            .map(|i| {
                f32::from_bits(0x7f_fff0 + (i as u32 % 32)) * if i % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect(),
        (0..k)
            .map(|i| {
                if i == 0 {
                    f32::MIN_POSITIVE
                } else {
                    f32::from_bits(i as u32 * 31)
                }
            })
            .collect(),
        (0..k)
            .map(|i| {
                if i == 5 {
                    -1.0e-30
                } else {
                    f32::from_bits(i as u32 * 977)
                }
            })
            .collect(),
        // Tiny amax (< 224 * 2^-31): scale clamped to 2^-31, values scaled up into the FP8 range.
        (0..k)
            .map(|i| {
                if i == 3 {
                    1.0e-8
                } else {
                    f32::from_bits(0x3200_0000 + (i as u32) * 0x1_0003)
                        * if i % 2 == 0 { 1.0 } else { -1.0 }
                }
            })
            .collect(),
    ];
    for (mode, shift) in [(1, 0), (1, 3), (1, -1), (0, 0), (0, 3)] {
        cases.push(Case::filtered(
            format!("tiny/subnormal rows mode {mode} shift {shift}"),
            k,
            mode,
            shift,
            rows.clone(),
        ));
    }
    // Mode 0 around the OCP minimum subnormal (2^-9) and its half (2^-10, the tie with zero).
    let near_min: Vec<f32> = (0..k)
        .map(|i| {
            let base = [
                pow2_f32(-9),
                pow2_f32(-10),
                pow2_f32(-11),
                1.5 * pow2_f32(-9),
                2.5 * pow2_f32(-9),
                7.0 * pow2_f32(-9),
            ][i % 6];
            let v = match (i / 6) % 3 {
                0 => base,
                1 => up(base),
                _ => down(base),
            };
            if (i / 18) % 2 == 0 {
                v
            } else {
                -v
            }
        })
        .collect();
    cases.push(Case::new(
        "near OCP minimum subnormal, mode 0",
        k,
        0,
        0,
        vec![near_min.clone()],
    ));
    cases.push(Case::new(
        "near OCP minimum subnormal, mode 0 ignores shift",
        k,
        0,
        -1,
        vec![near_min],
    ));
    cases
}

fn pow2_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (k, rows_n) in [(256, 1), (512, 3)] {
        for shift in [0, -1, 1] {
            // Row per amax = 2^a (a = -40..=17), elements ±2^(a-i%16).
            let by_amax: Vec<Vec<f32>> = (-40..=17)
                .map(|a| {
                    (0..k)
                        .map(|i| {
                            let v = pow2_f64(a - (i % 16) as i32) as f32;
                            if (i / 16) % 2 == 0 {
                                v
                            } else {
                                -v
                            }
                        })
                        .collect()
                })
                .collect();
            cases.push(Case::filtered(
                format!("powers of two by amax K{k}"),
                k,
                1,
                shift,
                by_amax,
            ));
            // Wide ramps: 2^-26..2^7 and the whole f32 subnormal/normal low range.
            let ramps: Vec<Vec<f32>> = (0..rows_n)
                .map(|r| {
                    (0..k)
                        .map(|i| {
                            let v = pow2_f64(((i + 5 * r) % 34) as i32 - 26) as f32;
                            if (i + r) % 3 == 0 {
                                -v
                            } else {
                                v
                            }
                        })
                        .collect()
                })
                .collect();
            cases.push(Case::new(
                format!("power-of-two ramp K{k}"),
                k,
                1,
                shift,
                ramps,
            ));
        }
    }
    let low: Vec<f32> = (0..256u32)
        .map(|i| {
            if i < 24 {
                f32::from_bits(1 << i)
            } else {
                pow2_f32(((i - 24) % 126) as i32 - 126)
            }
        })
        .collect();
    cases.push(Case::new(
        "power-of-two subnormal/low-normal row",
        256,
        1,
        0,
        vec![low.clone()],
    ));
    cases.push(Case::new(
        "power-of-two mode 0",
        256,
        0,
        0,
        vec![
            (0..256)
                .map(|i| {
                    let v = pow2_f32(i % 18 - 12);
                    if i % 2 == 0 {
                        v
                    } else {
                        -v
                    }
                })
                .collect(),
            low,
        ],
    ));
    cases
}

/// `amax = 224 * 2^k` (t is exactly a power of two) and ±ulps around it, for
/// every `k` from below the lower clamp to above the upper one.
fn boundary_cases() -> Vec<Case> {
    let mut rng = Rng(0x2245_112a_b0b5_c0de);
    let mut cases = Vec::new();
    for shift in [0, -1, 1, 3] {
        let mut rows = Vec::new();
        for kexp in -40..=10 {
            let base = (224.0 * pow2_f64(kexp)) as f32;
            for d in -2i32..=2 {
                let amax = f32::from_bits((base.to_bits() as i32 + d) as u32);
                let mut row: Vec<f32> = (0..256)
                    .map(|_| {
                        (rng.sign() * (rng.next() >> 11) as f64 / (1u64 << 53) as f64
                            * f64::from(amax)) as f32
                    })
                    .collect();
                row[(rng.below(256)) as usize] = if (kexp + d) % 2 == 0 { amax } else { -amax };
                rows.push(row);
            }
        }
        cases.push(Case::filtered(
            format!("amax = 224*2^k ± 2 ulp, k=-40..10"),
            256,
            1,
            shift,
            rows,
        ));
    }
    // The 112 boundary (t = 0.5 -> e = -1) and 224 boundary (t = 1.0 -> e = 0) with the whole row on
    // exactly representable binary fractions of amax.
    for amax in [
        112.0f32,
        up(112.0),
        down(112.0),
        224.0,
        up(224.0),
        down(224.0),
        448.0,
        up(448.0),
        down(448.0),
        56.0,
        896.0,
    ] {
        let row: Vec<f32> = (0..256)
            .map(|i| {
                if i == 0 {
                    amax
                } else {
                    amax * [0.5, -0.25, 0.75, -0.9375, 0.03125, -0.0078125, 0.99, -1.0][i % 8]
                }
            })
            .collect();
        for shift in [0, -1, 2] {
            if fits(&row, 1, shift) {
                cases.push(Case::new(
                    format!("amax {amax} (112/224 boundary)"),
                    256,
                    1,
                    shift,
                    vec![row.clone()],
                ))
            }
        }
    }
    cases
}

fn clamp_cases() -> Vec<Case> {
    let mut rng = Rng(0xc1a3_9000_0000_0001);
    let mut cases = Vec::new();
    // Upper clamp e = 9: amax >= 224*2^9 = 114688 (t >= 512) up to 224*2^10 (scaled 448).
    let uppers = [
        114_688.0f32,
        up(114_688.0),
        down(114_688.0),
        114_688.0 * 1.5,
        200_000.0,
        down(229_376.0),
        229_376.0,
        130_000.5,
        114_689.0,
        4096.0 * 28.0,
    ];
    let upper_rows: Vec<Vec<f32>> = uppers
        .iter()
        .enumerate()
        .map(|(r, &amax)| {
            (0..512)
                .map(|i| {
                    if i == 17 * r % 512 {
                        amax
                    } else {
                        (rng.sign() * rng.mantissa() / 2.0 * f64::from(amax)) as f32
                    }
                })
                .collect()
        })
        .collect();
    for shift in [0, 1, 3] {
        cases.push(Case::new(
            format!("upper clamp e=9 shift {shift}"),
            512,
            1,
            shift,
            upper_rows.clone(),
        ))
    }
    // Rows past the clamp whose scaled values still fit: scaled amax > 224.
    let past: Vec<Vec<f32>> = [
        (114_688.0f32 * 1.0001),
        114_688.0 * 1.9,
        114_688.0 * 1.99999,
    ]
    .iter()
    .map(|&amax| {
        (0..256)
            .map(|i| {
                if i == 1 {
                    -amax
                } else {
                    amax * [0.1, 0.5, -0.99, 0.0, -0.0, 0.8765, -0.4321, 0.2][i % 8]
                }
            })
            .collect()
    })
    .collect();
    cases.push(Case::new(
        "past upper clamp (scaled amax in (224, 448))",
        256,
        1,
        0,
        past,
    ));
    // Lower clamp e = -31: amax <= 224*2^-31, down to subnormals; other scales below it.
    let base = (224.0 * pow2_f64(-31)) as f32;
    let lowers = [
        base,
        up(base),
        down(base),
        base * 0.5,
        base * 0.25,
        base * 0.01,
        pow2_f32(-60),
        pow2_f32(-100),
        1.0e-38,
        1.0e-44,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1.0e-30,
        3.0e-12,
    ];
    let lower_rows: Vec<Vec<f32>> = lowers
        .iter()
        .enumerate()
        .map(|(r, &amax)| {
            (0..256)
                .map(|i| {
                    if i == 31 * r % 256 {
                        -amax
                    } else {
                        (rng.sign() * rng.mantissa() / 2.0 * f64::from(amax)) as f32
                    }
                })
                .collect()
        })
        .collect();
    for shift in [0, -1, 3] {
        cases.push(Case::filtered(
            format!("lower clamp e=-31 shift {shift}"),
            256,
            1,
            shift,
            lower_rows.clone(),
        ))
    }
    cases
}

fn tie_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    // Mode 1: x[0] = 224*2^e, ties of the final scale 2^(e+shift) up to the cap 224*2^-shift.
    for shift in [-1, 0, 2, 3] {
        let rows: Vec<Vec<f32>> = [0, -1, 5, -31, 9, -20, 3]
            .iter()
            .map(|&e| ties_row(2048, e + shift, 224.0 * pow2_f64(-shift)))
            .collect();
        cases.push(Case::new(
            format!("rounding ties, mode 1 shift {shift}"),
            2048,
            1,
            shift,
            rows,
        ));
    }
    // Mode 0: scale 1, every tie up to 448 (including those between 224 and 448).
    cases.push(Case::new(
        "rounding ties, mode 0",
        2048,
        0,
        0,
        vec![ties_row(2048, 0, 448.0), ties_row(2048, 0, 224.0)],
    ));
    cases.push(Case::new(
        "rounding ties, mode 0 ignores shift",
        2048,
        0,
        3,
        vec![ties_row(2048, 0, 448.0)],
    ));
    // Ties scaled by a non-trivial scale and then values at the exact top of the grid.
    let top: Vec<f32> = (0..256)
        .map(|i| [448.0f32, 432.0, 416.0, 400.0, 384.0, -448.0, -432.0, -440.0][i % 8])
        .collect();
    cases.push(Case::new(
        "top of the OCP grid, mode 0",
        256,
        0,
        0,
        vec![top.clone()],
    ));
    cases.push(Case::new(
        "top of the OCP grid, mode 1 shift -1",
        256,
        1,
        -1,
        vec![(0..256)
            .map(|i| if i == 0 { 224.0 } else { top[i] / 2.0 })
            .collect()],
    ));
    cases
}

fn random_cases() -> Vec<Case> {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut cases = Vec::new();
    for k in KS {
        for (mode, shift) in [(0, 0), (0, -1), (1, 0), (1, -1), (1, 2), (1, 3)] {
            let n = if k == 2048 { 3 } else { 6 };
            let rows = random_rows(&mut rng, n, k, mode, shift);
            cases.push(Case::new(format!("random rows K{k}"), k, mode, shift, rows));
        }
    }
    cases
}

fn all_cases() -> Vec<Case> {
    let mut v = zero_cases();
    v.extend(tiny_cases());
    v.extend(pow2_cases());
    v.extend(boundary_cases());
    v.extend(clamp_cases());
    v.extend(tie_cases());
    v.extend(random_cases());
    v
}

// ----------------------------------------------------------------------------
// Tests

/// CPU only: `reference` equals the independent original-formula model on all
/// case sets (codes byte for byte except OCP 0x80 -> 0, half sums, row scales).
#[test]
fn reference_equals_independent_original_formula() {
    let cases = all_cases();
    assert!(cases.len() > 100, "{} cases", cases.len());
    for c in &cases {
        check_reference(c)
    }
}

/// The OCP model itself: spot values of the grid, ties to even, -0 handling.
#[test]
fn ocp_model_self_check() {
    assert_eq!(ocp_value(0x01), pow2_f64(-9));
    assert_eq!(ocp_value(0x08), pow2_f64(-6));
    assert_eq!(ocp_value(0x38), 1.0);
    assert_eq!(ocp_value(0x76), 224.0);
    assert_eq!(ocp_value(0x7e), 448.0);
    assert_eq!(ocp_code(1.0625), 0x38);
    assert_eq!(ocp_code(1.1875), 0x3a);
    assert_eq!(ocp_code(f64::from(up(1.0625))), 0x39);
    assert_eq!(ocp_code(432.0), 0x7e);
    assert_eq!(ocp_code(0.5 * pow2_f64(-9)), 0x00);
    assert_eq!(ocp_code(-0.5 * pow2_f64(-9)), 0x80);
    assert_eq!(ocp_code(1.5 * pow2_f64(-9)), 0x02);
    assert_eq!(ocp_code(-0.0), 0x80);
    // Every byte round-trips through its own value.
    for b in (0..=0x7eu8).chain(0x80..=0xfe) {
        if b == 0x80 {
            continue;
        }
        let v = ocp_decode(b);
        assert_eq!(ocp_code(f64::from(v)), b, "{b:#04x}");
    }
}

#[test]
fn build_shape_and_kernarg_abi() {
    let k = build(Arch::Gfx942).unwrap();
    assert_eq!(k.name, SYMBOL);
    assert_eq!(k.arch, peacemaker_cdna::Arch::Gfx942);
    assert_eq!(k.resources.kernarg_size, 48);
    assert_eq!(k.resources.workgroup_size, 64);
    assert_eq!(k.resources.group_segment_fixed_size, 0);
    assert!(
        k.resources.workgroup_id[0] && !k.resources.workgroup_id[1] && !k.resources.workgroup_id[2]
    );
    assert_eq!(k.resources.workitem_id, 0);
    let abi: Vec<(u32, u32)> = k.kernargs.iter().map(|a| (a.offset, a.size)).collect();
    assert_eq!(
        abi,
        [
            (0, 8),
            (8, 8),
            (16, 8),
            (24, 8),
            (32, 4),
            (36, 4),
            (40, 4),
            (44, 4)
        ]
    );
    for other in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
        assert!(build(other).is_err(), "{other:?}");
    }
    // Deterministic: the same words twice.
    assert_eq!(build(Arch::Gfx942).unwrap().words, k.words);
}

#[test]
fn reference_rejects_unsupported_inputs() {
    assert!(reference(&[], 0, 1, 0).is_err(), "K = 0");
    assert!(
        reference(&vec![1.0; 128], 128, 1, 0).is_err(),
        "K not a multiple of 256"
    );
    assert!(reference(&vec![1.0; 300], 300, 1, 0).is_err(), "K = 300");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x = vec![1.0f32; 256];
        x[100] = bad;
        assert!(reference(&x, 256, 1, 0).is_err(), "{bad}");
        assert!(reference(&x, 256, 0, 0).is_err(), "{bad} mode 0");
    }
    // A well-formed row passes and has K codes, K/128 half sums, one scale.
    let r = reference(&vec![1.0; 512], 512, 1, 0).unwrap();
    assert_eq!(
        (r.codes.len(), r.half_sums.len(), r.row_scales.len()),
        (512, 4, 1)
    );
}

/// `row_scale_shift` is supported only in [-1, 3]: `reference` rejects everything else, in
/// both scale modes (mode 0 ignores the shift but still validates it).
#[test]
fn reference_rejects_shift_outside_minus1_to_3() {
    let x = vec![1.0f32; 256];
    for mode in [0u32, 1] {
        for shift in [-40, -3, -2, 4, 5, 9, 40, 100, i32::MIN, i32::MAX] {
            assert!(
                reference(&x, 256, mode, shift).is_err(),
                "mode {mode} shift {shift} accepted"
            );
        }
        for shift in SHIFTS {
            assert!(
                reference(&x, 256, mode, shift).is_ok(),
                "mode {mode} shift {shift} rejected"
            );
        }
    }
}

/// Row scale `a_n` pinned by hand: `t = RN(amax * RN(1/224))`, `e = ceil(log2 t)`
/// from the bit pattern, clamp [-31, 9], zero -> 1, then `*2^shift` (mode 1 only).
#[test]
fn row_scales_pinned_by_hand() {
    let row = |amax: f32| {
        let mut r = vec![0.0f32; 256];
        r[3] = -amax;
        r
    };
    let scale = |amax: f32, shift: i32| reference(&row(amax), 256, 1, shift).unwrap().row_scales[0];
    assert_eq!(scale(1.0, 0), pow2_f32(-7));
    assert_eq!(scale(128.0, 0), 1.0);
    assert_eq!(scale(224.0, 0), 1.0);
    assert_eq!(scale(448.0, 0), 2.0);
    assert_eq!(scale(0.0, 0), 1.0);
    assert_eq!(scale(0.0, 3), 8.0);
    assert_eq!(scale(224.0, 3), 8.0);
    assert_eq!(scale(224.0, -1), 0.5);
    assert_eq!(scale(1.0e-20, 0), pow2_f32(-31));
    assert_eq!(scale(1.0e-20, 2), pow2_f32(-29));
    assert_eq!(scale(f32::from_bits(1), 0), pow2_f32(-31));
    assert_eq!(scale(1.0e6, 0), pow2_f32(9));
    assert_eq!(scale(1.0e6, -1), pow2_f32(8));
    assert_eq!(
        reference(&row(1.0e6), 256, 0, 3).unwrap().row_scales[0],
        1.0
    );
}

/// The RN(1/224) edge. `RN(1/224) * 224 == 1.0` exactly, so amax = 224 gives
/// `t = 1.0`, `e = 0`, a_n = 1 — NOT 2: the rounded reciprocal does not push
/// the 224 boundary over. One ulp above 224 (t = 1 + 2^-23, mantissa != 0) gives
/// e = 1, a_n = 2. The same holds at every 224*2^k boundary: swept below over
/// k = -40..=12 and ±6 ulps, the rounded-reciprocal exponent equals the exact
/// `ceil(log2(amax / 224))`, so replacing the reciprocal by a division changes
/// nothing there. Kernel, reference and model must all agree.
#[test]
fn rounded_reciprocal_scale_edge() {
    let r = 1.0f32 / 224.0;
    assert_eq!(224.0f32 * r, 1.0);
    assert_eq!((224.0f32 * r).to_bits(), 0x3f80_0000);
    assert_eq!(scale_exponents(224.0).0, 0);
    assert_eq!(scale_exponents(up(224.0)).0, 1);
    assert_eq!(scale_exponents(down(224.0)).0, 0);
    assert_eq!(scale_exponents(112.0).0, -1);
    assert_eq!(scale_exponents(up(112.0)).0, 0);
    assert_eq!(scale_exponents(448.0).0, 1);

    let mut rows = Vec::new();
    for kexp in -40..=12 {
        let base = (224.0 * pow2_f64(kexp)) as f32;
        for d in -6i32..=6 {
            let amax = f32::from_bits((base.to_bits() as i32 + d) as u32);
            // Exact: smallest e with amax <= 224 * 2^e.
            let mut exact = -200;
            while f64::from(amax) > 224.0 * pow2_f64(exact) {
                exact += 1
            }
            assert_eq!(
                scale_exponents(amax).0,
                exact,
                "amax {amax:?} = 224*2^{kexp} {d:+} ulp"
            );
            let mut row = vec![0.0f32; 256];
            row[(kexp + 40) as usize * 3 % 256] = if d % 2 == 0 { amax } else { -amax };
            rows.push(row);
        }
    }
    check_sim(&Case::filtered(
        "224*2^k boundary sweep",
        256,
        1,
        0,
        rows.clone(),
    ));
    check_sim(&Case::filtered(
        "224*2^k boundary sweep, shift -1",
        256,
        1,
        -1,
        rows,
    ));

    // The headline rows, through the kernel: scales and the codes they imply.
    let amaxes = [
        224.0f32,
        up(224.0),
        down(224.0),
        112.0,
        up(112.0),
        448.0,
        up(448.0),
    ];
    let rows: Vec<Vec<f32>> = amaxes
        .iter()
        .map(|&a| {
            let mut r = vec![0.0f32; 256];
            r[0] = a;
            r[1] = -a / 2.0;
            r[2] = a / 7.0;
            r
        })
        .collect();
    let got = run_sim(&rows.concat(), 256, 1, 0);
    let want = [1.0f32, 2.0, 1.0, 0.5, 1.0, 2.0, 4.0];
    assert_bits("row_scales", "edge rows", &got.row_scales, &want, 1);
    // amax = 224 -> a_n = 1 -> x[0] = 224 = 0x76 (OCP), x[1] = -112 = 0xee.
    assert_eq!(got.codes[0], 0x76);
    assert_eq!(got.codes[1], 0xee);
    check_sim(&Case::new("224/112 headline rows", 256, 1, 0, rows));
}

#[test]
fn sim_zero_and_negative_zero_rows() {
    for c in zero_cases() {
        check_sim(&c)
    }
}

#[test]
fn sim_tiny_and_subnormal_rows() {
    for c in tiny_cases() {
        check_sim(&c)
    }
}

#[test]
fn sim_powers_of_two() {
    for c in pow2_cases() {
        check_sim(&c)
    }
}

#[test]
fn sim_224_112_boundaries() {
    for c in boundary_cases() {
        check_sim(&c)
    }
}

#[test]
fn sim_upper_and_lower_clamps() {
    for c in clamp_cases() {
        check_sim(&c)
    }
}

#[test]
fn sim_rounding_ties() {
    for c in tie_cases() {
        check_sim(&c)
    }
}

/// K = 256/512/768/2048 with random rows, both modes and several shifts.
#[test]
fn sim_random_rows_k_modes_shifts() {
    for c in random_cases() {
        check_sim(&c)
    }
}

/// Mode 0 ignores the shift: identical outputs for different shifts.
#[test]
fn sim_mode0_ignores_row_scale_shift() {
    let mut rng = Rng(77);
    let rows = random_rows(&mut rng, 4, 512, 0, 0);
    let x = rows.concat();
    let a = run_sim(&x, 512, 0, 0);
    for shift in [1, -1, 2, 3] {
        let b = run_sim(&x, 512, 0, shift);
        assert_eq!(a.codes, b.codes, "shift {shift}");
        assert_bits("half_sums", "mode 0 shifts", &a.half_sums, &b.half_sums, 4);
        assert!(
            b.row_scales.iter().all(|s| s.to_bits() == 1.0f32.to_bits()),
            "shift {shift}"
        );
    }
}

/// The offline HIP compile of the twin for gfx942 (no GPU is used).
#[test]
fn hip_twin_compiles_offline_for_gfx942() {
    let _ = require_rocm_tool!("clang");
    let hipcc = rocm::hipcc();
    assert!(
        hipcc.is_file(),
        "required {} missing; set HIPFIRE_TEST_REQUIRE_ROCM=0 only on no-ROCm hosts",
        hipcc.display()
    );
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/src/pm_cdna/fp8_act_producer.gfx942.hip");
    let text = std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    assert!(
        text.contains(SYMBOL),
        "{} does not define {SYMBOL}",
        src.display()
    );
    let obj =
        std::env::temp_dir().join(format!("pm-cdna-fp8-act-producer-{}.o", std::process::id()));
    let out = std::process::Command::new(&hipcc)
        .args(["--offload-arch=gfx942", "-O3", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .env("HIP_VISIBLE_DEVICES", "-1")
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", hipcc.display()));
    let size = std::fs::metadata(&obj).map(|m| m.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&obj);
    assert!(
        out.status.success(),
        "hipcc failed ({}):\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(size > 0, "hipcc produced no object");
}

/// Rows whose scaled magnitude exceeds 480 (above the FNUZ maximum 240·2): with
/// FP16_OVFL = false the converts overflow to FNUZ `0x80`. The OCP formula has no such code, so
/// the OCP model is NOT consulted here: only `reference` (the kernel contract) and the emulator.
fn overflow_cases() -> Vec<Case> {
    let mut rng = Rng(0x0f10_0480_dead_beef);
    let mut cases = Vec::new();
    // Every scaled magnitude of the fixed rows is far from (448, 480] for every shift in [-1, 3]
    // (Case::overflow asserts it).
    let amaxes = [250_000.0f32, 300_000.0, 1.0e6, 1.0e12, 1.0e30, f32::MAX];
    let fixed: Vec<Vec<f32>> = amaxes
        .iter()
        .map(|&a| {
            (0..512)
                .map(|i| match i % 8 {
                    0 => 0.0,
                    1 => a,
                    2 => -a,
                    3 => a * 0.001,
                    4 => -a * 0.0001,
                    5 => -0.0,
                    6 => 1.0,
                    _ => -3.0,
                })
                .collect()
        })
        .collect();
    for shift in SHIFTS {
        // Random rows with amax in [2^18, 2^20): past the upper clamp at e = 9, overflow at shifts <= 1.
        let mut rows = fixed.clone();
        let mut tries = 0;
        while rows.len() < fixed.len() + 4 {
            tries += 1;
            assert!(
                tries < 100_000,
                "no gray-zone-free random overflow rows for shift {shift}"
            );
            let b = 18 + rng.below(2) as i32;
            let row = random_row(&mut rng, 512, b);
            if Case::gray_free(&row, 1, shift) {
                rows.push(row)
            }
        }
        cases.push(Case::overflow(
            format!("overflow above 480, mode 1 shift {shift}"),
            512,
            1,
            shift,
            rows,
        ));
    }
    // Unit scale: values outside the grid map to 0x80, in-range values keep their codes.
    let mode0: Vec<Vec<f32>> = [0, 1]
        .iter()
        .map(|&r| {
            (0..256)
                .map(|i| {
                    [
                        600.0f32,
                        -1000.0,
                        1.0e20,
                        500.0,
                        -3.0,
                        0.5,
                        -0.0,
                        481.5,
                        f32::MAX,
                        -f32::MAX,
                        100.0,
                        -0.001,
                    ][(i + 5 * r) % 12]
                })
                .collect()
        })
        .collect();
    for shift in [0, 3] {
        cases.push(Case::overflow(
            format!("overflow above 480, mode 0 shift {shift}"),
            256,
            0,
            shift,
            mode0.clone(),
        ))
    }
    cases
}

/// Past the FNUZ range the emulated kernel equals `reference` bitwise (codes `0x80`, the sums
/// that follow, row scales), with no comparison to the OCP oracle; every element whose scaled
/// magnitude exceeds 480 must come out as the FNUZ overflow byte `0x80`.
#[test]
fn sim_overflow_above_480_matches_reference_without_ocp_oracle() {
    for c in overflow_cases() {
        let x = c.flat();
        let r = reference(&x, c.k, c.mode, c.shift)
            .unwrap_or_else(|e| panic!("{}: reference: {e}", c.ctx()));
        let got = run_sim(&x, c.k, c.mode, c.shift);
        let ctx = c.ctx();
        assert_bytes("x_fp8", &ctx, &got.codes, &r.codes, c.k);
        assert_bits("row_scales", &ctx, &got.row_scales, &r.row_scales, 1);
        assert_bits("half_sums", &ctx, &got.half_sums, &r.half_sums, c.k / 128);
        let mut overflowed = 0;
        for (row_idx, row) in c.rows.iter().enumerate() {
            let (_, ef) = model_scale(row, c.mode, c.shift);
            for (i, v) in row.iter().enumerate() {
                if f64::from(v.abs()) * pow2_f64(-ef) > 480.0 {
                    overflowed += 1;
                    assert_eq!(
                        got.codes[row_idx * c.k + i],
                        0x80,
                        "{ctx}: row {row_idx} element {i} ({v:?}) must overflow to FNUZ 0x80"
                    );
                }
            }
        }
        assert!(overflowed > 0, "{ctx}");
    }
}

/// Every `scale_mode` other than 1 is unit scale, exactly as the source's `scale_mode == 1`
/// test: outputs equal mode 0 and the shift (inside [-1, 3]) is ignored.
#[test]
fn sim_scale_modes_other_than_1_are_unit_scale() {
    let mut rng = Rng(0x5ca1_e000_0000_0002);
    for mode in [2u32, 3, 0x8000_0000, u32::MAX] {
        for shift in SHIFTS {
            let k = 512;
            let rows = random_rows(&mut rng, 3, k, mode, shift);
            let case = Case::new(format!("scale_mode {mode:#x}"), k, mode, shift, rows);
            check_sim(&case);
            let unit = run_sim(&case.flat(), k, 0, 0);
            let got = run_sim(&case.flat(), k, mode, shift);
            assert_eq!(got.codes, unit.codes, "{}", case.ctx());
            assert_bits(
                "half_sums",
                &case.ctx(),
                &got.half_sums,
                &unit.half_sums,
                k / 128,
            );
            assert!(
                got.row_scales
                    .iter()
                    .all(|s| s.to_bits() == 1.0f32.to_bits()),
                "{}",
                case.ctx()
            );
        }
    }
    // Zero and -0 rows too (a_n = 1.0, not 2^shift).
    let zeros = vec![vec![0.0f32; 256], vec![-0.0f32; 256]];
    for shift in SHIFTS {
        check_sim(&Case::new(
            "zero rows, scale_mode 2",
            256,
            2,
            shift,
            zeros.clone(),
        ))
    }
}
