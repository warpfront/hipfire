// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware byte oracle of the Qwen4 EP A8 MoE prefill chain (int8 per-128 activations x symmetric-IU4
//! weights, `qwen4_moe_ep_a8.gfx12.hip` + the two producers) on exact gfx1201.
//!
//! The CPU reference is independent of the kernels (integer/f32-exact):
//!   * activation block per K128: `amax = max|x|`, `d = RN(amax / 127)`, `q = clamp(rint(RN(x / d)), -127, 127)`
//!     (ties to even), zero block => `d = 0, q = 0`, `s = sum q`; qs in the "nibble-pair order";
//!   * GEMM per output and ascending K128 epoch h: exact i32 `C_h = sum (u - 8) * q`,
//!     `sum = fmaf(RN(f32(sc_h) * d_h), (float)C_h, sum)` from +0;
//!   * gate/up epilogue `bf16(silu(bf16 g) * bf16 u)` with the shipped hipcc expression
//!     `(v / (1.0f + expf(-v))) * w` (device `expf` evaluated on the device over every BF16 pattern);
//!     down epilogue `bf16(sum)`; padding slots +0; sentinel tiles untouched (0xA5 poison).
//!   * producers: FWHT256 (gate) / FWHT128 (down) in the kernels' exact f32 operation order.
//!
//! The RAW build (`#define EP8_RAW_F32 1`, renamed symbols, module `qwen4_moe_ep_a8_raw_gfx12`) stores the F32
//! fold sums: compared bit-exact with the CPU fold sums. The normal build is driven only through the PUBLIC
//! `Gpu` launchers. Activations are SYNTHETIC F32 (gaussian + outlier channels); weights are synthetic (section
//! 1, 3) or real Flash-Next experts (section 2).
//!
//! Tests (all `#[ignore]`, `--test-threads 1`, JSON summary into `EP_A8_OUT`):
//!   s1_synthetic  s2_real_layer0  s2_real_layer17  s2_real_layer47  s3_negative_controls
//! Pure unit tests (`unit_*`) need no card.

use hip_bridge::KernargBlob;
use rayon::prelude::*;
use rdna_compute::kernels::QWEN4_MOE_EP_A8_SRC;
use rdna_compute::qwen4_moe_ep_a8::{
    ep_a8_down_reserve_k, ep_a8_down_symbol, ep_a8_gate_up_symbol, ep_a8_select_nt,
};
use rdna_compute::{DType, Gpu, GpuTensor, Int8MmqPrepared};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const MODEL_DEFAULT: &str =
    "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts";
const ROUTES_DIR: &str = "/home/kaden/qcal/release-0.4.5/parity/kernel-runs/k0/routes";
const GUARD: usize = 512;
const POISON: u8 = 0xA5;
const RAW_MODULE: &str = "qwen4_moe_ep_a8_raw_gfx12";

// ---------------------------------------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn u32(&mut self) -> u32 {
        (self.u64() >> 32) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u64() % (n as u64)) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.u32() >> 8) as f32 / 16_777_216.0
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.u64() >> 11) as f64 + 1.0) / 9_007_199_254_740_993.0;
        let u2 = (self.u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// f16 / bf16
// ---------------------------------------------------------------------------------------------------------

fn f16_to_f32(b: u16) -> f32 {
    let s = (b >> 15) & 1;
    let e = ((b >> 10) & 0x1F) as i32;
    let m = (b & 0x3FF) as u32;
    let v = if e == 0 {
        (m as f32) * f32::from_bits(0x3380_0000) // 2^-24
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits((((e + 112) as u32) << 23) | (m << 13))
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        if rem > mid || (rem == mid && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// RNE BF16 bits, non-finite passthrough of the high half (the kernel's `ep8_bf16_bits`).
fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if !v.is_finite() {
        return (u >> 16) as u16;
    }
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}
fn bf16_rt(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    f32::from_bits((bf16_bits(v) as u32) << 16)
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `zp` f16 bits of a symmetric header: the f16 of the f32 `-8 * sc`.
fn zp_bits(sc: u16) -> u16 {
    f32_to_f16(-8.0 * f16_to_f32(sc))
}

// ---------------------------------------------------------------------------------------------------------
// Weights: logical form + the two byte layouts
// ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    GateUp,
    Down,
}
impl Kind {
    fn group_cols(self) -> usize {
        match self {
            Kind::GateUp => 256,
            Kind::Down => 128,
        }
    }
    fn group_bytes(self) -> usize {
        match self {
            Kind::GateUp => 136,
            Kind::Down => 68,
        }
    }
    fn hdr_bytes(self) -> usize {
        match self {
            Kind::GateUp => 8,
            Kind::Down => 4,
        }
    }
}

/// Logical `[rows, k]` expert tensor: `w = u - 8` as int8 (row-major) and the f16 scale bits per (row, K128).
#[derive(Clone, PartialEq, Debug)]
struct Wlog {
    rows: usize,
    k: usize,
    w: Vec<i8>,
    sc: Vec<u16>,
}
impl Wlog {
    fn nk(&self) -> usize {
        self.k / 128
    }
}

fn put16(out: &mut [u8], pos: usize, v: u16) {
    out[pos..pos + 2].copy_from_slice(&v.to_le_bytes());
}
fn get16(b: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([b[pos], b[pos + 1]])
}

fn shipped_bytes(kind: Kind, rows: usize, k: usize) -> usize {
    rows * (k / kind.group_cols()) * kind.group_bytes()
}
fn planar_bytes(rows: usize, k: usize) -> usize {
    rows * k / 2 + 2 * rows * (k / 128)
}

fn pack_shipped(kind: Kind, w: &Wlog) -> Vec<u8> {
    let (gc, gb, hb) = (kind.group_cols(), kind.group_bytes(), kind.hdr_bytes());
    let ng = w.k / gc;
    let nk = w.nk();
    let mut out = vec![0u8; w.rows * ng * gb];
    for r in 0..w.rows {
        for g in 0..ng {
            let base = (r * ng + g) * gb;
            match kind {
                Kind::GateUp => {
                    for hh in 0..2 {
                        let sc = w.sc[r * nk + 2 * g + hh];
                        put16(&mut out, base + 4 * hh, sc);
                        put16(&mut out, base + 4 * hh + 2, zp_bits(sc));
                    }
                }
                Kind::Down => {
                    let sc = w.sc[r * nk + g];
                    put16(&mut out, base, sc);
                    put16(&mut out, base + 2, zp_bits(sc));
                }
            }
            for j in 0..gc / 2 {
                let c = g * gc + 2 * j;
                let lo = (w.w[r * w.k + c] + 8) as u8;
                let hi = (w.w[r * w.k + c + 1] + 8) as u8;
                out[base + hb + j] = lo | (hi << 4);
            }
        }
    }
    out
}

/// Inverse of `pack_shipped`; also returns the number of scale halves whose stored zero is not `-8 * sc`.
fn unpack_shipped(kind: Kind, rows: usize, k: usize, b: &[u8]) -> (Wlog, u64) {
    let (gc, gb, hb) = (kind.group_cols(), kind.group_bytes(), kind.hdr_bytes());
    let ng = k / gc;
    let nk = k / 128;
    assert_eq!(b.len(), rows * ng * gb);
    let mut w = Wlog {
        rows,
        k,
        w: vec![0i8; rows * k],
        sc: vec![0u16; rows * nk],
    };
    let mut asym = 0u64;
    for r in 0..rows {
        for g in 0..ng {
            let base = (r * ng + g) * gb;
            match kind {
                Kind::GateUp => {
                    for hh in 0..2 {
                        let sc = get16(b, base + 4 * hh);
                        let zp = get16(b, base + 4 * hh + 2);
                        asym += (f16_to_f32(zp) != -8.0 * f16_to_f32(sc)) as u64;
                        w.sc[r * nk + 2 * g + hh] = sc;
                    }
                }
                Kind::Down => {
                    let sc = get16(b, base);
                    let zp = get16(b, base + 2);
                    asym += (f16_to_f32(zp) != -8.0 * f16_to_f32(sc)) as u64;
                    w.sc[r * nk + g] = sc;
                }
            }
            for j in 0..gc / 2 {
                let c = g * gc + 2 * j;
                let byte = b[base + hb + j];
                w.w[r * k + c] = (byte & 15) as i8 - 8;
                w.w[r * k + c + 1] = (byte >> 4) as i8 - 8;
            }
        }
    }
    (w, asym)
}

/// Direct logical -> planar (`[codes M*K/2][scales M*(K/128) LE f16]`).
fn pack_planar(w: &Wlog) -> Vec<u8> {
    let codes = w.rows * w.k / 2;
    let nk = w.nk();
    let mut out = vec![0u8; planar_bytes(w.rows, w.k)];
    for r in 0..w.rows {
        for j in 0..w.k / 2 {
            let lo = (w.w[r * w.k + 2 * j] + 8) as u8;
            let hi = (w.w[r * w.k + 2 * j + 1] + 8) as u8;
            out[r * (w.k / 2) + j] = lo | (hi << 4);
        }
        for h in 0..nk {
            put16(&mut out, codes + 2 * (r * nk + h), w.sc[r * nk + h]);
        }
    }
    out
}

/// Byte-level shipped -> planar repack (re-implementation of `hipfire-runtime::mq4_planar::repack_tensor`).
fn repack_shipped_to_planar(kind: Kind, rows: usize, k: usize, s: &[u8]) -> Vec<u8> {
    let (gc, gb, hb) = (kind.group_cols(), kind.group_bytes(), kind.hdr_bytes());
    let ng = k / gc;
    let nk = k / 128;
    let codes = rows * k / 2;
    let mut out = vec![0u8; planar_bytes(rows, k)];
    for r in 0..rows {
        for g in 0..ng {
            let base = (r * ng + g) * gb;
            let nib = gc / 2;
            out[r * (k / 2) + g * nib..r * (k / 2) + (g + 1) * nib].copy_from_slice(&s[base + hb..base + hb + nib]);
            match kind {
                Kind::GateUp => {
                    for hh in 0..2 {
                        let sc = get16(s, base + 4 * hh);
                        put16(&mut out, codes + 2 * (r * nk + 2 * g + hh), sc);
                    }
                }
                Kind::Down => {
                    let sc = get16(s, base);
                    put16(&mut out, codes + 2 * (r * nk + g), sc);
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum WProf {
    Random,
    AllZero,
    AllFifteen,
    ExtremeScales,
    MixedSign,
    Adversarial,
    HugeScales,
}
const PROFILES: [WProf; 12] = [
    WProf::Random,
    WProf::AllZero,
    WProf::AllFifteen,
    WProf::ExtremeScales,
    WProf::MixedSign,
    WProf::Adversarial,
    WProf::HugeScales,
    WProf::Random,
    WProf::Random,
    WProf::ExtremeScales,
    WProf::MixedSign,
    WProf::Random,
];
const EXTREME_SC: [u16; 12] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00, 0x2E66, 0x0200,
];

fn gen_wlog(rng: &mut Rng, rows: usize, k: usize, prof: WProf) -> Wlog {
    let nk = k / 128;
    let mut w = vec![0i8; rows * k];
    let mut sc = vec![0u16; rows * nk];
    for r in 0..rows {
        for c in 0..k {
            let u: i8 = match prof {
                WProf::AllZero => 0,
                WProf::AllFifteen => 15,
                WProf::MixedSign => {
                    if (r + c) % 2 == 0 {
                        0
                    } else {
                        15
                    }
                }
                _ => (rng.u32() & 15) as i8,
            };
            w[r * k + c] = u - 8;
        }
        for h in 0..nk {
            sc[r * nk + h] = match prof {
                WProf::ExtremeScales => EXTREME_SC[rng.below(EXTREME_SC.len())],
                WProf::HugeScales => 0x7BFF,
                WProf::MixedSign => {
                    let v = f32_to_f16(rng.unit() * 0.04 + 0.002);
                    if rng.u32() & 1 == 1 {
                        v | 0x8000
                    } else {
                        v
                    }
                }
                WProf::Adversarial => {
                    let e = rng.below(25) as i32 - 12;
                    let v = f32_to_f16(2f32.powi(e) * (1.0 + rng.unit()));
                    if rng.u32() & 1 == 1 {
                        v | 0x8000
                    } else {
                        v
                    }
                }
                _ => f32_to_f16(rng.unit() * 0.04 + 0.002),
            };
        }
    }
    Wlog { rows, k, w, sc }
}

// ---------------------------------------------------------------------------------------------------------
// Activation blocks (block_i8_128) and sets
// ---------------------------------------------------------------------------------------------------------

/// CPU quantizer of one K128 block (`div` = 127 contract; 128 for the wrong-scale negative control).
fn quant128(x: &[f32], div: f32, q: &mut [i8]) -> (f32, i32) {
    debug_assert!(x.len() == 128 && q.len() == 128);
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    if amax == 0.0 {
        q.fill(0);
        return (0.0, 0);
    }
    let d = amax / div;
    let mut s = 0i32;
    for i in 0..128 {
        let r = (x[i] / d).round_ties_even();
        let c = r.max(-127.0).min(127.0); // fminf(fmaxf(r, -127), 127): NaN -> -127
        q[i] = c as i8;
        s += c as i32;
    }
    (d, s)
}

/// 136-byte block: `[f32 d][i32 s][128 i8]` in the nibble-pair order.
fn encode_blk(d: f32, s: i32, q: &[i8], out: &mut [u8]) {
    out[0..4].copy_from_slice(&d.to_le_bytes());
    out[4..8].copy_from_slice(&s.to_le_bytes());
    for g in 0..16 {
        for j in 0..4 {
            out[8 + 8 * g + j] = q[8 * g + 2 * j] as u8;
            out[8 + 8 * g + 4 + j] = q[8 * g + 2 * j + 1] as u8;
        }
    }
}
fn decode_blk(b: &[u8]) -> (f32, i32, [i8; 128]) {
    let d = f32::from_le_bytes(b[0..4].try_into().unwrap());
    let s = i32::from_le_bytes(b[4..8].try_into().unwrap());
    let mut q = [0i8; 128];
    for g in 0..16 {
        for j in 0..4 {
            q[8 * g + 2 * j] = b[8 + 8 * g + j] as i8;
            q[8 * g + 2 * j + 1] = b[8 + 8 * g + 4 + j] as i8;
        }
    }
    (d, s, q)
}

/// `n` rows (tokens or flat slots) of `nk` K128 blocks, token-major.
#[derive(Clone)]
struct XSet {
    n: usize,
    nk: usize,
    q: Vec<i8>,
    d: Vec<f32>,
    s: Vec<i32>,
}
impl XSet {
    fn new(n: usize, nk: usize) -> XSet {
        XSet {
            n,
            nk,
            q: vec![0; n * nk * 128],
            d: vec![0.0; n * nk],
            s: vec![0; n * nk],
        }
    }
    fn set(&mut self, i: usize, h: usize, d: f32, s: i32, q: &[i8]) {
        let b = i * self.nk + h;
        self.d[b] = d;
        self.s[b] = s;
        self.q[b * 128..(b + 1) * 128].copy_from_slice(q);
    }
    /// Epoch-major `[res_nk, n]` sidecar bytes (136 B per block; epochs `>= nk` zero).
    fn to_bytes(&self, res_nk: usize) -> Vec<u8> {
        let mut out = vec![0u8; res_nk * self.n * 136];
        for h in 0..self.nk {
            for i in 0..self.n {
                let b = i * self.nk + h;
                let off = (h * self.n + i) * 136;
                encode_blk(self.d[b], self.s[b], &self.q[b * 128..(b + 1) * 128], &mut out[off..off + 136]);
            }
        }
        out
    }
}

/// Sidecar compare restricted to rows `present[i]`; returns (bytes, bad, first (idx, exp, got)).
fn cmp_sidecar(dev: &[u8], cpu: &[u8], n: usize, nk: usize, present: &[bool]) -> Cmp {
    let mut c = Cmp::default();
    for h in 0..nk {
        for i in 0..n {
            if !present[i] {
                continue;
            }
            let off = (h * n + i) * 136;
            let (a, b) = (&cpu[off..off + 136], &dev[off..off + 136]);
            c.bytes += 136;
            if a != b {
                for j in 0..136 {
                    if a[j] != b[j] {
                        c.bad += 1;
                        if c.first.is_none() {
                            c.first = Some(((off + j) as u64, a[j], b[j]));
                        }
                    }
                }
            }
        }
    }
    c
}

/// Hand-built block kinds (edge cases the quantizer cannot produce): returns (d, q).
fn gen_edge_block(rng: &mut Rng, q: &mut [i8; 128]) -> f32 {
    let pick = rng.below(100);
    let rand_d = |rng: &mut Rng| -> f32 {
        const DS: [f32; 6] = [3.7e-5, 0.011, 0.5, 1.0, 17.0, 90.0];
        DS[rng.below(DS.len())]
    };
    match pick {
        0..=54 => {
            let sig = [0.01f32, 1.0, 30.0][rng.below(3)];
            let mut x = [0f32; 128];
            for v in x.iter_mut() {
                *v = rng.gauss() * sig;
            }
            if rng.u32() & 7 == 0 {
                x[rng.below(128)] *= 40.0;
            }
            let (d, _) = quant128(&x, 127.0, q);
            d
        }
        55..=59 => {
            q.fill(0);
            0.0
        }
        60..=64 => {
            q.fill(127);
            rand_d(rng)
        }
        65..=69 => {
            q.fill(-127);
            rand_d(rng)
        }
        70..=74 => {
            q.fill(-128);
            rand_d(rng)
        }
        75..=79 => {
            for (i, v) in q.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 127 } else { -128 };
            }
            rand_d(rng)
        }
        80..=84 => {
            q.fill(0);
            q[rng.below(128)] = if rng.u32() & 1 == 1 { 127 } else { -128 };
            rand_d(rng)
        }
        85..=89 => {
            for v in q.iter_mut() {
                *v = (rng.u32() & 255) as u8 as i8;
            }
            rand_d(rng)
        }
        90..=94 => {
            for v in q.iter_mut() {
                *v = (rng.u32() & 255) as u8 as i8;
            }
            0.0
        }
        _ => {
            for v in q.iter_mut() {
                *v = if rng.u32() & 1 == 1 { 127 } else { -127 };
            }
            [1e-30f32, 1e-40, 90.0, 1e-8][rng.below(4)]
        }
    }
}

fn gen_xset_hand(rng: &mut Rng, n: usize, nk: usize) -> XSet {
    let mut x = XSet::new(n, nk);
    for i in 0..n {
        for h in 0..nk {
            let mut q = [0i8; 128];
            let d = gen_edge_block(rng, &mut q);
            let s: i32 = q.iter().map(|&v| v as i32).sum();
            x.set(i, h, d, s, &q);
        }
    }
    x
}

// ---------------------------------------------------------------------------------------------------------
// Producers (CPU)
// ---------------------------------------------------------------------------------------------------------

/// In-place radix-2 Walsh-Hadamard, lower index gets `a + b`, upper `a - b` (strides 1, 2, ...).
fn fwht(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// `mq_rotate_x_i4.hip` body: `x * signs1`, FWHT256, `* 0.0625 * signs2`.
fn rotate256_row(x: &[f32], s1: &[f32], s2: &[f32], out: &mut [f32]) {
    let mut v = [0f32; 256];
    for i in 0..256 {
        v[i] = x[i] * s1[i];
    }
    fwht(&mut v);
    for i in 0..256 {
        out[i] = v[i] * 0.0625f32 * s2[i];
    }
}

/// Gate producer: returns (A8 sidecar set `[tokens]`, rotated F32 rows).
fn produce_gate(x: &[f32], k: usize, rows: usize, s1: &[f32], s2: &[f32], div: f32) -> (XSet, Vec<f32>) {
    let nk = k / 128;
    let mut xr = vec![0f32; rows * k];
    xr.par_chunks_mut(k).zip(x.par_chunks(k)).for_each(|(o, i)| {
        for g in 0..k / 256 {
            rotate256_row(&i[g * 256..(g + 1) * 256], s1, s2, &mut o[g * 256..(g + 1) * 256]);
        }
    });
    let mut set = XSet::new(rows, nk);
    let blocks: Vec<(f32, i32, [i8; 128])> = (0..rows * nk)
        .into_par_iter()
        .map(|b| {
            let mut q = [0i8; 128];
            let (d, s) = quant128(&xr[b * 128..(b + 1) * 128], div, &mut q);
            (d, s, q)
        })
        .collect();
    for (b, (d, s, q)) in blocks.iter().enumerate() {
        set.set(b / nk, b % nk, *d, *s, q);
    }
    (set, xr)
}

/// Down producer: grouped BF16 rows `h [grouped_rows, k]` -> sidecar set over flat slots (`present[slot]`).
fn produce_down(
    h: &[u16],
    sorted: &[i32],
    k: usize,
    l: usize,
    s1: &[f32],
    s2: &[f32],
    div: f32,
) -> (XSet, Vec<bool>) {
    let nk = k / 128;
    let mut present = vec![false; l];
    let work: Vec<(usize, usize)> = sorted
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= 0)
        .map(|(p, &s)| (p, s as usize))
        .collect();
    for &(_, s) in &work {
        present[s] = true;
    }
    let blocks: Vec<(usize, usize, f32, i32, [i8; 128])> = work
        .par_iter()
        .flat_map_iter(|&(p, slot)| {
            (0..nk).map(move |g| {
                let mut v = [0f32; 128];
                for i in 0..128 {
                    v[i] = bf16_to_f32(h[p * k + g * 128 + i]) * s1[i];
                }
                fwht(&mut v);
                let mut xv = [0f32; 128];
                for i in 0..128 {
                    xv[i] = v[i] * 0.0883883476f32 * s2[i];
                }
                let mut q = [0i8; 128];
                let (d, s) = quant128(&xv, div, &mut q);
                (slot, g, d, s, q)
            })
        })
        .collect();
    let mut set = XSet::new(l, nk);
    for (slot, g, d, s, q) in &blocks {
        set.set(*slot, *g, *d, *s, q);
    }
    (set, present)
}

// ---------------------------------------------------------------------------------------------------------
// Routing / layouts
// ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Layout {
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    g: usize,
    total_slots: usize,
}
impl Layout {
    fn tile_valid(&self, t: usize, n_exp: usize) -> bool {
        let e = self.tiles[t];
        e >= 0 && (e as usize) < n_exp
    }
}

/// A4-chain contract: per expert ascending, its flat slots (ascending) padded to 16 per expert; `-1` ids dropped.
/// `trailing` sentinel tiles (id -1, sorted -1) follow.
fn build_pad16(ids: &[i32], n_exp: usize, trailing: usize) -> Layout {
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); n_exp];
    for (s, &e) in ids.iter().enumerate() {
        if e >= 0 {
            per[e as usize].push(s as i32);
        }
    }
    let mut tiles = Vec::new();
    let mut sorted = Vec::new();
    for (e, v) in per.iter().enumerate() {
        for chunk in v.chunks(16) {
            tiles.push(e as i32);
            sorted.extend_from_slice(chunk);
            sorted.extend(std::iter::repeat(-1).take(16 - chunk.len()));
        }
    }
    for _ in 0..trailing {
        tiles.push(-1);
        sorted.extend(std::iter::repeat(-1).take(16));
    }
    Layout {
        g: tiles.len() * 16,
        tiles,
        sorted,
        total_slots: ids.len(),
    }
}

/// One tile of a synthetic plan.
#[derive(Clone, Copy, Debug)]
enum Ts {
    /// expert id, live-lane mask
    Run(i32, u16),
    /// id -1, all padding (kernel returns before any store)
    Sentinel,
    /// id -1 (owner mask) but lanes hold real slots
    Masked(u16),
}
const FULL: u16 = 0xFFFF;

/// Synthetic layout: live lanes get a random permutation of `0..L` as flat slots.
fn build_plan(specs: &[Ts], rng: &mut Rng) -> Layout {
    let mut tiles = Vec::new();
    let mut live_pos = Vec::new();
    for (t, s) in specs.iter().enumerate() {
        let mask = match *s {
            Ts::Run(e, m) => {
                tiles.push(e);
                m
            }
            Ts::Sentinel => {
                tiles.push(-1);
                0
            }
            Ts::Masked(m) => {
                tiles.push(-1);
                m
            }
        };
        for lane in 0..16 {
            if mask >> lane & 1 == 1 {
                live_pos.push(t * 16 + lane);
            }
        }
    }
    let l = live_pos.len();
    let mut perm: Vec<i32> = (0..l as i32).collect();
    rng.shuffle(&mut perm);
    let mut sorted = vec![-1i32; specs.len() * 16];
    for (i, &p) in live_pos.iter().enumerate() {
        sorted[p] = perm[i];
    }
    Layout {
        tiles,
        sorted,
        g: specs.len() * 16,
        total_slots: l,
    }
}

fn plan_one() -> Vec<Ts> {
    vec![Ts::Run(3, 0x01FF)]
}
fn plan_two() -> Vec<Ts> {
    vec![Ts::Run(2, FULL), Ts::Run(2, FULL)]
}
/// Odd-start runs: the NT2 boundary cuts runs of expert 1 (tiles 1..=3) and expert 4 (tiles 9..=13).
fn plan_cut() -> Vec<Ts> {
    vec![
        Ts::Run(0, FULL),
        Ts::Run(1, FULL),
        Ts::Run(1, FULL),
        Ts::Run(1, 0x7FFF),
        Ts::Run(2, 0x00FF),
        Ts::Run(2, FULL),
        Ts::Run(3, FULL),
        Ts::Sentinel,
        Ts::Run(5, 0xAAAA),
        Ts::Run(4, FULL),
        Ts::Run(4, FULL),
        Ts::Run(4, FULL),
        Ts::Run(4, FULL),
        Ts::Run(4, 0x0001),
    ]
}
/// Runs of 1, 2, 3, 5, 9 tiles; sentinels, a masked tile, an all-padding tile, ragged/holey masks.
fn plan_runs() -> Vec<Ts> {
    let mut v = vec![Ts::Run(0, FULL)];
    v.push(Ts::Run(1, FULL)); // run 1
    for _ in 0..2 {
        v.push(Ts::Run(2, FULL)); // run 2
    }
    v.push(Ts::Sentinel);
    for i in 0..3 {
        v.push(Ts::Run(3, if i == 2 { 0x1FF3 } else { FULL })); // run 3
    }
    for i in 0..5 {
        v.push(Ts::Run(4, if i == 4 { 0x0007 } else { FULL })); // run 5
    }
    v.push(Ts::Masked(FULL));
    v.push(Ts::Sentinel);
    v.push(Ts::Sentinel);
    for i in 0..9 {
        v.push(Ts::Run(5, if i == 4 { 0x5555 } else if i == 8 { 0x8000 } else { FULL })); // run 9
    }
    v.push(Ts::Run(6, 0)); // all-padding tile of a live expert
    v.push(Ts::Run(7, 0x000F));
    v.push(Ts::Masked(0x00FF));
    v.push(Ts::Run(8, FULL));
    v.push(Ts::Run(8, FULL));
    v.push(Ts::Run(9, 0x0001));
    v.push(Ts::Run(10, FULL));
    v.push(Ts::Run(11, 0xFFFE));
    v.push(Ts::Sentinel);
    v
}
fn plan_big(rng: &mut Rng, tiles: usize, n_e: usize) -> Vec<Ts> {
    let mut v = Vec::new();
    while v.len() < tiles {
        let r = rng.below(100);
        if r < 5 {
            v.push(Ts::Sentinel);
        } else if r < 8 {
            v.push(Ts::Masked(if rng.below(2) == 0 { FULL } else { (rng.u32() & 0xFFFF) as u16 }));
        } else if r < 10 {
            v.push(Ts::Run(rng.below(n_e) as i32, 0));
        } else {
            let e = rng.below(n_e) as i32;
            let len = [1usize, 2, 3, 5, 9][rng.below(5)];
            for _ in 0..len {
                if v.len() >= tiles {
                    break;
                }
                let mask = if rng.below(10) < 8 { FULL } else { (rng.u32() & 0xFFFF) as u16 };
                v.push(Ts::Run(e, mask));
            }
        }
    }
    v
}
fn plan_hot(rng: &mut Rng, tiles: usize, n_e: usize) -> Vec<Ts> {
    let mut v = Vec::new();
    for _ in 0..tiles * 7 / 10 {
        v.push(Ts::Run(3, FULL));
    }
    while v.len() < tiles {
        v.push(Ts::Run(rng.below(n_e) as i32, if rng.below(3) == 0 { 0x00F0 } else { FULL }));
    }
    v
}

// ---------------------------------------------------------------------------------------------------------
// CPU GEMM reference
// ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fold {
    Asc,
    Desc,
}

#[inline]
fn dot128(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Raw fold sum of weight row `r` against X row `xi`.
fn fold_row(w: &Wlog, r: usize, x: &XSet, xi: usize, fold: Fold) -> f32 {
    let nk = w.nk();
    let mut sum = 0.0f32;
    for step in 0..nk {
        let h = match fold {
            Fold::Asc => step,
            Fold::Desc => nk - 1 - step,
        };
        let wr = &w.w[r * w.k + h * 128..r * w.k + (h + 1) * 128];
        let b = xi * nk + h;
        let c = dot128(wr, &x.q[b * 128..(b + 1) * 128]);
        let sd = f16_to_f32(w.sc[r * nk + h]) * x.d[b];
        sum = sd.mul_add(c as f32, sum);
    }
    sum
}

/// `[G, m]` raw F32 fold sums (column = weight row); padding / sentinel positions 0.
fn ref_raw(lay: &Layout, w: &[Wlog], x: &XSet, div: usize, m: usize, fold: Fold) -> Vec<f32> {
    let n_exp = w.len();
    let mut out = vec![0f32; lay.g * m];
    out.par_chunks_mut(16 * m).enumerate().for_each(|(t, chunk)| {
        let e = lay.tiles[t];
        if e < 0 || e as usize >= n_exp {
            return;
        }
        let we = &w[e as usize];
        let live: Vec<(usize, usize)> = (0..16)
            .filter_map(|lr| {
                let s = lay.sorted[t * 16 + lr];
                (s >= 0).then(|| (lr, s as usize / div))
            })
            .collect();
        for r in 0..m {
            for &(lr, xi) in &live {
                chunk[lr * m + r] = fold_row(we, r, x, xi, fold);
            }
        }
    });
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn live_pos(lay: &Layout, p: usize, n_exp: usize) -> Option<bool> {
    // None: sentinel tile (poison); Some(live)
    if !lay.tile_valid(p / 16, n_exp) {
        return None;
    }
    Some(lay.sorted[p] >= 0)
}

/// Expected raw F32 bytes `[G, m]`: poison in sentinel tiles.
fn exp_raw_bytes(lay: &Layout, sums: &[f32], m: usize, n_exp: usize) -> Vec<u8> {
    let mut out = vec![POISON; lay.g * m * 4];
    for p in 0..lay.g {
        if lay.tile_valid(p / 16, n_exp) {
            for c in 0..m {
                out[(p * m + c) * 4..(p * m + c) * 4 + 4].copy_from_slice(&sums[p * m + c].to_le_bytes());
            }
        }
    }
    out
}

fn bf16_bytes_into(out: &mut [u8], idx: usize, v: u16) {
    out[idx * 2..idx * 2 + 2].copy_from_slice(&v.to_le_bytes());
}

/// Expected BF16 SwiGLU bytes `[G, m/2]` from raw sums (gate col c, up col m/2 + c); `tab[bits]` = device `expf(-v)`.
fn exp_bf16_gate(lay: &Layout, sums: &[f32], m: usize, n_exp: usize, tab: &[f32]) -> Vec<u8> {
    let mi = m / 2;
    let mut out = vec![POISON; lay.g * mi * 2];
    for p in 0..lay.g {
        match live_pos(lay, p, n_exp) {
            None => {}
            Some(false) => {
                for c in 0..mi {
                    bf16_bytes_into(&mut out, p * mi + c, 0);
                }
            }
            Some(true) => {
                for c in 0..mi {
                    let v = bf16_rt(sums[p * m + c]);
                    let w = bf16_rt(sums[p * m + mi + c]);
                    assert!(!v.is_nan() && !w.is_nan(), "NaN fold sum in the oracle inputs");
                    let e = tab[(v.to_bits() >> 16) as usize];
                    let r = bf16_rt((v / (1.0f32 + e)) * w);
                    bf16_bytes_into(&mut out, p * mi + c, bf16_bits(r));
                }
            }
        }
    }
    out
}

fn exp_bf16_down(lay: &Layout, sums: &[f32], m: usize, n_exp: usize) -> Vec<u8> {
    let mut out = vec![POISON; lay.g * m * 2];
    for p in 0..lay.g {
        match live_pos(lay, p, n_exp) {
            None => {}
            Some(false) => {
                for c in 0..m {
                    bf16_bytes_into(&mut out, p * m + c, 0);
                }
            }
            Some(true) => {
                for c in 0..m {
                    bf16_bytes_into(&mut out, p * m + c, bf16_bits(sums[p * m + c]));
                }
            }
        }
    }
    out
}

/// Gate and up halves of every row swapped (negative control).
fn swap_halves(sums: &[f32], g: usize, m: usize) -> Vec<f32> {
    let mi = m / 2;
    let mut out = sums.to_vec();
    for p in 0..g {
        for c in 0..mi {
            out.swap(p * m + c, p * m + mi + c);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// Comparator
// ---------------------------------------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
struct Cmp {
    bytes: u64,
    bad: u64,
    bad_guard: u64,
    first: Option<(u64, u8, u8)>,
}
impl Cmp {
    fn total(&self) -> u64 {
        self.bad + self.bad_guard
    }
}

fn cmp(exp: &[u8], got: &[u8]) -> Cmp {
    assert_eq!(exp.len(), got.len());
    let mut c = Cmp {
        bytes: exp.len() as u64,
        ..Default::default()
    };
    for (ci, (a, b)) in exp.chunks(4096).zip(got.chunks(4096)).enumerate() {
        if a == b {
            continue;
        }
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            if x != y {
                c.bad += 1;
                if c.first.is_none() {
                    c.first = Some(((ci * 4096 + j) as u64, *x, *y));
                }
            }
        }
    }
    c
}

/// `got` = Y bytes followed by the 0xA5 guard.
fn cmp_y(exp_y: &[u8], got: &[u8]) -> Cmp {
    let n = exp_y.len();
    assert!(got.len() >= n + 256, "need >= 256 guard bytes");
    let mut c = cmp(exp_y, &got[..n]);
    let guard = &got[n..];
    c.bytes += guard.len() as u64;
    for (j, &b) in guard.iter().enumerate() {
        if b != POISON {
            c.bad_guard += 1;
            if c.first.is_none() {
                c.first = Some(((n + j) as u64, POISON, b));
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------------------

struct Report {
    section: String,
    cases: Vec<Value>,
    controls: Vec<Value>,
    extra: serde_json::Map<String, Value>,
    bad: u64,
    cases_n: u64,
    bytes: u64,
    failed_controls: Vec<String>,
    nz_words: u64,
    total_words: u64,
}
impl Report {
    /// Non-vacuity statistic: how many expected fold sums are non-zero.
    fn note_sums(&mut self, sums: &[f32]) {
        self.total_words += sums.len() as u64;
        self.nz_words += sums.iter().filter(|v| **v != 0.0).count() as u64;
    }
    fn new(section: &str) -> Report {
        Report {
            section: section.to_string(),
            cases: vec![],
            controls: vec![],
            extra: serde_json::Map::new(),
            bad: 0,
            cases_n: 0,
            bytes: 0,
            failed_controls: vec![],
            nz_words: 0,
            total_words: 0,
        }
    }
    /// `coord` = (element bytes, columns) to turn the first mismatching byte into (row, col).
    fn add(&mut self, name: &str, c: &Cmp, coord: Option<(usize, usize)>) {
        self.cases_n += 1;
        self.bytes += c.bytes;
        self.bad += c.total();
        let first = c.first.map(|(i, e, g)| {
            let mut v = json!({"byte": i, "expected": e, "got": g});
            if let Some((eb, cols)) = coord {
                let el = i as usize / eb;
                v["row"] = json!(el / cols);
                v["col"] = json!(el % cols);
            }
            v
        });
        let status = if c.total() == 0 { "PASS" } else { "FAIL" };
        eprintln!(
            "CASE {status} {}: {} bytes, {} bad (+{} guard){}",
            name,
            c.bytes,
            c.bad,
            c.bad_guard,
            first.as_ref().map(|f| format!(" first={f}")).unwrap_or_default()
        );
        self.cases.push(json!({
            "name": name, "status": status, "bytes": c.bytes,
            "mismatching_bytes": c.bad, "mismatching_guard_bytes": c.bad_guard, "first": first
        }));
    }
    fn control(&mut self, name: &str, mismatches: u64) {
        let status = if mismatches > 0 { "PASS(detected)" } else { "FAIL(not detected)" };
        eprintln!("CONTROL {status} {name}: {mismatches} mismatching bytes");
        if mismatches == 0 {
            self.failed_controls.push(name.to_string());
        }
        self.controls.push(json!({"name": name, "mismatching_bytes": mismatches, "status": status}));
    }
    fn finish(self, gpu_arch: &str) {
        let dir = std::env::var("EP_A8_OUT").unwrap_or_else(|_| {
            format!(
                "{}/../../../parity/kernel-runs/e1pf/oracle-json",
                env!("CARGO_MANIFEST_DIR")
            )
        });
        std::fs::create_dir_all(&dir).ok();
        let verdict = if self.bad == 0 && self.failed_controls.is_empty() {
            "PASS"
        } else {
            "FAIL"
        };
        let doc = json!({
            "section": self.section,
            "verdict": verdict,
            "arch": gpu_arch,
            "mismatching_bytes_total": self.bad,
            "cases": self.cases_n,
            "bytes_compared": self.bytes,
            "failed_controls": self.failed_controls,
            "expected_fold_sum_words": {"total": self.total_words, "nonzero": self.nz_words},
            "controls": self.controls,
            "extra": Value::Object(self.extra),
            "case_list": self.cases,
            "activations": "synthetic F32 (gaussian, a few outlier channels / hand-built edge blocks); weights synthetic or real Flash-Next experts as stated per section",
            "arithmetic": "A8 chain vs independent CPU reference (integer/f32-exact); raw-F32 build bit-exact; BF16 build with device expf table",
        });
        let path = format!("{dir}/ep_a8_{}.json", self.section);
        let text = serde_json::to_string_pretty(&doc).unwrap();
        std::fs::write(&path, &text).ok();
        println!("EP_A8_SUMMARY {}", serde_json::to_string(&doc.as_object().map(|o| {
            let mut s = o.clone();
            s.remove("case_list");
            Value::Object(s)
        }).unwrap()).unwrap());
        println!("EP_A8_JSON_PATH {path}");
        assert_eq!(self.bad, 0, "{}: mismatching bytes", self.section);
        assert!(self.failed_controls.is_empty(), "negative controls not detected: {:?}", self.failed_controls);
    }
}

// ---------------------------------------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------------------------------------

struct Stack {
    data: GpuTensor,
    table: GpuTensor,
}
struct DevW {
    ship: Stack,
    plan: Stack,
}

struct Ctx {
    g: Gpu,
    s1: Vec<f32>,
    s2: Vec<f32>,
    s1_128: Vec<f32>,
    s2_128: Vec<f32>,
    exp_tab: Vec<f32>,
}

const EXPF_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void ep8_oracle_expf_table(float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536u) {
        const float v = __uint_as_float(i << 16);
        out[i] = expf(-v);
    }
}
"#;
const QUANT_SRC: &str = include_str!("../../../kernels/src/block_i8_128_quant.hip");

fn raw_src() -> String {
    let mut s = String::from("#define EP8_RAW_F32 1\n");
    for kind in ["gate_up", "gate_up_planar", "down", "down_planar"] {
        for nt in [1, 2] {
            s.push_str(&format!(
                "#define qwen4_moe_ep_a8_{kind}_nt{nt} ep8raw_{kind}_nt{nt}\n"
            ));
        }
    }
    s.push_str(QWEN4_MOE_EP_A8_SRC);
    s
}
fn raw_sym(sym: &str) -> String {
    sym.replace("qwen4_moe_ep_a8_", "ep8raw_")
}

fn init_ctx() -> Ctx {
    let mut g = Gpu::init().expect("Gpu::init");
    if g.arch != "gfx1201" {
        eprintln!("SKIP: the EP A8 oracle needs exact gfx1201, device is {}", g.arch);
        panic!("FAIL: not gfx1201 (device {}): SKIP counts as FAIL", g.arch);
    }
    g.ensure_mq_signs().expect("signs256");
    g.ensure_mq_signs_128().expect("signs128");
    let dl = |t: &Option<GpuTensor>, g: &Gpu| g.download_f32(t.as_ref().unwrap()).unwrap();
    let s1 = dl(&g.scratch.mq_signs1, &g);
    let s2 = dl(&g.scratch.mq_signs2, &g);
    let s1_128 = dl(&g.scratch.mq_signs1_128, &g);
    let s2_128 = dl(&g.scratch.mq_signs2_128, &g);
    assert!(s1.len() >= 256 && s2.len() >= 256 && s1_128.len() >= 128 && s2_128.len() >= 128);
    for v in s1[..256].iter().chain(&s2[..256]).chain(&s1_128[..128]).chain(&s2_128[..128]) {
        assert!(*v == 1.0 || *v == -1.0, "FWHT sign table is not +-1: {v}");
    }
    // Device expf(-v) over every BF16 pattern.
    g.ensure_kernel_public("ep8_oracle_expf", EXPF_SRC, "ep8_oracle_expf_table")
        .expect("expf table kernel");
    let t = g.alloc_tensor(&[65536], DType::F32).unwrap();
    let mut b = KernargBlob::new();
    b.push_ptr(t.buf.as_ptr());
    g.launch_kernel_blob("ep8_oracle_expf_table", [256, 1, 1], [256, 1, 1], 0, b.as_mut_slice())
        .expect("expf table launch");
    g.hip.device_synchronize().unwrap();
    let exp_tab = g.download_f32(&t).unwrap();
    g.free_tensor(t).unwrap();
    Ctx {
        g,
        s1: s1[..256].to_vec(),
        s2: s2[..256].to_vec(),
        s1_128: s1_128[..128].to_vec(),
        s2_128: s2_128[..128].to_vec(),
        exp_tab,
    }
}

fn up_i32(g: &Gpu, v: &[i32]) -> GpuTensor {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.upload_raw(&b, &[v.len()]).unwrap()
}
fn up_u16(g: &Gpu, v: &[u16]) -> GpuTensor {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.upload_raw(&b, &[v.len()]).unwrap()
}
fn alloc_poisoned(g: &mut Gpu, ybytes: usize) -> GpuTensor {
    let n = (ybytes + GUARD).div_ceil(4);
    let t = g.alloc_tensor(&[n], DType::F32).unwrap();
    g.hip.memset(&t.buf, POISON as i32, n * 4).unwrap();
    t
}
fn dl_bytes(g: &Gpu, t: &GpuTensor, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    g.hip.memcpy_dtoh(&mut v, &t.buf).unwrap();
    v
}
fn sync(g: &Gpu) {
    g.hip.device_synchronize().unwrap();
}

fn upload_stack(g: &Gpu, bytes: &[u8], n: usize, per: usize) -> Stack {
    assert_eq!(bytes.len(), n * per);
    let data = g.upload_raw(bytes, &[bytes.len()]).unwrap();
    let base = data.buf.as_ptr() as usize as u64;
    let table: Vec<u8> = (0..n).flat_map(|i| (base + (i * per) as u64).to_le_bytes()).collect();
    let table = g.upload_raw(&table, &[n * 8]).unwrap();
    Stack { data, table }
}

fn upload_w(g: &Gpu, kind: Kind, w: &[Wlog]) -> DevW {
    let (rows, k) = (w[0].rows, w[0].k);
    let sp = shipped_bytes(kind, rows, k);
    let pp = planar_bytes(rows, k);
    let mut ship = Vec::with_capacity(sp * w.len());
    let mut plan = Vec::with_capacity(pp * w.len());
    for e in w {
        ship.extend(pack_shipped(kind, e));
        plan.extend(pack_planar(e));
    }
    DevW {
        ship: upload_stack(g, &ship, w.len(), sp),
        plan: upload_stack(g, &plan, w.len(), pp),
    }
}

fn free_w(g: &mut Gpu, d: DevW) {
    for s in [d.ship, d.plan] {
        g.free_tensor(s.data).unwrap();
        g.free_tensor(s.table).unwrap();
    }
}

/// Write `bytes` as the live int8 sidecar of `(reserve_k, n)` and seal the handle.
fn put_sidecar(g: &mut Gpu, bytes: &[u8], reserve_k: usize, n: usize) -> Int8MmqPrepared {
    let res = g.reserve_int8_mmq(reserve_k, n).expect("reserve_int8_mmq");
    let buf = g.scratch.int8_mmq_x_scratch.as_ref().unwrap();
    assert!(bytes.len() <= buf.size());
    g.hip.memcpy_htod(buf, bytes).unwrap();
    Int8MmqPrepared::from_reservation(res)
}
fn dl_sidecar(g: &Gpu, nbytes: usize) -> Vec<u8> {
    let buf = g.scratch.int8_mmq_x_scratch.as_ref().unwrap();
    let mut v = vec![0u8; nbytes];
    g.hip.memcpy_dtoh(&mut v, buf).unwrap();
    v
}

#[derive(Clone, Copy)]
struct Shape {
    down: bool,
    m: usize,
    k: usize,
    div: usize,
}
struct LayDev {
    tiles: GpuTensor,
    sorted: GpuTensor,
}
fn upload_layout(g: &Gpu, lay: &Layout) -> LayDev {
    LayDev {
        tiles: up_i32(g, &lay.tiles),
        sorted: up_i32(g, &lay.sorted),
    }
}
fn free_layout(g: &mut Gpu, d: LayDev) {
    g.free_tensor(d.tiles).unwrap();
    g.free_tensor(d.sorted).unwrap();
}

fn ybytes(sh: Shape, lay: &Layout) -> usize {
    lay.g * if sh.down { sh.m } else { sh.m / 2 } * 2
}

/// PUBLIC BF16 launcher over the live sidecar; returns Y bytes + guard.
fn run_public(
    g: &mut Gpu,
    sh: Shape,
    lay: &Layout,
    ld: &LayDev,
    ptrs: &GpuTensor,
    xq: &Int8MmqPrepared,
    xsrc: usize,
    tiles_override: Option<&GpuTensor>,
) -> Vec<u8> {
    let yb = ybytes(sh, lay);
    let y = alloc_poisoned(g, yb);
    let tiles = tiles_override.unwrap_or(&ld.tiles);
    if sh.down {
        g.gemm_qwen4_moe_down_a8_sym(ptrs, tiles, &ld.sorted, xq, &y, sh.m, sh.k, sh.div, lay.g, xsrc)
            .expect("gemm_qwen4_moe_down_a8_sym");
    } else {
        g.gemm_qwen4_moe_gate_up_silu_a8_sym(ptrs, tiles, &ld.sorted, xq, &y, sh.m, sh.k, sh.div, lay.g, xsrc)
            .expect("gemm_qwen4_moe_gate_up_silu_a8_sym");
    }
    sync(g);
    let out = dl_bytes(g, &y, yb + GUARD);
    g.free_tensor(y).unwrap();
    out
}

/// RAW-F32 build with an explicit NT width over the live sidecar; returns F32 bytes + guard.
fn run_raw(
    g: &mut Gpu,
    planar: bool,
    nt: usize,
    sh: Shape,
    lay: &Layout,
    ld: &LayDev,
    ptrs: &GpuTensor,
    xsrc: usize,
    tiles_override: Option<&GpuTensor>,
) -> Vec<u8> {
    let sym = if sh.down {
        ep_a8_down_symbol(planar, nt)
    } else {
        ep_a8_gate_up_symbol(planar, nt)
    }
    .unwrap();
    let rs = raw_sym(sym);
    g.ensure_kernel_public(RAW_MODULE, &raw_src(), &rs).expect("raw build");
    let yb = lay.g * sh.m * 4;
    let y = alloc_poisoned(g, yb);
    let (_, xp) = g.scratch.int8_mmq_live();
    let tiles = tiles_override.unwrap_or(&ld.tiles);
    let nev = (ptrs.buf.size() / 8) as i32;
    let mut b = KernargBlob::new();
    b.push_ptr(ptrs.buf.as_ptr());
    b.push_ptr(tiles.buf.as_ptr());
    b.push_ptr(ld.sorted.buf.as_ptr());
    b.push_ptr(xp as *const _);
    b.push_ptr(y.buf.as_ptr());
    for v in [sh.m, sh.k, sh.div, lay.g, xsrc] {
        b.push_i32(v as i32);
    }
    b.push_i32(nev);
    assert_eq!(b.len(), 64, "kernarg ABI is 64 bytes");
    let rows = if sh.down { sh.m } else { sh.m / 2 };
    g.launch_kernel_blob(&rs, [(rows / 64) as u32, (lay.g / 16) as u32, 1], [128, 1, 1], 0, b.as_mut_slice())
        .expect("raw launch");
    sync(g);
    let out = dl_bytes(g, &y, yb + GUARD);
    g.free_tensor(y).unwrap();
    out
}

fn shipped_or_planar<'a>(w: &'a DevW, planar: bool) -> &'a GpuTensor {
    if planar {
        &w.plan.table
    } else {
        &w.ship.table
    }
}

/// CPU expectations of one (shape, layout, X) case.
struct Expect {
    sums: Vec<f32>,
    raw: Vec<u8>,
    bf16: Vec<u8>,
}
fn expect_case(ctx: &Ctx, sh: Shape, lay: &Layout, w: &[Wlog], x: &XSet, fold: Fold) -> Expect {
    let sums = ref_raw(lay, w, x, sh.div, sh.m, fold);
    let raw = exp_raw_bytes(lay, &sums, sh.m, w.len());
    let bf16 = if sh.down {
        exp_bf16_down(lay, &sums, sh.m, w.len())
    } else {
        exp_bf16_gate(lay, &sums, sh.m, w.len(), &ctx.exp_tab)
    };
    Expect { sums, raw, bf16 }
}

/// Run the public BF16 + both raw NT widths on both layouts over the live sidecar and compare.
fn compare_all(
    ctx: &mut Ctx,
    rep: &mut Report,
    tag: &str,
    sh: Shape,
    lay: &Layout,
    ld: &LayDev,
    dw: &DevW,
    xq: &Int8MmqPrepared,
    xsrc: usize,
    exp: &Expect,
) {
    rep.note_sums(&exp.sums);
    let cols_bf = if sh.down { sh.m } else { sh.m / 2 };
    for planar in [false, true] {
        ctx.g.set_mq4_experts_planar(planar);
        let lname = if planar { "planar" } else { "shipped" };
        let ptrs = shipped_or_planar(dw, planar);
        let got = run_public(&mut ctx.g, sh, lay, ld, ptrs, xq, xsrc, None);
        let nt_sel = ep_a8_select_nt(sh.down, lay.g);
        rep.add(
            &format!("{tag} {lname} bf16 public(nt{nt_sel})"),
            &cmp_y(&exp.bf16, &got),
            Some((2, cols_bf)),
        );
        for nt in [1usize, 2] {
            let got = run_raw(&mut ctx.g, planar, nt, sh, lay, ld, ptrs, xsrc, None);
            rep.add(&format!("{tag} {lname} raw_f32 nt{nt}"), &cmp_y(&exp.raw, &got), Some((4, sh.m)));
        }
    }
    ctx.g.set_mq4_experts_planar(false);
}

// ---------------------------------------------------------------------------------------------------------
// Synthetic activations for the producer variants
// ---------------------------------------------------------------------------------------------------------

/// F32 rows: gaussian + outlier channels with some zero / outlier / +-0.0 / wide-range rows (no denormals).
fn gen_x_f32(rng: &mut Rng, n: usize, k: usize, edgy: bool) -> Vec<f32> {
    let mut x = vec![0f32; n * k];
    let chans: Vec<usize> = (0..12).map(|_| rng.below(k)).collect();
    for t in 0..n {
        let row = &mut x[t * k..(t + 1) * k];
        let kind = if edgy { rng.below(100) } else { 99 };
        match kind {
            0..=5 => {} // zero row
            6..=9 => {
                for v in row.iter_mut() {
                    *v = rng.gauss() * 0.01;
                }
                row[rng.below(k)] = 1e30;
            }
            10..=13 => {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = if i % 3 == 0 { -0.0 } else { 0.0 };
                }
            }
            14..=17 => {
                for c in 0..k / 128 {
                    let sc = 10f32.powf(rng.unit() * 12.0 - 6.0);
                    for v in &mut row[c * 128..(c + 1) * 128] {
                        *v = rng.gauss() * sc;
                    }
                }
            }
            18..=21 => row.fill(1.0),
            _ => {
                let sig = [0.01f32, 1.0, 50.0][rng.below(3)];
                for v in row.iter_mut() {
                    *v = rng.gauss() * sig;
                }
                for &c in &chans {
                    row[c] *= 30.0;
                }
            }
        }
    }
    x
}

fn gen_h_bf16(rng: &mut Rng, rows: usize, k: usize) -> Vec<u16> {
    let mut h = vec![0u16; rows * k];
    for p in 0..rows {
        let row = &mut h[p * k..(p + 1) * k];
        match rng.below(20) {
            0 => {}
            1 => {
                for v in row.iter_mut() {
                    *v = bf16_bits(rng.gauss() * 0.02);
                }
                row[rng.below(k)] = bf16_bits(1e30);
            }
            2 => {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = if i % 2 == 0 { 0x8000 } else { 0x0000 };
                }
            }
            3 => {
                for c in 0..k / 128 {
                    let sc = 10f32.powf(rng.unit() * 8.0 - 4.0);
                    for v in &mut row[c * 128..(c + 1) * 128] {
                        *v = bf16_bits(rng.gauss() * sc);
                    }
                }
            }
            _ => {
                let sig = [0.05f32, 1.0, 8.0][rng.below(3)];
                for v in row.iter_mut() {
                    *v = bf16_bits(rng.gauss() * sig);
                }
            }
        }
    }
    h
}

// ---------------------------------------------------------------------------------------------------------
// Section 1: synthetic
// ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Geom {
    down: bool,
    m: usize,
    k: usize,
    tag: &'static str,
}

fn gen_experts(rng: &mut Rng, g: &Geom, n_e: usize) -> Vec<Wlog> {
    (0..n_e).map(|e| gen_wlog(rng, g.m, g.k, PROFILES[e % PROFILES.len()])).collect()
}

fn gemm_suite(ctx: &mut Ctx, rep: &mut Report, geom: Geom, seed: u64) {
    let n_e = 12;
    let mut rng = Rng::new(seed);
    let w = gen_experts(&mut rng, &geom, n_e);
    let kind = if geom.down { Kind::Down } else { Kind::GateUp };
    let dw = upload_w(&ctx.g, kind, &w);
    let sh = Shape {
        down: geom.down,
        m: geom.m,
        k: geom.k,
        div: if geom.down { 1 } else { 10 },
    };
    let big = geom.m * geom.k >= 1_000_000;
    let plans: Vec<(&str, Vec<Ts>)> = vec![
        ("one-tile", plan_one()),
        ("two-tiles", plan_two()),
        ("odd-cut", plan_cut()),
        ("runs-1-2-3-5-9", plan_runs()),
        ("big4096", plan_big(&mut rng, 256, n_e)),
        ("hot", plan_hot(&mut rng, if big { 96 } else { 160 }, n_e)),
    ];
    for (pname, specs) in plans {
        let lay = build_plan(&specs, &mut rng);
        let ld = upload_layout(&ctx.g, &lay);
        let l = lay.total_slots;
        let n_x = if geom.down { l } else { l.div_ceil(10) };
        let nk = geom.k / 128;
        let res_k = if geom.down { ep_a8_down_reserve_k(geom.k) } else { geom.k };
        let res_nk = res_k / 128;
        // (a) hand-built edge sidecar
        {
            let x = gen_xset_hand(&mut rng, n_x, nk);
            let exp = expect_case(ctx, sh, &lay, &w, &x, Fold::Asc);
            let xq = put_sidecar(&mut ctx.g, &x.to_bytes(res_nk), res_k, n_x);
            let tag = format!("S1 {} {pname} G={} hand-sidecar", geom.tag, lay.g);
            compare_all(ctx, rep, &tag, sh, &lay, &ld, &dw, &xq, n_x, &exp);
        }
        // (b) device-producer sidecar
        if pname == "runs-1-2-3-5-9" || pname == "big4096" || pname == "odd-cut" {
            let (x, xq, present, bytes_len) = if geom.down {
                let h = gen_h_bf16(&mut rng, lay.g, geom.k);
                let ht = up_u16(&ctx.g, &h);
                let xq = ctx
                    .g
                    .qwen4_moe_rotate128_i8(&ht, &ld.sorted, geom.k, lay.g, l)
                    .expect("rotate128_i8");
                sync(&ctx.g);
                let (xs, present) = produce_down(&h, &lay.sorted, geom.k, l, &ctx.s1_128, &ctx.s2_128, 127.0);
                ctx.g.free_tensor(ht).unwrap();
                (xs, xq, present, res_nk * l * 136)
            } else {
                let xf = gen_x_f32(&mut rng, n_x, geom.k, true);
                let xt = ctx.g.upload_f32(&xf, &[n_x * geom.k]).unwrap();
                let xq = ctx.g.qwen4_moe_rotate256_i8(&xt, geom.k, n_x).expect("rotate256_i8");
                sync(&ctx.g);
                let (xs, _) = produce_gate(&xf, geom.k, n_x, &ctx.s1, &ctx.s2, 127.0);
                ctx.g.free_tensor(xt).unwrap();
                (xs, xq, vec![true; n_x], nk * n_x * 136)
            };
            let dev = dl_sidecar(&ctx.g, bytes_len);
            let cpu = x.to_bytes(res_nk);
            let tag = format!("S1 {} {pname} G={} producer-sidecar", geom.tag, lay.g);
            rep.add(&format!("{tag} device-vs-cpu producer"), &cmp_sidecar(&dev, &cpu, n_x, nk, &present), None);
            let exp = expect_case(ctx, sh, &lay, &w, &x, Fold::Asc);
            compare_all(ctx, rep, &tag, sh, &lay, &ld, &dw, &xq, n_x, &exp);
        }
        free_layout(&mut ctx.g, ld);
    }
    free_w(&mut ctx.g, dw);
}

/// Standalone quantizer (the producers' shared `quantize_block_i8_128_wave`) on hand-picked edge blocks.
fn special_blocks(rng: &mut Rng) -> Vec<[f32; 128]> {
    let mut v: Vec<[f32; 128]> = Vec::new();
    v.push([0.0; 128]); // zero block
    v.push([-0.0; 128]);
    let mut mixed = [0.0f32; 128];
    for (i, x) in mixed.iter_mut().enumerate() {
        *x = if i % 2 == 0 { 0.0 } else { -0.0 };
    }
    v.push(mixed);
    let mut outlier = [1e-3f32; 128];
    outlier[77] = 3.0e38;
    v.push(outlier);
    let mut outlier2 = [0.0f32; 128];
    outlier2[0] = -3.4e38;
    outlier2[127] = 1.0;
    v.push(outlier2);
    v.push([1e-40; 128]); // denormal block
    v.push([1.4e-45; 128]); // smallest denormal
    let mut dn = [0.0f32; 128];
    for (i, x) in dn.iter_mut().enumerate() {
        *x = f32::from_bits(1 + (i as u32 * 977) % 0x7F_FFFF);
    }
    v.push(dn);
    let mut dn2 = [0.0f32; 128];
    dn2[3] = 1.4e-45;
    dn2[4] = -2.8e-45;
    v.push(dn2);
    // exact .5 ties: amax = 127 * s so d == s exactly (power of two), ties at odd multiples of s/2.
    for e in [0i32, 1, -3, 10, -20, 60] {
        let s = 2f32.powi(e);
        let mut b = [0.0f32; 128];
        b[0] = 127.0 * s;
        let ties = [0.5f32, 1.5, 2.5, 3.5, 4.5, 63.5, 125.5, 126.5, -0.5, -1.5, -2.5, -126.5, -125.5, 100.5, -101.5];
        for (i, t) in ties.iter().enumerate() {
            b[1 + i] = t * s;
        }
        v.push(b);
        let mut b2 = [0.0f32; 128];
        b2[5] = -127.0 * s;
        for (i, t) in ties.iter().enumerate() {
            b2[20 + i] = -t * s;
        }
        v.push(b2);
    }
    // ties after the divide rounds: amax not a power-of-two multiple
    for _ in 0..40 {
        let mut b = [0.0f32; 128];
        let d = 0.0137f32 * (1.0 + rng.unit());
        for x in b.iter_mut() {
            let q = (rng.below(255) as i32 - 127) as f32;
            *x = (q + 0.5) * d;
        }
        v.push(b);
    }
    // ordinary gaussian / wide range
    for _ in 0..120 {
        let sig = 10f32.powf(rng.unit() * 8.0 - 4.0);
        let mut b = [0.0f32; 128];
        for x in b.iter_mut() {
            *x = rng.gauss() * sig;
        }
        if rng.below(4) == 0 {
            b[rng.below(128)] *= 1000.0;
        }
        v.push(b);
    }
    v
}

fn quantizer_edge(ctx: &mut Ctx, rep: &mut Report) {
    let mut rng = Rng::new(0xE0_0017);
    let blocks = special_blocks(&mut rng);
    let k = 1024usize;
    let per_row = k / 128;
    let n = blocks.len().div_ceil(per_row);
    let mut x = vec![0f32; n * k];
    for (i, b) in blocks.iter().enumerate() {
        x[i * 128..(i + 1) * 128].copy_from_slice(b);
    }
    ctx.g
        .ensure_kernel_public("ep8_oracle_quant_i8", QUANT_SRC, "quantize_int8_mmq_ds128")
        .expect("quantizer kernel");
    let xt = ctx.g.upload_f32(&x, &[n * k]).unwrap();
    let ysz = per_row * n * 136;
    let y = alloc_poisoned(&mut ctx.g, ysz);
    let mut b = KernargBlob::new();
    b.push_ptr(xt.buf.as_ptr());
    b.push_ptr(y.buf.as_ptr());
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    let grid = (k.div_ceil(1024) * n) as u32;
    ctx.g
        .launch_kernel_blob("quantize_int8_mmq_ds128", [grid, 1, 1], [256, 1, 1], 0, b.as_mut_slice())
        .unwrap();
    sync(&ctx.g);
    let dev = dl_bytes(&ctx.g, &y, ysz);
    // CPU
    let mut cpu = vec![0u8; ysz];
    let mut present = vec![false; n];
    for t in 0..n {
        present[t] = true;
        for kb in 0..per_row {
            let bi = t * per_row + kb;
            let mut xb = [0f32; 128];
            xb.copy_from_slice(&x[bi * 128..(bi + 1) * 128]);
            let mut q = [0i8; 128];
            let (d, s) = quant128(&xb, 127.0, &mut q);
            let off = (kb * n + t) * 136;
            encode_blk(d, s, &q, &mut cpu[off..off + 136]);
        }
    }
    rep.add(
        &format!("S1 quantizer-edge ({} blocks: zero, +-0, outlier 3e38, denormal, exact .5 ties)", blocks.len()),
        &cmp_sidecar(&dev, &cpu, n, per_row, &present),
        None,
    );
    ctx.g.free_tensor(xt).unwrap();
    ctx.g.free_tensor(y).unwrap();
}

/// Gate / down producers on edge rows including denormal-small inputs.
fn producer_edge(ctx: &mut Ctx, rep: &mut Report) {
    let mut rng = Rng::new(0xE0_0042);
    let k = 2560usize;
    let rows = 96usize;
    let mut x = gen_x_f32(&mut rng, rows, k, true);
    // denormal-small / min-denormal / -0.0 / huge+tiny rows on top of the generator's rows
    for (t, kind) in [(1usize, 0), (2, 1), (3, 2), (4, 3), (5, 4)] {
        let row = &mut x[t * k..(t + 1) * k];
        for (i, v) in row.iter_mut().enumerate() {
            *v = match kind {
                0 => 1e-40 * (1.0 + (i % 7) as f32),
                1 => f32::from_bits(1 + (i as u32 * 31) % 4096),
                2 => -f32::from_bits(1 + (i as u32 * 97) % 0x7F_FFFF),
                3 => if i % 2 == 0 { 1e-20 } else { 1e-44 },
                _ => if i == 129 { -3e37 } else { 1e-30 },
            };
        }
    }
    let xt = ctx.g.upload_f32(&x, &[rows * k]).unwrap();
    let xq = ctx.g.qwen4_moe_rotate256_i8(&xt, k, rows).expect("rotate256_i8");
    sync(&ctx.g);
    let nk = k / 128;
    let dev = dl_sidecar(&ctx.g, nk * rows * 136);
    let (xs, _) = produce_gate(&x, k, rows, &ctx.s1, &ctx.s2, 127.0);
    let c = cmp_sidecar(&dev, &xs.to_bytes(nk), rows, nk, &vec![true; rows]);
    rep.add("S1 gate-producer-edge rotate256_i8 (zero/outlier/denormal/+-0 rows) device-vs-cpu", &c, None);
    let _ = xq;
    ctx.g.free_tensor(xt).unwrap();

    // down producer: rows of the grouped plane with specials
    let kd = 640usize;
    let g = 160usize;
    let mut h = gen_h_bf16(&mut rng, g, kd);
    for (p, kind) in [(1usize, 0), (2, 1), (3, 2), (4, 3)] {
        for (i, v) in h[p * kd..(p + 1) * kd].iter_mut().enumerate() {
            *v = match kind {
                0 => 0x0001 + (i as u16 % 64),   // bf16 denormals
                1 => 0x8001 + (i as u16 % 8),
                2 => if i % 2 == 0 { 0x8000 } else { 0x0000 },
                _ => if i == 5 { bf16_bits(3e38) } else { 0x0040 },
            };
        }
    }
    let mut sorted: Vec<i32> = (0..g as i32).collect();
    rng.shuffle(&mut sorted);
    for p in (0..g).step_by(7) {
        sorted[p] = -1;
    }
    // renumber live slots densely so total_slots == live count
    let mut live = 0;
    for s in sorted.iter_mut() {
        if *s >= 0 {
            *s = live;
            live += 1;
        }
    }
    let ht = up_u16(&ctx.g, &h);
    let st = up_i32(&ctx.g, &sorted);
    let l = live as usize;
    let xq = ctx.g.qwen4_moe_rotate128_i8(&ht, &st, kd, g, l).expect("rotate128_i8");
    sync(&ctx.g);
    let res_nk = ep_a8_down_reserve_k(kd) / 128;
    let dev = dl_sidecar(&ctx.g, res_nk * l * 136);
    let (xs, present) = produce_down(&h, &sorted, kd, l, &ctx.s1_128, &ctx.s2_128, 127.0);
    let c = cmp_sidecar(&dev, &xs.to_bytes(res_nk), l, kd / 128, &present);
    rep.add("S1 down-producer-edge rotate128_i8 (zero/outlier/bf16-denormal/+-0 rows, padding rows) device-vs-cpu", &c, None);
    let _ = xq;
    ctx.g.free_tensor(ht).unwrap();
    ctx.g.free_tensor(st).unwrap();
}

#[test]
#[ignore = "needs exact gfx1201 (fabric card C)"]
fn s1_synthetic() {
    let mut ctx = init_ctx();
    let mut rep = Report::new("s1_synthetic");
    let tab_diff = (0..65536u32)
        .filter(|&i| {
            let v = f32::from_bits(i << 16);
            v.is_finite() && (-v).exp().to_bits() != ctx.exp_tab[i as usize].to_bits()
        })
        .count();
    rep.extra.insert("device_expf_vs_rust_exp_diff_patterns".into(), json!(tab_diff));
    quantizer_edge(&mut ctx, &mut rep);
    producer_edge(&mut ctx, &mut rep);
    let geoms = [
        Geom { down: false, m: 256, k: 256, tag: "gate_up m256 k256" },
        Geom { down: false, m: 256, k: 512, tag: "gate_up m256 k512" },
        Geom { down: false, m: 1280, k: 2560, tag: "gate_up m1280 k2560" },
        Geom { down: true, m: 256, k: 128, tag: "down m256 k128" },
        Geom { down: true, m: 192, k: 384, tag: "down m192 k384" },
        Geom { down: true, m: 2560, k: 640, tag: "down m2560 k640" },
    ];
    for (i, g) in geoms.iter().enumerate() {
        gemm_suite(&mut ctx, &mut rep, *g, 0x5100 + i as u64);
    }
    let arch = ctx.g.arch.clone();
    rep.finish(&arch);
}

// ---------------------------------------------------------------------------------------------------------
// Section 3: negative controls
// ---------------------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs exact gfx1201 (fabric card C)"]
fn s3_negative_controls() {
    let mut ctx = init_ctx();
    let mut rep = Report::new("s3_negative_controls");
    let n_e = 12usize;
    // Adversarial-fold-order and plain experts: expert 5 = Adversarial, others per PROFILES.
    for (geom, tag) in [
        (Geom { down: false, m: 256, k: 512, tag: "gate_up" }, "gate_up m256 k512"),
        (Geom { down: true, m: 192, k: 384, tag: "down" }, "down m192 k384"),
    ] {
        let mut rng = Rng::new(0x3E6 + geom.m as u64);
        let mut w = gen_experts(&mut rng, &geom, n_e);
        // make experts 0..=4 well-conditioned random so a one-nibble corruption is visible
        for e in [0usize, 1, 2] {
            w[e] = gen_wlog(&mut rng, geom.m, geom.k, WProf::Random);
        }
        let kind = if geom.down { Kind::Down } else { Kind::GateUp };
        let sh = Shape { down: geom.down, m: geom.m, k: geom.k, div: if geom.down { 1 } else { 10 } };
        let specs = vec![
            Ts::Run(0, FULL),
            Ts::Run(0, FULL),
            Ts::Run(1, FULL),
            Ts::Run(1, FULL),
            Ts::Run(1, 0x0FFF),
            Ts::Run(2, FULL),
            Ts::Run(5, FULL),
            Ts::Run(5, FULL),
            Ts::Run(5, FULL),
            Ts::Run(6, FULL),
        ];
        let lay = build_plan(&specs, &mut rng);
        let ld = upload_layout(&ctx.g, &lay);
        let l = lay.total_slots;
        let n_x = if geom.down { l } else { l.div_ceil(10) };
        let nk = geom.k / 128;
        let res_k = if geom.down { ep_a8_down_reserve_k(geom.k) } else { geom.k };
        let res_nk = res_k / 128;
        let dw = upload_w(&ctx.g, kind, &w);
        let x = gen_xset_hand(&mut rng, n_x, nk);
        let exp = expect_case(&ctx, sh, &lay, &w, &x, Fold::Asc);
        let xq = put_sidecar(&mut ctx.g, &x.to_bytes(res_nk), res_k, n_x);
        let cols_bf = if geom.down { sh.m } else { sh.m / 2 };
        let _ = cols_bf;

        // positive anchor: unmodified device vs CPU must be 0 (guards against a vacuous control)
        let ptrs = &dw.ship.table;
        ctx.g.set_mq4_experts_planar(false);
        let got_bf = run_public(&mut ctx.g, sh, &lay, &ld, ptrs, &xq, n_x, None);
        let got_raw = run_raw(&mut ctx.g, false, 2, sh, &lay, &ld, ptrs, n_x, None);
        let anchor = cmp_y(&exp.bf16, &got_bf).total() + cmp_y(&exp.raw, &got_raw).total();
        rep.add(&format!("S3 anchor {tag}: unmodified device vs CPU (must be 0)"), &Cmp { bytes: 1, bad: anchor, ..Default::default() }, None);

        // (a) flip one byte of a device Y copy
        let mut flipped = got_bf.clone();
        flipped[exp.bf16.len() / 3] ^= 0x01;
        rep.control(&format!("(a) flip one byte of a device Y copy [{tag}]"), cmp_y(&exp.bf16, &flipped).total());

        // (b) corrupt one nibble of one expert weight on the device only (both layouts)
        {
            let mut wc = w.clone();
            let (e0, r0, c0) = (1usize, 5usize, 7usize);
            let cur = wc[e0].w[r0 * geom.k + c0];
            wc[e0].w[r0 * geom.k + c0] = if cur > -4 { cur - 5 } else { cur + 5 };
            let dwc = upload_w(&ctx.g, kind, &wc);
            let mut bad_bf = 0;
            let mut bad_raw = 0;
            for planar in [false, true] {
                ctx.g.set_mq4_experts_planar(planar);
                let p = shipped_or_planar(&dwc, planar);
                bad_bf += cmp_y(&exp.bf16, &run_public(&mut ctx.g, sh, &lay, &ld, p, &xq, n_x, None)).total();
                bad_raw += cmp_y(&exp.raw, &run_raw(&mut ctx.g, planar, 2, sh, &lay, &ld, p, n_x, None)).total();
            }
            ctx.g.set_mq4_experts_planar(false);
            rep.control(&format!("(b) one corrupted nibble on device only, raw F32 [{tag}]"), bad_raw);
            rep.extra.insert(format!("control_b_bf16_mismatching_bytes[{tag}]"), json!(bad_bf));
            free_w(&mut ctx.g, dwc);
        }

        // (c) CPU reference with a non-ascending epoch fold order on adversarial scales
        {
            let wrong = expect_case(&ctx, sh, &lay, &w, &x, Fold::Desc);
            rep.control(
                &format!("(c) CPU fold in descending epoch order vs device raw (adversarial scales) [{tag}]"),
                cmp_y(&wrong.raw, &got_raw).total(),
            );
        }

        // (e) swapped gate/up halves (gate/up only)
        if !geom.down {
            let sw = swap_halves(&exp.sums, lay.g, sh.m);
            let raw_sw = exp_raw_bytes(&lay, &sw, sh.m, n_e);
            let bf_sw = exp_bf16_gate(&lay, &sw, sh.m, n_e, &ctx.exp_tab);
            rep.control(&format!("(e) swapped gate/up halves, raw [{tag}]"), cmp_y(&raw_sw, &got_raw).total());
            rep.control(&format!("(e) swapped gate/up halves, bf16 SwiGLU [{tag}]"), cmp_y(&bf_sw, &got_bf).total());
        }

        // (f) leader-rule violation: tile ids edited after the CPU reference (one tile runs the wrong expert)
        {
            let mut t2 = lay.tiles.clone();
            let t_idx = 2; // first tile of expert 1's run
            t2[t_idx] = 7;
            let tt = up_i32(&ctx.g, &t2);
            let bf = run_public(&mut ctx.g, sh, &lay, &ld, ptrs, &xq, n_x, Some(&tt));
            let rw = run_raw(&mut ctx.g, false, 2, sh, &lay, &ld, ptrs, n_x, Some(&tt));
            rep.control(&format!("(f) tile id edited after the CPU reference, raw [{tag}]"), cmp_y(&exp.raw, &rw).total());
            rep.control(&format!("(f) tile id edited after the CPU reference, bf16 [{tag}]"), cmp_y(&exp.bf16, &bf).total());
            ctx.g.free_tensor(tt).unwrap();
        }
        free_w(&mut ctx.g, dw);
        free_layout(&mut ctx.g, ld);
    }

    // (d) X quantized at the wrong scale: d = RN(amax / 128)
    {
        let mut rng = Rng::new(0xD0D);
        let (k, rows) = (512usize, 48usize);
        let x = gen_x_f32(&mut rng, rows, k, false);
        let xt = ctx.g.upload_f32(&x, &[rows * k]).unwrap();
        let xq = ctx.g.qwen4_moe_rotate256_i8(&xt, k, rows).unwrap();
        sync(&ctx.g);
        let nk = k / 128;
        let dev = dl_sidecar(&ctx.g, nk * rows * 136);
        let (good, _) = produce_gate(&x, k, rows, &ctx.s1, &ctx.s2, 127.0);
        let (wrong, _) = produce_gate(&x, k, rows, &ctx.s1, &ctx.s2, 128.0);
        let present = vec![true; rows];
        rep.add(
            "S3 anchor (d): device producer vs CPU quantizer at the correct scale (must be 0)",
            &cmp_sidecar(&dev, &good.to_bytes(nk), rows, nk, &present),
            None,
        );
        rep.control(
            "(d) CPU X quantized at d=RN(amax/128) vs device sidecar bytes",
            cmp_sidecar(&dev, &wrong.to_bytes(nk), rows, nk, &present).total(),
        );
        // and through the GEMM: device raw output vs CPU GEMM over the wrong-scale X
        let geom = Geom { down: false, m: 256, k, tag: "gate_up" };
        let w = gen_experts(&mut rng, &geom, 4);
        let dw = upload_w(&ctx.g, Kind::GateUp, &w);
        let specs = vec![Ts::Run(0, FULL), Ts::Run(1, FULL), Ts::Run(1, FULL), Ts::Run(2, FULL)];
        let mut lay = build_plan(&specs, &mut rng);
        // gate rows: 10 slots per token; keep slots < rows*10
        let live = lay.total_slots;
        assert!(live <= rows * 10);
        lay.total_slots = live;
        let ld = upload_layout(&ctx.g, &lay);
        let sh = Shape { down: false, m: 256, k, div: 10 };
        let raw = run_raw(&mut ctx.g, false, 2, sh, &lay, &ld, &dw.ship.table, rows, None);
        let sums_wrong = ref_raw(&lay, &w, &wrong, 10, 256, Fold::Asc);
        let sums_good = ref_raw(&lay, &w, &good, 10, 256, Fold::Asc);
        rep.add(
            "S3 anchor (d): device raw GEMM over the device sidecar vs CPU GEMM over the correct-scale X (must be 0)",
            &cmp_y(&exp_raw_bytes(&lay, &sums_good, 256, 4), &raw),
            Some((4, 256)),
        );
        rep.control(
            "(d) device raw GEMM vs CPU GEMM over the wrong-scale X",
            cmp_y(&exp_raw_bytes(&lay, &sums_wrong, 256, 4), &raw).total(),
        );
        let _ = xq;
        free_w(&mut ctx.g, dw);
        free_layout(&mut ctx.g, ld);
        ctx.g.free_tensor(xt).unwrap();
    }
    let arch = ctx.g.arch.clone();
    rep.finish(&arch);
}

// ---------------------------------------------------------------------------------------------------------
// Section 2: real tensors
// ---------------------------------------------------------------------------------------------------------

struct TensorEnt {
    name: String,
    offset: u64,
    len: usize,
}

fn hfqm_index(f: &mut File) -> Vec<TensorEnt> {
    let mut h = [0u8; 32];
    f.read_exact(&mut h).unwrap();
    assert_eq!(&h[..4], b"HFQM");
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap()) as usize;
    let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
    let meta = u64_at(&h[16..24]);
    let data = u64_at(&h[24..32]);
    f.seek(SeekFrom::Start(meta)).unwrap();
    let mut b = vec![0u8; (data - meta) as usize];
    f.read_exact(&mut b).unwrap();
    let (mut depth, mut quoted, mut escape, mut p) = (0i32, false, false, 0usize);
    for (i, &c) in b.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if quoted && c == b'\\' {
            escape = true;
            continue;
        }
        if c == b'"' {
            quoted = !quoted;
        }
        if !quoted {
            if c == b'{' {
                depth += 1;
            }
            if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    p = i + 1;
                    break;
                }
            }
        }
    }
    assert!(p > 0);
    let n = u32_at(&b[p..p + 4]);
    p += 4;
    assert_eq!(n, u32_at(&h[12..16]));
    let mut offset = data;
    let mut out = Vec::new();
    for _ in 0..n {
        let l = u16::from_le_bytes(b[p..p + 2].try_into().unwrap()) as usize;
        p += 2;
        let name = String::from_utf8(b[p..p + l].to_vec()).unwrap();
        p += l + 1;
        let nd = b[p] as usize;
        p += 1 + 4 * nd + 4;
        let len = u64_at(&b[p..p + 8]) as usize;
        p += 8;
        out.push(TensorEnt { name, offset, len });
        offset += len as u64;
    }
    out
}

fn read_range(f: &mut File, off: u64, len: usize) -> Vec<u8> {
    f.seek(SeekFrom::Start(off)).unwrap();
    let mut v = vec![0u8; len];
    f.read_exact(&mut v).unwrap();
    v
}

fn spread_ids(total: usize, count: usize) -> Vec<usize> {
    (0..count).map(|i| i * total / count).collect()
}

/// layer -> rows of 10 global expert ids; plus exact (layer, rows) records.
struct Routes {
    pool: HashMap<usize, Vec<[i32; 10]>>,
    exact: HashMap<(usize, usize), Vec<i32>>,
}
fn load_routes() -> Routes {
    let mut r = Routes {
        pool: HashMap::new(),
        exact: HashMap::new(),
    };
    let mut files: Vec<_> = std::fs::read_dir(ROUTES_DIR)
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    files.retain(|p| p.extension().map(|e| e == "bin").unwrap_or(false));
    files.sort();
    for p in files {
        let b = std::fs::read(&p).unwrap();
        let mut pos = 0;
        while pos + 4 <= b.len() {
            let layer = u16::from_le_bytes([b[pos], b[pos + 1]]) as usize;
            let rows = u16::from_le_bytes([b[pos + 2], b[pos + 3]]) as usize;
            pos += 4;
            if pos + rows * 40 > b.len() {
                break;
            }
            let ids: Vec<i32> = (0..rows * 10)
                .map(|i| i32::from_le_bytes(b[pos + 4 * i..pos + 4 * i + 4].try_into().unwrap()))
                .collect();
            pos += rows * 40;
            for row in ids.chunks(10) {
                r.pool.entry(layer).or_default().push(row.try_into().unwrap());
            }
            r.exact.entry((layer, rows)).or_insert(ids);
        }
    }
    r
}

/// Flat per-slot expert ids (`rows * 10`, mapped onto `0..n_sel`) for `(layer, rows)`.
fn route_real(routes: &Routes, layer: usize, rows: usize, n_sel: usize) -> (Vec<i32>, &'static str) {
    if let Some(ids) = routes.exact.get(&(layer, rows)) {
        return (ids.iter().map(|&e| e.rem_euclid(n_sel as i32)).collect(), "exact fixture record");
    }
    let pool = routes.pool.get(&layer).expect("no route fixtures for this layer");
    let mut rng = Rng::new(0xA0_0000 + layer as u64);
    let mut idx: Vec<usize> = Vec::with_capacity(rows);
    while idx.len() < rows {
        let mut cyc: Vec<usize> = (0..pool.len()).collect();
        if !idx.is_empty() {
            rng.shuffle(&mut cyc);
        }
        idx.extend(cyc.into_iter().take(rows - idx.len()));
    }
    let ids = idx
        .iter()
        .flat_map(|&i| pool[i].iter().map(|&e| e.rem_euclid(n_sel as i32)))
        .collect();
    (
        ids,
        if rows <= pool.len() { "fixture rows (layer pool, first rows)" } else { "fixture rows tiled+permuted" },
    )
}

/// Synthetic routing: 10 distinct experts per token; `skew` = zipf-ish weights.
fn route_synth(rows: usize, n_sel: usize, skew: bool, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    let wts: Vec<f64> = (0..n_sel).map(|i| if skew { 1.0 / ((i + 1) as f64).powf(1.3) } else { 1.0 }).collect();
    let mut out = Vec::with_capacity(rows * 10);
    for _ in 0..rows {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < 10 {
            let tot: f64 = (0..n_sel).filter(|i| !chosen.contains(i)).map(|i| wts[i]).sum();
            let mut r = (rng.u64() >> 11) as f64 / 9_007_199_254_740_992.0 * tot;
            for i in 0..n_sel {
                if chosen.contains(&i) {
                    continue;
                }
                if r < wts[i] {
                    chosen.push(i);
                    break;
                }
                r -= wts[i];
            }
        }
        out.extend(chosen.iter().map(|&e| e as i32));
    }
    out
}

fn gen_real_x(rng: &mut Rng, rows: usize, k: usize) -> Vec<f32> {
    let chans: Vec<usize> = (0..14).map(|_| rng.below(k)).collect();
    let mut x = vec![0f32; rows * k];
    for t in 0..rows {
        let tok_scale = (rng.gauss() * 0.5).exp();
        for c in 0..k {
            x[t * k + c] = rng.gauss() * tok_scale;
        }
        for &c in &chans {
            x[t * k + c] *= 30.0 + 20.0 * rng.unit();
        }
    }
    x
}

// ---- quality (f64) ---------------------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Acc {
    num: f64,
    den: f64,
    maxd: f64,
    n: usize,
}
impl Acc {
    fn add(&mut self, a: f64, b: f64) {
        let d = (a - b).abs();
        self.num += d * d;
        self.den += b * b;
        self.maxd = self.maxd.max(d);
        self.n += 1;
    }
    fn json(&self) -> Value {
        let rel = if self.den > 0.0 { (self.num / self.den).sqrt() } else { f64::NAN };
        let rms = (self.den / self.n.max(1) as f64).sqrt();
        json!({"rel_l2": rel, "max_abs_over_rms": if rms > 0.0 { self.maxd / rms } else { f64::NAN }, "n": self.n})
    }
}

fn dot_f64(w: &Wlog, r: usize, x: &[f64]) -> f64 {
    let nk = w.nk();
    let mut tot = 0.0f64;
    for h in 0..nk {
        let s = f16_to_f32(w.sc[r * nk + h]) as f64;
        let wr = &w.w[r * w.k + h * 128..r * w.k + (h + 1) * 128];
        let xr = &x[h * 128..(h + 1) * 128];
        let mut a = 0.0f64;
        for i in 0..128 {
            a += wr[i] as f64 * xr[i];
        }
        tot += s * a;
    }
    tot
}

fn bf16v(y: &[u8], idx: usize) -> f64 {
    bf16_to_f32(u16::from_le_bytes([y[idx * 2], y[idx * 2 + 1]])) as f64
}

#[allow(clippy::too_many_arguments)]
fn quality(
    lay: &Layout,
    gate_w: &[Wlog],
    down_w: &[Wlog],
    xr: &[f32],
    rows: usize,
    s1_128: &[f32],
    s2_128: &[f32],
    a8: (&[u8], &[u8]),
    a4: Option<(&[u8], &[u8])>,
) -> Value {
    let (mi, hid) = (640usize, 2560usize);
    let nsamp = rows.min(128);
    let toks: Vec<usize> = (0..nsamp).map(|i| i * rows / nsamp).collect();
    let mut pos_of = vec![usize::MAX; rows * 10];
    for (p, &s) in lay.sorted.iter().enumerate() {
        if s >= 0 {
            pos_of[s as usize] = p;
        }
    }
    let tasks: Vec<(usize, usize)> = toks.iter().flat_map(|&t| (0..10).map(move |j| (t, j))).collect();
    let refs: Vec<(usize, Vec<f64>, Vec<f64>)> = tasks
        .par_iter()
        .map(|&(t, j)| {
            let slot = t * 10 + j;
            let pos = pos_of[slot];
            let e = lay.tiles[pos / 16] as usize;
            let xt: Vec<f64> = xr[t * hid..(t + 1) * hid].iter().map(|&v| v as f64).collect();
            let gw = &gate_w[e];
            let mut h64 = vec![0f64; mi];
            for c in 0..mi {
                let g = dot_f64(gw, c, &xt);
                let u = dot_f64(gw, mi + c, &xt);
                h64[c] = g / (1.0 + (-g).exp()) * u;
            }
            // FWHT128 rotate in f64
            let mut hr = vec![0f64; mi];
            for grp in 0..mi / 128 {
                let mut v = [0f64; 128];
                for i in 0..128 {
                    v[i] = h64[grp * 128 + i] * s1_128[i] as f64;
                }
                let mut hh = 1;
                while hh < 128 {
                    let mut i = 0;
                    while i < 128 {
                        for q in i..i + hh {
                            let (a, b) = (v[q], v[q + hh]);
                            v[q] = a + b;
                            v[q + hh] = a - b;
                        }
                        i += 2 * hh;
                    }
                    hh *= 2;
                }
                let sc = 1.0 / (128f64).sqrt();
                for i in 0..128 {
                    hr[grp * 128 + i] = v[i] * sc * s2_128[i] as f64;
                }
            }
            let dw = &down_w[e];
            let y64: Vec<f64> = (0..hid).map(|r| dot_f64(dw, r, &hr)).collect();
            (pos, h64, y64)
        })
        .collect();
    let mut acc: HashMap<&str, Acc> = HashMap::new();
    for (pos, h64, y64) in &refs {
        for c in 0..mi {
            acc.entry("y1_a8_vs_f32").or_default().add(bf16v(a8.0, pos * mi + c), h64[c]);
            if let Some(a4) = a4 {
                acc.entry("y1_a4_vs_f32").or_default().add(bf16v(a4.0, pos * mi + c), h64[c]);
                acc.entry("y1_a8_vs_a4").or_default().add(bf16v(a8.0, pos * mi + c), bf16v(a4.0, pos * mi + c));
            }
        }
        for c in 0..hid {
            acc.entry("y2_a8_vs_f32").or_default().add(bf16v(a8.1, pos * hid + c), y64[c]);
            if let Some(a4) = a4 {
                acc.entry("y2_a4_vs_f32").or_default().add(bf16v(a4.1, pos * hid + c), y64[c]);
                acc.entry("y2_a8_vs_a4").or_default().add(bf16v(a8.1, pos * hid + c), bf16v(a4.1, pos * hid + c));
            }
        }
    }
    let mut o = serde_json::Map::new();
    o.insert("sampled_tokens".into(), json!(nsamp));
    o.insert("sampled_slots".into(), json!(refs.len()));
    for (k, v) in &acc {
        o.insert((*k).to_string(), v.json());
    }
    Value::Object(o)
}

fn run_a4(
    g: &mut Gpu,
    x: &GpuTensor,
    rows: usize,
    lay: &Layout,
    ld: &LayDev,
    gate_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let e = |x: rdna_compute::HipError| format!("{x}");
    g.set_mq4_experts_planar(false);
    g.reserve_qwen4_moe_sym(rows, 10, 512, 2560, 640).map_err(e)?;
    let l = rows * 10;
    let y1b = lay.g * 640 * 2;
    let y2b = lay.g * 2560 * 2;
    let y1 = alloc_poisoned(g, y1b);
    let y2 = alloc_poisoned(g, y2b);
    let gate = g.qwen4_moe_rotate256_i4(x, 2560, rows).map_err(e)?;
    g.gemm_qwen4_moe_gate_up_silu_iu4_sym(gate_ptrs, &ld.tiles, &ld.sorted, &gate, &y1, 1280, 2560, 10, lay.g, rows, false)
        .map_err(e)?;
    let d = g.qwen4_moe_rotate128_i4_gp(&y1, &ld.sorted, 640, lay.g, l).map_err(e)?;
    g.gemm_qwen4_moe_down_iu4_sym(down_ptrs, &ld.tiles, &ld.sorted, &d, &y2, 2560, 640, 1, lay.g, l, false)
        .map_err(e)?;
    g.hip.device_synchronize().map_err(e)?;
    let r = (dl_bytes(g, &y1, y1b), dl_bytes(g, &y2, y2b));
    g.free_tensor(y1).unwrap();
    g.free_tensor(y2).unwrap();
    Ok(r)
}

struct RealLayer {
    gate_w: Vec<Wlog>,
    down_w: Vec<Wlog>,
    gate_dev: DevW,
    down_dev: DevW,
}

fn load_real_layer(ctx: &mut Ctx, f: &mut File, idx: &[TensorEnt], layer: usize, n_sel: usize, rep: &mut Report) -> RealLayer {
    let find = |suffix: &str| {
        let name = format!("model.language_model.layers.{layer}.mlp.experts.{suffix}");
        idx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("tensor {name} not found"))
    };
    let (gu, dn) = (find("gate_up_proj"), find("down_proj"));
    let (gper, dper) = (shipped_bytes(Kind::GateUp, 1280, 2560), shipped_bytes(Kind::Down, 2560, 640));
    assert_eq!(gper, 1_740_800);
    assert_eq!(dper, 870_400);
    assert_eq!(gu.len % gper, 0, "{}: not a multiple of one shipped expert", gu.name);
    assert_eq!(dn.len % dper, 0, "{}: not a multiple of one shipped expert", dn.name);
    let total = gu.len / gper;
    assert_eq!(dn.len / dper, total);
    let ids = spread_ids(total, n_sel);
    let (mut gship, mut dship) = (Vec::new(), Vec::new());
    let (mut gw, mut dw) = (Vec::new(), Vec::new());
    let (mut gplan, mut dplan) = (Vec::new(), Vec::new());
    let mut asym = 0u64;
    for &e in &ids {
        let gb = read_range(f, gu.offset + (e * gper) as u64, gper);
        let db = read_range(f, dn.offset + (e * dper) as u64, dper);
        let (gl, a1) = unpack_shipped(Kind::GateUp, 1280, 2560, &gb);
        let (dl, a2) = unpack_shipped(Kind::Down, 2560, 640, &db);
        asym += a1 + a2;
        gplan.extend(repack_shipped_to_planar(Kind::GateUp, 1280, 2560, &gb));
        dplan.extend(repack_shipped_to_planar(Kind::Down, 2560, 640, &db));
        // the independent logical -> planar packer must agree with the byte repack
        assert_eq!(pack_planar(&gl), repack_shipped_to_planar(Kind::GateUp, 1280, 2560, &gb), "planar pack != repack (gate)");
        gship.extend(gb);
        dship.extend(db);
        gw.push(gl);
        dw.push(dl);
    }
    rep.extra.insert(
        format!("layer{layer}_weights"),
        json!({
            "gate_up_tensor": gu.name, "down_tensor": dn.name, "stacked_experts": total,
            "uploaded_expert_ids": ids, "asymmetric_scale_halves": asym,
            "gate_up_bytes_per_expert": gper, "down_bytes_per_expert": dper,
            "planar_gate_up_bytes_per_expert": planar_bytes(1280, 2560), "planar_down_bytes_per_expert": planar_bytes(2560, 640),
        }),
    );
    assert_eq!(asym, 0, "layer {layer}: real tensors carry non-symmetric headers");
    let gate_dev = DevW {
        ship: upload_stack(&ctx.g, &gship, n_sel, gper),
        plan: upload_stack(&ctx.g, &gplan, n_sel, planar_bytes(1280, 2560)),
    };
    let down_dev = DevW {
        ship: upload_stack(&ctx.g, &dship, n_sel, dper),
        plan: upload_stack(&ctx.g, &dplan, n_sel, planar_bytes(2560, 640)),
    };
    RealLayer { gate_w: gw, down_w: dw, gate_dev, down_dev }
}

struct DownRef {
    y1: Vec<u8>,
    xs: XSet,
    present: Vec<bool>,
    raw: Vec<u8>,
    bf16: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn chain_case(
    ctx: &mut Ctx,
    rep: &mut Report,
    rl: &RealLayer,
    layer: usize,
    label: &str,
    ids: &[i32],
    rows: usize,
    n_sel: usize,
    quality_out: &mut Vec<Value>,
) {
    let (hid, mi) = (2560usize, 640usize);
    let lay = build_pad16(ids, n_sel, 2);
    let l = rows * 10;
    assert_eq!(lay.total_slots, l);
    let ld = upload_layout(&ctx.g, &lay);
    let mut rng = Rng::new(0xC0DE ^ (layer as u64) << 8 ^ rows as u64);
    let x = gen_real_x(&mut rng, rows, hid);
    let (gx, xr) = produce_gate(&x, hid, rows, &ctx.s1, &ctx.s2, 127.0);
    let sh1 = Shape { down: false, m: 1280, k: hid, div: 10 };
    let sh2 = Shape { down: true, m: hid, k: mi, div: 1 };
    let e1 = expect_case(ctx, sh1, &lay, &rl.gate_w, &gx, Fold::Asc);
    rep.note_sums(&e1.sums);
    let tag0 = format!("S2 layer{layer} {label} rows={rows} G={}", lay.g);
    let xt = ctx.g.upload_f32(&x, &[rows * hid]).unwrap();
    let nk1 = hid / 128;
    let res_nk2 = ep_a8_down_reserve_k(mi) / 128;
    let mut down_cache: Option<DownRef> = None;
    let mut a8_ship: Option<(Vec<u8>, Vec<u8>)> = None;
    for planar in [false, true] {
        ctx.g.set_mq4_experts_planar(planar);
        let ln = if planar { "planar" } else { "shipped" };
        let tag = format!("{tag0} {ln}");
        // gate producer
        let xq1 = ctx.g.qwen4_moe_rotate256_i8(&xt, hid, rows).expect("rotate256_i8");
        sync(&ctx.g);
        let dev1 = dl_sidecar(&ctx.g, nk1 * rows * 136);
        rep.add(
            &format!("{tag} gate sidecar device-vs-cpu"),
            &cmp_sidecar(&dev1, &gx.to_bytes(nk1), rows, nk1, &vec![true; rows]),
            None,
        );
        let gptrs = shipped_or_planar(&rl.gate_dev, planar);
        let y1b = ybytes(sh1, &lay);
        let y1 = alloc_poisoned(&mut ctx.g, y1b);
        ctx.g
            .gemm_qwen4_moe_gate_up_silu_a8_sym(gptrs, &ld.tiles, &ld.sorted, &xq1, &y1, 1280, hid, 10, lay.g, rows)
            .expect("gate/up a8");
        sync(&ctx.g);
        let y1_all = dl_bytes(&ctx.g, &y1, y1b + GUARD);
        rep.add(&format!("{tag} gate/up y1 bf16 (public nt{})", ep_a8_select_nt(false, lay.g)), &cmp_y(&e1.bf16, &y1_all), Some((2, mi)));
        for nt in [1usize, 2] {
            let raw = run_raw(&mut ctx.g, planar, nt, sh1, &lay, &ld, gptrs, rows, None);
            rep.add(&format!("{tag} gate/up raw_f32 nt{nt}"), &cmp_y(&e1.raw, &raw), Some((4, 1280)));
        }
        // down producer over the device's own BF16 intermediate
        let y1_bytes = y1_all[..y1b].to_vec();
        let xq2 = ctx
            .g
            .qwen4_moe_rotate128_i8(&y1, &ld.sorted, mi, lay.g, l)
            .expect("rotate128_i8");
        sync(&ctx.g);
        let dev2 = dl_sidecar(&ctx.g, res_nk2 * l * 136);
        if down_cache.as_ref().map(|d| d.y1 != y1_bytes).unwrap_or(true) {
            let h: Vec<u16> = y1_bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let (xs, present) = produce_down(&h, &lay.sorted, mi, l, &ctx.s1_128, &ctx.s2_128, 127.0);
            let sums = ref_raw(&lay, &rl.down_w, &xs, 1, hid, Fold::Asc);
            let raw = exp_raw_bytes(&lay, &sums, hid, n_sel);
            let bf16 = exp_bf16_down(&lay, &sums, hid, n_sel);
            rep.note_sums(&sums);
            down_cache = Some(DownRef { y1: y1_bytes.clone(), xs, present, raw, bf16 });
        }
        let dr = down_cache.as_ref().unwrap();
        rep.add(
            &format!("{tag} down sidecar device-vs-cpu (CPU A8 of the device's own BF16 intermediate)"),
            &cmp_sidecar(&dev2, &dr.xs.to_bytes(res_nk2), l, mi / 128, &dr.present),
            None,
        );
        let dptrs = shipped_or_planar(&rl.down_dev, planar);
        let y2b = ybytes(sh2, &lay);
        let y2 = alloc_poisoned(&mut ctx.g, y2b);
        ctx.g
            .gemm_qwen4_moe_down_a8_sym(dptrs, &ld.tiles, &ld.sorted, &xq2, &y2, hid, mi, 1, lay.g, l)
            .expect("down a8");
        sync(&ctx.g);
        let y2_all = dl_bytes(&ctx.g, &y2, y2b + GUARD);
        rep.add(&format!("{tag} down y2 bf16 (public nt{})", ep_a8_select_nt(true, lay.g)), &cmp_y(&dr.bf16, &y2_all), Some((2, hid)));
        for nt in [1usize, 2] {
            let raw = run_raw(&mut ctx.g, planar, nt, sh2, &lay, &ld, dptrs, l, None);
            rep.add(&format!("{tag} down raw_f32 nt{nt}"), &cmp_y(&dr.raw, &raw), Some((4, hid)));
        }
        if !planar {
            a8_ship = Some((y1_bytes, y2_all[..y2b].to_vec()));
        }
        ctx.g.free_tensor(y1).unwrap();
        ctx.g.free_tensor(y2).unwrap();
    }
    ctx.g.set_mq4_experts_planar(false);
    // A4 chain (quality column) on the same layout / activations
    let a4 = {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_a4(&mut ctx.g, &xt, rows, &lay, &ld, &rl.gate_dev.ship.table, &rl.down_dev.ship.table)
        }));
        match r {
            Ok(Ok(v)) => Some(v),
            Ok(Err(e)) => {
                eprintln!("A4 chain not drivable: {e}");
                rep.extra.insert(format!("a4_skip_layer{layer}_{label}_{rows}"), json!(e));
                None
            }
            Err(_) => {
                eprintln!("A4 chain panicked");
                rep.extra.insert(format!("a4_skip_layer{layer}_{label}_{rows}"), json!("panic"));
                None
            }
        }
    };
    let (y1a8, y2a8) = a8_ship.unwrap();
    let mut q = quality(
        &lay,
        &rl.gate_w,
        &rl.down_w,
        &xr,
        rows,
        &ctx.s1_128,
        &ctx.s2_128,
        (&y1a8, &y2a8),
        a4.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())),
    );
    q["layer"] = json!(layer);
    q["rows"] = json!(rows);
    q["routing"] = json!(label);
    q["a4_available"] = json!(a4.is_some());
    eprintln!("QUALITY layer{layer} {label} rows={rows}: {q}");
    quality_out.push(q);
    ctx.g.free_tensor(xt).unwrap();
    free_layout(&mut ctx.g, ld);
}

fn real_layer_test(layer: usize) {
    let mut ctx = init_ctx();
    let mut rep = Report::new(&format!("s2_real_layer{layer}"));
    let model = std::env::var("FN_MODEL").unwrap_or_else(|_| MODEL_DEFAULT.to_string());
    let mut f = File::open(&model).unwrap_or_else(|e| panic!("model {model}: {e}"));
    let idx = hfqm_index(&mut f);
    rep.extra.insert("model".into(), json!(model));
    rep.extra.insert("routes_dir".into(), json!(ROUTES_DIR));
    let n_sel = 24usize;
    let rl = load_real_layer(&mut ctx, &mut f, &idx, layer, n_sel, &mut rep);
    let routes = load_routes();
    let mut qual: Vec<Value> = Vec::new();
    let row_counts: Vec<usize> = std::env::var("EP_A8_ROWS")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![9, 64, 256, 512, 2048]);
    for &rows in &row_counts {
        let (ids, how) = route_real(&routes, layer, rows, n_sel);
        rep.extra.insert(format!("route_layer{layer}_rows{rows}"), json!(how));
        chain_case(&mut ctx, &mut rep, &rl, layer, "real-route", &ids, rows, n_sel, &mut qual);
    }
    if std::env::var("EP_A8_SYNTH_ROUTES").map(|v| v != "0").unwrap_or(true) {
        for (label, skew) in [("uniform-synth", false), ("skewed-synth", true)] {
            let rows = 2048;
            let ids = route_synth(rows, n_sel, skew, 0x5EED + layer as u64);
            chain_case(&mut ctx, &mut rep, &rl, layer, label, &ids, rows, n_sel, &mut qual);
        }
    }
    rep.extra.insert("quality".into(), Value::Array(qual));
    free_w(&mut ctx.g, rl.gate_dev);
    free_w(&mut ctx.g, rl.down_dev);
    let arch = ctx.g.arch.clone();
    rep.finish(&arch);
}

#[test]
#[ignore = "needs exact gfx1201 (fabric card C) + the Flash-Next mq4-xts model"]
fn s2_real_layer0() {
    real_layer_test(0);
}
#[test]
#[ignore = "needs exact gfx1201 (fabric card C) + the Flash-Next mq4-xts model"]
fn s2_real_layer17() {
    real_layer_test(17);
}
#[test]
#[ignore = "needs exact gfx1201 (fabric card C) + the Flash-Next mq4-xts model"]
fn s2_real_layer47() {
    real_layer_test(47);
}

// ---------------------------------------------------------------------------------------------------------
// Pure unit tests (no GPU)
// ---------------------------------------------------------------------------------------------------------

#[test]
fn unit_f16_roundtrip_all_finite() {
    for b in 0..=u16::MAX {
        let exp = (b >> 10) & 0x1F;
        if exp == 31 {
            continue;
        }
        let v = f16_to_f32(b);
        assert_eq!(f32_to_f16(v), b, "f16 {b:#06x} -> {v}");
    }
    assert_eq!(f32_to_f16(70000.0), 0x7C00);
    assert_eq!(f32_to_f16(-70000.0), 0xFC00);
    // RNE ties at the f16 grid: 1 + 2^-11 is the tie between 1.0 and 1 + 2^-10
    assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
    assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
}

#[test]
fn unit_bf16_rne_and_nonfinite() {
    assert_eq!(bf16_bits(1.0), 0x3F80);
    // tie to even: 1 + 2^-8 sits exactly between 0x3F80 and 0x3F81 -> even
    assert_eq!(bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
    assert_eq!(bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
    assert_eq!(bf16_bits(f32::INFINITY), 0x7F80);
    assert_eq!(bf16_bits(f32::NEG_INFINITY), 0xFF80);
    assert_eq!(bf16_bits(f32::from_bits(0x7FC0_0001)), 0x7FC0);
    assert!(bf16_rt(f32::NAN).is_nan());
    assert_eq!(bf16_rt(f32::MAX).to_bits(), 0x7F80_0000); // RNE overflow to +inf
}

#[test]
fn unit_quantizer_ties_zero_block_and_clamp() {
    let mut q = [9i8; 128];
    // zero block
    let (d, s) = quant128(&[0.0; 128], 127.0, &mut q);
    assert_eq!((d.to_bits(), s), (0, 0));
    assert!(q.iter().all(|&v| v == 0));
    let (d, _) = quant128(&[-0.0; 128], 127.0, &mut q);
    assert_eq!(d.to_bits(), 0);
    // amax = 127 -> d = 1; ties to even
    let mut x = [0f32; 128];
    x[0] = 127.0;
    let ties = [0.5f32, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 126.5, -126.5, 100.5];
    for (i, t) in ties.iter().enumerate() {
        x[1 + i] = *t;
    }
    let (d, s) = quant128(&x, 127.0, &mut q);
    assert_eq!(d, 1.0);
    let want = [127i8, 0, 2, 2, 4, 0, -2, -2, 126, -126, 100];
    assert_eq!(&q[..11], &want);
    assert_eq!(s, want.iter().map(|&v| v as i32).sum::<i32>());
    // d = RN(amax/127) for a non-representable ratio and the clamp at +-127
    let mut x = [0f32; 128];
    x[0] = -1.0;
    x[1] = 1.0;
    let (d, _) = quant128(&x, 127.0, &mut q);
    assert_eq!(d, 1.0f32 / 127.0);
    assert_eq!((q[0], q[1]), (-127, 127));
    // wrong-scale control really differs
    let mut q2 = [0i8; 128];
    let (d2, _) = quant128(&x, 128.0, &mut q2);
    assert_ne!(d2.to_bits(), d.to_bits());
    // denormal underflow: d == 0, zero lane -> NaN -> -127 (fminf(fmaxf(NaN, -127), 127) semantics)
    let mut x = [0f32; 128];
    x[0] = 1.4e-45;
    let (d, _) = quant128(&x, 127.0, &mut q);
    assert_eq!(d.to_bits(), 0);
    assert_eq!((q[0], q[1]), (127, -127));
}

#[test]
fn unit_block_encode_decode_nibble_pair_order() {
    let mut rng = Rng::new(1);
    let mut q = [0i8; 128];
    for v in q.iter_mut() {
        *v = (rng.u32() & 255) as u8 as i8;
    }
    let mut b = [0u8; 136];
    encode_blk(0.25, -7, &q, &mut b);
    // qs[8g+0..3] = q[8g+0,2,4,6]; qs[8g+4..7] = q[8g+1,3,5,7]
    for g in 0..16 {
        for j in 0..4 {
            assert_eq!(b[8 + 8 * g + j] as i8, q[8 * g + 2 * j]);
            assert_eq!(b[8 + 8 * g + 4 + j] as i8, q[8 * g + 2 * j + 1]);
        }
    }
    let (d, s, q2) = decode_blk(&b);
    assert_eq!((d, s), (0.25, -7));
    assert_eq!(q, q2);
}

#[test]
fn unit_shipped_planar_roundtrip_and_sizes() {
    let mut rng = Rng::new(2);
    for (kind, rows, k) in [(Kind::GateUp, 128usize, 512usize), (Kind::Down, 64, 384), (Kind::GateUp, 128, 256), (Kind::Down, 64, 128)] {
        for prof in [WProf::Random, WProf::ExtremeScales, WProf::AllZero, WProf::AllFifteen, WProf::MixedSign, WProf::Adversarial] {
            let w = gen_wlog(&mut rng, rows, k, prof);
            let s = pack_shipped(kind, &w);
            assert_eq!(s.len(), shipped_bytes(kind, rows, k));
            let (u, asym) = unpack_shipped(kind, rows, k, &s);
            assert_eq!(u, w);
            // symmetric headers by construction (inf/NaN zero aside: extreme scales stay finite*8 <= 65504*8 -> f16 inf)
            if prof != WProf::ExtremeScales && prof != WProf::HugeScales {
                assert_eq!(asym, 0);
            }
            let p = pack_planar(&w);
            assert_eq!(p.len(), planar_bytes(rows, k));
            assert_eq!(p, repack_shipped_to_planar(kind, rows, k, &s), "{kind:?} {prof:?}");
        }
    }
    assert_eq!(shipped_bytes(Kind::GateUp, 1280, 2560), 1_740_800);
    assert_eq!(shipped_bytes(Kind::Down, 2560, 640), 870_400);
    assert_eq!(planar_bytes(1280, 2560), 1_689_600);
    assert_eq!(planar_bytes(2560, 640), 844_800);
}

#[test]
fn unit_pad16_builder() {
    // expert 2 has 17 slots (2 tiles), expert 0 one slot, expert 5 16 slots, one dropped slot
    let mut ids = Vec::new();
    ids.extend(std::iter::repeat(2).take(17));
    ids.push(0);
    ids.extend(std::iter::repeat(5).take(16));
    ids.push(-1);
    let lay = build_pad16(&ids, 8, 2);
    assert_eq!(lay.tiles, vec![0, 2, 2, 5, -1, -1]);
    assert_eq!(lay.g, 96);
    assert_eq!(lay.total_slots, ids.len());
    // every routed slot exactly once, ascending per expert, padding -1, dropped slot absent
    let mut seen: Vec<i32> = lay.sorted.iter().copied().filter(|&s| s >= 0).collect();
    seen.sort();
    let want: Vec<i32> = (0..ids.len() as i32).filter(|&s| ids[s as usize] >= 0).collect();
    assert_eq!(seen, want);
    assert_eq!(lay.sorted[0], 17);
    assert_eq!(&lay.sorted[16..19], &[0, 1, 2]);
    assert_eq!(lay.sorted[32 + 1], -1); // second tile of expert 2 holds only slot 16
    assert_eq!(lay.sorted[32], 16);
    assert!(lay.sorted[64..].iter().all(|&s| s == -1));
}

#[test]
fn unit_plan_builder_every_live_lane_gets_one_slot() {
    let mut rng = Rng::new(3);
    let specs = plan_runs();
    let lay = build_plan(&specs, &mut rng);
    assert_eq!(lay.g, specs.len() * 16);
    let mut seen: Vec<i32> = lay.sorted.iter().copied().filter(|&s| s >= 0).collect();
    seen.sort();
    assert_eq!(seen, (0..lay.total_slots as i32).collect::<Vec<_>>());
    for (t, s) in specs.iter().enumerate() {
        match s {
            Ts::Sentinel => {
                assert_eq!(lay.tiles[t], -1);
                assert!(lay.sorted[t * 16..t * 16 + 16].iter().all(|&v| v < 0));
            }
            Ts::Masked(m) => {
                assert_eq!(lay.tiles[t], -1);
                for lane in 0..16 {
                    assert_eq!(lay.sorted[t * 16 + lane] >= 0, m >> lane & 1 == 1);
                }
            }
            Ts::Run(e, m) => {
                assert_eq!(lay.tiles[t], *e);
                for lane in 0..16 {
                    assert_eq!(lay.sorted[t * 16 + lane] >= 0, m >> lane & 1 == 1);
                }
            }
        }
    }
    let big = plan_big(&mut rng, 256, 12);
    assert_eq!(big.len(), 256);
    assert_eq!(build_plan(&big, &mut rng).g, 4096);
}

#[test]
fn unit_fwht_is_an_involution_and_rotation_order_matches() {
    let mut rng = Rng::new(4);
    let s1: Vec<f32> = (0..256).map(|_| if rng.u32() & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let s2: Vec<f32> = (0..256).map(|_| if rng.u32() & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let x: Vec<f32> = (0..256).map(|_| rng.gauss()).collect();
    let mut y = vec![0f32; 256];
    rotate256_row(&x, &s1, &s2, &mut y);
    let mut z = vec![0f32; 256];
    rotate256_row(&y, &s2, &s1, &mut z); // T' T = I with swapped signs
    for i in 0..256 {
        assert!((z[i] - x[i]).abs() < 1e-4, "{i}: {} vs {}", z[i], x[i]);
    }
    // orthonormal: norm preserved
    let n0: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
    let n1: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum();
    assert!((n0 - n1).abs() / n0 < 1e-4);
    // lower index gets a+b, upper a-b (n = 2)
    let mut v = [3.0f32, 5.0];
    fwht(&mut v);
    assert_eq!(v, [8.0, -2.0]);
}

#[test]
fn unit_cpu_gemm_fold_order_matters_and_is_ascending() {
    // explicit ascending fold with the exact i32 C per epoch
    let mut rng = Rng::new(7);
    let w = gen_wlog(&mut rng, 64, 1024, WProf::Adversarial);
    let x = gen_xset_hand(&mut rng, 4, 8);
    let mut differ = 0;
    for r in 0..64 {
        for xi in 0..4 {
            let asc = fold_row(&w, r, &x, xi, Fold::Asc);
            let mut sum = 0.0f32;
            for h in 0..8 {
                let c: i32 = (0..128)
                    .map(|i| w.w[r * 1024 + h * 128 + i] as i32 * x.q[(xi * 8 + h) * 128 + i] as i32)
                    .sum();
                let sd = f16_to_f32(w.sc[r * 8 + h]) * x.d[xi * 8 + h];
                sum = sd.mul_add(c as f32, sum);
            }
            assert_eq!(asc.to_bits(), sum.to_bits());
            differ += (asc.to_bits() != fold_row(&w, r, &x, xi, Fold::Desc).to_bits()) as u32;
        }
    }
    assert!(differ > 0, "adversarial scales must make the fold order observable");
}

#[test]
fn unit_cpu_reference_poison_padding_and_epilogue() {
    // one expert, one tile with 2 live lanes, one sentinel tile
    let mut rng = Rng::new(5);
    let w = vec![gen_wlog(&mut rng, 128, 256, WProf::Random)];
    let specs = vec![Ts::Run(0, 0b101), Ts::Sentinel];
    let lay = build_plan(&specs, &mut rng);
    assert_eq!(lay.total_slots, 2);
    let x = gen_xset_hand(&mut rng, 1, 2); // 2 slots -> 1 token for div 10
    let sums = ref_raw(&lay, &w, &x, 10, 128, Fold::Asc);
    let raw = exp_raw_bytes(&lay, &sums, 128, 1);
    assert!(raw[32 * 128 * 4..].iter().all(|&b| b == POISON), "sentinel rows poisoned");
    // padding lane sums are exactly +0
    assert!(sums[128..2 * 128].iter().all(|v| v.to_bits() == 0));
    // down epilogue: padding +0, live bf16(sum)
    let bf = exp_bf16_down(&lay, &sums, 128, 1);
    assert!(bf[128 * 2..2 * 128 * 2].iter().all(|&b| b == 0));
    let first = u16::from_le_bytes([bf[0], bf[1]]);
    assert_eq!(first, bf16_bits(sums[0]));
    // gate epilogue uses the supplied expf table and is +0 on padding lanes
    let tab: Vec<f32> = (0..65536u32).map(|i| (-f32::from_bits(i << 16)).exp()).collect();
    let g = exp_bf16_gate(&lay, &sums, 128, 1, &tab);
    assert_eq!(g.len(), 32 * 64 * 2);
    assert!(g[64 * 2..2 * 64 * 2].iter().all(|&b| b == 0));
    assert!(g[16 * 64 * 2..].iter().all(|&b| b == POISON));
}

#[test]
fn unit_comparator_counts_bytes_and_guards() {
    let a = vec![1u8, 2, 3, 4];
    let b = vec![1u8, 9, 3, 7];
    let c = cmp(&a, &b);
    assert_eq!((c.bytes, c.bad, c.first), (4, 2, Some((1, 2, 9))));
    let mut got = vec![0u8; 8];
    got.extend(std::iter::repeat(POISON).take(300));
    assert_eq!(cmp_y(&[0u8; 8], &got).total(), 0);
    got[8 + 100] = 0;
    let c = cmp_y(&[0u8; 8], &got);
    assert_eq!((c.bad, c.bad_guard), (0, 1));
}

#[test]
fn unit_down_producer_rotation_matches_a_direct_formula() {
    let mut rng = Rng::new(6);
    let s1: Vec<f32> = (0..128).map(|_| if rng.u32() & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let s2: Vec<f32> = (0..128).map(|_| if rng.u32() & 1 == 1 { 1.0 } else { -1.0 }).collect();
    let k = 256;
    let h: Vec<u16> = (0..2 * k).map(|_| bf16_bits(rng.gauss())).collect();
    let sorted = vec![1i32, -1];
    let (xs, present) = produce_down(&h, &sorted, k, 2, &s1, &s2, 127.0);
    assert_eq!(present, vec![false, true]);
    // slot 1 comes from row 0; compare block 0 with the explicit Hadamard matrix product in f64 (tolerance on q)
    let mut v = vec![0f64; 128];
    for i in 0..128 {
        v[i] = bf16_to_f32(h[i]) as f64 * s1[i] as f64;
    }
    let mut r = vec![0f64; 128];
    for i in 0..128 {
        for j in 0..128 {
            let sign = if ((i & j) as u32).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            r[i] += sign * v[j];
        }
        r[i] *= 0.0883883476483184 * s2[i] as f64;
    }
    let amax = r.iter().fold(0f64, |a, &b| a.max(b.abs()));
    let d = amax / 127.0;
    assert!((xs.d[1 * 2] as f64 - d).abs() / d < 1e-5);
    for i in 0..128 {
        let qi = xs.q[(1 * 2) * 128 + i] as f64;
        assert!((qi - r[i] / d).abs() <= 1.01, "{i}: {qi} vs {}", r[i] / d);
    }
}
