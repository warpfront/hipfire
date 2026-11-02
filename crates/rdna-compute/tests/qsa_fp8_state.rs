// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware tests of the QSA E4M3 ("fp8") K/V state on exact gfx1201
//! (n_heads 24, n_kv_heads 2, head_dim 256).
//!
//! * `qsa_fp8_append_encodes_e4m3_rows`: `indexed_attention_cache_append_batch`
//!   writes F32 and Fp8 caches from the same rows at the same nonzero position.
//!   The Fp8 cache bytes (256 codes per KV head, then one f16 scale per KV
//!   head) must equal a CPU model of `qsa_kv_store` byte for byte: head absmax,
//!   scale = RN_f16(amax / 448) (zero row: 1.0; nonzero underflow: bits 1; one
//!   bit bump when 448 * scale < amax, clamped at 0x7bff), `value / scale` in
//!   f32 and a nearest-even OCP E4M3 code (finite saturation, signed zero kept).
//!   The rows cover random, zero, all -0, negative, dynamic-range, scale-
//!   underflow, single-element, explicit E4M3 subnormal/grid and exact/perturbed
//!   tie values. The decoded element must stay within
//!   `|x|/16 + scale/1024 + 2^-22 |x|` of the F32 cache element, and the cache
//!   prefix/suffix around the appended rows must stay zero.
//! * `qsa_fp8_attention_matches_decoded_reference`: same-codec production-
//!   versus-sequential-production G1 for every R=1..64 in Eager/Recorded AR
//!   arithmetic scopes, using full live bytes and poisoned inactive/tail bytes.
//!   Separately, the single-row entry and production batch (rows 1..=8)
//!   at lengths 64/1131/2048/8195, and the
//!   gathered F16 WMMA lab entry (`pm = false`, only when the route is enabled)
//!   at modest shapes, against an f64 CPU attention over the *decoded* Fp8
//!   cache bytes (not the original F32 data). Exact routes: per head
//!   `maxdiff <= 1e-5 * maxabsref + 1e-7`; gathered: relative L2 < 5e-3 (the
//!   F16 operand/probability tolerance of `qsa_gathered_wmma_matches_per_head_kernel`).
//! * `qsa_fp8_attention_within_bound_of_f32`: the same q/K/V/selection through
//!   the F32 and Fp8 GPU routes (single, batch 1..=8, gathered where enabled).
//!   The element-wise difference is bounded analytically (below) plus the
//!   numerical allowance `2 * (1e-5 * maxabsref + 1e-7)` per head. The CPU is
//!   not used for attention there.
//!
//! Error bound of the quantized state. Per element the E4M3 store gives
//! `E(x, s) = |x|/16 + s/1024 + 2^-22 |x|` (half a 3-bit mantissa ulp for normal
//! codes, half the 2^-9 subnormal step times the scale, plus the f32 division
//! rounding). For a KV head let `EK`, `EV` be the maxima of `E` over every cache
//! element of K, V, `Vmax` the maximum |V| of the F32 cache and `Qabs =
//! sum_d |q_d|` of a query head. Each logit `q.K/16` moves by at most
//! `eps = EK * Qabs / 16`. Softmax has L1 sensitivity (Jacobian row L1 norm
//! `sum_j p_j |eta_j - mean_p(eta)| <= max|eta|`, integrated along the path) of
//! at most `eps`, so `|p' - p|_1 <= 2 min(1, eps)` holds (the trivial bound 2 and
//! a factor 2 of slack). Then
//! `|sum p' V' - sum p V| <= sum p' |V' - V| + |sum (p' - p) V| <= EV +
//! 2 Vmax min(1, eps)`, and the output of channel c is that times
//! `sigmoid(gate_c)`. Every selection is a subset of the cache, so cache-wide
//! maxima bound it.
//!
//! Gathered route (F16 WMMA), additionally, with `u = 2^-11` and `a = 2^-25` (F16
//! rounding: relative `u`, absolute half step `a` in the subnormal range) and `n`
//! the selected length: K/Q/V are rounded to F16 (`|dK| <= u|K| + a`), so the
//! dot error is `(2u + u^2) Qabs Kmax + a (1 + u) (Qabs + 256 Kmax) + 256 a^2`
//! (`Kmax` from the decoded cache) plus f32 accumulation `256 2^-24 (1+u)^2 Qabs
//! Kmax` and score/exp rounding `2^-22 Qabs Kmax / 16`, all added to `eps`.
//! Probabilities are rounded to F16 (`u + 2^-22` relative incl. `expf`, `a`
//! absolute) while the normaliser is the f32 sum of the unrounded ones (`l >= 1`
//! for the max entry), adding `Vmax (u + 2^-22 + n a)`; rounded V adds
//! `u Vmax_dec + a`; both multiplied by `1 + u + 2^-22 + n a` for the sum of rounded
//! probabilities; f32 accumulation of `n` terms and rescales adds
//! `2 n 2^-24 Vmax_dec`. The gathered F32 side is the exact per-head F32 kernel:
//! gfx1201 has no F32 gathered route (and the dense F16 WMMA route of the
//! production F32 dispatch ignores selections), so the production batch is not
//! the reference there.
//!
//! `#[ignore]`d: needs an exact gfx1201 device (the tests panic on any other
//! arch). Run explicitly, one test thread:
//!
//!   cargo test -p rdna-compute --release --features lab --test qsa_fp8_state \
//!       -- --ignored --nocapture --test-threads=1
//!
//! No environment variable is set by this file.

#![cfg(feature = "lab")]

use rayon::prelude::*;
use rdna_compute::tensor_ops::{
    indexed_attention_attention, indexed_attention_attention_batch,
    indexed_attention_attention_batch_exact, indexed_attention_cache_append_batch,
    indexed_attention_gathered_batch, qsa_gathered_wmma_enabled, IndexedAttentionAttention,
    IndexedAttentionAttentionBatch, IndexedAttentionCacheAppendBatch, QsaKvFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const NH: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
/// Bytes of one Fp8 K or V token row: codes of every KV head, then f16 scales.
const ROW_BYTES: usize = NKV * (HD + 2);
/// Cache rows: above the longest context (8195) with slack, so tokens in
/// [visible, FULL_CAP) are in-bounds cache rows the kernels must still reject.
const FULL_CAP: usize = 8300;
/// Budget large enough never to bind.
const HUGE: usize = 1 << 20;
const SENTINEL: u32 = 0x7fc0_dead;
const LENS: [usize; 4] = [64, 1131, 2048, 8195];
const GATHER_BUDGET: usize = 512;
const GATHER_COMPRESS: usize = 4;
const GATHER_CAPACITY: usize = GATHER_BUDGET * GATHER_COMPRESS + GATHER_COMPRESS - 1;
/// F16 relative rounding `u` and absolute half step `a` of the subnormal range.
const U16: f64 = 1.0 / 2048.0;
const A16: f64 = 1.0 / 33_554_432.0;

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Deterministic value in [-scale, scale).
    fn value(&mut self, scale: f32) -> f32 {
        (self.next() as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 2.0 * scale - scale
    }
}

// ---------------------------------------------------------------------------
// CPU model of the E4M3 store
// ---------------------------------------------------------------------------

/// Magnitude of the non-negative OCP E4M3 code `c` (0..=127; 127 is NaN).
fn e4m3_mag(c: u8) -> f32 {
    let e = (c >> 3) as i32;
    let m = (c & 7) as f32;
    if c == 0x7f {
        f32::NAN
    } else if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (8.0 + m) * 2f32.powi(e - 10)
    }
}

fn e4m3_table() -> [f32; 128] {
    let mut t = [0f32; 128];
    for (c, slot) in t.iter_mut().enumerate() {
        *slot = e4m3_mag(c as u8);
    }
    t
}

/// f32 -> f16 bits, round to nearest even (finite inputs; overflow to inf).
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let abs = b & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        return sign | 0x7c00 | if abs > 0x7f80_0000 { 0x200 } else { 0 };
    }
    if abs >= 0x477f_f000 {
        return sign | 0x7c00;
    }
    // At or below 2^-25 the result rounds to zero (2^-25 is a tie, even is 0).
    if abs <= 0x3300_0000 {
        return sign;
    }
    if abs < 0x3880_0000 {
        // f16 subnormal result: unit 2^-24.
        let e = (abs >> 23) as i32;
        let mant = (abs & 0x7f_ffff) | 0x80_0000;
        let shift = (126 - e) as u32;
        let q = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        return sign | q as u16;
    }
    let m = abs - 0x3800_0000;
    let lsb = (m >> 13) & 1;
    sign | ((m + 0xfff + lsb) >> 13) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as f64;
    assert!(e != 31, "f16 inf/nan scale {bits:#06x}");
    let v = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1024.0 + m) * 2f64.powi(e - 25)
    };
    (sign * v) as f32
}

/// f16 scale bits of a head with absolute maximum `amax` (`qsa_kv_store`).
fn scale_bits(amax: f32) -> u16 {
    let mut s16 = if amax == 0.0 {
        0x3c00
    } else {
        f32_to_f16_bits(amax / 448.0)
    };
    if amax > 0.0 && s16 == 0 {
        s16 = 1;
    }
    let sf = f16_to_f32(s16);
    if amax > 0.0 && 448.0f32 * sf < amax {
        s16 = if s16 >= 0x7bff { 0x7bff } else { s16 + 1 };
    }
    s16
}

/// Nearest-even E4M3 code (sign in bit 7, finite saturation at 448) of `v`.
fn e4m3_encode(v: f32, tab: &[f32; 128]) -> u8 {
    assert!(!v.is_nan(), "NaN in the E4M3 encoder");
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs() as f64;
    let idx = tab[..127].partition_point(|&d| (d as f64) < a);
    let code = if idx >= 127 {
        126
    } else if tab[idx] as f64 == a {
        idx
    } else {
        let lo = idx - 1;
        let dl = a - tab[lo] as f64;
        let dh = tab[idx] as f64 - a;
        if dl < dh {
            lo
        } else if dh < dl {
            idx
        } else if lo % 2 == 0 {
            lo
        } else {
            idx
        }
    };
    sign | code as u8
}

/// One Fp8 token row (codes of every KV head, then little-endian f16 scales).
fn encode_row(x: &[f32], tab: &[f32; 128]) -> Vec<u8> {
    assert_eq!(x.len(), NKV * HD);
    let mut row = vec![0u8; ROW_BYTES];
    for h in 0..NKV {
        let xs = &x[h * HD..(h + 1) * HD];
        let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let s16 = scale_bits(amax);
        let sf = f16_to_f32(s16);
        for (c, &v) in xs.iter().enumerate() {
            row[h * HD + c] = e4m3_encode(v / sf, tab);
        }
        row[NKV * HD + 2 * h..NKV * HD + 2 * h + 2].copy_from_slice(&s16.to_le_bytes());
    }
    row
}

/// Decoded element of code `code` with f32 scale `s` (exact in f32).
fn e4m3_decode(code: u8, s: f32, tab: &[f32; 128]) -> f32 {
    let m = tab[(code & 0x7f) as usize];
    assert!(!m.is_nan(), "E4M3 NaN code {code:#04x} in a cache row");
    let v = m * s;
    if code & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// Element error allowance `E(x, s)` of the module docs.
fn elem_bound(x: f32, s: f32) -> f64 {
    let ax = (x as f64).abs();
    ax / 16.0 + s as f64 / 1024.0 + 2f64.powi(-22) * ax
}

// ---------------------------------------------------------------------------
// Append test data
// ---------------------------------------------------------------------------

const KINDS: usize = 13;
const KIND_NAMES: [&str; KINDS] = [
    "uniform+-1",
    "log-uniform dynamic range",
    "zero",
    "all -0",
    "negative with -0",
    "exact e4m3 ties",
    "perturbed e4m3 ties",
    "e4m3 grid scale 1",
    "e4m3 grid odd scale",
    "tiny amax (scale underflow/bump)",
    "single nonzero element",
    "huge amax",
    "mid-range amax (scale bump)",
];

const SCALE_BITS: [u16; 6] = [0x3c00, 0x3555, 0x1234, 0x0400, 0x0010, 0x5a00];

/// Uniform in +-amax with one element forced to +-amax.
fn rand_with_amax(rng: &mut Rng, amax: f32) -> Vec<f32> {
    let mut x: Vec<f32> = (0..HD).map(|_| rng.value(amax)).collect();
    let pos = rng.below(HD as u64) as usize;
    x[pos] = if rng.below(2) == 0 { amax } else { -amax };
    x
}

/// amax 448 * s: every midpoint between consecutive positive codes (times s,
/// exact in f32) as +/- values, optionally moved one f32 ulp away from the tie.
fn tie_head(tab: &[f32; 128], s: f32, perturb: bool) -> Vec<f32> {
    let mut x = vec![0f32; HD];
    x[0] = 448.0 * s;
    for c in 0..126 {
        let mut m = (tab[c] + tab[c + 1]) * 0.5 * s;
        if perturb {
            let b = m.to_bits();
            m = f32::from_bits(if c % 2 == 0 { b + 1 } else { b - 1 });
        }
        x[1 + c] = m;
        x[127 + c] = -m;
    }
    x[253] = -0.0;
    x[254] = 0.0;
    x[255] = -(2f32.powi(-11)) * s;
    x
}

/// amax 448 * s: every code point (subnormals included) times s, both signs.
fn grid_head(tab: &[f32; 128], s: f32) -> Vec<f32> {
    let mut x = vec![0f32; HD];
    x[0] = 448.0 * s;
    for c in 0..=126 {
        x[1 + c] = tab[c] * s;
        x[128 + c] = -(tab[c] * s);
    }
    x[255] = 0.0;
    x
}

fn gen_head(kind: usize, rng: &mut Rng, tab: &[f32; 128]) -> Vec<f32> {
    match kind {
        0 => (0..HD).map(|_| rng.value(1.0)).collect(),
        1 => (0..HD)
            .map(|_| {
                let u = rng.value(1.0) * 0.5 + 0.5;
                let mag = 10f32.powf(-4.0 + 7.0 * u);
                if rng.below(2) == 0 {
                    mag
                } else {
                    -mag
                }
            })
            .collect(),
        2 => vec![0.0; HD],
        3 => vec![-0.0; HD],
        4 => (0..HD)
            .map(|c| {
                if c % 7 == 3 {
                    -0.0
                } else {
                    -(rng.value(1.0) + 1.0)
                }
            })
            .collect(),
        5 => {
            let s = f16_to_f32(SCALE_BITS[rng.below(6) as usize]);
            tie_head(tab, s, false)
        }
        6 => {
            let s = f16_to_f32(SCALE_BITS[rng.below(6) as usize]);
            tie_head(tab, s, true)
        }
        7 => grid_head(tab, 1.0),
        8 => grid_head(tab, f16_to_f32(SCALE_BITS[1 + rng.below(5) as usize])),
        9 => {
            const AMAX: [f32; 8] = [1e-9, 3e-8, 1e-6, 1.5e-5, 3e-5, 6e-5, 1e-4, 2.9e-5];
            let amax = AMAX[rng.below(8) as usize];
            rand_with_amax(rng, amax)
        }
        10 => {
            let mut x = vec![0.0; HD];
            let pos = rng.below(HD as u64) as usize;
            x[pos] = rng.value(10.0) + 10.5;
            if rng.below(2) == 0 {
                x[pos] = -x[pos];
            }
            x
        }
        11 => {
            let amax = if rng.below(2) == 0 { 1.0e7 } else { 2.9e7 };
            rand_with_amax(rng, amax)
        }
        12 => {
            const AMAX: [f32; 6] = [1.0, 0.7, 3.3, 100.0, 0.013, 12345.6];
            let amax = AMAX[rng.below(6) as usize];
            rand_with_amax(rng, amax)
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

fn init_gpu() -> Gpu {
    let gpu = Gpu::init().expect("HIP device init");
    assert_eq!(
        gpu.arch.as_str(),
        "gfx1201",
        "these tests need an exact gfx1201 device"
    );
    gpu
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().expect("device sync");
}

fn free(gpu: &mut Gpu, tensors: Vec<GpuTensor>) {
    for t in tensors {
        gpu.free_tensor(t).expect("free tensor");
    }
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let data = vec![f32::from_bits(SENTINEL); n];
    gpu.upload_f32(&data, &[n]).expect("upload sentinel")
}

fn upload_selected(gpu: &Gpu, rows: &[Vec<i32>]) -> GpuTensor {
    let bytes: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("upload selected")
}

// ---------------------------------------------------------------------------
// Test 1: append encodes E4M3 rows
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs an exact gfx1201 device"]
fn qsa_fp8_append_encodes_e4m3_rows() {
    let mut gpu = init_gpu();
    let tab = e4m3_table();
    const P: usize = 7;
    const N: usize = 52;
    const ROWS_CAP: usize = P + N + 9;
    let width = NKV * HD;

    let mut rng = Rng(0xf8e4_0001);
    let mut k = vec![0f32; N * width];
    let mut v = vec![0f32; N * width];
    // Generator kind of (K = 0 / V = 1, row, head), flat.
    let mut kinds = vec![0usize; 2 * N * NKV];
    for i in 0..N {
        for h in 0..NKV {
            let kk = (i + 5 * h) % KINDS;
            let vk = (i + 7 + 11 * h) % KINDS;
            kinds[i * NKV + h] = kk;
            kinds[(N + i) * NKV + h] = vk;
            let off = i * width + h * HD;
            k[off..off + HD].copy_from_slice(&gen_head(kk, &mut rng, &tab));
            v[off..off + HD].copy_from_slice(&gen_head(vk, &mut rng, &tab));
        }
    }

    let k_in = gpu.upload_f32(&k, &[k.len()]).expect("upload K");
    let v_in = gpu.upload_f32(&v, &[v.len()]).expect("upload V");
    let k32 = gpu
        .zeros(&[ROWS_CAP * width], DType::F32)
        .expect("F32 K cache");
    let v32 = gpu
        .zeros(&[ROWS_CAP * width], DType::F32)
        .expect("F32 V cache");
    let k8 = gpu
        .zeros(&[ROWS_CAP * ROW_BYTES], DType::Raw)
        .expect("Fp8 K cache");
    let v8 = gpu
        .zeros(&[ROWS_CAP * ROW_BYTES], DType::Raw)
        .expect("Fp8 V cache");
    for (format, kc, vc) in [(QsaKvFormat::F32, &k32, &v32), (QsaKvFormat::Fp8, &k8, &v8)] {
        indexed_attention_cache_append_batch(
            &mut gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &k_in,
                value: &v_in,
                full_keys: kc,
                full_values: vc,
                rows: N,
                position_start: P,
                kv_heads: NKV,
                head_dim: HD,
                format,
            },
        )
        .unwrap_or_else(|e| panic!("append {format:?} failed: {e:?}"));
    }
    sync(&gpu);
    let f32_k = gpu.download_f32(&k32).expect("download F32 K");
    let f32_v = gpu.download_f32(&v32).expect("download F32 V");
    let fp8_k = gpu.download_raw_bytes(&k8).expect("download Fp8 K");
    let fp8_v = gpu.download_raw_bytes(&v8).expect("download Fp8 V");
    free(&mut gpu, vec![k_in, v_in, k32, v32, k8, v8]);

    let mut fails: Vec<String> = Vec::new();
    for (name, which, host, f32c, fp8c) in [
        ("K", 0usize, &k, &f32_k, &fp8_k),
        ("V", 1, &v, &f32_v, &fp8_v),
    ] {
        assert_eq!(f32c.len(), ROWS_CAP * width);
        assert_eq!(fp8c.len(), ROWS_CAP * ROW_BYTES);
        // Prefix and suffix stay zero in both caches.
        if let Some(i) = f32c[..P * width].iter().position(|x| x.to_bits() != 0) {
            fails.push(format!("{name}: F32 cache prefix element {i} is not +0"));
        }
        if let Some(i) = f32c[(P + N) * width..]
            .iter()
            .position(|x| x.to_bits() != 0)
        {
            fails.push(format!("{name}: F32 cache suffix element {i} is not +0"));
        }
        if let Some(i) = fp8c[..P * ROW_BYTES].iter().position(|&b| b != 0) {
            fails.push(format!("{name}: Fp8 cache prefix byte {i} is not 0"));
        }
        if let Some(i) = fp8c[(P + N) * ROW_BYTES..].iter().position(|&b| b != 0) {
            fails.push(format!("{name}: Fp8 cache suffix byte {i} is not 0"));
        }
        let (mut bad_rows, mut first) = (0usize, None::<String>);
        let mut worst_ratio = 0.0f64;
        for i in 0..N {
            let pos = P + i;
            let x = &host[i * width..(i + 1) * width];
            // The F32 append is a plain copy.
            let got32 = &f32c[pos * width..(pos + 1) * width];
            if let Some(j) = (0..width).find(|&j| got32[j].to_bits() != x[j].to_bits()) {
                fails.push(format!(
                    "{name}: F32 cache row {i} (position {pos}) element {j}: got {:#010x}, input {:#010x}",
                    got32[j].to_bits(),
                    x[j].to_bits()
                ));
            }
            let want = encode_row(x, &tab);
            let got = &fp8c[pos * ROW_BYTES..(pos + 1) * ROW_BYTES];
            if let Some(b) = (0..ROW_BYTES).find(|&b| got[b] != want[b]) {
                bad_rows += 1;
                if first.is_none() {
                    let msg = if b < NKV * HD {
                        let (h, c) = (b / HD, b % HD);
                        let s_got =
                            u16::from_le_bytes([got[NKV * HD + 2 * h], got[NKV * HD + 2 * h + 1]]);
                        let s_want = u16::from_le_bytes([
                            want[NKV * HD + 2 * h],
                            want[NKV * HD + 2 * h + 1],
                        ]);
                        format!(
                            "{name} row {i} (position {pos}) head {h} channel {c} [{}]: code got {:#04x} want {:#04x} for input {:e} ({:#010x}); scale bits got {s_got:#06x} want {s_want:#06x}",
                            KIND_NAMES[kinds[(which * N + i) * NKV + h]],
                            got[b],
                            want[b],
                            x[b],
                            x[b].to_bits()
                        )
                    } else {
                        let h = (b - NKV * HD) / 2;
                        format!(
                            "{name} row {i} (position {pos}) head {h} [{}]: scale byte {} got {:#04x} want {:#04x} (scale bits got {:#06x} want {:#06x})",
                            KIND_NAMES[kinds[(which * N + i) * NKV + h]],
                            (b - NKV * HD) % 2,
                            got[b],
                            want[b],
                            u16::from_le_bytes([got[NKV * HD + 2 * h], got[NKV * HD + 2 * h + 1]]),
                            u16::from_le_bytes([want[NKV * HD + 2 * h], want[NKV * HD + 2 * h + 1]])
                        )
                    };
                    first = Some(msg);
                }
            }
            // Decoded element against the F32 cache element.
            for h in 0..NKV {
                let s = f16_to_f32(u16::from_le_bytes([
                    got[NKV * HD + 2 * h],
                    got[NKV * HD + 2 * h + 1],
                ]));
                for c in 0..HD {
                    let d = e4m3_decode(got[h * HD + c], s, &tab);
                    let xe = got32[h * HD + c];
                    let allow = elem_bound(xe, s);
                    let diff = (d as f64 - xe as f64).abs();
                    worst_ratio = worst_ratio.max(diff / allow);
                    if diff.is_nan() || diff > allow {
                        fails.push(format!(
                            "{name} row {i} (position {pos}) head {h} channel {c} [{}]: decoded {d:e} vs F32 {xe:e}: |diff| {diff:.3e} > allowance {allow:.3e} (scale {s:e}, code {:#04x})",
                            KIND_NAMES[kinds[(which * N + i) * NKV + h]],
                            got[h * HD + c]
                        ));
                    }
                }
            }
        }
        println!(
            "{name}: {N} rows at position {P}: {bad_rows} rows differ from the CPU encoder; worst decoded error / allowance = {worst_ratio:.3}"
        );
        if let Some(m) = first {
            fails.push(format!("first byte mismatch ({bad_rows} rows differ): {m}"));
        }
    }
    finish(fails);
}

fn finish(fails: Vec<String>) {
    if !fails.is_empty() {
        for f in fails.iter().take(40) {
            eprintln!("FAIL: {f}");
        }
        panic!("{} failure(s); first: {}", fails.len(), fails[0]);
    }
}

// ---------------------------------------------------------------------------
// Shared attention fixtures
// ---------------------------------------------------------------------------

/// F32 and Fp8 caches of one K/V stream (all `FULL_CAP` rows), the decoded Fp8
/// values on the host and the per-KV-head maxima of the module-doc bound.
struct Cache {
    kd: Vec<f32>,
    vd: Vec<f32>,
    k32: GpuTensor,
    v32: GpuTensor,
    k8: GpuTensor,
    v8: GpuTensor,
    /// Maxima of `E(x, s)` over K and V.
    ek: [f64; NKV],
    ev: [f64; NKV],
    /// Maximum |V| of the F32 cache; maxima of |decoded K| and |decoded V|.
    vmax: [f64; NKV],
    kdmax: [f64; NKV],
    vdmax: [f64; NKV],
}

fn gen_cache(rng: &mut Rng) -> Vec<f32> {
    const GAIN: [f32; 4] = [1.0, 0.5, 0.25, 0.0625];
    let mut x = vec![0f32; FULL_CAP * NKV * HD];
    for tok in 0..FULL_CAP {
        for h in 0..NKV {
            let gain = GAIN[(tok * 3 + h) % 4];
            for c in 0..HD {
                x[(tok * NKV + h) * HD + c] = rng.value(1.0) * gain;
            }
        }
    }
    x
}

/// Decode the Fp8 cache bytes; per KV head: (max E, max |x|, max |decoded|).
fn decode_cache(
    host: &[f32],
    bytes: &[u8],
    tab: &[f32; 128],
    name: &str,
) -> (Vec<f32>, [f64; NKV], [f64; NKV], [f64; NKV]) {
    assert_eq!(bytes.len(), FULL_CAP * ROW_BYTES);
    let mut dec = vec![0f32; host.len()];
    let (mut emax, mut xmax, mut dmax) = ([0f64; NKV], [0f64; NKV], [0f64; NKV]);
    for tok in 0..FULL_CAP {
        let row = &bytes[tok * ROW_BYTES..(tok + 1) * ROW_BYTES];
        for h in 0..NKV {
            let s = f16_to_f32(u16::from_le_bytes([
                row[NKV * HD + 2 * h],
                row[NKV * HD + 2 * h + 1],
            ]));
            for c in 0..HD {
                let i = (tok * NKV + h) * HD + c;
                let d = e4m3_decode(row[h * HD + c], s, tab);
                let x = host[i];
                let e = elem_bound(x, s);
                assert!(
                    (d as f64 - x as f64).abs() <= e,
                    "{name} cache token {tok} head {h} channel {c}: decoded {d:e} vs {x:e} exceeds E = {e:e}"
                );
                dec[i] = d;
                emax[h] = emax[h].max(e);
                xmax[h] = xmax[h].max((x as f64).abs());
                dmax[h] = dmax[h].max((d as f64).abs());
            }
        }
    }
    (dec, emax, xmax, dmax)
}

impl Cache {
    fn build(gpu: &mut Gpu) -> Self {
        let tab = e4m3_table();
        let n = FULL_CAP * NKV * HD;
        let mut rng = Rng(0xcac4_e001);
        let kf = gen_cache(&mut rng);
        let vf = gen_cache(&mut rng);
        let k32 = gpu.upload_f32(&kf, &[n]).expect("upload F32 K cache");
        let v32 = gpu.upload_f32(&vf, &[n]).expect("upload F32 V cache");
        let k8 = gpu
            .zeros(&[FULL_CAP * ROW_BYTES], DType::Raw)
            .expect("Fp8 K cache");
        let v8 = gpu
            .zeros(&[FULL_CAP * ROW_BYTES], DType::Raw)
            .expect("Fp8 V cache");
        indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &k32,
                value: &v32,
                full_keys: &k8,
                full_values: &v8,
                rows: FULL_CAP,
                position_start: 0,
                kv_heads: NKV,
                head_dim: HD,
                format: QsaKvFormat::Fp8,
            },
        )
        .expect("Fp8 cache append");
        sync(gpu);
        let kb = gpu.download_raw_bytes(&k8).expect("download Fp8 K");
        let vb = gpu.download_raw_bytes(&v8).expect("download Fp8 V");
        let (kd, ek, _kxmax, kdmax) = decode_cache(&kf, &kb, &tab, "K");
        let (vd, ev, vmax, vdmax) = decode_cache(&vf, &vb, &tab, "V");
        Cache {
            kd,
            vd,
            k32,
            v32,
            k8,
            v8,
            ek,
            ev,
            vmax,
            kdmax,
            vdmax,
        }
    }

    fn free(self, gpu: &mut Gpu) {
        free(gpu, vec![self.k32, self.v32, self.k8, self.v8]);
    }

    fn tensors(&self, format: QsaKvFormat) -> (&GpuTensor, &GpuTensor) {
        match format {
            QsaKvFormat::F32 => (&self.k32, &self.v32),
            QsaKvFormat::Fp8 => (&self.k8, &self.v8),
            other => panic!("no {other:?} cache in this test"),
        }
    }
}

/// `rows` query rows: per head `HD` query values (+/-`qscale`) then `HD` gate
/// values (+/-2).
fn make_q(rng: &mut Rng, rows: usize, qscale: f32) -> Vec<f32> {
    let mut q = Vec::with_capacity(rows * 2 * NH * HD);
    for _ in 0..rows * NH {
        q.extend((0..HD).map(|_| rng.value(qscale)));
        q.extend((0..HD).map(|_| rng.value(2.0)));
    }
    q
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Mixed,
    AllValid,
    AllInvalid,
}

#[derive(Clone, Copy)]
struct Layout {
    budget: usize,
    compress: usize,
    capacity: usize,
}

const GATHER_LAYOUT: Layout = Layout {
    budget: GATHER_BUDGET,
    compress: GATHER_COMPRESS,
    capacity: GATHER_CAPACITY,
};

/// The kernels' own selected length of the row with `visible` keys.
fn selected_len(visible: usize, l: Layout) -> usize {
    let blocks = visible / l.compress;
    (l.budget.min(blocks) * l.compress + visible - blocks * l.compress).min(l.capacity)
}

fn is_valid(token: i32, hi: usize) -> bool {
    token >= 0 && (token as usize) < hi
}

/// One `capacity`-wide selected row: valid tokens are below `visible` (the
/// batched kernels' bound, and the single-row entry's `full_capacity`); the
/// invalid ones are negative, in [visible, FULL_CAP) or >= FULL_CAP. Slots at or
/// past `sel_len` hold valid-looking tokens unless the mode is AllInvalid.
fn gen_row(rng: &mut Rng, mode: Mode, visible: usize, sel_len: usize, capacity: usize) -> Vec<i32> {
    let valid = |rng: &mut Rng| -> i32 { rng.below(visible as u64) as i32 };
    let invalid = |rng: &mut Rng, kind: usize| -> i32 {
        match kind % 3 {
            0 => -1 - rng.below(1000) as i32,
            1 if visible < FULL_CAP => {
                visible as i32 + rng.below((FULL_CAP - visible) as u64) as i32
            }
            1 => i32::MIN + rng.below(1000) as i32,
            _ => {
                if rng.below(2) == 0 {
                    FULL_CAP as i32 + rng.below(100) as i32
                } else {
                    i32::MAX - rng.below(100) as i32
                }
            }
        }
    };
    (0..capacity)
        .map(|j| {
            if j >= sel_len {
                return if mode == Mode::AllInvalid {
                    invalid(rng, j)
                } else {
                    valid(rng)
                };
            }
            match mode {
                Mode::AllValid => valid(rng),
                Mode::AllInvalid => invalid(rng, j),
                Mode::Mixed => match rng.below(10) {
                    0..=6 => valid(rng),
                    7 => invalid(rng, 0),
                    8 => invalid(rng, 1),
                    _ => invalid(rng, 2),
                },
            }
        })
        .collect()
}

struct Case {
    rows: usize,
    position_start: usize,
    layout: Layout,
    q: Vec<f32>,
    sel: Vec<Vec<i32>>,
    sel_len: Vec<usize>,
    /// Exclusive upper bound of the valid tokens of each row.
    hi: Vec<usize>,
}

/// `rows` rows ending at `len` visible tokens (`single`: the single-row entry
/// over `len` selected slots). Row 1 of every batch of two or more rows has no
/// valid token; every third row from 2 is all-valid, the rest mixed.
fn build_case(
    rng: &mut Rng,
    rows: usize,
    len: usize,
    layout: Layout,
    single: bool,
    qscale: f32,
) -> Case {
    let rows = if single { 1 } else { rows };
    let position_start = len - rows;
    let (mut sel, mut sel_len, mut hi) = (Vec::new(), Vec::new(), Vec::new());
    let layout = if single {
        Layout {
            capacity: len,
            ..layout
        }
    } else {
        layout
    };
    for r in 0..rows {
        let visible = position_start + r + 1;
        let n = if single {
            len
        } else {
            selected_len(visible, layout)
        };
        let mode = if r == 1 {
            Mode::AllInvalid
        } else if r >= 2 && r % 3 == 2 {
            Mode::AllValid
        } else {
            Mode::Mixed
        };
        sel.push(gen_row(rng, mode, visible, n, layout.capacity));
        sel_len.push(n);
        hi.push(visible.min(FULL_CAP));
    }
    Case {
        rows,
        position_start,
        layout,
        q: make_q(rng, rows, qscale),
        sel,
        sel_len,
        hi,
    }
}

impl Case {
    fn valid_rows(&self) -> Vec<bool> {
        (0..self.rows)
            .map(|r| {
                self.sel[r][..self.sel_len[r]]
                    .iter()
                    .any(|&t| is_valid(t, self.hi[r]))
            })
            .collect()
    }

    fn upload(&self, gpu: &mut Gpu) -> (GpuTensor, GpuTensor) {
        let q = gpu.upload_f32(&self.q, &[self.q.len()]).expect("upload q");
        (q, upload_selected(gpu, &self.sel))
    }
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Single,
    Exact,
    Production,
    Gathered,
}

fn run(
    gpu: &mut Gpu,
    route: Route,
    format: QsaKvFormat,
    cache: &Cache,
    q: &GpuTensor,
    sel: &GpuTensor,
    case: &Case,
) -> Vec<f32> {
    let (k, v) = cache.tensors(format);
    let out = sentinel(gpu, case.rows * NH * HD);
    let result = match route {
        Route::Single => indexed_attention_attention(
            gpu,
            &IndexedAttentionAttention {
                q_with_gate: q,
                full_keys: k,
                full_values: v,
                selected: sel,
                output: &out,
                n_heads: NH,
                n_kv_heads: NKV,
                head_dim: HD,
                selected_len: case.sel_len[0],
                // The single-row kernels bound tokens only by `full_capacity`:
                // pass the visible end, as the batched kernels' cutoff. The
                // backing cache keeps all FULL_CAP rows.
                full_capacity: case.hi[0],
                end_position: case.hi[0],
                format,
            },
        ),
        _ => {
            let p = IndexedAttentionAttentionBatch {
                q_with_gate: q,
                full_keys: k,
                full_values: v,
                selected: sel,
                output: &out,
                rows: case.rows,
                position_start: case.position_start,
                n_heads: NH,
                n_kv_heads: NKV,
                head_dim: HD,
                budget_blocks: case.layout.budget,
                compress: case.layout.compress,
                capacity: case.layout.capacity,
                full_capacity: FULL_CAP,
                format,
                shape_selected: case.layout.capacity,
            };
            match route {
                Route::Exact => indexed_attention_attention_batch_exact(gpu, &p),
                Route::Production => indexed_attention_attention_batch(gpu, &p),
                Route::Gathered => indexed_attention_gathered_batch(gpu, &p, false),
                Route::Single => unreachable!(),
            }
        }
    };
    if let Err(e) = result {
        panic!(
            "{route:?} {format:?} failed (rows {}, position_start {}): {e:?}",
            case.rows, case.position_start
        );
    }
    sync(gpu);
    let got = gpu.download_f32(&out).expect("download output");
    free(gpu, vec![out]);
    got
}

/// Same-codec G1, independent of the approximate decoded/F32 checks below.
/// Recorded binds the existing AR arithmetic scope; it does not pretend to
/// capture a graph. Receipts describe only the actual terminal invocation.
fn check_fp8_wide_bytes(gpu: &mut Gpu, cache: &Cache) {
    use rdna_compute::prefill_invariant::Qwen4VerifyReference;
    use rdna_compute::tensor_ops::qsa_decode_attention_policy_for_row_for_test;

    let old_wide = gpu.set_wide_verify_rows(64);
    let original_k = gpu.download_raw_bytes(&cache.k8).expect("snapshot FP8 K");
    let original_v = gpu.download_raw_bytes(&cache.v8).expect("snapshot FP8 V");
    let guard = 37;
    let row_elements = NH * HD;
    for mode in [Qwen4VerifyReference::Eager, Qwen4VerifyReference::Recorded] {
        let old_mode = gpu.set_qwen4_verify_reference(Some(mode));
        for seed in [0x47f8_0001, 0x47f8_0002] {
            let mut rng = Rng(seed);
            for rows in 1..=64 {
                // 8188 is the last parallel selection: selected*8 + 32 <=
                // 65536. Include windows crossing it, short/empty selection,
                // and every compression residue at a nonzero position.
                for (position, layout) in [
                    (
                        1,
                        Layout {
                            budget: 0,
                            compress: 1,
                            capacity: 1,
                        },
                    ),
                    (
                        67,
                        Layout {
                            budget: 1,
                            compress: 4,
                            capacity: 7,
                        },
                    ),
                    (
                        128,
                        Layout {
                            budget: HUGE,
                            compress: 4,
                            capacity: 64,
                        },
                    ),
                    (
                        129,
                        Layout {
                            budget: HUGE,
                            compress: 4,
                            capacity: 64,
                        },
                    ),
                    (
                        130,
                        Layout {
                            budget: HUGE,
                            compress: 4,
                            capacity: 64,
                        },
                    ),
                    (
                        131,
                        Layout {
                            budget: HUGE,
                            compress: 4,
                            capacity: 64,
                        },
                    ),
                    (
                        8187,
                        Layout {
                            budget: HUGE,
                            compress: 4,
                            capacity: 8195,
                        },
                    ),
                    (
                        8188,
                        Layout {
                            budget: HUGE,
                            compress: 4,
                            capacity: 8195,
                        },
                    ),
                ] {
                    let case = build_case(&mut rng, rows, position + rows, layout, false, 3.0);
                    let mut outputs = Vec::new();
                    for singleton in [false, true] {
                        let mut live = Vec::new();
                        for r in 0..if singleton { rows } else { 1 } {
                            let count = if singleton { 1 } else { rows };
                            let first = if singleton { r } else { 0 };
                            let q_host = &case.q
                                [first * 2 * row_elements..(first + count) * 2 * row_elements];
                            let q = gpu.upload_f32(q_host, &[q_host.len()]).expect("G1 query");
                            let selected = upload_selected(gpu, &case.sel[first..first + count]);
                            let q_before = gpu.download_raw_bytes(&q).expect("snapshot query");
                            let selected_before = gpu
                                .download_raw_bytes(&selected)
                                .expect("snapshot selection");
                            // Retain the real owning allocation around its view:
                            // prefix, two inactive rows, and non-row-aligned tail.
                            let active = count * row_elements;
                            let owner = sentinel(gpu, (count + 2) * row_elements + 2 * guard);
                            let out = owner.sub_offset(guard, (count + 2) * row_elements);
                            let before = gpu
                                .download_raw_bytes(&owner)
                                .expect("snapshot output owner");
                            let p = IndexedAttentionAttentionBatch {
                                q_with_gate: &q,
                                full_keys: &cache.k8,
                                full_values: &cache.v8,
                                selected: &selected,
                                output: &out,
                                rows: count,
                                position_start: position + first,
                                n_heads: NH,
                                n_kv_heads: NKV,
                                head_dim: HD,
                                budget_blocks: layout.budget,
                                compress: layout.compress,
                                capacity: layout.capacity,
                                full_capacity: FULL_CAP,
                                format: QsaKvFormat::Fp8,
                                shape_selected: layout.capacity,
                            };
                            let policies: Vec<_> = (0..count)
                                .map(|i| {
                                    qsa_decode_attention_policy_for_row_for_test(gpu, &p, i)
                                        .expect("G1 actual wide production row not admitted")
                                })
                                .collect();
                            indexed_attention_attention_batch(gpu, &p)
                                .expect("G1 actual FP8 production dispatch");
                            let symbol = gpu
                                .last_launched_kernel()
                                .expect("G1 missing actual launch")
                                .to_owned();
                            assert!(
                                matches!(
                                    symbol.as_str(),
                                    "indexed_attention_attention_fp8_batched"
                                        | "indexed_attention_attention_fp8_batched_serial"
                                ),
                                "G1 unexpected actual FP8 production kernel: {symbol}"
                            );
                            let (module, container, device) = gpu
                                .lab_kernel_device_identity(&symbol)
                                .expect("G1 missing loaded raw device ELF identity");
                            sync(gpu);
                            let after = gpu
                                .download_raw_bytes(&owner)
                                .expect("G1 output owner bytes");
                            assert_eq!(
                                &after[..guard * 4],
                                &before[..guard * 4],
                                "G1 prefix overwrite R={rows} mode={mode:?} position={position}"
                            );
                            assert_eq!(&after[(guard + active) * 4..], &before[(guard + active) * 4..],
                                "G1 inactive/tail overwrite R={rows} mode={mode:?} position={position}");
                            assert_eq!(gpu.download_raw_bytes(&q).expect("query after"), q_before);
                            assert_eq!(
                                gpu.download_raw_bytes(&selected).expect("selection after"),
                                selected_before
                            );
                            assert_eq!(
                                gpu.download_raw_bytes(&cache.k8).expect("K after"),
                                original_k
                            );
                            assert_eq!(
                                gpu.download_raw_bytes(&cache.v8).expect("V after"),
                                original_v
                            );
                            eprintln!("G1 FP8 arch={} R={rows} mode={mode:?} seed={seed} singleton={singleton} row={first} position={} rows={count} budget={} compress={} capacity={} shape_selected={} policies={policies:?} live_bytes={} allocation_bytes={} query_bytes={} selection_bytes={} kv_bytes={} terminal_symbol={symbol} module={module} container_sha256={container} raw_device_elf_sha256={device}",
                                gpu.arch, position + first, layout.budget, layout.compress,
                                layout.capacity, layout.capacity, active * 4, after.len(),
                                q_before.len(), selected_before.len(), original_k.len());
                            eprintln!("G1 FP8 output_live_offset_bytes={} prefix_guard_bytes={} inactive_bytes={} tail_guard_bytes={}",
                                guard * 4, guard * 4, 2 * row_elements * 4, guard * 4);
                            live.extend_from_slice(&after[guard * 4..(guard + active) * 4]);
                            drop(out);
                            free(gpu, vec![q, selected, owner]);
                        }
                        outputs.push(live);
                    }
                    assert_eq!(outputs[0], outputs[1],
                        "G1 FP8 full live bytes R={rows} mode={mode:?} seed={seed} position={position}");
                }
            }
        }
        gpu.set_qwen4_verify_reference(old_mode);
    }
    gpu.set_wide_verify_rows(old_wide);
}

// ---------------------------------------------------------------------------
// Test 2: GPU Fp8 attention against the f64 decoded reference
// ---------------------------------------------------------------------------

fn dot(q: &[f64], k: &[f32]) -> f64 {
    let mut a = [0f64; 4];
    for (qc, kc) in q.as_chunks::<4>().0.iter().zip(k.as_chunks::<4>().0) {
        for i in 0..4 {
            a[i] += qc[i] * kc[i] as f64;
        }
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

/// f64 attention over the decoded K/V: `rows * NH * HD` outputs, exactly zero
/// for a row/head without a valid token.
fn cpu_reference(case: &Case, kd: &[f32], vd: &[f32]) -> Vec<f64> {
    let per_head: Vec<Vec<f64>> = (0..case.rows * NH)
        .into_par_iter()
        .map(|rh| {
            let (r, h) = (rh / NH, rh % NH);
            let kv = h / (NH / NKV);
            let qh = &case.q[rh * 2 * HD..(rh + 1) * 2 * HD];
            let q64: Vec<f64> = qh[..HD].iter().map(|&x| x as f64).collect();
            let toks: Vec<usize> = case.sel[r][..case.sel_len[r]]
                .iter()
                .filter(|&&t| is_valid(t, case.hi[r]))
                .map(|&t| t as usize)
                .collect();
            let mut out = vec![0f64; HD];
            if toks.is_empty() {
                return out;
            }
            let scores: Vec<f64> = toks
                .iter()
                .map(|&t| dot(&q64, &kd[(t * NKV + kv) * HD..(t * NKV + kv + 1) * HD]) / 16.0)
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut den = 0f64;
            for (&t, &s) in toks.iter().zip(&scores) {
                let w = (s - m).exp();
                den += w;
                let vr = &vd[(t * NKV + kv) * HD..(t * NKV + kv + 1) * HD];
                for (o, &x) in out.iter_mut().zip(vr) {
                    *o += w * x as f64;
                }
            }
            for (c, o) in out.iter_mut().enumerate() {
                let gate = qh[HD + c] as f64;
                *o = *o / den / (1.0 + (-gate).exp());
            }
            out
        })
        .collect();
    per_head.into_iter().flatten().collect()
}

fn rel_l2(want: &[f64], got: &[f32]) -> f64 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        err += (w - *g as f64).powi(2);
        norm += w * w;
    }
    (err / norm).sqrt()
}

/// Exact-route check: invalid rows exactly +0, every other head within
/// `1e-5 * maxabsref + 1e-7`. Returns (worst maxdiff, worst diff/tol).
fn check_exact(
    label: &str,
    case: &Case,
    got: &[f32],
    want: &[f64],
    fails: &mut Vec<String>,
) -> (f64, f64) {
    let valid = case.valid_rows();
    let row_len = NH * HD;
    let (mut worst, mut ratio) = (0.0f64, 0.0f64);
    for (r, &row_valid) in valid.iter().enumerate() {
        if !row_valid {
            if let Some(i) = (r * row_len..(r + 1) * row_len).find(|&i| got[i].to_bits() != 0) {
                fails.push(format!(
                    "{label}: row {r} has no valid token, expected +0, got bits {:#010x} at {}",
                    got[i].to_bits(),
                    i - r * row_len
                ));
            }
            continue;
        }
        for h in 0..NH {
            let lo = (r * NH + h) * HD;
            let (g, w) = (&got[lo..lo + HD], &want[lo..lo + HD]);
            let maxabs = w.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            let tol = 1e-5 * maxabs + 1e-7;
            let mut diff = 0.0f64;
            let mut bad = g.iter().any(|x| !x.is_finite());
            for (a, b) in g.iter().zip(w) {
                diff = diff.max((*a as f64 - b).abs());
            }
            if diff.is_nan() || diff > tol {
                bad = true;
            }
            if maxabs == 0.0 {
                fails.push(format!(
                    "{label}: row {r} head {h} reference is all zero despite valid tokens"
                ));
            }
            if bad {
                fails.push(format!(
                    "{label}: row {r} head {h} maxdiff={diff:.3e} > tol={tol:.3e} (maxabsref={maxabs:.3e})"
                ));
            }
            worst = worst.max(diff);
            ratio = ratio.max(diff / tol);
        }
    }
    println!(
        "{label}: valid rows {}/{} worst maxdiff={worst:.3e} worst diff/tol={ratio:.3}",
        valid.iter().filter(|&&v| v).count(),
        case.rows
    );
    (worst, ratio)
}

#[test]
#[ignore = "needs an exact gfx1201 device"]
fn qsa_fp8_attention_matches_decoded_reference() {
    let mut gpu = init_gpu();
    let cache = Cache::build(&mut gpu);
    check_fp8_wide_bytes(&mut gpu, &cache);
    let gathered_on = qsa_gathered_wmma_enabled(&gpu, QsaKvFormat::Fp8);
    let mut fails: Vec<String> = Vec::new();
    let mut rng = Rng(0x2ef8_0002);
    // Query scale 3: logits with a standard deviation near 1, a peaked softmax.
    const QSCALE: f32 = 3.0;

    for &len in &LENS {
        // Single-row entry: full_capacity = visible, so tokens >= visible are invalid.
        let layout = Layout {
            budget: HUGE,
            compress: 4,
            capacity: len,
        };
        let case = build_case(&mut rng, 1, len, layout, true, QSCALE);
        let (q, sel) = case.upload(&mut gpu);
        let got = run(
            &mut gpu,
            Route::Single,
            QsaKvFormat::Fp8,
            &cache,
            &q,
            &sel,
            &case,
        );
        let want = cpu_reference(&case, &cache.kd, &cache.vd);
        check_exact(&format!("len {len} single"), &case, &got, &want, &mut fails);
        free(&mut gpu, vec![q, sel]);

        // Production batch of 1..=8 rows ending at `len` visible tokens.
        for rows in 1..=8usize {
            let case = build_case(&mut rng, rows, len, layout, false, QSCALE);
            let (q, sel) = case.upload(&mut gpu);
            let got = run(
                &mut gpu,
                Route::Production,
                QsaKvFormat::Fp8,
                &cache,
                &q,
                &sel,
                &case,
            );
            let want = cpu_reference(&case, &cache.kd, &cache.vd);
            check_exact(
                &format!("len {len} batch rows {rows}"),
                &case,
                &got,
                &want,
                &mut fails,
            );
            free(&mut gpu, vec![q, sel]);
        }
    }

    if gathered_on {
        for (rows, len) in [(64usize, 64usize), (64, 1131), (512, 1131)] {
            let case = build_case(&mut rng, rows, len, GATHER_LAYOUT, false, QSCALE);
            let (q, sel) = case.upload(&mut gpu);
            let got = run(
                &mut gpu,
                Route::Gathered,
                QsaKvFormat::Fp8,
                &cache,
                &q,
                &sel,
                &case,
            );
            let want = cpu_reference(&case, &cache.kd, &cache.vd);
            let label = format!("gathered rows {rows} len {len}");
            let valid = case.valid_rows();
            let row_len = NH * HD;
            for r in 0..rows {
                let g = &got[r * row_len..(r + 1) * row_len];
                if g.iter().any(|x| !x.is_finite()) {
                    fails.push(format!("{label}: row {r} not finite/unwritten"));
                } else if !valid[r] && g.iter().any(|x| x.to_bits() != 0) {
                    fails.push(format!("{label}: row {r} has no valid token but is not +0"));
                }
            }
            let maxdiff = got
                .iter()
                .zip(&want)
                .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs()));
            let rel = rel_l2(&want, &got);
            println!(
                "{label}: valid rows {}/{rows} relL2={rel:.3e} maxabsdiff={maxdiff:.3e}",
                valid.iter().filter(|&&v| v).count()
            );
            if rel.is_nan() || rel >= 5e-3 {
                fails.push(format!("{label}: relL2 {rel:.3e} >= 5e-3"));
            }
            free(&mut gpu, vec![q, sel]);
        }
    } else {
        println!("gathered F16 WMMA route disabled for Fp8 here: gathered comparisons skipped");
    }
    cache.free(&mut gpu);
    finish(fails);
}

// ---------------------------------------------------------------------------
// Test 3: Fp8 route within the analytic bound of the F32 route
// ---------------------------------------------------------------------------

/// Bound of the module docs on `|out_fp8 - out_f32| / sigmoid(gate)` for one
/// (row, head) with query L1 norm `qabs` and selected length `n`.
fn analytic_base(cache: &Cache, kv: usize, qabs: f64, n: usize, gathered: bool) -> f64 {
    let (ek, ev, vmax) = (cache.ek[kv], cache.ev[kv], cache.vmax[kv]);
    let eps_q = ek * qabs / 16.0;
    if !gathered {
        return ev + 2.0 * vmax * eps_q.min(1.0);
    }
    let (kd, vd) = (cache.kdmax[kv], cache.vdmax[kv]);
    let (u, a, hd, n) = (U16, A16, HD as f64, n as f64);
    let dot16 = (2.0 * u + u * u) * qabs * kd + a * (1.0 + u) * (qabs + hd * kd) + hd * a * a;
    let accum = hd * 2f64.powi(-24) * (1.0 + u) * (1.0 + u) * qabs * kd;
    let score_round = 2f64.powi(-22) * qabs * kd / 16.0;
    let eps_tot = eps_q + (dot16 + accum) / 16.0 + score_round;
    let pr_rel = u + 2f64.powi(-22);
    let norm = 1.0 + pr_rel + n * a;
    let v16 = u * vd + a;
    norm * (ev + v16)
        + 2.0 * vmax * eps_tot.min(1.0)
        + vmax * (pr_rel + n * a)
        + 2.0 * n * 2f64.powi(-24) * vd
}

#[allow(clippy::too_many_arguments)]
fn compare_routes(
    label: &str,
    case: &Case,
    cache: &Cache,
    gathered: bool,
    f32_out: &[f32],
    fp8_out: &[f32],
    fails: &mut Vec<String>,
) {
    let valid = case.valid_rows();
    let row_len = NH * HD;
    let (mut max_diff, mut max_bound, mut worst_ratio) = (0.0f64, 0.0f64, 0.0f64);
    let (mut err2, mut norm2) = (0.0f64, 0.0f64);
    for (r, &row_valid) in valid.iter().enumerate() {
        let rr = r * row_len..(r + 1) * row_len;
        if !row_valid {
            for (name, out) in [("F32", f32_out), ("Fp8", fp8_out)] {
                if let Some(i) = rr.clone().find(|&i| out[i].to_bits() != 0) {
                    fails.push(format!(
                        "{label}: row {r} has no valid token, {name} output not +0 (bits {:#010x} at {})",
                        out[i].to_bits(),
                        i - r * row_len
                    ));
                }
            }
            continue;
        }
        for h in 0..NH {
            let kv = h / (NH / NKV);
            let qh = &case.q[(r * NH + h) * 2 * HD..(r * NH + h + 1) * 2 * HD];
            let qabs: f64 = qh[..HD].iter().map(|&x| (x as f64).abs()).sum();
            let base = analytic_base(cache, kv, qabs, case.sel_len[r], gathered);
            let lo = r * row_len + h * HD;
            let (a, b) = (&f32_out[lo..lo + HD], &fp8_out[lo..lo + HD]);
            let maxabsref = a.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()));
            let allowance = 2.0 * (1e-5 * maxabsref + 1e-7);
            for c in 0..HD {
                let gate = qh[HD + c] as f64;
                let bound = base / (1.0 + (-gate).exp());
                let tol = bound + allowance;
                let (x, y) = (a[c] as f64, b[c] as f64);
                let diff = (x - y).abs();
                if !(a[c].is_finite() && b[c].is_finite() && diff <= tol) {
                    fails.push(format!(
                        "{label}: row {r} head {h} channel {c}: f32 {x:e} fp8 {y:e} |diff| {diff:.3e} > bound {bound:.3e} + allowance {allowance:.3e}"
                    ));
                }
                max_diff = max_diff.max(diff);
                max_bound = max_bound.max(bound);
                worst_ratio = worst_ratio.max(diff / tol);
                err2 += (x - y) * (x - y);
                norm2 += x * x;
            }
        }
    }
    println!(
        "{label}: valid rows {}/{} max|diff|={max_diff:.3e} relL2={:.3e} largest analytic bound={max_bound:.3e} worst diff/(bound+allowance)={worst_ratio:.3e}",
        valid.iter().filter(|&&v| v).count(),
        case.rows,
        (err2 / norm2).sqrt()
    );
}

#[test]
#[ignore = "needs an exact gfx1201 device"]
fn qsa_fp8_attention_within_bound_of_f32() {
    let mut gpu = init_gpu();
    let cache = Cache::build(&mut gpu);
    let gathered_on = qsa_gathered_wmma_enabled(&gpu, QsaKvFormat::Fp8);
    let mut fails: Vec<String> = Vec::new();
    let mut rng = Rng(0x3b0d_0003);
    // Small queries keep the analytic bound far from the trivial `2 Vmax`.
    const QSCALE: f32 = 0.25;

    for &len in &LENS {
        let layout = Layout {
            budget: HUGE,
            compress: 4,
            capacity: len,
        };
        let case = build_case(&mut rng, 1, len, layout, true, QSCALE);
        let (q, sel) = case.upload(&mut gpu);
        let a = run(
            &mut gpu,
            Route::Single,
            QsaKvFormat::F32,
            &cache,
            &q,
            &sel,
            &case,
        );
        let b = run(
            &mut gpu,
            Route::Single,
            QsaKvFormat::Fp8,
            &cache,
            &q,
            &sel,
            &case,
        );
        compare_routes(
            &format!("len {len} single"),
            &case,
            &cache,
            false,
            &a,
            &b,
            &mut fails,
        );
        free(&mut gpu, vec![q, sel]);

        for rows in 1..=8usize {
            let case = build_case(&mut rng, rows, len, layout, false, QSCALE);
            let (q, sel) = case.upload(&mut gpu);
            let a = run(
                &mut gpu,
                Route::Production,
                QsaKvFormat::F32,
                &cache,
                &q,
                &sel,
                &case,
            );
            let b = run(
                &mut gpu,
                Route::Production,
                QsaKvFormat::Fp8,
                &cache,
                &q,
                &sel,
                &case,
            );
            compare_routes(
                &format!("len {len} batch rows {rows}"),
                &case,
                &cache,
                false,
                &a,
                &b,
                &mut fails,
            );
            free(&mut gpu, vec![q, sel]);
        }

        if gathered_on {
            for rows in [64usize, 512] {
                if rows > len {
                    continue;
                }
                // The F32 side is the exact per-head kernel: gfx1201 has no F32
                // gathered route, and the production F32 dispatch would take the
                // selection-blind dense WMMA route at 512 rows.
                let case = build_case(&mut rng, rows, len, GATHER_LAYOUT, false, QSCALE);
                let (q, sel) = case.upload(&mut gpu);
                let a = run(
                    &mut gpu,
                    Route::Exact,
                    QsaKvFormat::F32,
                    &cache,
                    &q,
                    &sel,
                    &case,
                );
                let b = run(
                    &mut gpu,
                    Route::Gathered,
                    QsaKvFormat::Fp8,
                    &cache,
                    &q,
                    &sel,
                    &case,
                );
                compare_routes(
                    &format!("len {len} gathered rows {rows} (f32 exact vs fp8 gathered)"),
                    &case,
                    &cache,
                    true,
                    &a,
                    &b,
                    &mut fails,
                );
                free(&mut gpu, vec![q, sel]);
            }
        }
    }
    if !gathered_on {
        println!("gathered F16 WMMA route disabled for Fp8 here: gathered comparisons skipped");
    }
    cache.free(&mut gpu);
    finish(fails);
}
