// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the lane-batched A3B lm_head (`Gpu::gemv_mq4g256v2_head_lanes`)
//! and batched final norm (`Gpu::rmsnorm_f32_rowsplit_rows` + the batched
//! `mq_rotate_x`) against the REAL singleton launches of the one-row decode and
//! against the PM multi-column twin the exact route used before, at the
//! production geometry (vocab 248320 rows x K = 2048, MQ4G256V2).
//!
//! Test 1 `head_lanes_match_singleton_and_pm_xbatch_hw`: random MQ4G256V2 weight
//! bytes in the real layout (sane f16 headers; the lm_head's 270 MB), columns
//! drawn from random and adversarial activations (zeros, +-large, overflow to
//! inf / NaN, denormals, all-equal ties, sparse, alternating signs, cancelling
//! pairs), N in 1..=8 columns, consecutive and permuted slots, and row counts
//! that exercise every tile / workgroup tail. Per column the new kernel's row
//! must equal the singleton `Gpu::gemv_mq4g256v2` (the launch the one-row
//! decode makes) AND the `gemv_mq4g256v2_multirow_r2_xbatch_pm` column, byte
//! for byte; slot rows that are not columns stay untouched; a negative control
//! (one flipped activation bit) changes exactly that column; 50 back-to-back
//! launches repeat.
//!
//! Test 2 `head_norm_rows_match_singleton_hw`: the batched
//! `rmsnorm_f32_rowsplit_rows` + `rotate_x_mq_batched` equal, per row, the
//! singleton `rmsnorm_f32` + `rotate_x_mq` pair on random / adversarial rows.
//!
//! Test 3 `head_lanes_timing_hw` (`--ignored`, run under `--class timing`):
//! hipEvent medians of OLD (PM x-batch) vs NEW at the production shape.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab --test head_lanes \
//!       -- --ignored --nocapture --test-threads=1
#![cfg(feature = "lab")]

use rdna_compute::head_lanes::HEAD_LANES_ROW_BYTES;
use rdna_compute::{DType, Gpu, GpuTensor};

const K: usize = 2048;
const VOCAB: usize = 248_320;
const POISON: u32 = 0x7FC0_DEAD;
const EPS: f32 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_unit(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        f32::from_bits(0x3F80_0000 | a) - 1.0
    }
    fn f32_signed(&mut self) -> f32 {
        let v = self.f32_unit();
        if self.next_u32() & 1 == 1 {
            -v
        } else {
            v
        }
    }
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

fn f32_bytes(d: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 4) }
}
fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip
        .memcpy_htod(&t.buf, f32_bytes(data))
        .expect("htod f32");
    t
}
fn alloc_f32(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.alloc_tensor(&[n], DType::F32).expect("alloc")
}
fn poison(gpu: &mut Gpu, t: &GpuTensor, n: usize) {
    let v = vec![POISON; n];
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("poison");
}
fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// MQ4G256V2 rows: 8 groups x 136 B `[sc0:f16][zp0:f16][sc1:f16][zp1:f16]
/// [128 nibble bytes]` (header dword 0 serves lanes 0..16, dword 1 lanes 16..32).
fn synth_rows(rows: usize, seed: u64, sc_lo: u32, sc_hi: u32) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; rows * HEAD_LANES_ROW_BYTES];
    for g in 0..rows * 8 {
        let off = g * 136;
        for half in 0..2 {
            let sc = rng.f16_normal(sc_lo, sc_hi, false);
            let zp = rng.f16_normal(sc_lo - 1, sc_hi, true);
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for w in 0..32 {
            let pk = rng.next_u32() ^ (rng.next_u32() << 16);
            out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Pat {
    Random(f32),
    Zeros,
    /// +-large finite values (sums stay finite or overflow to inf).
    Large,
    /// Magnitudes beyond f32 range of the dot: inf / NaN results.
    Overflow,
    Denormal,
    /// All ones: exact ties between lanes, partial sums are small integers.
    Ones,
    /// One nonzero element.
    Sparse,
    Alternating,
    /// x[2i] = v, x[2i+1] = -v: cancelling pairs.
    Cancel,
    /// Mixed magnitudes spanning ~1e-30..1e30.
    Wide,
}

fn make_column(pat: Pat, rng: &mut Lcg) -> Vec<f32> {
    (0..K)
        .map(|i| match pat {
            Pat::Random(mag) => rng.f32_signed() * mag,
            Pat::Zeros => 0.0,
            Pat::Large => rng.f32_signed() * 1.0e30,
            Pat::Overflow => {
                if i % 7 == 0 {
                    f32::MAX
                } else if i % 11 == 0 {
                    -f32::MAX
                } else {
                    rng.f32_signed() * 1.0e37
                }
            }
            Pat::Denormal => {
                f32::from_bits(rng.next_u32() & 0x007F_FFFF)
                    * if rng.next_u32() & 1 == 1 { -1.0 } else { 1.0 }
            }
            Pat::Ones => 1.0,
            Pat::Sparse => {
                if i == 777 {
                    3.5
                } else {
                    0.0
                }
            }
            Pat::Alternating => {
                if i % 2 == 0 {
                    0.75
                } else {
                    -0.75
                }
            }
            Pat::Cancel => {
                let v = (i / 2) as f32 * 0.37 + 0.11;
                if i % 2 == 0 {
                    v
                } else {
                    -v
                }
            }
            Pat::Wide => {
                let e = (rng.next_u32() % 60) as i32 - 30;
                rng.f32_signed() * 10f32.powi(e)
            }
        })
        .collect()
}

struct Rig {
    gpu: Gpu,
    w: GpuTensor,
}

fn rig() -> Rig {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(
        gpu.arch_caps.is_gfx1201(),
        "this gate requires an exact gfx1201 device (got {})",
        gpu.arch
    );
    assert!(
        gpu.flags.pm_decode,
        "kernel.pm_decode is off: the PM x-batch reference would run the hipcc fallback"
    );
    let bytes = synth_rows(VOCAB, 0x4EAD_0001, 7, 11);
    let w = upload_u8(&mut gpu, &bytes);
    Rig { gpu, w }
}

/// First differing element: (row, ref bits, got bits).
fn first_diff(a: &[u32], b: &[u32]) -> Option<(usize, u32, u32)> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .map(|i| (i, a[i], b[i]))
}

/// Rows compared; returns the number of mismatching elements.
#[allow(clippy::too_many_arguments)]
fn check_case(
    rig: &mut Rig,
    tag: &str,
    m: usize,
    cols: &[Vec<f32>],
    slots: &[usize],
    stats: &mut (usize, usize),
) {
    let n = cols.len();
    let gpu = &mut rig.gpu;
    let flat: Vec<f32> = cols.iter().flatten().copied().collect();
    let x_all = upload_f32(gpu, &flat);

    // singleton reference, one launch per column on its own row buffer
    let mut refs: Vec<Vec<u32>> = Vec::new();
    for c in 0..n {
        let xc = x_all.sub_offset(c * K, K);
        let yc = alloc_f32(gpu, m);
        poison(gpu, &yc, m);
        gpu.gemv_mq4g256v2(&rig.w, &xc, &yc, m, K)
            .expect("singleton gemv_mq4g256v2");
        refs.push(dl(gpu, &yc, m));
    }
    // PM x-batch twin (what the exact route used before)
    let y_pm = alloc_f32(gpu, n * m);
    poison(gpu, &y_pm, n * m);
    gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(&rig.w, &x_all, &y_pm, m, K, n)
        .expect("pm xbatch");
    let pm = dl(gpu, &y_pm, n * m);

    // new kernel into slot rows of a bigger buffer
    let max_slot = *slots.iter().max().unwrap();
    let y_new = alloc_f32(gpu, (max_slot + 1) * m);
    poison(gpu, &y_new, (max_slot + 1) * m);
    gpu.gemv_mq4g256v2_head_lanes(&rig.w, &x_all, &y_new, m, slots)
        .expect("head lanes");
    let got = dl(gpu, &y_new, (max_slot + 1) * m);

    let mut bad = 0usize;
    for c in 0..n {
        let row = &got[slots[c] * m..(slots[c] + 1) * m];
        if let Some((i, r, g)) = first_diff(&refs[c], row) {
            bad += 1;
            eprintln!(
                "FAIL {tag} col{c} slot{} vs SINGLETON: first diff row {i}: singleton {r:#010x} new {g:#010x}",
                slots[c]
            );
        }
        if let Some((i, r, g)) = first_diff(&pm[c * m..(c + 1) * m], row) {
            bad += 1;
            eprintln!(
                "FAIL {tag} col{c} vs PM xbatch: first diff row {i}: pm {r:#010x} new {g:#010x}"
            );
        }
        stats.0 += m;
    }
    for s in 0..=max_slot {
        if !slots.contains(&s) {
            let row = &got[s * m..(s + 1) * m];
            if row.iter().any(|&b| b != POISON) {
                bad += 1;
                eprintln!("FAIL {tag}: unused slot row {s} was written");
            }
        }
    }
    stats.1 += bad;
    assert_eq!(bad, 0, "{tag}: mismatches");
}

#[test]
#[ignore = "needs an exact gfx1201 device"]
fn head_lanes_match_singleton_and_pm_xbatch_hw() {
    let mut rig = rig();
    let mut stats = (0usize, 0usize);
    let pats = [
        Pat::Random(1.0),
        Pat::Random(40.0),
        Pat::Zeros,
        Pat::Large,
        Pat::Overflow,
        Pat::Denormal,
        Pat::Ones,
        Pat::Sparse,
        Pat::Alternating,
        Pat::Cancel,
        Pat::Wide,
    ];
    let permuted: [usize; 8] = [5, 1, 7, 2, 0, 6, 3, 4];
    // every column count, production vocab, consecutive and permuted slots
    for n in 1..=8usize {
        for round in 0..4u64 {
            let mut rng = Lcg(0xFACE_0000 ^ ((n as u64) << 12) ^ round);
            let cols: Vec<Vec<f32>> = (0..n)
                .map(|c| {
                    let p = pats[(c + round as usize * 3 + n) % pats.len()];
                    make_column(p, &mut rng)
                })
                .collect();
            let consecutive: Vec<usize> = (0..n).collect();
            check_case(
                &mut rig,
                &format!("N{n} r{round} consecutive"),
                VOCAB,
                &cols,
                &consecutive,
                &mut stats,
            );
            let perm: Vec<usize> = permuted[..n].to_vec();
            check_case(
                &mut rig,
                &format!("N{n} r{round} permuted"),
                VOCAB,
                &cols,
                &perm,
                &mut stats,
            );
        }
    }
    // each adversarial pattern in every column position of an 8-lane launch
    for &p in &pats {
        let mut rng = Lcg(0xBEEF_0001);
        let cols: Vec<Vec<f32>> = (0..8)
            .map(|c| {
                if c % 2 == 0 {
                    make_column(p, &mut rng)
                } else {
                    make_column(Pat::Random(2.0), &mut rng)
                }
            })
            .collect();
        let slots: Vec<usize> = (0..8).collect();
        check_case(
            &mut rig,
            &format!("N8 all-{p:?}"),
            VOCAB,
            &cols,
            &slots,
            &mut stats,
        );
    }
    // tails: row counts around the tile (4 rows) / workgroup (128 rows) edges
    for &m in &[
        1usize, 2, 3, 4, 5, 7, 8, 31, 32, 33, 127, 128, 129, 130, 255, 256, 257, 511, 512, 1000,
        4097, 65_537, 248_319, 248_320,
    ] {
        for &n in &[1usize, 3, 8] {
            let mut rng = Lcg(0x7A11_0000 ^ ((m as u64) << 8) ^ n as u64);
            let cols: Vec<Vec<f32>> = (0..n)
                .map(|c| make_column(pats[(c * 5 + m) % pats.len()], &mut rng))
                .collect();
            let slots: Vec<usize> = permuted[..n].to_vec();
            check_case(
                &mut rig,
                &format!("tail m{m} N{n}"),
                m,
                &cols,
                &slots,
                &mut stats,
            );
        }
    }

    // negative control: flip one input bit of one column -> exactly that
    // column's row changes (and only it).
    {
        let n = 5usize;
        let mut rng = Lcg(0xC0FF_EE00);
        let mut cols: Vec<Vec<f32>> = (0..n)
            .map(|_| make_column(Pat::Random(1.0), &mut rng))
            .collect();
        let slots = vec![3usize, 0, 6, 1, 4];
        let gpu = &mut rig.gpu;
        let max_slot = 6usize;
        let run = |gpu: &mut Gpu, w: &GpuTensor, cols: &[Vec<f32>]| -> Vec<u32> {
            let flat: Vec<f32> = cols.iter().flatten().copied().collect();
            let x = upload_f32(gpu, &flat);
            let y = alloc_f32(gpu, (max_slot + 1) * VOCAB);
            poison(gpu, &y, (max_slot + 1) * VOCAB);
            gpu.gemv_mq4g256v2_head_lanes(w, &x, &y, VOCAB, &slots)
                .expect("neg");
            dl(gpu, &y, (max_slot + 1) * VOCAB)
        };
        let base = run(gpu, &rig.w, &cols);
        for victim in 0..n {
            let saved = cols[victim][1234];
            cols[victim][1234] = f32::from_bits(saved.to_bits() ^ (1 << 20));
            let flipped = run(gpu, &rig.w, &cols);
            cols[victim][1234] = saved;
            for c in 0..n {
                let a = &base[slots[c] * VOCAB..(slots[c] + 1) * VOCAB];
                let b = &flipped[slots[c] * VOCAB..(slots[c] + 1) * VOCAB];
                let changed = a != b;
                assert_eq!(
                    changed,
                    c == victim,
                    "negative control: victim {victim}, column {c} changed={changed}"
                );
            }
        }
    }

    // repeatability: 50 back-to-back launches, same bytes each time
    {
        let n = 8usize;
        let mut rng = Lcg(0x5EED_5EED);
        let cols: Vec<Vec<f32>> = (0..n)
            .map(|_| make_column(Pat::Random(3.0), &mut rng))
            .collect();
        let flat: Vec<f32> = cols.iter().flatten().copied().collect();
        let gpu = &mut rig.gpu;
        let x = upload_f32(gpu, &flat);
        let y = alloc_f32(gpu, n * VOCAB);
        let slots: Vec<usize> = (0..n).collect();
        gpu.gemv_mq4g256v2_head_lanes(&rig.w, &x, &y, VOCAB, &slots)
            .unwrap();
        let first = dl(gpu, &y, n * VOCAB);
        for _ in 0..50 {
            poison(gpu, &y, n * VOCAB);
            gpu.gemv_mq4g256v2_head_lanes(&rig.w, &x, &y, VOCAB, &slots)
                .unwrap();
        }
        assert_eq!(first, dl(gpu, &y, n * VOCAB), "repeat launches differ");
    }
    println!(
        "head_lanes identity: compared {} logits, {} mismatching cases",
        stats.0, stats.1
    );
    assert_eq!(stats.1, 0);
}

#[test]
#[ignore = "needs an exact gfx1201 device"]
fn head_norm_rows_match_singleton_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(gpu.arch_caps.is_gfx1201(), "exact gfx1201 required");
    assert!(
        gpu.rmsnorm_f32_rowsplit_rows_admitted(K, 8),
        "decode norm grids must be on: the singleton picks rmsnorm_f32_rowsplit"
    );
    gpu.ensure_mq_signs().expect("signs");
    let mut rng = Lcg(0x1234_5678);
    let weight: Vec<f32> = (0..K).map(|_| 0.5 + rng.f32_unit() * 2.0).collect();
    let wt = upload_f32(&mut gpu, &weight);
    let mut bad = 0usize;
    let mut compared = 0usize;
    for &n in &[1usize, 2, 3, 5, 8] {
        for (round, pat) in [
            Pat::Random(1.0),
            Pat::Random(1000.0),
            Pat::Zeros,
            Pat::Large,
            Pat::Denormal,
            Pat::Ones,
            Pat::Sparse,
            Pat::Wide,
        ]
        .into_iter()
        .enumerate()
        {
            let mut r2 = Lcg(0xAA00 ^ ((n as u64) << 8) ^ round as u64);
            let rows: Vec<Vec<f32>> = (0..n)
                .map(|c| {
                    if c % 3 == 2 {
                        make_column(Pat::Random(5.0), &mut r2)
                    } else {
                        make_column(pat, &mut r2)
                    }
                })
                .collect();
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            let x = upload_f32(&mut gpu, &flat);
            // singleton: one rmsnorm_f32 + one rotate_x_mq per row
            let mut want: Vec<Vec<u32>> = Vec::new();
            for r in 0..n {
                let xr = x.sub_offset(r * K, K);
                let tmp = alloc_f32(&mut gpu, K);
                let rot = alloc_f32(&mut gpu, K);
                poison(&mut gpu, &tmp, K);
                poison(&mut gpu, &rot, K);
                gpu.rmsnorm_f32(&xr, &wt, &tmp, EPS).expect("rmsnorm");
                gpu.rotate_x_mq(&tmp, &rot, K).expect("rotate");
                want.push(dl(&gpu, &rot, K));
            }
            // batched: one rowsplit launch (grid.y = rows) + one batched rotate
            let normed = alloc_f32(&mut gpu, n * K);
            let rot = alloc_f32(&mut gpu, n * K);
            poison(&mut gpu, &normed, n * K);
            poison(&mut gpu, &rot, n * K);
            gpu.rmsnorm_f32_rowsplit_rows(&x, &wt, &normed, K, n, EPS)
                .expect("rows norm");
            gpu.rotate_x_mq_batched(&normed, &rot, K, n)
                .expect("batched rotate");
            let got = dl(&gpu, &rot, n * K);
            for r in 0..n {
                compared += K;
                if let Some((i, w, g)) = first_diff(&want[r], &got[r * K..(r + 1) * K]) {
                    bad += 1;
                    eprintln!(
                        "FAIL n{n} {pat:?} row{r} elem{i}: singleton {w:#010x} batched {g:#010x}"
                    );
                }
            }
        }
    }
    println!("head norm rows: compared {compared} floats, {bad} mismatching rows");
    assert_eq!(bad, 0);
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "timing: run under --class timing on card A"]
fn head_lanes_timing_hw() {
    let mut rig = rig();
    let iters: usize = std::env::var("HEAD_LANES_TIMING_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let gpu = &mut rig.gpu;
    let weight_bytes = (VOCAB * HEAD_LANES_ROW_BYTES) as f64;
    println!(
        "head timing: vocab {VOCAB} K {K}, weight {:.1} MB, {iters} timed launches each (hipEvent per launch, after 30 warmup)",
        weight_bytes / 1e6
    );
    for n in [8usize, 4, 2, 1] {
        let mut rng = Lcg(0x71D1_0000 + n as u64);
        let flat: Vec<f32> = (0..n * K).map(|_| rng.f32_signed()).collect();
        let x = upload_f32(gpu, &flat);
        let y_old = alloc_f32(gpu, n * VOCAB);
        let y_new = alloc_f32(gpu, n * VOCAB);
        let slots: Vec<usize> = (0..n).collect();
        for (name, which) in [("OLD pm_xbatch multirow_r2", 0), ("NEW head_lanes", 1)] {
            let run = |gpu: &mut Gpu| {
                if which == 0 {
                    gpu.gemv_mq4g256v2_multirow_r2_xbatch_pm(&rig.w, &x, &y_old, VOCAB, K, n)
                        .unwrap();
                } else {
                    gpu.gemv_mq4g256v2_head_lanes(&rig.w, &x, &y_new, VOCAB, &slots)
                        .unwrap();
                }
            };
            for _ in 0..30 {
                run(gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            let mut pairs = Vec::new();
            for _ in 0..iters {
                let s = gpu.hip.event_create().unwrap();
                let e = gpu.hip.event_create().unwrap();
                gpu.hip.event_record(&s, None).unwrap();
                run(gpu);
                gpu.hip.event_record(&e, None).unwrap();
                pairs.push((s, e));
            }
            gpu.hip.event_synchronize(&pairs.last().unwrap().1).unwrap();
            let mut us: Vec<f64> = pairs
                .iter()
                .map(|(s, e)| gpu.hip.event_elapsed_ms(s, e).unwrap() as f64 * 1000.0)
                .collect();
            let min = us.iter().cloned().fold(f64::MAX, f64::min);
            let med = median(&mut us);
            println!(
                "  N={n} {name:<28} median {med:8.1} us  min {min:8.1} us  weight {:.0} GB/s (median)",
                weight_bytes / (med * 1e-6) / 1e9
            );
            for (s, e) in pairs {
                let _ = gpu.hip.event_destroy(s);
                let _ = gpu.hip.event_destroy(e);
            }
        }
    }
}

/// Signed zeros: every weight negative (header zero point -1, scale 0.01), so
/// products with +0 / -0 activations are -0 / +0 and a whole chain can be -0.
/// The singleton's accumulators start at +0.0f, so an all-zero dot is +0.0; the
/// new kernel must reproduce the sign of zero exactly.
#[test]
#[ignore = "needs an exact gfx1201 device"]
fn head_lanes_signed_zero_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(gpu.arch_caps.is_gfx1201(), "exact gfx1201 required");
    let m = 1024usize;
    let mut bytes = synth_rows(m, 0x2E80, 7, 11);
    for g in 0..m * 8 {
        let off = g * 136;
        for half in 0..2 {
            bytes[off + 4 * half..off + 4 * half + 2].copy_from_slice(&0x2122u16.to_le_bytes()); // 0.01
            bytes[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&0xBC00u16.to_le_bytes());
            // -1.0
        }
    }
    let w = upload_u8(&mut gpu, &bytes);
    let cols: Vec<Vec<f32>> = vec![
        vec![0.0; K],
        vec![-0.0; K],
        (0..K)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect(),
        (0..K).map(|i| if i == 5 { 1.0 } else { 0.0 }).collect(),
        (0..K)
            .map(|i| if i % 256 < 128 { 0.0 } else { -0.0 })
            .collect(),
    ];
    let n = cols.len();
    let flat: Vec<f32> = cols.iter().flatten().copied().collect();
    let x = upload_f32(&mut gpu, &flat);
    let y = alloc_f32(&mut gpu, n * m);
    poison(&mut gpu, &y, n * m);
    let slots: Vec<usize> = (0..n).collect();
    gpu.gemv_mq4g256v2_head_lanes(&w, &x, &y, m, &slots)
        .expect("head lanes");
    let got = dl(&gpu, &y, n * m);
    for c in 0..n {
        let yc = alloc_f32(&mut gpu, m);
        gpu.gemv_mq4g256v2(&w, &x.sub_offset(c * K, K), &yc, m, K)
            .expect("singleton");
        let want = dl(&gpu, &yc, m);
        assert_eq!(
            first_diff(&want, &got[c * m..(c + 1) * m]),
            None,
            "signed-zero column {c}"
        );
    }
}
