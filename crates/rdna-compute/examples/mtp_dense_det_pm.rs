// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP append fc_hidden DET dense F16 WMMA first-stage KILL harness (gfx1151 only;
//! developer example, needs a GPU: build only on other hosts).
//!
//! Question: is the standalone native module `mtp_dense_det_pm_gfx1151`
//! (`Gpu::gemm_bf16_xf32_mtp_dense_det_pm`, entry `mtp_dense_det_pm_gfx1151_w8`; BF16 weight
//! `[M, K]` through its model-lifetime F16 shadow, F32 X `[N, K]` converted to F16 on every
//! call, F32 Y `[N, M]`, one fixed 128 x 256 tile, one ascending K16 WMMA chain per output,
//! no split-K) at least 5x the incumbent conversion-free BF16 x F32 multirow GEMM at the
//! fc_hidden shape (default M = K = 2560, N = 4096), conversion INCLUDED?  It is explicitly DET,
//! not byte-equal to the incumbent; the F16 narrower exponent range may under/overflow.
//!
//! Arms (all over the same deterministic finite BF16 W and F32 X, guarded 4 KiB poison either
//! side of every output):
//!   * baseline: `gemm_bf16_xf32_multirow_r16w4_forced` (the incumbent
//!     `gemm_bf16_xf32_multirow_r16w4_gfx1151` tile, forced);
//!   * oracle: the existing dense F16 WMMA GEMM (`gemm_f16_x_f16_wmma_lds_tiled`, fixed plain
//!     128 x 256 / k64 tile, fixed K, no split-K) over the SAME converted operands (the weight
//!     shadow and an own copy of the X conversion);
//!   * native conversion-inclusive (`gemm_bf16_xf32_mtp_dense_det_pm`) and native launch-only
//!     (`mtp_dense_det_pm_launch` over the own X16 copy).
//!
//! Receipts: the device is gfx1151 at PCI 0000:c0:00.0; the first (cold) weight shadow
//! overhead is reported apart (the conversion kernels are preloaded by a tiny shadow first);
//! shadow and X16 bytes are checked against a host BF16 -> F32 -> F16 RNE; native outputs equal
//! the oracle over the WHOLE buffers (guards included) after the first run and again after the
//! timed loops; error against the incumbent (max abs, max rel against a floor of 1e-3 of the
//! incumbent RMS, RMS) is REPORTED, equality is not required; row-local invariance: partitions
//! N in {1, 15, 16, 17, 255, 256, 257} at three offsets, reordered, duplicated, scattered and
//! zero-padded batches, each row byte-equal to the same row of the full-batch native output.
//! Timing: 3 warm + `--reps` interleaved event-timed rounds; the gate is the median of the paired
//! baseline / native-conversion-inclusive ratios AND the ratio of medians both >= 5.0.
//!
//! Output: JSON lines on stdout (`start`, `receipt`, `kill_verdict`) and a final `KILL PASS|FAIL`
//! line. Exit 0 only on a kill PASS (target shape, every correctness check, speed gate); 1 FAIL,
//! 2 usage, 3 not the admitted gfx1151 device / HIP init failure.
//!
//! Usage: `mtp_dense_det_pm [--m 2560] [--k 2560] [--n 4096] [--reps 20] [--seed 184]`

use rdna_compute::gemm::{mtp_dense_det_pm_shape, LdsTile};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const GUARD: usize = 4096;
const POISON: u8 = 0xA5;
const WARM: usize = 3;
const REQUIRED_PCI: &str = "0000:c0:00.0";
const REQUIRED_RATIO: f64 = 5.0;
/// The oracle's fixed plain (non-pipelined) tile: 128 channels x 256 rows, K64 stage, no split-K.
const ORACLE_TILE: LdsTile = LdsTile::new(128, 256, 32, 64, 64, false);
const PARTITIONS: [usize; 7] = [1, 15, 16, 17, 255, 256, 257];
/// Rows of the largest invariance fixture (scattered).
const INV_MAX_ROWS: usize = 515;

// ----------------------------------------------------------------- generators

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// BF16 bits of a finite F32, round to nearest even.
fn bf16_rne(f: f32) -> u16 {
    let b = f.to_bits();
    (b.wrapping_add(0x7fff).wrapping_add((b >> 16) & 1) >> 16) as u16
}

/// Finite BF16 weight: small magnitudes (row-dependent scale), ~1/64 each +0 / -0.
fn gen_w(row: u64, col: u64, k: u64, seed: u64) -> u16 {
    let h = mix64((row * k + col).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ seed);
    match h & 63 {
        0 => 0x0000,
        1 => 0x8000,
        _ => {
            let u = ((h >> 16) & 0xff_ffff) as f32 / 16_777_216.0;
            let scale = 0.08 / (1u32 << (row % 5)) as f32;
            bf16_rne((u - 0.5) * scale)
        }
    }
}

/// Finite F32 activation with a full 24-bit mantissa (the F16 conversion rounds), ~1/64 each +-0.
fn gen_x(idx: u64, seed: u64) -> f32 {
    let h = mix64(idx.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed);
    match h & 63 {
        0 => 0.0,
        1 => -0.0,
        _ => (((h >> 16) & 0xff_ffff) as f32 / 16_777_216.0 - 0.5) * 4.0,
    }
}

/// F32 -> F16 bits, round to nearest even (overflow to Inf, subnormals, NaN payload truncation).
fn f32_to_f16_rne(f: f32) -> u16 {
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let ne = e - 127 + 15;
    if ne >= 31 {
        return s | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - ne) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) {
            half + 1
        } else {
            half
        };
        return s | r as u16;
    }
    let half = ((ne as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
        half + 1
    } else {
        half
    };
    s | r as u16
}

// -------------------------------------------------------------------- device

/// F32 storage with `GUARD` bytes of poison either side.
struct Guarded {
    store: GpuTensor,
    payload_elems: usize,
}
impl Guarded {
    fn new(gpu: &mut Gpu, elems: usize) -> Res<Self> {
        let total = elems + 2 * GUARD / 4;
        let store = gpu.alloc_tensor(&[total], DType::F32)?;
        let g = Self {
            store,
            payload_elems: elems,
        };
        g.poison(gpu)?;
        Ok(g)
    }
    fn poison(&self, gpu: &Gpu) -> Res<()> {
        gpu.hip
            .memset(&self.store.buf, i32::from(POISON), self.store.byte_size())?;
        Ok(())
    }
    fn view(&self) -> GpuTensor {
        self.store.sub_offset(GUARD / 4, self.payload_elems)
    }
    fn fetch(&self, gpu: &Gpu) -> Res<Vec<u8>> {
        Ok(gpu.download_raw_bytes(&self.store)?)
    }
}

fn upload_dtype(gpu: &Gpu, bytes: &[u8], dtype: DType, elems: usize) -> Res<GpuTensor> {
    let mut t = gpu.upload_raw(bytes, &[elems])?;
    t.dtype = dtype;
    t.shape = vec![elems];
    Ok(t)
}

fn payload(buf: &[u8], payload_bytes: usize) -> &[u8] {
    &buf[GUARD..GUARD + payload_bytes]
}

/// Both guards (and everything past the `written` payload bytes, up to the end) still poison.
fn region_ok(buf: &[u8], written: usize) -> bool {
    buf[..GUARD].iter().all(|&b| b == POISON) && buf[GUARD + written..].iter().all(|&b| b == POISON)
}

/// F32 words of the payload that are still the poison pattern.
fn poison_left(p: &[u8]) -> usize {
    p.chunks_exact(4)
        .filter(|c| c.iter().all(|&b| b == POISON))
        .count()
}

fn first_diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (
        n + a.len().abs_diff(b.len()),
        first.or((a.len() != b.len()).then_some(a.len().min(b.len()))),
    )
}

fn f32s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

struct ErrStats {
    max_abs: f64,
    max_abs_at: usize,
    max_rel: f64,
    rms: f64,
    rms_ref: f64,
    nonfinite_cand: usize,
    nonfinite_ref: usize,
}

/// Candidate against reference (both F32 payloads). Relative error uses the denominator
/// `max(|ref|, 1e-3 * rms_ref)`.
fn err_stats(cand: &[u8], reference: &[u8]) -> ErrStats {
    let count = cand.len() / 4;
    let ss: f64 = f32s(reference)
        .filter(|v| v.is_finite())
        .map(|v| f64::from(v) * f64::from(v))
        .sum();
    let rms_ref = (ss / count.max(1) as f64).sqrt();
    let floor = 1e-3 * rms_ref.max(f64::MIN_POSITIVE);
    let (mut max_abs, mut max_abs_at, mut max_rel, mut sq) = (0f64, 0usize, 0f64, 0f64);
    let (mut nf_c, mut nf_r) = (0usize, 0usize);
    for (i, (c, r)) in f32s(cand).zip(f32s(reference)).enumerate() {
        if !c.is_finite() {
            nf_c += 1;
        }
        if !r.is_finite() {
            nf_r += 1;
        }
        if !c.is_finite() || !r.is_finite() {
            continue;
        }
        let d = (f64::from(c) - f64::from(r)).abs();
        sq += d * d;
        if d > max_abs {
            max_abs = d;
            max_abs_at = i;
        }
        max_rel = max_rel.max(d / f64::from(r).abs().max(floor));
    }
    ErrStats {
        max_abs,
        max_abs_at,
        max_rel,
        rms: (sq / count.max(1) as f64).sqrt(),
        rms_ref,
        nonfinite_cand: nf_c,
        nonfinite_ref: nf_r,
    }
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn usage(msg: &str) -> ! {
    eprintln!(
        "{msg}\nusage: mtp_dense_det_pm [--m 2560] [--k 2560] [--n 4096] [--reps 20] [--seed 184]"
    );
    std::process::exit(2);
}

fn parse<T: std::str::FromStr>(args: &[String], k: &str, d: &str) -> T {
    match arg(args, k, d).parse() {
        Ok(v) => v,
        Err(_) => usage(&format!("{k}: not a number")),
    }
}

/// Expected rows of one invariance fixture: `Some(r)` = row `r` of the full-batch native output,
/// `None` = a zero-padding row (exactly +0.0).
struct Fixture {
    name: &'static str,
    rows: Vec<Option<usize>>,
}

fn fixtures(n: usize) -> Vec<Fixture> {
    let reorder: Vec<Option<usize>> = (0..n.min(300)).rev().map(Some).collect();
    let mut dup: Vec<Option<usize>> = Vec::new();
    dup.extend(std::iter::repeat(Some(n / 3)).take(129));
    dup.extend(std::iter::repeat(Some(n - 1)).take(100));
    dup.extend(std::iter::repeat(Some(0)).take(28));
    let scattered: Vec<Option<usize>> = (0..INV_MAX_ROWS)
        .map(|i| Some((i * 131 + 17) % n))
        .collect();
    let padded: Vec<Option<usize>> = (0..300)
        .map(|j| {
            if j % 2 == 0 {
                Some((j * 37 + 5) % n)
            } else {
                None
            }
        })
        .collect();
    vec![
        Fixture {
            name: "reordered",
            rows: reorder,
        },
        Fixture {
            name: "duplicated",
            rows: dup,
        },
        Fixture {
            name: "scattered",
            rows: scattered,
        },
        Fixture {
            name: "zero_padded",
            rows: padded,
        },
    ]
}

fn run(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    reps: usize,
    seed: u64,
    fails: &mut Vec<String>,
) -> Res<Value> {
    // ---- operands
    let mut w_bytes = Vec::with_capacity(m * k * 2);
    for r in 0..m as u64 {
        for c in 0..k as u64 {
            w_bytes.extend(gen_w(r, c, k as u64, seed ^ 0x57A11).to_le_bytes());
        }
    }
    let x: Vec<f32> = (0..(n * k) as u64)
        .map(|e| gen_x(e, seed ^ 0xC0FFEE))
        .collect();
    let w = upload_dtype(gpu, &w_bytes, DType::BF16, m * k)?;
    let x32 = gpu.upload_f32(&x, &[n * k])?;

    // ---- cold shadow overhead, kernels preloaded by tiny operands (kept alive: the shadow cache
    // is keyed by weight pointer, so no operand may be freed and recycled)
    let w_tiny = upload_dtype(gpu, &w_bytes[..16 * 64 * 2], DType::BF16, 16 * 64)?;
    let _tiny_shadow = gpu.mtp_dense_det_pm_weight_shadow(&w_tiny, 16, 64)?;
    let _tiny_x16 = gpu.qwen4_convert_f16_x(&x32.sub_offset(0, 64), 64)?;
    gpu.hip.device_synchronize()?;
    let t0 = Instant::now();
    let shadow = gpu.mtp_dense_det_pm_weight_shadow(&w, m, k)?;
    gpu.hip.device_synchronize()?;
    let first_shadow_ms = t0.elapsed().as_secs_f64() * 1e3;

    // ---- conversion receipts against the host RNE; X16 own copy for the oracle / launch-only arms
    let shadow_bytes = gpu.download_raw_bytes(&shadow)?;
    let shadow_expect: Vec<u8> = w_bytes
        .chunks_exact(2)
        .flat_map(|c| {
            f32_to_f16_rne(f32::from_bits(
                u32::from(u16::from_le_bytes([c[0], c[1]])) << 16,
            ))
            .to_le_bytes()
        })
        .collect();
    let (shadow_diff, _) = first_diff(&shadow_bytes, &shadow_expect);
    let x16_view = gpu.qwen4_convert_f16_x(&x32, n * k)?;
    let x16_bytes = gpu.download_raw_bytes(&x16_view)?;
    let x16_expect: Vec<u8> = x
        .iter()
        .flat_map(|v| f32_to_f16_rne(*v).to_le_bytes())
        .collect();
    let (x16_diff, _) = first_diff(&x16_bytes, &x16_expect);
    let x16 = upload_dtype(gpu, &x16_bytes, DType::F16, n * k)?;
    drop((shadow_expect, x16_expect, x16_bytes));
    let conv_ok = shadow_diff == 0 && x16_diff == 0;
    if !conv_ok {
        fails.push(format!(
            "conversion differs from host RNE (shadow {shadow_diff} bytes, x16 {x16_diff} bytes)"
        ));
    }

    // ---- buffers + arms
    let y_base = Guarded::new(gpu, n * m)?;
    let y_orc = Guarded::new(gpu, n * m)?;
    let y_nat = Guarded::new(gpu, n * m)?;
    let y_lo = Guarded::new(gpu, n * m)?;
    let (yb, yo, yn, yl) = (y_base.view(), y_orc.view(), y_nat.view(), y_lo.view());
    let pay = n * m * 4;

    let base = |gpu: &mut Gpu| -> Res<()> {
        Ok(gpu.gemm_bf16_xf32_multirow_r16w4_forced(&w, &x32, &yb, m, k, n)?)
    };
    let oracle = |gpu: &mut Gpu| -> Res<()> {
        Ok(gpu.gemm_f16_x_f16_wmma_lds_tiled(&shadow, &x16, &yo, None, m, k, n, ORACLE_TILE)?)
    };
    let nat = |gpu: &mut Gpu| -> Res<()> {
        Ok(gpu.gemm_bf16_xf32_mtp_dense_det_pm(&w, &x32, &yn, m, k, n)?)
    };
    let launch =
        |gpu: &mut Gpu| -> Res<()> { Ok(gpu.mtp_dense_det_pm_launch(&w, &x16, &yl, m, k, n)?) };

    // ---- first run of every arm
    base(gpu)?;
    oracle(gpu)?;
    nat(gpu)?;
    launch(gpu)?;
    gpu.hip.device_synchronize()?;
    let (b1, o1, n1, l1) = (
        y_base.fetch(gpu)?,
        y_orc.fetch(gpu)?,
        y_nat.fetch(gpu)?,
        y_lo.fetch(gpu)?,
    );
    let first = check_outputs("first", &b1, &o1, &n1, &l1, pay, fails);
    let err_nat = err_stats(payload(&n1, pay), payload(&b1, pay));
    let err_orc = err_stats(payload(&o1, pay), payload(&b1, pay));
    if err_nat.nonfinite_cand != 0 || err_nat.nonfinite_ref != 0 {
        fails.push(format!(
            "non-finite outputs (native {}, incumbent {})",
            err_nat.nonfinite_cand, err_nat.nonfinite_ref
        ));
    }

    // ---- timing: WARM warm rounds, then `reps` interleaved rounds of baseline / native incl. / native launch-only
    let ev = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let time = |gpu: &mut Gpu, which: usize| -> Res<f64> {
        gpu.hip.event_record(&ev.0, None)?;
        match which {
            0 => base(gpu)?,
            1 => nat(gpu)?,
            _ => launch(gpu)?,
        }
        gpu.hip.event_record(&ev.1, None)?;
        gpu.hip.event_synchronize(&ev.1)?;
        Ok(f64::from(gpu.hip.event_elapsed_ms(&ev.0, &ev.1)?))
    };
    for _ in 0..WARM {
        for which in 0..3 {
            time(gpu, which)?;
        }
    }
    let mut ms: [Vec<f64>; 3] = Default::default();
    for rep in 0..reps {
        let order = if rep % 2 == 0 { [0, 1, 2] } else { [1, 0, 2] };
        for which in order {
            ms[which].push(time(gpu, which)?);
        }
    }
    let med: Vec<f64> = ms.iter().map(|v| median(v.clone()) * 1000.0).collect();
    let ratios: Vec<f64> = ms[0].iter().zip(&ms[1]).map(|(b, c)| b / c).collect();
    let pair_ratio = median(ratios.clone());
    let med_ratio = med[0] / med[1];
    let ratio_launch = med[0] / med[2];
    let speed_pass = reps > 0 && pair_ratio >= REQUIRED_RATIO && med_ratio >= REQUIRED_RATIO;
    if !speed_pass {
        fails.push(format!("speed gate: median paired ratio {pair_ratio:.3}, ratio of medians {med_ratio:.3}, required >= {REQUIRED_RATIO}"));
    }

    // ---- post-timing: the last timed outputs, then a fresh poisoned re-run of the oracle / native arms
    gpu.hip.device_synchronize()?;
    let (b2, n2, l2) = (y_base.fetch(gpu)?, y_nat.fetch(gpu)?, y_lo.fetch(gpu)?);
    let timed_same = payload(&b2, pay) == payload(&b1, pay)
        && payload(&n2, pay) == payload(&n1, pay)
        && payload(&l2, pay) == payload(&l1, pay)
        && region_ok(&b2, pay)
        && region_ok(&n2, pay)
        && region_ok(&l2, pay);
    if !timed_same {
        fails.push(
            "outputs after the timed loops differ from the first run (or a guard was written)"
                .into(),
        );
    }
    y_orc.poison(gpu)?;
    y_nat.poison(gpu)?;
    y_lo.poison(gpu)?;
    oracle(gpu)?;
    nat(gpu)?;
    launch(gpu)?;
    gpu.hip.device_synchronize()?;
    let (o3, n3, l3) = (y_orc.fetch(gpu)?, y_nat.fetch(gpu)?, y_lo.fetch(gpu)?);
    let post = check_outputs("post", &b2, &o3, &n3, &l3, pay, fails);
    let post_same_as_first =
        payload(&o3, pay) == payload(&o1, pay) && payload(&n3, pay) == payload(&n1, pay);
    if !post_same_as_first {
        fails.push("fresh post-timing re-run differs from the first run".into());
    }

    // ---- row-local invariance (production weights, partitions / permutations / padding of X)
    let part = Guarded::new(gpu, INV_MAX_ROWS * m)?;
    let row_bytes = m * 4;
    let ref_nat = payload(&n1, pay);
    let mut inv_ok = true;
    let mut partitions = Vec::new();
    let mut inv_complete = true;
    for &s in &PARTITIONS {
        if s > n {
            inv_complete = false;
            partitions.push(json!({"rows": s, "skipped": "larger than N"}));
            continue;
        }
        let mut starts = vec![0usize, 129usize.min(n - s), n - s];
        starts.dedup();
        for start in starts {
            part.poison(gpu)?;
            let xv = x32.sub_offset(start * k, s * k);
            gpu.gemm_bf16_xf32_mtp_dense_det_pm(&w, &xv, &part.view(), m, k, s)?;
            gpu.hip.device_synchronize()?;
            let got = part.fetch(gpu)?;
            let written = s * row_bytes;
            let (d, _) = first_diff(
                payload(&got, written),
                &ref_nat[start * row_bytes..(start + s) * row_bytes],
            );
            let ok = d == 0 && region_ok(&got, written);
            inv_ok &= ok;
            partitions.push(json!({"rows": s, "start": start, "diff_bytes": d, "guards_ok": region_ok(&got, written), "ok": ok}));
        }
    }
    let mut fixture_json = Vec::new();
    for f in fixtures(n) {
        let rows = f.rows.len();
        let mut xg: Vec<f32> = Vec::with_capacity(rows * k);
        for r in &f.rows {
            match r {
                Some(r) => xg.extend_from_slice(&x[r * k..(r + 1) * k]),
                None => xg.extend(std::iter::repeat(0.0f32).take(k)),
            }
        }
        let xt = gpu.upload_f32(&xg, &[rows * k])?;
        part.poison(gpu)?;
        gpu.gemm_bf16_xf32_mtp_dense_det_pm(&w, &xt, &part.view(), m, k, rows)?;
        gpu.hip.device_synchronize()?;
        let got = part.fetch(gpu)?;
        let written = rows * row_bytes;
        let mut bad_rows = 0usize;
        let mut first_bad = None;
        let zero = vec![0u8; row_bytes];
        for (i, r) in f.rows.iter().enumerate() {
            let want: &[u8] = match r {
                Some(r) => &ref_nat[r * row_bytes..(r + 1) * row_bytes],
                None => &zero,
            };
            if payload(&got, written)[i * row_bytes..(i + 1) * row_bytes] != *want {
                bad_rows += 1;
                first_bad.get_or_insert(i);
            }
        }
        let ok = bad_rows == 0 && region_ok(&got, written);
        inv_ok &= ok;
        fixture_json.push(json!({"fixture": f.name, "rows": rows, "bad_rows": bad_rows, "first_bad_row": first_bad, "guards_ok": region_ok(&got, written), "ok": ok}));
        gpu.free_tensor(xt)?;
    }
    if !inv_ok {
        fails.push("row-local invariance failed".into());
    }
    if !inv_complete {
        fails.push("row-local invariance incomplete: N smaller than a required partition".into());
    }

    Ok(json!({
        "event": "receipt",
        "shape": {"M": m, "K": k, "N": n}, "seed": seed, "reps": reps, "warmups": WARM,
        "oracle_tile": ORACLE_TILE.entry(),
        "first_shadow_ms_cold_wall": first_shadow_ms,
        "conversion_vs_host_rne": {"shadow_diff_bytes": shadow_diff, "x16_diff_bytes": x16_diff, "ok": conv_ok},
        "first_run": first, "post_timing": post,
        "timed_outputs_equal_first_run": timed_same, "post_rerun_equal_first_run": post_same_as_first,
        "error_native_vs_incumbent": err_json(&err_nat, m),
        "error_oracle_vs_incumbent": err_json(&err_orc, m),
        "median_us": {"baseline_r16w4": med[0], "native_conversion_inclusive": med[1], "native_launch_only": med[2]},
        "all_us": {"baseline_r16w4": ms[0].iter().map(|v| v * 1000.0).collect::<Vec<_>>(),
                   "native_conversion_inclusive": ms[1].iter().map(|v| v * 1000.0).collect::<Vec<_>>(),
                   "native_launch_only": ms[2].iter().map(|v| v * 1000.0).collect::<Vec<_>>()},
        "speed": {"median_paired_ratio": pair_ratio, "ratio_of_medians": med_ratio, "launch_only_ratio_of_medians": ratio_launch,
                   "required": REQUIRED_RATIO, "pass": speed_pass},
        "row_invariance": {"ok": inv_ok, "complete": inv_complete, "partitions": partitions, "fixtures": fixture_json},
    }))
}

fn err_json(e: &ErrStats, m: usize) -> Value {
    json!({
        "max_abs": e.max_abs, "max_abs_at": {"n": e.max_abs_at / m, "c": e.max_abs_at % m},
        "max_rel": e.max_rel, "max_rel_denominator": "max(|incumbent|, 1e-3*rms_incumbent)",
        "rms": e.rms, "rms_incumbent": e.rms_ref, "rel_rms": e.rms / e.rms_ref,
        "nonfinite_candidate": e.nonfinite_cand, "nonfinite_incumbent": e.nonfinite_ref,
    })
}

/// Native (conversion-inclusive and launch-only) against the oracle over the whole buffers,
/// guards and un-overwritten poison of every arm.
fn check_outputs(
    stage: &str,
    b: &[u8],
    o: &[u8],
    n: &[u8],
    l: &[u8],
    pay: usize,
    fails: &mut Vec<String>,
) -> Value {
    let (nd, nf) = first_diff(n, o);
    let (ld, lf) = first_diff(l, o);
    let guards = [
        region_ok(b, pay),
        region_ok(o, pay),
        region_ok(n, pay),
        region_ok(l, pay),
    ];
    let left = [
        poison_left(payload(b, pay)),
        poison_left(payload(o, pay)),
        poison_left(payload(n, pay)),
        poison_left(payload(l, pay)),
    ];
    if nd != 0 {
        fails.push(format!(
            "{stage}: native conversion-inclusive differs from the oracle ({nd} bytes)"
        ));
    }
    if ld != 0 {
        fails.push(format!(
            "{stage}: native launch-only differs from the oracle ({ld} bytes)"
        ));
    }
    if guards.iter().any(|g| !g) {
        fails.push(format!(
            "{stage}: a guard was written {guards:?} (baseline, oracle, native, launch-only)"
        ));
    }
    if left.iter().any(|&p| p != 0) {
        fails.push(format!("{stage}: poison words left in an output {left:?} (baseline, oracle, native, launch-only)"));
    }
    json!({
        "stage": stage,
        "native_vs_oracle_diff_bytes": nd, "native_first_diff_byte": nf,
        "launch_only_vs_oracle_diff_bytes": ld, "launch_only_first_diff_byte": lf,
        "guards_intact": {"baseline": guards[0], "oracle": guards[1], "native": guards[2], "launch_only": guards[3]},
        "poison_words_left": {"baseline": left[0], "oracle": left[1], "native": left[2], "launch_only": left[3]},
    })
}

fn verdict_fail(reason: String, code: i32) -> ! {
    println!(
        "{}",
        json!({"event": "kill_verdict", "kill_pass": false, "reason": reason})
    );
    println!("KILL FAIL");
    std::process::exit(code);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = parse(&args, "--m", "2560");
    let k: usize = parse(&args, "--k", "2560");
    let n: usize = parse(&args, "--n", "4096");
    let reps: usize = parse(&args, "--reps", "20");
    let seed: u64 = parse(&args, "--seed", "184");
    if !mtp_dense_det_pm_shape(m, k, n) || k % 256 != 0 || n < 64 {
        usage(&format!("shape M={m} K={k} N={n}: need M%16==0, K%256==0 (the incumbent r16w4 baseline), N>=64, extents < 2^31"));
    }
    if reps == 0 {
        usage("--reps must be at least 1");
    }
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => verdict_fail(format!("gpu init: {e}"), 3),
    };
    let pci = gpu
        .hip
        .device_pci_bus_id(gpu.device_id)
        .unwrap_or_default()
        .to_ascii_lowercase();
    println!(
        "{}",
        json!({"event": "start", "arch": gpu.arch, "pci_bus_id": pci, "device_id": gpu.device_id, "shape": {"M": m, "K": k, "N": n}, "seed": seed, "reps": reps})
    );
    if gpu.arch != "gfx1151" || pci != REQUIRED_PCI {
        verdict_fail(
            format!(
                "device is {} at {pci:?}; the admitted device is gfx1151 at {REQUIRED_PCI}",
                gpu.arch
            ),
            3,
        );
    }
    let mut fails: Vec<String> = Vec::new();
    let receipt = match run(&mut gpu, m, k, n, reps, seed, &mut fails) {
        Ok(r) => r,
        Err(e) => verdict_fail(format!("error: {e}"), 1),
    };
    println!("{receipt}");
    let target = (m, k, n) == (2560, 2560, 4096);
    let speed_pass = receipt["speed"]["pass"].as_bool().unwrap_or(false);
    let correctness_pass = fails.iter().all(|f| f.starts_with("speed gate"));
    if !target {
        fails.push("not the first-kill shape (2560, 2560, 4096)".into());
    }
    let kill_pass = fails.is_empty();
    println!(
        "{}",
        json!({
            "event": "kill_verdict", "kill_pass": kill_pass, "target_shape": target,
            "correctness_pass": correctness_pass, "speed_pass": speed_pass,
            "median_paired_ratio": receipt["speed"]["median_paired_ratio"], "ratio_of_medians": receipt["speed"]["ratio_of_medians"],
            "required_ratio": REQUIRED_RATIO, "failures": fails,
        })
    );
    println!("KILL {}", if kill_pass { "PASS" } else { "FAIL" });
    std::process::exit(if kill_pass { 0 } else { 1 });
}
