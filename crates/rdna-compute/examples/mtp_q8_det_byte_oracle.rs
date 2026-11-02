// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP append DET Q8_0 byte oracle (gfx1151 only; developer example, needs a GPU: build only on
//! other hosts). Small bounded cell, no timing, no measurement claim.
//!
//! Exercises the SAME native module as the product route and the dense kill harness
//! (`mtp_dense_det_pm.rs`): the Q8_0 -> F16 weight shadow converter
//! (`Gpu::mtp_q8_det_pm_weight_shadow`, module `mtp_q8_det_pm_gfx1151`, element
//! `F16-RNE(f32(scale) * f32(q))`) and the native dense launch over that shadow
//! (`Gpu::gemm_q8_0_xf32_mtp_dense_det_pm` / `Gpu::mtp_q8_det_pm_launch`). No second numerics
//! model and no tolerance: every comparison is on bytes.
//!
//! 1. Converter. A deterministic finite Q8_0 `[16, 2048]` weight (1024 blocks = 128 scales x 8
//!    blocks): every scale appears with all 256 signed int8 values (-128..127) and, over the 128
//!    scales, every int8 value at every one of the 32 block slots. Scales are finite F16 bit
//!    patterns: +-0, F16 subnormals (smallest, largest), min normal, 1/-1, the max finite (+-65504),
//!    scales whose product with an int8 lands exactly on / around the 65504 / 65520 overflow
//!    boundary (65520 is the round-to-even tie that overflows), and random finite patterns. The
//!    host oracle is independent of the device: `F16RNE(scaleF32 * qF32)` with its own exact
//!    F16 -> F32 and RNE F32 -> F16 (self-checked by an exhaustive round trip). Every converted
//!    byte of the shadow is compared; the shadow cache is checked immutable (same pointer, same
//!    bytes after the device weight is overwritten, then restored) and the API's own guards
//!    (dtype, K % 64, packed capacity, cached-shadow size) must each return an error.
//!    Non-finite scales are NOT probed: NaN/Inf propagation is outside this oracle. F16 underflow
//!    of a non-zero product cannot occur here (a finite F16 scale times an int8 is an integer
//!    multiple of 2^-24, exact on the F16 subnormal grid); the receipt reports that count.
//! 2. Dense. A production-representative Q8_0 weight (M = 320, K = 2560; bounded finite scales
//!    so every output is finite) and F32 X `[17, 2560]`. For N in {17, 1, 15, 16} the native
//!    conversion-inclusive and launch-only outputs equal, byte for byte over the WHOLE guarded
//!    buffers, the existing ascending-K F16 HIP WMMA oracle (`gemm_f16_x_f16_wmma_lds_tiled`,
//!    fixed plain 128 x 256 tile) over the same shadow and an own copy of the X conversion; X16
//!    and the shadow are also checked against the host RNE. Row identity: ragged N rows equal the
//!    N = 17 rows, and reordered / duplicated / scattered / zero-padded / sub-partitioned batches
//!    each reproduce the matching N = 17 rows. API guards (shape, dtype, capacity) must error and
//!    leave the output poison untouched.
//!
//! Environment: `HIPFIRE_NO_DEVICE_COMPILER` must be set (non-empty, not `0`); the device is
//! exactly gfx1151 at PCI 0000:c0:00.0. No fallback: anything else exits 3.
//!
//! Output: JSON lines on stdout (`start`, `receipt`, `verdict`) and a final
//! `Q8 DET BYTE ORACLE PASS|FAIL` line. Exit 0 only on PASS, 1 FAIL, 2 usage, 3 environment /
//! device / HIP init failure.
//!
//! Usage: `HIPFIRE_NO_DEVICE_COMPILER=1 mtp_q8_det_byte_oracle [--seed 184]`

use rdna_compute::gemm::LdsTile;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const GUARD: usize = 4096;
const POISON: u8 = 0xA5;
const REQUIRED_PCI: &str = "0000:c0:00.0";
/// The oracle's fixed plain (non-pipelined) tile: 128 channels x 256 rows, K64 stage, no split-K.
const ORACLE_TILE: LdsTile = LdsTile::new(128, 256, 32, 64, 64, false);

/// Converter weight: 16 rows x 2048 = 1024 Q8_0 blocks = 128 scales x 8 blocks (256 int8 values).
const CONV_M: usize = 16;
const CONV_K: usize = 2048;
const CONV_SCALES: usize = 128;
/// Dense weight / batch: production-representative fixed-row projection, ragged N.
const PM: usize = 320;
const PK: usize = 2560;
const PN: usize = 17;
const RAGGED: [usize; 4] = [17, 1, 15, 16];
/// Most rows of any row-identity fixture (the scattered batch).
const FIXTURE_MAX_ROWS: usize = 40;

// ----------------------------------------------------------------- generators

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
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

/// Bounded finite F16 scale bits of dense block `block` in weight row `row`: ~1/64 each +0 / -0,
/// ~1/32 subnormal, else a row-dependent small magnitude (|scale| <= 1/64, so |w| <= 2).
fn gen_scale(block: u64, row: u64, seed: u64) -> u16 {
    let h = mix64(block.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ seed);
    match h & 63 {
        0 => 0x0000,
        1 => 0x8000,
        2 | 3 => (((h >> 8) & 0x3ff) as u16 | 1) | if h & 0x1_0000 != 0 { 0x8000 } else { 0 },
        _ => {
            let u = ((h >> 16) & 0xff_ffff) as f32 / 16_777_216.0;
            f32_to_f16_rne((u - 0.5) * (1.0 / 32.0) / (1u32 << (row % 5)) as f32)
        }
    }
}

/// Dense int8 weight value: uniform over the full signed range.
fn gen_q(elem: u64, seed: u64) -> i8 {
    (mix64(elem ^ seed ^ 0xD1B5_4A32) & 0xff) as u8 as i8
}

// ------------------------------------------------------------------ host F16

/// Exact F16 bits -> F32.
fn f16_to_f32(h: u16) -> f32 {
    let s = u32::from(h & 0x8000) << 16;
    let e = u32::from((h >> 10) & 31);
    let m = u32::from(h & 0x3ff);
    match e {
        0 => {
            // m * 2^-24, exact in F32
            let v = m as f32 * f32::from_bits(0x3380_0000);
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

/// The converter's element contract on the host: `F16RNE(f32(scale) * f32(q))`.
fn host_convert(scale: u16, q: i8) -> u16 {
    f32_to_f16_rne(f16_to_f32(scale) * f32::from(q))
}

/// Append one 34-byte Q8_0 block (F16 scale LE, 32 signed int8).
fn push_block(packed: &mut Vec<u8>, scale: u16, qs: impl Iterator<Item = i8>) {
    packed.extend(scale.to_le_bytes());
    let before = packed.len();
    packed.extend(qs.map(|q| q as u8));
    debug_assert_eq!(packed.len() - before, 32);
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

/// (differing byte count incl. length difference, first differing byte offset).
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

fn row_bad(a: &[u8], b: &[u8], row_bytes: usize) -> usize {
    a.chunks_exact(row_bytes)
        .zip(b.chunks_exact(row_bytes))
        .filter(|(x, y)| x != y)
        .count()
}

fn f32s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

/// Count of executed checks and the failure messages.
struct Ledger {
    checks: usize,
    fails: Vec<String>,
}
impl Ledger {
    fn check(&mut self, ok: bool, msg: impl FnOnce() -> String) {
        self.checks += 1;
        if !ok {
            self.fails.push(msg());
        }
    }
}

/// An API guard: the call must return an error whose message contains `needle`.
fn expect_err<T, E: std::fmt::Display>(
    lg: &mut Ledger,
    name: &str,
    r: Result<T, E>,
    needle: &str,
) -> Value {
    let (errored, msg) = match r {
        Ok(_) => (false, String::new()),
        Err(e) => (true, e.to_string()),
    };
    let ok = errored && msg.contains(needle);
    lg.check(ok, || {
        format!(
            "guard {name}: expected an error containing {needle:?}, got {}",
            if errored { msg.clone() } else { "Ok".into() }
        )
    });
    json!({"guard": name, "errored": errored, "needle": needle, "message": msg, "ok": ok})
}

fn hex16(v: u16) -> String {
    format!("0x{v:04x}")
}

// ----------------------------------------------------------------- converter

fn converter(gpu: &mut Gpu, seed: u64, lg: &mut Ledger) -> Res<Value> {
    let (m, k) = (CONV_M, CONV_K);
    let elems = m * k;
    let blocks = elems / 32;
    assert_eq!(blocks, CONV_SCALES * 8);

    // Host oracle self-check: every finite F16 survives F16 -> F32 -> F16 RNE.
    let mut rt_checked = 0usize;
    let mut rt_bad = 0usize;
    for h in 0..=u16::MAX {
        if (h >> 10) & 31 == 31 {
            continue;
        }
        rt_checked += 1;
        if f32_to_f16_rne(f16_to_f32(h)) != h {
            rt_bad += 1;
        }
    }
    lg.check(rt_bad == 0, || {
        format!("host F16 round trip: {rt_bad} of {rt_checked} finite patterns differ")
    });

    // Finite scales: boundary list, then random finite patterns up to CONV_SCALES.
    let mut scales: Vec<u16> = vec![
        0x0000, 0x8000, // +-0
        0x0001, 0x8001, 0x0002, // smallest subnormals
        0x03ff, 0x83ff, // largest subnormal
        0x0200, // mid subnormal
        0x0400, 0x8400, 0x0401, // min normal
        0x3555, // ~1/3
        0x3c00, 0xbc00, 0x3c01, 0x3bff, // +-1 and neighbours
        0x4000, // 2
        0x7bff, 0xfbff, 0x7bfe, 0x7800, // max finite and neighbours
        0x6000, // 512: q=-128 -> -65536 overflows, q=127 -> 65024 finite
        0x6010, // 520: q=126 -> 65520 = round-to-even tie that overflows
        0x600f, // 519
        0x5fff, // 511.75: q=-128 -> -65504 exactly the max finite
        0xdfff, 0xe010, // negatives of the above
    ];
    {
        let mut i = 0u64;
        while scales.len() < CONV_SCALES {
            let mut b = (mix64(i ^ seed ^ 0x5CA1E) & 0xffff) as u16;
            if (b >> 10) & 31 == 31 {
                b &= !0x0400; // exponent 30: stay finite
            }
            if !scales.contains(&b) {
                scales.push(b);
            }
            i += 1;
        }
    }
    assert_eq!(scales.len(), CONV_SCALES);

    // Blocks: scale s occupies 8 consecutive blocks j = 0..8 carrying q = -128 + 32 j + ((t + s%32) % 32)
    // at slot t, i.e. all 256 int8 values per scale, rotated so every value visits every slot.
    let mut packed: Vec<u8> = Vec::with_capacity(blocks * 34);
    let mut qs: Vec<i8> = Vec::with_capacity(elems);
    let mut expect_bytes: Vec<u8> = Vec::with_capacity(elems * 2);
    let mut seen_q = [false; 256];
    let mut seen_slot = vec![false; 256 * 32];
    let (mut ovf_pos, mut ovf_neg, mut subnormal, mut plus_zero, mut minus_zero) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut max_fin, mut inexact, mut underflow_nonzero, mut normal_finite) =
        (0usize, 0usize, 0usize, 0usize);
    for b in 0..blocks {
        let (s_idx, j) = (b / 8, b % 8);
        let scale = scales[s_idx];
        let rot = s_idx % 32;
        let block_q: Vec<i8> = (0..32)
            .map(|t| (-128 + 32 * j as i32 + ((t + rot) % 32) as i32) as i8)
            .collect();
        push_block(&mut packed, scale, block_q.iter().copied());
        for (t, &q) in block_q.iter().enumerate() {
            qs.push(q);
            let want = host_convert(scale, q);
            expect_bytes.extend(want.to_le_bytes());
            seen_q[(i16::from(q) + 128) as usize] = true;
            seen_slot[(i16::from(q) + 128) as usize * 32 + t] = true;
            let product = f16_to_f32(scale) * f32::from(q);
            let exp_field = (want >> 10) & 31;
            match want {
                0x7c00 => ovf_pos += 1,
                0xfc00 => ovf_neg += 1,
                0x0000 => plus_zero += 1,
                0x8000 => minus_zero += 1,
                _ => {}
            }
            if want & 0x7fff == 0 && product != 0.0 {
                underflow_nonzero += 1;
            }
            if exp_field == 0 && want & 0x3ff != 0 {
                subnormal += 1;
            }
            if want & 0x7fff == 0x7bff {
                max_fin += 1;
            }
            if exp_field != 0 && exp_field != 31 {
                normal_finite += 1;
            }
            if exp_field != 31 && f16_to_f32(want) != product {
                inexact += 1;
            }
        }
    }
    let q_values = seen_q.iter().filter(|&&s| s).count();
    let q_slot_pairs = seen_slot.iter().filter(|&&s| s).count();
    lg.check(q_values == 256, || {
        format!("coverage: {q_values} of 256 int8 values")
    });
    lg.check(q_slot_pairs == 256 * 32, || {
        format!("coverage: {q_slot_pairs} of 8192 (int8, slot) pairs")
    });
    lg.check(ovf_pos > 0 && ovf_neg > 0, || {
        format!("coverage: overflow +Inf {ovf_pos} / -Inf {ovf_neg}")
    });
    lg.check(subnormal > 0 && max_fin > 0, || {
        format!("coverage: subnormal {subnormal}, max finite {max_fin}")
    });
    lg.check(plus_zero > 0 && minus_zero > 0, || {
        format!("coverage: +0 {plus_zero}, -0 {minus_zero}")
    });
    lg.check(inexact > 0 && normal_finite > 0, || {
        format!("coverage: inexact {inexact}, normal finite {normal_finite}")
    });

    let w = upload_dtype(gpu, &packed, DType::Q8_0, elems)?;
    let shadow = gpu.mtp_q8_det_pm_weight_shadow(&w, m, k)?;
    gpu.hip.device_synchronize()?;
    let shadow_ptr = shadow.buf.as_ptr() as usize;
    let got = gpu.download_raw_bytes(&shadow)?;
    lg.check(got.len() == elems * 2, || {
        format!("shadow is {} bytes, expected {}", got.len(), elems * 2)
    });
    let (diff_bytes, first_byte) = first_diff(&got, &expect_bytes);
    let mut bad_elems = 0usize;
    let mut first_bad: Vec<Value> = Vec::new();
    for i in 0..elems.min(got.len() / 2) {
        let g = u16::from_le_bytes([got[2 * i], got[2 * i + 1]]);
        let want = u16::from_le_bytes([expect_bytes[2 * i], expect_bytes[2 * i + 1]]);
        if g != want {
            bad_elems += 1;
            if first_bad.len() < 8 {
                first_bad.push(json!({
                    "elem": i, "block": i / 32, "slot": i % 32, "scale_bits": hex16(scales[i / 32 / 8]),
                    "q": qs[i], "got": hex16(g), "want": hex16(want),
                }));
            }
        }
    }
    lg.check(diff_bytes == 0 && bad_elems == 0, || {
        format!("converter: {bad_elems} elements / {diff_bytes} bytes differ from the host F16RNE(scale*q) (first byte {first_byte:?})")
    });

    // Immutable cache: overwrite the device weight, the shadow must be the same buffer with the
    // same bytes (no reconversion); restore the weight afterwards.
    gpu.hip.memcpy_htod(&w.buf, &vec![0u8; packed.len()])?;
    let shadow2 = gpu.mtp_q8_det_pm_weight_shadow(&w, m, k)?;
    gpu.hip.device_synchronize()?;
    let same_ptr = shadow2.buf.as_ptr() as usize == shadow_ptr;
    let got2 = gpu.download_raw_bytes(&shadow2)?;
    let (reuse_diff, _) = first_diff(&got2, &got);
    gpu.hip.memcpy_htod(&w.buf, &packed)?;
    gpu.hip.device_synchronize()?;
    lg.check(same_ptr, || {
        "cache: second shadow call returned a different buffer".into()
    });
    lg.check(reuse_diff == 0, || {
        format!("cache: shadow bytes changed after the weight was overwritten ({reuse_diff} bytes)")
    });

    // API guards (each errors before allocating or launching; the cached shadow stays intact).
    let small = upload_dtype(gpu, &vec![0u8; 34 * 2], DType::F16, 64)?;
    let guards = vec![
        expect_err(
            lg,
            "dtype_not_q8_0",
            gpu.mtp_q8_det_pm_weight_shadow(&small, 1, 64),
            "needs a Q8_0 weight",
        ),
        expect_err(
            lg,
            "k_not_multiple_of_64",
            gpu.mtp_q8_det_pm_weight_shadow(&w, m, k - 32),
            "K % 64 == 0",
        ),
        expect_err(
            lg,
            "m_zero",
            gpu.mtp_q8_det_pm_weight_shadow(&w, 0, k),
            "K % 64 == 0",
        ),
        expect_err(
            lg,
            "packed_capacity_too_small",
            gpu.mtp_q8_det_pm_weight_shadow(&w, m * 2, k),
            "packed capacity",
        ),
        expect_err(
            lg,
            "cached_shadow_other_size",
            gpu.mtp_q8_det_pm_weight_shadow(&w, m, k / 2),
            "cached shadow is",
        ),
    ];
    gpu.hip.device_synchronize()?;
    let got3 = gpu.download_raw_bytes(&shadow)?;
    let (after_guard_diff, _) = first_diff(&got3, &expect_bytes);
    lg.check(after_guard_diff == 0, || {
        format!(
            "shadow differs from the host oracle after the guard calls ({after_guard_diff} bytes)"
        )
    });

    Ok(json!({
        "shape": {"M": m, "K": k, "blocks": blocks, "elements": elems, "scales": CONV_SCALES},
        "host_f16_roundtrip": {"checked": rt_checked, "bad": rt_bad},
        "excluded": "non-finite scales (NaN/Inf) are not probed",
        "coverage": {
            "int8_values": q_values, "int8_slot_pairs": q_slot_pairs,
            "overflow_pos_inf": ovf_pos, "overflow_neg_inf": ovf_neg, "subnormal_results": subnormal,
            "plus_zero": plus_zero, "minus_zero": minus_zero, "max_finite_results": max_fin,
            "normal_finite_results": normal_finite, "inexact_roundings": inexact,
            "underflow_to_zero_of_nonzero_product": underflow_nonzero,
            "underflow_note": "a finite F16 scale times an int8 is an integer multiple of 2^-24, exact on the F16 subnormal grid: only q = 0 reaches zero",
        },
        "bytes_compared": got.len(),
        "diff_bytes": diff_bytes, "diff_elements": bad_elems, "first_diff": first_bad,
        "cache": {"same_buffer": same_ptr, "diff_bytes_after_overwrite": reuse_diff},
        "guards": guards,
        "diff_bytes_after_guards": after_guard_diff,
    }))
}

// --------------------------------------------------------------------- dense

/// Whole guarded buffers of native conversion-inclusive and launch-only against the oracle.
#[allow(clippy::too_many_arguments)]
fn check_arms(stage: &str, n: usize, o: &[u8], nat: &[u8], lo: &[u8], lg: &mut Ledger) -> Value {
    let pay = n * PM * 4;
    let row_bytes = PM * 4;
    let (nd, nf) = first_diff(nat, o);
    let (ld, lf) = first_diff(lo, o);
    let nat_bad_rows = row_bad(payload(nat, pay), payload(o, pay), row_bytes);
    let lo_bad_rows = row_bad(payload(lo, pay), payload(o, pay), row_bytes);
    let guards = [region_ok(o, pay), region_ok(nat, pay), region_ok(lo, pay)];
    let left = [
        poison_left(payload(o, pay)),
        poison_left(payload(nat, pay)),
        poison_left(payload(lo, pay)),
    ];
    let nonfinite = f32s(payload(nat, pay)).filter(|v| !v.is_finite()).count();
    let nonzero = f32s(payload(nat, pay)).filter(|v| *v != 0.0).count();
    lg.check(nd == 0, || format!("{stage} N={n}: native conversion-inclusive differs from the oracle ({nd} bytes, {nat_bad_rows} rows)"));
    lg.check(ld == 0, || format!("{stage} N={n}: native launch-only differs from the oracle ({ld} bytes, {lo_bad_rows} rows)"));
    lg.check(guards.iter().all(|&g| g), || {
        format!("{stage} N={n}: a guard was written {guards:?} (oracle, native, launch-only)")
    });
    lg.check(left.iter().all(|&p| p == 0), || {
        format!(
            "{stage} N={n}: poison words left in an output {left:?} (oracle, native, launch-only)"
        )
    });
    lg.check(nonfinite == 0, || {
        format!("{stage} N={n}: {nonfinite} non-finite native outputs")
    });
    lg.check(nonzero > 0, || {
        format!("{stage} N={n}: every native output is zero")
    });
    json!({
        "stage": stage, "N": n, "payload_bytes": pay,
        "native_vs_oracle": {"diff_bytes": nd, "first_diff_byte": nf, "bad_rows": nat_bad_rows},
        "launch_only_vs_oracle": {"diff_bytes": ld, "first_diff_byte": lf, "bad_rows": lo_bad_rows},
        "guards_intact": {"oracle": guards[0], "native": guards[1], "launch_only": guards[2]},
        "poison_words_left": {"oracle": left[0], "native": left[1], "launch_only": left[2]},
        "nonfinite_native_outputs": nonfinite, "nonzero_native_outputs": nonzero,
    })
}

/// Expected rows of one identity fixture: `Some(r)` = row `r` of the full N = 17 output,
/// `None` = a zero-padding row (exactly +0.0).
struct Fixture {
    name: &'static str,
    rows: Vec<Option<usize>>,
}

fn fixtures() -> Vec<Fixture> {
    let reordered: Vec<Option<usize>> = (0..PN).rev().map(Some).collect();
    let mut duplicated: Vec<Option<usize>> = Vec::new();
    duplicated.extend(std::iter::repeat(Some(5)).take(19));
    duplicated.extend(std::iter::repeat(Some(PN - 1)).take(9));
    duplicated.extend(std::iter::repeat(Some(0)).take(4));
    let scattered: Vec<Option<usize>> = (0..FIXTURE_MAX_ROWS)
        .map(|i| Some((i * 7 + 3) % PN))
        .collect();
    let padded: Vec<Option<usize>> = (0..20)
        .map(|j| {
            if j % 2 == 0 {
                Some((j * 5 + 2) % PN)
            } else {
                None
            }
        })
        .collect();
    vec![
        Fixture {
            name: "reordered",
            rows: reordered,
        },
        Fixture {
            name: "duplicated",
            rows: duplicated,
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

fn dense(gpu: &mut Gpu, seed: u64, lg: &mut Ledger) -> Res<Value> {
    let blocks_per_row = PK / 32;
    let blocks = PM * blocks_per_row;

    // ---- operands: Q8_0 weight (packed 34-byte blocks) and its host F16 shadow expectation
    let mut packed: Vec<u8> = Vec::with_capacity(blocks * 34);
    let mut shadow_expect: Vec<u8> = Vec::with_capacity(PM * PK * 2);
    let (mut q_min, mut q_max) = (i8::MAX, i8::MIN);
    for b in 0..blocks {
        let scale = gen_scale(b as u64, (b / blocks_per_row) as u64, seed ^ 0x57A11);
        let q: Vec<i8> = (0..32)
            .map(|t| gen_q((b * 32 + t) as u64, seed ^ 0xA11CE))
            .collect();
        for &v in &q {
            q_min = q_min.min(v);
            q_max = q_max.max(v);
            shadow_expect.extend(host_convert(scale, v).to_le_bytes());
        }
        push_block(&mut packed, scale, q.into_iter());
    }
    let x: Vec<f32> = (0..(PN * PK) as u64)
        .map(|e| gen_x(e, seed ^ 0xC0FFEE))
        .collect();
    let w = upload_dtype(gpu, &packed, DType::Q8_0, PM * PK)?;
    let x32 = gpu.upload_f32(&x, &[PN * PK])?;

    let shadow = gpu.mtp_q8_det_pm_weight_shadow(&w, PM, PK)?;
    gpu.hip.device_synchronize()?;
    let shadow_bytes = gpu.download_raw_bytes(&shadow)?;
    let (shadow_diff, _) = first_diff(&shadow_bytes, &shadow_expect);
    lg.check(shadow_diff == 0, || {
        format!("dense weight shadow differs from the host F16RNE(scale*q) ({shadow_diff} bytes)")
    });
    drop(shadow_bytes);

    // ---- ragged cells: oracle / native / launch-only over the whole guarded buffers
    let mut cells = Vec::new();
    let mut full_nat: Vec<u8> = Vec::new();
    let mut full_orc: Vec<u8> = Vec::new();
    for &n in &RAGGED {
        let pay = n * PM * 4;
        let xv = x32.sub_offset(0, n * PK);
        let x16_view = gpu.qwen4_convert_f16_x(&xv, n * PK)?;
        gpu.hip.device_synchronize()?;
        let x16_bytes = gpu.download_raw_bytes(&x16_view)?;
        let x16_expect: Vec<u8> = x[..n * PK]
            .iter()
            .flat_map(|v| f32_to_f16_rne(*v).to_le_bytes())
            .collect();
        let (x16_diff, _) = first_diff(&x16_bytes, &x16_expect);
        lg.check(x16_diff == 0, || {
            format!("N={n}: X16 differs from the host F32 -> F16 RNE ({x16_diff} bytes)")
        });
        // Own copy: the conversion-inclusive arm reuses the shared X16 scratch.
        let x16 = upload_dtype(gpu, &x16_bytes, DType::F16, n * PK)?;

        let y_orc = Guarded::new(gpu, n * PM)?;
        let y_nat = Guarded::new(gpu, n * PM)?;
        let y_lo = Guarded::new(gpu, n * PM)?;
        let (yo, yn, yl) = (y_orc.view(), y_nat.view(), y_lo.view());
        gpu.gemm_f16_x_f16_wmma_lds_tiled(&shadow, &x16, &yo, None, PM, PK, n, ORACLE_TILE)?;
        gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &xv, &yn, PM, PK, n)?;
        gpu.mtp_q8_det_pm_launch(&w, &x16, &yl, PM, PK, n)?;
        gpu.hip.device_synchronize()?;
        let (o1, n1, l1) = (y_orc.fetch(gpu)?, y_nat.fetch(gpu)?, y_lo.fetch(gpu)?);
        let first = check_arms("first", n, &o1, &n1, &l1, lg);

        // Fresh poisoned re-run of the native arms.
        y_nat.poison(gpu)?;
        y_lo.poison(gpu)?;
        gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &xv, &yn, PM, PK, n)?;
        gpu.mtp_q8_det_pm_launch(&w, &x16, &yl, PM, PK, n)?;
        gpu.hip.device_synchronize()?;
        let (n2, l2) = (y_nat.fetch(gpu)?, y_lo.fetch(gpu)?);
        let rerun = check_arms("rerun", n, &o1, &n2, &l2, lg);

        // Row identity of the ragged batch against the N = 17 output.
        let prefix = if n == PN {
            full_nat = payload(&n1, pay).to_vec();
            full_orc = payload(&o1, pay).to_vec();
            json!({"reference": true})
        } else {
            let (nd, _) = first_diff(payload(&n1, pay), &full_nat[..pay]);
            let (od, _) = first_diff(payload(&o1, pay), &full_orc[..pay]);
            lg.check(nd == 0, || {
                format!("N={n}: native rows differ from the same rows of N={PN} ({nd} bytes)")
            });
            lg.check(od == 0, || {
                format!("N={n}: oracle rows differ from the same rows of N={PN} ({od} bytes)")
            });
            json!({"native_vs_n17_prefix_diff_bytes": nd, "oracle_vs_n17_prefix_diff_bytes": od})
        };
        cells.push(json!({"N": n, "x16_vs_host_diff_bytes": x16_diff, "first_run": first, "rerun": rerun, "prefix_identity": prefix}));
    }

    // ---- row reorder / duplicate / scatter / zero-pad / sub-partition identity (vs the N = 17 rows)
    let row_bytes = PM * 4;
    let part = Guarded::new(gpu, FIXTURE_MAX_ROWS * PM)?;
    let zero = vec![0u8; row_bytes];
    let mut fixture_json = Vec::new();
    for f in fixtures() {
        let rows = f.rows.len();
        let mut xg: Vec<f32> = Vec::with_capacity(rows * PK);
        for r in &f.rows {
            match r {
                Some(r) => xg.extend_from_slice(&x[r * PK..(r + 1) * PK]),
                None => xg.extend(std::iter::repeat(0.0f32).take(PK)),
            }
        }
        let xt = gpu.upload_f32(&xg, &[rows * PK])?;
        part.poison(gpu)?;
        gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &xt, &part.view(), PM, PK, rows)?;
        gpu.hip.device_synchronize()?;
        let got = part.fetch(gpu)?;
        let written = rows * row_bytes;
        let mut bad_rows = 0usize;
        let mut first_bad = None;
        for (i, r) in f.rows.iter().enumerate() {
            let want: &[u8] = match r {
                Some(r) => &full_nat[r * row_bytes..(r + 1) * row_bytes],
                None => &zero,
            };
            if payload(&got, written)[i * row_bytes..(i + 1) * row_bytes] != *want {
                bad_rows += 1;
                first_bad.get_or_insert(i);
            }
        }
        let guards_ok = region_ok(&got, written);
        lg.check(bad_rows == 0, || {
            format!(
                "fixture {}: {bad_rows} of {rows} rows differ (first {first_bad:?})",
                f.name
            )
        });
        lg.check(guards_ok, || {
            format!("fixture {}: a guard was written", f.name)
        });
        fixture_json.push(json!({"fixture": f.name, "rows": rows, "bad_rows": bad_rows, "first_bad_row": first_bad, "guards_ok": guards_ok}));
    }
    let mut subparts = Vec::new();
    for &(start, len) in &[
        (0usize, 1usize),
        (8, 1),
        (16, 1),
        (0, 15),
        (2, 15),
        (0, 16),
        (1, 16),
    ] {
        part.poison(gpu)?;
        let xv = x32.sub_offset(start * PK, len * PK);
        gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &xv, &part.view(), PM, PK, len)?;
        gpu.hip.device_synchronize()?;
        let got = part.fetch(gpu)?;
        let written = len * row_bytes;
        let (d, _) = first_diff(
            payload(&got, written),
            &full_nat[start * row_bytes..(start + len) * row_bytes],
        );
        let guards_ok = region_ok(&got, written);
        lg.check(d == 0, || {
            format!(
                "subpartition rows {start}..{}: {d} bytes differ from the N={PN} rows",
                start + len
            )
        });
        lg.check(guards_ok, || {
            format!(
                "subpartition rows {start}..{}: a guard was written",
                start + len
            )
        });
        subparts
            .push(json!({"start": start, "rows": len, "diff_bytes": d, "guards_ok": guards_ok}));
    }

    // ---- dense API guards: error, nothing written
    part.poison(gpu)?;
    let ysmall = part.view().sub_offset(0, PM);
    let guards = vec![
        expect_err(
            lg,
            "m_not_multiple_of_16",
            gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &x32, &part.view(), 24, PK, 1),
            "outside the kernel's limits",
        ),
        expect_err(
            lg,
            "n_zero",
            gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &x32, &part.view(), PM, PK, 0),
            "outside the kernel's limits",
        ),
        expect_err(
            lg,
            "launch_x_not_f16",
            gpu.mtp_q8_det_pm_launch(&w, &x32, &part.view(), PM, PK, 1),
            "dtypes must be weight",
        ),
        expect_err(
            lg,
            "y_capacity_too_small",
            gpu.gemm_q8_0_xf32_mtp_dense_det_pm(&w, &x32, &ysmall, PM, PK, 2),
            "buffer capacity is too small",
        ),
    ];
    gpu.hip.device_synchronize()?;
    let untouched = part.fetch(gpu)?.iter().all(|&b| b == POISON);
    lg.check(untouched, || {
        "a rejected dense call wrote its output".into()
    });

    // ---- the shadow is immutable across every launch above
    let shadow_after = gpu.download_raw_bytes(&shadow)?;
    let (shadow_after_diff, _) = first_diff(&shadow_after, &shadow_expect);
    lg.check(shadow_after_diff == 0, || {
        format!("dense weight shadow changed during the launches ({shadow_after_diff} bytes)")
    });

    Ok(json!({
        "shape": {"M": PM, "K": PK, "N": PN, "ragged_N": RAGGED}, "oracle_tile": ORACLE_TILE.entry(),
        "weight": {"q8_blocks": blocks, "packed_bytes": packed.len(), "int8_min": q_min, "int8_max": q_max},
        "shadow_vs_host_diff_bytes": shadow_diff, "shadow_after_launches_diff_bytes": shadow_after_diff,
        "cells": cells,
        "row_identity": {"fixtures": fixture_json, "subpartitions": subparts},
        "guards": guards, "rejected_calls_left_output_untouched": untouched,
    }))
}

// ---------------------------------------------------------------------- main

fn verdict_fail(reason: String, code: i32) -> ! {
    println!(
        "{}",
        json!({"event": "verdict", "pass": false, "reason": reason})
    );
    println!("Q8 DET BYTE ORACLE FAIL");
    std::process::exit(code);
}

fn usage(msg: &str) -> ! {
    eprintln!("{msg}\nusage: mtp_q8_det_byte_oracle [--seed 184]");
    std::process::exit(2);
}

fn main() {
    let start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u64 = 184;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seed" => match it.next().and_then(|v| v.parse().ok()) {
                Some(v) => seed = v,
                None => usage("--seed: not a number"),
            },
            other => usage(&format!("unknown argument {other}")),
        }
    }
    let no_compiler = hipfire_config::developer_var_os("HIPFIRE_NO_DEVICE_COMPILER")
        .is_some_and(|v| v != "0" && !v.is_empty());
    if !no_compiler {
        verdict_fail("HIPFIRE_NO_DEVICE_COMPILER is not set: this oracle runs the pre-built pack only, with no runtime compile".into(), 3);
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
        json!({
            "event": "start", "arch": gpu.arch, "pci_bus_id": pci, "device_id": gpu.device_id, "seed": seed,
            "no_device_compiler": no_compiler,
        })
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

    let mut lg = Ledger {
        checks: 0,
        fails: Vec::new(),
    };
    let conv = match converter(&mut gpu, seed, &mut lg) {
        Ok(v) => v,
        Err(e) => verdict_fail(format!("converter error: {e}"), 1),
    };
    let dense = match dense(&mut gpu, seed, &mut lg) {
        Ok(v) => v,
        Err(e) => verdict_fail(format!("dense error: {e}"), 1),
    };
    println!(
        "{}",
        json!({
            "event": "receipt", "arch": gpu.arch, "pci_bus_id": pci, "seed": seed,
            "converter": conv, "dense": dense,
        })
    );
    let pass = lg.fails.is_empty();
    println!(
        "{}",
        json!({
            "event": "verdict", "pass": pass, "checks": lg.checks, "failed": lg.fails.len(), "failures": lg.fails,
            "elapsed_s": start.elapsed().as_secs_f64(),
        })
    );
    println!("Q8 DET BYTE ORACLE {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
