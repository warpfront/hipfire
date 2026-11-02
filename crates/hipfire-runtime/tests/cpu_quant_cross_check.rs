// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Live cross-check of the host-side weight decoder over the **real** tensors of
//! the on-disk fixtures.
//!
//! `weight_backend::dequantize_to_f32` now delegates every block-quantized
//! format to `hipfire_cpu::quant::dequant_group`, so for those formats this test
//! compares the canonical decoder to itself. Its remaining two-sided value is
//! the formats `dequantize_to_f32` still decodes locally (qt 1/2/3/14/16). The
//! independent bit-exact oracle for the delegated formats is `hipfire-cpu`'s own
//! `EXPECTED_*` table test (`quant.rs`), whose tables were generated from the
//! canonical decoder before the cutover — keep both green.
//!
//! Not part of `scripts/no-gpu-ci.sh`'s curated list (it needs model files), and
//! it decodes real tensors, so run it optimized:
//!
//! ```text
//! cargo test --release -p hipfire-runtime --test cpu_quant_cross_check -- --nocapture
//! ```
//!
//! Fixtures are looked up in `$HIPFIRE_MODELS_DIR` (default `~/.hipfire/models`)
//! and a missing file is reported and skipped, like the `skip: no GPU` shape the
//! rest of the suite uses. A file that *is* present but yields no checkable
//! tensor is a failure: that would mean the check itself stopped exercising the
//! decode path.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Rows of each tensor checked. The group stride is fully exercised by one row
/// (all `k / group_elems` groups); the row stride by taking more than one, which
/// is why this is 4 and not 1. Bounded so the check stays fast on the 9B's
/// `[4096, 12288]` and `[12288, 4096]` tensors.
const CHECK_ROWS: usize = 4;

use hipfire_cpu::quant::{dequant_group, CpuQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::dequantize_weight_to_f32;

/// Registry fixtures, smallest first. The 2B variants are the per-format
/// fixtures (`mq3` = qt 17, `mq6` = qt 15, `hf6` = qt 8, `mq4` = qt 13).
const FIXTURES: [&str; 7] = [
    "qwen3.5-2b.mq4",
    "qwen3.5-2b.mq3",
    "qwen3.5-2b.mq6",
    "qwen3.5-2b.hf6",
    "qwen3.5-9b.mq4",
    "qwen3.8-27b.mq4",
    "ornith-1.5-35b-a3b.mq4",
];

fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| PathBuf::from("/").into());
    PathBuf::from(home).join(".hipfire").join("models")
}

#[test]
fn dequant_group_matches_the_canonical_decoder_on_real_tensors() {
    let dir = models_dir();
    let mut checked = 0usize;
    let mut files_seen = 0usize;
    let mut formats: BTreeSet<u8> = BTreeSet::new();

    for name in FIXTURES {
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            continue;
        }
        files_seen += 1;
        let hfq = HfqFile::open(&path).expect("open fixture");
        for info in hfq.tensors() {
            let Some(q) = CpuQuant::from_quant_type(info.quant_type) else {
                continue;
            };
            let elems: usize = info.shape.iter().map(|d| *d as usize).product();
            if elems == 0 || elems % q.group_elems() != 0 {
                continue;
            }
            let (_, data) = hfq
                .tensor_data_vec(&info.name)
                .unwrap_or_else(|| panic!("{}: no bytes for {}", name, info.name));
            assert_eq!(
                data.len(),
                info.data_size,
                "{name}: {} data_size",
                info.name
            );

            // A 2-D tensor's byte count must be exactly `m` rows of the group
            // layout — the property `hipfire_cpu::gemv` derives its row pointer
            // from, checked here against the file rather than against itself.
            let m = if info.shape.len() == 2 {
                let rows = info.shape[0] as usize;
                assert_eq!(
                    data.len(),
                    rows * hipfire_cpu::gemv::row_bytes(q, elems / rows),
                    "{name}: {} byte count vs m * row_bytes",
                    info.name
                );
                rows
            } else {
                1
            };
            let k = elems / m;
            let (ge, gb) = (q.group_elems(), q.group_bytes());
            let rb = hipfire_cpu::gemv::row_bytes(q, k);

            let rows = m.min(CHECK_ROWS);
            let mut got = vec![0.0f32; k];
            for row in 0..rows {
                let row_bytes = &data[row * rb..(row + 1) * rb];
                // Canonical decoder over the row …
                let want = dequantize_weight_to_f32(info.quant_type, row_bytes, k);
                assert_eq!(want.len(), k, "{name}: {} decode length", info.name);

                // … versus the transcription, group by group.
                for g in 0..k / ge {
                    if q.row_header_bytes() == 0 {
                        dequant_group(q, &row_bytes[g * gb..], &mut got[g * ge..(g + 1) * ge]);
                    } else {
                        // E8 keeps its row scale once, before all FWHT groups.
                        let mut group = [0u8; 152];
                        group[..16].copy_from_slice(&row_bytes[..16]);
                        group[16..].copy_from_slice(&row_bytes[16 + g * gb..16 + (g + 1) * gb]);
                        dequant_group(q, &group, &mut got[g * ge..(g + 1) * ge]);
                    }
                }

                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{name}: {} (qt {}) row {row} element {i}: {a} (0x{:08x}) != {b} (0x{:08x})",
                        info.name,
                        info.quant_type,
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
            checked += 1;
            formats.insert(info.quant_type);
        }
    }

    eprintln!(
        "cpu_quant_cross_check: {checked} tensors over {files_seen} fixtures, quant types {formats:?}"
    );
    if files_seen == 0 {
        eprintln!("skip: no fixture found in {}", dir.display());
        return;
    }
    assert!(
        checked > 0,
        "no tensor in {files_seen} fixture(s) mapped to a CpuQuant — the cross-check \
         stopped exercising the decode path"
    );
    // Every fixture on disk is MQ4/MQ3/MQ6/HF6/F16/Q8, so a run against real
    // files must cover at least the rotated 4-bit path.
    assert!(
        formats.contains(&13) || formats.contains(&17),
        "expected at least one FWHT-rotated format among the checked tensors, got {formats:?}"
    );
}
/// Bit-exact equivalence of the canonical `dequantize_to_f32` arms for the
/// V2 MQ family (qt 44/45/47-50) against `hipfire_cpu::quant::dequant_group`,
/// on synthetic groups — the fixture files carry none of these dtypes, so the
/// live test above never exercises them. An fp16 header is built from
/// non-trivial bit patterns and the code payload cycles every valid code, so
/// both the scale/zero plumbing and the bit unpacking are covered.
fn f16le(v: f32) -> [u8; 2] {
    // Only need a stable, non-trivial bit pattern — the decoders widen it.
    let bits = half::f16_from_f32_bits(v);
    bits.to_le_bytes()
}

/// Minimal f32→f16 encoder (round-to-nearest-even) sufficient for the test's
/// chosen values — avoids pulling a half crate for one test.
mod half {
    pub fn f16_from_f32_bits(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x7fffff;
        if exp == 0 {
            return sign;
        }
        let e = exp - 127 + 15;
        if e >= 31 {
            return sign | 0x7c00;
        }
        if e <= 0 {
            // subnormal / underflow — flush through round-to-nearest.
            let m = mant | 0x800000;
            let shift = (14 - e) as u32;
            let mut m16 = (m >> shift) as u16;
            if (m >> (shift - 1)) & 1 == 1 {
                m16 = m16.wrapping_add(1);
            }
            return sign | m16;
        }
        let mut m16 = (mant >> 13) as u16;
        if mant & 0x1000 != 0 && (mant & 0x0fff != 0 || m16 & 1 == 1) {
            m16 = m16.wrapping_add(1);
        }
        sign | ((e as u16) << 10) | (m16 & 0x3ff)
    }
}

/// Build one synthetic group for `qt`, `v2_halves` layout `[s0 z0 s1 z1]` as
/// fp16 (qt45 reuses half-0's pair for both halves, per its single-grid spec).
fn synthetic_group(qt: u8) -> Vec<u8> {
    let (payload_len, group_bytes) = match qt {
        44 | 45 => (128, 136),
        47 => (192, 200),
        48 => (160, 168),
        49 => (96, 104),
        50 => (64, 72),
        _ => unreachable!(),
    };
    let max_code = match qt {
        47 => 63u32,
        48 => 31,
        49 => 7,
        50 => 3,
        _ => 15, // 44, 45
    };
    let mut g = vec![0u8; group_bytes];
    // Header: distinct half scale/zeros so per-half plumbing is exercised.
    // Chosen values are exactly representable in f16, so encode/decode is lossless.
    let (s0, z0, s1, z1) = (0.75f32, -0.5f32, 1.25f32, 0.25f32);
    g[0..2].copy_from_slice(&f16le(s0));
    g[2..4].copy_from_slice(&f16le(z0));
    g[4..6].copy_from_slice(&f16le(if qt == 45 { s0 } else { s1 }));
    g[6..8].copy_from_slice(&f16le(if qt == 45 { z0 } else { z1 }));

    // Codes 0..max_code cycling.
    let codes: Vec<u32> = (0..256).map(|i| (i as u32) % (max_code + 1)).collect();
    let p = &mut g[8..8 + payload_len];
    match qt {
        44 | 45 => {
            for (i, c) in codes.iter().enumerate() {
                let b = &mut p[i / 2];
                if i % 2 == 0 {
                    *b |= *c as u8;
                } else {
                    *b |= (*c as u8) << 4;
                }
            }
        }
        47 => {
            for ch in codes.chunks(4).enumerate() {
                let (bo, c) = (ch.0 * 3, ch.1);
                p[bo] = (c[0] | (c[1] << 6)) as u8;
                p[bo + 1] = ((c[1] >> 2) | (c[2] << 4)) as u8;
                p[bo + 2] = ((c[2] >> 4) | (c[3] << 2)) as u8;
            }
        }
        48 => {
            for ch in codes.chunks(8).enumerate() {
                let (bo, c) = (ch.0 * 5, ch.1);
                let w = (0..8).fold(0u64, |w, i| w | (c[i] as u64) << (5 * i));
                p[bo..bo + 5].copy_from_slice(&w.to_le_bytes()[..5]);
            }
        }
        49 => {
            for ch in codes.chunks(8).enumerate() {
                let (bo, c) = (ch.0 * 3, ch.1);
                let w = (0..8).fold(0u32, |w, i| w | c[i] << (3 * i));
                p[bo..bo + 3].copy_from_slice(&w.to_le_bytes()[..3]);
            }
        }
        50 => {
            for (i, c) in codes.iter().enumerate() {
                p[i / 4] |= (*c as u8) << (2 * (i % 4));
            }
        }
        _ => unreachable!(),
    }
    g
}

#[test]
fn dequantize_v2_family_matches_cpu_transcription_on_synthetic_groups() {
    for qt in [44u8, 45, 47, 48, 49, 50] {
        let q =
            CpuQuant::from_quant_type(qt).unwrap_or_else(|| panic!("qt {qt} unmapped in CpuQuant"));
        let group = synthetic_group(qt);
        assert_eq!(group.len(), q.group_bytes(), "qt {qt} group size");

        // Two groups back-to-back so the canonical decoder's group loop and
        // the per-group stride are both exercised.
        let mut data = group.clone();
        data.extend_from_slice(&group);

        let want = dequantize_weight_to_f32(qt, &data, 512);
        assert_eq!(want.len(), 512, "qt {qt} decode length");

        let gb = q.group_bytes();
        let ge = q.group_elems();
        let mut got = vec![0.0f32; 512];
        for gidx in 0..2 {
            dequant_group(
                q,
                &data[gidx * gb..(gidx + 1) * gb],
                &mut got[gidx * ge..(gidx + 1) * ge],
            );
        }
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "qt {qt} element {i}: {a} (0x{:08x}) != {b} (0x{:08x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}

// ---- planar expert-slab codecs (qt 131/132/135/136) --------------------------------
//
// `hipfire-cpu` transcribes the E8P grid / `decode8` and the W3 bit layout so it
// stays a rayon-only leaf. These tests hold the transcription to the runtime's
// oracles (`e8p_i4_reference`, `mq_i4`, `mq_i4_reference`), which are the single
// source of truth for the wire formats.

use hipfire_cpu::gemv::{gemv_with_simd, row_bytes, tensor_bytes};
use hipfire_cpu::quant::{decode_planar_row, e8p_decode_word};
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4::{self, MqI4Profile, MqI4Rotation};
use hipfire_runtime::mq_i4_reference;

/// xorshift64* — deterministic fixtures.
struct Xs(u64);

impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// A finite, strictly positive, normal f16 bit pattern (exponent field 1..=30):
/// legal for both writers' scale guards (`>= 2^-14`, finite, sign clear).
fn random_scale_bits(rng: &mut Xs) -> u16 {
    let exp = 1 + (rng.next() % 30) as u16;
    let man = (rng.next() % 1024) as u16;
    (exp << 10) | man
}

fn random_x(rng: &mut Xs, k: usize) -> Vec<f32> {
    (0..k)
        .map(|_| ((rng.next() % 4096) as f32 - 2048.0) / 1024.0)
        .collect()
}

/// Planar E8P tensor bytes: the words plane (little-endian u16) then the scales.
fn e8p_tensor_bytes(words: &[u16], scale_bits: &[u16]) -> Vec<u8> {
    words
        .iter()
        .chain(scale_bits)
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// Every row of a planar `[m, k]` tensor through `hipfire_cpu`'s scalar decoder.
fn cpu_decode_rows(q: CpuQuant, bytes: &[u8], m: usize, k: usize) -> Vec<f32> {
    let (rb, spr) = (row_bytes(q, k), 2 * (k / 128));
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let at = m * rb + r * spr;
        decode_planar_row(
            q,
            &bytes[r * rb..(r + 1) * rb],
            &bytes[at..at + spr],
            k,
            &mut out[r * k..(r + 1) * k],
        );
    }
    out
}

/// (a) The transcribed E8P grid equals `E8P_ABS2`, and the transcribed decode
/// equals `decode8` for **every** 16-bit wire word — both `e8p_decode_word`
/// directly and through `decode_planar_row` (k = 128 rows, 16 words each, scale
/// `d = 4.0` so the decoded weight is exactly `v`).
#[test]
fn e8p_grid_and_decode_match_the_runtime_oracle_for_every_word() {
    for idx in 0..256usize {
        assert_eq!(
            hipfire_cpu::quant::E8P_ABS2[idx],
            e8p::E8P_ABS2[idx],
            "grid row {idx}"
        );
    }
    let q = CpuQuant::E8pI4G128R128;
    let scales = 0x4400u16.to_le_bytes(); // f16 4.0 -> (4.0 * 0.25) = 1.0
    let mut out = vec![0.0f32; 128];
    for base in (0..65536u32).step_by(16) {
        let mut codes = [0u8; 32];
        for w in 0..16u32 {
            let word = (base + w) as u16;
            codes[2 * w as usize..2 * w as usize + 2].copy_from_slice(&word.to_le_bytes());
        }
        decode_planar_row(q, &codes, &scales, 128, &mut out);
        for w in 0..16usize {
            let word = (base as usize + w) as u16;
            let (idx, sign7, shift) = e8p::unpack_word_fields(word);
            let want = e8p::decode8(idx, sign7, shift);
            let direct = e8p_decode_word(word);
            for lane in 0..8 {
                assert_eq!(
                    direct[lane], want[lane],
                    "word {word:#06x} lane {lane}: e8p_decode_word"
                );
                assert_eq!(
                    out[8 * w + lane],
                    want[lane] as f32,
                    "word {word:#06x} lane {lane}: decode_planar_row"
                );
            }
        }
    }
}

/// (b) `decode_planar_row` over random planar tensors equals the runtime's decoded
/// weights bit for bit: E8P vs `decode_weights_f32`, W3 vs `mq_i4::decode_tensor`'s
/// `f16(d) * (2c - 7)`. Also pins `tensor_bytes` to the runtime's byte contract.
#[test]
fn planar_row_decode_matches_the_runtime_decoders_bit_for_bit() {
    let mut rng = Xs(0x1234_5678_9abc_def1);

    // E8P (both rotations' shapes, incl. the down k = 640 and a lone group).
    for (q, m, k) in [
        (CpuQuant::E8pI4G128R256, 5usize, 512usize),
        (CpuQuant::E8pI4G128R256, 2, 2560),
        (CpuQuant::E8pI4G128R128, 4, 640),
        (CpuQuant::E8pI4G128R128, 3, 128),
    ] {
        let words: Vec<u16> = (0..m * k / 8).map(|_| rng.next() as u16).collect();
        let scale_bits: Vec<u16> = (0..m * k / 128)
            .map(|_| random_scale_bits(&mut rng))
            .collect();
        let want = e8p::decode_weights_f32(m, k, &words, &scale_bits);
        let bytes = e8p_tensor_bytes(&words, &scale_bits);
        assert_eq!(bytes.len(), tensor_bytes(q, m, k), "{q:?} {m}x{k} bytes");
        let got = cpu_decode_rows(q, &bytes, m, k);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{q:?} {m}x{k} element {i}: {a} != {b}"
            );
        }
    }

    // W3.
    for (q, profile, m, k) in [
        (
            CpuQuant::Mq3I4G128R256,
            MqI4Profile::MQ3_R256,
            5usize,
            512usize,
        ),
        (CpuQuant::Mq3I4G128R256, MqI4Profile::MQ3_R256, 2, 2560),
        (CpuQuant::Mq3I4G128R128, MqI4Profile::MQ3_R128, 4, 640),
        (CpuQuant::Mq3I4G128R128, MqI4Profile::MQ3_R128, 3, 128),
    ] {
        let codes: Vec<u8> = (0..m * k).map(|_| (rng.next() % 8) as u8).collect();
        let scale_bits: Vec<u16> = (0..m * k / 128)
            .map(|_| random_scale_bits(&mut rng))
            .collect();
        let bytes =
            mq_i4::encode_tensor(profile, m, k, &codes, &scale_bits).expect("encode W3 tensor");
        let total = mq_i4::tensor_bytes(profile, m, k).expect("tensor_bytes");
        assert_eq!(bytes.len(), total);
        assert_eq!(tensor_bytes(q, m, k), total, "{q:?} {m}x{k} tensor_bytes");
        let t = mq_i4::decode_tensor(profile, m, k, &bytes).expect("decode W3 tensor");
        let want: Vec<f32> = (0..m * k)
            .map(|i| {
                ::half::f16::from_bits(t.scale_bits[i / 128]).to_f32()
                    * (2 * t.codes[i] as i32 - 7) as f32
            })
            .collect();
        let got = cpu_decode_rows(q, &bytes, m, k);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{q:?} {m}x{k} element {i}: {a} != {b}"
            );
        }
    }

    // The planar byte contract at the real Flash-Next expert shapes: E8P is the
    // 2-bit plane, so it has the W2 profiles' sizes.
    for (profile, q3, qe, m, k) in [
        (
            MqI4Profile::MQ2_R256,
            CpuQuant::Mq3I4G128R256,
            CpuQuant::E8pI4G128R256,
            1280usize,
            2560usize,
        ),
        (
            MqI4Profile::MQ2_R128,
            CpuQuant::Mq3I4G128R128,
            CpuQuant::E8pI4G128R128,
            2560,
            640,
        ),
    ] {
        assert_eq!(
            tensor_bytes(qe, m, k),
            mq_i4::tensor_bytes(profile, m, k).unwrap()
        );
        let w3 = MqI4Profile {
            codec: mq_i4::MqI4Codec::W3,
            rotation: profile.rotation,
        };
        assert_eq!(
            tensor_bytes(q3, m, k),
            mq_i4::tensor_bytes(w3, m, k).unwrap()
        );
    }
}

/// (c) `hipfire-cpu` GEMV (forced scalar and runtime-detected AVX2) against the
/// runtime's GEMV twins — `decode_gemv_f32_lanepair` for E8P, `decode_gemv_f32`
/// for W3 — at both rotations' shapes.
///
/// The two sides are different f32 summation orders (the twins use the frozen F4
/// tree), and the rows are random-sign sums that can cancel almost entirely, so
/// the `1e-5` relative bound is taken against `Σ|w·x|`, the scale f32
/// accumulation error is proportional to, rather than against `|y|` — which
/// would fail a few rows per hundred for reasons that are not decode errors.
/// A decode error moves a row by a whole weight (`~Σ|w·x| / k`) and is orders of
/// magnitude over the bound.
#[test]
fn planar_gemv_matches_the_runtime_gemv_twins() {
    let mut rng = Xs(0x0bad_cafe_f00d_1234);
    let check = |q: CpuQuant,
                 m: usize,
                 k: usize,
                 bytes: &[u8],
                 weights: &[f32],
                 x: &[f32],
                 twin: &[f32]| {
        for requested in [Some(false), None] {
            let mut y = vec![0.0f32; m];
            gemv_with_simd(q, bytes, m, k, x, &mut y, requested);
            for r in 0..m {
                let l1: f64 = (0..k)
                    .map(|i| (weights[r * k + i] as f64 * x[i] as f64).abs())
                    .sum();
                let err = (y[r] as f64 - twin[r] as f64).abs();
                assert!(
                    err <= 1e-5 * l1,
                    "{q:?} {m}x{k} {requested:?} row {r}: cpu {} vs twin {} (err {err:.3e}, Σ|wx| {l1:.3e})",
                    y[r],
                    twin[r]
                );
            }
        }
    };

    for (q, rotation, m, k) in [
        (
            CpuQuant::E8pI4G128R256,
            MqI4Rotation::R256,
            3usize,
            256usize,
        ),
        (CpuQuant::E8pI4G128R256, MqI4Rotation::R256, 40, 2560),
        (CpuQuant::E8pI4G128R128, MqI4Rotation::R128, 3, 128),
        (CpuQuant::E8pI4G128R128, MqI4Rotation::R128, 40, 640),
    ] {
        let words: Vec<u16> = (0..m * k / 8).map(|_| rng.next() as u16).collect();
        let scale_bits: Vec<u16> = (0..m * k / 128)
            .map(|_| random_scale_bits(&mut rng))
            .collect();
        let x = random_x(&mut rng, k);
        let weights = e8p::decode_weights_f32(m, k, &words, &scale_bits);
        let twin = e8p::decode_gemv_f32_lanepair(rotation, m, k, &words, &scale_bits, &x);
        let bytes = e8p_tensor_bytes(&words, &scale_bits);
        check(q, m, k, &bytes, &weights, &x, &twin);
    }

    for (q, profile, m, k) in [
        (
            CpuQuant::Mq3I4G128R256,
            MqI4Profile::MQ3_R256,
            3usize,
            256usize,
        ),
        (CpuQuant::Mq3I4G128R256, MqI4Profile::MQ3_R256, 40, 2560),
        (CpuQuant::Mq3I4G128R128, MqI4Profile::MQ3_R128, 3, 128),
        (CpuQuant::Mq3I4G128R128, MqI4Profile::MQ3_R128, 40, 640),
    ] {
        let codes: Vec<u8> = (0..m * k).map(|_| (rng.next() % 8) as u8).collect();
        let scale_bits: Vec<u16> = (0..m * k / 128)
            .map(|_| random_scale_bits(&mut rng))
            .collect();
        let x = random_x(&mut rng, k);
        let bytes =
            mq_i4::encode_tensor(profile, m, k, &codes, &scale_bits).expect("encode W3 tensor");
        let weights = cpu_decode_rows(q, &bytes, m, k);
        let twin = mq_i4_reference::decode_gemv_f32(profile, m, k, &bytes, &x);
        check(q, m, k, &bytes, &weights, &x, &twin);
    }
}
