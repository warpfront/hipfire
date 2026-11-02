// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! S1 packing oracle: `pack_matrix` / `pack_shape` / `pack_trunk` against an independent CPU oracle.
//!
//! The oracle (`mod oracle`) is the lab CPU twin, copied verbatim where marked from
//! `lab/npu-a8emu` `crates/rdna-compute/examples/npu_a8emu_check.rs` (worktree
//! `/home/kaden/ClaudeCode/warpfront/wt-npu-a8emu`): `Rng` (l.41-56), `f16_bits_to_f32` (l.58-75),
//! `f32_to_f16_rne` (l.77-110), `pack_group` (l.127-136), `unpack_group` (l.138-147), `decode_mq6`
//! (l.175-196) and the weight half of `cpu_w8a8` (l.202-217: `w = q*s + z` separate mul then add,
//! `ws = ((absmax as f64)/127.0) as f32`, `w8 = clamp(round_ties_even(w/ws), ±127)`, `ws == 0` => zero codes).
//! That is the contract of INTERFACES.md §0 (`npu_a8emu_wprep_w8`).
//!
//! The 6-bit code decode is cross-checked inside the oracle against a second, independent bit-stream
//! decoder (field `j` = bits `6j..6j+6` of the little-endian 192 B payload), so the oracle is not tied
//! to the shift/mask layout the packer under test may share with the lab code.
//!
//! The panel byte layout is owned by `peacemaker-npu` `Geometry::pack_b` and is not re-derived here; panel
//! tests check structure, determinism and the `ws` vector only.

use hipfire_arch_qwen4::npu_trunk::{
    pack_matrix, pack_shape, pack_shape_with_contract, pack_trunk, pack_trunk_with_contract,
    Family, FamilyMask, LayerTrunkRefs, MatrixRef, Shape, PANEL_COLS, TRUNK_K,
};
use hipfire_runtime::coproc::EpilogueContract;

mod oracle {
    //! Lab CPU oracle, see the file header for the source citation.

    pub const GROUP_BYTES: usize = 200;

    // npu_a8emu_check.rs:41-56
    pub struct Rng(pub u64);
    impl Rng {
        pub fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    // npu_a8emu_check.rs:58-75
    pub fn f16_bits_to_f32(h: u16) -> f32 {
        let s = ((h >> 15) as u32) << 31;
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as u32;
        let bits = if e == 0 {
            if m == 0 {
                s
            } else {
                let v = m as f32 * (1.0 / 16_777_216.0);
                return if s != 0 { -v } else { v };
            }
        } else if e == 31 {
            s | 0x7f80_0000 | (m << 13)
        } else {
            s | (((e - 15 + 127) as u32) << 23) | (m << 13)
        };
        f32::from_bits(bits)
    }

    // npu_a8emu_check.rs:77-110
    pub fn f32_to_f16_rne(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x007f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 31 {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = man | 0x0080_0000;
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
        let rem = man & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1;
        }
        sign | h as u16
    }

    // npu_a8emu_check.rs:127-136
    pub fn pack_group(codes: &[u8; 256], payload: &mut [u8]) {
        for i in (0..256).step_by(4) {
            let bo = (i / 4) * 3;
            let (q0, q1, q2, q3) = (
                codes[i] & 63,
                codes[i + 1] & 63,
                codes[i + 2] & 63,
                codes[i + 3] & 63,
            );
            payload[bo] = q0 | (q1 << 6);
            payload[bo + 1] = (q1 >> 2) | (q2 << 4);
            payload[bo + 2] = (q2 >> 4) | (q3 << 2);
        }
    }

    // npu_a8emu_check.rs:138-147
    pub fn unpack_group(payload: &[u8], q: &mut [u8; 256]) {
        for c in 0..64 {
            let (b0, b1, b2) = (payload[c * 3], payload[c * 3 + 1], payload[c * 3 + 2]);
            q[c * 4] = b0 & 63;
            q[c * 4 + 1] = (b0 >> 6) | ((b1 & 0x0f) << 2);
            q[c * 4 + 2] = (b1 >> 4) | ((b2 & 0x03) << 4);
            q[c * 4 + 3] = b2 >> 2;
        }
    }

    /// Independent second decoder (not in the lab file): 6-bit little-endian bit-stream fields.
    pub fn unpack_group_bitstream(payload: &[u8], q: &mut [u8; 256]) {
        for (j, out) in q.iter_mut().enumerate() {
            let bit = j * 6;
            let byte = bit / 8;
            let lo = payload[byte] as u16;
            let hi = payload.get(byte + 1).copied().unwrap_or(0) as u16;
            *out = ((((hi << 8) | lo) >> (bit % 8)) & 63) as u8;
        }
    }

    // npu_a8emu_check.rs:175-196 (decode), extended with the bit-stream cross-check.
    pub struct DecodedWeights {
        pub codes: Vec<u8>,
        pub s: Vec<f32>,
        pub z: Vec<f32>,
    }

    pub fn decode_mq6(blob: &[u8], m: usize, k: usize) -> DecodedWeights {
        assert_eq!(blob.len(), m * (k / 256) * GROUP_BYTES);
        let epochs = k / 128;
        let mut out = DecodedWeights {
            codes: vec![0; m * k],
            s: vec![0.0; m * epochs],
            z: vec![0.0; m * epochs],
        };
        let mut q = [0u8; 256];
        let mut q2 = [0u8; 256];
        for (gi, g) in blob.chunks_exact(GROUP_BYTES).enumerate() {
            let h = |o: usize| f16_bits_to_f32(u16::from_le_bytes([g[o], g[o + 1]]));
            unpack_group(&g[8..], &mut q);
            unpack_group_bitstream(&g[8..], &mut q2);
            assert_eq!(q[..], q2[..], "oracle decoders disagree in group {gi}");
            out.codes[gi * 256..(gi + 1) * 256].copy_from_slice(&q);
            out.s[gi * 2] = h(0);
            out.z[gi * 2] = h(2);
            out.s[gi * 2 + 1] = h(4);
            out.z[gi * 2 + 1] = h(6);
        }
        out
    }

    // npu_a8emu_check.rs:198-200
    fn clamp_code(q: f32, qmax: i32) -> i32 {
        (q.round_ties_even() as i32).clamp(-qmax, qmax)
    }

    // npu_a8emu_check.rs:204-217 (weights half of cpu_w8a8)
    pub fn wprep_w8(w: &DecodedWeights, m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
        let epochs = k / 128;
        let mut w8 = vec![0i8; m * k];
        let mut ws = vec![0f32; m];
        for n in 0..m {
            let wf = |i: usize| {
                (w.codes[n * k + i] as f32) * w.s[n * epochs + i / 128] + w.z[n * epochs + i / 128]
            };
            let absmax = (0..k).map(|i| wf(i).abs()).fold(0.0f32, f32::max);
            let wsn = (absmax as f64 / 127.0) as f32;
            ws[n] = wsn;
            if wsn != 0.0 {
                for i in 0..k {
                    w8[n * k + i] = clamp_code(wf(i) / wsn, 127) as i8;
                }
            }
        }
        (w8, ws)
    }
}

use oracle::*;

// ---------------------------------------------------------------------------------------------
// Generators.
// ---------------------------------------------------------------------------------------------

/// One 256-weight group: f16 header bits `[s0, z0, s1, z1]` and 256 codes.
#[derive(Clone)]
struct Group {
    hdr: [u16; 4],
    codes: [u8; 256],
}

fn blob_of(groups: &[Group]) -> Vec<u8> {
    let mut blob = vec![0u8; groups.len() * GROUP_BYTES];
    for (g, out) in groups.iter().zip(blob.chunks_exact_mut(GROUP_BYTES)) {
        for (i, h) in g.hdr.iter().enumerate() {
            out[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
        }
        pack_group(&g.codes, &mut out[8..]);
    }
    blob
}

fn finite_f16(rng: &mut Rng) -> u16 {
    loop {
        let h = (rng.next() >> 20) as u16;
        if (h >> 10) & 0x1f != 31 {
            return h;
        }
    }
}

fn f16(x: f32) -> u16 {
    let h = f32_to_f16_rne(x);
    assert_eq!(f16_bits_to_f32(h), x, "{x} is not f16-exact");
    h
}

fn random_codes(rng: &mut Rng) -> [u8; 256] {
    let mut c = [0u8; 256];
    for v in c.iter_mut() {
        *v = (rng.next() % 64) as u8;
    }
    c
}

/// Every one of the 64 codes appears, rotated per group so positions differ.
fn all_codes(rot: usize) -> [u8; 256] {
    let mut c = [0u8; 256];
    for (j, v) in c.iter_mut().enumerate() {
        *v = ((j + rot) % 64) as u8;
    }
    c
}

/// Fast random blob: every payload byte pattern is a valid MQ6 payload; headers are finite f16 of any
/// magnitude and sign (including subnormals and zeros).
fn random_blob_fast(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let mut blob = vec![0u8; m * (k / 256) * GROUP_BYTES];
    for g in blob.chunks_exact_mut(GROUP_BYTES) {
        for i in 0..4 {
            g[2 * i..2 * i + 2].copy_from_slice(&finite_f16(rng).to_le_bytes());
        }
        for chunk in g[8..].chunks_exact_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    blob
}

/// Lab-style mixed magnitudes (npu_a8emu_check.rs `synth_mq6` headers): four decades, offset z.
fn realistic_group(rng: &mut Rng, rot: usize) -> Group {
    let mut hdr = [0u16; 4];
    for h in 0..2 {
        let sc = 10f32.powf(-3.5 + 2.0 * rng.unit()) * (1.0 + rng.unit());
        let sc16 = f32_to_f16_rne(sc);
        let sc_back = f16_bits_to_f32(sc16);
        let z16 = f32_to_f16_rne(-sc_back * (16.0 + 32.0 * rng.unit()));
        hdr[2 * h] = sc16;
        hdr[2 * h + 1] = z16;
    }
    let codes = if rot == usize::MAX {
        random_codes(rng)
    } else {
        all_codes(rot)
    };
    Group { hdr, codes }
}

// ---------------------------------------------------------------------------------------------
// Checks.
// ---------------------------------------------------------------------------------------------

fn check_against_oracle(label: &str, blob: &[u8], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let dec = decode_mq6(blob, m, k);
    let (w8_ref, ws_ref) = wprep_w8(&dec, m, k);
    let (w8, ws) = pack_matrix(blob, m, k);
    assert_eq!(ws.len(), m, "{label}: ws length");
    assert_eq!(w8.len(), m * k, "{label}: w8 length");
    for n in 0..m {
        assert_eq!(
            ws[n].to_bits(),
            ws_ref[n].to_bits(),
            "{label}: ws[{n}] = {:e} (bits {:08x}), oracle {:e} (bits {:08x})",
            ws[n],
            ws[n].to_bits(),
            ws_ref[n],
            ws_ref[n].to_bits()
        );
        let (a, b) = (&w8[n * k..(n + 1) * k], &w8_ref[n * k..(n + 1) * k]);
        if a != b {
            let i = (0..k).find(|&i| a[i] != b[i]).unwrap();
            panic!(
                "{label}: w8[{n}][{i}] = {}, oracle {} (ws {:e})",
                a[i], b[i], ws_ref[n]
            );
        }
        assert!(
            a.iter().all(|&v| v != i8::MIN),
            "{label}: -128 code in row {n}"
        );
        if ws[n] == 0.0 {
            assert!(
                a.iter().all(|&v| v == 0),
                "{label}: ws==0 row {n} must have zero codes"
            );
        }
    }
    (w8, ws)
}

// ---------------------------------------------------------------------------------------------
// pack_matrix tests.
// ---------------------------------------------------------------------------------------------

#[test]
fn oracle_self_check_roundtrips_generated_codes() {
    let mut rng = Rng::new(0x5eed);
    let groups: Vec<Group> = (0..16)
        .map(|i| realistic_group(&mut rng, if i % 2 == 0 { i } else { usize::MAX }))
        .collect();
    let blob = blob_of(&groups);
    let dec = decode_mq6(&blob, 4, 1024);
    for (gi, g) in groups.iter().enumerate() {
        assert_eq!(
            dec.codes[gi * 256..(gi + 1) * 256],
            g.codes[..],
            "group {gi}"
        );
        assert_eq!(dec.s[gi * 2].to_bits(), f16_bits_to_f32(g.hdr[0]).to_bits());
        assert_eq!(
            dec.z[gi * 2 + 1].to_bits(),
            f16_bits_to_f32(g.hdr[3]).to_bits()
        );
    }
}

#[test]
fn random_finite_headers_all_f16_magnitudes() {
    for (seed, (m, k)) in [
        (1u64, (48usize, TRUNK_K)),
        (2, (17, 256)),
        (3, (9, 768)),
        (4, (1, TRUNK_K)),
    ] {
        let mut rng = Rng::new(0xa11c0de ^ seed);
        let blob = random_blob_fast(&mut rng, m, k);
        check_against_oracle(
            &format!("random_finite seed={seed} m={m} k={k}"),
            &blob,
            m,
            k,
        );
    }
}

#[test]
fn realistic_mixed_magnitudes_all_codes_reachable() {
    let (m, k) = (64usize, TRUNK_K);
    let mut rng = Rng::new(0x6d71);
    let groups: Vec<Group> = (0..m * k / 256)
        .map(|gi| realistic_group(&mut rng, if gi % 3 == 0 { gi % 64 } else { usize::MAX }))
        .collect();
    let blob = blob_of(&groups);
    let dec = decode_mq6(&blob, m, k);
    let mut seen = [false; 64];
    dec.codes.iter().for_each(|&c| seen[c as usize] = true);
    assert!(seen.iter().all(|&s| s), "all 64 codes must be present");
    let (_, ws) = check_against_oracle("realistic", &blob, m, k);
    assert!(ws.iter().all(|w| w.is_finite() && *w > 0.0));
}

#[test]
fn every_code_in_every_position_and_half() {
    // Group g holds code (j + g) % 64 at position j: every code at every position class and in both halves.
    let k = TRUNK_K;
    let mut rng = Rng::new(0xc0de5);
    let groups: Vec<Group> = (0..64 * (k / 256))
        .map(|gi| {
            let mut g = realistic_group(&mut rng, gi);
            g.hdr = [f16(0.015625), f16(-0.5), f16(0.03125), f16(-1.0)];
            g
        })
        .collect();
    check_against_oracle("every_code", &blob_of(&groups), 64, k);
}

#[test]
fn zero_rows_and_degenerate_scales() {
    let k = TRUNK_K;
    let ngroups = k / 256;
    let mut rng = Rng::new(0x2e50);
    let mut groups = Vec::new();
    let mk_row = |rng: &mut Rng, hdr: [u16; 4], random: bool| -> Vec<Group> {
        (0..ngroups)
            .map(|i| Group {
                hdr,
                codes: if random {
                    random_codes(rng)
                } else {
                    all_codes(i)
                },
            })
            .collect()
    };
    // 0: +0 scales/zeros, random codes. 1: -0 headers. 2: s=0 but z nonzero (constant row).
    // 3: z=0, s nonzero, all codes 0. 4: subnormal-only header. 5: s=0,z=-subnormal. 6: normal row.
    groups.extend(mk_row(&mut rng, [0, 0, 0, 0], true));
    groups.extend(mk_row(&mut rng, [0x8000, 0x8000, 0x8000, 0x8000], true));
    groups.extend(mk_row(&mut rng, [0, f16(0.75), 0, f16(-0.75)], true));
    groups.extend((0..ngroups).map(|_| Group {
        hdr: [f16(1.5), 0, f16(-2.0), 0],
        codes: [0; 256],
    }));
    groups.extend(mk_row(&mut rng, [0x0001, 0x0001, 0x8001, 0x0003], true));
    groups.extend(mk_row(&mut rng, [0, 0x8001, 0, 0x8001], false));
    groups.extend(mk_row(
        &mut rng,
        [f16(0.125), f16(-1.0), f16(0.25), f16(-2.0)],
        true,
    ));
    // Zero row between two nonzero rows with all-zero scale on one half only.
    groups.extend(mk_row(&mut rng, [0, 0, f16(1.0), f16(-1.0)], true));
    let m = groups.len() / ngroups;
    let (w8, ws) = check_against_oracle("zero_rows", &blob_of(&groups), m, k);
    assert_eq!(ws[0], 0.0);
    assert_eq!(ws[1], 0.0);
    assert!(w8[..2 * k].iter().all(|&v| v == 0));
    assert_eq!(ws[3], 0.0, "s != 0 but all codes 0 and z = 0 is a zero row");
    assert!(ws[6] > 0.0);
}

#[test]
fn huge_outlier_rows() {
    let k = TRUNK_K;
    let ngroups = k / 256;
    let tiny = [f16(2f32.powi(-14)), 0, f16(2f32.powi(-14)), 0];
    let mut groups = Vec::new();
    // Rows: (group, position, positive?) of one outlier amid tiny weights.
    for &(og, pos, neg) in &[
        (0usize, 0usize, false),
        (3, 127, true),
        (4, 128, false),
        (ngroups - 1, 255, true),
        (5, 77, false),
    ] {
        for gi in 0..ngroups {
            let mut g = Group {
                hdr: tiny,
                codes: random_codes(&mut Rng::new(0x1234 + gi as u64 + pos as u64)),
            };
            if gi == og {
                let half = pos / 128;
                let big = if neg { 0xfbff } else { 0x7bff }; // +-65504 = f16 max finite
                g.hdr[2 * half] = big;
                g.hdr[2 * half + 1] = 0;
                // The outlier's half is otherwise all zero (w = 0 * s + 0), so it is a single outlier.
                g.codes[half * 128..(half + 1) * 128].fill(0);
                g.codes[pos] = 63;
            }
            groups.push(g);
        }
    }
    // Largest possible |w| = 63*65504 + 65504 with the offset sharing the sign: same-sign outlier z.
    for gi in 0..ngroups {
        let mut g = Group {
            hdr: tiny,
            codes: random_codes(&mut Rng::new(0x99 + gi as u64)),
        };
        if gi == 2 {
            g.hdr[0] = 0x7bff;
            g.hdr[1] = 0x7bff;
            g.codes[10] = 63;
        }
        groups.push(g);
    }
    let m = groups.len() / ngroups;
    let (w8, ws) = check_against_oracle("outlier", &blob_of(&groups), m, k);
    // The outlier saturates at +-127.
    for n in 0..m {
        let row = &w8[n * k..(n + 1) * k];
        assert_eq!(
            row.iter().map(|&v| (v as i32).abs()).max(),
            Some(127),
            "row {n}"
        );
        assert!(ws[n] > 100.0);
    }
}

#[test]
fn exact_round_ties_even_rows() {
    // Half 1 holds |w| = 127 * 2^e exactly (s = 2^(e+1), z = 2^e, q = 63), so ws = 2^e exactly and half 0
    // holds w = (q + 0.5) * 2^e, i.e. w / ws = q + 0.5 is an exact tie for every code.
    let k = TRUNK_K;
    let ngroups = k / 256;
    let mut groups = Vec::new();
    let mut rows = Vec::new(); // (e, negative)
    for e in -8i32..=4 {
        for &neg in &[false, true] {
            let u = 2f32.powi(e);
            let sg = if neg { -1.0 } else { 1.0 };
            for gi in 0..ngroups {
                // Same header in every group: half 0 w = (q + 0.5) u, half 1 w = (2q + 1) u, |w|max = 127 u.
                let hdr = [
                    f16(sg * u),
                    f16(sg * 0.5 * u),
                    f16(sg * 2.0 * u),
                    f16(sg * u),
                ];
                let mut codes = all_codes(gi);
                if gi == 0 {
                    codes[128 + 5] = 63;
                }
                groups.push(Group { hdr, codes });
            }
            rows.push((e, neg));
        }
    }
    let m = rows.len();
    let blob = blob_of(&groups);
    let (w8, ws) = check_against_oracle("ties", &blob, m, k);
    for (n, &(e, neg)) in rows.iter().enumerate() {
        assert_eq!(ws[n], 2f32.powi(e), "row {n}: ws must be exactly 2^{e}");
        let row = &w8[n * k..(n + 1) * k];
        for q in 0..64usize {
            // Group 0, half 0, position j has code (j % 64): j = q.
            let expect = (q as f32 + 0.5).round_ties_even() as i8;
            assert_eq!(
                row[q],
                if neg { -expect } else { expect },
                "row {n} (e={e}, neg={neg}) code {q}: tie to even"
            );
        }
        assert_eq!(
            &row[0..4],
            if neg { &[0, -2, -2, -4] } else { &[0, 2, 2, 4] },
            "row {n}"
        );
        assert_eq!(row[128 + 5], if neg { -127 } else { 127 });
    }
}

#[test]
fn edge_shapes_and_invalid_arguments() {
    let (w8, ws) = pack_matrix(&[], 0, TRUNK_K);
    assert!(w8.is_empty() && ws.is_empty(), "m=0 is permitted");

    let mut rng = Rng::new(7);
    let good = random_blob_fast(&mut rng, 2, 256);
    let short = good[..good.len() - 1].to_vec();
    assert!(
        std::panic::catch_unwind(|| pack_matrix(&short, 2, 256)).is_err(),
        "short blob must panic"
    );
    assert!(
        std::panic::catch_unwind(|| pack_matrix(&good, 2, 255)).is_err(),
        "k not a multiple of 256 must panic"
    );
}

// ---------------------------------------------------------------------------------------------
// pack_shape / pack_trunk.
// ---------------------------------------------------------------------------------------------

struct ShapeData {
    blobs: Vec<(Family, usize, Vec<u8>)>, // fused order
}

fn shape_data(shape: Shape, seed: u64) -> ShapeData {
    let mut rng = Rng::new(seed);
    let blobs = shape
        .families()
        .iter()
        .map(|&f| {
            let m = f.cols();
            (f, m, random_blob_fast(&mut rng, m, TRUNK_K))
        })
        .collect();
    ShapeData { blobs }
}

fn refs(d: &ShapeData) -> Vec<MatrixRef<'_>> {
    d.blobs
        .iter()
        .map(|(f, m, b)| MatrixRef {
            family: *f,
            mq6: b,
            m: *m,
            k: TRUNK_K,
        })
        .collect()
}

fn parts_of(map: &hipfire_arch_qwen4::npu_trunk::ColumnMap) -> Vec<(u32, u32)> {
    map.parts.iter().map(|&(_, c0, n)| (c0, n)).collect()
}

/// Oracle W8 row scale rounded up to the direct-BF16 grid15 scale `Sm * 2^Se` (what the packer stores in `ws`).
fn grid_ws(raw: f32) -> f32 {
    let (sm, se) = peacemaker_npu::kernels::w8a8_epi::grid15_up(raw);
    (f64::from(sm) * 2f64.powi(i32::from(se))) as f32
}

fn expected_ws(blobs: &[(Family, usize, Vec<u8>)]) -> Vec<Vec<f32>> {
    blobs
        .iter()
        .map(|(_, m, b)| {
            wprep_w8(&decode_mq6(b, *m, TRUNK_K), *m, TRUNK_K)
                .1
                .into_iter()
                .map(grid_ws)
                .collect()
        })
        .collect()
}

#[test]
fn pack_shape_full_mask_matches_oracle_ws_and_structure() {
    for (i, shape) in Shape::ALL.into_iter().enumerate() {
        let d = shape_data(shape, 0xface + i as u64);
        let mats = refs(&d);
        let mask = FamilyMask::ALL;
        let packed = match pack_shape(shape, &mats, mask) {
            Ok(p) => p,
            Err(e) => panic!("{shape:?}: {e}"),
        };
        let map = shape.column_map(mask);
        assert_eq!(parts_of(&packed.map), parts_of(&map));
        let useful: usize = d.blobs.iter().map(|(_, m, _)| *m).sum();
        assert_eq!(map.useful_cols(), useful);
        assert!(map.padded_cols() >= useful && map.padded_cols() % PANEL_COLS == 0);
        assert_eq!(packed.panels.len(), map.panel_count());
        assert_eq!(packed.panels.len() * PANEL_COLS, map.padded_cols());
        assert!(packed
            .panels
            .iter()
            .all(|p| p.len() == packed.panels[0].len() && !p.is_empty()));
        assert_eq!(
            packed.ws.len(),
            map.padded_cols(),
            "ws is padded to the panel multiple"
        );
        let mut col = 0;
        for (ws_ref, (_, m, _)) in expected_ws(&d.blobs).iter().zip(&d.blobs) {
            for n in 0..*m {
                assert_eq!(
                    packed.ws[col + n].to_bits(),
                    ws_ref[n].to_bits(),
                    "{shape:?} fused col {}",
                    col + n
                );
            }
            col += m;
        }
        assert!(
            packed.ws[useful..].iter().all(|&w| w == 0.0),
            "{shape:?}: padded ws must be zero"
        );
        // locate() agrees with the family boundaries.
        let mut c0 = 0;
        for &(f, m, _) in &d.blobs {
            assert_eq!(map.locate(c0).map(|(_, r)| r), Some(0));
            assert_eq!(map.locate(c0 + m - 1).map(|(_, r)| r), Some(m - 1));
            let _ = f;
            c0 += m;
        }
        assert!(
            map.locate(map.useful_cols()).is_none(),
            "padding columns belong to no projection"
        );
        // Deterministic.
        let again = pack_shape(shape, &mats, mask).ok().expect("repack");
        assert!(
            again.panels == packed.panels && again.ws == packed.ws,
            "{shape:?}: nondeterministic"
        );
        // The explicit contract is the default; scales are the grid15-up decomposition of `ws`.
        let direct =
            pack_shape_with_contract(EpilogueContract::Bf16Direct, shape, &mats, mask).unwrap();
        assert!(direct.panels == packed.panels && direct.ws == packed.ws);
        assert_eq!(direct.scales.len(), map.padded_cols());
        for (&rounded, &(sm, se)) in direct.ws.iter().zip(&direct.scales) {
            assert_eq!(rounded, (f64::from(sm) * 2f64.powi(i32::from(se))) as f32);
            assert_eq!(
                (sm, se),
                peacemaker_npu::kernels::w8a8_epi::grid15_up(rounded)
            );
        }
    }
}

#[test]
fn pack_shape_single_family_masks_and_validation() {
    for (i, shape) in Shape::ALL.into_iter().enumerate() {
        let d = shape_data(shape, 0xbeef + i as u64);
        let all = refs(&d);
        let ws_all = expected_ws(&d.blobs);
        for (fi, &(f, m, _)) in d.blobs.iter().enumerate() {
            let mask = FamilyMask::from(f);
            // Unselected entries may be omitted.
            let only = vec![MatrixRef {
                family: all[fi].family,
                mq6: all[fi].mq6,
                m: all[fi].m,
                k: all[fi].k,
            }];
            let packed = match pack_shape(shape, &only, mask) {
                Ok(p) => p,
                Err(e) => panic!("{shape:?} family #{fi}: {e}"),
            };
            let map = shape.column_map(mask);
            assert_eq!(parts_of(&packed.map), parts_of(&map));
            assert_eq!(packed.panels.len(), map.panel_count());
            assert_eq!(packed.ws.len(), map.padded_cols());
            // The selected columns' ws sit at the column map's own offsets.
            let (c0, n) = map
                .parts
                .iter()
                .map(|&(_, c, n)| (c as usize, n as usize))
                .next()
                .unwrap();
            assert_eq!(n, m);
            for r in 0..m {
                assert_eq!(
                    packed.ws[c0 + r].to_bits(),
                    ws_all[fi][r].to_bits(),
                    "{shape:?} fam #{fi} row {r}"
                );
            }
            // Full set given but only one selected: also accepted, same result.
            let with_all = pack_shape(shape, &all, mask).ok().expect("superset refs");
            assert!(with_all.panels == packed.panels && with_all.ws == packed.ws);
            // Selected family missing => error.
            assert!(
                pack_shape(shape, &[], mask).is_err(),
                "{shape:?} #{fi}: missing selected family"
            );
        }
        // Zero selected columns => zero panels.
        let none = pack_shape(shape, &[], FamilyMask::NONE)
            .ok()
            .expect("empty mask");
        assert!(none.panels.is_empty());

        // Validation of the full mask.
        let mask = FamilyMask::ALL;
        let mut dup = refs(&d);
        dup.push(MatrixRef {
            family: all[0].family,
            mq6: all[0].mq6,
            m: all[0].m,
            k: all[0].k,
        });
        assert!(
            pack_shape(shape, &dup, mask).is_err(),
            "{shape:?}: duplicate family"
        );
        let mut swapped = refs(&d);
        swapped.swap(0, 1);
        assert!(
            pack_shape(shape, &swapped, mask).is_err(),
            "{shape:?}: out-of-fused-order"
        );
        let mut missing = refs(&d);
        missing.pop();
        assert!(
            pack_shape(shape, &missing, mask).is_err(),
            "{shape:?}: missing selected family"
        );
        let mut bad_m = refs(&d);
        bad_m[0].m -= 1;
        bad_m[0].mq6 = &d.blobs[0].2[..d.blobs[0].2.len() - (TRUNK_K / 256) * GROUP_BYTES];
        assert!(
            pack_shape(shape, &bad_m, mask).is_err(),
            "{shape:?}: wrong family rows"
        );
        let mut bad_k = refs(&d);
        bad_k[0].k = 2048;
        assert!(
            pack_shape(shape, &bad_k, mask).is_err(),
            "{shape:?}: K must be 2560"
        );
        let mut short = refs(&d);
        short[0].mq6 = &d.blobs[0].2[..d.blobs[0].2.len() - 1];
        assert!(
            pack_shape(shape, &short, mask).is_err(),
            "{shape:?}: truncated blob"
        );
    }
}

#[test]
fn pack_trunk_layers_share_shape_map_and_ws() {
    let gdn = shape_data(Shape::GdnIn, 0x6d5);
    let qsa = shape_data(Shape::QsaIn, 0x95a);
    let layers = vec![
        LayerTrunkRefs {
            shape: Shape::GdnIn,
            matrices: refs(&gdn),
        },
        LayerTrunkRefs {
            shape: Shape::QsaIn,
            matrices: refs(&qsa),
        },
        LayerTrunkRefs {
            shape: Shape::GdnIn,
            matrices: refs(&gdn),
        },
    ];
    let mask = FamilyMask::ALL;
    let trunk = match pack_trunk(&layers, mask) {
        Ok(t) => t,
        Err(e) => panic!("pack_trunk: {e}"),
    };
    assert_eq!(trunk.per_layer.len(), 3);
    assert_eq!(trunk.ws.len(), 3);
    for (l, shape) in [(0, Shape::GdnIn), (1, Shape::QsaIn), (2, Shape::GdnIn)] {
        let (s, packed) = trunk.per_layer[l].as_ref().expect("layer selected by ALL");
        assert!(*s == shape);
        let direct = pack_shape(shape, &layers[l].matrices, mask)
            .ok()
            .expect("pack_shape");
        assert!(
            packed.panels == direct.panels && packed.ws == direct.ws,
            "layer {l}"
        );
        assert!(
            trunk.ws[l] == packed.ws,
            "layer {l}: PackedTrunk.ws mirrors the packed ws"
        );
        assert_eq!(
            parts_of(&trunk.map[shape.index()]),
            parts_of(&shape.column_map(mask))
        );
    }
    assert!(
        trunk.per_layer[0].as_ref().unwrap().1.panels
            == trunk.per_layer[2].as_ref().unwrap().1.panels
    );
    let direct = pack_trunk_with_contract(EpilogueContract::Bf16Direct, &layers, mask).unwrap();
    assert_eq!(direct.per_layer.len(), layers.len());
    assert_eq!(
        direct.per_layer[0].as_ref().unwrap().1.scales,
        direct.per_layer[2].as_ref().unwrap().1.scales
    );
    assert!(pack_trunk_with_contract(EpilogueContract::Bf16Direct, &layers, mask).is_ok());
}
