// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `peacemaker_cdna::float` against an independent reference.
//!
//! The fixture `tests/fixtures/fp8_reference.json` is produced by
//! `tools/fp8_reference.py` (exact `fractions.Fraction` arithmetic, numpy f16
//! cross-check, optional `ml_dtypes` cross-check). Regenerate it from the
//! repository root with
//!
//! ```text
//! python3 -m venv --system-site-packages /tmp/pmc-float-venv   # throwaway
//! /tmp/pmc-float-venv/bin/pip install ml_dtypes                 # optional
//! /tmp/pmc-float-venv/bin/python crates/peacemaker-cdna/tools/fp8_reference.py
//! ```
//!
//! (plain `python3 crates/peacemaker-cdna/tools/fp8_reference.py` works
//! without `ml_dtypes`, skipping that cross-check). No oracle is involved: every
//! test here always runs.

use peacemaker_cdna::arch::Fp8Encoding::{self, Fnuz, Ocp};
use peacemaker_cdna::float::{
    bf16_to_f32, f16_to_f32, f32_to_bf16_rne, f32_to_f16_rne, f32_to_f16_rtz, f32_to_f8_rne,
    f32_to_f8_sr, f8_to_f32, ocp_e4m3_to_fnuz_half, F8,
};

const FIXTURE: &str = include_str!("fixtures/fp8_reference.json");

/// Fixture column order (generator's `F8_FMTS`).
const FMTS: [(&str, Fp8Encoding, F8); 4] = [
    ("fnuz_e4m3", Fnuz, F8::Fp8),
    ("fnuz_e5m2", Fnuz, F8::Bf8),
    ("ocp_e4m3", Ocp, F8::Fp8),
    ("ocp_e5m2", Ocp, F8::Bf8),
];

/// The string array under `key` (the fixture holds only flat string arrays
/// and string scalars, written one value per quoted token).
fn strings(key: &str) -> Vec<&'static str> {
    let open = format!("\"{key}\": [");
    let start = FIXTURE
        .find(&open)
        .unwrap_or_else(|| panic!("fixture lacks {key}"))
        + open.len();
    let end = start + FIXTURE[start..].find(']').expect("unterminated array");
    FIXTURE[start..end].split('"').skip(1).step_by(2).collect()
}

fn hex32(s: &str) -> u32 {
    u32::from_str_radix(s, 16).unwrap()
}

/// `(x bits, [codes; 8])` rows of `f8_rne`: four formats with FP16_OVFL=0,
/// then the same four with FP16_OVFL=1.
fn f8_rows() -> Vec<(u32, [u8; 8])> {
    strings("f8_rne")
        .into_iter()
        .map(|row| {
            let (x, codes) = row.split_once(':').unwrap();
            let mut out = [0u8; 8];
            for (i, o) in out.iter_mut().enumerate() {
                *o = u8::from_str_radix(&codes[2 * i..2 * i + 2], 16).unwrap();
            }
            (hex32(x), out)
        })
        .collect()
}

#[test]
fn decode_all_codes_bit_exact() {
    for (name, enc, kind) in FMTS {
        let want = strings(&format!("decode_{name}"));
        assert_eq!(want.len(), 256, "{name}");
        for (code, w) in want.iter().enumerate() {
            let got = f8_to_f32(enc, kind, code as u8).to_bits();
            assert_eq!(got, hex32(w), "{name} code {code:#04x}");
        }
    }
}

#[test]
fn rne_sweep_bit_exact() {
    let rows = f8_rows();
    assert!(rows.len() > 2000, "sweep too small: {}", rows.len());
    for (x, want) in rows {
        for (slot, w) in want.iter().enumerate() {
            let (name, enc, kind) = FMTS[slot % 4];
            let ovfl = slot >= 4;
            let got = f32_to_f8_rne(enc, kind, f32::from_bits(x), ovfl);
            assert_eq!(
                got,
                *w,
                "{name} fp16_ovfl={ovfl} x={x:#010x} ({})",
                f32::from_bits(x)
            );
        }
    }
}

#[test]
fn f16_bf16_sweep_bit_exact() {
    let rows = strings("f16");
    assert!(rows.len() > 1000);
    for row in rows {
        let mut it = row.split(':');
        let x = f32::from_bits(hex32(it.next().unwrap()));
        let mut next16 = || u16::from_str_radix(it.next().unwrap(), 16).unwrap();
        let (rne, rtz, bf) = (next16(), next16(), next16());
        assert_eq!(f32_to_f16_rne(x), rne, "f16 rne {:#010x}", x.to_bits());
        assert_eq!(f32_to_f16_rtz(x), rtz, "f16 rtz {:#010x}", x.to_bits());
        assert_eq!(f32_to_bf16_rne(x), bf, "bf16 rne {:#010x}", x.to_bits());
    }
}

#[test]
fn every_finite_code_round_trips() {
    for (name, enc, kind) in FMTS {
        for code in 0..=255u8 {
            let v = f8_to_f32(enc, kind, code);
            if v.is_nan() || v.is_infinite() {
                continue;
            }
            for ovfl in [false, true] {
                assert_eq!(
                    f32_to_f8_rne(enc, kind, v, ovfl),
                    code,
                    "{name} {code:#04x}"
                );
                for rnd in [0, 0x8000_0000, u32::MAX] {
                    assert_eq!(
                        f32_to_f8_sr(enc, kind, v, rnd, ovfl),
                        code,
                        "{name} sr {code:#04x}"
                    );
                }
            }
        }
    }
}

#[test]
fn fnuz_signed_zero_policy() {
    for kind in [F8::Fp8, F8::Bf8] {
        for ovfl in [false, true] {
            assert_eq!(f32_to_f8_rne(Fnuz, kind, -0.0, ovfl), 0x00);
            assert_eq!(f32_to_f8_rne(Fnuz, kind, -1e-30, ovfl), 0x00);
            assert_eq!(f32_to_f8_sr(Fnuz, kind, -1e-30, u32::MAX, ovfl), 0x00);
            assert_eq!(f32_to_f8_rne(Ocp, kind, -0.0, ovfl), 0x80);
            assert_eq!(f32_to_f8_rne(Ocp, kind, -1e-30, ovfl), 0x80);
        }
    }
    assert!(f8_to_f32(Fnuz, F8::Fp8, 0x80).is_nan());
    assert!(f8_to_f32(Fnuz, F8::Bf8, 0x80).is_nan());
    assert_eq!(f8_to_f32(Ocp, F8::Fp8, 0x80).to_bits(), 0x8000_0000);
}

#[test]
fn fp16_ovfl_table() {
    // CDNA3 ISA §7.2 p.51.
    let inf = f32::INFINITY;
    for s in [1.0f32, -1.0] {
        let neg = if s < 0.0 { 0x80 } else { 0 };
        // FNUZ E4M3: max 240 (0x7f), no Inf.
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Fp8, s * inf, true), neg | 0x7f);
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Fp8, s * inf, false), 0x80);
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Fp8, s * 241.0, true), neg | 0x7f);
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Fp8, s * 241.0, false), 0x80);
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Fp8, s * 240.0, false), neg | 0x7f);
        // FNUZ E5M2: max 57344 (0x7f), Inf → NaN 0x80 without FP16_OVFL.
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Bf8, s * inf, true), neg | 0x7f);
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Bf8, s * inf, false), 0x80);
        assert_eq!(f32_to_f8_rne(Fnuz, F8::Bf8, s * 60000.0, false), 0x80);
        // OCP E4M3FN: max 448 (0x7e), NaN 0x7f.
        assert_eq!(f32_to_f8_rne(Ocp, F8::Fp8, s * inf, true), neg | 0x7e);
        assert_eq!(f32_to_f8_rne(Ocp, F8::Fp8, s * 449.0, false), neg | 0x7f);
        // OCP E5M2: max 57344 (0x7b), Inf 0x7c.
        assert_eq!(f32_to_f8_rne(Ocp, F8::Bf8, s * inf, false), neg | 0x7c);
        assert_eq!(f32_to_f8_rne(Ocp, F8::Bf8, s * 60000.0, true), neg | 0x7b);
        for (_, enc, kind) in FMTS {
            for ovfl in [false, true] {
                let nan =
                    f32::from_bits(f32::NAN.to_bits() | if s < 0.0 { 0x8000_0000 } else { 0 });
                let got = f32_to_f8_rne(enc, kind, nan, ovfl);
                assert!(
                    f8_to_f32(enc, kind, got).is_nan(),
                    "{enc:?} {kind:?} NaN → {got:#04x}"
                );
            }
        }
    }
}

/// Fold theorem, values: every finite OCP E4M3 byte is its own FNUZ image at
/// half the value (`0x80` → `0x00`).
#[test]
fn fold_theorem_decode() {
    for c in 0..=255u8 {
        let ocp = f8_to_f32(Ocp, F8::Fp8, c);
        match ocp_e4m3_to_fnuz_half(c) {
            None => assert!(ocp.is_nan(), "{c:#04x}"),
            Some(h) => {
                assert!(!ocp.is_nan());
                assert_eq!(
                    2.0 * f8_to_f32(Fnuz, F8::Fp8, h),
                    ocp,
                    "{c:#04x} → {h:#04x}"
                );
                assert_eq!(h, if c == 0x80 { 0x00 } else { c });
            }
        }
    }
}

/// Fold theorem, rounding: on [-448, 448] RNE onto the FNUZ grid of `x/2`
/// is the OCP RNE of `x`, mapped through `ocp_e4m3_to_fnuz_half`.
#[test]
fn fold_theorem_rne_commutes_with_halving() {
    let check = |x: f32| {
        for ovfl in [false, true] {
            let ocp = f32_to_f8_rne(Ocp, F8::Fp8, x, ovfl);
            let image = ocp_e4m3_to_fnuz_half(ocp).expect("in-range OCP RNE is finite");
            assert_eq!(
                f32_to_f8_rne(Fnuz, F8::Fp8, x * 0.5, ovfl),
                image,
                "x={x} ({:#010x})",
                x.to_bits()
            );
        }
    };
    let mut n = 0usize;
    for (x, _) in f8_rows() {
        let x = f32::from_bits(x);
        if x.is_finite() && x.abs() <= 448.0 {
            check(x);
            n += 1;
        }
    }
    assert!(n > 1000);
    // Plus a strided sweep of every f32 bit pattern in [0, 448], both signs.
    let top = 448.0f32.to_bits();
    let mut b = 0u32;
    while b <= top {
        check(f32::from_bits(b));
        check(-f32::from_bits(b));
        b += 509;
    }
    check(448.0);
    check(-448.0);
}

#[test]
fn f16_round_trip_and_rounding_structure() {
    for h in 0..=u16::MAX {
        let v = f16_to_f32(h);
        if v.is_nan() {
            // Quieted, payload kept.
            assert_eq!(f32_to_f16_rne(v), h | 0x0200, "{h:#06x}");
            assert_eq!(f32_to_f16_rtz(v), h | 0x0200, "{h:#06x}");
            continue;
        }
        assert_eq!(f32_to_f16_rne(v), h, "rne {h:#06x}");
        assert_eq!(f32_to_f16_rtz(v), h, "rtz {h:#06x}");
    }
    // Adjacent positive finite pairs: midpoint ties to even, ±1 f32 ulp
    // around it picks the nearer neighbour, RTZ truncates.
    for h in 0..0x7bffu16 {
        let (a, b) = (f16_to_f32(h), f16_to_f32(h + 1));
        let mid = (a + b) * 0.5; // exact: f16 has 11 significant bits
        assert_eq!(f64::from(mid), (f64::from(a) + f64::from(b)) / 2.0);
        let even = if h & 1 == 0 { h } else { h + 1 };
        for s in [0u16, 0x8000] {
            let sg = |v: f32| if s != 0 { -v } else { v };
            assert_eq!(f32_to_f16_rne(sg(mid)), s | even, "{h:#06x}");
            assert_eq!(f32_to_f16_rne(sg(f32::from_bits(mid.to_bits() - 1))), s | h);
            assert_eq!(
                f32_to_f16_rne(sg(f32::from_bits(mid.to_bits() + 1))),
                s | (h + 1)
            );
            assert_eq!(f32_to_f16_rtz(sg(mid)), s | h);
            assert_eq!(f32_to_f16_rtz(sg(f32::from_bits(b.to_bits() - 1))), s | h);
        }
    }
    // Overflow: 65520 is the tie between 65504 and 2^16.
    assert_eq!(f32_to_f16_rne(65520.0), 0x7c00);
    assert_eq!(
        f32_to_f16_rne(f32::from_bits(65520.0f32.to_bits() - 1)),
        0x7bff
    );
    assert_eq!(f32_to_f16_rtz(1e30), 0x7bff);
    assert_eq!(f32_to_f16_rtz(-f32::INFINITY), 0xfc00);
    assert_eq!(f32_to_f16_rne(-0.0), 0x8000);
    // Below half the smallest subnormal → ±0; just above → smallest subnormal.
    assert_eq!(f32_to_f16_rne(2f32.powi(-25)), 0x0000);
    assert_eq!(
        f32_to_f16_rne(f32::from_bits(2f32.powi(-25).to_bits() + 1)),
        0x0001
    );
}

#[test]
fn bf16_round_trip_and_rounding_structure() {
    for h in 0..=u16::MAX {
        let v = bf16_to_f32(h);
        if v.is_nan() {
            assert_eq!(f32_to_bf16_rne(v), h | 0x0040, "{h:#06x}");
            continue;
        }
        assert_eq!(f32_to_bf16_rne(v), h, "{h:#06x}");
    }
    for h in 0..0x7f7fu16 {
        let lo = u32::from(h) << 16;
        let mid = lo | 0x8000; // exactly halfway to h + 1
        let even = if h & 1 == 0 { h } else { h + 1 };
        for s in [0u32, 0x8000_0000] {
            let s16 = (s >> 16) as u16;
            assert_eq!(f32_to_bf16_rne(f32::from_bits(s | mid)), s16 | even);
            assert_eq!(f32_to_bf16_rne(f32::from_bits(s | (mid - 1))), s16 | h);
            assert_eq!(
                f32_to_bf16_rne(f32::from_bits(s | (mid + 1))),
                s16 | (h + 1)
            );
        }
    }
    // Largest finite + half ulp ties to even → Inf.
    assert_eq!(f32_to_bf16_rne(f32::from_bits(0x7f7f_8000)), 0x7f80);
    assert_eq!(f32_to_bf16_rne(f32::INFINITY), 0x7f80);
}

/// Stochastic rounding under the documented bit-use model
/// (`float::f32_to_f8_sr`, `[UNVERIFIED: confirm on MI300X]`): the
/// probability of rounding up is exactly the dropped fraction for normal
/// results, rnd = 0 truncates, and the FP16_OVFL table matches RNE's.
#[test]
fn stochastic_rounding_model() {
    // FP8: 20 random bits rnd[31:12]; BF8: 21 bits rnd[31:11].
    for (enc, kind, bits) in [
        (Fnuz, F8::Fp8, 20u32),
        (Fnuz, F8::Bf8, 21),
        (Ocp, F8::Fp8, 20),
        (Ocp, F8::Bf8, 21),
    ] {
        let lo_code = 0x38u8; // a normal code in every format
        let lo = f8_to_f32(enc, kind, lo_code);
        let hi = f8_to_f32(enc, kind, lo_code + 1);
        let gap = hi.to_bits() - lo.to_bits();
        assert_eq!(gap, 1 << bits);
        for d in [1u32, 3, gap / 4, gap / 2, gap - 1] {
            let x = f32::from_bits(lo.to_bits() + d);
            assert_eq!(f32_to_f8_sr(enc, kind, x, 0, false), lo_code);
            assert_eq!(f32_to_f8_sr(enc, kind, x, u32::MAX, false), lo_code + 1);
            let up = (0..gap)
                .filter(|r| f32_to_f8_sr(enc, kind, x, r << (32 - bits), false) == lo_code + 1)
                .count();
            assert_eq!(up as u32, d, "{enc:?} {kind:?} d={d}");
            // Bits below the used field are ignored.
            assert_eq!(
                f32_to_f8_sr(enc, kind, x, (1 << (32 - bits)) - 1, false),
                lo_code
            );
            let neg = f32_to_f8_sr(enc, kind, -x, u32::MAX, false);
            assert_eq!(neg, 0x80 | (lo_code + 1));
        }
    }
    // max plus any random value truncates back to max; the source-magnitude
    // overflow test (|x| > max) applies before the add; NaN / Inf / > max as RNE.
    let just_above_max = f32::from_bits(240.0f32.to_bits() + 1);
    for ovfl in [false, true] {
        assert_eq!(f32_to_f8_sr(Fnuz, F8::Fp8, 240.0, u32::MAX, ovfl), 0x7f);
        assert_eq!(f32_to_f8_sr(Fnuz, F8::Fp8, -239.99, u32::MAX, ovfl), 0xff);
        assert_eq!(f32_to_f8_sr(Fnuz, F8::Bf8, 57344.0, u32::MAX, ovfl), 0x7f);
        assert_eq!(f32_to_f8_sr(Ocp, F8::Fp8, 448.0, u32::MAX, ovfl), 0x7e);
        assert_eq!(f32_to_f8_sr(Ocp, F8::Fp8, 440.0, u32::MAX, ovfl), 0x7e);
        assert_eq!(f32_to_f8_sr(Ocp, F8::Bf8, 57344.0, u32::MAX, ovfl), 0x7b);
    }
    assert_eq!(f32_to_f8_sr(Fnuz, F8::Fp8, just_above_max, 0, false), 0x80);
    assert_eq!(f32_to_f8_sr(Fnuz, F8::Fp8, just_above_max, 0, true), 0x7f);
    for (_, enc, kind) in FMTS {
        for ovfl in [false, true] {
            for x in [f32::NAN, f32::INFINITY, -f32::INFINITY, 1e9, -1e9] {
                for rnd in [0, 0x1234_5678, u32::MAX] {
                    assert_eq!(
                        f32_to_f8_sr(enc, kind, x, rnd, ovfl),
                        f32_to_f8_rne(enc, kind, x, ovfl)
                    );
                }
            }
        }
    }
}
