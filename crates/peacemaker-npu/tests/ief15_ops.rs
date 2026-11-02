// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! IEF15 N0 simulator ops (`local://ief15-spec.md` §4): every new instruction is assembled with the peacemaker-npu
//! encoder (`gen::ENCODINGS` + `bundle::encode_slot`/`pack_slots`), run on one `Tile` and compared with an
//! independent lane model written here (i128 / Vec arithmetic, not the simulator's helpers).
use peacemaker_npu::isa::{
    self, bundle,
    gen::{self, Slot},
};
use peacemaker_npu::sim::{decode::PreparedDecoder, Error, Step, Tile};

/// 16-bit nops after every instruction: longer than any itinerary latency used here (<= 7).
const GAP: usize = 8;
const MEM: u32 = 0x70000;

fn encode(name: &str, ops: &[(&str, &str)]) -> (Slot, u64) {
    let e = gen::ENCODINGS
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("no encoding {name}"));
    let raw: Vec<(&str, u64)> = ops
        .iter()
        .map(|&(field, v)| {
            let value = v.parse::<u64>().unwrap_or_else(|_| {
                let operand = e
                    .operands
                    .iter()
                    .find(|o| o.name == field)
                    .unwrap_or_else(|| panic!("{name}: no operand {field}"));
                operand
                    .registers
                    .iter()
                    .find(|r| r.name == v)
                    .unwrap_or_else(|| panic!("{name}.{field}: no register {v}"))
                    .value
            });
            let width = e
                .fields
                .iter()
                .filter(|f| f.name == field)
                .map(|f| f.source_lsb + f.width)
                .max()
                .unwrap_or_else(|| panic!("{name}: no field {field}"));
            (field, value & ((1u64 << width) - 1))
        })
        .collect();
    (e.slot, bundle::encode_slot(e, &raw, 0).unwrap())
}
struct Asm(Vec<u8>);
impl Asm {
    fn new() -> Self {
        Asm(Vec::new())
    }
    /// One single-slot bundle followed by `gap` 16-bit nop bundles (one functional cycle each).
    fn op_gap(mut self, name: &str, ops: &[(&str, &str)], gap: usize) -> Self {
        let (slot, bits) = encode(name, ops);
        self.0
            .extend(bundle::pack_slots(&[(slot, bits)]).unwrap().as_slice());
        for _ in 0..gap {
            self.0.extend([0u8, 0]);
        }
        self
    }
    fn op(self, name: &str, ops: &[(&str, &str)]) -> Self {
        self.op_gap(name, ops, GAP)
    }
    fn bytes(mut self) -> Vec<u8> {
        self.0.extend(isa::done().encode());
        self.0
    }
}
fn run_bytes(bytes: &[u8], setup: impl FnOnce(&mut Tile)) -> Result<Tile, Error> {
    let decoder = PreparedDecoder::new(bytes)?;
    let mut tile = Tile::new(bytes.to_vec())?;
    setup(&mut tile);
    assert_eq!(tile.run(&decoder, 100_000)?, Step::Done);
    Ok(tile)
}
fn run(asm: Asm, setup: impl FnOnce(&mut Tile)) -> Tile {
    run_bytes(&asm.bytes(), setup).unwrap()
}
fn run_err(asm: Asm, setup: impl FnOnce(&mut Tile)) -> Error {
    run_bytes(&asm.bytes(), setup)
        .err()
        .expect("expected an error")
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
fn vec16(v: &[u16; 32]) -> [u8; 64] {
    let mut o = [0; 64];
    for (i, x) in v.iter().enumerate() {
        o[2 * i..2 * i + 2].copy_from_slice(&x.to_le_bytes());
    }
    o
}
fn vec32(v: &[u32; 16]) -> [u8; 64] {
    let mut o = [0; 64];
    for (i, x) in v.iter().enumerate() {
        o[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
    }
    o
}
fn l16(v: &[u8; 64]) -> [u16; 32] {
    std::array::from_fn(|i| u16::from_le_bytes([v[2 * i], v[2 * i + 1]]))
}
fn l32(v: &[u8; 64]) -> [u32; 16] {
    std::array::from_fn(|i| u32::from_le_bytes(v[4 * i..4 * i + 4].try_into().unwrap()))
}
fn dm_bytes(dm: &[i32; 64]) -> [u8; 256] {
    let mut o = [0; 256];
    for (i, w) in dm.iter().enumerate() {
        o[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    o
}
fn dm_from(bytes: &[u8; 256]) -> [i32; 64] {
    std::array::from_fn(|i| i32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}
/// acc64 lane i = bytes 8i..8i+8 of the 256-byte accumulator.
fn acc64_lanes(dm: &[i32; 64]) -> [i64; 32] {
    let b = dm_bytes(dm);
    std::array::from_fn(|i| i64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
}
fn dm_of_acc64(l: &[i64; 32]) -> [i32; 64] {
    let mut b = [0u8; 256];
    for (i, x) in l.iter().enumerate() {
        b[8 * i..8 * i + 8].copy_from_slice(&x.to_le_bytes());
    }
    dm_from(&b)
}
fn random_dm(r: &mut Rng) -> [i32; 64] {
    std::array::from_fn(|_| r.next() as i32)
}
fn random_vec(r: &mut Rng) -> [u8; 64] {
    std::array::from_fn(|_| r.next() as u8)
}
/// Operand lanes with the interesting 16-bit corner values first, then pseudo-random.
fn corner16(r: &mut Rng) -> [u16; 32] {
    let corners = [
        0x8000u16, 0x7fff, 0xffff, 1, 0, 0x8001, 0xc000, 40000, 0x0100, 0xfffe, 2, 0x4000,
    ];
    std::array::from_fn(|i| {
        if i < corners.len() {
            corners[i]
        } else {
            r.next() as u16
        }
    })
}
fn mem64(t: &Tile, off: usize) -> [u8; 64] {
    t.memory[off..off + 64].try_into().unwrap()
}

// ------------------------------------------------------------------ multiply
fn conf(sx: bool, sy: bool, shift16: bool, zero: bool) -> u32 {
    0x5a | u32::from(sx) << 9 | u32::from(sy) << 8 | u32::from(shift16) << 10 | u32::from(zero)
}
fn product(a: u16, b: u16, sx: bool, sy: bool) -> i128 {
    let a = if sx {
        i128::from(a as i16)
    } else {
        i128::from(a)
    };
    let b = if sy {
        i128::from(b as i16)
    } else {
        i128::from(b)
    };
    a * b
}
/// `dst = sh(acc1) + acc2 + p` per acc64 lane, wrapping at 64 bits.
fn mac_model(
    acc1: &[i32; 64],
    acc2: Option<&[i32; 64]>,
    a: &[u8; 64],
    b: &[u8; 64],
    c: u32,
) -> [i32; 64] {
    let (a, b) = (l16(a), l16(b));
    let l1 = acc64_lanes(acc1);
    let l2 = acc2.map(acc64_lanes);
    let out: [i64; 32] = std::array::from_fn(|i| {
        let mut base = i128::from(l1[i]);
        if c & 1 != 0 {
            base = 0
        } else if c & 0x400 != 0 {
            base <<= 16
        }
        let sum = base
            + l2.map_or(0, |l| i128::from(l[i]))
            + product(a[i], b[i], c & 0x200 != 0, c & 0x100 != 0);
        sum as i64
    });
    dm_of_acc64(&out)
}
const SIGN_COMBOS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
#[test]
fn vmul_elementwise_all_sign_combinations() {
    let mut r = Rng(0x1234_5678_9abc_def1);
    let xa = vec16(&corner16(&mut r));
    let xb = {
        let mut v = corner16(&mut r);
        v.reverse();
        vec16(&v)
    };
    let mut asm = Asm::new();
    for k in 0..4 {
        asm = asm.op(
            "VMUL_vmul_cm_core_X_X",
            &[
                ("dst", ["dm0", "dm1", "dm2", "dm3"][k]),
                ("s1", "x0"),
                ("s2", "x2"),
                ("acc", ["r1", "r2", "r3", "r4"][k]),
            ],
        );
    }
    let t = run(asm, |t| {
        t.x[0] = xa;
        t.x[2] = xb;
        for (k, &(sx, sy)) in SIGN_COMBOS.iter().enumerate() {
            t.r[1 + k] = conf(sx, sy, false, false);
        }
    });
    for (k, &(sx, sy)) in SIGN_COMBOS.iter().enumerate() {
        assert_eq!(
            t.dm[k],
            mac_model(&[0; 64], None, &xa, &xb, conf(sx, sy, false, false)),
            "sx={sx} sy={sy}"
        );
    }
}
#[test]
fn vmul_named_corner_products() {
    let a = corner(&[0x8000, 0xffff, 0x8000, 0x7fff]);
    let b = corner(&[0x8000, 0xffff, 0x7fff, 0xffff]);
    fn corner(v: &[u16]) -> [u8; 64] {
        let mut o = [0u16; 32];
        o[..v.len()].copy_from_slice(v);
        vec16(&o)
    }
    for (c, expect) in [
        (
            conf(true, true, false, false),
            [1i64 << 30, 1, -32768 * 32767, -32767],
        ),
        (
            conf(false, false, false, false),
            [1 << 30, 0xffff * 0xffff, 0x8000 * 0x7fff, 0x7fff * 0xffff],
        ),
    ] {
        let t = run(
            Asm::new().op(
                "VMUL_vmul_cm_core_X_X",
                &[("dst", "dm0"), ("s1", "x0"), ("s2", "x1"), ("acc", "r1")],
            ),
            |t| {
                t.x[0] = a;
                t.x[1] = b;
                t.r[1] = c;
            },
        );
        assert_eq!(&acc64_lanes(&t.dm[0])[..4], &expect, "conf {c:#x}");
    }
}
#[test]
fn vmac_shift16_zero_acc_variants() {
    let mut r = Rng(0xdead_beef_0bad_f00d);
    let (xa, xb) = (vec16(&corner16(&mut r)), vec16(&corner16(&mut r)));
    let acc = random_dm(&mut r);
    for &(sx, sy) in &[(true, true), (false, false), (true, false)] {
        for (shift16, zero) in [(false, false), (true, false), (false, true), (true, true)] {
            let c = conf(sx, sy, shift16, zero);
            let t = run(
                Asm::new().op(
                    "VMAC_vmul_cm_core_X_X",
                    &[
                        ("dst", "dm4"),
                        ("acc1", "dm0"),
                        ("s1", "x0"),
                        ("s2", "x2"),
                        ("acc", "r1"),
                    ],
                ),
                |t| {
                    t.x[0] = xa;
                    t.x[2] = xb;
                    t.r[1] = c;
                    t.dm[0] = acc;
                },
            );
            assert_eq!(t.dm[4], mac_model(&acc, None, &xa, &xb, c), "conf {c:#x}");
            assert_eq!(t.dm[0], acc, "acc1 source untouched");
        }
    }
    // VMAC accumulating onto its own destination
    let t = run(
        Asm::new().op(
            "VMAC_vmul_cm_core_X_X",
            &[
                ("dst", "dm0"),
                ("acc1", "dm0"),
                ("s1", "x0"),
                ("s2", "x2"),
                ("acc", "r1"),
            ],
        ),
        |t| {
            t.x[0] = xa;
            t.x[2] = xb;
            t.r[1] = conf(true, true, false, false);
            t.dm[0] = acc;
        },
    );
    assert_eq!(
        t.dm[0],
        mac_model(&acc, None, &xa, &xb, conf(true, true, false, false))
    );
}
#[test]
fn horner_chain_acc_is_acc_shl16_plus_product() {
    let mut r = Rng(77);
    let v: Vec<[u8; 64]> = (0..6).map(|_| vec16(&corner16(&mut r))).collect();
    let (plain, horner) = (
        conf(false, false, false, false),
        conf(false, false, true, false),
    );
    let asm = Asm::new()
        .op(
            "VMUL_vmul_cm_core_X_X",
            &[("dst", "dm0"), ("s1", "x0"), ("s2", "x1"), ("acc", "r1")],
        )
        .op(
            "VMAC_vmul_cm_core_X_X",
            &[
                ("dst", "dm0"),
                ("acc1", "dm0"),
                ("s1", "x2"),
                ("s2", "x3"),
                ("acc", "r2"),
            ],
        )
        .op(
            "VMAC_vmul_cm_core_X_X",
            &[
                ("dst", "dm0"),
                ("acc1", "dm0"),
                ("s1", "x4"),
                ("s2", "x5"),
                ("acc", "r2"),
            ],
        );
    let t = run(asm, |t| {
        for (i, x) in v.iter().enumerate() {
            t.x[i] = *x;
        }
        t.r[1] = plain;
        t.r[2] = horner;
    });
    let (a, b) = (
        v.iter().step_by(2).map(l16).collect::<Vec<_>>(),
        v.iter().skip(1).step_by(2).map(l16).collect::<Vec<_>>(),
    );
    let expect: [i64; 32] = std::array::from_fn(|i| {
        let mut acc = i128::from(a[0][i]) * i128::from(b[0][i]);
        for k in 1..3 {
            acc = (acc << 16) + i128::from(a[k][i]) * i128::from(b[k][i]);
        }
        acc as i64
    });
    assert_eq!(acc64_lanes(&t.dm[0]), expect);
}
#[test]
fn vaddmac_adds_acc2_and_writes_acc1_register() {
    let mut r = Rng(4242);
    let (xa, xb) = (vec16(&corner16(&mut r)), vec16(&corner16(&mut r)));
    let (d1, d2) = (random_dm(&mut r), random_dm(&mut r));
    for &(sx, sy) in &SIGN_COMBOS {
        for (shift16, zero) in [(false, false), (true, false), (false, true), (true, true)] {
            let c = conf(sx, sy, shift16, zero);
            let asm = Asm::new().op(
                "VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X",
                &[
                    ("acc1", "dm1"),
                    ("acc2", "dm2"),
                    ("s1", "x0"),
                    ("s2", "x2"),
                    ("acc", "r1"),
                ],
            );
            let t = run(asm, |t| {
                t.x[0] = xa;
                t.x[2] = xb;
                t.r[1] = c;
                t.dm[1] = d1;
                t.dm[2] = d2;
            });
            assert_eq!(
                t.dm[1],
                mac_model(&d1, Some(&d2), &xa, &xb, c),
                "conf {c:#x}"
            );
            assert_eq!(t.dm[2], d2, "acc2 untouched");
        }
    }
}
#[test]
fn int8_matrix_mode_unchanged_and_other_modes_rejected() {
    let mut r = Rng(99);
    let (xa, xb) = (random_vec(&mut r), random_vec(&mut r));
    let t = run(
        Asm::new().op(
            "VMUL_vmul_cm_core_X_X",
            &[("dst", "dm0"), ("s1", "x0"), ("s2", "x2"), ("acc", "r1")],
        ),
        |t| {
            t.x[0] = xa;
            t.x[2] = xb;
            t.r[1] = 0x308;
        },
    );
    let expect: [i32; 64] = std::array::from_fn(|n| {
        let (row, col) = (n / 8, n % 8);
        (0..8)
            .map(|k| i32::from(xa[row * 8 + k] as i8) * i32::from(xb[k * 8 + col] as i8))
            .sum()
    });
    assert_eq!(t.dm[0], expect);
    let bad = [
        0x308 | 1,
        0x308 | 0x400,
        0x5a | 1 << 11,
        0x5a | 1 << 12,
        0x5a | 1 << 13,
        0x5a | 1 << 16,
        0x5a + 0x300 + 0x20,
        0x7a,
        0,
        2,
    ];
    for c in bad {
        let e = run_err(
            Asm::new().op(
                "VMUL_vmul_cm_core_X_X",
                &[("dst", "dm0"), ("s1", "x0"), ("s2", "x2"), ("acc", "r1")],
            ),
            |t| t.r[1] = c,
        );
        assert_eq!(e, Error::Unsupported, "conf {c:#x}");
    }
    let e = run_err(
        Asm::new().op(
            "VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X",
            &[
                ("acc1", "dm1"),
                ("acc2", "dm2"),
                ("s1", "x0"),
                ("s2", "x2"),
                ("acc", "r1"),
            ],
        ),
        |t| t.r[1] = 0x308,
    );
    assert_eq!(e, Error::Unsupported, "int8 matrix mode has no acc2");
}
#[test]
fn vneg_acc64_and_acc32_lanes() {
    let mut r = Rng(5);
    let mut d = random_dm(&mut r);
    d[0] = i32::MIN;
    d[1] = 0;
    d[2] = i32::MIN;
    d[3] = i32::MIN; // acc64 lane 0 = i64 -2^31 ... lane 1 = i32::MIN|MIN<<32
    for (c, wide) in [(2u32, true), (0, false)] {
        let t = run(
            Asm::new().op("VNEG", &[("dst", "dm2"), ("acc1", "dm1"), ("acc", "r1")]),
            |t| {
                t.dm[1] = d;
                t.r[1] = c;
            },
        );
        let expect: [i32; 64] = if wide {
            dm_of_acc64(&acc64_lanes(&d).map(|v| (-i128::from(v)) as i64))
        } else {
            d.map(|v| (-i64::from(v)) as i32)
        };
        assert_eq!(t.dm[2], expect, "amode wide={wide}");
    }
    let e = run_err(
        Asm::new().op("VNEG", &[("dst", "dm2"), ("acc1", "dm1"), ("acc", "r1")]),
        |t| t.r[1] = 3,
    );
    assert_eq!(e, Error::Unsupported);
}

// ------------------------------------------------------------------ SRS
/// Exact reference: choose between the integers around v/2^s from the exact remainder, then saturate.
fn srs_model(v: i128, s: u32, rnd: u32, sat: u32, bits: u32) -> i64 {
    let d = 1i128 << s;
    let fl = v.div_euclid(d);
    let rem = v.rem_euclid(d);
    let up = match rnd {
        0 => false,
        1 => rem > 0,
        2 => v < 0 && rem > 0,
        3 => v > 0 && rem > 0,
        _ => {
            let twice = 2 * rem;
            if twice > d {
                true
            } else if twice < d {
                false
            } else {
                match rnd {
                    8 => false,
                    9 => true,
                    10 => v < 0,
                    11 => v > 0,
                    12 => fl.rem_euclid(2) == 1,
                    13 => fl.rem_euclid(2) == 0,
                    _ => unreachable!(),
                }
            }
        }
    };
    let r = fl + i128::from(up);
    let (lo, hi) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
    let out = match sat {
        0 => {
            let m = 1i128 << bits;
            let w = r.rem_euclid(m);
            if w >= m / 2 {
                w - m
            } else {
                w
            }
        }
        1 => r.clamp(lo, hi),
        3 => r.clamp(lo + 1, hi),
        _ => unreachable!(),
    };
    out as i64
}
/// `cm`: None = whole dm (4x), Some(half) = cml/cmh (2x). `mode` = crSRSMode.
fn srs_expected(
    dm: &[i32; 64],
    cm: Option<usize>,
    mode: u32,
    s: u32,
    rnd: u32,
    sat: u32,
) -> [u8; 64] {
    let (lanes, bits): (Vec<i128>, u32) = match (cm, mode) {
        (None, 0) => (dm.iter().map(|&v| i128::from(v)).collect(), 8),
        (None, _) => (acc64_lanes(dm).iter().map(|&v| i128::from(v)).collect(), 16),
        (Some(h), 0) => (
            dm[32 * h..32 * h + 32]
                .iter()
                .map(|&v| i128::from(v))
                .collect(),
            16,
        ),
        (Some(h), _) => (
            acc64_lanes(dm)[16 * h..16 * h + 16]
                .iter()
                .map(|&v| i128::from(v))
                .collect(),
            32,
        ),
    };
    let mut out = Vec::new();
    for v in lanes {
        out.extend_from_slice(
            &srs_model(v, s, rnd, sat, bits).to_le_bytes()[..(bits / 8) as usize],
        );
    }
    out.try_into().unwrap()
}
/// Accumulator lane near a rounding boundary of `v >> s` (ties, ties +-1, exact, extremes).
fn srs_lane_value(r: &mut Rng, s: u32, wide: bool) -> i64 {
    let half = if s == 0 { 0i128 } else { 1i128 << (s - 1) };
    let (lo, hi) = if wide {
        (i128::from(i64::MIN), i128::from(i64::MAX))
    } else {
        (i128::from(i32::MIN), i128::from(i32::MAX))
    };
    let k = i128::from((r.next() as i64) >> (r.next() % 62));
    let offset =
        [0, half, half - 1, half + 1, -half, (1i128 << s) - 1, 1, -1][(r.next() % 8) as usize];
    if r.next() % 16 == 0 {
        return [lo as i64, hi as i64, 0, -1][(r.next() % 4) as usize];
    }
    (k * (1i128 << s) + offset).clamp(lo, hi) as i64
}
fn srs_data(r: &mut Rng, s: u32, wide: bool) -> [i32; 64] {
    if wide {
        dm_of_acc64(&std::array::from_fn(|_| srs_lane_value(r, s, true)))
    } else {
        std::array::from_fn(|_| srs_lane_value(r, s, false) as i32)
    }
}
const RNDS: [u32; 10] = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13];
#[test]
fn vsrs_to_register_every_shape_rounding_saturation() {
    // (encoding, src operand, cm half, crSRSMode)
    let cases = [
        ("VSRS_4x_mv_x_srs_dm_srsSign1", "dm1", None, 0u32),
        ("VSRS_4x_mv_x_srs_dm_srsSign1", "dm1", None, 1),
        ("VSRS_2x_mv_x_srs_cm_srsSign1", "cml1", Some(0usize), 0),
        ("VSRS_2x_mv_x_srs_cm_srsSign1", "cmh1", Some(1), 0),
        ("VSRS_2x_mv_x_srs_cm_srsSign1", "cml1", Some(0), 1),
        ("VSRS_2x_mv_x_srs_cm_srsSign1", "cmh1", Some(1), 1),
    ];
    let mut r = Rng(0x5eed_5eed);
    for (name, src, half, mode) in cases {
        let bytes = Asm::new()
            .op(name, &[("dst", "x4"), ("src", src), ("su", "s2")])
            .bytes();
        let decoder = PreparedDecoder::new(&bytes).unwrap();
        let wide = mode == 1;
        for s in if wide {
            vec![0u32, 16, 23, 39]
        } else {
            vec![0, 16, 23]
        } {
            for sat in [0u32, 1, 3] {
                for rnd in RNDS {
                    let dm = srs_data(&mut r, s, wide);
                    let mut t = Tile::new(bytes.clone()).unwrap();
                    t.dm[1] = dm;
                    t.s[2] = s;
                    t.cr_sat = sat;
                    t.cr_rnd = rnd;
                    t.cr_srs_mode = mode;
                    t.x[4] = [0xaa; 64];
                    assert_eq!(t.run(&decoder, 1000).unwrap(), Step::Done);
                    assert_eq!(
                        t.x[4],
                        srs_expected(&dm, half, mode, s, rnd, sat),
                        "{name} {src} mode {mode} s={s} sat={sat} rnd={rnd}"
                    );
                }
            }
        }
    }
}
#[test]
fn vst_srs_stores_all_shapes_and_advances_pointer() {
    let cases = [
        (
            "VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1",
            "dm1",
            None,
            0u32,
            true,
        ),
        (
            "VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1",
            "dm1",
            None,
            1,
            true,
        ),
        (
            "VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1",
            "cml1",
            Some(0usize),
            0,
            true,
        ),
        (
            "VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1",
            "cmh1",
            Some(1),
            0,
            true,
        ),
        (
            "VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1",
            "cml1",
            Some(0),
            1,
            true,
        ),
        (
            "VST_SRS_2x_dm_sts_srs_cm_pstm_nrm_imm_srsSign1",
            "cmh1",
            Some(1),
            1,
            true,
        ),
        (
            "VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign1",
            "cmh1",
            Some(1),
            1,
            false,
        ),
        (
            "VST_SRS_2x_dm_sts_srs_cm_idx_imm_srsSign1",
            "cml1",
            Some(0),
            0,
            false,
        ),
    ];
    let mut r = Rng(0xabcdef);
    for (name, src, half, mode, post) in cases {
        let bytes = Asm::new()
            .op(
                name,
                &[("src", src), ("su", "s1"), ("ptr", "p0"), ("imm", "1")],
            )
            .bytes();
        let decoder = PreparedDecoder::new(&bytes).unwrap();
        for (s, sat, rnd) in [(0u32, 1u32, 0u32), (16, 0, 12), (23, 3, 9), (7, 1, 13)] {
            let dm = srs_data(&mut r, s, mode == 1);
            let mut t = Tile::new(bytes.clone()).unwrap();
            t.dm[1] = dm;
            t.s[1] = s;
            t.cr_sat = sat;
            t.cr_rnd = rnd;
            t.cr_srs_mode = mode;
            t.p[0] = MEM + 0x100;
            assert_eq!(t.run(&decoder, 1000).unwrap(), Step::Done);
            let target = if post { 0x100 } else { 0x140 };
            assert_eq!(
                mem64(&t, target),
                srs_expected(&dm, half, mode, s, rnd, sat),
                "{name} {src} mode {mode} s={s}"
            );
            assert_eq!(
                t.p[0],
                if post { MEM + 0x140 } else { MEM + 0x100 },
                "{name} pointer"
            );
        }
    }
    // an invalid crSRSMode-independent request stays refused: shift 64 is not a shift
    let e = run_err(
        Asm::new().op(
            "VSRS_4x_mv_x_srs_dm_srsSign1",
            &[("dst", "x4"), ("src", "dm1"), ("su", "s0")],
        ),
        |t| {
            t.s[0] = 64;
            t.cr_srs_mode = 1;
        },
    );
    assert_eq!(e, Error::Unsupported);
    let e = run_err(
        Asm::new().op(
            "VSRS_2x_mv_x_srs_cm_srsSign0",
            &[("dst", "x4"), ("src", "cml0"), ("su", "s0")],
        ),
        |_| {},
    );
    assert_eq!(e, Error::Unsupported, "srsSign0 stays unsupported");
}

// ------------------------------------------------------------------ UPS
fn movx(asm: Asm, cr: &str, imm: u32) -> Asm {
    asm.op("MOVX_mvx_cr_imm", &[("dst", cr), ("src", &imm.to_string())])
}
fn fits(v: i128, signed: bool, bits: u32) -> bool {
    if signed {
        v >= -(1i128 << (bits - 1)) && v < (1i128 << (bits - 1))
    } else {
        v >= 0 && v < (1i128 << bits)
    }
}
#[test]
fn vups_4x_16bit_to_acc64_and_2x_modes() {
    let mut r = Rng(31337);
    let x = vec16(&corner16(&mut r));
    let x32 = vec32(&std::array::from_fn(|i| {
        [
            0x8000_0000u32,
            0x7fff_ffff,
            0xffff_ffff,
            1,
            0,
            0xdead_beef,
            0x8000_0001,
        ][i % 7]
    }));
    // 4x, crUPSMode = 1: 32 x 16-bit -> 32 acc64 (whole dm)
    for (name, signed) in [
        ("VUPS_4x_mv_ups_x2d_upsSign1", true),
        ("VUPS_4x_mv_ups_x2d_upsSign0", false),
    ] {
        for s in [0u32, 1, 15, 16, 32, 47] {
            let asm = movx(Asm::new(), "crUPSMode", 1)
                .op(name, &[("dst", "dm2"), ("src", "x0"), ("su", "s3")]);
            let t = run(asm, |t| {
                t.x[0] = x;
                t.s[3] = s;
                t.dm[2] = [0x5a5a5a5a; 64];
            });
            let expect: [i64; 32] = l16(&x).map(|v| {
                let e = if signed {
                    i128::from(v as i16)
                } else {
                    i128::from(v)
                };
                (e << s) as i64
            });
            assert_eq!(t.dm[2], dm_of_acc64(&expect), "{name} s={s}");
            assert_eq!(t.cr_ups_mode, 1);
        }
    }
    // 2x, crUPSMode = 1: 16 x 32-bit -> 16 acc64 into one cm half; the other half is preserved
    for (name, signed) in [
        ("VUPS_2x_mv_ups_x2c_upsSign1", true),
        ("VUPS_2x_mv_ups_x2c_upsSign0", false),
    ] {
        for (dst, half) in [("cml1", 0usize), ("cmh1", 1)] {
            for s in [0u32, 3, 16, 31] {
                let before = random_dm(&mut r);
                let asm = movx(Asm::new(), "crUPSMode", 1)
                    .op(name, &[("dst", dst), ("src", "x2"), ("su", "s0")]);
                let t = run(asm, |t| {
                    t.x[2] = x32;
                    t.s[0] = s;
                    t.dm[1] = before;
                });
                let mut lanes = acc64_lanes(&before);
                for (i, v) in l32(&x32).iter().enumerate() {
                    let e = if signed {
                        i128::from(*v as i32)
                    } else {
                        i128::from(*v)
                    };
                    lanes[16 * half + i] = (e << s) as i64;
                }
                assert_eq!(t.dm[1], dm_of_acc64(&lanes), "{name} {dst} s={s}");
            }
        }
    }
    // 2x, crUPSMode = 0: 32 x 16-bit -> 32 acc32 words of the half
    for (name, signed) in [
        ("VUPS_2x_mv_ups_x2c_upsSign1", true),
        ("VUPS_2x_mv_ups_x2c_upsSign0", false),
    ] {
        for (dst, half) in [("cml3", 0usize), ("cmh3", 1)] {
            for s in [0u32, 4, 16] {
                let before = random_dm(&mut r);
                let t = run(
                    Asm::new().op(name, &[("dst", dst), ("src", "x0"), ("su", "s1")]),
                    |t| {
                        t.x[0] = x;
                        t.s[1] = s;
                        t.dm[3] = before;
                    },
                );
                let mut expect = before;
                for (i, v) in l16(&x).iter().enumerate() {
                    let e = if signed {
                        i128::from(*v as i16)
                    } else {
                        i128::from(*v)
                    };
                    expect[32 * half + i] = ((e << s) as i64) as i32;
                }
                assert_eq!(t.dm[3], expect, "{name} {dst} acc32 s={s}");
            }
        }
    }
}
#[test]
fn vups_lane_overflow_and_unmodelled_modes_are_errors() {
    let one = vec16(&[1u16; 32]);
    // signed 1 << 63 does not fit a signed acc64 lane; unsigned 0xffff << 63 does not fit 64 bits
    let asm = movx(Asm::new(), "crUPSMode", 1).op(
        "VUPS_4x_mv_ups_x2d_upsSign1",
        &[("dst", "dm0"), ("src", "x0"), ("su", "s0")],
    );
    assert_eq!(
        run_err(asm, |t| {
            t.x[0] = one;
            t.s[0] = 63;
        }),
        Error::Unsupported
    );
    let asm = movx(Asm::new(), "crUPSMode", 1).op(
        "VUPS_4x_mv_ups_x2d_upsSign0",
        &[("dst", "dm0"), ("src", "x0"), ("su", "s0")],
    );
    assert_eq!(
        run_err(asm, |t| {
            t.x[0] = vec16(&[0xffff; 32]);
            t.s[0] = 63;
        }),
        Error::Unsupported
    );
    assert!(fits(1 << 62, true, 64) && !fits(1 << 63, true, 64) && fits(1 << 63, false, 64));
    // acc32 lane: 0x7fff << 17 overflows int32
    let asm = Asm::new().op(
        "VUPS_2x_mv_ups_x2c_upsSign1",
        &[("dst", "cml0"), ("src", "x0"), ("su", "s0")],
    );
    assert_eq!(
        run_err(asm, |t| {
            t.x[0] = vec16(&[0x7fff; 32]);
            t.s[0] = 17;
        }),
        Error::Unsupported
    );
    // 4x with crUPSMode = 0 is not modelled
    let asm = Asm::new().op(
        "VUPS_4x_mv_ups_x2d_upsSign1",
        &[("dst", "dm0"), ("src", "x0"), ("su", "s0")],
    );
    assert_eq!(run_err(asm, |_| {}), Error::Unsupported);
}

// ------------------------------------------------------------------ unpack
fn nibble_pattern() -> [u8; 64] {
    std::array::from_fn(|i| match i {
        0 => 0x98,
        1 => 0xba,
        2 => 0xdc,
        3 => 0xfe,
        4 => 0x07,
        5 => 0x70,
        6 => 0x80,
        7 => 0x08,
        _ => (i as u8).wrapping_mul(37).wrapping_add(11),
    })
}
fn unpack_model(src: &[u8; 64], size: u32, signed: bool) -> [u8; 128] {
    let mut out = [0u8; 128];
    if size == 0 {
        let nibbles: Vec<u8> = src.iter().flat_map(|b| [b & 15, b >> 4]).collect();
        for (n, &v) in nibbles.iter().enumerate() {
            out[n] = if signed && v >= 8 {
                (v as i8 - 16) as u8
            } else {
                v
            };
        }
    } else {
        for (n, &v) in src.iter().enumerate() {
            let e: u16 = if signed {
                i16::from(v as i8) as u16
            } else {
                u16::from(v)
            };
            out[2 * n..2 * n + 2].copy_from_slice(&e.to_le_bytes());
        }
    }
    out
}
#[test]
fn unpack_load_and_register_forms() {
    let src = nibble_pattern();
    for (size, signed) in [(0u32, true), (0, false), (1, true), (1, false)] {
        let sign = if signed { "unpackSign1" } else { "unpackSign0" };
        let expect = unpack_model(&src, size, signed);
        // VLDB.UNPACK: 64 B at p0, ptr += 64 (imm raw 1); y1 = (x2, x3)
        let asm = movx(Asm::new(), "crUnpackSize", size).op(
            &format!("VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_{sign}"),
            &[("dst", "y1"), ("ptr", "p0"), ("imm", "1")],
        );
        let t = run(asm, |t| {
            t.memory[0x80..0xc0].copy_from_slice(&src);
            t.p[0] = MEM + 0x80;
            t.x[2] = [0xee; 64];
            t.x[3] = [0xee; 64];
        });
        assert_eq!(
            &t.x[2][..],
            &expect[..64],
            "vldb.unpack size {size} {sign} lo"
        );
        assert_eq!(
            &t.x[3][..],
            &expect[64..],
            "vldb.unpack size {size} {sign} hi"
        );
        assert_eq!(t.p[0], MEM + 0xc0);
        assert_eq!(t.cr_unpack_size, size);
        // VUNPACK x0 -> y3 = (x6, x7)
        let asm = movx(Asm::new(), "crUnpackSize", size).op(
            &format!("VUNPACK_mv_unpack_x_{sign}"),
            &[("dst", "y3"), ("src", "x0")],
        );
        let t = run(asm, |t| t.x[0] = src);
        assert_eq!(
            (&t.x[6][..], &t.x[7][..]),
            (&expect[..64], &expect[64..]),
            "vunpack size {size} {sign}"
        );
    }
    // spot values: nibbles 0x8..0xF sign-extend to 0xf8..0xff, low nibble first
    let t = run(
        movx(Asm::new(), "crUnpackSize", 0).op(
            "VUNPACK_mv_unpack_x_unpackSign1",
            &[("dst", "y0"), ("src", "x4")],
        ),
        |t| t.x[4] = nibble_pattern(),
    );
    assert_eq!(
        &t.x[0][..8],
        &[0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff]
    );
    // crUnpackSize via movx_r; value 2 stays below the modelled sizes
    let t = run(
        Asm::new().op("MOVX_mvx_cr_r", &[("dst", "crUnpackSize"), ("src", "r5")]),
        |t| t.r[5] = 1,
    );
    assert_eq!(t.cr_unpack_size, 1);
    // misaligned source
    let asm = Asm::new().op(
        "VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1",
        &[("dst", "y1"), ("ptr", "p0"), ("imm", "1")],
    );
    assert_eq!(run_err(asm, |t| t.p[0] = MEM + 0x20), Error::Alignment);
}

// ------------------------------------------------------------------ x / bm memory forms
#[test]
fn x_and_bm_loads_and_stores_all_addressing_forms() {
    let mut r = Rng(2024);
    let img: Vec<u8> = (0..0x400).map(|_| r.next() as u8).collect();
    let asm = Asm::new()
        .op(
            "VLDB_dmx_ldb_x_pstm_nrm_imm",
            &[("dst", "x3"), ("ptr", "p0"), ("imm", "2")],
        ) // x3 = [p0]; p0 += 128
        .op(
            "VLDA_dmx_lda_x_pstm_nrm_imm",
            &[("dst", "x5"), ("ptr", "p0"), ("imm", "15")],
        ) // x5 = [p0]; p0 -= 64
        .op(
            "VST_dmx_sts_x_pstm_nrm_imm",
            &[("src", "x3"), ("ptr", "p1"), ("imm", "1")],
        ) // [p1] = x3; p1 += 64
        .op(
            "VLDA_dmx_lda_x_idx_imm",
            &[("dst", "x6"), ("ptr", "p2"), ("imm", "15")],
        ) // x6 = [p2 - 64]
        .op(
            "VLDB_dmx_ldb_x_idx_imm",
            &[("dst", "x7"), ("ptr", "p2"), ("imm", "1")],
        ) // x7 = [p2 + 64]
        .op(
            "VST_dmx_sts_x_idx_imm",
            &[("src", "x6"), ("ptr", "p3"), ("imm", "2")],
        ) // [p3 + 128] = x6
        .op(
            "VLDA_dmx_lda_bm_idx_imm",
            &[("dst", "bmhl1"), ("ptr", "p2"), ("imm", "2")],
        ) // bm quarter 6 = [p2 + 128]
        .op(
            "VST_dmx_sts_bm_idx_imm",
            &[("src", "bmhl1"), ("ptr", "p3"), ("imm", "3")],
        ) // [p3 + 192] = bm quarter 6
        .op(
            "VLDA_dmx_lda_x_pstm_nrm",
            &[("dst", "x8"), ("ptr", "p4"), ("mod", "m1")],
        ) // x8 = [p4]; p4 += m1
        .op(
            "VST_dmx_sts_x_pstm_nrm",
            &[("src", "x8"), ("ptr", "p5"), ("mod", "m1")],
        ) // [p5] = x8; p5 += m1
        .op(
            "VLDA_dmx_lda_bm_pstm_nrm",
            &[("dst", "bmlh2"), ("ptr", "p4"), ("mod", "m1")],
        ) // bm 9 = [p4]; p4 += m1
        .op(
            "VST_dmx_sts_bm_pstm_nrm",
            &[("src", "bmlh2"), ("ptr", "p6"), ("mod", "m1")],
        ); // [p6] = bm 9; p6 += m1
    let t = run(asm, |t| {
        t.memory[..0x400].copy_from_slice(&img);
        t.p[0] = MEM;
        t.p[1] = MEM + 0x800;
        t.p[2] = MEM + 0x200;
        t.p[3] = MEM + 0xa00;
        t.p[4] = MEM + 0x300;
        t.p[5] = MEM + 0xc00;
        t.p[6] = MEM + 0xd00;
        t.m[1] = 64;
    });
    let at = |o: usize| -> [u8; 64] { img[o..o + 64].try_into().unwrap() };
    assert_eq!(t.x[3], at(0));
    assert_eq!(t.x[5], at(128));
    assert_eq!((t.p[0], t.p[1]), (MEM + 128 - 64, MEM + 0x840));
    assert_eq!(mem64(&t, 0x800), at(0));
    assert_eq!(t.x[6], at(0x1c0));
    assert_eq!(t.x[7], at(0x240));
    assert_eq!(t.p[2], MEM + 0x200, "idx forms do not move the pointer");
    assert_eq!(mem64(&t, 0xa80), at(0x1c0));
    assert_eq!(&dm_bytes(&t.dm[1])[128..192], &at(0x280)[..]);
    assert_eq!(mem64(&t, 0xa00 + 192), at(0x280));
    assert_eq!(t.x[8], at(0x300));
    assert_eq!(mem64(&t, 0xc00), at(0x300));
    assert_eq!(&dm_bytes(&t.dm[2])[64..128], &at(0x340)[..]);
    assert_eq!(mem64(&t, 0xd00), at(0x340));
    assert_eq!(
        (t.p[4], t.p[5], t.p[6]),
        (MEM + 0x380, MEM + 0xc40, MEM + 0xd40)
    );
}

// ------------------------------------------------------------------ vector ALU
fn shuffle_model(a: &[u8; 64], b: &[u8; 64], mode: u32) -> [u8; 64] {
    let (w, intlv) = match mode {
        0..=11 => (1usize << (mode / 2), false),
        _ => (16 >> ((mode - 12) / 2), true),
    };
    let n = 64 / w;
    let hi = mode % 2 == 1;
    let el = |v: &[u8; 64], e: usize| v[e * w..(e + 1) * w].to_vec();
    let picked: Vec<Vec<u8>> = if intlv {
        let mut seq = Vec::new();
        for e in 0..n {
            seq.push(el(a, e));
            seq.push(el(b, e));
        }
        seq[if hi { n } else { 0 }..][..n].to_vec()
    } else {
        let cat: Vec<Vec<u8>> = (0..n)
            .map(|e| el(a, e))
            .chain((0..n).map(|e| el(b, e)))
            .collect();
        (0..n)
            .map(|j| cat[2 * j + usize::from(hi)].clone())
            .collect()
    };
    picked.concat().try_into().unwrap()
}
#[test]
fn vshuffle_every_interleave_and_deinterleave_mode() {
    let mut r = Rng(808);
    let (a, b) = (random_vec(&mut r), random_vec(&mut r));
    let bytes = Asm::new()
        .op(
            "VSHUFFLE_vec_shuffle_x",
            &[("dst", "x6"), ("s1", "x0"), ("s2", "x2"), ("mod", "r7")],
        )
        .bytes();
    let decoder = PreparedDecoder::new(&bytes).unwrap();
    for mode in 0..=21u32 {
        let mut t = Tile::new(bytes.clone()).unwrap();
        t.x[0] = a;
        t.x[2] = b;
        t.r[7] = mode;
        assert_eq!(t.run(&decoder, 1000).unwrap(), Step::Done);
        assert_eq!(t.x[6], shuffle_model(&a, &b, mode), "mode {mode}");
    }
    // spot value: T16_2x32_lo (18) = a0 b0 a1 b1 ... over 16-bit elements
    let (la, lb) = (l16(&a), l16(&b));
    let mut t = Tile::new(bytes.clone()).unwrap();
    t.x[0] = a;
    t.x[2] = b;
    t.r[7] = 18;
    t.run(&decoder, 1000).unwrap();
    assert_eq!(&l16(&t.x[6])[..4], &[la[0], lb[0], la[1], lb[1]]);
    // de-interleave then interleave is the identity
    let mut t = Tile::new(bytes.clone()).unwrap();
    t.x[0] = a;
    t.x[2] = b;
    t.r[7] = 2;
    t.run(&decoder, 1000).unwrap();
    let lo = t.x[6];
    let mut t = Tile::new(bytes.clone()).unwrap();
    t.x[0] = a;
    t.x[2] = b;
    t.r[7] = 3;
    t.run(&decoder, 1000).unwrap();
    let hi = t.x[6];
    let mut t = Tile::new(bytes.clone()).unwrap();
    t.x[0] = lo;
    t.x[2] = hi;
    t.r[7] = 18;
    t.run(&decoder, 1000).unwrap();
    assert_eq!(t.x[6], a);
    let mut t = Tile::new(bytes.clone()).unwrap();
    t.x[0] = lo;
    t.x[2] = hi;
    t.r[7] = 19;
    t.run(&decoder, 1000).unwrap();
    assert_eq!(t.x[6], b);
    for mode in [22u32, 23, 30, 63] {
        let mut t = Tile::new(bytes.clone()).unwrap();
        t.r[7] = mode;
        assert_eq!(
            t.run(&decoder, 1000).err(),
            Some(Error::Unsupported),
            "mode {mode}"
        );
    }
}
fn lanes_i(v: &[u8; 64], wide: bool, signed: bool) -> Vec<i64> {
    if wide {
        l32(v)
            .iter()
            .map(|&x| {
                if signed {
                    i64::from(x as i32)
                } else {
                    i64::from(x)
                }
            })
            .collect()
    } else {
        l16(v)
            .iter()
            .map(|&x| {
                if signed {
                    i64::from(x as i16)
                } else {
                    i64::from(x)
                }
            })
            .collect()
    }
}
fn mask_of(l: &[bool]) -> u32 {
    l.iter()
        .enumerate()
        .fold(0, |m, (i, &b)| m | u32::from(b) << i)
}
fn lane_corner_vectors(r: &mut Rng) -> ([u8; 64], [u8; 64]) {
    // equal lanes, sign-boundary lanes, zero lanes, then random
    let mut a = corner16(r);
    let mut b = corner16(r);
    for i in 0..8 {
        b[i] = a[i];
    }
    b[8] = a[8].wrapping_add(1);
    b[9] = a[9].wrapping_sub(1);
    a[12] = 0;
    b[12] = 0;
    a[13] = 0;
    b[13] = 0x8000;
    a[14] = 0x8000;
    b[14] = 0x7fff;
    (vec16(&a), vec16(&b))
}
#[test]
fn vector_compare_select_and_lanewise_alu() {
    let mut r = Rng(606);
    let (a, b) = lane_corner_vectors(&mut r);
    // 32-bit lanes: reuse the same bytes reinterpreted
    for wide in [false, true] {
        let w = if wide { "32" } else { "16" };
        for signed in [false, true] {
            let sign = if signed { "vaddSign1" } else { "vaddSign0" };
            let (la, lb) = (lanes_i(&a, wide, signed), lanes_i(&b, wide, signed));
            let t = run(
                Asm::new()
                    .op(
                        &format!("VLT_{w}_{sign}"),
                        &[("cmp", "r17"), ("s1", "x0"), ("s2", "x2")],
                    )
                    .op(
                        &format!("VGE_{w}_{sign}"),
                        &[("cmp", "r18"), ("s1", "x0"), ("s2", "x2")],
                    ),
                |t| {
                    t.x[0] = a;
                    t.x[2] = b;
                },
            );
            let lt: Vec<bool> = la.iter().zip(&lb).map(|(x, y)| x < y).collect();
            let ge: Vec<bool> = la.iter().zip(&lb).map(|(x, y)| x >= y).collect();
            assert_eq!(
                (t.r[17], t.r[18]),
                (mask_of(&lt), mask_of(&ge)),
                "vlt/vge.{w} {sign}"
            );
            // vmax_lt / vmin_ge
            for max in [true, false] {
                let name = if max {
                    format!("VMAX_LT_{w}_{sign}")
                } else {
                    format!("VMIN_GE_{w}_{sign}")
                };
                let t = run(
                    Asm::new().op(&name, &[("d", "x4"), ("s1", "x0"), ("s2", "x2")]),
                    |t| {
                        t.x[0] = a;
                        t.x[2] = b;
                        t.r[16] = 0xffff_ffff;
                    },
                );
                let sel: Vec<bool> = if max { lt.clone() } else { ge.clone() };
                let got = lanes_i(&t.x[4], wide, signed);
                for i in 0..la.len() {
                    let expect = if max {
                        la[i].max(lb[i])
                    } else {
                        la[i].min(lb[i])
                    };
                    assert_eq!(got[i], expect, "{name} lane {i}");
                }
                assert_eq!(t.r[16], mask_of(&sel), "{name} r16 mask");
            }
        }
        // veqz
        let z = if wide {
            vec32(&[
                0,
                1,
                0,
                0x8000_0000,
                0xffff_ffff,
                0,
                5,
                0x1_0000u32 >> 1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                7,
            ])
        } else {
            vec16(&std::array::from_fn(|i| {
                if i % 3 == 0 {
                    0
                } else {
                    i as u16 | 0x100
                }
            }))
        };
        let t = run(
            Asm::new().op(&format!("VEQZ_{w}"), &[("cmp", "r25"), ("s2", "x1")]),
            |t| t.x[1] = z,
        );
        let expect: Vec<bool> = lanes_i(&z, wide, false).iter().map(|&v| v == 0).collect();
        assert_eq!(t.r[25], mask_of(&expect), "veqz.{w}");
        // vsel: bit set -> s2 (second source), clear -> s1
        for sel in [0xa5a5_c3c3u32, 0, 0xffff_ffff, 0x8000_0001] {
            let t = run(
                Asm::new().op(
                    &format!("VSEL_{w}"),
                    &[("d", "x4"), ("s1", "x0"), ("s2", "x2"), ("sel", "r20")],
                ),
                |t| {
                    t.x[0] = a;
                    t.x[2] = b;
                    t.r[20] = sel;
                },
            );
            let (ua, ub) = (lanes_i(&a, wide, false), lanes_i(&b, wide, false));
            let got = lanes_i(&t.x[4], wide, false);
            for i in 0..ua.len() {
                assert_eq!(
                    got[i],
                    if sel >> i & 1 == 1 { ub[i] } else { ua[i] },
                    "vsel.{w} sel={sel:#x} lane {i}"
                );
            }
        }
        // vadd / vsub wrap
        for (op, f) in [
            (
                "VADD",
                (|x: i64, y: i64| x.wrapping_add(y)) as fn(i64, i64) -> i64,
            ),
            ("VSUB", |x, y| x.wrapping_sub(y)),
        ] {
            let t = run(
                Asm::new().op(
                    &format!("{op}_{w}"),
                    &[("d", "x4"), ("s1", "x0"), ("s2", "x2")],
                ),
                |t| {
                    t.x[0] = a;
                    t.x[2] = b;
                },
            );
            let (ua, ub) = (lanes_i(&a, wide, false), lanes_i(&b, wide, false));
            let mask = if wide { 0xffff_ffffi64 } else { 0xffff };
            let got = lanes_i(&t.x[4], wide, false);
            for i in 0..ua.len() {
                assert_eq!(got[i], f(ua[i], ub[i]) & mask, "{op}.{w} lane {i}");
            }
        }
    }
    let t = run(
        Asm::new()
            .op("VBAND", &[("d", "x4"), ("s1", "x0"), ("s2", "x2")])
            .op("VBOR", &[("d", "x6"), ("s1", "x0"), ("s2", "x2")]),
        |t| {
            t.x[0] = a;
            t.x[2] = b;
        },
    );
    assert_eq!(t.x[4], std::array::from_fn::<u8, 64, _>(|i| a[i] & b[i]));
    assert_eq!(t.x[6], std::array::from_fn::<u8, 64, _>(|i| a[i] | b[i]));
}
#[test]
fn vbcst_and_vmov_forms() {
    let t = run(
        Asm::new()
            .op("VBCST_16", &[("dst", "x4"), ("src", "r5")])
            .op("VBCST_32", &[("dst", "x6"), ("src", "r5")]),
        |t| t.r[5] = 0xdead_beef,
    );
    assert_eq!(t.x[4], vec16(&[0xbeef; 32]));
    assert_eq!(t.x[6], vec32(&[0xdead_beef; 16]));
    let mut r = Rng(11);
    let v = random_vec(&mut r);
    // x -> x, x -> bm quarter (bmlh1 = 4*1+1), bm quarter -> x
    let asm = Asm::new()
        .op("VMOV_alu_mv_mv_x", &[("dst", "x9"), ("src", "x0")])
        .op("VMOV_alu_mv_mv_x", &[("dst", "bmlh1"), ("src", "x0")])
        .op("VMOV_alu_mv_mv_x", &[("dst", "x10"), ("src", "bmlh1")]);
    let t = run(asm, |t| t.x[0] = v);
    assert_eq!(t.x[9], v);
    assert_eq!(t.x[10], v);
    assert_eq!(&dm_bytes(&t.dm[1])[64..128], &v[..]);
}

// ------------------------------------------------------------------ scalar ALU
#[test]
fn scalar_alu_and_or_add_lshl_eqz_nez() {
    let (a, b) = (0xf0f0_1234u32, 0x0ff3_ffffu32);
    let asm = Asm::new()
        .op("AND", &[("d0", "r10"), ("s0", "r1"), ("s1", "r2")])
        .op("OR", &[("d0", "r11"), ("s0", "r1"), ("s1", "r2")])
        .op("ADD_alu_r_rr", &[("d0", "r12"), ("s0", "r1"), ("s1", "r2")])
        .op("MOV_OR", &[("d0", "r13"), ("s0", "r1")])
        .op("EQZ", &[("d0", "r14"), ("s0", "r3")])
        .op("EQZ", &[("d0", "r15"), ("s0", "r1")])
        .op("NEZ", &[("d0", "r16"), ("s0", "r3")])
        .op("NEZ", &[("d0", "r17"), ("s0", "r1")]);
    let t = run(asm, |t| {
        t.r[1] = a;
        t.r[2] = b;
        t.r[3] = 0;
    });
    assert_eq!(
        &t.r[10..18],
        &[a & b, a | b, a.wrapping_add(b), a, 1, 0, 0, 1]
    );
    // LSHL: signed amount, negative = logical right shift (llvm-aie lowers `a>>n` to `lshl a, -n`)
    for (n, expect) in [
        (0i32, a),
        (4, a << 4),
        (31, a << 31),
        (-1, a >> 1),
        (-4, a >> 4),
        (-31, a >> 31),
    ] {
        let t = run(
            Asm::new().op("LSHL", &[("d0", "r10"), ("s0", "r1"), ("s1", "r2")]),
            |t| {
                t.r[1] = a;
                t.r[2] = n as u32;
            },
        );
        assert_eq!(t.r[10], expect, "lshl by {n}");
    }
    for n in [32i32, -32, 100, i32::MIN] {
        let e = run_err(
            Asm::new().op("LSHL", &[("d0", "r10"), ("s0", "r1"), ("s1", "r2")]),
            |t| {
                t.r[1] = a;
                t.r[2] = n as u32;
            },
        );
        assert_eq!(e, Error::Unsupported, "lshl by {n}");
    }
}
#[test]
fn movx_reaches_ups_and_unpack_control_registers() {
    let t = run(
        movx(movx(Asm::new(), "crUPSMode", 1), "crUnpackSize", 1),
        |_| {},
    );
    assert_eq!((t.cr_ups_mode, t.cr_unpack_size), (1, 1));
    let t = run(
        Asm::new().op("MOVX_mvx_cr_r", &[("dst", "crUPSMode"), ("src", "r4")]),
        |t| t.r[4] = 1,
    );
    assert_eq!((t.cr_ups_mode, t.cr_unpack_size), (1, 0));
    let t = run(Asm::new().op("NOPS", &[]), |_| {});
    assert_eq!((t.cr_ups_mode, t.cr_unpack_size), (0, 0), "reset value");
}

// ------------------------------------------------------------------ timing (itinerary latencies)
#[test]
fn consumer_reading_before_itinerary_latency_sees_old_value() {
    let mut r = Rng(1);
    let (xa, xb) = (vec16(&corner16(&mut r)), vec16(&corner16(&mut r)));
    let old = random_dm(&mut r);
    let c = conf(true, true, false, false);
    // VMUL dst: cycle 6 (VEC bypass); VNEG acc1: cycle 4 (VEC bypass) => consumer must be >= 2 bundles later.
    for (gap, fresh) in [(0usize, false), (1, true)] {
        let asm = Asm::new()
            .op_gap(
                "VMUL_vmul_cm_core_X_X",
                &[("dst", "dm0"), ("s1", "x0"), ("s2", "x2"), ("acc", "r1")],
                gap,
            )
            .op("VNEG", &[("dst", "dm1"), ("acc1", "dm0"), ("acc", "r2")]);
        let t = run(asm, |t| {
            t.x[0] = xa;
            t.x[2] = xb;
            t.r[1] = c;
            t.r[2] = 2;
            t.dm[0] = old;
        });
        let seen = if fresh {
            mac_model(&[0; 64], None, &xa, &xb, c)
        } else {
            old
        };
        assert_eq!(
            t.dm[1],
            dm_of_acc64(&acc64_lanes(&seen).map(|v| (-i128::from(v)) as i64)),
            "gap {gap}"
        );
        assert_eq!(
            t.dm[0],
            mac_model(&[0; 64], None, &xa, &xb, c),
            "the product still lands"
        );
    }
    // VSRS dst: cycle 4, no bypass; a following vst.x (src cycle 1) needs a 4th bundle.
    let dm = srs_data(&mut r, 3, false);
    for (gap, fresh) in [(2usize, false), (3, true)] {
        let asm = Asm::new()
            .op_gap(
                "VSRS_4x_mv_x_srs_dm_srsSign1",
                &[("dst", "x4"), ("src", "dm1"), ("su", "s0")],
                gap,
            )
            .op(
                "VST_dmx_sts_x_pstm_nrm_imm",
                &[("src", "x4"), ("ptr", "p0"), ("imm", "1")],
            );
        let t = run(asm, |t| {
            t.dm[1] = dm;
            t.s[0] = 3;
            t.p[0] = MEM + 0x40;
            t.cr_sat = 1;
            t.x[4] = [0x77; 64];
        });
        let expect = if fresh {
            srs_expected(&dm, None, 0, 3, 0, 1)
        } else {
            [0x77; 64]
        };
        assert_eq!(mem64(&t, 0x40), expect, "gap {gap}");
    }
    // MV bypass: vadd d (cycle 2, MV) feeds a next-bundle vadd s1 (cycle 1, MV) but not a next-bundle vst (no bypass).
    let (a, b) = (random_vec(&mut r), random_vec(&mut r));
    let t = run(
        Asm::new()
            .op_gap("VADD_16", &[("d", "x4"), ("s1", "x0"), ("s2", "x2")], 0)
            .op("VADD_16", &[("d", "x6"), ("s1", "x4"), ("s2", "x0")]),
        |t| {
            t.x[0] = a;
            t.x[2] = b;
            t.x[4] = [0; 64];
        },
    );
    let sum = |p: &[u8; 64], q: &[u8; 64]| {
        let (p, q) = (l16(p), l16(q));
        vec16(&std::array::from_fn(|i| p[i].wrapping_add(q[i])))
    };
    assert_eq!(
        t.x[6],
        sum(&sum(&a, &b), &a),
        "bypassed result is visible one bundle later"
    );
    let t = run(
        Asm::new()
            .op_gap("VADD_16", &[("d", "x4"), ("s1", "x0"), ("s2", "x2")], 0)
            .op(
                "VST_dmx_sts_x_pstm_nrm_imm",
                &[("src", "x4"), ("ptr", "p0"), ("imm", "1")],
            ),
        |t| {
            t.x[0] = a;
            t.x[2] = b;
            t.x[4] = [0x33; 64];
            t.p[0] = MEM + 0x40;
        },
    );
    assert_eq!(
        mem64(&t, 0x40),
        [0x33; 64],
        "no bypass into the store: old x4"
    );
    // unpack dst latency 7: VUNPACK then a VST after 6 vs 7 bundles
    let src = nibble_pattern();
    for (gap, fresh) in [(5usize, false), (6, true)] {
        let asm = Asm::new()
            .op_gap(
                "VUNPACK_mv_unpack_x_unpackSign1",
                &[("dst", "y1"), ("src", "x0")],
                gap,
            )
            .op(
                "VST_dmx_sts_x_pstm_nrm_imm",
                &[("src", "x2"), ("ptr", "p0"), ("imm", "1")],
            );
        let t = run(asm, |t| {
            t.x[0] = src;
            t.x[2] = [0x44; 64];
            t.p[0] = MEM + 0x40;
        });
        let full = unpack_model(&src, 0, true);
        let expect: [u8; 64] = if fresh {
            full[..64].try_into().unwrap()
        } else {
            [0x44; 64]
        };
        assert_eq!(mem64(&t, 0x40), expect, "unpack gap {gap}");
    }
}
