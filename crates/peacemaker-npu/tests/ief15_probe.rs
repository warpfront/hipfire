// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! IEF15 N0 semantics probe (`peacemaker_npu::kernels::iu4_ief15_probe`, design doc §4/§7 step 1): the exact probe PDI + TXN
//! runs in `config::Config`; (a) it completes, (b) every case's bytes equal an independent lane model written in this
//! file (i128 / Vec arithmetic, no simulator helpers: catches a wrong simulator), (c) the expected-bytes blob is
//! deterministic. `IEF15_PROBE_DUMP=<path>` writes the blob (the `npu-gemm --ief15-probe` expected bytes).
use peacemaker_npu::kernels::iu4_ief15_probe::{self as probe, Case, Op, CASES, NONE, OUT_BYTES};
use peacemaker_npu::sim::config::Config;

fn run_probe() -> Vec<u8> {
    let d = probe::design(0);
    let mut args = vec![probe::input(), vec![0xa5; OUT_BYTES]];
    let mut sim = Config::from_pdi(&d.pdi).unwrap();
    sim.tick_limit = 50_000_000;
    sim.submit(&d.insts, &mut args).unwrap();
    args.remove(1)
}

// ---- independent model ------------------------------------------------------------------------------------------
struct Model {
    input: Vec<u8>,
    horner: Vec<i64>,
}
fn l16(b: &[u8]) -> Vec<u16> {
    b.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn l32(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn l64(b: &[u8]) -> Vec<i64> {
    b.chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn b64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn b32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn b16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `v >> s` with floor (crRnd 0) or round-half-even (crRnd 12) on the exact integer, then wrap / saturate to `bits`.
fn srs_lane(v: i128, s: u32, rnd: i32, sat: i32, bits: u32) -> i128 {
    let q = v >> s; // floor
    let r = v - (q << s);
    let half = if s == 0 { 0 } else { 1i128 << (s - 1) };
    let rounded = match rnd {
        0 => q,
        12 => {
            if s == 0 {
                q
            } else if r > half {
                q + 1
            } else if r < half {
                q
            } else if q & 1 == 1 {
                q + 1
            } else {
                q
            }
        }
        _ => panic!("probe uses crRnd 0 / 12 only"),
    };
    let (lo, hi) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
    match sat {
        0 => {
            let m = rounded & ((1i128 << bits) - 1);
            if m > hi {
                m - (1i128 << bits)
            } else {
                m
            }
        }
        1 => rounded.clamp(lo, hi),
        _ => panic!("probe uses crSat 0 / 1 only"),
    }
}
fn pack_lanes(vals: &[i128], bits: u32) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| (0..bits / 8).map(move |b| ((*v >> (8 * b)) & 0xff) as u8))
        .collect()
}
fn mask_bits(preds: impl Iterator<Item = bool>) -> u32 {
    preds.enumerate().fold(0, |m, (i, p)| m | ((p as u32) << i))
}

impl Model {
    fn vec(&self, v: u16) -> &[u8] {
        &self.input[v as usize * 64..v as usize * 64 + 64]
    }
    fn acc(&self, v: u16) -> &[u8] {
        &self.input[v as usize * 64..v as usize * 64 + 256]
    }
    fn elem(
        &self,
        c: &Case,
        a: &[u16],
        b: &[u16],
        acc1: Option<Vec<i64>>,
        acc2: Option<Vec<i64>>,
    ) -> Vec<i64> {
        let conf = c.p[0];
        let (sa, sb) = (conf >> 9 & 1 == 1, conf >> 8 & 1 == 1);
        (0..32)
            .map(|i| {
                let x = if sa { a[i] as i16 as i64 } else { a[i] as i64 };
                let y = if sb { b[i] as i16 as i64 } else { b[i] as i64 };
                let p = x.wrapping_mul(y);
                let sh = |v: i64| {
                    if conf & 1 == 1 {
                        0
                    } else if conf >> 10 & 1 == 1 {
                        v.wrapping_shl(16)
                    } else {
                        v
                    }
                };
                match c.op {
                    Op::Vmul | Op::HornerMul => p,
                    Op::Vmac | Op::HornerMac => sh(acc1.as_ref().unwrap()[i]).wrapping_add(p),
                    _ => sh(acc1.as_ref().unwrap()[i])
                        .wrapping_add(acc2.as_ref().unwrap()[i])
                        .wrapping_add(p),
                }
            })
            .collect()
    }
    fn expect(&mut self, c: &Case) -> Vec<u8> {
        let s = |i: usize| self.vec(c.s[i]).to_vec();
        match c.op {
            Op::Vmul | Op::Vmac | Op::Vaddmac => {
                let (a, b) = (l16(&s(0)), l16(&s(1)));
                let acc1 = (c.s[2] != NONE).then(|| l64(self.acc(c.s[2])));
                let acc2 = (c.s[3] != NONE).then(|| l64(self.acc(c.s[3])));
                b64(&self.elem(c, &a, &b, acc1, acc2))
            }
            Op::HornerMul | Op::HornerMac => {
                let (a, b) = (l16(&s(0)), l16(&s(1)));
                let acc = if c.op == Op::HornerMul {
                    None
                } else {
                    Some(self.horner.clone())
                };
                self.horner = self.elem(c, &a, &b, acc, None);
                b64(&self.horner)
            }
            Op::Vneg => {
                let a = self.acc(c.s[2]);
                if c.p[0] == 2 {
                    b64(&l64(a).iter().map(|x| x.wrapping_neg()).collect::<Vec<_>>())
                } else {
                    b32(&l32(a)
                        .iter()
                        .map(|x| (*x as i32).wrapping_neg() as u32)
                        .collect::<Vec<_>>())
                }
            }
            Op::AccRt => self.acc(c.s[2]).to_vec(),
            Op::Srs4 | Op::Srs2 => {
                let (mode, shift, rnd, sat, half) = (
                    c.p[0],
                    c.p[1] as u32,
                    c.p[2],
                    c.p[3] & 0xff,
                    (c.p[3] >> 8) as usize,
                );
                let src = self.acc(c.s[2]);
                let (src, in_bits, out_bits) = match (c.op, mode) {
                    (Op::Srs4, 1) => (&src[..], 64, 16),
                    (Op::Srs4, _) => (&src[..], 32, 8),
                    (_, 1) => (&src[128 * half..128 * half + 128], 64, 32),
                    (_, _) => (&src[128 * half..128 * half + 128], 32, 16),
                };
                let lanes: Vec<i128> = if in_bits == 64 {
                    l64(src).into_iter().map(i128::from).collect()
                } else {
                    l32(src).into_iter().map(|x| i128::from(x as i32)).collect()
                };
                let outs: Vec<i128> = lanes
                    .iter()
                    .map(|&v| srs_lane(v, shift, rnd, sat, out_bits))
                    .collect();
                pack_lanes(&outs, out_bits)
            }
            Op::Ups4 | Op::Ups2 => {
                let (mode, signed, shift) = (c.p[0], c.p[1] == 1, c.p[2] as u32);
                let src = s(0);
                if c.op == Op::Ups4 {
                    let outs: Vec<i64> = l16(&src)
                        .iter()
                        .map(|&v| (if signed { v as i16 as i64 } else { v as i64 }) << shift)
                        .collect();
                    b64(&outs)
                } else if mode == 1 {
                    let outs: Vec<i64> = l32(&src)
                        .iter()
                        .map(|&v| (if signed { v as i32 as i64 } else { v as i64 }) << shift)
                        .collect();
                    b64(&outs)
                } else {
                    let outs: Vec<u32> = l16(&src)
                        .iter()
                        .map(|&v| {
                            (((if signed { v as i16 as i64 } else { v as i64 }) << shift) as i32)
                                as u32
                        })
                        .collect();
                    b32(&outs)
                }
            }
            Op::VldbUnpack | Op::Vunpack => {
                let src = s(0);
                (0..128)
                    .map(|n| {
                        let nib = (src[n / 2] >> (4 * (n & 1))) & 0xf;
                        if c.p[0] == 1 {
                            (((nib as i8) << 4) >> 4) as u8
                        } else {
                            nib
                        }
                    })
                    .collect()
            }
            Op::VldbX => s(0),
            Op::Shuffle => {
                let mode = c.p[0];
                let (s1, s2) = (s(0), s(1));
                let concat: Vec<u8> = s1.iter().chain(&s2).copied().collect();
                let (w, interleave) = match mode {
                    12 | 13 => (16, true),
                    14 | 15 => (8, true),
                    16 | 17 => (4, true),
                    18 | 19 => (2, true),
                    2 | 3 => (2, false),
                    4 | 5 => (4, false),
                    6 | 7 => (8, false),
                    8 | 9 => (16, false),
                    _ => panic!("mode {mode}"),
                };
                let hi = mode % 2 == 1;
                let full: Vec<u8> = if interleave {
                    let n = 64 / w;
                    (0..n)
                        .flat_map(|j| [&s1[j * w..j * w + w], &s2[j * w..j * w + w]])
                        .flatten()
                        .copied()
                        .collect()
                } else {
                    let n = 128 / w;
                    let evens: Vec<u8> = (0..n / 2)
                        .flat_map(|j| concat[2 * j * w..2 * j * w + w].to_vec())
                        .collect();
                    let odds: Vec<u8> = (0..n / 2)
                        .flat_map(|j| concat[(2 * j + 1) * w..(2 * j + 1) * w + w].to_vec())
                        .collect();
                    return if hi { odds } else { evens };
                };
                if hi {
                    full[64..].to_vec()
                } else {
                    full[..64].to_vec()
                }
            }
            Op::Vlt16 | Op::Vge16 | Op::Vlt32 | Op::Vge32 | Op::Veqz16 | Op::Veqz32 => {
                let (a, b) = (s(0), s(1));
                let signed = c.p[0] == 1;
                let wide = matches!(c.op, Op::Vlt32 | Op::Vge32 | Op::Veqz32);
                let (la, lb): (Vec<i128>, Vec<i128>) = if wide {
                    let f = |v: Vec<u32>| {
                        v.into_iter()
                            .map(|x| {
                                if signed {
                                    i128::from(x as i32)
                                } else {
                                    i128::from(x)
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    (f(l32(&a)), f(l32(&b)))
                } else {
                    let f = |v: Vec<u16>| {
                        v.into_iter()
                            .map(|x| {
                                if signed {
                                    i128::from(x as i16)
                                } else {
                                    i128::from(x)
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    (f(l16(&a)), f(l16(&b)))
                };
                let m = match c.op {
                    Op::Vlt16 | Op::Vlt32 => mask_bits(la.iter().zip(&lb).map(|(x, y)| x < y)),
                    Op::Vge16 | Op::Vge32 => mask_bits(la.iter().zip(&lb).map(|(x, y)| x >= y)),
                    _ => mask_bits(lb.iter().map(|&y| y == 0)),
                };
                m.to_le_bytes().to_vec()
            }
            Op::Vsel16 | Op::Vsel32 => {
                let (a, b) = (s(0), s(1));
                if c.op == Op::Vsel16 {
                    let (la, lb) = (l16(&a), l16(&b));
                    let m = if c.p[0] == 0 {
                        c.p[1] as u32
                    } else {
                        mask_bits(la.iter().zip(&lb).map(|(x, y)| (*x as i16) < (*y as i16)))
                    };
                    b16(&(0..32)
                        .map(|i| if m >> i & 1 == 1 { lb[i] } else { la[i] })
                        .collect::<Vec<_>>())
                } else {
                    let (la, lb) = (l32(&a), l32(&b));
                    let m = if c.p[0] == 0 {
                        c.p[1] as u32 & 0xffff
                    } else {
                        mask_bits(la.iter().zip(&lb).map(|(x, y)| (*x as i32) < (*y as i32)))
                    };
                    b32(&(0..16)
                        .map(|i| if m >> i & 1 == 1 { lb[i] } else { la[i] })
                        .collect::<Vec<_>>())
                }
            }
            Op::Vadd16 | Op::Vsub16 => {
                let (la, lb) = (l16(&s(0)), l16(&s(1)));
                b16(&la
                    .iter()
                    .zip(&lb)
                    .map(|(x, y)| {
                        if c.op == Op::Vadd16 {
                            x.wrapping_add(*y)
                        } else {
                            x.wrapping_sub(*y)
                        }
                    })
                    .collect::<Vec<_>>())
            }
            Op::Vadd32 | Op::Vsub32 => {
                let (la, lb) = (l32(&s(0)), l32(&s(1)));
                b32(&la
                    .iter()
                    .zip(&lb)
                    .map(|(x, y)| {
                        if c.op == Op::Vadd32 {
                            x.wrapping_add(*y)
                        } else {
                            x.wrapping_sub(*y)
                        }
                    })
                    .collect::<Vec<_>>())
            }
            Op::Vband => s(0).iter().zip(s(1)).map(|(x, y)| x & y).collect(),
            Op::Vbor => s(0).iter().zip(s(1)).map(|(x, y)| x | y).collect(),
            Op::Vmax16 | Op::Vmin16 | Op::Vmax32 | Op::Vmin32 => {
                let wide = matches!(c.op, Op::Vmax32 | Op::Vmin32);
                let max = matches!(c.op, Op::Vmax16 | Op::Vmax32);
                let (la, lb): (Vec<i64>, Vec<i64>) = if wide {
                    (
                        l32(&s(0)).iter().map(|&x| x as i32 as i64).collect(),
                        l32(&s(1)).iter().map(|&x| x as i32 as i64).collect(),
                    )
                } else {
                    (
                        l16(&s(0)).iter().map(|&x| x as i16 as i64).collect(),
                        l16(&s(1)).iter().map(|&x| x as i16 as i64).collect(),
                    )
                };
                // VMAX_LT: mask = s1 < s2, d = max; VMIN_GE: mask = s1 >= s2, d = min.
                let mask = mask_bits(
                    la.iter()
                        .zip(&lb)
                        .map(|(x, y)| if max { x < y } else { x >= y }),
                );
                if c.part == 1 {
                    return mask.to_le_bytes().to_vec();
                }
                let d: Vec<i128> = la
                    .iter()
                    .zip(&lb)
                    .map(|(x, y)| i128::from(if max { *x.max(y) } else { *x.min(y) }))
                    .collect();
                pack_lanes(&d, if wide { 32 } else { 16 })
            }
            Op::Vbcst16 => b16(&vec![c.p[0] as u16; 32]),
            Op::Vbcst32 => b32(&vec![c.p[0] as u32; 16]),
            Op::I8Mul | Op::I8Mac => {
                let i8s = |v: u16| {
                    self.vec(v)
                        .iter()
                        .map(|&x| x as i8 as i32)
                        .collect::<Vec<_>>()
                };
                let mm = |a: &[i32], b: &[i32]| -> Vec<i32> {
                    (0..64)
                        .map(|n| {
                            let (i, j) = (n / 8, n % 8);
                            (0..8).map(|k| a[i * 8 + k] * b[k * 8 + j]).sum()
                        })
                        .collect()
                };
                let v: Vec<Vec<i32>> = (0..4).map(|k| i8s(c.s[k])).collect();
                let dm2 = mm(&v[0], &v[1]);
                let dm = match c.p[1] {
                    0 => dm2,
                    1 => dm2
                        .iter()
                        .zip(mm(&v[2], &v[3]))
                        .map(|(x, y)| x.wrapping_add(y))
                        .collect(),
                    _ => {
                        let dm3: Vec<i32> = dm2
                            .iter()
                            .zip(mm(&v[2], &v[3]))
                            .map(|(x, y)| x.wrapping_add(y))
                            .collect();
                        dm3.iter()
                            .zip(mm(&v[0], &v[3]))
                            .map(|(x, y)| x.wrapping_add(y))
                            .collect()
                    }
                };
                b32(&dm.iter().map(|&x| x as u32).collect::<Vec<_>>())
            }
            Op::SAnd | Op::SOr | Op::SAdd | Op::SLshl | Op::SEqz | Op::SNez => {
                let (a, b) = (c.p[0] as u32, c.p[1] as u32);
                match c.op {
                    Op::SAnd => a & b,
                    Op::SOr => a | b,
                    Op::SAdd => a.wrapping_add(b),
                    Op::SLshl => a << (b & 31),
                    Op::SEqz => (a == 0) as u32,
                    _ => (a != 0) as u32,
                }
                .to_le_bytes()
                .to_vec()
            }
        }
    }
}

fn describe(c: &Case) -> String {
    format!(
        "case {} [{}] {} {}{} -> out[{}..{}]",
        c.id,
        c.group,
        c.instruction(),
        c.operands(),
        if c.part == 1 { " (part 1)" } else { "" },
        c.out_off,
        c.out_off + c.out_len as u32
    )
}

#[test]
fn probe_completes_and_every_case_matches_the_independent_lane_model() {
    let out = run_probe();
    assert_eq!(out.len(), OUT_BYTES);
    let mut model = Model {
        input: probe::input(),
        horner: vec![],
    };
    let mut bad = Vec::new();
    for c in CASES.iter() {
        let want = model.expect(c);
        assert_eq!(
            want.len(),
            c.out_len as usize,
            "model length: {}",
            describe(c)
        );
        let got = &out[c.out_off as usize..c.out_off as usize + c.out_len as usize];
        if got != &want[..] {
            let k = got.iter().zip(&want).position(|(g, w)| g != w).unwrap();
            bad.push(format!(
                "{}: first byte {k}: model {:#04x} sim {:#04x}",
                describe(c),
                want[k],
                got[k]
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} cases differ from the lane model; first 20:\n{}",
        bad.len(),
        CASES.len(),
        bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn expected_blob_is_deterministic() {
    let a = run_probe();
    let b = run_probe();
    assert!(a == b, "two simulator runs of the probe differ");
    if let Ok(path) = std::env::var("IEF15_PROBE_DUMP") {
        std::fs::write(path, &a).unwrap();
    }
}
