// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU (wave32 interpreter) execution of the PM canonical parallel target-row sampler against a
//! Rust model of the HIP twin law (`kernels/src/spec_sampling_cpu_ar_parallel.hip`). The model
//! evaluates `expf` with `f32::exp`, so probabilities agree to a few ulps and every integer
//! output (counts, token ids) must agree exactly; the hardware gate owns the bit-exact `expf`.
use peacemaker_ir::{
    emu::{Emulator, Launch, Memory},
    inst::Frontend,
};
use peacemaker_isa::{kernels::spec_sampling_rows_pm as rows, Arch};

const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];
const NEG_INF: f32 = f32::NEG_INFINITY;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[derive(Clone, Copy, Debug)]
struct Policy {
    pool: i32,
    cap: i32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
}

#[derive(Debug, PartialEq)]
struct Row {
    count: u32,
    ids: [u32; 64],
    probs: [f32; 64],
}

fn before(a: (f32, u32), b: (f32, u32)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// The HIP twin's law (canonical selection, fixed trees, serial nucleus).
fn model(l: &[f32], p: Policy) -> Row {
    let pool = p.pool.clamp(1, 64) as usize;
    let mut recs: Vec<(f32, u32)> = l
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (*v, i as u32))
        .collect();
    recs.sort_by(|a, b| {
        if before(*a, *b) {
            std::cmp::Ordering::Less
        } else if before(*b, *a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let top = recs.first().copied();
    let max = top.map_or(NEG_INF, |t| t.0);
    let inv = 1.0f32 / p.temperature;
    let mut w = [0f32; 64];
    let mut id = [0u32; 64];
    for i in 0..pool {
        if let Some(&(lg, tok)) = recs.get(i) {
            id[i] = tok;
            let e = ((lg - max) * inv).exp();
            w[i] = if e.is_finite() { e } else { 0.0 };
        } else {
            id[i] = u32::MAX;
        }
    }
    let mut leaves = w;
    let mut s = 32;
    while s >= 1 {
        for t in 0..s {
            leaves[t] = leaves[t] + leaves[t + s];
        }
        s /= 2;
    }
    let mut sum = leaves[0];
    let bad = sum <= 0.0 || !sum.is_finite();
    let mut out = Row {
        count: 0,
        ids: [0; 64],
        probs: [0.0; 64],
    };
    if bad {
        out.count = 1;
        out.ids[0] = top.map_or(0, |t| t.1);
        out.probs[0] = 1.0;
        return out;
    }
    // Stable insertion sort, descending weight.
    for i in 1..pool {
        let mut j = i;
        while j > 0 && w[j] > w[j - 1] {
            w.swap(j, j - 1);
            id.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut cap = p.cap.clamp(1, pool as i32) as usize;
    if p.min_p > 0.0 {
        let floor = p.min_p * w[0];
        for i in 1..cap {
            if w[i] < floor {
                cap = i;
                break;
            }
        }
    }
    if cap < pool {
        let mut leaves = [0f32; 64];
        for t in 0..cap {
            leaves[t] = w[t];
        }
        let mut s = 32;
        while s >= 1 {
            for t in 0..s {
                leaves[t] = leaves[t] + leaves[t + s];
            }
            s /= 2;
        }
        sum = leaves[0];
    }
    let threshold = p.top_p * sum;
    let mut cum = 0f32;
    let mut boundary = None;
    for i in 0..cap {
        cum = cum + w[i];
        if cum >= threshold {
            boundary = Some(i + 1);
            break;
        }
    }
    let n;
    match boundary {
        Some(b) => {
            n = b;
            for i in 0..n {
                w[i] = w[i] / cum;
            }
        }
        None => {
            n = cap;
            for i in 0..n {
                w[i] = w[i] / sum;
            }
            let corr = (1.0f32 + -(cum / sum)).max(0.0);
            w[0] = w[0] + corr;
        }
    }
    let mut m = 0;
    for i in 0..n {
        if w[i] > 0.0 {
            w[m] = w[i];
            id[m] = id[i];
            m += 1;
        }
    }
    out.count = m as u32;
    for i in 0..m {
        out.ids[i] = id[i];
        out.probs[i] = w[i];
    }
    out
}

fn run_kernel(arch: Arch, logits: &[f32], vocab: usize, rows_n: usize, p: Policy) -> Vec<Row> {
    let text = rows::emit(arch).unwrap().s_text;
    let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emulator = Emulator::new(&program, &rows::symbol(arch)).unwrap();
    let (lp, cp, ip, pp, kp) = (
        0x10_0000u64,
        0x4000_0000u64,
        0x4010_0000u64,
        0x4020_0000u64,
        0x4030_0000u64,
    );
    let mut mem = Memory::default();
    mem.map(
        "logits".into(),
        lp,
        logits.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map("counts".into(), cp, vec![0xAA; rows_n * 4])
        .unwrap();
    mem.map("ids".into(), ip, vec![0xAA; rows_n * 256]).unwrap();
    mem.map("probs".into(), pp, vec![0xAA; rows_n * 256])
        .unwrap();
    let mut kern = Vec::new();
    for ptr in [lp, cp, ip, pp] {
        kern.extend_from_slice(&ptr.to_le_bytes());
    }
    kern.extend_from_slice(&(vocab as i32).to_le_bytes());
    kern.extend_from_slice(&p.pool.to_le_bytes());
    kern.extend_from_slice(&p.cap.to_le_bytes());
    for f in [p.temperature, p.top_p, p.min_p] {
        kern.extend_from_slice(&f.to_le_bytes());
    }
    kern.resize(64, 0);
    mem.map("kernarg".into(), kp, kern).unwrap();
    for row in 0..rows_n {
        emulator
            .run_workgroup(
                &Launch {
                    kernarg_ptr: kp,
                    workgroup: [row as u32, 0, 0],
                    block: [256, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{arch:?} row {row}: {e}"));
    }
    let region = |name: &str| {
        mem.regions
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .bytes
            .clone()
    };
    let (c, i, f) = (region("counts"), region("ids"), region("probs"));
    (0..rows_n)
        .map(|r| {
            let mut row = Row {
                count: u32::from_le_bytes(c[4 * r..4 * r + 4].try_into().unwrap()),
                ids: [0; 64],
                probs: [0.0; 64],
            };
            for j in 0..64 {
                let o = r * 256 + 4 * j;
                row.ids[j] = u32::from_le_bytes(i[o..o + 4].try_into().unwrap());
                row.probs[j] = f32::from_le_bytes(f[o..o + 4].try_into().unwrap());
            }
            row
        })
        .collect()
}

fn check(arch: Arch, name: &str, logits: &[f32], vocab: usize, p: Policy) {
    let n = logits.len() / vocab;
    let got = run_kernel(arch, logits, vocab, n, p);
    for (r, got) in got.iter().enumerate() {
        let want = model(&logits[r * vocab..(r + 1) * vocab], p);
        assert_eq!(
            got.count, want.count,
            "{arch:?} {name} row {r} {p:?}: count\n got {got:?}\nwant {want:?}"
        );
        assert_eq!(
            got.ids, want.ids,
            "{arch:?} {name} row {r} {p:?}: ids\n got {got:?}\nwant {want:?}"
        );
        for j in 0..64 {
            let (a, b) = (got.probs[j], want.probs[j]);
            assert!(
                a == b || (a - b).abs() <= 4e-6 * b.abs().max(1e-30),
                "{arch:?} {name} row {r} {p:?}: prob[{j}] {a} vs {b}"
            );
        }
    }
}

fn random_row(rng: &mut Rng, vocab: usize) -> Vec<f32> {
    (0..vocab).map(|_| (rng.unit() - 0.5) * 24.0).collect()
}

#[test]
fn emulated_rows_match_the_hip_twin_law() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for arch in ARCHES {
        // Plain rows, several pools; vocab not a tile multiple.
        let vocab = 1500;
        let mut logits = Vec::new();
        for _ in 0..2 {
            logits.extend(random_row(&mut rng, vocab));
        }
        for pool in [1, 7, 32, 33, 64, 100] {
            check(
                arch,
                "plain",
                &logits,
                vocab,
                Policy {
                    pool,
                    cap: 64,
                    temperature: 0.8,
                    top_p: 0.95,
                    min_p: 0.0,
                },
            );
        }
        check(
            arch,
            "min_p",
            &logits,
            vocab,
            Policy {
                pool: 64,
                cap: 64,
                temperature: 1.0,
                top_p: 1.0,
                min_p: 0.2,
            },
        );
        check(
            arch,
            "cap",
            &logits,
            vocab,
            Policy {
                pool: 64,
                cap: 5,
                temperature: 1.3,
                top_p: 0.999,
                min_p: 0.0,
            },
        );
        check(
            arch,
            "cap0",
            &logits,
            vocab,
            Policy {
                pool: 40,
                cap: 0,
                temperature: 1.0,
                top_p: 0.5,
                min_p: 0.0,
            },
        );
        check(
            arch,
            "top_p_tiny",
            &logits,
            vocab,
            Policy {
                pool: 64,
                cap: 64,
                temperature: 1.0,
                top_p: 1e-9,
                min_p: 0.0,
            },
        );
        check(
            arch,
            "nucleus_b",
            &logits,
            vocab,
            Policy {
                pool: 64,
                cap: 64,
                temperature: 6.0,
                top_p: 0.97,
                min_p: 0.0,
            },
        );
        check(
            arch,
            "cold",
            &logits,
            vocab,
            Policy {
                pool: 64,
                cap: 64,
                temperature: 0.05,
                top_p: 0.9,
                min_p: 0.0,
            },
        );
    }
}

#[test]
fn emulated_rows_canonical_ties_and_non_finite_logits() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for arch in ARCHES {
        let vocab = 2100;
        // Heavy ties, +0 and -0, and non-finite logits mixed in.
        let mut a: Vec<f32> = (0..vocab)
            .map(|_| (rng.below(9) as f32 - 4.0) * 0.5)
            .collect();
        for i in [3usize, 500, 1500, 2099] {
            a[i] = 0.0;
            a[i + 1 - 2 * (i % 2)] = -0.0;
        }
        a[7] = f32::NAN;
        a[8] = f32::INFINITY;
        a[9] = f32::NEG_INFINITY;
        a[1030] = f32::NAN;
        a[2000] = f32::INFINITY;
        // All logits equal, then all non-finite, then a single finite token.
        let b = vec![1.25f32; vocab];
        let c = vec![f32::NAN; vocab];
        let mut d = vec![f32::NEG_INFINITY; vocab];
        d[1234] = 3.0;
        let mut all = a;
        all.extend(b);
        all.extend(c);
        all.extend(d);
        for pool in [1, 12, 64] {
            check(
                arch,
                "ties",
                &all,
                vocab,
                Policy {
                    pool,
                    cap: 64,
                    temperature: 1.0,
                    top_p: 0.9,
                    min_p: 0.05,
                },
            );
            check(
                arch,
                "ties_full",
                &all,
                vocab,
                Policy {
                    pool,
                    cap: 64,
                    temperature: 2.0,
                    top_p: 1.0,
                    min_p: 0.0,
                },
            );
        }
    }
}

#[test]
fn emulated_rows_short_vocab() {
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    for arch in ARCHES {
        for vocab in [1usize, 5, 64, 255, 1024, 1025] {
            let logits = random_row(&mut rng, vocab);
            check(
                arch,
                "short",
                &logits,
                vocab,
                Policy {
                    pool: 64,
                    cap: 64,
                    temperature: 0.7,
                    top_p: 0.9,
                    min_p: 0.0,
                },
            );
        }
    }
}

/// Negative temperature reverses the weight order (the stable sort restores it), zero temperature
/// makes `1 / temperature` infinite. Not part of the interpreter's qualified `v_div_fmas` domain
/// when the scaled path is taken, so only the cases it models are asserted.
#[test]
fn emulated_rows_negative_temperature() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for arch in ARCHES {
        let logits = random_row(&mut rng, 700);
        check(
            arch,
            "negative_temperature",
            &logits,
            700,
            Policy {
                pool: 20,
                cap: 64,
                temperature: -1.0,
                top_p: 0.9,
                min_p: 0.0,
            },
        );
    }
}
