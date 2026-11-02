// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The production gfx1201 device-accept kernel (certified builder module
//! `dflash_sampled_accept_pm_gfx1201`) on the PeaceMaker CPU interpreter,
//! compared decision word for decision word with the bit-exact host twin
//! `dflash_sampled_accept_reference` (the retained hipcc kernel's twin). No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::{kernels::dflash_accept, Arch};
use rdna_compute::dflash_sampled_accept::{
    dflash_accept_pack_payload, dflash_sampled_accept_reference, DflashDeviceDecision,
};
use std::sync::LazyLock;

const ARCH: Arch = Arch::Gfx1201;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xff_ffff) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

static PROGRAM: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| {
    let text = dflash_accept::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
    .program
});

fn run(
    probs: &[f32],
    tau: &[f32],
    z: &[f32],
    payload: &[u32],
    n: usize,
    vocab: usize,
) -> DflashDeviceDecision {
    let emu = Emulator::new(&PROGRAM, &dflash_accept::symbol(ARCH)).unwrap();
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let (kern, a_probs, a_tau, a_z, a_pay, a_out) = (
        0x1_0000u64,
        0x100_0000u64,
        0x200_0000u64,
        0x300_0000u64,
        0x400_0000u64,
        0x500_0000u64,
    );
    let mut kernarg = Vec::new();
    for p in [a_probs, a_tau, a_z, a_pay, a_out] {
        kernarg.extend(p.to_le_bytes());
    }
    kernarg.extend((n as u32).to_le_bytes());
    kernarg.extend((vocab as u32).to_le_bytes());
    assert_eq!(kernarg.len(), 48);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kern, kernarg).unwrap();
    mem.map("probs".into(), a_probs, bytes(probs)).unwrap();
    mem.map("tau".into(), a_tau, bytes(tau)).unwrap();
    mem.map("z".into(), a_z, bytes(z)).unwrap();
    mem.map(
        "payload".into(),
        a_pay,
        payload.iter().flat_map(|w| w.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map("out".into(), a_out, vec![0xa5; 16]).unwrap();
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: kern,
            workgroup: [0, 0, 0],
            block: [256, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )
    .unwrap();
    let out = &mem.regions.iter().find(|r| r.name == "out").unwrap().bytes;
    let w = |i: usize| i32::from_le_bytes(out[4 * i..4 * i + 4].try_into().unwrap());
    DflashDeviceDecision {
        accept_len: w(0) as usize,
        bonus: w(1) as u32,
        rejected: w(2) != 0,
        fallback: w(3) as u32,
    }
}

struct Rows {
    probs: Vec<f32>,
    tau: Vec<f32>,
    z: Vec<f32>,
}

/// `n_rows` softmax-like rows with a top-`keep` nucleus (`keep == 0`: untruncated, tau 0).
fn rows(vocab: usize, n_rows: usize, keep: usize, rng: &mut Lcg) -> Rows {
    let (mut probs, mut tau, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n_rows {
        let mut r: Vec<f32> = (0..vocab).map(|_| rng.unit().powi(8) + 1e-7).collect();
        let s: f32 = r.iter().sum();
        r.iter_mut().for_each(|v| *v /= s);
        if keep == 0 {
            tau.push(0.0);
            z.push(1.0);
        } else {
            let mut sorted = r.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let t = sorted[keep.min(vocab) - 1];
            tau.push(t);
            z.push(r.iter().filter(|&&p| p >= t).sum());
        }
        probs.extend(r);
    }
    Rows { probs, tau, z }
}

/// Sparse selector-shaped q: top ranks of the row plus random ids, weights summing to 1.
fn selector_q(
    rows: &Rows,
    vocab: usize,
    n: usize,
    support: usize,
    rng: &mut Lcg,
) -> Vec<Vec<(u32, f32)>> {
    (0..n)
        .map(|r| {
            let row = &rows.probs[r * vocab..(r + 1) * vocab];
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let mut ids: Vec<u32> = order[..(support / 2).max(1).min(vocab)]
                .iter()
                .map(|&i| i as u32)
                .collect();
            ids.extend((0..support / 2).map(|_| rng.below(vocab) as u32));
            ids.sort_unstable();
            ids.dedup();
            let w: Vec<f32> = ids.iter().map(|_| rng.unit() + 0.05).collect();
            let s: f32 = w.iter().sum();
            ids.into_iter().zip(w).map(|(i, w)| (i, w / s)).collect()
        })
        .collect()
}

fn draw(q: &[(u32, f32)], u: f32) -> (u32, f32) {
    let mut acc = 0.0;
    for &(i, v) in q {
        acc += v;
        if u < acc {
            return (i, v);
        }
    }
    *q.last().unwrap()
}

fn pack(
    tokens: &[u32],
    us: &[f32],
    u_fin: f32,
    max_accept: usize,
    selq: &[f32],
    q: &[Vec<(u32, f32)>],
) -> Vec<u32> {
    let mut off = vec![0u32];
    let (mut ids, mut vals) = (Vec::new(), Vec::new());
    for row in q {
        for &(i, v) in row {
            ids.push(i);
            vals.push(v);
        }
        off.push(ids.len() as u32);
    }
    let mut out = Vec::new();
    dflash_accept_pack_payload(
        &mut out, tokens, us, u_fin, max_accept, selq, &off, &ids, &vals,
    )
    .unwrap();
    out
}

/// `trials` windows of `n` rows: the interpreter's decision must equal the twin's.
fn equals_twin(
    name: &str,
    vocab: usize,
    n: usize,
    rows: &Rows,
    trials: usize,
    mut make_q: impl FnMut(&mut Lcg) -> Vec<Vec<(u32, f32)>>,
    max_accept: impl Fn(usize) -> usize,
) -> Vec<DflashDeviceDecision> {
    let mut rng = Lcg(0xD5A ^ (vocab as u64) << 8 ^ n as u64);
    let mut decisions = Vec::new();
    for k in 0..trials {
        let q = make_q(&mut rng);
        let (mut tokens, mut selq) = (Vec::new(), Vec::new());
        for row in &q {
            let (t, v) = draw(row, rng.unit());
            tokens.push(t);
            selq.push(v);
        }
        let us: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
        let payload = pack(&tokens, &us, rng.unit(), max_accept(k), &selq, &q);
        let dev = run(&rows.probs, &rows.tau, &rows.z, &payload, n, vocab);
        let twin =
            dflash_sampled_accept_reference(&rows.probs, &rows.tau, &rows.z, &payload, n, vocab);
        assert_eq!(dev, twin, "{name} #{k}: builder kernel != twin");
        decisions.push(dev);
    }
    let rej = decisions.iter().filter(|d| d.rejected).count();
    println!(
        "{name}: {trials} windows, {rej} rejected, fallbacks {:?}",
        decisions
            .iter()
            .map(|d| d.fallback)
            .collect::<std::collections::BTreeSet<_>>()
    );
    decisions
}

#[test]
fn nucleus_rows_equal_the_twin() {
    for (vocab, n, keep, support) in [
        (4096usize, 4usize, 64usize, 8usize),
        (5000, 5, 40, 12),
        (300, 3, 16, 6),
        (257, 2, 8, 4),
        (1024, 1, 32, 8),
        (4096, 4, 0, 8),
    ] {
        let mut rng = Lcg(7 + vocab as u64);
        let r = rows(vocab, n + 1, keep, &mut rng);
        let name = format!("v{vocab} n{n} keep{keep}");
        let d = equals_twin(
            &name,
            vocab,
            n,
            &r,
            40,
            |rng| selector_q(&r, vocab, n, support, rng),
            |k| [usize::MAX, 2, 0][k % 3].min(n),
        );
        assert!(
            d.iter().any(|d| d.rejected),
            "{name}: no residual window exercised"
        );
        assert!(
            d.iter().any(|d| !d.rejected),
            "{name}: no accepted window exercised"
        );
    }
}

/// q puts more mass than p on the proposed id (a point mass there): `p - q <= 0` everywhere.
#[test]
fn degenerate_residual_takes_the_argmax() {
    let (vocab, n) = (1000usize, 3usize);
    let mut probs = vec![0f32; (n + 1) * vocab];
    for r in 0..=n {
        probs[r * vocab + 7 + r] = 1.0;
    }
    let r = Rows {
        probs,
        tau: vec![1.0; n + 1],
        z: vec![1.0; n + 1],
    };
    let d = equals_twin(
        "degenerate",
        vocab,
        n,
        &r,
        30,
        |_| (0..n).map(|r| vec![((7 + r) as u32, 2.0)]).collect(),
        |_| n,
    );
    for w in d.iter().filter(|w| w.rejected) {
        assert_eq!((w.fallback, w.bonus as usize), (1, 7 + w.accept_len));
    }
    assert!(d.iter().any(|w| w.rejected));
}

/// A run of equal probabilities at the kept boundary.
#[test]
fn tie_rows_equal_the_twin() {
    let (vocab, n) = (4096usize, 3usize);
    let mut probs = vec![1e-7f32; (n + 1) * vocab];
    for r in 0..=n {
        for j in 0..40 {
            probs[r * vocab + 90 * j + r] = 0.025;
        }
    }
    let rows = Rows {
        probs,
        tau: vec![0.025; n + 1],
        z: vec![1.0; n + 1],
    };
    let mut rng0 = Lcg(11);
    let q = selector_q(&rows, vocab, n, 8, &mut rng0);
    equals_twin(
        "tie_rows",
        vocab,
        n,
        &rows,
        40,
        |rng| selector_q(&rows, vocab, n, 8, rng),
        |_| n,
    );
    let _ = q;
}

/// Every instruction of the module has interpreter semantics.
#[test]
fn interpreter_covers_the_module() {
    let mut missing = std::collections::BTreeSet::new();
    for k in &PROGRAM.kernels {
        for id in &k.body.layout {
            let i = k.body.insts.get(*id).unwrap();
            let row = peacemaker_ir::isa::lookup(PROGRAM.target.arch, i.op, i.form).unwrap();
            if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                missing.insert(row.name);
            }
        }
    }
    assert!(missing.is_empty(), "no interpreter semantics: {missing:?}");
}

/// More than 256 rows: the accept tests take a second trip, and the first
/// rejection lies on it (or nowhere).
#[test]
fn long_window_crosses_accept_trips() {
    let (vocab, n) = (64usize, 300usize);
    let mut rng = Lcg(31);
    let r = rows(vocab, n + 1, 8, &mut rng);
    let q = selector_q(&r, vocab, n, 4, &mut rng);
    // Every row accepts (u = 0) except `bad`, whose u is near 1 with a tiny p at its token.
    for bad in [None, Some(270usize), Some(255), Some(256), Some(299)] {
        let (mut tokens, mut selq) = (Vec::new(), Vec::new());
        for row in &q {
            let (t, v) = draw(row, 0.5);
            tokens.push(t);
            selq.push(v);
        }
        let mut us = vec![0f32; n];
        if let Some(b) = bad {
            let (_, low) = r.probs[b * vocab..(b + 1) * vocab]
                .iter()
                .enumerate()
                .fold((0, f32::MAX), |a, (i, &p)| if p < a.1 { (i, p) } else { a });
            let id = r.probs[b * vocab..(b + 1) * vocab]
                .iter()
                .position(|&p| p == low)
                .unwrap() as u32;
            tokens[b] = id;
            selq[b] = 1.0;
            us[b] = 0.999;
        }
        let payload = pack(&tokens, &us, 0.37, usize::MAX, &selq, &q);
        let dev = run(&r.probs, &r.tau, &r.z, &payload, n, vocab);
        let twin = dflash_sampled_accept_reference(&r.probs, &r.tau, &r.z, &payload, n, vocab);
        assert_eq!(dev, twin, "bad row {bad:?}");
        if let Some(b) = bad {
            assert_eq!((dev.accept_len, dev.rejected), (b, true), "bad row {b}");
        } else {
            assert_eq!((dev.accept_len, dev.rejected), (n, false));
        }
    }
}

/// A NaN final uniform never lands in any CDF interval: the last support id (code 2).
#[test]
fn fall_off_the_end_takes_the_last_support_id() {
    let (vocab, n) = (4096usize, 3usize);
    let mut rng = Lcg(5);
    let r = rows(vocab, n + 1, 40, &mut rng);
    let q = selector_q(&r, vocab, n, 8, &mut rng);
    let mut seen = std::collections::BTreeSet::new();
    for k in 0..24 {
        let (mut tokens, mut selq) = (Vec::new(), Vec::new());
        for row in &q {
            let (t, v) = draw(row, rng.unit());
            tokens.push(t);
            selq.push(v);
        }
        let us: Vec<f32> = (0..n)
            .map(|_| if k % 2 == 0 { 0.0 } else { rng.unit() })
            .collect();
        let payload = pack(&tokens, &us, f32::NAN, usize::MAX, &selq, &q);
        let dev = run(&r.probs, &r.tau, &r.z, &payload, n, vocab);
        let twin = dflash_sampled_accept_reference(&r.probs, &r.tau, &r.z, &payload, n, vocab);
        assert_eq!(dev, twin, "#{k}");
        assert_eq!(dev.fallback, 2, "#{k}");
        seen.insert(dev.rejected);
    }
    assert_eq!(
        seen.len(),
        2,
        "both the target-row and the residual fall-off"
    );
}

/// A real vocabulary: eight chunks per lane.
#[test]
fn vocab_32768_rows_equal_the_twin() {
    let (vocab, n) = (32768usize, 4usize);
    let mut rng = Lcg(77);
    let r = rows(vocab, n + 1, 40, &mut rng);
    let d = equals_twin(
        "v32768",
        vocab,
        n,
        &r,
        6,
        |rng| selector_q(&r, vocab, n, 8, rng),
        |_| n,
    );
    assert!(d.iter().any(|d| d.rejected));
}
