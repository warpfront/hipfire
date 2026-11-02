// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! IEF15 (`peacemaker_npu::kernels::iu4_ief15::Ief15Gemm`, core `iu4_ief15_core`) on the exact PDI/TXN simulator: the C stream
//! unpacks to the CPU oracle `reference` (one RNE to binary32 of `I = sum S*D*C`, exponent `a_r + b_t`) byte for byte.
//!
//! Gates (every `#[test]` runs the real core; none is ignored):
//!   * (a) smallest 256 x 256 x K128, (b) padding 300 x 700 x K640 (E = 5), (c) E = 20 / 40 / 48 / 136 at small M, N,
//!     (d) extreme `C` (+8192 and -7168 in every epoch), S / D in {0, 1, 32767}, `a` / `b` reaching subnormal results,
//!     overflow to inf, -0, exact ties at the final rounding (normal and subnormal), zero rows and columns,
//!     (e) three submits with changed operands on one context, (f) full and lean submits interleaved (bytes equal a fresh
//!     full run, retained state equal to a fresh full run), (g) the dense shapes at 256 tokens x 256 features,
//!     K = 5120 / 6144 / 17408;
//!   * `ief15_standin_core_moves_every_byte`: the data movement (descriptors, routes, TXN, locks) with the V8 `no_compute`
//!     pair program standing in for the IEF15 core: completes and moves exactly the byte counts of the layout. It does
//!     not need the IEF15 core.
use peacemaker_npu::kernels::gemm_core::{self, Control, Epilogue, PairRole, Probe};
use peacemaker_npu::kernels::iu4::{self, Acts, Weights};
use peacemaker_npu::kernels::iu4_ief15::{self as ief, Ief15Gemm};
use peacemaker_npu::sim::{config::Config, dma::Direction};

const POISON: u8 = 0xa5;

/// Native operands and the four sidecars of one problem.
struct Problem {
    tokens: usize,
    features: usize,
    k: usize,
    w: Vec<u8>,
    x: Vec<u8>,
    s: Vec<u16>,
    a: Vec<i16>,
    d: Vec<u16>,
    b: Vec<i16>,
}

impl Problem {
    fn weights(&self) -> Weights<'_> {
        Weights::new(self.features, self.k, &self.w)
    }
    fn acts(&self) -> Acts<'_> {
        Acts::new(self.tokens, self.k, &self.x)
    }
    fn reference(&self, g: &Ief15Gemm) -> Vec<u8> {
        g.reference(
            &self.weights(),
            &self.acts(),
            &self.s,
            &self.a,
            &self.d,
            &self.b,
        )
    }
    fn pack(&self, g: &Ief15Gemm) -> [Vec<u8>; 2] {
        g.pack_in(
            &self.weights(),
            &self.acts(),
            &self.s,
            &self.a,
            &self.d,
            &self.b,
        )
        .unwrap_or_else(|e| panic!("pack_in: {e}"))
    }
}

/// Random native operands (`extremes` = block 0 / features 0, 1 extreme) with the production encoders' sidecars.
fn encoded(tokens: usize, features: usize, k: usize, seed: u64, extremes: bool) -> Problem {
    let (w, x) = iu4::random_operands(tokens, features, k, seed, extremes);
    let (s, a) = ief::encode_weight_scales(&Weights::new(features, k, &w));
    let (d, b) = ief::encode_act_scales(&Acts::new(tokens, k, &x));
    Problem {
        tokens,
        features,
        k,
        w,
        x,
        s,
        a,
        d,
        b,
    }
}

/// `C = +8192` (feature 0) and `-7168` (feature 1) in every epoch; the sidecars cycle through the classes of
/// [`wild_sidecars`].
fn wild(tokens: usize, features: usize, k: usize, seed: u64) -> Problem {
    let (w, x) = ief::extreme_operands(tokens, features, k, seed);
    let (s, a, d, b) = wild_sidecars(tokens, features, k / 128);
    Problem {
        tokens,
        features,
        k,
        w,
        x,
        s,
        a,
        d,
        b,
    }
}

/// Per-feature `(S, a)` and per-token `(D, b)` classes: zero row / column, mantissa 0 / 1 / 32767, ties (`S = 32 * 1023`,
/// odd `D`, with `C = 8192` of feature 0: `I = 2^13 * S * D` has an exact half-ulp remainder; `S = 1`, odd `D`, `b = -60`,
/// `a = -100` with `C = -7168` of feature 1: an exact subnormal tie), overflow and underflow exponents up to `+-16383`.
fn wild_sidecars(
    tokens: usize,
    features: usize,
    e: usize,
) -> (Vec<u16>, Vec<i16>, Vec<u16>, Vec<i16>) {
    const FEATURE: [(u16, i16); 8] = [
        (32736, 0),
        (1, -100),
        (0, 0),
        (32767, -100),
        (2, -120),
        (255, 50),
        (1, 0),
        (32767, 16383),
    ];
    const TOKEN: [(u16, i16); 8] = [
        (32767, -60),
        (32765, 0),
        (0, 0),
        (1, -163),
        (32767, 130),
        (12345, -300),
        (7, -16383),
        (32767, 16383),
    ];
    let (s, a): (Vec<u16>, Vec<i16>) = (0..features).map(|r| FEATURE[r % 8]).unzip();
    let (dm, b): (Vec<u16>, Vec<i16>) = (0..tokens).map(|t| TOKEN[t % 8]).unzip();
    let s = s
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(e))
        .collect();
    let d = (0..e).flat_map(|_| dm.iter().copied()).collect();
    (s, a, d, b)
}

/// Run `insts` on `sim` for the packed operands and return the C argument; the packed operands must come back unchanged.
fn run_on(sim: &mut Config, g: &Ief15Gemm, insts: &[u8], [a0, a1]: [Vec<u8>; 2]) -> Vec<u8> {
    let d = g.design();
    let mut args = vec![a0.clone(), a1.clone(), vec![POISON; d.args[2].bytes]];
    sim.submit(insts, &mut args)
        .unwrap_or_else(|e| panic!("submit: {e}"));
    assert!(
        args[0] == a0 && args[1] == a1,
        "the packed operands must not change"
    );
    args.pop().unwrap()
}

/// Cheap helper: one design, a fresh simulator context, one full submit; returns the C bytes (arg2).
fn run_design(g: &Ief15Gemm, packed: [Vec<u8>; 2]) -> Vec<u8> {
    let mut sim = Config::from_pdi(&g.design().pdi).unwrap();
    run_on(&mut sim, g, &g.design().insts, packed)
}

fn first_diff(got: &[u8], want: &[u8]) -> Option<usize> {
    if got.len() != want.len() {
        return Some(got.len().min(want.len()) / 4);
    }
    (0..got.len() / 4).find(|&i| got[i * 4..i * 4 + 4] != want[i * 4..i * 4 + 4])
}

/// `unpack_out(c)` equals `want` byte for byte; panics with the first mismatching `(token, feature)` and the count.
fn assert_exact(g: &Ief15Gemm, p: &Problem, c: &[u8], want: &[u8], what: &str) {
    let got = g.unpack_out(c);
    if let Some(i) = first_diff(&got, want) {
        let bad = (0..got.len() / 4)
            .filter(|&j| got[j * 4..j * 4 + 4] != want[j * 4..j * 4 + 4])
            .count();
        let at = |v: &[u8]| u32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap());
        panic!("{what} {}x{}x{}: {bad} mismatches, first token {} feature {}: got {:#010x} want {:#010x}",
            p.tokens, p.features, p.k, i / p.features, i % p.features, at(&got), at(want));
    }
}

/// The full gate of one problem: fresh context, exact, plus the reference class summary.
fn shape(tokens: usize, features: usize, k: usize, ctl: Control, p: &Problem) -> Ief15Gemm {
    let g = Ief15Gemm::new(tokens, features, k, ctl)
        .unwrap_or_else(|e| panic!("{tokens}x{features}x{k}: {e}"));
    let want = p.reference(&g);
    let c = run_design(&g, p.pack(&g));
    assert_exact(&g, p, &c, &want, &format!("{ctl:?}"));
    println!(
        "iu4 ief15 {tokens}x{features}x{k} {ctl:?}: {} waves, {} outputs exact",
        g.design().waves(),
        want.len() / 4
    );
    g
}

// ---------------------------------------------------------------- (a) (b) (c)
/// (a) The smallest problem: one wave, one epoch, both control disciplines.
#[test]
fn ief15_256_256_128() {
    for ctl in [Control::Fast, Control::Slow] {
        shape(256, 256, 128, ctl, &encoded(256, 256, 128, 1, true));
    }
}
/// (b) Token and feature padding (300 -> 512, 700 -> 768), five epochs (odd), 6 waves.
#[test]
fn ief15_padding_300_700_640() {
    shape(
        300,
        700,
        640,
        Control::Fast,
        &encoded(300, 700, 640, 2, true),
    );
}
/// (c) Epoch counts 20 / 40 / 48 / 136 at small M, N (one wave, padded).
#[test]
fn ief15_e20_small() {
    shape(
        70,
        90,
        20 * 128,
        Control::Fast,
        &encoded(70, 90, 20 * 128, 3, true),
    );
}
#[test]
fn ief15_e40_small() {
    shape(
        70,
        90,
        40 * 128,
        Control::Fast,
        &encoded(70, 90, 40 * 128, 4, true),
    );
}
#[test]
fn ief15_e48_small() {
    shape(
        70,
        90,
        48 * 128,
        Control::Fast,
        &encoded(70, 90, 48 * 128, 5, true),
    );
}
#[test]
fn ief15_e136_small() {
    shape(
        70,
        90,
        136 * 128,
        Control::Fast,
        &encoded(70, 90, 136 * 128, 6, true),
    );
}
/// Two M waves and two N waves (wave order `w = mw * NW + nw`) at an odd epoch count.
#[test]
fn ief15_four_waves_e3() {
    shape(
        300,
        300,
        3 * 128,
        Control::Fast,
        &encoded(300, 300, 3 * 128, 7, false),
    );
}

// ---------------------------------------------------------------- (d) extremes
/// `(round_scaled tie, subnormal result)` of `I * 2^x`, independent of the packed path.
fn classify(i: i128, x: i32) -> (bool, bool) {
    if i == 0 {
        return (false, false);
    }
    let mag = i.unsigned_abs();
    let top = 127 - mag.leading_zeros() as i32;
    let res_exp = top + x;
    let shift = if res_exp >= -126 { top - 23 } else { -149 - x };
    let tie = (1..128).contains(&shift) && mag & ((1u128 << shift) - 1) == 1u128 << (shift - 1);
    (tie, res_exp < -126)
}

/// Class counts `[normal ties, subnormal ties, subnormal results, inf, -0, I = 0, extreme C partials, |a+b| > 300]` of the
/// outputs, from the integer state `I` rebuilt out of the GPU epoch partials (independent of the packed path).
fn classes(p: &Problem) -> [usize; 8] {
    let parts = iu4::partials(&p.weights(), &p.acts());
    let e = p.k / 128;
    let mut n = [0usize; 8];
    let y = p.reference_values();
    for t in 0..p.tokens {
        for r in 0..p.features {
            let i: i128 = (0..e)
                .map(|ep| {
                    i128::from(p.s[r * e + ep])
                        * i128::from(p.d[ep * p.tokens + t])
                        * i128::from(parts[(ep * p.tokens + t) * p.features + r])
                })
                .sum();
            let x = i32::from(p.a[r]) + i32::from(p.b[t]);
            let (tie, subn) = classify(i, x);
            let bits = y[t * p.features + r].to_bits();
            n[0] += usize::from(tie && !subn);
            n[1] += usize::from(tie && subn);
            n[2] += usize::from(subn && bits & 0x7fff_ffff != 0);
            n[3] += usize::from(bits & 0x7fff_ffff == 0x7f80_0000);
            n[4] += usize::from(bits == 0x8000_0000);
            n[5] += usize::from(i == 0);
            n[7] += usize::from(x.abs() > 300);
        }
    }
    n[6] = parts.iter().filter(|&&c| c == 8192 || c == -7168).count();
    n
}

impl Problem {
    fn reference_values(&self) -> Vec<f32> {
        ief::reference_values(
            &self.weights(),
            &self.acts(),
            &self.s,
            &self.a,
            &self.d,
            &self.b,
        )
    }
}

fn extreme(tokens: usize, features: usize, k: usize, ctl: Control, seed: u64) {
    let p = wild(tokens, features, k, seed);
    let g = shape(tokens, features, k, ctl, &p);
    let [normal_tie, sub_tie, sub, inf, neg_zero, zero_i, full_c, huge_x] = classes(&p);
    if k == 128 {
        assert!(
            normal_tie > 0 && sub_tie > 0,
            "tie classes present: normal {normal_tie}, subnormal {sub_tie}"
        );
    }
    assert!(sub > 0 && inf > 0 && neg_zero > 0 && zero_i > 0 && full_c > 0 && huge_x > 0,
        "classes present: subnormal {sub}, inf {inf}, -0 {neg_zero}, I = 0 {zero_i}, extreme C {full_c}, |a+b| > 300 {huge_x}");
    println!("extreme {tokens}x{features}x{k}: normal ties {normal_tie}, subnormal ties {sub_tie}, subnormal {sub}, inf {inf}, -0 {neg_zero}, I=0 {zero_i}, waves {}", g.design().waves());
}
/// One epoch: every extreme class, the tie classes included (all epochs share the sidecar classes, so ties need E = 1).
#[test]
fn ief15_extreme_classes_e1() {
    extreme(256, 256, 128, Control::Fast, 21)
}
/// Padding plus 20 epochs of full-range `C` (sum of 20 equal terms stays below 2^51).
#[test]
fn ief15_extreme_classes_e20_padded() {
    extreme(300, 140, 20 * 128, Control::Fast, 22)
}
/// Largest epoch count with `S = D = 32767` on the full-range `C`: `|I| < 2^51`, the integer state never wraps.
#[test]
fn ief15_extreme_max_state_e136() {
    let (tokens, features, k) = (64, 64, 136 * 128);
    let mut p = wild(tokens, features, k, 23);
    let e = 136;
    p.s = vec![32767; features * e];
    p.d = vec![32767; e * tokens];
    p.a = (0..features)
        .map(|r| [0i16, -90, 40, -200][r % 4])
        .collect();
    p.b = (0..tokens).map(|t| [0i16, -60, 20, 100][t % 4]).collect();
    let g = shape(tokens, features, k, Control::Fast, &p);
    let parts = iu4::partials(&p.weights(), &p.acts());
    let i0: i128 = (0..e)
        .map(|ep| 32767i128 * 32767 * i128::from(parts[ep * tokens * features]))
        .sum();
    assert!(
        i0 > 1 << 50 && i0 < 1 << 51,
        "feature 0 / token 0 reaches the 2^51 bound: {i0}"
    );
    drop(g);
}
/// Only zero scales: every output is +0; only zero mantissa tokens / features still produce exact +0.
#[test]
fn ief15_all_zero_scales() {
    let mut p = encoded(130, 70, 384, 24, true);
    p.s.fill(0);
    shape(130, 70, 384, Control::Fast, &p);
    let mut p = encoded(130, 70, 384, 25, true);
    p.d.fill(0);
    shape(130, 70, 384, Control::Fast, &p);
}

// ---------------------------------------------------------------- (e) (f) reuse and lean
/// (e) One context, three full submits with different operands.
#[test]
fn ief15_three_submits_one_context() {
    let (tokens, features, k) = (300, 300, 4 * 128);
    let g = Ief15Gemm::new(tokens, features, k, Control::Fast).unwrap();
    let mut sim = Config::from_pdi(&g.design().pdi).unwrap();
    for (i, seed) in [11u64, 12, 13].into_iter().enumerate() {
        let p = if i == 1 {
            wild(tokens, features, k, seed)
        } else {
            encoded(tokens, features, k, seed, i == 2)
        };
        let c = run_on(&mut sim, &g, &g.design().insts, p.pack(&g));
        assert_exact(&g, &p, &c, &p.reference(&g), &format!("submit {i}"));
    }
}

/// (f) Full and lean submits interleaved on one context: bytes exact, equal to a fresh full run, retained state equal to
/// the state a fresh full run leaves.
fn lean_interleave(tokens: usize, features: usize, k: usize) {
    let g = Ief15Gemm::new(tokens, features, k, Control::Fast).unwrap();
    let what = format!(
        "{tokens}x{features}x{k} (J = {}, W = {})",
        g.epochs() * g.design().waves(),
        g.design().waves()
    );
    let lean = g
        .design()
        .lean_insts()
        .unwrap_or_else(|e| panic!("{what}: lean_insts: {e}"));
    assert!(
        lean.len() < g.design().insts.len(),
        "{what}: lean TXN must be smaller"
    );
    let mut sim = Config::from_pdi(&g.design().pdi).unwrap();
    let mut initial = None;
    for (i, (seed, is_lean)) in [
        (31u64, false),
        (32, true),
        (33, true),
        (34, false),
        (35, true),
    ]
    .into_iter()
    .enumerate()
    {
        let p = if i % 2 == 0 {
            encoded(tokens, features, k, seed, true)
        } else {
            wild(tokens, features, k, seed)
        };
        let mut fresh = Config::from_pdi(&g.design().pdi).unwrap();
        let reference = run_on(&mut fresh, &g, &g.design().insts, p.pack(&g));
        let init = initial.get_or_insert_with(|| fresh.state_snapshot());
        assert!(
            fresh.state_snapshot() == *init,
            "{what}: fresh full run state differs between operands"
        );
        let c = run_on(
            &mut sim,
            &g,
            if is_lean { &lean } else { &g.design().insts },
            p.pack(&g),
        );
        assert_eq!(
            c,
            reference,
            "{what}: submit {i} ({}) differs from a fresh full run",
            if is_lean { "lean" } else { "full" }
        );
        assert_exact(&g, &p, &c, &p.reference(&g), &what);
        assert!(
            sim.state_snapshot() == *init,
            "{what}: retained state after submit {i} differs from a fresh full run"
        );
    }
}
/// `J` and `W` both odd (E = 3, one wave).
#[test]
fn ief15_lean_interleave_j_odd() {
    lean_interleave(256, 256, 3 * 128)
}
/// `J` and `W` even.
#[test]
fn ief15_lean_interleave_j_even() {
    lean_interleave(512, 256, 2 * 128)
}
/// `W` odd, `J` even (E = 2, one wave) and `W` even with odd E (two waves, E = 3).
#[test]
fn ief15_lean_interleave_w_odd_e_even() {
    lean_interleave(256, 256, 2 * 128)
}
#[test]
fn ief15_lean_interleave_w_even_e_odd() {
    lean_interleave(256, 512, 3 * 128)
}
/// Padding, N waves and 5 epochs.
#[test]
fn ief15_lean_interleave_padded() {
    lean_interleave(300, 300, 5 * 128)
}

// ---------------------------------------------------------------- (g) dense shapes at reduced M
/// K = 5120 (E = 40, the o_proj / down K of the 27B), 6144 (E = 48) and 17408 (E = 136, the down projection).
#[test]
fn ief15_dense_256_256_k5120() {
    shape(
        256,
        256,
        5120,
        Control::Fast,
        &encoded(256, 256, 5120, 41, true),
    );
}
#[test]
fn ief15_dense_256_256_k6144() {
    shape(
        256,
        256,
        6144,
        Control::Fast,
        &encoded(256, 256, 6144, 42, true),
    );
}
#[test]
fn ief15_dense_256_256_k17408() {
    shape(
        256,
        256,
        17408,
        Control::Fast,
        &encoded(256, 256, 17408, 43, true),
    );
}

// ---------------------------------------------------------------- data movement with a stand-in core
/// The V8 `no_compute` pair program (lock and DMA protocol, no data access) runs the IEF15 descriptors, routes, locks and
/// TXN to completion and every shim channel moves exactly the layout's byte count: A `waves*E*2176` on MM2S1 and B on MM2S0
/// per column, `waves * 2 * 8192` on each of S2MM0 / S2MM1. Full and lean bodies both complete.
fn standin(tokens: usize, features: usize, k: usize) {
    let ctl = Control::Probe(Probe {
        no_compute: true,
        ..Probe::default()
    });
    let epochs = k / 128;
    let layout = ief::Ief15Layout::new(tokens, features, k, ctl).unwrap();
    let programs = |role| {
        gemm_core::program_pair(
            epochs,
            layout.waves(),
            role,
            Epilogue::Int8 { shift: 0 },
            ctl,
        )
        .finish()
    };
    let g = Ief15Gemm::with_programs(
        tokens,
        features,
        k,
        ctl,
        programs(PairRole::Lower),
        programs(PairRole::Upper),
    )
    .unwrap();
    let [wa, wb, wc] = layout.arg_bytes();
    let waves = layout.waves();
    let mut sim = Config::from_pdi(&g.design().pdi).unwrap();
    let lean = g.design().lean_insts().unwrap();
    for (i, insts) in [&g.design().insts[..], &lean[..], &g.design().insts[..]]
        .into_iter()
        .enumerate()
    {
        let before: Vec<u64> = sim.shim_stats().map(|c| c.dram_bytes()).collect();
        let mut args = vec![vec![0x5au8; wa], vec![0xa5u8; wb], vec![0u8; wc]];
        assert!(sim.shim_stats().count() > 0, "no shim stats");
        sim.submit(insts, &mut args)
            .unwrap_or_else(|e| panic!("{tokens}x{features}x{k} submit {i}: {e}"));
        for (c, b0) in sim.shim_stats().zip(before) {
            let want = match (c.direction, c.channel) {
                (Direction::Mm2s, 0) => wb / 8,
                (Direction::Mm2s, 1) => wa / 8,
                (Direction::S2mm, 0 | 1) => waves * 2 * 8192,
                other => panic!("unexpected shim channel {other:?}"),
            } as u64;
            assert_eq!(
                c.dram_bytes() - b0,
                want,
                "{tokens}x{features}x{k} submit {i} col {} {:?}{}",
                c.col,
                c.direction,
                c.channel
            );
        }
    }
    println!(
        "stand-in core {tokens}x{features}x{k}: full + lean + full completed, byte counts exact"
    );
}
#[test]
fn ief15_standin_core_moves_every_byte() {
    standin(256, 256, 128);
    standin(300, 700, 640);
    standin(256, 256, 3 * 128);
    standin(70, 90, 20 * 128);
}

/// One final-pack group on a single tile (`iu4_ief15_core::final_pack_debug_program`) against the lane model on random
/// and tie-heavy lanes over the whole exponent range (subnormal, overflow, -0, carries).
fn run_pack(ops: usize, i: &[i64; 32], xv: &[i16; 32]) -> peacemaker_npu::sim::Tile {
    use peacemaker_npu::kernels::iu4_ief15_core as core;
    let prog = core::final_pack_debug_program(ops);
    let dec = peacemaker_npu::sim::decode::PreparedDecoder::new(&prog).unwrap();
    let mut tile = peacemaker_npu::sim::Tile::new(prog).unwrap();
    for (l, v) in i.iter().enumerate() {
        let o = core::I_ADDR as usize + 8 * l;
        tile.memory[o..o + 8].copy_from_slice(&v.to_le_bytes());
    }
    for (l, v) in xv.iter().enumerate() {
        let o = core::C16_ADDR as usize + 2 * l;
        tile.memory[o..o + 2].copy_from_slice(&v.to_le_bytes());
    }
    let r = tile.run(&dec, 100_000);
    assert!(matches!(r, Ok(peacemaker_npu::sim::Step::Done)), "{r:?}");
    tile
}
#[test]
fn ief15_final_pack_group_matches_lane_model() {
    let mut seed = 0x1234_5678_9abc_def1u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut bad = 0;
    for it in 0..300 {
        let mut i = [0i64; 32];
        let mut xv = [0i16; 32];
        for l in 0..32 {
            let bits = (next() % 52) as u32;
            let mut v = (next() >> 13) as i64 & ((1i64 << bits) - 1).max(0);
            if it % 3 == 0 {
                v = (((next() >> 40) as i64 | (1 << 24)) * 2 + 1) << (next() % 26);
                v &= (1 << 51) - 1;
            }
            i[l] = if next() & 1 == 1 { -v } else { v };
            xv[l] = ((next() % 641) as i32 - 320) as i16;
        }
        let t = run_pack(usize::MAX, &i, &xv);
        for l in 0..32 {
            let o = peacemaker_npu::kernels::iu4_ief15_core::COUT_ADDR as usize + 4 * l;
            let got = u32::from_le_bytes(t.memory[o..o + 4].try_into().unwrap());
            let want = peacemaker_npu::kernels::iu4_ief15_core::pack_lane_model(i[l], xv[l] as i32);
            if got != want {
                bad += 1;
                if bad < 6 {
                    eprintln!("I {} X {}: got {got:08x} want {want:08x}", i[l], xv[l]);
                }
            }
        }
    }
    assert_eq!(bad, 0, "final pack mismatches");
}
