// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted gfx1201 draft-head partial top-8 `topk8_partial_f32_blocks_pm_gfx1201`, the
//! byte twin of the hipcc `topk8_partial_f32_blocks` (`kernels/src/qwen4_gemv_q8_0.hip`). The
//! emitted program runs on the PeaceMaker CPU interpreter (256-thread workgroups, LDS and
//! workgroup barriers included); every pv/pi word of every launched group must equal the host
//! reference of the HIP law (value bits, `+0`/`-0` and NaN-skipping included) and nothing outside
//! the launched groups' slots may be written. No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::kernels::topk8_partial_blocks_pm as tk;
use peacemaker_isa::Arch;
use std::sync::LazyLock;

const ARCH: Arch = Arch::Gfx1201;
const GROUPS: usize = 128;
const POISON: u8 = 0x5A;
const TAIL_GUARD: usize = 16;

#[test]
fn emission_is_deterministic_and_targets_gfx1201_only() {
    let a = tk::emit_module(ARCH).unwrap();
    let b = tk::emit_module(ARCH).unwrap();
    assert_eq!(a.1, b.1);
    assert_eq!(a.2.s_text_sha256, b.2.s_text_sha256);
    assert!(a.0[0].shape.barriers > 0);
    assert!(a.0[0].shape.next_free_vgpr <= 64 && a.0[0].shape.next_free_sgpr <= 104);
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(tk::emit(arch).is_err(), "{arch:?}");
    }
    assert_eq!(tk::symbol(ARCH), "topk8_partial_f32_blocks_pm_gfx1201");
    assert_eq!(tk::module(ARCH), "topk8_partial_blocks_pm_gfx1201");
}

/// Kernarg ABI of `topk8_partial_f32_blocks`: v, n, chunk, pv, pi, front, special, tail, block0.
#[test]
fn kernarg_layout_matches_the_hip_kernel() {
    let text = tk::emit_module(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let offsets: Vec<u32> = metadata
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse::<u32>().unwrap())
        })
        .collect();
    assert_eq!(offsets, [0, 8, 12, 16, 24, 32, 36, 40, 44]);
    assert!(text.contains(".amdhsa_kernarg_size 48\n"));
    assert!(text.contains(".amdhsa_group_segment_fixed_size 64\n"));
    assert!(text.contains(".max_flat_workgroup_size: 256"));
    assert!(!text.contains("atomic"));
    for l in text.lines().map(str::trim_start) {
        assert!(
            !(l.starts_with("scratch_") || l.starts_with("flat_") || l.starts_with("global_")),
            "{l}"
        );
    }
}

/// `topk8_better`: larger value, then smaller layout index; NaN is never better.
fn better(v: f32, i: i32, bv: f32, bi: i32) -> bool {
    v > bv || (v == bv && i < bi)
}

/// The HIP kernel for group `group`: the eight (value bits, token) rounds.
fn reference(
    row: &[f32],
    n: usize,
    chunk: usize,
    group: usize,
    (front, special, tail): (i32, i32, i32),
) -> ([u32; 8], [i32; 8]) {
    let begin = group * chunk;
    let end = n.min(begin + chunk);
    // The 256 x 8 candidate slots (j = begin + t + 256 u); beyond `end` the sentinel.
    let mut cand: Vec<(f32, i32)> = (0..2048usize)
        .map(|k| {
            let j = begin + (k % 256) + (k / 256) * 256;
            if j < end {
                (row[j], j as i32)
            } else {
                (f32::NEG_INFINITY, i32::MAX)
            }
        })
        .collect();
    let (mut pv, mut pi) = ([0u32; 8], [0i32; 8]);
    for round in 0..8 {
        let (mut bv, mut bi) = (f32::NEG_INFINITY, i32::MAX);
        for &(v, i) in &cand {
            if better(v, i, bv, bi) {
                bv = v;
                bi = i;
            }
        }
        for c in &mut cand {
            if c.1 == bi {
                *c = (f32::NEG_INFINITY, i32::MAX);
            }
        }
        let token = if bi < front {
            bi
        } else if bi < front.wrapping_add(tail) {
            bi.wrapping_sub(front).wrapping_add(special)
        } else {
            bi.wrapping_sub(tail)
        };
        pv[round] = bv.to_bits();
        pi[round] = token;
    }
    (pv, pi)
}

static PROGRAM: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| {
    let text = tk::emit_module(ARCH).unwrap().1;
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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f(&mut self) -> f32 {
        (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
    }
}

/// One launch of `blocks` workgroups from `block0` over `row` (`n` = `row.len()`); every launched
/// group's pv/pi slots must equal the reference and every other word must keep its poison.
fn check(name: &str, row: &[f32], order: (i32, i32, i32), block0: usize, blocks: usize) {
    let n = row.len();
    let chunk = n.div_ceil(GROUPS);
    let emu = Emulator::new(&PROGRAM, &tk::symbol(ARCH)).unwrap();
    let (kp, vp, pvp, pip) = (0x1_0000u64, 0x10_0000u64, 0x4000_0000u64, 0x5000_0000u64);
    let mut kern = Vec::new();
    kern.extend(vp.to_le_bytes());
    kern.extend((n as i32).to_le_bytes());
    kern.extend((chunk as i32).to_le_bytes());
    kern.extend(pvp.to_le_bytes());
    kern.extend(pip.to_le_bytes());
    kern.extend(order.0.to_le_bytes());
    kern.extend(order.1.to_le_bytes());
    kern.extend(order.2.to_le_bytes());
    kern.extend((block0 as i32).to_le_bytes());
    assert_eq!(kern.len(), tk::KERNARG_BYTES as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    mem.map(
        "v".into(),
        vp,
        row.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
    )
    .unwrap();
    let out_bytes = GROUPS * 8 * 4 + TAIL_GUARD;
    mem.map("pv".into(), pvp, vec![POISON; out_bytes]).unwrap();
    mem.map("pi".into(), pip, vec![POISON; out_bytes]).unwrap();
    for b in 0..blocks {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: kp,
                workgroup: [b as u32, 0, 0],
                block: [tk::BLOCK, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("{name}: n {n} order {order:?} block {b}: {e}"));
    }
    let region = |name: &str| {
        mem.regions
            .iter()
            .find(|r| r.name == name)
            .unwrap()
            .bytes
            .clone()
    };
    let (pv, pi) = (region("pv"), region("pi"));
    let word =
        |bytes: &[u8], k: usize| u32::from_le_bytes(bytes[4 * k..4 * k + 4].try_into().unwrap());
    let poison = u32::from_le_bytes([POISON; 4]);
    for g in 0..GROUPS {
        let launched = (block0..block0 + blocks).contains(&g);
        if launched {
            let (wv, wi) = reference(row, n, chunk, g, order);
            for r in 0..8 {
                assert_eq!(
                    word(&pv, g * 8 + r),
                    wv[r],
                    "{name}: n {n} order {order:?} group {g} round {r} pv bits"
                );
                assert_eq!(
                    word(&pi, g * 8 + r) as i32,
                    wi[r],
                    "{name}: n {n} order {order:?} group {g} round {r} pi"
                );
            }
        } else {
            for r in 0..8 {
                assert_eq!(word(&pv, g * 8 + r), poison, "{name}: group {g} pv written");
                assert_eq!(word(&pi, g * 8 + r), poison, "{name}: group {g} pi written");
            }
        }
    }
    assert!(
        pv[GROUPS * 32..]
            .iter()
            .chain(&pi[GROUPS * 32..])
            .all(|&b| b == POISON),
        "{name}: written past the partial buffer"
    );
}

fn orders(n: usize) -> Vec<(i32, i32, i32)> {
    let n = n as i32;
    vec![
        (n, 0, 0),
        (0, 0, 0),
        (n / 3, n / 2, n / 5),
        (n / 2, n, n / 4),
    ]
}

fn flavours(rng: &mut Rng, n: usize, chunk: usize) -> Vec<(&'static str, Vec<f32>)> {
    let edge = (64 * chunk).min(n.saturating_sub(1));
    let mut out: Vec<(&'static str, Vec<f32>)> = Vec::new();
    out.push(("random", (0..n).map(|_| rng.f() * 8.0).collect()));
    out.push(("ties", (0..n).map(|_| (rng.next() % 5) as f32).collect()));
    out.push(("all-equal", vec![1.5; n]));
    out.push((
        "zero-signs",
        (0..n)
            .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
            .collect(),
    ));
    out.push(("all-nan", vec![f32::NAN; n]));
    out.push(("all-neg-inf", vec![f32::NEG_INFINITY; n]));
    out.push(("all-pos-inf", vec![f32::INFINITY; n]));
    out.push((
        "random-bits",
        (0..n).map(|_| f32::from_bits(rng.next() as u32)).collect(),
    ));
    out.push((
        "denormals",
        (0..n)
            .map(|_| {
                let sign = if rng.next() & 1 == 0 { 0 } else { 0x8000_0000 };
                f32::from_bits(sign | (rng.next() % 8) as u32)
            })
            .collect(),
    ));
    out.push((
        "nonfinite-mix",
        (0..n)
            .map(|_| match rng.below(6) {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                2 => f32::INFINITY,
                3 => -0.0,
                4 => 0.0,
                _ => rng.f(),
            })
            .collect(),
    ));
    let mut tie_edge: Vec<f32> = (0..n).map(|_| -1.0 - rng.f().abs()).collect();
    for k in 0..8 {
        if edge >= k + 1 {
            tie_edge[edge - 1 - k] = 20.0;
        }
        if edge + k < n {
            tie_edge[edge + k] = 20.0;
        }
    }
    out.push(("ties-at-64-chunk", tie_edge));
    // Fewer than eight finite values per group; NaN and -inf padding.
    let mut sparse = vec![f32::NEG_INFINITY; n];
    for _ in 0..20.min(n) {
        let i = rng.below(n);
        sparse[i] = rng.f();
    }
    if n > 2 {
        sparse[n / 2] = f32::NAN;
    }
    out.push(("sparse-neg-inf", sparse));
    out
}

#[test]
fn emulated_groups_equal_the_hip_law() {
    let mut rng = Rng(0x70_9C8);
    for n in [1usize, 5, 257, 2048, 5000, 20_000] {
        let chunk = n.div_ceil(GROUPS);
        for (name, row) in flavours(&mut rng, n, chunk) {
            // Every token-map order on the first two groups; the middle pair (the 64-chunk
            // boundary) and the last group (past `n` when the row is short) on two orders.
            for (k, order) in orders(n).into_iter().enumerate() {
                check(name, &row, order, 0, 2);
                if k % 2 == 0 {
                    check(name, &row, order, 63, 2);
                    check(name, &row, order, GROUPS - 1, 1);
                }
            }
        }
    }
}

/// n = 100,276 and 248,320 (chunk 784 and 1,940): the two groups at the 64-chunk boundary, ties
/// across it, and the block-0-offset launch against a one-launch run of the same groups.
#[test]
fn emulated_boundary_groups_at_the_production_vocabularies() {
    let mut rng = Rng(0xB0_0D);
    for n in [100_276usize, 248_320] {
        let chunk = n.div_ceil(GROUPS);
        let edge = 64 * chunk;
        let mut row: Vec<f32> = (0..n).map(|_| -1.0 - rng.f().abs()).collect();
        for k in 0..8 {
            row[edge - 1 - k] = 20.0;
            row[edge + k] = 20.0;
        }
        row[edge + 7] = f32::NEG_INFINITY;
        row[edge - 3] = f32::NAN;
        for order in [(n as i32, 0, 0), (90_000, 100_000, 276)] {
            check("boundary-ties", &row, order, 63, 2);
            check("boundary-ties-single", &row, order, 64, 1);
        }
    }
}
