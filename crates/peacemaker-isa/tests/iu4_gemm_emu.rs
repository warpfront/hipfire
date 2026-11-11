// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The production dense `iu4_gemm` kernel on the PeaceMaker CPU emulator, through the reusable harness of
//! `support/iu4_emu.rs`: hand-checkable cases (`Y = K` exactly, signed extremes) and a random case against the exact
//! CPU fold, bitwise, for SET (token and slab activations) and ADD.
use iu4_emu::*;
use peacemaker_isa::kernels::iu4_gemm::{
    verify64_tune, ALayout, Epi, FetchPlace, FoldSched, IdleHalf, Spec, Tune, WaitPlace,
};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;
#[path = "support/iu4_emu.rs"]
mod iu4_emu;

fn control(epi: Epi) -> Spec {
    Spec::control(epi)
}

fn bitwise_eq(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// All weight codes 9 (signed +1 after `q - 8`), all activation q = +1, unit scales: `Y = K` exactly. M = 136 and
/// N = 40 cover a second, partial row tile and a partial token tile.
#[test]
fn dense_all_ones_gives_k_exactly() {
    require_rocm_tool!("llvm-mc");
    for (k, epi) in [(256, Epi::Set), (512, Epi::Set), (256, Epi::Add)] {
        let p = Problem::uniform(136, 40, k, 9, 1);
        let y = run_gemm(control(epi), &p).unwrap();
        assert!(
            y.iter().all(|&v| v == k as f32),
            "{epi:?} K={k}: y[0]={} y[last]={}",
            y[0],
            y[y.len() - 1]
        );
    }
}

/// Signed extremes: code 0 (-8) against q = -8 gives +64 per K, code 15 (+7) against q = -8 gives -56.
#[test]
fn dense_signed_extremes_are_exact() {
    require_rocm_tool!("llvm-mc");
    for (code, q) in [(0u8, -8), (15, -8), (0, 7)] {
        let p = Problem::uniform(128, 16, 256, code, q);
        let y = run_gemm(control(Epi::Set), &p).unwrap();
        let want = (256 * (i32::from(code) - 8) * q) as f32;
        assert!(y.iter().all(|&v| v == want), "code {code} q {q}: {}", y[0]);
    }
}

/// Random weights and activations against the CPU fold, bitwise: SET with token and slab activations, ADD.
#[test]
fn dense_random_matches_the_cpu_fold_bitwise() {
    require_rocm_tool!("llvm-mc");
    let p = Problem::synth(136, 40, 384 + 128, 0x9e37_79b9_7f4a_7c15);
    for (epi, act) in [
        (Epi::Set, ALayout::Token),
        (Epi::Set, ALayout::Slab),
        (Epi::Add, ALayout::Token),
    ] {
        let spec = Spec {
            act,
            ..control(epi)
        };
        let got = run_gemm(spec, &p).unwrap();
        let want = reference(&p, epi == Epi::Add);
        assert_eq!(
            bitwise_eq(&got, &want),
            0,
            "{epi:?} {act:?}: first got {} want {}",
            got[0],
            want[0]
        );
    }
}

/// `WaitPlace::Window` (the shipped schedule of the 27B gate/up `_b1s` export, `Tune::shipped_gate_up_slab`) is
/// scheduling only: the split B1 rendezvous with sub-block 1's WMMA group in its window and `d_loads` hoisted into phase
/// A leave every result bit equal to the default schedule's and to the CPU fold's, at K = 256 (peeled group only),
/// 512, 1024 and 5120 (the 27B hidden size: a rolled loop of full trips). SiLU is not driven by the emulator
/// (`run_gemm` takes set/add): the epilogue follows the K-loop and is untouched by the schedule, so SET and ADD on the
/// slab layout (the export's activation layout) and the token layout cover the window block.
#[test]
fn window_schedule_is_bit_identical_to_the_default() {
    require_rocm_tool!("llvm-mc");
    for k in [256, 512, 1024, 5120] {
        let p = Problem::synth(136, 40, k, 0x9e37_79b9_7f4a_7c15 ^ k as u64);
        for (epi, act) in [
            (Epi::Set, ALayout::Slab),
            (Epi::Add, ALayout::Slab),
            (Epi::Set, ALayout::Token),
        ] {
            let base = Spec {
                act,
                ..control(epi)
            };
            let want = run_gemm(base, &p).unwrap();
            let got = run_gemm(
                Spec {
                    tune: Tune::shipped_gate_up_slab(),
                    ..base
                },
                &p,
            )
            .unwrap();
            assert_eq!(
                bitwise_eq(&got, &want),
                0,
                "{epi:?} {act:?} K={k}: window vs default"
            );
            assert_eq!(
                bitwise_eq(&got, &reference(&p, epi == Epi::Add)),
                0,
                "{epi:?} {act:?} K={k}: window vs the CPU fold"
            );
        }
    }
    assert_eq!(Tune::shipped_gate_up_slab().wait, WaitPlace::Window);
}

/// `FoldSched::Chain` (stage 1 of one accumulator between the WMMAs, rotated chained stage 2, wave priorities) is scheduling
/// only: at every wait placement, SET and ADD on the slab layout, K = 256 (peeled group only), 512, 1024 and 5120 (a rolled
/// loop of full trips), every result bit equals the default schedule's and the CPU fold's.
#[test]
fn chain_fold_is_bit_identical_to_the_default() {
    require_rocm_tool!("llvm-mc");
    for k in [256, 512, 1024, 5120] {
        let p = Problem::synth(136, 40, k, 0x517c_c1b7_2722_0a95 ^ k as u64);
        for epi in [Epi::Set, Epi::Add] {
            let base = Spec {
                act: ALayout::Slab,
                ..control(epi)
            };
            let want = run_gemm(base, &p).unwrap();
            assert_eq!(bitwise_eq(&want, &reference(&p, epi == Epi::Add)), 0);
            for wait in [
                WaitPlace::AfterMma,
                WaitPlace::MidMma,
                WaitPlace::BeforeMma,
                WaitPlace::Window,
            ] {
                let got = run_gemm(
                    Spec {
                        tune: Tune {
                            wait,
                            fold_sched: FoldSched::Chain,
                            ..Tune::default()
                        },
                        ..base
                    },
                    &p,
                )
                .unwrap();
                assert_eq!(
                    bitwise_eq(&got, &want),
                    0,
                    "{epi:?} K={k} wait={}: chain vs default",
                    wait.name()
                );
            }
        }
    }
}

/// `Tune::idle` (skip the compute of the empty token half's waves) is scheduling only: for `skip` and `skiphi` (the wave
/// remap), with the `after` and `window` K-loop schedules, SET and ADD, slab and token activations, every result bit
/// equals the default schedule's and the CPU fold's. N = 16, 40 and 64 leave the second token half empty (dead waves
/// skip every K block and their epilogue), 65 and 96 and 128 keep it live (the predicate is false: the same code runs),
/// 129 has a second token tile with one token (`N - bs = 1`: that tile skips, the first does not). K = 256 is the
/// peeled group only, 512 one loop trip plus the group, 5120 (the 27B hidden size) a rolled loop; the long K runs only
/// the shipped window schedule on the slab layout at the dead-half boundary shapes.
#[test]
fn idle_skip_is_bit_identical() {
    require_rocm_tool!("llvm-mc");
    for n in [16usize, 40, 64, 65, 96, 128, 129] {
        for k in [256usize, 512, 5120] {
            let long = k == 5120;
            if long && ![64, 65, 129].contains(&n) {
                continue;
            }
            let p = Problem::synth(
                136,
                n,
                k,
                0x9e37_79b9_7f4a_7c15 ^ k as u64 ^ ((n as u64) << 40),
            );
            let acts: &[ALayout] = if long {
                &[ALayout::Slab]
            } else {
                &[ALayout::Slab, ALayout::Token]
            };
            let waits: &[WaitPlace] = if long {
                &[WaitPlace::Window]
            } else {
                &[WaitPlace::AfterMma, WaitPlace::Window]
            };
            for epi in [Epi::Set, Epi::Add] {
                let cpu = reference(&p, epi == Epi::Add);
                for &act in acts {
                    let base = Spec {
                        act,
                        ..control(epi)
                    };
                    let want = run_gemm(base, &p).unwrap();
                    assert_eq!(
                        bitwise_eq(&want, &cpu),
                        0,
                        "{epi:?} {act:?} N={n} K={k}: default vs the CPU fold"
                    );
                    for &wait in waits {
                        for idle in [IdleHalf::Skip, IdleHalf::SkipHi] {
                            let got = run_gemm(
                                Spec {
                                    tune: Tune {
                                        wait,
                                        idle,
                                        ..Tune::default()
                                    },
                                    ..base
                                },
                                &p,
                            )
                            .unwrap_or_else(|e| {
                                panic!(
                                    "{epi:?} {act:?} N={n} K={k} {} {}: {e}",
                                    wait.name(),
                                    idle.name()
                                )
                            });
                            assert_eq!(
                                bitwise_eq(&got, &want),
                                0,
                                "{epi:?} {act:?} N={n} K={k} wait {} idle {}: vs default",
                                wait.name(),
                                idle.name()
                            );
                            assert_eq!(
                                bitwise_eq(&got, &cpu),
                                0,
                                "{epi:?} {act:?} N={n} K={k} wait {} idle {}: vs the CPU fold",
                                wait.name(),
                                idle.name()
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The product small-N (`_v64`) exports run exactly as shipped: `Spec { tune: verify64_tune(epi, false), .. }` (slab layout,
/// 128x128x8 tile) is bit-identical to the same spec with `Tune::default()`, and for SET and ADD to the CPU fold, for N = 1
/// through 129 (both sides of the dead-half boundary 64 and the second token tile) and K = 256 (peeled group), 512 and
/// 5120 (rolled loop; only at N = 64 and 65). M = 136 leaves a partial second row tile. Gate/up SiLU is not driven: the
/// emulator stops on a NaN operand of the SiLU region's `v_dual_sub_f32` (an unspecified-NaN opcode) in the shipped
/// `_b1s` kernel too, so only its tune is checked here. The fused projection's `_v64` export (`QkvzaGdn`,
/// `QkvStore::Set`) is not driven either (120-byte kernarg ABI, three row segments); the M7 and tune tests cover it.
#[test]
fn verify64_exports_are_bit_identical_to_the_default_schedule() {
    require_rocm_tool!("llvm-mc");
    // The tune is what the export is documented to be.
    for epi in [Epi::Set, Epi::Add, Epi::GateUpSilu] {
        let t = verify64_tune(epi, false);
        assert!(t.n64 && t.wait == WaitPlace::Window, "{epi:?}");
        assert_eq!(t.fetch, FetchPlace::PairW, "{epi:?}");
        assert_eq!(t.idle, IdleHalf::SkipHi, "{epi:?}");
        let v64 = Spec {
            act: ALayout::Slab,
            tune: t,
            ..control(epi)
        };
        let shipped = Spec {
            act: ALayout::Slab,
            ..control(epi)
        };
        assert_eq!(v64.symbol(), format!("{}_v64", shipped.symbol()));
        v64.validate().unwrap();
    }
    for n in [1usize, 16, 40, 63, 64, 65, 96, 128, 129] {
        for k in [256usize, 512, 5120] {
            if k == 5120 && ![64, 65].contains(&n) {
                continue;
            }
            let seed = 0x9e37_79b9_7f4a_7c15 ^ k as u64 ^ ((n as u64) << 40);
            let p = Problem::synth(136, n, k, seed);
            for epi in [Epi::Set, Epi::Add] {
                let add = epi == Epi::Add;
                let base = Spec {
                    act: ALayout::Slab,
                    ..control(epi)
                };
                let want = run_gemm(base, &p).unwrap();
                let cpu = reference(&p, add);
                assert_eq!(
                    bitwise_eq(&want, &cpu),
                    0,
                    "{epi:?} N={n} K={k}: default vs the CPU fold"
                );
                let got = run_gemm(
                    Spec {
                        tune: verify64_tune(epi, false),
                        ..base
                    },
                    &p,
                )
                .unwrap_or_else(|e| panic!("{epi:?} N={n} K={k}: {e}"));
                assert_eq!(
                    bitwise_eq(&got, &want),
                    0,
                    "{epi:?} N={n} K={k}: _v64 vs default"
                );
                assert_eq!(
                    bitwise_eq(&got, &cpu),
                    0,
                    "{epi:?} N={n} K={k}: _v64 vs the CPU fold"
                );
            }
        }
    }
}
