// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `iu4_gemm::Tune` generator knobs (`ring`, `wait`, `fold_op`): the default is today's codegen, every legal
//! non-default point emits and assembles for the 27B prefill gate/up spec (`gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s_bf16`),
//! the schedule-only knobs are bit-identical to the control on the CPU emulator, `fold_op = muladd` is the exact
//! non-fused fold, and `Spec::validate` rejects what has no generator.
use iu4_emu::*;
use peacemaker_isa::kernels::iu4_gemm::{
    emit, module, ALayout, Epi, FetchPlace, FoldOp, FoldSched, IdleHalf, Mma, QkvStore, RingDepth,
    Spec, Tile, Tune, WaitPlace,
};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;
#[path = "support/iu4_emu.rs"]
mod iu4_emu;

const WAITS: [WaitPlace; 3] = [WaitPlace::AfterMma, WaitPlace::MidMma, WaitPlace::BeforeMma];

/// The 27B prefill gate/up node: slab activations, packed bf16 h, the production tile.
fn node() -> Spec {
    Spec {
        act: ALayout::Slab,
        ..Spec::control(Epi::GateUpSiluBf16)
    }
}

fn with(spec: Spec, tune: Tune) -> Spec {
    Spec { tune, ..spec }
}

fn bitwise_eq(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

#[test]
fn default_tune_is_the_shipped_codegen() {
    let spec = node();
    assert!(spec.tune.is_default());
    assert_eq!(
        spec.tune,
        Tune {
            ring: RingDepth::Two,
            wait: WaitPlace::AfterMma,
            fold_op: FoldOp::Fma,
            fold_sched: FoldSched::Paired,
            fetch: FetchPlace::Split,
            idle: IdleHalf::Off,
            qkv: QkvStore::Gdn,
            n64: false,
        }
    );
    // The default spec keeps its variant name and symbol; a tuned one keeps the symbol (the product ABI) and
    // only extends the variant name.
    assert_eq!(spec.variant_name(), "k128-128x128x8-cacc1-slab-silu-bf16");
    let tuned = with(
        spec,
        Tune {
            wait: WaitPlace::MidMma,
            fold_op: FoldOp::MulAdd,
            ..Tune::default()
        },
    );
    assert_eq!(tuned.symbol(), spec.symbol());
    assert_eq!(
        tuned.variant_name(),
        "k128-128x128x8-cacc1-slab-silu-bf16-wmid-fmuladd"
    );
}

/// Every legal wait placement emits and assembles at the 27B spec; each differs from the control, from the
/// others, and re-emits to the same bytes.
#[test]
fn wait_placements_emit_distinct_schedules() {
    let spec = node();
    let mut texts = Vec::new();
    for wait in WAITS {
        let s = with(
            spec,
            Tune {
                wait,
                ..Tune::default()
            },
        );
        s.validate().unwrap();
        let e = emit(s).unwrap_or_else(|e| panic!("wait {}: {e}", wait.name()));
        peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
        assert_eq!(
            emit(s).unwrap().s_text,
            e.s_text,
            "wait {} is not deterministic",
            wait.name()
        );
        texts.push(e.s_text);
    }
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            assert_ne!(
                texts[i], texts[j],
                "{:?} and {:?} emit the same text",
                WAITS[i], WAITS[j]
            );
        }
    }
    assert_eq!(texts[0], emit(spec).unwrap().s_text, "after is the default");
}

/// The other epilogues and the 16-wave tile take every wait placement too.
#[test]
fn wait_placements_emit_for_every_dense_variant() {
    for (epi, act, tile) in [
        (Epi::Set, ALayout::Token, Tile::T128x128x8),
        (Epi::Add, ALayout::Slab, Tile::T128x128x8),
        (Epi::GateUpSilu, ALayout::Token, Tile::T256x128x16),
        (Epi::QkvzaGdn, ALayout::Token, Tile::T128x128x8),
    ] {
        for wait in WAITS {
            let s = Spec {
                act,
                tile,
                tune: Tune {
                    wait,
                    ..Tune::default()
                },
                ..Spec::control(epi)
            };
            let e = emit(s)
                .unwrap_or_else(|e| panic!("{} wait {}: {e}", s.variant_name(), wait.name()));
            peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
        }
    }
}

#[test]
fn fold_op_muladd_emits_and_replaces_the_fused_accumulate() {
    let spec = node();
    let base = emit(spec).unwrap().s_text;
    let s = with(
        spec,
        Tune {
            fold_op: FoldOp::MulAdd,
            ..Tune::default()
        },
    );
    s.validate().unwrap();
    let e = emit(s).unwrap();
    peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
    let count = |t: &str, m: &str| t.lines().filter(|l| l.trim_start().starts_with(m)).count();
    assert!(base.contains("v_dual_fmac_f32"));
    // One non-fused product and one add per accumulated element, each trip's and the peeled group's blocks.
    let muls = count(&e.s_text, "v_mul_f32_e32") - count(&base, "v_mul_f32_e32");
    let adds = count(&e.s_text, "v_add_f32_e32") - count(&base, "v_add_f32_e32");
    assert!(muls > 0 && muls == adds, "muls {muls} adds {adds}");
}

#[test]
fn validate_rejects_what_has_no_generator() {
    let spec = node();
    // Ring depth 3 has a generator for the dense K-loop (`tests/iu4_gemm_ring3.rs`); the fused GDN projection, whose
    // epilogue reuses the K-loop's LDS, has none, and the reason survives `emit`.
    let gdn = Spec::control(Epi::QkvzaGdn);
    let ring3 = Tune {
        ring: RingDepth::Three,
        ..Tune::default()
    };
    assert!(with(spec, ring3).validate().is_ok());
    let e = with(gdn, ring3).validate().unwrap_err();
    assert!(e.contains("ring depth 3"), "{e}");
    let e = emit(with(gdn, ring3)).err().unwrap();
    assert!(e.contains("ring depth 3"), "{e}");
    // A non-default wait placement is for the dense K-loop block only.
    let sparse = Spec {
        mma: Mma::S24X64,
        tune: Tune {
            wait: WaitPlace::MidMma,
            ..Tune::default()
        },
        ..Spec::control(Epi::Set)
    };
    let e = sparse.validate().unwrap_err();
    assert!(e.contains("wait placement mid"), "{e}");
    assert!(Spec {
        tune: Tune::default(),
        ..sparse
    }
    .validate()
    .is_ok());
}

#[test]
fn knob_strings_round_trip() {
    for (s, r) in [("2", RingDepth::Two), ("3", RingDepth::Three)] {
        assert_eq!(s.parse::<RingDepth>().unwrap(), r);
        assert_eq!(r.name(), s);
    }
    for w in WAITS {
        assert_eq!(w.name().parse::<WaitPlace>().unwrap(), w);
    }
    for f in [FoldOp::Fma, FoldOp::MulAdd] {
        assert_eq!(f.name().parse::<FoldOp>().unwrap(), f);
    }
    assert!("4".parse::<RingDepth>().is_err());
    assert!("late".parse::<WaitPlace>().is_err());
    assert!("fused".parse::<FoldOp>().is_err());
}

/// The schedule-only knobs are bit-identical to the control on the emulator (SET and ADD, token and slab, three
/// K groups so the rolled loop and the peeled tail both run).
#[test]
fn wait_placements_are_bit_identical_on_the_emulator() {
    require_rocm_tool!("llvm-mc");
    let p = Problem::synth(136, 40, 768, 0x9e37_79b9_7f4a_7c15);
    for (epi, act) in [
        (Epi::Set, ALayout::Token),
        (Epi::Set, ALayout::Slab),
        (Epi::Add, ALayout::Token),
    ] {
        let want = run_gemm(
            Spec {
                act,
                ..Spec::control(epi)
            },
            &p,
        )
        .unwrap();
        for wait in [WaitPlace::MidMma, WaitPlace::BeforeMma] {
            let got = run_gemm(
                Spec {
                    act,
                    tune: Tune {
                        wait,
                        ..Tune::default()
                    },
                    ..Spec::control(epi)
                },
                &p,
            )
            .unwrap();
            assert_eq!(
                bitwise_eq(&got, &want),
                0,
                "{epi:?} {act:?} wait {}",
                wait.name()
            );
        }
    }
}

/// `FetchPlace::PairW` (the next block's two weight slabs fetched back to back into a third 4-VGPR set) is
/// bit-identical to the control on the emulator for SET, ADD and the SiLU gate/up, with every wait placement, over K
/// of one group (peeled tail only), two (one loop trip) and three (rolled loop + tail); and it is refused where it
/// has no generator.
#[test]
fn pairw_fetch_is_bit_identical_on_the_emulator() {
    use peacemaker_isa::kernels::iu4_gemm::FetchPlace;
    require_rocm_tool!("llvm-mc");
    for k in [256usize, 512, 768] {
        let p = Problem::synth(136, 40, k, 0x9e37_79b9_7f4a_7c15 ^ k as u64);
        for (epi, act) in [
            (Epi::Set, ALayout::Token),
            (Epi::Set, ALayout::Slab),
            (Epi::Add, ALayout::Slab),
        ] {
            let want = run_gemm(
                Spec {
                    act,
                    ..Spec::control(epi)
                },
                &p,
            )
            .unwrap();
            for wait in [WaitPlace::AfterMma, WaitPlace::MidMma, WaitPlace::Window] {
                let got = run_gemm(
                    Spec {
                        act,
                        tune: Tune {
                            wait,
                            fetch: FetchPlace::PairW,
                            ..Tune::default()
                        },
                        ..Spec::control(epi)
                    },
                    &p,
                )
                .unwrap();
                assert_eq!(
                    bitwise_eq(&got, &want),
                    0,
                    "{epi:?} {act:?} k={k} wait {}",
                    wait.name()
                );
            }
        }
    }
    let ring3 = Spec {
        tune: Tune {
            ring: RingDepth::Three,
            fetch: FetchPlace::PairW,
            ..Tune::default()
        },
        ..Spec::control(Epi::Set)
    };
    assert!(ring3.validate().unwrap_err().contains("pairw"));
    let gdn = Spec {
        tune: Tune {
            fetch: FetchPlace::PairW,
            ..Tune::default()
        },
        ..Spec::control(Epi::QkvzaGdn)
    };
    assert!(gdn.validate().unwrap_err().contains("pairw"));
    assert_eq!("pairw".parse::<FetchPlace>().unwrap(), FetchPlace::PairW);
    assert!("pair".parse::<FetchPlace>().is_err());
}

/// `Tune::idle` (skip the compute of the empty token half's waves): its names parse, it appends `-idle<name>` to the
/// variant name and keeps the symbol (the product ABI), every admitted spec emits, assembles and is deterministic and
/// distinct from the default (and `skip` from `skiphi`), a module of two such kernels assembles (the epilogue labels
/// stay unique), and it is refused where the skip has no generator.
#[test]
fn idle_half_knob_names_emits_and_refuses() {
    assert_eq!("off".parse::<IdleHalf>().unwrap(), IdleHalf::Off);
    assert_eq!("skip".parse::<IdleHalf>().unwrap(), IdleHalf::Skip);
    assert_eq!("skiphi".parse::<IdleHalf>().unwrap(), IdleHalf::SkipHi);
    assert!("hi".parse::<IdleHalf>().is_err());
    assert_eq!(IdleHalf::default(), IdleHalf::Off);
    let idle_tune = |idle, wait| Tune {
        idle,
        wait,
        ..Tune::default()
    };
    // Admitted: dense 128x128x8, ring 2, wait after|window, every epilogue family (the fused projection only with
    // plain QKV stores), both activation layouts where the epilogue has them.
    for (epi, act, qkv) in [
        (Epi::Set, ALayout::Token, QkvStore::Gdn),
        (Epi::Set, ALayout::Slab, QkvStore::Gdn),
        (Epi::Add, ALayout::Token, QkvStore::Gdn),
        (Epi::Add, ALayout::Slab, QkvStore::Gdn),
        (Epi::GateUpSilu, ALayout::Slab, QkvStore::Gdn),
        (Epi::GateUpSiluBf16, ALayout::Slab, QkvStore::Gdn),
        (Epi::AddSigmoid, ALayout::Token, QkvStore::Gdn),
        (Epi::QkvzaGdn, ALayout::Token, QkvStore::Set),
        (Epi::QkvzaGdn, ALayout::Slab, QkvStore::Set),
    ] {
        let base = Spec {
            act,
            tune: Tune {
                qkv,
                ..Tune::default()
            },
            ..Spec::control(epi)
        };
        let mut seen = vec![emit(base).unwrap().s_text];
        for wait in [WaitPlace::AfterMma, WaitPlace::Window] {
            for idle in [IdleHalf::Skip, IdleHalf::SkipHi] {
                let s = Spec {
                    tune: Tune {
                        qkv,
                        ..idle_tune(idle, wait)
                    },
                    ..base
                };
                s.validate()
                    .unwrap_or_else(|e| panic!("{}: {e}", s.variant_name()));
                assert_eq!(s.symbol(), base.symbol());
                assert!(
                    s.variant_name().contains(&format!("-idle{}", idle.name())),
                    "{}",
                    s.variant_name()
                );
                let e = emit(s).unwrap_or_else(|e| panic!("{}: {e}", s.variant_name()));
                peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
                assert_eq!(
                    emit(s).unwrap().s_text,
                    e.s_text,
                    "{} is not deterministic",
                    s.variant_name()
                );
                assert!(
                    !seen.contains(&e.s_text),
                    "{} emits the text of an earlier variant",
                    s.variant_name()
                );
                seen.push(e.s_text);
            }
        }
    }
    // `idle = off` is the default: the same bytes and name.
    let base = Spec::control(Epi::Set);
    assert_eq!(
        with(base, idle_tune(IdleHalf::Off, WaitPlace::AfterMma)).variant_name(),
        base.variant_name()
    );
    // Idle kernels of different symbols share one module: their local labels stay unique (the whole
    // variant name qualifies them) and the default kernel beside them keeps its own tag.
    let a = Spec {
        tune: idle_tune(IdleHalf::Skip, WaitPlace::Window),
        ..Spec::control(Epi::Set)
    };
    let b = Spec {
        tune: idle_tune(IdleHalf::SkipHi, WaitPlace::AfterMma),
        ..Spec::control(Epi::Add)
    };
    let (text, _) = module(
        &[
            emit(a).unwrap(),
            emit(b).unwrap(),
            emit(Spec::control(Epi::GateUpSilu)).unwrap(),
        ],
        "idle",
    )
    .unwrap();
    peacemaker_isa::native::assemble(&text, a.arch).unwrap();
    // Refused: wait placements that interleave the publish between WMMA groups, ring depth 3, the 16-wave tile, the
    // GDN-preparation projection (its epilogue exchanges across waves through LDS), the 2:4-sparse blocks.
    for (spec, what) in [
        (
            Spec {
                tune: idle_tune(IdleHalf::Skip, WaitPlace::MidMma),
                ..Spec::control(Epi::Set)
            },
            "wait",
        ),
        (
            Spec {
                tune: idle_tune(IdleHalf::Skip, WaitPlace::BeforeMma),
                ..Spec::control(Epi::Set)
            },
            "wait",
        ),
        (
            Spec {
                tune: Tune {
                    ring: RingDepth::Three,
                    idle: IdleHalf::Skip,
                    ..Tune::default()
                },
                ..Spec::control(Epi::Set)
            },
            "ring depth 2",
        ),
        (
            Spec {
                tile: Tile::T256x128x16,
                tune: idle_tune(IdleHalf::SkipHi, WaitPlace::AfterMma),
                ..Spec::control(Epi::Set)
            },
            "Tile::T128x128x8",
        ),
        (
            Spec {
                tune: idle_tune(IdleHalf::Skip, WaitPlace::AfterMma),
                ..Spec::control(Epi::QkvzaGdn)
            },
            "GDN-preparation",
        ),
        (
            Spec {
                mma: Mma::S24X64,
                tune: idle_tune(IdleHalf::Skip, WaitPlace::AfterMma),
                ..Spec::control(Epi::Set)
            },
            "dense",
        ),
    ] {
        let e = spec.validate().unwrap_err();
        assert!(e.contains(what), "{}: {e}", spec.variant_name());
    }
}

/// `fold_op = muladd` reproduces the non-fused CPU fold bit for bit, and (on this data) is not the fused result.
#[test]
fn muladd_matches_the_non_fused_cpu_fold_bitwise() {
    require_rocm_tool!("llvm-mc");
    let p = Problem::synth(136, 40, 768, 0x9e37_79b9_7f4a_7c15);
    for (epi, act) in [(Epi::Set, ALayout::Token), (Epi::Add, ALayout::Slab)] {
        let spec = Spec {
            act,
            tune: Tune {
                fold_op: FoldOp::MulAdd,
                ..Tune::default()
            },
            ..Spec::control(epi)
        };
        let got = run_gemm(spec, &p).unwrap();
        let add = epi == Epi::Add;
        assert_eq!(
            bitwise_eq(&got, &reference_with(&p, add, false)),
            0,
            "{epi:?} {act:?}: muladd must equal the non-fused fold"
        );
        assert!(
            bitwise_eq(&got, &reference_with(&p, add, true)) > 0,
            "{epi:?} {act:?}: this data does not separate fused from non-fused"
        );
    }
}

/// `WaitPlace::Window` (the shipped 27B gate/up `_b1s` export, `Tune::shipped_gate_up_slab`): a distinct schedule that
/// emits and assembles, keeps the symbol, names itself `-wwindow-fschain`, round-trips its knob string, and is refused where
/// it has no generator (ring depth 3, the 16-wave tile, the sparse K-loop).
#[test]
fn window_placement_emits_and_is_refused_where_it_has_no_generator() {
    let spec = Spec {
        act: ALayout::Slab,
        ..Spec::control(Epi::GateUpSilu)
    };
    let window = with(spec, Tune::shipped_gate_up_slab());
    assert_eq!(window.tune.wait, WaitPlace::Window);
    window.validate().unwrap();
    assert_eq!(window.symbol(), spec.symbol());
    assert_eq!(
        window.variant_name(),
        "k128-128x128x8-cacc1-slab-silu-wwindow-fschain"
    );
    let e = emit(window).unwrap();
    peacemaker_isa::native::assemble(&e.s_text, window.arch).unwrap();
    assert_eq!(emit(window).unwrap().s_text, e.s_text, "not deterministic");
    for wait in WAITS {
        let other = emit(with(
            spec,
            Tune {
                wait,
                ..Tune::default()
            },
        ))
        .unwrap();
        assert_ne!(other.s_text, e.s_text, "{} equals window", wait.name());
    }
    // B1 stays one signal/wait pair (split around the WMMA group); no barrier is added or dropped.
    let count = |t: &str, m: &str| t.lines().filter(|l| l.trim_start().starts_with(m)).count();
    assert_eq!(
        count(&e.s_text, "s_barrier_signal"),
        count(&emit(spec).unwrap().s_text, "s_barrier_signal")
    );
    assert_eq!("window".parse::<WaitPlace>().unwrap(), WaitPlace::Window);
    assert_eq!(WaitPlace::Window.name(), "window");
    // The window placement is generated for the dense K-loop at ring depth 2 on the 128x128x8 tile only.
    let ring3 = with(
        spec,
        Tune {
            ring: RingDepth::Three,
            wait: WaitPlace::Window,
            ..Tune::default()
        },
    );
    let msg = ring3.validate().unwrap_err();
    assert!(
        msg.contains("window") && msg.contains("ring depth 2"),
        "{msg}"
    );
    assert!(emit(ring3).is_err());
    let t256 = Spec {
        tile: Tile::T256x128x16,
        tune: Tune::shipped_gate_up_slab(),
        ..Spec::control(Epi::GateUpSilu)
    };
    let msg = t256.validate().unwrap_err();
    assert!(
        msg.contains("window") && msg.contains("T128x128x8"),
        "{msg}"
    );
    let sparse = Spec {
        mma: Mma::S24X64,
        tune: Tune::shipped_gate_up_slab(),
        ..Spec::control(Epi::Set)
    };
    let msg = sparse.validate().unwrap_err();
    assert!(msg.contains("wait placement window"), "{msg}");
}

/// `FoldSched::Chain` (the shipped gate/up export's fold, F3-01-lead1rot-prio): a distinct schedule at every wait
/// placement that emits and assembles, keeps the symbol, raises wave priority to 2 at each block head and drops it to 1
/// at each fold, is selected only by the shipped gate/up tune, and is refused where it has no generator.
#[test]
fn chain_fold_schedule_emits_and_is_refused_where_it_has_no_generator() {
    assert_eq!("chain".parse::<FoldSched>().unwrap(), FoldSched::Chain);
    assert_eq!("paired".parse::<FoldSched>().unwrap(), FoldSched::Paired);
    assert!("fused".parse::<FoldSched>().is_err());
    assert_eq!(FoldSched::Chain.name(), "chain");
    assert_eq!(Tune::shipped_gate_up_slab().fold_sched, FoldSched::Chain);
    for epi in [
        Epi::Set,
        Epi::Add,
        Epi::GateUpSilu,
        Epi::GateUpSiluBf16,
        Epi::QkvzaGdn,
    ] {
        for act in [ALayout::Token, ALayout::Slab] {
            let want = if epi == Epi::GateUpSilu && act == ALayout::Slab {
                Tune::shipped_gate_up_slab()
            } else {
                Tune::default()
            };
            assert_eq!(
                peacemaker_isa::kernels::iu4_gemm::module_tune(Tile::T128x128x8, act, epi),
                want,
                "{epi:?} {act:?}"
            );
        }
    }
    let spec = node();
    let count = |t: &str, m: &str| t.lines().filter(|l| l.trim() == m).count();
    let paired = emit(spec).unwrap();
    for wait in [
        WaitPlace::AfterMma,
        WaitPlace::MidMma,
        WaitPlace::BeforeMma,
        WaitPlace::Window,
    ] {
        let chain = with(
            spec,
            Tune {
                wait,
                fold_sched: FoldSched::Chain,
                ..Tune::default()
            },
        );
        chain.validate().unwrap();
        assert_eq!(chain.symbol(), spec.symbol());
        assert!(
            chain.variant_name().ends_with("-fschain"),
            "{}",
            chain.variant_name()
        );
        let e = emit(chain).unwrap();
        peacemaker_isa::native::assemble(&e.s_text, chain.arch).unwrap();
        assert_eq!(emit(chain).unwrap().s_text, e.s_text, "not deterministic");
        let plain = emit(with(
            spec,
            Tune {
                wait,
                ..Tune::default()
            },
        ))
        .unwrap();
        assert_ne!(
            e.s_text,
            plain.s_text,
            "{} chain equals paired",
            wait.name()
        );
        // Four K-loop blocks (m0, m1, t0, t1): a priority-2 head each, a priority-1 fold each.
        assert_eq!(count(&e.s_text, "s_setprio 2"), 4, "{}", wait.name());
        assert_eq!(
            count(&e.s_text, "s_setprio 1"),
            count(&plain.s_text, "s_setprio 1") + 4
        );
        assert_eq!(count(&plain.s_text, "s_setprio 2"), 0);
    }
    assert_eq!(count(&paired.s_text, "s_setprio 2"), 0);
    // Chain is generated for the dense k128 128x128x8 cacc1 fused-accumulate fold at ring depth 2 only.
    let bad = |tune: Tune, spec: Spec| {
        let msg = with(spec, tune).validate().unwrap_err();
        assert!(msg.contains("fold schedule chain"), "{msg}");
    };
    let chain = Tune {
        fold_sched: FoldSched::Chain,
        ..Tune::default()
    };
    // The wider tile supports neither packed bf16 nor slab activations, so use the
    // plain token-layout SiLU control to reach the chain-specific refusals.
    let silu = Spec::control(Epi::GateUpSilu);
    bad(
        Tune {
            ring: RingDepth::Three,
            ..chain
        },
        silu,
    );
    bad(
        Tune {
            fold_op: FoldOp::MulAdd,
            ..chain
        },
        silu,
    );
    bad(
        chain,
        Spec {
            tile: Tile::T256x128x16,
            ..silu
        },
    );
    bad(
        chain,
        Spec {
            mma: Mma::S24X64,
            ..Spec::control(Epi::Set)
        },
    );
}

/// `QkvStore::Set` (the fused projection's QKV tiles store like SET): a distinct `..._qkvzaset` symbol that emits and
/// assembles, differs from the GDN-prep kernel, shares its kernel arguments, and is refused for every other epilogue.
#[test]
fn qkv_store_set_is_a_distinct_fused_symbol() {
    use peacemaker_isa::kernels::iu4_gemm::QkvStore;
    let gdn = Spec {
        act: ALayout::Slab,
        ..Spec::control(Epi::QkvzaGdn)
    };
    let set = with(
        gdn,
        Tune {
            qkv: QkvStore::Set,
            ..Tune::default()
        },
    );
    set.validate().unwrap();
    assert_eq!(set.symbol(), "gemm_mq4g256v2_residual_mmq_iu4_qkvzaset_b1s");
    assert_eq!(gdn.symbol(), "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s");
    assert_eq!(
        set.variant_name(),
        "k128-128x128x8-cacc1-slab-qkvzagdn-qset"
    );
    assert_eq!(
        format!("{:?}", set.kernargs()),
        format!("{:?}", gdn.kernargs())
    );
    let (a, b) = (emit(gdn).unwrap(), emit(set).unwrap());
    peacemaker_isa::native::assemble(&b.s_text, set.arch).unwrap();
    assert_ne!(a.s_text, b.s_text);
    // The GDN preparation is gone: no GDN label, and the K-loop is the GDN kernel's.
    assert!(a.s_text.contains(".Liu4_gdn:"));
    assert!(!b.s_text.contains(".Liu4_gdn:"));
    assert_eq!(emit(set).unwrap().s_text, b.s_text, "deterministic");
    for epi in [Epi::Set, Epi::Add, Epi::GateUpSilu] {
        let e = with(
            Spec::control(epi),
            Tune {
                qkv: QkvStore::Set,
                ..Tune::default()
            },
        )
        .validate()
        .unwrap_err();
        assert!(e.contains("qkv store set"), "{e}");
    }
    assert_eq!("set".parse::<QkvStore>().unwrap(), QkvStore::Set);
    assert_eq!(QkvStore::Gdn.name(), "gdn");
    assert!("fold".parse::<QkvStore>().is_err());
}

/// `Tune::n64` names the small-N (`_v64`) export: it appends `_v64` to the symbol and `-n64` to the variant name, changes
/// no code by itself, and is refused outside the dense slab 128x128x8 module. The `verify64_tune` schedules are what the
/// four `_v64` exports of the slab module carry.
#[test]
fn n64_names_the_small_n_export() {
    let base = Spec {
        act: ALayout::Slab,
        ..Spec::control(Epi::Set)
    };
    let named = with(
        base,
        Tune {
            n64: true,
            ..Tune::default()
        },
    );
    named.validate().unwrap();
    assert_eq!(
        named.symbol(),
        "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1s_v64"
    );
    assert_eq!(named.variant_name(), format!("{}-n64", base.variant_name()));
    // Only the symbol's name differs: the code is the shipped export's.
    assert_eq!(
        emit(named).unwrap().s_text,
        emit(base)
            .unwrap()
            .s_text
            .replace(&base.symbol(), &named.symbol())
    );
    let token = with(Spec::control(Epi::Set), named.tune);
    let e = token.validate().unwrap_err();
    assert!(e.contains("n64"), "{e}");
}

/// The new export passes the certifier (lift/emit identity, wait and hazard replay, barrier pairing).
#[cfg(feature = "toolchain")]
#[test]
fn qkv_store_set_passes_m7() {
    use peacemaker_isa::kernels::iu4_gemm::QkvStore;
    for act in [ALayout::Token, ALayout::Slab] {
        let s = Spec {
            act,
            tune: Tune {
                qkv: QkvStore::Set,
                ..Tune::default()
            },
            ..Spec::control(Epi::QkvzaGdn)
        };
        let e = emit(s).unwrap();
        let elf = peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
        let path = std::env::temp_dir().join(format!(
            "iu4_qset_{}_{}.elf",
            std::process::id(),
            s.variant_name()
        ));
        std::fs::write(&path, &elf).unwrap();
        let receipt = peacemaker_isa::pm_check::m7(&path, "gfx1201", &s.symbol())
            .unwrap_or_else(|e| panic!("{}: {e}", s.variant_name()));
        std::fs::remove_file(&path).ok();
        println!("{} {receipt}", s.variant_name());
    }
}

/// Every `Tune::idle` kernel of the slab module's exports passes the certifier with zero obligations (the skip bodies'
/// registers are defined on every path from the entry: `kloop::define_dead_regs`). `m7` returns `Err` on any obligation.
#[cfg(feature = "toolchain")]
#[test]
fn idle_half_passes_m7() {
    use peacemaker_isa::kernels::iu4_gemm::{IdleHalf, QkvStore};
    for (epi, qkv) in [
        (Epi::Set, QkvStore::Gdn),
        (Epi::Add, QkvStore::Gdn),
        (Epi::GateUpSilu, QkvStore::Gdn),
        (Epi::QkvzaGdn, QkvStore::Set),
    ] {
        for wait in [WaitPlace::AfterMma, WaitPlace::Window] {
            for idle in [IdleHalf::Skip, IdleHalf::SkipHi] {
                let s = Spec {
                    act: ALayout::Slab,
                    tune: Tune {
                        wait,
                        idle,
                        qkv,
                        ..Tune::default()
                    },
                    ..Spec::control(epi)
                };
                let e = emit(s).unwrap_or_else(|e| panic!("{}: {e}", s.variant_name()));
                let elf = peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
                let path = std::env::temp_dir().join(format!(
                    "iu4_idle_{}_{}.elf",
                    std::process::id(),
                    s.variant_name()
                ));
                std::fs::write(&path, &elf).unwrap();
                let receipt = peacemaker_isa::pm_check::m7(&path, "gfx1201", &s.symbol())
                    .unwrap_or_else(|e| panic!("{}: {e}", s.variant_name()));
                std::fs::remove_file(&path).ok();
                assert!(
                    receipt["obligations"]
                        .as_object()
                        .is_none_or(|o| o.is_empty()),
                    "{}: {receipt}",
                    s.variant_name()
                );
            }
        }
    }
}
