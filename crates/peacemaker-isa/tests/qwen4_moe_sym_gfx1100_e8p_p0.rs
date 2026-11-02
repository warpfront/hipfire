// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2 E8P gfx1100 (`src/kernels/qwen4_moe_sym_gfx1100/e8p.rs`): CPU-only proofs and the builder gates.
//!
//! * `e8p::prove()`: emitted per-word unpack network vs the K-E0 reference for every `(idx, sign7, shift)`, the
//!   `2C + s` identity over random epochs, the exact `2a + (s - MAGIC)` / quarter / fma fold (and the
//!   `wrong-fold-epoch` term `2C/4`), the LDS stage partition and bounds, the 32-byte epoch windows and the
//!   register layout, and that both table negatives differ from the true tables at every entry.
//! * Every E8P entry (`gate_up`/`down` x BF16/`_acc` x candidate + the six E8P mutations) must pass the
//!   builder (wait ledger, hazards, LDS typestate and barrier, VOPD pairing, register plan) and declare the
//!   1,536 B static LDS; the MQ4/W2/W3 entries still declare none, and a mutation that does not exist for a
//!   codec is refused.
#![allow(dead_code, unused_imports)]

#[path = "../src/kernels/qwen4_moe_sym_gfx1100/mod.rs"]
mod gfx1100;

use gfx1100::{Codec, Kind, Negative, Spec};
use peacemaker_isa::Arch;

fn spec(codec: Codec, kind: Kind, acc: bool, negative: Negative) -> Spec {
    Spec {
        arch: Arch::Gfx1100,
        codec,
        kind,
        acc,
        negative,
    }
}

#[test]
fn e8p_cpu_proofs_pass() {
    let p = gfx1100::e8p::prove();
    assert!(
        p.failures.is_empty(),
        "{:?}",
        &p.failures[..p.failures.len().min(8)]
    );
    assert_eq!(p.checks.len(), 7, "{:?}", p.checks);
}

#[test]
fn e8p_names_and_abi() {
    assert_eq!(
        Codec::E8p.module(Arch::Gfx1100),
        "qwen4_moe_iu4_sym_e8p_pm_gfx1100"
    );
    assert_eq!(
        spec(Codec::E8p, Kind::GateUp, false, Negative::None).symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1100"
    );
    assert_eq!(
        spec(Codec::E8p, Kind::Down, false, Negative::None).symbol(),
        "qwen4_moe_down_iu4_sym_e8p_pm_gfx1100"
    );
    assert_eq!(
        spec(Codec::E8p, Kind::Down, true, Negative::None).symbol(),
        "qwen4_moe_down_iu4_sym_e8p_pm_gfx1100_acc"
    );
    assert_eq!(gfx1100::KERNARG_BYTES, 64);
    let layout = spec(Codec::E8p, Kind::Down, false, Negative::None).kernargs();
    assert_eq!(layout.size, 64);
    assert_eq!(
        layout
            .args
            .last()
            .map(|a| (a.name.as_str(), a.offset, a.size)),
        Some(("n_experts", 60, 4))
    );
    assert_eq!(Codec::ALL.len(), 3, "the legacy module set is unchanged");
    assert_eq!(
        Negative::ALL.len(),
        4,
        "the legacy negative set is unchanged"
    );
}

/// The rc5 expert-range guard of every gfx1100 K2 entry (MQ4 and E8P, BF16 and `_acc`, every mutation): `n_experts`
/// (kernarg 0x3c) is loaded into s5 and the routed expert id s23 is range-checked with ONE unsigned
/// `s_cmp_ge_u32 s23, s5` branching to the no-store exit before the `EID * 8` (or the `WrongExpertPointer`
/// `s_mov_b32 s4, 0`) pointer index, the expert-pointer load, the first buffer load and the table barrier. The old
/// signed negative compare is gone.
#[test]
fn expert_guard_is_one_unsigned_compare_before_pointer_load_barrier_and_loads() {
    for codec in [Codec::Mq4, Codec::E8p] {
        let negatives: Vec<Negative> = std::iter::once(Negative::None)
            .chain([
                Negative::WrongScaleRow,
                Negative::ReversedEpoch,
                Negative::WrongPackSelector,
                Negative::WrongExpertPointer,
                Negative::WrongLut,
                Negative::WrongSignMask,
                Negative::WrongFoldEpoch,
            ])
            .filter(|n| n.applies(codec))
            .collect();
        for kind in Kind::ALL {
            for acc in [false, true] {
                for &negative in &negatives {
                    if acc && negative == Negative::WrongPackSelector {
                        continue;
                    }
                    let s = spec(codec, kind, acc, negative);
                    let e = gfx1100::emit(s).unwrap_or_else(|err| panic!("{s:?}: {err}"));
                    let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
                    let at = |p: &str| lines.iter().position(|l| l.starts_with(p));
                    assert!(
                        e.s_text.contains(".kernarg_segment_size: 64"),
                        "{s:?}: kernarg segment"
                    );
                    assert_eq!(
                        lines
                            .iter()
                            .filter(|l| l.starts_with("s_cmp_ge_u32 s23, s5"))
                            .count(),
                        1,
                        "{s:?}: one unsigned guard"
                    );
                    assert!(
                        !lines.iter().any(|l| l.starts_with("s_cmp_lt_i32 s23, 0")),
                        "{s:?}: old signed compare"
                    );
                    let load = at("s_load_b32 s5, s[0:1], 0x3c")
                        .unwrap_or_else(|| panic!("{s:?}: no n_experts load"));
                    let cmp = at("s_cmp_ge_u32 s23, s5").unwrap();
                    assert!(load < cmp, "{s:?}: n_experts is loaded before the guard");
                    assert!(
                        lines[cmp + 1].starts_with("s_cbranch_scc1"),
                        "{s:?}: the guard branches to the exit"
                    );
                    let pointer_index = if negative == Negative::WrongExpertPointer {
                        at("s_mov_b32 s4, 0")
                    } else {
                        at("s_lshl_b32 s4, s23, 3")
                    }
                    .expect("expert pointer index");
                    let pointer_load =
                        at("s_load_b64 s[58:59], s[6:7], null").expect("expert pointer load");
                    let first_buffer = at("buffer_load").expect("first buffer load");
                    for (what, i) in [
                        ("pointer index", pointer_index),
                        ("pointer load", pointer_load),
                        ("buffer load", first_buffer),
                    ] {
                        assert!(cmp < i, "{s:?}: guard must precede the {what}");
                    }
                    if codec == Codec::E8p {
                        let barrier = at("s_barrier").expect("table barrier");
                        assert!(cmp < barrier, "{s:?}: guard must precede the table barrier");
                    }
                }
            }
        }
    }
}

#[test]
fn e8p_entries_and_mutations_build_with_the_lds_stage() {
    for kind in Kind::ALL {
        for acc in [false, true] {
            for negative in std::iter::once(Negative::None).chain(Negative::E8P) {
                let s = spec(Codec::E8p, kind, acc, negative);
                let e = gfx1100::emit(s).unwrap_or_else(|err| panic!("{s:?}: {err}"));
                assert!(
                    e.s_text.contains(".group_segment_fixed_size: 1536"),
                    "{s:?}: missing the 1,536 B static LDS"
                );
                assert!(
                    e.s_text.contains("s_barrier"),
                    "{s:?}: missing the table publication barrier"
                );
                assert!(e.s_text.contains("ds_load_b32"), "{s:?}: no LDS table read");
            }
        }
    }
}

#[test]
fn legacy_entries_stay_lds_free_and_mutations_are_codec_scoped() {
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            let e = gfx1100::emit(spec(codec, kind, false, Negative::None))
                .expect("legacy entry builds");
            assert!(
                e.s_text.contains(".group_segment_fixed_size: 0"),
                "{codec:?} {kind:?}"
            );
            // The BF16 MQ4 gate/up (lockstep) and down (K=640 pipeline) entries carry workgroup
            // barriers, never LDS accesses; every other legacy entry has neither.
            assert_eq!(
                e.s_text.contains("s_barrier"),
                codec == Codec::Mq4,
                "{codec:?} {kind:?}"
            );
            assert!(!e.s_text.contains("ds_load_b32"), "{codec:?} {kind:?}");
            for bad in [
                Negative::WrongLut,
                Negative::WrongSignMask,
                Negative::WrongFoldEpoch,
            ] {
                assert!(
                    gfx1100::emit(spec(codec, kind, false, bad)).is_err(),
                    "{codec:?} accepted {bad:?}"
                );
            }
        }
    }
    for kind in Kind::ALL {
        assert!(gfx1100::emit(spec(Codec::E8p, kind, false, Negative::WrongPackSelector)).is_err());
    }
}

#[test]
fn e8p_mutations_change_the_source() {
    for kind in Kind::ALL {
        let base = gfx1100::emit(spec(Codec::E8p, kind, false, Negative::None))
            .unwrap()
            .s_text;
        for negative in Negative::E8P {
            let mutated = gfx1100::emit(spec(Codec::E8p, kind, false, negative))
                .unwrap()
                .s_text;
            assert_ne!(
                base, mutated,
                "{kind:?} {negative:?} left the source unchanged"
            );
        }
    }
}

#[test]
fn wrong_fold_epoch_drops_only_the_activation_sum() {
    for kind in Kind::ALL {
        for acc in [false, true] {
            let base = gfx1100::emit(spec(Codec::E8p, kind, acc, Negative::None))
                .unwrap()
                .s_text;
            let bad = gfx1100::emit(spec(Codec::E8p, kind, acc, Negative::WrongFoldEpoch))
                .unwrap()
                .s_text;
            let adds = |t: &str| t.matches("v_add_nc_u32_e32").count();
            let movs = |t: &str| t.matches("v_mov_b32_e32").count();
            assert_eq!(
                adds(&base),
                adds(&bad) + 1,
                "{kind:?} acc {acc}: exactly one `s - MAGIC` add is replaced"
            );
            assert_eq!(
                movs(&bad),
                movs(&base) + 1,
                "{kind:?} acc {acc}: by exactly one `-MAGIC` move"
            );
            assert_eq!(
                base.lines().count(),
                bad.lines().count(),
                "{kind:?} acc {acc}: instruction count unchanged"
            );
        }
    }
}

/// R2 card winners: the BF16 MQ4 gate/up entry (two lockstep barriers per two-epoch trip) and the
/// BF16 MQ4 down entry (K = 640 pipeline: one barrier after each of its five epoch issues) are the
/// only legacy entries with workgroup barriers; the `_acc` twins and W2/W3 keep the old loop.
#[test]
fn mq4_bf16_entries_carry_the_round2_barriers() {
    let barriers = |kind, acc| {
        let e = gfx1100::emit(spec(Codec::Mq4, kind, acc, Negative::None)).expect("builds");
        e.s_text
            .lines()
            .filter(|l| l.trim_start().starts_with("s_barrier"))
            .count()
    };
    assert_eq!(barriers(Kind::GateUp, false), 2);
    assert_eq!(barriers(Kind::Down, false), 5);
    assert_eq!(barriers(Kind::GateUp, true), 0);
    assert_eq!(barriers(Kind::Down, true), 0);
    for codec in [Codec::W2, Codec::W3] {
        for kind in Kind::ALL {
            let e = gfx1100::emit(spec(codec, kind, false, Negative::None)).expect("builds");
            assert!(!e.s_text.contains("s_barrier"), "{codec:?} {kind:?}");
        }
    }
    // A mutated down entry keeps the plain loop (the pipeline is the candidate's only).
    let m = gfx1100::emit(spec(Codec::Mq4, Kind::Down, false, Negative::ReversedEpoch))
        .expect("builds");
    assert!(!m.s_text.contains("s_barrier"));
}
