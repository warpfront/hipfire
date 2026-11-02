// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every valid `iu4_gemm::Spec` re-emits the bytes recorded before the
//! building blocks (`iu4_gemm::blocks`) were factored out of `emit`: the
//! sha256 of `emit(spec).s_text` per spec, computed from origin/beta's unrefactored
//! `iu4_gemm::emit` (a scratch worktree of origin/beta, this file copied in). A change to any digest is a change
//! to a production kernel and must be intentional.

use peacemaker_isa::kernels::iu4_gemm::{
    emit, emit_module, module_tune, verify64_tune, ALayout, Cacc, Epi, Fold, Mma, QkvStore, Spec,
    Tile, Tune, WaitPlace,
};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

/// `(variant name, sha256 of s_text)` for every valid spec, in iteration order.
const PINNED: &[(&str, &str)] = &[
    (
        "k128-128x128x8-cacc1-set",
        "079960944bdfc28bb64cf80c89de13438d64f9b51ece918ecbb08d8e86099fc0",
    ),
    (
        "k128-128x128x8-cacc1-slab-set",
        "e769dd8109c5b66783661b507a297add3b20286acd3e119be3201f911c9b183f",
    ),
    (
        "k128-128x128x8-cacc1-add",
        "3ae2eeab50318099623b726019a0d08b7a3b7af623ea89898f32198b9484df92",
    ),
    (
        "k128-128x128x8-cacc1-slab-add",
        "dff9618681be419f4df102f9e75e1ebf19779ff4408449f749520366413fb45e",
    ),
    (
        "k128-128x128x8-cacc1-addsigmoid",
        "4af41d09171edbc9065b613c54a0d0c35a085e19bc0333072f2bb018ed37bfd0",
    ),
    (
        "k128-128x128x8-cacc1-silu",
        "eb12a394f3a62f8454fbd977b7e6b257503f440fce456749ac9aaa933d482e19",
    ),
    (
        "k128-128x128x8-cacc1-slab-silu",
        "3f50cdeff5482a41a923d3150871963c9025c9b6968fb084a381819d64df4924",
    ),
    (
        "k128-128x128x8-cacc1-silu-bf16",
        "0142a4330302a5a9a32d222d259c84836a26e97380099d72514e6d1bfc7aa428",
    ),
    (
        "k128-128x128x8-cacc1-slab-silu-bf16",
        "3e388e47115501d1de4e32108b3b779814a03713729d62a837eddf0880c46066",
    ),
    (
        "k128-128x128x8-cacc1-qkvzagdn",
        "a26cedbe8c421d48fd31cb587a48fe863aae2b9afcc29397961ab4a15523367f",
    ),
    (
        "k128-128x128x8-cacc1-slab-qkvzagdn",
        "a26b3207951c032b948024294970af5900bdff204900d83fa7a28a1eaa66acf0",
    ),
    (
        "k128-256x128x16-cacc1-set",
        "28489b418ce74a6a8378c9ca0d5f09b79d27aed906164c38cc90febe229ddf0b",
    ),
    (
        "k128-256x128x16-cacc1-add",
        "d246a5009d97d897b3e0747d26e6f7947a0edf7a0eaf982be73158b15036d899",
    ),
    (
        "k128-256x128x16-cacc1-silu",
        "0177ce76b08e295200c647b2dc3a56b0e0d9522c3696c417f0e1803337e4a934",
    ),
];

fn digests() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for fold in [Fold::K128, Fold::K256Shared, Fold::K256Pow2] {
        for tile in [Tile::T128x128x8, Tile::T256x128x16] {
            for cacc in [Cacc::One, Cacc::Two] {
                for epi in [
                    Epi::Set,
                    Epi::Add,
                    Epi::AddSigmoid,
                    Epi::GateUpSilu,
                    Epi::GateUpSiluBf16,
                    Epi::QkvzaGdn,
                ] {
                    for act in [ALayout::Token, ALayout::Slab] {
                        let spec = Spec {
                            fold,
                            tile,
                            cacc,
                            epi,
                            act,
                            arch: Arch::Gfx1201,
                            mma: Mma::Dense,
                            tune: Tune::default(),
                        };
                        if spec.validate().is_err() {
                            continue;
                        }
                        let e = emit(spec).unwrap_or_else(|e| {
                            panic!("{} is valid but does not emit: {e}", spec.variant_name())
                        });
                        out.push((
                            spec.variant_name(),
                            format!("{:x}", Sha256::digest(e.s_text.as_bytes())),
                        ));
                    }
                }
            }
        }
    }
    out
}

#[test]
fn iu4_gemm_emit_digests_pinned() {
    let got = digests();
    if std::env::var_os("IU4_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, PINNED, "iu4_gemm emitted bytes changed");
}

/// The shipped 27B gate/up export (`gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`, `Tune::shipped_gate_up_slab`): the sha256 of
/// its `emit(spec).s_text` and of the symbol's code in the committed `_b1s` module (see `native_identity.rs`).
const PINNED_GATE_UP_SLAB_WINDOW: (&str, &str) = (
    "k128-128x128x8-cacc1-slab-silu-wwindow-fschain",
    "baf26d529895ede82d7484ed851ced347fa777c28bc9c869d219bbf6a67fdacc",
);

fn gate_up_slab(tune: Tune) -> Spec {
    Spec {
        fold: Fold::K128,
        tile: Tile::T128x128x8,
        cacc: Cacc::One,
        epi: Epi::GateUpSilu,
        act: ALayout::Slab,
        arch: Arch::Gfx1201,
        mma: Mma::Dense,
        tune,
    }
}

#[test]
fn iu4_gemm_gate_up_slab_window_digest_pinned() {
    let spec = gate_up_slab(Tune::shipped_gate_up_slab());
    assert_eq!(spec.tune.wait, WaitPlace::Window);
    let e = emit(spec).unwrap();
    assert_eq!(spec.variant_name(), PINNED_GATE_UP_SLAB_WINDOW.0);
    assert_eq!(
        format!("{:x}", Sha256::digest(e.s_text.as_bytes())),
        PINNED_GATE_UP_SLAB_WINDOW.1,
        "the shipped gate/up window schedule changed"
    );
}

/// The module emission gives the window tune to that one export and the default tune to every other symbol of every
/// module (token and slab, both tiles): each module symbol's bytes are `emit` of its spec with `module_tune`, and only
/// the 128x128x8 slab `silu` one has a non-default tune.
#[test]
fn only_the_gate_up_slab_export_takes_the_window_tune() {
    for tile in [Tile::T128x128x8, Tile::T256x128x16] {
        for act in [ALayout::Token, ALayout::Slab] {
            // The 16-wave tile has no slab layout: no such module.
            let Ok((emitted, _, _)) = emit_module(Fold::K128, tile, Cacc::One, act, Arch::Gfx1201)
            else {
                assert_eq!((tile, act), (Tile::T256x128x16, ALayout::Slab));
                continue;
            };
            let mut matched = 0;
            for epi in [
                Epi::Set,
                Epi::Add,
                Epi::GateUpSilu,
                Epi::GateUpSiluBf16,
                Epi::QkvzaGdn,
                Epi::AddSigmoid,
            ] {
                let tune = module_tune(tile, act, epi);
                let spec = Spec {
                    epi,
                    tile,
                    act,
                    tune,
                    ..gate_up_slab(Tune::default())
                };
                if spec.validate().is_err() {
                    continue;
                }
                let Some(e) = emitted
                    .iter()
                    .find(|e| e.proof.variant == spec.variant_name())
                else {
                    continue;
                };
                matched += 1;
                let windowed =
                    tile == Tile::T128x128x8 && act == ALayout::Slab && epi == Epi::GateUpSilu;
                assert_eq!(tune.is_default(), !windowed, "{}", e.proof.variant);
                assert_eq!(e.s_text, emit(spec).unwrap().s_text, "{}", e.proof.variant);
            }
            // The slab module's extra export: the fused projection with plain QKV stores.
            if tile == Tile::T128x128x8 && act == ALayout::Slab {
                let spec = Spec {
                    epi: Epi::QkvzaGdn,
                    tile,
                    act,
                    tune: Tune {
                        qkv: QkvStore::Set,
                        ..Tune::default()
                    },
                    ..gate_up_slab(Tune::default())
                };
                let e = emitted
                    .iter()
                    .find(|e| e.proof.variant == spec.variant_name())
                    .expect("the slab module exports the qkvzaset kernel");
                matched += 1;
                assert_eq!(e.s_text, emit(spec).unwrap().s_text, "{}", e.proof.variant);
                // The small-N `_v64` exports: the four N-keyed schedules of `verify64_tune`.
                for (epi, qkv_set) in [
                    (Epi::GateUpSilu, false),
                    (Epi::Add, false),
                    (Epi::Set, false),
                    (Epi::QkvzaGdn, true),
                ] {
                    let tune = verify64_tune(epi, qkv_set);
                    assert!(tune.n64 && !tune.is_default());
                    let spec = Spec {
                        epi,
                        tile,
                        act,
                        tune,
                        ..gate_up_slab(Tune::default())
                    };
                    assert!(spec.symbol().ends_with("_b1s_v64"), "{}", spec.symbol());
                    let e = emitted
                        .iter()
                        .find(|e| e.proof.variant == spec.variant_name())
                        .expect("the slab module exports the v64 kernels");
                    matched += 1;
                    assert_eq!(e.s_text, emit(spec).unwrap().s_text, "{}", e.proof.variant);
                }
            }
            assert_eq!(
                matched,
                emitted.len(),
                "{tile:?} {act:?}: every module symbol"
            );
        }
    }
}
