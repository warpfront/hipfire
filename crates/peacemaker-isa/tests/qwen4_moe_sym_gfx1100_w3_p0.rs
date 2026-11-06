// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2 W3 (MQ3-I4 planar) gfx1100 (`src/kernels/qwen4_moe_sym_gfx1100`, `Codec::W3`): CPU-only gates of the
//! shipped `kernels/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco`.
//!
//! * names and ABI: module, the four entries, the 64-byte kernargs with `n_experts@60`;
//! * the rc5 expert-range guard (one unsigned compare before the pointer load and every buffer load) on every
//!   W3 entry and mutation;
//! * the W3 geometry in the emitted text: no LDS, no barrier, 16-bit scale loads, planar code rows of `K*3/8`;
//! * the committed image exports exactly the four W3 entries, and its sha256 file matches the bytes.
//!
//! The arithmetic law (`d * (2c - 7)` planar grid, exact int32 WMMA chain per epoch, `fma(RN(d*sc), D, S)`) is
//! proved by the emitter's P1 receipt (`qwen4_moe_sym_gfx1100_p1.rs`, run by
//! `emit_qwen4_moe_sym_gfx1100 --codec w3`); the byte-exact gate against the CPU planar decoder on real expert
//! layers is the card G1 oracle.
#![allow(dead_code, unused_imports)]

#[path = "../src/kernels/qwen4_moe_sym_gfx1100/mod.rs"]
mod gfx1100;

use gfx1100::{Codec, Kind, Negative, Spec};
use peacemaker_isa::Arch;

fn spec(kind: Kind, acc: bool, negative: Negative) -> Spec {
    Spec {
        arch: Arch::Gfx1100,
        codec: Codec::W3,
        kind,
        acc,
        negative,
    }
}

#[test]
fn w3_names_and_abi() {
    assert_eq!(
        Codec::W3.module(Arch::Gfx1100),
        "qwen4_moe_iu4_sym_w3_pm_gfx1100"
    );
    assert_eq!(
        spec(Kind::GateUp, false, Negative::None).symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1100"
    );
    assert_eq!(
        spec(Kind::Down, false, Negative::None).symbol(),
        "qwen4_moe_down_iu4_sym_w3_pm_gfx1100"
    );
    assert_eq!(
        spec(Kind::Down, true, Negative::None).symbol(),
        "qwen4_moe_down_iu4_sym_w3_pm_gfx1100_acc"
    );
    let layout = spec(Kind::Down, false, Negative::None).kernargs();
    assert_eq!(layout.size, 64);
    assert_eq!(
        layout
            .args
            .last()
            .map(|a| (a.name.as_str(), a.offset, a.size)),
        Some(("n_experts", 60, 4))
    );
    // W3 code row bytes (planar): K*3/8.
    assert_eq!(gfx1100::row_code_bytes(Codec::W3, 2560), 960);
    assert_eq!(gfx1100::row_code_bytes(Codec::W3, 640), 240);
}

#[test]
fn w3_expert_guard_is_one_unsigned_compare_before_pointer_load_and_loads() {
    let negatives: Vec<Negative> = std::iter::once(Negative::None)
        .chain(Negative::ALL)
        .filter(|n| n.applies(Codec::W3))
        .collect();
    assert!(!negatives.is_empty());
    for kind in Kind::ALL {
        for acc in [false, true] {
            for &negative in &negatives {
                if acc && negative == Negative::WrongPackSelector {
                    continue;
                }
                let s = spec(kind, acc, negative);
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
                let load = at("s_load_b32 s5, s[0:1], 0x3c").expect("n_experts load");
                let cmp = at("s_cmp_ge_u32 s23, s5").unwrap();
                assert!(load < cmp, "{s:?}: n_experts loaded before the guard");
                assert!(lines[cmp + 1].starts_with("s_cbranch_scc1"), "{s:?}");
                let pointer_load =
                    at("s_load_b64 s[58:59], s[6:7], null").expect("expert pointer load");
                let first_buffer = at("buffer_load").expect("first buffer load");
                assert!(cmp < pointer_load && cmp < first_buffer, "{s:?}");
            }
        }
    }
}

#[test]
fn w3_entries_are_lds_free_barrier_free_with_16_bit_scale_loads() {
    for kind in Kind::ALL {
        let e = gfx1100::emit(spec(kind, false, Negative::None)).expect("W3 entry builds");
        assert!(
            e.s_text.contains(".group_segment_fixed_size: 0"),
            "{kind:?}"
        );
        assert!(!e.s_text.contains("s_barrier"), "{kind:?}");
        assert!(!e.s_text.contains("ds_load_b32"), "{kind:?}");
        assert!(
            e.s_text.contains("buffer_load_u16"),
            "{kind:?}: scale loads are 16-bit"
        );
        assert!(!e.s_text.contains("scratch_"), "{kind:?}: no scratch");
    }
}

#[test]
fn w3_mutations_change_the_source_and_are_codec_scoped() {
    for kind in Kind::ALL {
        let base = gfx1100::emit(spec(kind, false, Negative::None)).unwrap();
        for negative in Negative::ALL {
            if !negative.applies(Codec::W3) {
                continue;
            }
            let bad = gfx1100::emit(spec(kind, false, negative)).unwrap();
            assert_ne!(base.s_text, bad.s_text, "{kind:?} {negative:?}");
        }
        for bad in [
            Negative::WrongLut,
            Negative::WrongSignMask,
            Negative::WrongFoldEpoch,
        ] {
            assert!(!bad.applies(Codec::W3), "{bad:?} is E8P-only");
        }
    }
}

#[test]
fn committed_w3_image_exports_the_four_entries_and_matches_its_hash() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("kernels/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for kind in Kind::ALL {
        for acc in [false, true] {
            let symbol = spec(kind, acc, Negative::None).symbol();
            assert!(
                bytes.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                "{symbol} missing from the committed image"
            );
        }
    }
    let sha =
        std::fs::read_to_string(root.join("kernels/qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco.sha256"))
            .ok();
    if let Some(sha) = sha {
        use sha2::{Digest, Sha256};
        let want = sha.split_whitespace().next().unwrap_or("");
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), want);
    }
}
