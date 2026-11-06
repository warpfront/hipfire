// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The F1 drop-zero planar twins (`WeightCodec::Mq4SymPlanar`) of the gfx1201 sealed IU4 symmetric MoE images:
//! `qwen4_moe_iu4_sym_p_pm_gfx1201` (gate/up and down NT4/NT8) and `qwen4_moe_down_gp_p_pm_gfx1201` (the grouped
//! producer plus the grouped-position down NT4/NT8). Names, module order, refusals, the instruction-level relation to
//! the shipped-layout entries they twin (the same fold, only the weight and scale addressing differ), and the committed
//! bundles. The numeric twin test (planar vs shipped layout through the emulator) is `qwen4_moe_sym_planar_emu.rs`.
use peacemaker_isa::kernels::qwen4_moe_sym::{self as moe, Kind, Spec, WeightCodec};
use peacemaker_isa::kernels::{moe_rotate_i4_pm, qwen4_moe_down_gp};
use peacemaker_isa::{Arch, Emitted};
use std::collections::BTreeMap;

const ARCH: Arch = Arch::Gfx1201;
const PLANAR: WeightCodec = WeightCodec::Mq4SymPlanar;

fn spec(kind: Kind, nt: u8, codec: WeightCodec) -> Spec {
    Spec {
        arch: ARCH,
        kind,
        nt,
        rr: 1,
        codec,
    }
}

/// Instruction lines of the body of `symbol`, comments and directives dropped.
fn body(e: &Emitted, symbol: &str) -> Vec<String> {
    let start = e
        .s_text
        .lines()
        .position(|l| l == format!("{symbol}:"))
        .unwrap_or_else(|| panic!("{symbol}: label"));
    e.s_text
        .lines()
        .skip(start + 1)
        .take_while(|l| !l.starts_with(".amdhsa_kernel") && !l.starts_with(".Lfunc_end"))
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with(';') && !l.starts_with('.'))
        .map(str::to_owned)
        .collect()
}

fn mnemonics(lines: &[String]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for l in lines {
        if let Some(m) = l.split_whitespace().next() {
            *out.entry(m.to_owned()).or_insert(0) += 1;
        }
    }
    out
}

#[test]
fn planar_names_symbols_and_modules() {
    assert_eq!(
        Spec::module_codec(ARCH, PLANAR),
        "qwen4_moe_iu4_sym_p_pm_gfx1201"
    );
    assert_eq!(
        spec(Kind::GateUp, 4, PLANAR).symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt4"
    );
    assert_eq!(
        spec(Kind::GateUp, 8, PLANAR).symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt8"
    );
    assert_eq!(
        spec(Kind::Down, 4, PLANAR).symbol(),
        "qwen4_moe_down_iu4_sym_p_pm_gfx1201_nt4"
    );
    assert_eq!(
        spec(Kind::Down, 8, PLANAR).symbol(),
        "qwen4_moe_down_iu4_sym_p_pm_gfx1201_nt8"
    );
    assert_eq!(
        spec(Kind::Down, 4, PLANAR).symbol_gp(),
        "qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt4"
    );
    assert_eq!(
        spec(Kind::Down, 8, PLANAR).symbol_gp(),
        "qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt8"
    );
    // The shipped codec's names are untouched.
    assert_eq!(
        spec(Kind::Down, 8, WeightCodec::Mq4Sym).symbol_gp(),
        "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8"
    );
    assert_eq!(Spec::module(ARCH), "qwen4_moe_iu4_sym_pm_gfx1201");
    assert_eq!(spec(Kind::Down, 4, PLANAR).variant(), "down_nt4_p");
}

#[test]
fn planar_module_holds_the_four_nt4_nt8_entries_in_order() {
    let specs = moe::module_specs_codec(ARCH, PLANAR);
    let symbols: Vec<String> = specs.iter().map(|s| s.symbol()).collect();
    assert_eq!(
        symbols,
        [
            "qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt4",
            "qwen4_moe_down_iu4_sym_p_pm_gfx1201_nt4",
            "qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt8",
            "qwen4_moe_down_iu4_sym_p_pm_gfx1201_nt8",
        ]
    );
    let (emitted, text, proof) = moe::emit_module_codec(ARCH, PLANAR).unwrap();
    assert_eq!(emitted.len(), 4);
    assert_eq!(proof.module, "qwen4_moe_iu4_sym_p_pm_gfx1201");
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 4);
    for symbol in &symbols {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}"
        );
        assert!(text.contains(&format!(".name: {symbol}\n")), "{symbol}");
    }
    // None of the shipped-layout exports is part of the planar module.
    for s in moe::module_specs(ARCH) {
        assert!(
            !text.contains(&format!(".amdhsa_kernel {}\n", s.symbol())),
            "{}",
            s.symbol()
        );
    }
    // 64-byte ABI, grid and LDS as the shipped entries.
    for (e, s) in emitted.iter().zip(&specs) {
        let shipped = Spec {
            codec: WeightCodec::Mq4Sym,
            ..*s
        };
        assert_eq!(
            serde_json::to_value(s.kernargs()).unwrap(),
            serde_json::to_value(shipped.kernargs()).unwrap()
        );
        assert_eq!(s.threads(), shipped.threads());
        assert_eq!(s.lds_bytes(), shipped.lds_bytes());
        assert_eq!(e.proof.variant, format!("{}_p", shipped.variant()));
    }
}

#[test]
fn planar_emission_is_deterministic_and_refuses_everything_but_gfx1201_nt4_nt8() {
    let a = moe::emit_module_codec(ARCH, PLANAR).unwrap();
    let b = moe::emit_module_codec(ARCH, PLANAR).unwrap();
    assert!(a.1 == b.1);
    assert_eq!(
        serde_json::to_value(&a.2).unwrap(),
        serde_json::to_value(&b.2).unwrap()
    );
    for arch in [Arch::Gfx1151, Arch::Gfx1100] {
        assert!(moe::emit_module_codec(arch, PLANAR).is_err(), "{arch:?}");
        assert!(moe::module_specs_codec(arch, PLANAR).is_empty(), "{arch:?}");
        for kind in Kind::ALL {
            for nt in [1u8, 4] {
                assert!(
                    moe::emit(Spec {
                        arch,
                        kind,
                        nt,
                        rr: 1,
                        codec: PLANAR
                    })
                    .is_err(),
                    "{arch:?} {kind:?} nt{nt}"
                );
            }
        }
    }
    for kind in Kind::ALL {
        for nt in [1u8, 2] {
            assert!(
                moe::emit(spec(kind, nt, PLANAR)).is_err(),
                "{kind:?} nt{nt}"
            );
        }
        let mut rr = spec(kind, 4, PLANAR);
        rr.rr = 2;
        assert!(moe::emit(rr).is_err());
    }
    // The grouped-position twin is a down NT4/NT8 entry only.
    assert!(moe::emit_down_gp(spec(Kind::GateUp, 4, PLANAR)).is_err());
    assert!(moe::emit_down_gp(spec(Kind::Down, 1, PLANAR)).is_err());
    assert!(qwen4_moe_down_gp::emit_module_codec(Arch::Gfx1151, PLANAR).is_err());
    assert!(qwen4_moe_down_gp::emit_module_codec(ARCH, WeightCodec::Mq3I4).is_err());
}

/// The planar entry is its shipped-layout twin with only the weight/scale addressing replaced: every vector ALU, WMMA,
/// DS, activation/weight-nibble load, store and barrier instruction is the same (counted by mnemonic), the 32-bit header
/// word load becomes a 16-bit scale load, and nothing but scalar address arithmetic differs.
fn assert_planar_twin(shipped: &Emitted, ssym: &str, planar: &Emitted, psym: &str) {
    let (sb, pb) = (body(shipped, ssym), body(planar, psym));
    let (sm, pm) = (mnemonics(&sb), mnemonics(&pb));
    let get = |m: &BTreeMap<String, usize>, k: &str| m.get(k).copied().unwrap_or(0);
    assert!(get(&pm, "buffer_load_u16") > 0, "{psym}: no scale loads");
    assert_eq!(get(&sm, "buffer_load_u16"), 0, "{ssym}");
    assert_eq!(
        get(&pm, "buffer_load_b32") + get(&pm, "buffer_load_u16"),
        get(&sm, "buffer_load_b32"),
        "{psym}: the header word loads become scale loads one for one"
    );
    // Every scale load reads the scale descriptor (s[84:87]) at an even byte offset below the 4-byte trip step.
    for l in pb.iter().filter(|l| l.starts_with("buffer_load_u16")) {
        assert!(l.contains("s[84:87]"), "{l}");
        assert!(
            l.ends_with("null offen") || l.ends_with("null offen offset:2"),
            "{l}"
        );
    }
    let mut names: Vec<&String> = sm.keys().chain(pm.keys()).collect();
    names.sort();
    names.dedup();
    for n in names {
        if get(&sm, n) == get(&pm, n) || n == "buffer_load_b32" || n == "buffer_load_u16" {
            continue;
        }
        assert!(
            n.starts_with("s_") && !n.starts_with("s_barrier"),
            "{psym}: `{n}` differs ({} shipped vs {} planar)",
            get(&sm, n),
            get(&pm, n)
        );
    }
    assert_eq!(get(&sm, "s_barrier_signal"), get(&pm, "s_barrier_signal"));
    // Row bytes are no longer scaled by the 136/68 group size.
    assert!(sb.iter().any(|l| l.starts_with("s_mulk_i32 s46, 0x")));
    assert!(!pb.iter().any(|l| l.starts_with("s_mulk_i32 s46, 0x")));
}

#[test]
fn flat_planar_entries_keep_the_shipped_fold() {
    for kind in Kind::ALL {
        for nt in [4u8, 8] {
            let (s, p) = (spec(kind, nt, WeightCodec::Mq4Sym), spec(kind, nt, PLANAR));
            let (es, ep) = (moe::emit(s).unwrap(), moe::emit(p).unwrap());
            assert_planar_twin(&es, &s.symbol(), &ep, &p.symbol());
        }
    }
}

#[test]
fn grouped_position_planar_down_keeps_the_shipped_gp_fold() {
    for nt in [4u8, 8] {
        let (s, p) = (
            spec(Kind::Down, nt, WeightCodec::Mq4Sym),
            spec(Kind::Down, nt, PLANAR),
        );
        let (es, ep) = (moe::emit_down_gp(s).unwrap(), moe::emit_down_gp(p).unwrap());
        assert_planar_twin(&es, &s.symbol_gp(), &ep, &p.symbol_gp());
        // The planar gp twin is the planar flat entry with only the activation gather replaced.
        let flat = moe::emit(p).unwrap();
        let (gb, fb) = (body(&ep, &p.symbol_gp()), body(&flat, &p.symbol()));
        let (gm, fm) = (mnemonics(&gb), mnemonics(&fb));
        assert_eq!(
            gm.get("buffer_load_u16"),
            fm.get("buffer_load_u16"),
            "nt{nt}"
        );
        assert_eq!(
            gm.get("v_wmma_i32_16x16x32_iu4"),
            fm.get("v_wmma_i32_16x16x32_iu4")
        );
    }
}

#[test]
fn down_gp_planar_module_exports_the_producer_and_the_two_planar_twins() {
    assert_eq!(
        qwen4_moe_down_gp::module_name_codec(ARCH, PLANAR),
        "qwen4_moe_down_gp_p_pm_gfx1201"
    );
    assert_eq!(
        qwen4_moe_down_gp::module_name(ARCH),
        "qwen4_moe_down_gp_pm_gfx1201"
    );
    let symbols = qwen4_moe_down_gp::symbols_codec(ARCH, PLANAR);
    assert_eq!(
        symbols,
        [
            "moe_rotate_i4_gp_pm_gfx1201",
            "qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt4",
            "qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt8",
        ]
    );
    let (emitted, text, proof) = qwen4_moe_down_gp::emit_module_codec(ARCH, PLANAR).unwrap();
    assert_eq!(emitted.len(), 3);
    assert_eq!(proof.module, "qwen4_moe_down_gp_p_pm_gfx1201");
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 3);
    for symbol in &symbols {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}"
        );
    }
    // The producer is byte for byte the shipped module's (the A4 plane does not depend on the weight layout).
    let shipped = qwen4_moe_down_gp::emit_module(ARCH).unwrap().0;
    assert_eq!(
        body(&emitted[0], &moe_rotate_i4_pm::symbol_gp(ARCH)),
        body(&shipped[0], &moe_rotate_i4_pm::symbol_gp(ARCH))
    );
}

fn bundle(path: &str, text: &str) {
    let committed = std::fs::read(path).unwrap_or_else(|e| {
        panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel qwen4_moe_sym_p|qwen4_moe_down_gp_p_pm --arch gfx1201 ... --bundle)")
    });
    let elf = peacemaker_isa::native::assemble(text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "fresh emission differs from the committed bundle {path}"
    );
}

/// The committed planar images are exactly what the builder emits today.
#[test]
fn committed_planar_bundles_equal_fresh_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let (_, text, _) = moe::emit_module_codec(ARCH, PLANAR).unwrap();
    bundle(
        &format!("{root}/qwen4_moe_iu4_sym_p_pm_gfx1201.hxaco"),
        &text,
    );
    let (_, text, _) = qwen4_moe_down_gp::emit_module_codec(ARCH, PLANAR).unwrap();
    bundle(
        &format!("{root}/qwen4_moe_down_gp_p_pm_gfx1201.hxaco"),
        &text,
    );
}
