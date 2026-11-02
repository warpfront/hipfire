// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WeightCodec::E8pI4` NT1 entries of the grouped symmetric IU4 MoE GEMM
//! (gfx1201): spec validation, entry ABI and the 1,536-byte per-CTA LDS stage,
//! the `2C + s` / quarter fold arithmetic and the once-per-CTA publication.
//! Qualification-only route: nothing here claims a G1 or Gp result.
use peacemaker_isa::kernels::iu4_fold::MAGIC;
use peacemaker_isa::kernels::qwen4_moe_sym::{
    self, Spec,
    WeightCodec::{E8pI4, Mq3I4},
};
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

/// `LUT[256]` (one u32 of eight nibbles per abs-grid row) plus `MASK[128]`
/// (one u32 per sign7): 1,024 + 512 bytes, staged once per CTA.
const LDS_STAGE_BYTES: u32 = 256 * 4 + 128 * 4;
/// The 64-byte kernarg block (60-byte frozen ABI plus `n_experts@60`) shared with the MQ4 and W3 NT1 entries.
const KERNARG_BYTES: u32 = 64;

fn e8p_specs() -> Vec<Spec> {
    qwen4_moe_sym::module_specs_codec(Arch::Gfx1201, E8pI4)
}

/// The E8P codec is built for the gfx1201 NT1 entries only; the same gate
/// rejects every other arch, tile width and row repeat.
#[test]
fn spec_validation_is_gfx1201_nt1_only() {
    let specs = e8p_specs();
    assert_eq!(specs.len(), 2);
    assert_eq!(
        (specs[0].kind, specs[1].kind),
        (qwen4_moe_sym::Kind::GateUp, qwen4_moe_sym::Kind::Down)
    );
    for spec in &specs {
        assert_eq!(
            (spec.arch, spec.nt, spec.rr, spec.codec),
            (Arch::Gfx1201, 1, 1, E8pI4)
        );
        spec.validate().unwrap();
    }
    for arch in [Arch::Gfx1151, Arch::Gfx1100] {
        assert!(
            qwen4_moe_sym::module_specs_codec(arch, E8pI4).is_empty(),
            "{arch:?}"
        );
    }
    for spec in &specs {
        for bad in [
            Spec {
                arch: Arch::Gfx1151,
                ..*spec
            },
            Spec {
                arch: Arch::Gfx1100,
                ..*spec
            },
            Spec { nt: 4, ..*spec },
            Spec { nt: 8, ..*spec },
            Spec { rr: 2, ..*spec },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
            assert!(qwen4_moe_sym::emit(bad).is_err(), "{bad:?}");
        }
    }
}

/// Own module, own symbols (`_e8p` infix) and variant tags; the 64-byte ABI,
/// the 1,536-byte LDS reservation and the thread count agree between the
/// `Spec` and the emitted descriptor/metadata of the shipped and `_acc` entries.
#[test]
fn entry_abi_names_and_lds_stage() {
    let specs = e8p_specs();
    assert_eq!(
        specs[0].symbol(),
        "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1201"
    );
    assert_eq!(specs[1].symbol(), "qwen4_moe_down_iu4_sym_e8p_pm_gfx1201");
    assert_eq!(
        Spec::module_codec(Arch::Gfx1201, E8pI4),
        "qwen4_moe_iu4_sym_e8p_pm_gfx1201"
    );
    assert_ne!(
        specs[0].variant(),
        Spec {
            codec: Mq3I4,
            ..specs[0]
        }
        .variant()
    );
    for spec in specs {
        assert_eq!(spec.lds_bytes(), LDS_STAGE_BYTES, "{spec:?}");
        assert_eq!(spec.kernargs().size, KERNARG_BYTES, "{spec:?}");
        assert_eq!(
            spec.kernargs()
                .args
                .last()
                .map(|a| (a.name.as_str(), a.offset, a.size)),
            Some(("n_experts", 60, 4)),
            "{spec:?}"
        );
        let acc = qwen4_moe_sym::e8p::emit_acc(spec).unwrap();
        let symbol = spec.symbol();
        for (e, name) in [
            (qwen4_moe_sym::emit(spec).unwrap(), symbol.clone()),
            (acc, format!("{symbol}_acc")),
        ] {
            let s = &e.s_text;
            assert!(s.contains(&format!(".name: {name}\n")), "{name}");
            assert!(
                s.contains(&format!(
                    ".amdhsa_group_segment_fixed_size {LDS_STAGE_BYTES}\n"
                )),
                "{name}: LDS directive"
            );
            assert!(
                s.contains(&format!(".group_segment_fixed_size: {LDS_STAGE_BYTES}\n")),
                "{name}: LDS metadata"
            );
            assert!(
                s.contains(&format!(".kernarg_segment_size: {KERNARG_BYTES}\n")),
                "{name}: kernarg size"
            );
            assert!(
                s.contains(&format!(".max_flat_workgroup_size: {}\n", spec.threads())),
                "{name}: threads"
            );
        }
    }
}

/// The rc5 expert-range guard: `n_experts` (kernarg 0x3c) is loaded into s5 and the routed expert id (s23) is
/// compared with ONE unsigned `s_cmp_ge_u32 s23, s5` that exits before the `EID * 8` pointer index, the
/// expert-pointer load, the first buffer load and the table publication barrier. The old signed negative
/// compare is gone, so the -1 sentinel, INT_MIN and any id `>= n_experts` skip the tile with no store.
#[test]
fn expert_guard_is_one_unsigned_compare_before_pointer_load_barrier_and_loads() {
    for spec in e8p_specs() {
        let acc = qwen4_moe_sym::e8p::emit_acc(spec).unwrap();
        for e in [qwen4_moe_sym::emit(spec).unwrap(), acc] {
            let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
            let at = |p: &str| lines.iter().position(|l| l.starts_with(p));
            assert_eq!(
                lines
                    .iter()
                    .filter(|l| l.starts_with("s_cmp_ge_u32 s23, s5"))
                    .count(),
                1,
                "{spec:?}: one unsigned guard"
            );
            assert!(
                !lines.iter().any(|l| l.starts_with("s_cmp_lt_i32 s23, 0")),
                "{spec:?}: old signed compare"
            );
            let load = lines
                .iter()
                .position(|l| l.starts_with("s_load_b32 s5, s[0:1], 0x3c"))
                .expect("n_experts load");
            let cmp = at("s_cmp_ge_u32 s23, s5").expect("guard compare");
            assert!(load < cmp, "{spec:?}: n_experts is loaded before the guard");
            assert!(
                lines[cmp + 1].starts_with("s_cbranch_scc1"),
                "{spec:?}: the guard branches to the exit"
            );
            let pointer_index = at("s_lshl_b32 s4, s23, 3").expect("expert pointer index");
            let pointer_load = at("s_load_b64 s[58:59], s[6:7], null")
                .or_else(|| at("s_load_b64 s[58:59], s[6:7], 0x0"))
                .expect("expert pointer load");
            let first_buffer = at("buffer_load").expect("first buffer load");
            let barrier = at("s_barrier").expect("table barrier");
            for (what, i) in [
                ("pointer index", pointer_index),
                ("pointer load", pointer_load),
                ("buffer load", first_buffer),
                ("barrier", barrier),
            ] {
                assert!(cmp < i, "{spec:?}: guard must precede the {what}");
            }
        }
    }
}

/// Emission is deterministic; the shipped entry stages the table through LDS
/// once per CTA: stores, then one uniform barrier, then the lookups. Every
/// entry has exactly one barrier (no NT>1 epilogue handoff exists at NT1).
#[test]
fn staging_is_published_once_per_cta() {
    for spec in e8p_specs() {
        let (a, b) = (
            qwen4_moe_sym::emit(spec).unwrap(),
            qwen4_moe_sym::emit(spec).unwrap(),
        );
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        let acc = qwen4_moe_sym::e8p::emit_acc(spec).unwrap();
        for e in [&a, &acc] {
            assert_eq!(e.shape.barriers, 1, "{spec:?}");
            let s = &e.s_text;
            let (store, barrier, load) =
                (s.find("ds_store"), s.find("s_barrier"), s.find("ds_load"));
            let (store, barrier, load) = (
                store.expect("staging ds_store"),
                barrier.expect("publication barrier"),
                load.expect("table ds_load"),
            );
            assert!(
                store < barrier && barrier < load,
                "{spec:?}: stage -> barrier -> lookup order"
            );
            // `s_barrier` (gfx11) / `s_barrier_signal`+`s_barrier_wait` (gfx12): one publication point.
            assert_eq!(s.matches("s_barrier_signal").count(), 1, "{spec:?}");
            assert_eq!(s.matches("s_barrier_wait").count(), 1, "{spec:?}");
        }
    }
}

/// The fold network on the emitted text: the magic-seeded accumulator is folded
/// by one `v_lshl_add_u32 a', a, 1, s - MAGIC` (`a' = 2a + (s - MAGIC)`), and the
/// converted integer is quartered by the exact power-of-two `0.25`. No FP16 `d/4`
/// is formed: the quarter is applied to the F32 integer-side result.
#[test]
fn fold_uses_two_c_plus_s_and_the_exact_quarter() {
    for spec in e8p_specs() {
        for e in [
            qwen4_moe_sym::emit(spec).unwrap(),
            qwen4_moe_sym::e8p::emit_acc(spec).unwrap(),
        ] {
            let s = &e.s_text;
            let shl1 = s
                .lines()
                .filter(|l| {
                    let l = l.trim();
                    l.starts_with("v_lshl_add_u32 ")
                        && l.split(',').nth(2).map(str::trim) == Some("1")
                })
                .count();
            assert!(
                shl1 > 0,
                "{spec:?}: no `v_lshl_add_u32 dst, a, 1, s_minus_magic` fold"
            );
            assert!(
                s.contains("0.25") || s.contains("0x3e800000"),
                "{spec:?}: no exact-quarter multiply"
            );
            assert!(
                s.contains("0xcb400000"),
                "{spec:?}: no magic-subtract constant"
            );
            assert!(!s.contains("v_cvt_f16_f32"), "{spec:?}: FP16 d/4 formed");
        }
    }
}

/// `a' = 2a + (s - MAGIC)` with `a = MAGIC + C` (mod 2^32) gives
/// `MAGIC + 2C + s`; subtracting the magic gives the exact F32 `2C + s`, and
/// the quarter is exact. `|C| <= 6*8*128`, `|s| <= 8*128` for IU4 `o` in
/// `[-6, 5]` and `q` in `[-8, 7]`.
#[test]
fn two_c_plus_s_fold_and_quarter_are_exact() {
    let magic_f = f32::from_bits(MAGIC);
    for c in -6144i32..=6144 {
        for s in [
            -1024i32, -1023, -513, -255, -64, -7, -1, 0, 1, 5, 64, 127, 511, 1023,
        ] {
            let a = MAGIC.wrapping_add(c as u32);
            let folded = a
                .wrapping_shl(1)
                .wrapping_add((s as u32).wrapping_sub(MAGIC));
            assert_eq!(
                folded,
                MAGIC.wrapping_add((2 * c + s) as u32),
                "c {c} s {s}"
            );
            let t = f32::from_bits(folded) - magic_f;
            assert_eq!(t, (2 * c + s) as f32, "c {c} s {s}");
            assert_eq!(
                (t * 0.25) as f64,
                f64::from(2 * c + s) * 0.25,
                "c {c} s {s}"
            );
        }
    }
    // Weight side: v = 2o + 1 for every IU4 operand, so sum(v*q) = 2C + s.
    for o in -6i32..=5 {
        for q in -8i32..=7 {
            assert_eq!((2 * o + 1) * q, 2 * (o * q) + q);
        }
    }
}

/// The SWAR decode `w = (word ^ m) + (m & 0x1111_1111) - (shift ? 0x1111_1111 : 0)`
/// on nibbles of `2|g| ∈ {1,3,5}` gives each lane's `o = sign * 2|g| - shift`
/// in `[-6, 5]` with no carry or borrow across nibbles, for every lane/sign/shift.
#[test]
fn swar_sign_and_shift_decode_matches_per_lane_model() {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..200_000 {
        let r = next();
        let (mut word, mut m, mut want) = (0u32, 0u32, [0i32; 8]);
        let shift = (r >> 60) & 1 == 1;
        for k in 0..8 {
            let a = [1u32, 3, 5][((r >> (2 * k)) & 3) as usize % 3];
            let neg = (r >> (16 + k)) & 1 == 1;
            word |= a << (4 * k);
            if neg {
                m |= 0xf << (4 * k);
            }
            want[k] = (if neg { -(a as i32) } else { a as i32 }) - i32::from(shift);
        }
        let w = (word ^ m)
            .wrapping_add(m & 0x1111_1111)
            .wrapping_sub(if shift { 0x1111_1111 } else { 0 });
        for k in 0..8 {
            let nib = ((w >> (4 * k)) & 15) as i32;
            assert_eq!(
                ((nib << 28) >> 28),
                want[k],
                "lane {k} word {word:08x} mask {m:08x} shift {shift}"
            );
            assert!((-6..=5).contains(&want[k]));
        }
    }
}

/// The E8P route adds an own module and changes no byte of the MQ4 or W3 ones:
/// neither mentions `_e8p`, the E8P module mentions no `_w3`, and both anchors
/// stay deterministic (the committed-bundle equality for MQ4/W3 is
/// `committed_bundles_equal_fresh_emission` / `committed_w3_bundle_equals_fresh_emission`
/// in `qwen4_moe_sym.rs`).
#[test]
fn module_is_isolated_from_mq4_and_w3() {
    let arch = Arch::Gfx1201;
    let (emitted, text, proof) = qwen4_moe_sym::emit_module_codec(arch, E8pI4).unwrap();
    assert_eq!(emitted.len(), 4, "shipped + qualification entry per kind");
    assert_eq!(proof.module, "qwen4_moe_iu4_sym_e8p_pm_gfx1201");
    for spec in e8p_specs() {
        let symbol = spec.symbol();
        assert!(
            text.contains(&format!(".name: {symbol}\n"))
                && text.contains(&format!(".name: {symbol}_acc\n")),
            "{symbol}"
        );
    }
    // The W3 codec namespace is its `_sym_w3_` symbols and `.Lq4s_<kind>_w3[acc]_` labels; a bare `_w3` also
    // matches the E8P stage label `.Lq4s_down_e8p_stage_w3` (wave 3, not the codec).
    for kind in ["gate_up", "down"] {
        for needle in [
            format!("_sym_w3_"),
            format!(".Lq4s_{kind}_w3_"),
            format!(".Lq4s_{kind}_w3acc_"),
        ] {
            assert!(
                !text.contains(&needle),
                "E8P module mentions the W3 codec namespace: {needle}"
            );
        }
    }
    assert_eq!(
        text,
        qwen4_moe_sym::emit_module_codec(arch, E8pI4).unwrap().1,
        "module emission is deterministic"
    );
    let mq4 = qwen4_moe_sym::emit_module(arch).unwrap().1;
    let w3 = qwen4_moe_sym::emit_module_codec(arch, Mq3I4).unwrap().1;
    for other in [&mq4, &w3] {
        assert!(
            !other.contains("_e8p"),
            "an existing module mentions the E8P entries"
        );
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, Toolchain},
    };

    /// All four E8P symbols (shipped and `_acc`) replay their wait ledger and
    /// lift byte-exact with no M7 obligations. Qualification of the entries only;
    /// numerics are a separate receipt.
    #[test]
    fn e8p_module_passes_m7_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let arch = Arch::Gfx1201;
        let (emitted, text, _) = qwen4_moe_sym::emit_module_codec(arch, E8pI4).unwrap();
        assert_eq!(emitted.len(), 4);
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, arch).unwrap();
        }
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-q4s-e8p-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let build = build(
            &Toolchain::oracle(),
            &s,
            &dir.join("module.hsaco"),
            arch.name(),
        )
        .unwrap();
        for spec in e8p_specs() {
            let base = spec.symbol();
            for symbol in [base.clone(), format!("{base}_acc")] {
                let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                    .unwrap_or_else(|e| panic!("{symbol}: {e}"));
                assert_eq!(
                    (m7["lift"].as_str(), &m7["obligations"]),
                    (Some("byte-exact"), &serde_json::json!({})),
                    "{symbol}"
                );
            }
        }
    }
}
