// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 planar MQ3-I4 / MQ2-I4 one-row decode GEMV module `mq_i4_rows_pm_gfx1201`
//! (`mq_i4_rows_{w3,w2}_{r256,r128}`): symbols, 32-byte kernarg ABI, exact-arch
//! gating, the instruction census of the frozen DAG, the committed contract
//! identities, and (feature `toolchain`) wait replay, M7 and the
//! radiowave certification of every symbol.
use peacemaker_isa::kernels::mq_i4_rows::{self, Codec, Rotation, Spec};
use peacemaker_isa::Arch;
#[cfg(feature = "toolchain")]
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "mq_i4_rows_pm_gfx1201";
const SYMBOLS: [&str; 4] = [
    "mq_i4_rows_w3_r256",
    "mq_i4_rows_w3_r128",
    "mq_i4_rows_w2_r256",
    "mq_i4_rows_w2_r128",
];
/// Contract ABI: `codes x y` pointers, then `M K` u32.
const ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4)];

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}
fn spec(codec: Codec, rotation: Rotation) -> Spec {
    Spec {
        arch: ARCH,
        codec,
        rotation,
    }
}

#[test]
fn module_exports_are_the_four_frozen_symbols_in_order() {
    let specs = mq_i4_rows::module_specs(ARCH);
    assert_eq!(
        specs,
        [
            spec(Codec::W3, Rotation::R256),
            spec(Codec::W3, Rotation::R128),
            spec(Codec::W2, Rotation::R256),
            spec(Codec::W2, Rotation::R128)
        ]
    );
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), SYMBOLS);
    assert_eq!(Spec::module(ARCH), MODULE);
    let (emitted, text, proof) = mq_i4_rows::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    assert_eq!(text.matches(".amdhsa_kernel ").count(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: not global"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
}

#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in mq_i4_rows::module_specs(ARCH) {
        let (a, b) = (
            mq_i4_rows::emit(spec).unwrap(),
            mq_i4_rows::emit(spec).unwrap(),
        );
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
    }
    let (_, ta, pa) = mq_i4_rows::emit_module(ARCH).unwrap();
    let (_, tb, pb) = mq_i4_rows::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        for spec in mq_i4_rows::module_specs(arch) {
            assert!(mq_i4_rows::emit(spec).is_err(), "{arch:?} {spec:?}");
        }
        assert!(mq_i4_rows::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// Every symbol is wave32, one wave per workgroup, without scratch, spills or LDS.
#[test]
fn every_export_is_wave32_without_scratch_or_lds() {
    let (emitted, text, _) = mq_i4_rows::emit_module(ARCH).unwrap();
    for (e, symbol) in emitted.iter().zip(SYMBOLS) {
        assert_eq!(e.proof.arch, ARCH, "{symbol}");
        assert!(
            e.s_text.contains(".amdhsa_wavefront_size32 1"),
            "{symbol}: wave32"
        );
        assert!(
            e.s_text.contains(".private_segment_fixed_size: 0\n"),
            "{symbol}: private segment"
        );
        assert!(
            e.s_text.contains(".sgpr_spill_count: 0\n")
                && e.s_text.contains(".vgpr_spill_count: 0\n"),
            "{symbol}: spills"
        );
        assert!(
            e.s_text.contains(".group_segment_fixed_size: 0\n"),
            "{symbol}: no LDS"
        );
        assert_eq!(e.shape.barriers, 0, "{symbol}: barriers");
        let ds = e.s_text.lines().find(|l| {
            l.split_whitespace()
                .next()
                .is_some_and(|m| m.starts_with("ds_load") || m.starts_with("ds_store"))
        });
        assert!(ds.is_none(), "{symbol}: LDS access {ds:?}");
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1201"));
}

/// The kernarg block, per-argument offsets/sizes, access and the launch bound of every export.
#[test]
fn kernarg_abi_matches_the_contract() {
    assert_eq!(mq_i4_rows::KERNARG_BYTES, 32);
    assert_eq!(mq_i4_rows::WORKGROUP_SIZE, 32);
    for spec in mq_i4_rows::module_specs(ARCH) {
        let e = mq_i4_rows::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert!(
            e.s_text.contains(".kernarg_segment_size: 32\n"),
            "{symbol}: kernarg size"
        );
        assert_eq!(
            field(&e.s_text, ".offset:"),
            ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{symbol}: kernarg offsets"
        );
        assert_eq!(
            field(&e.s_text, ".size:"),
            ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{symbol}: kernarg sizes"
        );
        assert_eq!(
            field(&e.s_text, ".max_flat_workgroup_size:"),
            [32],
            "{symbol}: block 32"
        );
        let (last_off, last_size) = *ARGS.last().unwrap();
        assert_eq!(
            last_off + last_size,
            mq_i4_rows::KERNARG_BYTES,
            "{symbol}: arguments cover the kernarg block"
        );
        // codes and x are read-only, y write-only (LLVM key order: the first key carries the list dash).
        for (name, access) in [
            ("codes", "read_only"),
            ("x", "read_only"),
            ("y", "write_only"),
        ] {
            assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{symbol}: {name} access {access}");
        }
        // The first loads: all five arguments in one 32-byte scalar load from kernarg offset 0.
        assert!(
            e.s_text.contains("s_load_b256 s[8:15], s[0:1], 0x0"),
            "{symbol}: kernarg load"
        );
    }
}

/// The frozen DAG's census per symbol: codec packet loads, the weight/dot/accumulate
/// chain per group, the R256 four-chain fold vs the R128 single chain, five snapshot shuffle
/// steps with out-of-range lanes left unchanged (sum into scratch, then a mask select),
/// one lane-0 store. W3 reads two bytes per lane packet, W2 one; there is no
/// buffer access, MQ4 XOR rebias or fma-mix dequant.
#[test]
fn instruction_census_matches_the_frozen_dag() {
    for spec in mq_i4_rows::module_specs(ARCH) {
        let t = mq_i4_rows::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        // R256: four chains in the quad loop plus three tail slots; R128: one group body.
        let groups = match spec.rotation {
            Rotation::R256 => 7,
            Rotation::R128 => 1,
        };
        let packet_loads = match spec.codec {
            Codec::W3 => 2,
            Codec::W2 => 1,
        };
        assert_eq!(
            count(&t, "global_load_u8"),
            packet_loads * groups,
            "{symbol}: packet byte loads"
        );
        assert_eq!(
            count(&t, "global_load_u16"),
            groups,
            "{symbol}: scale loads"
        );
        assert_eq!(count(&t, "global_load_b128"), groups, "{symbol}: x loads");
        assert_eq!(count(&t, "global_store_b32"), 1, "{symbol}: store");
        assert_eq!(
            count(&t, "v_bfe_u32"),
            4 * groups,
            "{symbol}: code extracts"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_i32_e32"),
            4 * groups,
            "{symbol}: weight integer converts"
        );
        assert_eq!(
            count(&t, "v_cvt_f32_f16_e64"),
            groups,
            "{symbol}: scale converts"
        );
        // Weight multiplies (4) + first dot product term (1) per group; three FMAs per group.
        assert_eq!(
            count(&t, "v_mul_f32_e32"),
            5 * groups,
            "{symbol}: multiplies"
        );
        assert_eq!(
            count(&t, "v_fmac_f32_e32"),
            3 * groups,
            "{symbol}: fused steps"
        );
        // One group-partial add per group, the R256 fold's three adds, five shuffle adds.
        let fold = match spec.rotation {
            Rotation::R256 => 3,
            Rotation::R128 => 0,
        };
        assert_eq!(
            count(&t, "v_add_f32_e32"),
            groups + fold + 5,
            "{symbol}: adds"
        );
        assert_eq!(count(&t, "ds_bpermute_b32"), 5, "{symbol}: shuffle steps");
        assert_eq!(
            count(&t, "v_lshl_or_b32"),
            if spec.codec == Codec::W3 { groups } else { 0 },
            "{symbol}: W3 packet combine"
        );
        for forbidden in [
            "buffer_",
            "ds_swizzle",
            "v_fma_mix_f32",
            "v_xor_b32",
            "flat_",
            "scratch_",
            "v_wmma",
        ] {
            let hit = t.lines().find(|l| {
                l.split_whitespace()
                    .next()
                    .is_some_and(|m| m.starts_with(forbidden))
            });
            assert!(hit.is_none(), "{symbol}: emits {hit:?}");
        }
        // Signed grid 2c - L: the add of -L (never an XOR rebias).
        assert!(
            t.contains(&format!(
                "v_add_nc_u32_e32 v12, -{}, v12",
                spec.codec.levels()
            )),
            "{symbol}: 2c-L"
        );
        // Shuffle offsets 16, 8, 4, 2, 1 in that order.
        let offsets: Vec<u32> = t
            .lines()
            .filter(|l| l.trim().starts_with("v_cndmask_b32_e64 v4, 0,"))
            .map(|l| l.trim().split(", ").nth(2).unwrap().parse().unwrap())
            .collect();
        assert_eq!(offsets, [16, 8, 4, 2, 1], "{symbol}: shuffle offsets");
        // Out-of-range lanes stay unchanged: each step sums into the scratch register, then
        // v_cndmask keeps the old accumulator where the step's compare mask is clear.
        assert_eq!(
            count(&t, "v_cndmask_b32_e64"),
            10,
            "{symbol}: address + accumulator selects"
        );
        assert_eq!(
            t.lines()
                .filter(|l| l.trim() == "v_cndmask_b32_e64 v20, v20, v24, vcc_lo")
                .count(),
            5,
            "{symbol}: accumulator selects"
        );
        assert_eq!(
            t.lines()
                .filter(|l| l.trim() == "v_add_f32_e32 v24, v20, v24")
                .count(),
            5,
            "{symbol}: shuffle sums into scratch"
        );
        assert!(
            !t.lines().any(|l| l.trim() == "v_add_f32_e32 v20, v20, v24"),
            "{symbol}: unconditional shuffle add"
        );
    }
}

#[test]
fn codec_and_rotation_constants_match_the_planar_format() {
    assert_eq!(
        (
            Codec::W3.bits(),
            Codec::W3.levels(),
            Codec::W3.group_bytes()
        ),
        (3, 7, 48)
    );
    assert_eq!(
        (
            Codec::W2.bits(),
            Codec::W2.levels(),
            Codec::W2.group_bytes()
        ),
        (2, 3, 32)
    );
    assert_eq!((Rotation::R256.width(), Rotation::R128.width()), (256, 128));
    assert_eq!(mq_i4_rows::GROUP, 128);
}

/// The committed shape contracts are exactly what `contract()` derives from fresh emission
/// (`peacemaker-isa emit --kernel mq_i4_rows ... --co ... --shape crates/peacemaker-isa/kernels`).
#[test]
fn committed_contracts_equal_fresh_contracts() {
    for spec in mq_i4_rows::module_specs(ARCH) {
        let path = format!(
            "{}/kernels/mq_i4_rows.{}.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            ARCH.name(),
            spec.tag()
        );
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
                .unwrap();
        let fresh = mq_i4_rows::contract(spec, &mq_i4_rows::emit(spec).unwrap());
        assert_eq!(committed, fresh, "{path}");
        assert_eq!(committed["symbol"], spec.symbol().as_str());
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// Every symbol's waits replay independently, and M7 lifts the natively linked
    /// object byte-exactly with no outstanding obligations (no ROCm tools).
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = mq_i4_rows::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-mqi4rows-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 =
                pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every symbol certifies against the ROCm oracle build: parse-back, independent wait replay,
    /// its committed shape contract, and M7 of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = mq_i4_rows::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-mqi4rows-cert-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        for spec in mq_i4_rows::module_specs(ARCH) {
            let symbol = spec.symbol();
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
            let tag = spec.tag();
            let path = format!(
                "{}/kernels/mq_i4_rows.{}.{tag}.contract.json",
                env!("CARGO_MANIFEST_DIR"),
                ARCH.name()
            );
            let contract = serde_json::from_slice(
                &std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
            )
            .unwrap();
            certify(
                &toolchain,
                &build,
                &s,
                ARCH.name(),
                &dir.join(format!("{tag}.manifest.json")),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
