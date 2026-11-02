// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QSA selector BF16-pooled score module (gfx1151): the two symbols
//! `indexed_attention_select_scores_rows16_bf16_pm_gfx1151` and
//! `indexed_attention_select_scores_rows32_bf16_pm_gfx1151` (rows16 first) in module
//! `qsa_select_bf16_pm_gfx1151`. The F32 pair module carries its own rows16/rows32 scores.
use peacemaker_isa::kernels::{
    qsa_score,
    qsa_select::{self, Kind, Spec},
};
use peacemaker_isa::Arch;

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;
/// The host target stamped into the committed bundles (trailing hyphen).
const HOST: &str = "host-x86_64-unknown-linux-gnu-";
const SYMBOL: &str = "indexed_attention_select_scores_rows16_bf16_pm_gfx1151";
const SYMBOL32: &str = "indexed_attention_select_scores_rows32_bf16_pm_gfx1151";
const MODULE: &str = "qsa_select_bf16_pm_gfx1151";

fn bf16() -> (
    Vec<peacemaker_isa::Emitted>,
    String,
    peacemaker_isa::kernels::iu4_gemm::ModuleProof,
) {
    qsa_select::emit_module_bf16(ARCH).unwrap()
}

/// Instruction lines of the emitted assembly (everything before the `.rodata`).
fn instructions(text: &str) -> Vec<&str> {
    let body = &text[..text.find(".section .rodata").unwrap()];
    body.lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty() && !l.starts_with('.') && !l.starts_with(';') && !l.ends_with(':')
        })
        .collect()
}

fn count(lines: &[&str], mnemonic: &str) -> usize {
    lines
        .iter()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}

#[test]
fn entries_are_deterministic_two_symbols_and_target_only_gfx1151() {
    for (kind, symbol) in [(Kind::ScoreBf16, SYMBOL), (Kind::ScoreBf16Rows32, SYMBOL32)] {
        let spec = Spec { arch: ARCH, kind };
        let (a, b) = (
            qsa_select::emit(spec).unwrap(),
            qsa_select::emit(spec).unwrap(),
        );
        assert_eq!(a.s_text, b.s_text);
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        assert_eq!(
            serde_json::to_value(&a.proof).unwrap(),
            serde_json::to_value(&b.proof).unwrap()
        );
        assert_eq!(a.shape.barriers, 0, "no LDS and so no barrier");
        assert!(a.s_text.contains(&format!("{symbol}:")));
    }
    assert_eq!(
        qsa_select::emit(Spec {
            arch: ARCH,
            kind: Kind::ScoreBf16
        })
        .unwrap()
        .s_text,
        qsa_score::emit_bf16(ARCH).unwrap().s_text
    );
    assert_eq!(
        qsa_select::emit(Spec {
            arch: ARCH,
            kind: Kind::ScoreBf16Rows32
        })
        .unwrap()
        .s_text,
        qsa_score::emit_bf16_rows32(ARCH).unwrap().s_text
    );
    assert_eq!(qsa_score::symbol_bf16(ARCH), SYMBOL);
    assert_eq!(qsa_score::symbol_bf16_rows32(ARCH), SYMBOL32);
    assert_eq!(Kind::BF16, [Kind::ScoreBf16, Kind::ScoreBf16Rows32]);

    for arch in [Arch::Gfx1100, Arch::Gfx1201] {
        for kind in Kind::BF16 {
            assert!(
                qsa_select::emit(Spec { arch, kind }).is_err(),
                "{arch:?} {kind:?}"
            )
        }
        assert!(qsa_score::emit_bf16(arch).is_err() && qsa_score::emit_bf16_rows32(arch).is_err());
        assert!(qsa_select::emit_module_bf16(arch).is_err());
    }

    // Exactly two symbols (rows16 then rows32), one module, deterministic bytes.
    let (emitted, text, proof) = bf16();
    assert_eq!(emitted.len(), 2);
    assert_eq!(qsa_select::module_bf16(ARCH), MODULE);
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 2);
    assert_eq!(text.matches(&format!("{SYMBOL}:")).count(), 1);
    assert_eq!(text.matches(&format!("{SYMBOL32}:")).count(), 1);
    assert!(
        text.find(&format!("{SYMBOL}:")).unwrap() < text.find(&format!("{SYMBOL32}:")).unwrap(),
        "module order is rows16 then rows32"
    );
    assert!(!text.contains("indexed_attention_select_scores_rows16_f32_pm_gfx1151"));
    assert!(!text.contains("indexed_attention_select_scores_rows32_f32_pm_gfx1151"));
    assert!(!text.contains("indexed_attention_select_from_scores_pm_gfx1151"));
    let (again_emitted, again_text, again_proof) = bf16();
    assert_eq!((again_emitted.len(), &again_text), (2, &text));
    assert_eq!(
        serde_json::to_value(&again_proof).unwrap(),
        serde_json::to_value(&proof).unwrap(),
        "ModuleProof"
    );
}

/// The rows32 BF16 symbol is the rows16 emission with the row tile changed and
/// nothing else: after the symbol name is normalized, the texts differ exactly
/// in the workgroup-y shift (4 -> 5) and the row clamp (16 -> 32).
#[test]
fn rows32_differs_from_rows16_only_in_row_geometry() {
    let a = qsa_select::emit(Spec {
        arch: ARCH,
        kind: Kind::ScoreBf16,
    })
    .unwrap();
    let b = qsa_select::emit(Spec {
        arch: ARCH,
        kind: Kind::ScoreBf16Rows32,
    })
    .unwrap();
    // Symbols (and the `.L<symbol>_end` labels derived from them) and the module-global
    // `.Lqsc32_*` loop/exit labels are the only names that differ from rows16.
    let b_as_16 = b
        .s_text
        .replace(SYMBOL32, SYMBOL)
        .replace(".Lqsc32_", ".Lqsc_");
    assert_eq!(a.s_text.lines().count(), b_as_16.lines().count());
    let diff: Vec<(&str, &str)> = a
        .s_text
        .lines()
        .zip(b_as_16.lines())
        .filter(|(x, y)| x != y)
        .collect();
    assert_eq!(
        diff,
        [
            ("\ts_lshl_b32 s27, s3, 4", "\ts_lshl_b32 s27, s3, 5"),
            ("\ts_min_i32 s28, s28, 16", "\ts_min_i32 s28, s28, 32")
        ]
    );
    assert_eq!(
        (
            a.shape.instructions,
            a.shape.vopd_pairs,
            a.shape.vmem,
            a.shape.waits,
            a.shape.next_free_vgpr,
            a.shape.next_free_sgpr
        ),
        (
            b.shape.instructions,
            b.shape.vopd_pairs,
            b.shape.vmem,
            b.shape.waits,
            b.shape.next_free_vgpr,
            b.shape.next_free_sgpr
        )
    );
}

/// The F32 pair carries rows16 and rows32 scores; the BF16 module's kinds are separate.
#[test]
fn f32_pair_module_is_unchanged() {
    assert_eq!(Kind::ALL, [Kind::Score, Kind::Score32, Kind::Select]);
    assert!(!Kind::ALL.contains(&Kind::ScoreBf16) && !Kind::ALL.contains(&Kind::ScoreBf16Rows32));
    let (emitted, text, _) = qsa_select::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 3);
    assert_eq!(qsa_select::module(ARCH), "qsa_select_pm_gfx1151");
    assert!(!text.contains(SYMBOL) && !text.contains(SYMBOL32));
    assert!(text.contains("indexed_attention_select_scores_rows16_f32_pm_gfx1151:"));
    assert!(text.contains("indexed_attention_select_scores_rows32_f32_pm_gfx1151:"));
    assert!(text.contains("indexed_attention_select_from_scores_pm_gfx1151:"));
    let committed = std::fs::read(format!(
        "{}/../../kernels/{}.hxaco",
        env!("CARGO_MANIFEST_DIR"),
        qsa_select::module(ARCH)
    ))
    .unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    assert!(
        peacemaker_isa::native::bundle(&elf, ARCH, HOST) == committed,
        "F32 pair: fresh emission differs from the committed bundle"
    );
}

/// Kernarg ABI of both score kernels: pointer/scalar offsets, 52 bytes each.
#[test]
fn kernarg_layout_matches_the_f32_score_abi() {
    let text = bf16().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let args = |l: &str| {
        l.trim()
            .strip_prefix(".offset:")
            .map(|v| v.trim().parse::<u32>().unwrap())
    };
    let offsets: Vec<u32> = metadata.lines().filter_map(args).collect();
    let abi = [0, 8, 16, 24, 28, 32, 36, 40, 44, 48];
    assert_eq!(offsets, [&abi[..], &abi[..]].concat());
    assert_eq!(
        text.matches(".amdhsa_kernarg_size 52\n").count(),
        2,
        "kernarg size"
    );
    assert_eq!(
        text.matches(".kernarg_segment_size: 52\n").count(),
        2,
        "metadata kernarg size"
    );

    for kind in [Kind::Score, Kind::Score32] {
        let f32_text = qsa_select::emit(Spec { arch: ARCH, kind }).unwrap().s_text;
        let f32_metadata = &f32_text[f32_text.find("amdhsa.kernels:").unwrap()..];
        let f32_offsets: Vec<u32> = f32_metadata.lines().filter_map(args).collect();
        assert_eq!(
            abi[..],
            f32_offsets[..],
            "BF16 score ABI offsets equal the {kind:?} F32 score ABI offsets"
        );
    }
}

/// Resources, waits and instruction shape of each kernel.
#[test]
fn resources_waits_and_shape() {
    let text = bf16().1;
    for field in [
        ".amdhsa_next_free_vgpr 152\n",
        ".amdhsa_next_free_sgpr 103\n",
        ".amdhsa_group_segment_fixed_size 0\n",
        ".amdhsa_private_segment_fixed_size 0\n",
        ".amdhsa_wavefront_size32 1\n",
        ".amdhsa_uses_dynamic_stack 0\n",
        ".amdhsa_enable_private_segment 0\n",
    ] {
        assert_eq!(text.matches(field).count(), 2, "{field:?}");
    }
    for field in [
        ".vgpr_count: 152\n",
        ".sgpr_count: 105\n",
        ".vgpr_spill_count: 0\n",
        ".sgpr_spill_count: 0\n",
        ".group_segment_fixed_size: 0\n",
        ".private_segment_fixed_size: 0\n",
        ".wavefront_size: 32\n",
        ".max_flat_workgroup_size: 256\n",
        ".uses_dynamic_stack: false\n",
    ] {
        assert_eq!(text.matches(field).count(), 2, "{field:?}");
    }
    for kind in Kind::BF16 {
        let e = qsa_select::emit(Spec { arch: ARCH, kind }).unwrap();
        assert_eq!(
            (e.shape.next_free_vgpr, e.shape.next_free_sgpr),
            (152, 103),
            "{kind:?}"
        );
        assert_eq!(
            (e.proof.next_free_vgpr, e.proof.next_free_sgpr),
            (152, 103),
            "{kind:?}"
        );
        assert_eq!(e.shape.barriers, 0);
        assert_eq!(e.shape.ds, 0, "no LDS");

        let lines = instructions(&e.s_text);
        assert_eq!(
            count(&lines, "buffer_load_b128"),
            16,
            "one 256-byte BF16 key row as 16 b128 loads"
        );
        assert_eq!(count(&lines, "buffer_store_b32"), 1);
        assert_eq!(
            count(&lines, "v_lshlrev_b32_e32") + count(&lines, "v_lshlrev_b32"),
            66
        );
        assert_eq!(
            count(&lines, "v_and_b32_e32") + count(&lines, "v_and_b32"),
            64
        );
        assert_eq!(count(&lines, "s_barrier"), 0);
        assert_eq!(count(&lines, "s_endpgm"), 1);
        assert_eq!(count(&lines, "v_div_scale_f32"), 8);
        assert_eq!(count(&lines, "v_div_fmas_f32"), 4);
        assert_eq!(count(&lines, "v_div_fixup_f32"), 4);
        for forbidden in [
            "scratch_",
            "flat_",
            "global_",
            "ds_",
            "buffer_atomic_",
            "v_readlane",
            "v_writelane",
            "s_setpc",
            "s_swappc",
        ] {
            assert!(
                !lines.iter().any(|l| l.starts_with(forbidden)),
                "forbidden {forbidden}*"
            );
        }

        // The 16 loads are drained by one `vmcnt(0)` wait before the first unpack mask.
        let last_load = lines
            .iter()
            .rposition(|l| l.starts_with("buffer_load_b128"))
            .unwrap();
        let wait = lines
            .iter()
            .enumerate()
            .skip(last_load)
            .find(|(_, l)| l.starts_with("s_waitcnt") && l.contains("vmcnt(0)"))
            .map(|(i, _)| i)
            .expect("vmcnt(0) after the last key load");
        let first_unpack_and = lines
            .iter()
            .position(|l| l.starts_with("v_and_b32"))
            .unwrap();
        assert!(
            wait < first_unpack_and,
            "the first unpack must follow the vmcnt(0) wait"
        );
        assert!(e.shape.waits >= 1);
        assert!(e.proof.waits.len() >= 1);
    }
}

/// The committed bundle is exactly today's native emission: complete bundle
/// bytes, not only `.text`.
#[test]
fn committed_bundle_equals_fresh_native_emission() {
    let path = format!(
        "{}/../../kernels/{MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = bf16().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    assert!(
        peacemaker_isa::native::bundle(&elf, ARCH, HOST) == committed,
        "{path}: fresh emission differs from the committed bundle"
    );
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// Both symbols certify: parse-back, independent wait replay, the
    /// committed BF16 shape contracts, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = bf16();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-qsa-select-bf16-cert-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain {
            host_target: HOST.into(),
            ..Toolchain::oracle()
        };
        let hsaco = dir.join("module.hsaco");
        let build = build(&toolchain, &s, &hsaco, ARCH.name()).unwrap();
        let committed = std::fs::read(format!(
            "{}/../../kernels/{MODULE}.hxaco",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        assert!(
            std::fs::read(&hsaco).unwrap() == committed,
            "{MODULE}: certified build differs from the committed bundle"
        );
        for (symbol, tag) in [(SYMBOL, "score_bf16"), (SYMBOL32, "score_bf16_rows32")] {
            let m7 = pm_check::m7(&build.elf, ARCH.name(), symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}: {m7}"
            );
            let path = format!(
                "{}/kernels/qsa_select.{}.{tag}.contract.json",
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
    }

    /// The BF16 score kernels, like the F32 ones, use no LDS.
    #[test]
    fn no_lds_accesses() {
        for (kind, symbol) in [(Kind::ScoreBf16, SYMBOL), (Kind::ScoreBf16Rows32, SYMBOL32)] {
            let e = qsa_select::emit(Spec { arch: ARCH, kind }).unwrap();
            assert!(
                !e.s_text.lines().any(|l| l.trim_start().starts_with("ds_")),
                "{kind:?} uses no LDS"
            );
            assert_eq!(
                peacemaker_isa::pm_check::lds_bounds(&e.s_text, symbol, 8, 0).unwrap(),
                0
            );
        }
    }
}
