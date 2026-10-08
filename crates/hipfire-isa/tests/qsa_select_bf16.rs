// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QSA selector BF16-pooled score module (gfx1151): the single symbol
//! `indexed_attention_select_scores_rows16_bf16_pm_gfx1151` in module
//! `qsa_select_bf16_pm_gfx1151`. The F32 pair module is untouched.
use hipfire_isa::Arch;
use hipfire_isa::kernels::{qsa_score, qsa_select::{self, Kind, Spec}};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;
/// The host target stamped into the committed bundles (trailing hyphen).
const HOST: &str = "host-x86_64-unknown-linux-gnu-";
const SYMBOL: &str = "indexed_attention_select_scores_rows16_bf16_pm_gfx1151";
const MODULE: &str = "qsa_select_bf16_pm_gfx1151";

fn bf16() -> (Vec<hipfire_isa::Emitted>, String, hipfire_isa::kernels::iu4_gemm::ModuleProof) {
    qsa_select::emit_module_bf16(ARCH).unwrap()
}

/// Instruction lines of the emitted assembly (everything before the `.rodata`).
fn instructions(text: &str) -> Vec<&str> {
    let body = &text[..text.find(".section .rodata").unwrap()];
    body.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('.') && !l.starts_with(';') && !l.ends_with(':')).collect()
}

fn count(lines: &[&str], mnemonic: &str) -> usize {
    lines.iter().filter(|l| l.split_whitespace().next() == Some(mnemonic)).count()
}

#[test]
fn entry_is_deterministic_single_symbol_and_target_only_gfx1151() {
    let spec = Spec { arch: ARCH, kind: Kind::ScoreBf16 };
    let (a, b) = (qsa_select::emit(spec).unwrap(), qsa_select::emit(spec).unwrap());
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(serde_json::to_value(&a.proof).unwrap(), serde_json::to_value(&b.proof).unwrap());
    assert_eq!(a.s_text, qsa_score::emit_bf16(ARCH).unwrap().s_text);
    assert_eq!(a.shape.barriers, 0, "no LDS and so no barrier");
    assert!(a.s_text.contains(&format!("{SYMBOL}:")));
    assert_eq!(qsa_score::symbol_bf16(ARCH), SYMBOL);

    for arch in [Arch::Gfx1100, Arch::Gfx1201] {
        assert!(qsa_select::emit(Spec { arch, kind: Kind::ScoreBf16 }).is_err(), "{arch:?}");
        assert!(qsa_score::emit_bf16(arch).is_err());
        assert!(qsa_select::emit_module_bf16(arch).is_err());
    }

    // Exactly one symbol, one module, deterministic bytes.
    let (emitted, text, proof) = bf16();
    assert_eq!(emitted.len(), 1);
    assert_eq!(emitted[0].s_text, a.s_text);
    assert_eq!(qsa_select::module_bf16(ARCH), MODULE);
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 1);
    assert_eq!(text.matches(&format!("{SYMBOL}:")).count(), 1);
    assert!(!text.contains("indexed_attention_select_scores_rows16_f32_pm_gfx1151"));
    assert!(!text.contains("indexed_attention_select_from_scores_pm_gfx1151"));
    let (again_emitted, again_text, again_proof) = bf16();
    assert_eq!((again_emitted.len(), &again_text), (1, &text));
    assert_eq!(serde_json::to_value(&again_proof).unwrap(), serde_json::to_value(&proof).unwrap(), "ModuleProof");
}

/// The F32 pair and `Kind::ALL` are unaffected by the BF16 module.
#[test]
fn f32_pair_module_is_unchanged() {
    assert_eq!(Kind::ALL, [Kind::Score, Kind::Select]);
    assert!(!Kind::ALL.contains(&Kind::ScoreBf16));
    let (emitted, text, _) = qsa_select::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 2);
    assert_eq!(qsa_select::module(ARCH), "qsa_select_pm_gfx1151");
    assert!(!text.contains(SYMBOL));
    assert!(text.contains("indexed_attention_select_scores_rows16_f32_pm_gfx1151:"));
    assert!(text.contains("indexed_attention_select_from_scores_pm_gfx1151:"));
    let committed = std::fs::read(format!("{}/../../kernels/{}.hxaco", env!("CARGO_MANIFEST_DIR"), qsa_select::module(ARCH))).unwrap();
    let elf = hipfire_isa::native::assemble(&text, ARCH).unwrap();
    assert!(hipfire_isa::native::bundle(&elf, ARCH, HOST) == committed, "F32 pair: fresh emission differs from the committed bundle");
}

/// Kernarg ABI of the score kernel: pointer/scalar offsets, 52 bytes.
#[test]
fn kernarg_layout_matches_the_f32_score_abi() {
    let text = bf16().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let args = |l: &str| l.trim().strip_prefix(".offset:").map(|v| v.trim().parse::<u32>().unwrap());
    let offsets: Vec<u32> = metadata.lines().filter_map(args).collect();
    assert_eq!(offsets, [0, 8, 16, 24, 28, 32, 36, 40, 44, 48]);
    assert!(text.contains(".amdhsa_kernarg_size 52\n"), "kernarg size");
    assert!(text.contains(".kernarg_segment_size: 52\n"), "metadata kernarg size");

    let f32_text = qsa_select::emit(Spec { arch: ARCH, kind: Kind::Score }).unwrap().s_text;
    let f32_metadata = &f32_text[f32_text.find("amdhsa.kernels:").unwrap()..];
    let f32_offsets: Vec<u32> = f32_metadata.lines().filter_map(args).collect();
    assert_eq!(offsets, f32_offsets, "BF16 score ABI offsets equal the F32 score ABI offsets");
}

/// Resources, waits and instruction shape of the single kernel.
#[test]
fn resources_waits_and_shape() {
    let text = bf16().1;
    for field in [
        ".amdhsa_next_free_vgpr 152\n", ".amdhsa_next_free_sgpr 103\n", ".amdhsa_group_segment_fixed_size 0\n",
        ".amdhsa_private_segment_fixed_size 0\n", ".amdhsa_wavefront_size32 1\n", ".amdhsa_uses_dynamic_stack 0\n",
        ".amdhsa_enable_private_segment 0\n",
    ] {
        assert!(text.contains(field), "{field:?}");
    }
    for field in [
        ".vgpr_count: 152\n", ".sgpr_count: 105\n", ".vgpr_spill_count: 0\n", ".sgpr_spill_count: 0\n",
        ".group_segment_fixed_size: 0\n", ".private_segment_fixed_size: 0\n", ".wavefront_size: 32\n",
        ".max_flat_workgroup_size: 256\n", ".uses_dynamic_stack: false\n",
    ] {
        assert!(text.contains(field), "{field:?}");
    }
    let e = qsa_select::emit(Spec { arch: ARCH, kind: Kind::ScoreBf16 }).unwrap();
    assert_eq!((e.shape.next_free_vgpr, e.shape.next_free_sgpr), (152, 103));
    assert_eq!((e.proof.next_free_vgpr, e.proof.next_free_sgpr), (152, 103));
    assert_eq!(e.shape.barriers, 0);
    assert_eq!(e.shape.ds, 0, "no LDS");

    let lines = instructions(&e.s_text);
    assert_eq!(count(&lines, "buffer_load_b128"), 16, "one 256-byte BF16 key row as 16 b128 loads");
    assert_eq!(count(&lines, "buffer_store_b32"), 1);
    assert_eq!(count(&lines, "v_lshlrev_b32_e32") + count(&lines, "v_lshlrev_b32"), 66);
    assert_eq!(count(&lines, "v_and_b32_e32") + count(&lines, "v_and_b32"), 64);
    assert_eq!(count(&lines, "s_barrier"), 0);
    assert_eq!(count(&lines, "s_endpgm"), 1);
    assert_eq!(count(&lines, "v_div_scale_f32"), 8);
    assert_eq!(count(&lines, "v_div_fmas_f32"), 4);
    assert_eq!(count(&lines, "v_div_fixup_f32"), 4);
    for forbidden in ["scratch_", "flat_", "global_", "ds_", "buffer_atomic_", "v_readlane", "v_writelane", "s_setpc", "s_swappc"] {
        assert!(!lines.iter().any(|l| l.starts_with(forbidden)), "forbidden {forbidden}*");
    }

    // The 16 loads are drained by one `vmcnt(0)` wait before the first unpack mask.
    let last_load = lines.iter().rposition(|l| l.starts_with("buffer_load_b128")).unwrap();
    let wait = lines.iter().enumerate().skip(last_load).find(|(_, l)| l.starts_with("s_waitcnt") && l.contains("vmcnt(0)")).map(|(i, _)| i)
        .expect("vmcnt(0) after the last key load");
    let first_unpack_and = lines.iter().position(|l| l.starts_with("v_and_b32")).unwrap();
    assert!(wait < first_unpack_and, "the first unpack must follow the vmcnt(0) wait");
    assert!(e.shape.waits >= 1);
    assert!(e.proof.waits.len() >= 1);
}

/// The committed bundle is exactly today's native emission: complete bundle
/// bytes, not only `.text`.
#[test]
fn committed_bundle_equals_fresh_native_emission() {
    let path = format!("{}/../../kernels/{MODULE}.hxaco", env!("CARGO_MANIFEST_DIR"));
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = bf16().1;
    let elf = hipfire_isa::native::assemble(&text, ARCH).unwrap();
    assert!(hipfire_isa::native::bundle(&elf, ARCH, HOST) == committed, "{path}: fresh emission differs from the committed bundle");
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check, toolchain::{build, certify, Toolchain}};

    /// The single symbol certifies: parse-back, independent wait replay, the
    /// committed BF16 shape contract, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification_for_its_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = bf16();
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-qsa-select-bf16-cert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain { host_target: HOST.into(), ..Toolchain::oracle() };
        let hsaco = dir.join("module.hsaco");
        let build = build(&toolchain, &s, &hsaco, ARCH.name()).unwrap();
        let committed = std::fs::read(format!("{}/../../kernels/{MODULE}.hxaco", env!("CARGO_MANIFEST_DIR"))).unwrap();
        assert!(std::fs::read(&hsaco).unwrap() == committed, "{MODULE}: certified build differs from the committed bundle");
        let m7 = pm_check::m7(&build.elf, ARCH.name(), SYMBOL).unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{SYMBOL}: {m7}");
        let path = format!("{}/kernels/qsa_select.{}.score_bf16.contract.json", env!("CARGO_MANIFEST_DIR"), ARCH.name());
        let contract = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap();
        certify(&toolchain, &build, &s, ARCH.name(), &dir.join("score_bf16.manifest.json"), Some(&contract), "test", "test")
            .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
    }

    /// The BF16 score kernel, like the F32 one, uses no LDS.
    #[test]
    fn no_lds_accesses() {
        let e = qsa_select::emit(Spec { arch: ARCH, kind: Kind::ScoreBf16 }).unwrap();
        assert!(!e.s_text.lines().any(|l| l.trim_start().starts_with("ds_")), "score uses no LDS");
        assert_eq!(hipfire_isa::pm_check::lds_bounds(&e.s_text, SYMBOL, 8, 0).unwrap(), 0);
    }
}
