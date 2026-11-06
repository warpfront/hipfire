// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 register-resident-Q native-fp8 FA2-GQA PeaceMaker twin `fa_qresident_a3b_pm_gfx1201` (H16 / KV2 / D256,
//! two exports: the direct `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_pm_gfx1201` and the range
//! `..._range_a3b_pm_gfx1201`, both with the hipcc twins' 68-byte kernarg ABI, 768-thread block and 49 936-byte dynamic
//! LDS). Exact spec / arch admission, determinism, the descriptor and kernarg layout, the committed bundle / contracts /
//! M7 receipts against fresh emission, and the independent M7 analysis of the linked object (no wait / hazard
//! obligations). The arithmetic is gated on hardware by the byte oracle (`fa_pm_fp8_oracle --pm-qres`: PM vs the hipcc
//! twin on the same inputs, raw bytes); the emitted text is the generator's own schedule, not a copy of the hipcc object.
use peacemaker_isa::kernels::fa_qresident::{self, FaQresSpec};
use peacemaker_isa::{native, Arch};
use serde_json::Value;

const ARCH: Arch = Arch::Gfx1201;

fn emitted() -> (Vec<peacemaker_isa::plan::Emitted>, String) {
    let (e, text, _) = fa_qresident::emit_module(ARCH, &FaQresSpec::A3B).unwrap();
    (e, text)
}

// ------------------------------------------------------------------------------------------------ 1. spec / domain
#[test]
fn spec_and_arch_are_exact() {
    assert!(fa_qresident::emit(ARCH, &FaQresSpec::A3B, false).is_ok());
    assert!(fa_qresident::emit(ARCH, &FaQresSpec::A3B, true).is_ok());
    for (heads, kv_heads, head_dim) in [
        (0u32, 0u32, 0u32),
        (16, 2, 128),
        (16, 4, 256),
        (24, 4, 256),
        (8, 1, 256),
        (32, 2, 256),
    ] {
        let s = FaQresSpec {
            heads,
            kv_heads,
            head_dim,
        };
        assert!(
            fa_qresident::emit(ARCH, &s, false).is_err(),
            "{heads}/{kv_heads}/{head_dim}"
        );
        assert!(fa_qresident::emit_module(ARCH, &s).is_err());
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        assert!(fa_qresident::emit(arch, &FaQresSpec::A3B, false).is_err());
        assert!(fa_qresident::emit_module(arch, &FaQresSpec::A3B).is_err());
    }
}

#[test]
fn grid_is_the_hipcc_launchers() {
    // [ceil(min(batch, 512) * 8 / 384), 2, ceil(batch / 512)]
    for (batch, g) in [
        (1u32, [1u32, 2, 1]),
        (48, [1, 2, 1]),
        (49, [2, 2, 1]),
        (96, [2, 2, 1]),
        (511, [11, 2, 1]),
        (512, [11, 2, 1]),
        (513, [11, 2, 2]),
        (8192, [11, 2, 16]),
    ] {
        assert_eq!(fa_qresident::grid(batch), g, "batch {batch}");
    }
}

// ------------------------------------------------------------------------------------------------ 2. determinism / layout
#[test]
fn emission_is_deterministic_and_the_module_is_direct_then_range() {
    let (a, b) = (
        fa_qresident::emit(ARCH, &FaQresSpec::A3B, false).unwrap(),
        fa_qresident::emit(ARCH, &FaQresSpec::A3B, false).unwrap(),
    );
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(
        serde_json::to_string(&a.proof).unwrap(),
        serde_json::to_string(&b.proof).unwrap()
    );
    let (ea, ta, ma) = fa_qresident::emit_module(ARCH, &FaQresSpec::A3B).unwrap();
    let (_, tb, mb) = fa_qresident::emit_module(ARCH, &FaQresSpec::A3B).unwrap();
    assert_eq!(ta, tb);
    assert_eq!(
        serde_json::to_string(&ma).unwrap(),
        serde_json::to_string(&mb).unwrap()
    );
    assert_eq!((ma.module.as_str(), ma.arch), (fa_qresident::MODULE, ARCH));
    assert_eq!(ea.len(), 2);
    assert_eq!(
        ea[0].s_text, a.s_text,
        "the module entry is the standalone direct export"
    );
    assert_eq!(ta.matches(".amdhsa_kernel ").count(), 2);
    let direct = ta
        .find(&format!(".amdhsa_kernel {}\n", fa_qresident::SYMBOL_DIRECT))
        .unwrap();
    let range = ta
        .find(&format!(".amdhsa_kernel {}\n", fa_qresident::SYMBOL_RANGE))
        .unwrap();
    assert!(direct < range, "direct export precedes the range export");
}

#[test]
fn descriptor_kernarg_and_resources_match_the_hipcc_twin() {
    let (entries, _) = emitted();
    for (e, symbol) in entries
        .iter()
        .zip([fa_qresident::SYMBOL_DIRECT, fa_qresident::SYMBOL_RANGE])
    {
        let t = &e.s_text;
        assert!(
            t.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}"
        );
        let field = |key: &str| -> u32 {
            t.lines()
                .filter_map(|l| l.trim().strip_prefix(key))
                .next()
                .unwrap_or_else(|| panic!("{symbol}: no {key}"))
                .trim()
                .parse()
                .unwrap()
        };
        assert_eq!(field(".amdhsa_kernarg_size"), fa_qresident::KERNARG_BYTES);
        assert_eq!(field(".amdhsa_group_segment_fixed_size"), 0);
        assert_eq!(field(".amdhsa_private_segment_fixed_size"), 0);
        assert!(
            field(".amdhsa_next_free_vgpr") <= 240,
            "{symbol}: VGPR budget"
        );
        assert_eq!(field(".amdhsa_wavefront_size32"), 1);
        assert_eq!(field(".amdhsa_system_sgpr_workgroup_id_z"), 1);
        assert!(
            t.contains(".workgroup_size_hint")
                || t.contains(&format!(
                    ".max_flat_workgroup_size: {}",
                    fa_qresident::BLOCK
                )),
            "{symbol}: workgroup size"
        );
        // Pointer kernargs and the five by-value scalars at the hipcc twin's offsets.
        for (name, off, size) in [
            ("q_codes", 0u32, 8u32),
            ("q_scales", 8, 8),
            ("k_cache", 16, 8),
            ("v_cache", 24, 8),
            ("out", 32, 8),
            ("positions", 40, 8),
            ("n_heads", 48, 4),
            ("n_kv_heads", 52, 4),
            ("head_dim", 56, 4),
            ("batch_size", 60, 4),
            ("scale_attn", 64, 4),
        ] {
            assert!(
                t.contains(&format!(".name: {name}\n"))
                    || t.contains(&format!(".name:       {name}\n")),
                "{symbol}: kernarg {name}@{off}+{size}"
            );
        }
        // No private memory, no hazards the builder could not close.
        assert!(!t.contains("scratch_"));
        assert!(!t.contains("buffer_"));
        assert!(!t.contains("global_atomic"));
        assert!(!t.contains("s_waitcnt"));
    }
}

#[test]
fn text_census_counts_the_arithmetic_of_one_sub() {
    let t = &fa_qresident::emit(ARCH, &FaQresSpec::A3B, false)
        .unwrap()
        .s_text;
    let n = |m: &str| {
        t.lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(m))
            .count()
    };
    // Two buffer copies of the sub loop: 16 QK + 16 PV WMMAs each, nothing else on the matrix unit.
    assert_eq!(n("v_wmma_f32_16x16x16_fp8_fp8"), 64);
    // Barriers: prologue rendezvous + one per tile copy.
    assert_eq!(n("s_barrier_signal"), 3);
    assert_eq!(n("s_barrier_wait"), 3);
    // The P / Q code conversions: 4 + 4 bracket cvts per copy, 4 recompute cvts per copy.
    assert_eq!(n("v_cvt_pk_fp8_f32"), 2 * (8 + 4));
    assert_eq!(n("s_endpgm"), 1);
}

// ------------------------------------------------------------------------------------------------ 3. committed artifacts
fn kernels_dir() -> String {
    format!("{}/kernels", env!("CARGO_MANIFEST_DIR"))
}
fn top_bundle_path() -> String {
    format!(
        "{}/../../kernels/{}.hxaco",
        env!("CARGO_MANIFEST_DIR"),
        fa_qresident::MODULE
    )
}

#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = top_bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{path}: {e} (emit it: peacemaker-isa emit --kernel fa_qresident_a3b_pm --arch gfx1201 ... --bundle kernels/{}.hxaco)",
            fa_qresident::MODULE
        )
    });
    let (_, text) = emitted();
    let fresh = native::bundle(
        &native::assemble(&text, ARCH).unwrap(),
        ARCH,
        native::DEFAULT_HOST_TARGET,
    );
    assert!(
        committed == fresh,
        "{}: fresh native emission differs from the committed bundle {path}",
        fa_qresident::MODULE
    );
    let lifted = peacemaker_lift::lift_object(
        &committed,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap_or_else(|e| panic!("the committed bundle does not lift: {e}"));
    assert_eq!(format!("{:?}", lifted.program.target.arch), "Gfx1201");
    let symbols: Vec<&str> = lifted
        .program
        .kernels
        .iter()
        .map(|k| k.symbol.0.as_str())
        .collect();
    assert_eq!(
        symbols,
        [fa_qresident::SYMBOL_DIRECT, fa_qresident::SYMBOL_RANGE],
        "the committed bundle must hold exactly the direct then the range export"
    );
}

#[test]
fn committed_contracts_and_m7_receipts_equal_fresh_ones() {
    let (entries, _) = emitted();
    for (name, symbol, e) in [
        ("direct", fa_qresident::SYMBOL_DIRECT, &entries[0]),
        ("range", fa_qresident::SYMBOL_RANGE, &entries[1]),
    ] {
        let fresh = fa_qresident::contract(ARCH, e, symbol);
        let file = format!(
            "{}/fa_qresident_a3b_pm.gfx1201.{name}.contract.json",
            kernels_dir()
        );
        let committed: Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed, fresh,
            "{file}: committed contract differs from the derived contract"
        );
    }
    let file = format!("{}/fa_qresident_a3b_pm.gfx1201.m7.json", kernels_dir());
    let m7: Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
            .unwrap();
    let mut keys: Vec<&str> = m7.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut want = [fa_qresident::SYMBOL_DIRECT, fa_qresident::SYMBOL_RANGE];
    want.sort_unstable();
    assert_eq!(keys, want, "{file}: receipt keys");
    for symbol in want {
        let r = &m7[symbol];
        assert_eq!(r["lift"], "byte-exact", "{symbol}");
        assert_eq!(r["obligations"], serde_json::json!({}), "{symbol}");
        assert_eq!(r["ambiguous_delays"], 0, "{symbol}");
        assert_eq!(r["barrier_pairs"], 3, "{symbol}");
        assert!(r["vgpr_high_water"].as_u64().unwrap() <= 240, "{symbol}");
    }
}
