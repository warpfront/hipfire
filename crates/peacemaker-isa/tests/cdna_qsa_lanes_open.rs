// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OPEN, negative-only pin of the immutable original QSA lanes anchor.
//! No deployable twin or ledger allowance is established by these refusals.
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, KernelSpec};
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] = include_bytes!("fixtures/cdna/gfx942_qsa_lanes_open.co");
const SOURCE: &[u8] = include_bytes!("fixtures/cdna/gfx942_qsa_lanes_open.hip");
const CASES: &[(&str, &str)] = &[
    ("qwen4_qsa_lanes_prologue_f32_gfx942", "CDNA certification of `qwen4_qsa_lanes_prologue_f32_gfx942`: qwen4_qsa_lanes_prologue_f32_gfx942: instruction 62 `flat_store_dword v[6:7], v4`: Flat segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled"),
    ("qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942", "CDNA certification of `qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942`: qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942: dword 1212: generated instruction v_cmp_eq_f16 (ENC_VOP3 opcode 34, gfx942) has no hand-written model"),
    ("qwen4_qsa_lanes_pool_f32_gfx942", "CDNA certification of `qwen4_qsa_lanes_pool_f32_gfx942`: qwen4_qsa_lanes_pool_f32_gfx942: instruction 87 `flat_load_dword v5, v[6:7]`: Flat segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled"),
    ("qwen4_qsa_lanes_scores_rows8_f32_gfx942", "CDNA certification of `qwen4_qsa_lanes_scores_rows8_f32_gfx942`: qwen4_qsa_lanes_scores_rows8_f32_gfx942: instruction 100 `flat_load_dword v11, v[6:7]`: Flat segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled"),
    ("qwen4_qsa_lanes_select_from_scores_gfx942", "CDNA certification of `qwen4_qsa_lanes_select_from_scores_gfx942`: qwen4_qsa_lanes_select_from_scores_gfx942: instruction 72 `flat_store_dword v[6:7], v1`: Flat segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled"),
    ("qwen4_qsa_lanes_attention_f32_gfx942", "CDNA certification of `qwen4_qsa_lanes_attention_f32_gfx942`: qwen4_qsa_lanes_attention_f32_gfx942: instruction 223 `flat_load_dword v6, v[6:7]`: Flat segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled"),
    ("qwen4_qsa_lanes_attention_fp8_fnuz_gfx942", "CDNA certification of `qwen4_qsa_lanes_attention_fp8_fnuz_gfx942`: v_fma_mix_f32 v18, v27, 1.0, v27 op_sel_hi:[1,1,1]: unverified encoding: unverified(xml-only: XML mnemonic V_MAD_MIX_F32 decoded as v_fma_mix_f32 v0, v0, v0, v0 op_sel_hi:[1,1,1])"),
];

#[test]
fn immutable_original_anchor_and_source_are_pinned() {
    assert_eq!(
        format!("{:x}", Sha256::digest(ANCHOR)),
        "bd4b17da271be63fef94e0e81711b08fe4c8c8d3123018ab8a7d0d2a653e334d"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(SOURCE)),
        "086364291c76819ae8f9a24603d32f64c40b2216a136d6923a1fb8eaa4224e06"
    );
}

#[test]
fn every_original_entry_remains_explicitly_open() {
    for &(symbol, expected) in CASES {
        let result = seal_many(
            ANCHOR,
            Arch::Gfx942,
            &[KernelSpec {
                symbol,
                abi_ref: None,
            }],
        );
        let error = match result {
            Ok(_) => panic!("{symbol}: OPEN obligation unexpectedly discharged; review pin"),
            Err(error) => error,
        };
        assert_eq!(error, expected, "{symbol}");
    }
}
