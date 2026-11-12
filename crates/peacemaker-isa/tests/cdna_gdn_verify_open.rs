// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Negative pin of the frozen four-entry GDN verification module: its rollback entry (FLAT
//! segment) stays refused, so it is not a four-entry certificate and no deployable twin or ledger
//! allowance is established for it.
//!
//! History (do not lose it): the convolution entry was also refused here, with
//! `wait ledger replay: pc 674 global_load_dword v1, v[6:7], off (after label L648): needs Vm
//! wait 0 (Waw { reg: "v1" })`. That witness crossed `s_branch L648` (pc 665) after a pending
//! `global_load_ushort v1`; the predecessor of that edge stores 0 in `s[14:15]`, so the join's
//! `s_and_b64 s[14:15], s[14:15], exec; s_cselect_b32; s_cmp_lg_u32; s_cbranch_scc1` always
//! takes its branch over the load. The edge-conditioned scalar proof (`cdna::lifted`
//! `entry_proofs`) removes exactly that edge; the fixture bytes are unchanged.
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, KernelSpec};
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] = include_bytes!("fixtures/cdna/gfx942_gdn_verify_open.co");
const SOURCE: &[u8] = include_bytes!("fixtures/cdna/gfx942_gdn_verify_open.hip");
const SYMBOLS: [&str; 4] = [
    "qwen4_g942_gdn_verify_capture_f32",
    "qwen4_g942_gdn_verify_rollback_f32",
    "qwen4_g942_gdn_verify_conv_params_k4",
    "qwen4_g942_gdn_verify_gate_batched",
];

fn refusal(symbol: &str) -> String {
    match seal_many(
        ANCHOR,
        Arch::Gfx942,
        &[KernelSpec {
            symbol,
            abi_ref: None,
        }],
    ) {
        Ok(_) => panic!("{symbol}: OPEN obligation unexpectedly discharged; review pin"),
        Err(error) => error,
    }
}

#[test]
fn frozen_source_and_anchor_are_immutable() {
    assert_eq!(
        format!("{:x}", Sha256::digest(ANCHOR)),
        "a9c68299ce5bf448d5172a5dba6f46d101ffb86b620acd4772c73fae8272f355"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(SOURCE)),
        "5773927855c0585d95d190762a251076b7bb01b7907647a921c15abe108d66c7"
    );
}

#[test]
fn rollback_flat_segment_stays_open() {
    assert_eq!(refusal(SYMBOLS[1]), "CDNA certification of `qwen4_g942_gdn_verify_rollback_f32`: qwen4_g942_gdn_verify_rollback_f32: instruction 60 `flat_load_dword v142, v[2:3]`: Flat segment access: FLAT counts on VM_CNT and LGKM_CNT and may address LDS, SCRATCH needs a private segment; only GLOBAL is modelled");
}

#[test]
fn convolution_vmem_waw_at_pc674_is_discharged_by_the_join_edge_proof() {
    let sealed = seal_many(
        ANCHOR,
        Arch::Gfx942,
        &[KernelSpec {
            symbol: SYMBOLS[2],
            abi_ref: None,
        }],
    )
    .unwrap_or_else(|error| panic!("{}: {error}", SYMBOLS[2]));
    assert_eq!(sealed.len(), 1);
    let report = &sealed[0].report;
    assert_eq!(report["hazards"]["checked"], true);
    for field in ["findings", "blocking", "bypasses"] {
        assert_eq!(report["hazards"][field], 0, "{field}");
    }
}

#[test]
fn capture_conv_and_gate_are_individually_provable_but_not_a_four_entry_certificate() {
    for symbol in [SYMBOLS[0], SYMBOLS[2], SYMBOLS[3]] {
        let sealed = seal_many(
            ANCHOR,
            Arch::Gfx942,
            &[KernelSpec {
                symbol,
                abi_ref: None,
            }],
        )
        .unwrap_or_else(|error| panic!("{symbol}: {error}"));
        assert_eq!(sealed.len(), 1);
        let report = &sealed[0].report;
        assert_eq!(
            report["elf_sha256"],
            "60ebb9c7b0bb236a37d2a216df426fee61edc88556e4cfc4e81290a64909c0b4"
        );
        assert_eq!(report["hazards"]["checked"], true);
        for field in ["findings", "blocking", "bypasses"] {
            assert_eq!(report["hazards"][field], 0);
        }
    }
    let specs: Vec<_> = SYMBOLS
        .iter()
        .map(|symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect();
    let error = match seal_many(ANCHOR, Arch::Gfx942, &specs) {
        Ok(_) => panic!("full frozen module unexpectedly certified; review OPEN pins"),
        Err(error) => error,
    };
    assert!(
        error.starts_with("CDNA certification of `qwen4_g942_gdn_verify_rollback_f32`:"),
        "{error}"
    );
}
