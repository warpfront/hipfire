// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The exact gfx942 P1b wide64 hipcc anchor, byte-pinned, must certify **all three** entries.
//!
//! History (do not lose it): before the block-local scalar branch proofs the same pinned
//! bundle was refused for the first entry, `gemm_bf16_mfma_gfx942_wide64`:
//! `wait ledger replay: pc 975 global_load_dwordx4 v[34:37], v[4:5], off (after label L872):
//! needs Vm wait 0 (Waw { reg: "v[34:37]" })`. That witness fell through pc 893
//! (`s_mov_b64 s[0:1], 0; s_and_b64 s[0:1], s[0:1], exec; s_cselect_b32 s0, 1, 0;
//! s_cmp_lg_u32 s0, 1; s_cbranch_scc1 L695`), whose branch is taken for every EXEC value, so
//! the syntactic fall-through into L872 does not exist (`pm/LEDGER-PATH-PROOF.md` section 2).
//! The old refusal is preserved here only as this comment and in the private all-Both
//! reference assertion owned by `cdna::lifted`'s unit tests; there is no allowlist row.
//!
//! This is an acceptance test of the reviewed verifier change. It makes no claim about any
//! observed run until the integration owner executes it; a different refusal for any of the
//! three symbols is a failure to be investigated, never to be suppressed here. No verifier
//! limit, obligation allowlist, anchor instruction, or scheduler is changed.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] = include_bytes!("fixtures/cdna/gfx942_p1b_wide.co");
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/gemm_bf16_mfma_wide.gfx942.hip");
const SYMBOLS: [&str; 3] = [
    "gemm_bf16_mfma_gfx942_wide64",
    "gemm_bf16_mfma_gfx942_wide64_m32",
    "gemm_bf16_mfma_gfx942_wide64_m64",
];
const SOURCE_SHA256: &str = "49f356151cf2fac0c31e6e965bf7cf7e5a9968e2c8970e90f691e4acac512567";
const BUNDLE_SHA256: &str = "5ecf282d0a3129cc9cfc0d4268677bc1b67e11454931818ba808a33a30061bcc";
const DEVICE_ELF_SHA256: &str = "141276895e3dd53cb51746e7fb8f761d2e29929dc36147c719d0fe8355dec1e6";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn p1b_owner_anchor_is_byte_exact_and_all_three_entries_certify() {
    // Approved source owner commit: 17183b9b89. Re-copy and re-pin if the owner changes it.
    assert_eq!(sha(SOURCE), SOURCE_SHA256);
    assert_eq!(sha(ANCHOR), BUNDLE_SHA256);
    let (bundle, payloads) = Bundle::read(ANCHOR).expect("pinned offload bundle");
    let entries: Vec<_> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.triple == "hipv4-amdgcn-amd-amdhsa--gfx942")
        .collect();
    assert_eq!(entries.len(), 1);
    let elf = payloads[entries[0].0];
    assert_eq!(sha(elf), DEVICE_ELF_SHA256);
    let lifted = peacemaker_lift::cdna::lift_object(elf).expect("fully typed P1b image");
    assert_eq!(peacemaker_lift::cdna::checked(&lifted).unwrap(), elf);
    let mut names: Vec<_> = lifted
        .kernels
        .iter()
        .map(|k| k.image.name.as_str())
        .collect();
    names.sort_unstable();
    let mut expected = SYMBOLS;
    expected.sort_unstable();
    assert_eq!(names, expected);

    let specs: Vec<_> = SYMBOLS
        .iter()
        .map(|symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect();
    let sealed = seal_many(ANCHOR, Arch::Gfx942, &specs).unwrap_or_else(|e| {
        panic!("the pinned P1b anchor must certify all three entries, got: {e}")
    });
    assert_eq!(sealed.len(), SYMBOLS.len());
    for (symbol, s) in SYMBOLS.iter().zip(&sealed) {
        let h = &s.report["hazards"];
        assert_eq!(s.report["kernel"], *symbol, "{symbol}");
        assert_eq!(h["checked"], true, "{symbol}: {h}");
        assert_eq!(h["blocking"], 0, "{symbol}: {h}");
        assert_eq!(h["findings"], 0, "{symbol}: {h}");
        assert_eq!(h["bypasses"], 0, "{symbol}: {h}");
        assert_eq!(h["hazard"]["violations"], 0, "{symbol}: {h}");
        assert_eq!(h["hazard"]["inserted_nops"], 0, "{symbol}: {h}");
        assert_eq!(h["hazard"]["rewritten_instructions"], 0, "{symbol}: {h}");
        // The device image is unchanged by certification.
        assert_eq!(s.elf_sha256, DEVICE_ELF_SHA256, "{symbol}");
    }
    // One shared bundle; every symbol re-verifies the exact same bytes.
    let hxaco = &sealed[0].hxaco;
    assert!(sealed.iter().all(|s| &s.hxaco == hxaco));
    let verified = verify_hxaco_many(hxaco, Arch::Gfx942, &SYMBOLS)
        .unwrap_or_else(|e| panic!("sealed P1b bundle must re-verify all three entries: {e}"));
    assert_eq!(verified.len(), SYMBOLS.len());
    for (a, b) in sealed.iter().zip(&verified) {
        assert_eq!(a.hxaco_sha256, b.hxaco_sha256);
        assert_eq!(a.elf_sha256, b.elf_sha256);
        assert_eq!(a.text_sha256, b.text_sha256);
        assert_eq!(b.report["hazards"]["blocking"], 0);
        assert_eq!(b.report["hazards"]["findings"], 0);
        assert_eq!(b.report["hazards"]["hazard"]["rewritten_instructions"], 0);
    }
}
