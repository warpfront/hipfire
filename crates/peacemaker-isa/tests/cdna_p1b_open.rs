// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OPEN, not a certified twin: the exact P1b anchor remains refused for a VMEM WAW.
//! A future sound path/EXEC proof must deliberately replace this negative assertion.
//! No verifier limit, obligation allowlist, anchor instruction, or scheduler is changed.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] = include_bytes!("fixtures/cdna/gfx942_p1b_wide_open.co");
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/gemm_bf16_mfma_wide.gfx942.hip");
const SYMBOLS: [&str; 3] = [
    "gemm_bf16_mfma_gfx942_wide64",
    "gemm_bf16_mfma_gfx942_wide64_m32",
    "gemm_bf16_mfma_gfx942_wide64_m64",
];

#[test]
fn p1b_owner_anchor_is_byte_exact_but_wait_certification_stays_open() {
    // Approved source owner commit: 17183b9b89. Re-copy and re-pin if the owner changes it.
    assert_eq!(
        format!("{:x}", Sha256::digest(SOURCE)),
        "49f356151cf2fac0c31e6e965bf7cf7e5a9968e2c8970e90f691e4acac512567"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(ANCHOR)),
        "5ecf282d0a3129cc9cfc0d4268677bc1b67e11454931818ba808a33a30061bcc"
    );
    let (bundle, payloads) = Bundle::read(ANCHOR).expect("pinned offload bundle");
    let entries: Vec<_> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.triple == "hipv4-amdgcn-amd-amdhsa--gfx942")
        .collect();
    assert_eq!(entries.len(), 1);
    let elf = payloads[entries[0].0];
    assert_eq!(
        format!("{:x}", Sha256::digest(elf)),
        "141276895e3dd53cb51746e7fb8f761d2e29929dc36147c719d0fe8355dec1e6"
    );
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
    let refusal = match seal_many(ANCHOR, Arch::Gfx942, &specs) {
        Ok(_) => panic!("OPEN P1b anchor unexpectedly sealed; review the path/EXEC proof"),
        Err(error) => error.to_string(),
    };
    for required in [
        SYMBOLS[0],
        "wait ledger replay",
        "pc 975",
        "global_load_dwordx4 v[34:37], v[4:5], off",
        "L872",
        "needs Vm wait 0",
        "Waw { reg: \"v[34:37]\" }",
    ] {
        assert!(
            refusal.contains(required),
            "missing {required:?}: {refusal}"
        );
    }
}
