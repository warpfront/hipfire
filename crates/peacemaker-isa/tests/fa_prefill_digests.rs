// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every `fa_prefill` export, both modules and the three shape contracts re-emit the bytes recorded
//! before the building blocks (`fa_prefill::blocks`) were factored out of `emit`: the sha256 of each
//! export's `s_text` and proof JSON, of each module's text and proof, and of each contract's JSON,
//! computed from the unrefactored generator. A change to any digest is a change to a production
//! kernel and must be intentional.

use peacemaker_isa::kernels::fa_prefill::{
    contract, emit, emit_fp8_module, emit_module, emit_preconvert, fp8_contract,
    preconvert_contract, FaKvCodec, FaPrefillSpec,
};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

const ARCH: Arch = Arch::Gfx1201;

fn spec(kv: FaKvCodec) -> FaPrefillSpec {
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv,
    }
}

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn json_sha<T: serde::Serialize>(v: &T) -> String {
    sha(&serde_json::to_string(v).unwrap())
}

// `hipfire-engine` intentionally enables serde_json/preserve_order (b26ba3b495).
// Workspace feature unification must not turn JSON object ordering into a
// kernel change: retain the sorted-key representation used by the original pins.
fn contract_sha(mut value: serde_json::Value) -> String {
    value.sort_all_objects();
    json_sha(&value)
}

/// `(name, sha256)` in a fixed order.
fn digests() -> Vec<(&'static str, String)> {
    let q8 = spec(FaKvCodec::Q8_0);
    let fp8 = spec(FaKvCodec::Fp8E4M3);
    let pre = emit_preconvert(ARCH, &q8).unwrap();
    let direct = emit(ARCH, &q8).unwrap();
    let direct8 = emit(ARCH, &fp8).unwrap();
    let (_, module_text, module_proof) = emit_module(ARCH, &q8).unwrap();
    let (_, module8_text, module8_proof) = emit_fp8_module(ARCH, &fp8).unwrap();
    vec![
        ("preconvert.s_text", sha(&pre.s_text)),
        ("preconvert.proof", json_sha(&pre.proof)),
        ("direct.s_text", sha(&direct.s_text)),
        ("direct.proof", json_sha(&direct.proof)),
        ("fp8_direct.s_text", sha(&direct8.s_text)),
        ("fp8_direct.proof", json_sha(&direct8.proof)),
        ("module.text", sha(&module_text)),
        ("module.proof", json_sha(&module_proof)),
        ("fp8_module.text", sha(&module8_text)),
        ("fp8_module.proof", json_sha(&module8_proof)),
        (
            "contract.preconvert",
            contract_sha(preconvert_contract(ARCH, &pre)),
        ),
        ("contract.direct", contract_sha(contract(ARCH, &direct))),
        (
            "contract.fp8_direct",
            contract_sha(fp8_contract(ARCH, &direct8)),
        ),
    ]
}

/// Recorded from the generator before `fa_prefill::blocks` existed.
const PINNED: &[(&str, &str)] = &[
    (
        "preconvert.s_text",
        "cb251dd9508c1ec9986685cbd87e820b08f20cdd8e9da719541baed4cce71053",
    ),
    (
        "preconvert.proof",
        "247dabe88d0b2ea20d89aded3f94337887fb7966303b70acd75bb9f647c8d796",
    ),
    (
        "direct.s_text",
        "5710c49e341e4de332676f034fb2f8926c1a8b3e4e18e51833216672755519f8",
    ),
    (
        "direct.proof",
        "112529296a91a8b53d60cc1ca6ef278cb0093a23f3b38b937754ef6d97709978",
    ),
    (
        "fp8_direct.s_text",
        "dfcfd35268655249d0259e6284d103fce4277a3f0d9089a27451eda693b995e5",
    ),
    (
        "fp8_direct.proof",
        "05ab6c84517c00ea482a76d6851d55f60671ad1a09df130abf7b6a2f802ad2a0",
    ),
    (
        "module.text",
        "31cafebb9aa9ba63a06516f6441bbcb08d22c190b3a1a92546e144fa02c98a94",
    ),
    (
        "module.proof",
        "0ddfba68856f81f067baca72b9c2dea24d029e99e5231439766d1bbc7cb281d0",
    ),
    (
        "fp8_module.text",
        "dfcfd35268655249d0259e6284d103fce4277a3f0d9089a27451eda693b995e5",
    ),
    (
        "fp8_module.proof",
        "34589da6fd799c330fe17a1409844448eca6d361a9c3798fca61ebf2698d6e65",
    ),
    (
        "contract.preconvert",
        "243b716f27c9d0dd31008ce52d39c89ec4c4ca4bfd9cf144e2e6c74bece541dd",
    ),
    (
        "contract.direct",
        "2446050d978df5e38dff1951ca8a133a77650570098829c79e6051dfed99c71c",
    ),
    (
        "contract.fp8_direct",
        "b4469d6d931584101c6bd771f5c9c6c73bda7e51f15d7afca48d2c9aade9e616",
    ),
];

#[test]
fn fa_prefill_digests_pinned() {
    let got = digests();
    if std::env::var_os("FA_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (*n, d.as_str())).collect();
    assert_eq!(got, PINNED, "fa_prefill emitted bytes changed");
}
