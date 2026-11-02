// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every valid `fp8_gemm::Spec` (and the product module) re-emits the bytes recorded before the
//! building blocks (`fp8_gemm::blocks`) were factored out of `emit`: the sha256 of
//! `emit(spec).s_text` per spec and of the `emit_module` text, computed from the unrefactored
//! generator. A change to any digest is a change to a production kernel and must be intentional.

use peacemaker_isa::kernels::fp8_gemm::{emit, emit_module, ActScale, Epi, Spec};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

/// `(variant name, sha256 of s_text)` for every valid spec, in iteration order.
const PINNED: &[(&str, &str)] = &[
    (
        "ratio-256x128x8-row.set",
        "0724466e5048a33f09bdef5baa57214f1e4e92c85139e035212af76edba22d56",
    ),
    (
        "ratio-256x128x8-row.add",
        "e41fce5a4ea552f4ec1df2c3984c2215261271cc17f085def1bb32b45f7d0eb2",
    ),
    (
        "ratio-256x128x8-row.silu",
        "536bf970692ba601dd91fbdb33cb3374589f63083356a8931fa5944c134205a6",
    ),
    (
        "ratio-256x128x8-row.silu-bf16",
        "2616121567d950c1b154f04594dfbd8010f407ad1be8d3b4c7cec7f09ba0be88",
    ),
    (
        "ratio-256x128x8-row.qkv",
        "0fb2c84f5809e2519bbc532953933dfc5524e6b737467da3cf5e1ba6dad2a7e6",
    ),
    (
        "ratio-256x128x8-row.qkvza",
        "15b1bd087990fea7ff388cb7c2c48bbefb6520c19a5b59ce4808c5e676d9a540",
    ),
    (
        "ratio-256x128x8-row.qkvzagdn",
        "c05466e0904cd7a29f6fd6c40fcde46bc91cad6eb2ba6a65d2eb71b260f3d4aa",
    ),
    (
        "ratio-256x128x8-k128.set",
        "4d27623779232d7c29c5e04cd77a716b0b35632c89f81981cdfd2ab3a86e5227",
    ),
    (
        "ratio-256x128x8-k128.add",
        "c7d7e0f10ca2afbcae64f3ef1ba431b7bcd178b2d4885a6aa0d59db91a4180c4",
    ),
    (
        "ratio-256x128x8-k128.silu",
        "0db5296cc3c2dab350e72991c99296f0b1850ca92d836d5c35126d210cb82fc6",
    ),
    (
        "ratio-256x128x8-k128.qkv",
        "9171218ead262f5eda9005ad2037d70b62d3c7fb434f7c3be9078e3bea938c46",
    ),
    (
        "ratio-256x128x8-k128.qkvza",
        "93333a78d5ea42007868c1f9cc9382132c160a2532376c2505720506a65a0887",
    ),
];

/// sha256 of the `emit_module` text for the row-scaled product selection (the SiLU bf16 sibling and
/// the GDN-fused QKVZA ride along).
const PINNED_MODULE: &str = "365ecb4e3bcafa186b4a9ad62389c65c14791b01df292ce336d55a94c37c588d";

const EPIS: [Epi; 7] = [
    Epi::Set,
    Epi::Add,
    Epi::GateUpSilu,
    Epi::GateUpSiluBf16,
    Epi::Qkv,
    Epi::Qkvza,
    Epi::QkvzaGdn,
];

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn digests() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for act_scale in [ActScale::Row, ActScale::K128] {
        for epi in EPIS {
            let spec = Spec {
                arch: Arch::Gfx1201,
                act_scale,
                epi,
            };
            if spec.validate().is_err() {
                continue;
            }
            let e = emit(spec)
                .unwrap_or_else(|e| panic!("{} is valid but does not emit: {e}", spec.variant()));
            out.push((spec.variant(), sha(&e.s_text)));
        }
    }
    out
}

#[test]
fn fp8_gemm_emit_digests_pinned() {
    let got = digests();
    if std::env::var_os("FP8_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, PINNED, "fp8_gemm emitted bytes changed");
}

#[test]
fn fp8_gemm_module_digest_pinned() {
    // The CLI's `--epi all` selection per scale, then the module the runtime bundle is built from.
    let (_, text, proof) = emit_module(
        Arch::Gfx1201,
        ActScale::Row,
        &[Epi::Set, Epi::Add, Epi::GateUpSilu, Epi::Qkv, Epi::Qkvza],
    )
    .unwrap();
    if std::env::var_os("FP8_PRINT_DIGESTS").is_some() {
        println!("module {}", sha(&text));
    }
    assert_eq!(sha(&text), PINNED_MODULE, "fp8_gemm module text changed");
    assert_eq!(proof.s_text_sha256, sha(&text));
}
