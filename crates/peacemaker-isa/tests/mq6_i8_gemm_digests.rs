// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every valid `mq6_i8_gemm::Spec` (and the product module) re-emits the bytes recorded
//! before the building blocks (`mq6_i8_gemm::blocks`) were factored out of `emit`: the sha256 of
//! `emit(spec).s_text` per spec and of the `emit_module` text, computed from beta's unrefactored
//! generator. A change to any digest is a change to a production kernel and must be intentional.

use peacemaker_isa::kernels::mq6_i8_gemm::{emit, emit_module, Epi, Raster, Spec, Tile};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

/// `(variant name, sha256 of s_text)` for every valid spec, in iteration order.
const PINNED: &[(&str, &str)] = &[
    (
        "128x128x8-set",
        "cff953525f9a0cc0f0123408d77825b06465f92335a6b2c67b235a58f85813c0",
    ),
    (
        "128x128x8-set-linear",
        "3dc3a3a20928e1d2b4a207ea6bb38066c9d3c19879ee34597e78527f3f925121",
    ),
    (
        "128x128x8-set_bf16",
        "75ff9e9c6f6468028691f9974c99ae89f40b121d3c213c4cf6e731814da6b576",
    ),
    (
        "128x128x8-set_bf16-linear",
        "bfe9929739fd35729475af2ed10e970b6cf3c8c73ca69f373edf5ae212a19238",
    ),
];

/// `(variant name, sha256 of s_text)` of the non-shipped tiles.
const PINNED_TILES: &[(&str, &str)] = &[
    (
        "256x128x16-set",
        "40dc38a1ccfcfee38c1616d8fecf6ff77a128325d32d3b79ff76057c10c1a762",
    ),
    (
        "256x128x16-set-linear",
        "40fcf1a0d5893bf4686a13db0b49bff1d2f9960fdc821e904aa58ba7bbbe112f",
    ),
    (
        "256x128x16-set_bf16",
        "94248d998a3b9e39639681635eea4b8f9bf64cfd1aedeea106454bc0ae051b69",
    ),
    (
        "256x128x16-set_bf16-linear",
        "855cadf5bf87a856465266fa8c2a5917817559799cdc683fdf4165da603fb2fd",
    ),
    (
        "256x128x16-set_nr",
        "75c4f0a9a6efe60d749316dfa0176b65b83a756db769dc0b2bdfc870dfc05a09",
    ),
    (
        "256x128x16-set_nr-linear",
        "d8a6e33915bb02752c7c26d422c14db586e538749ff0d3cefe247985160a1eee",
    ),
    (
        "128x256x16-set",
        "f3de033b28bd4aed0a1573c887f2abb87b7a687260161f3fe0e4a147347ae472",
    ),
    (
        "128x256x16-set-linear",
        "b88327f8c636f9a7d642152cf85a1340b1f0ff613f76456765c1c0279deb8666",
    ),
    (
        "128x256x16-set_bf16",
        "b0e4cbcb7ad03e02cb242d8adbdae217ab52966d59072b0bf622e6305d910a4a",
    ),
    (
        "128x256x16-set_bf16-linear",
        "192c289bd029721c35eb6b8ed28b6080f46b85138ee64c489f914bcde3965258",
    ),
];

/// sha256 of the `emit_module` text (the five banded product symbols: both epilogues on the
/// shipped 128x128 tile, whose entries are `PINNED`, on 128x256, and the QSA norm+RoPE `set_nr`
/// on 256x128, `PINNED_TILES`).
const PINNED_MODULE: &str = "220be0f421fecf946b2d063e355689498c35b00be4cd9bfcdfcddf28bfc78c95";

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn digests(tiles: &[Tile]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for &tile in tiles {
        for epi in [Epi::Set, Epi::SetBf16, Epi::SetNr] {
            for raster in [Raster::Banded, Raster::Linear] {
                let spec = Spec {
                    epi,
                    raster,
                    tile,
                    arch: Arch::Gfx1201,
                    tune: Default::default(),
                };
                if spec.validate().is_err() {
                    continue;
                }
                let e = emit(spec).unwrap_or_else(|e| {
                    panic!("{} is valid but does not emit: {e}", spec.variant_name())
                });
                out.push((spec.variant_name(), sha(&e.s_text)));
            }
        }
    }
    out
}

fn check(tiles: &[Tile], pinned: &[(&str, &str)], what: &str) {
    let got = digests(tiles);
    if std::env::var_os("MQ6_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, pinned, "{what} emitted bytes changed");
}

#[test]
fn mq6_i8_gemm_emit_digests_pinned() {
    check(&[Tile::T128x128], PINNED, "mq6_i8_gemm");
}

/// The tile axis: the 256x128 and 128x256 tiles, pinned once each was bit-equal to the exact
/// reference and to the shipped tile on the interpreter (`tests/mq6_i8_gemm.rs`). These digests
/// guard them; the shipped tile's are `PINNED`.
#[test]
fn mq6_i8_gemm_tile_digests_pinned() {
    check(
        &[Tile::T256x128, Tile::T128x256],
        PINNED_TILES,
        "mq6_i8_gemm tile",
    );
}

#[test]
fn mq6_i8_gemm_module_digest_pinned() {
    let (_, text, proof) = emit_module(Arch::Gfx1201).unwrap();
    if std::env::var_os("MQ6_PRINT_DIGESTS").is_some() {
        println!("module {}", sha(&text));
    }
    assert_eq!(sha(&text), PINNED_MODULE, "mq6_i8_gemm module text changed");
    assert_eq!(proof.s_text_sha256, sha(&text));
}
