// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certification of the exact gfx1100 (7900 XTX) code object that carries the MQ8G256 attention-class decode GEMVs of
//! the Raon plain-AR decode (`raon-1-mini-preview.mq4-xts` with `mq8attn` tensors, beta `6315981bd6`): the fixture is the
//! object (and radiowave sidecar) `raon_oracle railgun-probe --phase inventory` found loaded on the XTX at ctx 512 and
//! ctx 4096, byte for byte (the object SHA-256 is pinned below). Its three kernels launch 93 / 25 / 7 times per token.
//!
//! The gfx1201 owner declines this model with `no_receipt: gemv_mq8g256_bx_r1_u5_w24_p1`; on gfx1100 the same
//! kernels lift on the existing table and every one is Proven with a passing sidecar differential: four read-only
//! pointer arguments (the quantised weights, their scales/metadata and the activation planes), one write-only output
//! (`+32`), no atomics, no LDS-DMA, no tables, no indexed effects and no witnesses.
use railgun_cert::{certify, Receipt, STATUS_PROVEN};

const OBJECT: &[u8] = include_bytes!("fixtures/raon_mq8g256_bx_gemv.gfx1100.hsaco");
const SIDECAR: &str = include_str!("fixtures/raon_mq8g256_bx_gemv.gfx1100.radiowave.json");
const SHA: &str = "f829307f36df50207500f5a653c152e47968dc17a125cf37dc9d3eb5ece14dcf";
/// (symbol, launches per decode token)
const KERNELS: [(&str, u32); 3] = [
    ("gemv_mq8g256_bx_r1_u5_w24_p1", 93),
    ("gemv_mq8g256_bx_r1_u10_w20_p1", 25),
    ("gemv_mq8g256_bx_r1_u1_w10_p1", 7),
];

#[test]
fn the_gfx1100_mq8_decode_gemvs_are_proven_read_only_weights_one_output() {
    let r: Receipt =
        certify(OBJECT, Some(SIDECAR)).expect("the gfx1100 MQ8 object lifts and certifies");
    assert_eq!(r.arch, "gfx1100");
    assert_eq!(r.object_sha256, SHA, "the fixture is the loaded object");
    // The object is the whole `gemv_mq8g256_b` module: 22 kernels (every `bx` shape variant the dispatcher can pick), of
    // which the decode launches three. All of them are Proven, so another n / shape never meets an uncertified kernel.
    assert_eq!(r.railgun.kernels.len(), 22);
    assert!(
        r.railgun.kernels.iter().all(|k| k.status == STATUS_PROVEN),
        "{:?}",
        r.railgun
            .kernels
            .iter()
            .filter(|k| k.status != STATUS_PROVEN)
            .map(|k| (&k.symbol, &k.status))
            .collect::<Vec<_>>()
    );
    for (symbol, _launches) in KERNELS {
        let k = r
            .railgun
            .kernels
            .iter()
            .find(|k| k.symbol == symbol)
            .unwrap_or_else(|| panic!("{symbol} missing"));
        assert_eq!(k.status, STATUS_PROVEN, "{symbol}: {:?}", k.unresolved);
        assert!(k.unresolved.is_empty(), "{symbol}: {:?}", k.unresolved);
        assert!(
            k.required_witnesses.is_empty()
                && k.sealed_tables.is_empty()
                && k.indexed_effects.is_empty()
                && k.guards.is_empty(),
            "{symbol}: a proven kernel carries no conditional case"
        );
        assert_eq!(
            (
                k.differential.source.as_str(),
                k.differential.status.as_str()
            ),
            ("sidecar", "pass"),
            "{symbol}"
        );
        assert_eq!(k.kernarg_size, 48, "{symbol}");
        let pointers: Vec<(u32, Option<&str>)> = k
            .args
            .iter()
            .filter(|a| a.value_kind == "global_buffer")
            .map(|a| (a.offset, a.mode.as_deref()))
            .collect();
        assert_eq!(
            pointers,
            [
                (0, Some("read")),
                (8, Some("read")),
                (16, Some("read")),
                (24, Some("read")),
                (32, Some("write")),
            ],
            "{symbol}: four read-only inputs and one output at +32"
        );
        let out = k.args.iter().find(|a| a.offset == 32).expect("output arg");
        assert_eq!(
            out.classes,
            ["vmem_store"],
            "{symbol}: the output is only stored to"
        );
        assert!(
            k.args
                .iter()
                .filter(|a| a.offset < 32 && a.value_kind == "global_buffer")
                .all(|a| !a
                    .classes
                    .iter()
                    .any(|c| c.contains("store") || c.contains("atomic"))),
            "{symbol}: the inputs are never stored to or atomically updated"
        );
        assert!(
            k.validate_case(&r.object_sha256, &r.device_elf_sha256, &r.arch)
                .is_ok(),
            "{symbol}"
        );
    }
    // The serialised receipt round-trips.
    let back: Receipt = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back.object_sha256, r.object_sha256);
    assert_eq!(back.railgun.kernels.len(), r.railgun.kernels.len());
}
