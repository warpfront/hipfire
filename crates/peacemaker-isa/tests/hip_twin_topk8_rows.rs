// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Portable proof-only Route-A twins; dispatch remains HIP.
use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco_many};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] = include_bytes!("../../../kernels/topk8_rows_mq6g256v2_hipcc_gfx1201.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/topk8_rows_mq6g256v2_pm_gfx1201.hxaco");
const MANIFEST: &str = include_str!("../../../kernels/topk8_rows_mq6g256v2_pm_gfx1201.twin.json");
const SHA: &str = include_str!("../../../kernels/topk8_rows_mq6g256v2_pm_gfx1201.hxaco.sha256");
const SYMBOLS: [&str; 2] = [
    "topk8_partial_f32_rows",
    "topk8_rescore_mq6g256v2_k2560_rows",
];
const PARTS: [(&str, &[u8]); 3] = [
    (
        "kernels/src/topk8_select.hip",
        include_bytes!("../../../kernels/src/topk8_select.hip"),
    ),
    (
        "kernels/src/qwen4_gemv_mq6g256v2.hip",
        include_bytes!("../../../kernels/src/qwen4_gemv_mq6g256v2.hip"),
    ),
    (
        "kernels/src/topk8_rows.gfx1201.hip",
        include_bytes!("../../../kernels/src/topk8_rows.gfx1201.hip"),
    ),
];
fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

#[test]
fn provenance_and_portability() {
    let m: Value = serde_json::from_str(MANIFEST).unwrap();
    assert_eq!(m["arch"], "gfx1201");
    assert_eq!(m["symbols"], serde_json::json!(SYMBOLS));
    assert_eq!(m["anchor"]["sha256"], hex(ANCHOR));
    assert_eq!(m["hxaco"]["sha256"], hex(HXACO));
    assert_eq!(m["anchor"]["bytes"], ANCHOR.len());
    assert_eq!(m["hxaco"]["bytes"], HXACO.len());
    assert_eq!(SHA.trim(), hex(HXACO));
    let mut combined = Vec::new();
    for ((path, bytes), recorded) in PARTS
        .iter()
        .zip(m["module_source"]["parts"].as_array().unwrap())
    {
        assert_eq!(recorded["path"], *path);
        assert_eq!(recorded["sha256"], hex(bytes));
        assert_eq!(recorded["bytes"], bytes.len());
        combined.extend_from_slice(bytes);
    }
    assert_eq!(m["module_source"]["combined_sha256"], hex(&combined));
    assert_eq!(m["module_source"]["combined_bytes"], combined.len());
    assert_eq!(
        m["compiler"]["flags"],
        serde_json::json!([
            "--genco",
            "--offload-arch=gfx1201",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-mcode-object-version=6"
        ])
    );
}

#[test]
fn full_elf_identity_and_explicit_abi() {
    fn device(bytes: &[u8]) -> &[u8] {
        let (bundle, payloads) = Bundle::read(bytes).unwrap();
        payloads[bundle
            .device_entry(peacemaker_ir::inst::Arch::Gfx1201)
            .unwrap()]
    }
    assert_eq!(device(ANCHOR), device(HXACO), "full device ELF identity");
    let m: Value = serde_json::from_str(MANIFEST).unwrap();
    let verified = verify_hxaco_many(HXACO, Arch::Gfx1201, &SYMBOLS).unwrap();
    let wants = [
        vec![
            (0, 8, true),
            (8, 4, false),
            (12, 4, false),
            (16, 8, true),
            (24, 4, false),
            (28, 4, false),
            (32, 4, false),
            (36, 4, false),
        ],
        vec![
            (0, 8, true),
            (8, 8, true),
            (16, 8, true),
            (24, 8, true),
            (32, 4, false),
            (36, 4, false),
        ],
    ];
    for (i, (symbol, v)) in SYMBOLS.iter().zip(verified).enumerate() {
        let sealed = seal(ANCHOR, Arch::Gfx1201, symbol, Some((ANCHOR, symbol))).unwrap();
        assert_eq!(sealed.hxaco, HXACO);
        assert_eq!(v.hxaco, HXACO);
        assert_eq!(v.elf_sha256, hex(device(ANCHOR)));
        assert_eq!(m["seal"]["reports"][i]["kernel"], *symbol);
        assert_eq!(m["seal"]["reports"][i]["text_sha256"], v.text_sha256);
        assert_eq!(m["seal"]["reports"][i]["hazards"]["blocking"], 0);
        let args: Vec<_> = v.report["abi"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_"))
            .map(|a| {
                (
                    a["offset"].as_u64().unwrap(),
                    a["size"].as_u64().unwrap(),
                    a["pointer"].as_bool().unwrap(),
                )
            })
            .collect();
        assert_eq!(args, wants[i]);
    }
}

#[test]
fn wrong_arch_and_symbol_fail_closed() {
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1200, &SYMBOLS).is_err());
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1201, &["no_such_topk8_entry"]).is_err());
}
