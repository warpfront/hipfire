// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Downstream-only (hipfire vendored copy). Upstream `CompileManifest` has no
//! `peacemaker` field and upstream has no `PeacemakerRecord`, so this file is
//! NOT part of the byte-identical shared set and must not be sent upstream.
//! It pins that the union schema leaves the downstream Peacemaker-carrying
//! manifest unchanged and that the two schemas still refuse each other.

use radiowave::union::CertifiedExecutableRecord;
use radiowave::{CompileManifest, PeacemakerRecord};
use serde_json::{Value, json};

fn manifest_with_peacemaker() -> Value {
    json!({
        "schema_version": 3,
        "compiler": "radiowave",
        "generated_unix_seconds": 0,
        "source": "kernel.hip",
        "output": "kernel.hsaco",
        "arch": "gfx1201",
        "wavefront": "wave64",
        "scheduler_profile": "default",
        "hipcc": "hipcc",
        "hipcc_version": "test",
        "command": [],
        "source_sha256": "source",
        "support_header_sha256": "header",
        "output_sha256": "output",
        "inspection": null,
        "peacemaker": {
            "arm": "custom-isa",
            "producer": {
                "builder_crate_version": "0.4.2",
                "builder_git_sha": "abc",
                "proof_sha256": "def"
            },
            "tools": [],
            "s_text_sha256": "a",
            "text_section_sha256": "b",
            "shape_contract": {},
            "shape_result": {},
            "oracle_receipt": {
                "suite": "mq4",
                "fixtures_sha256": "c",
                "cases": 4,
                "differing_bits": 0,
                "device_uuid": "GPU-x",
                "date": "2026-10-11"
            }
        }
    })
}

#[test]
fn peacemaker_manifest_round_trips_unchanged() {
    let original = manifest_with_peacemaker();
    let manifest: CompileManifest = serde_json::from_value(original.clone()).unwrap();
    assert_eq!(serde_json::to_value(&manifest).unwrap(), original);
    assert!(!original.to_string().contains("union"));
    assert_eq!(manifest.schema_version, 3);
    assert!(manifest.peacemaker.is_some());

    let record: PeacemakerRecord = serde_json::from_value(original["peacemaker"].clone()).unwrap();
    assert!(record.oracle_receipt.is_some());
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        original["peacemaker"]
    );
}

#[test]
fn peacemaker_manifest_is_not_a_union_record() {
    let text = manifest_with_peacemaker().to_string();
    assert!(CertifiedExecutableRecord::from_json(&text).is_err());
}
