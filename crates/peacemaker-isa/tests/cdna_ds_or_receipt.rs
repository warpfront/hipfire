// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact old-value/final-LDS evidence, without imposing a lane order on contended atomics.
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::{Arch, Src};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const RECEIPT: &str = include_str!("fixtures/cdna/gfx942_ds_or_receipt.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_ds_or_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_ds_or_oracle.txt");
const OFFSETS: [u16; 7] = [0, 4, 252, 1024, 4096, 16380, 32384];
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_sha(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}
fn vreg(value: &Value) -> Src {
    Src::Vgpr(
        value
            .as_str()
            .unwrap()
            .strip_prefix('v')
            .unwrap()
            .parse()
            .unwrap(),
    )
}

#[test]
fn ds_or_hardware_receipt_is_bound_to_its_exercised_sources() {
    let receipt: Value = serde_json::from_str(RECEIPT).unwrap();
    let compare: Value = serde_json::from_str(include_str!(
        "fixtures/cdna/gfx942_float_compare_oracle.json"
    ))
    .unwrap();
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["raw_oracle_sha256"], sha(ORACLE.as_bytes()));
    assert_eq!(receipt["raw_log_sha256"], sha(LOG.as_bytes()));
    assert_eq!(receipt["ledger"]["exit"], 0);
    assert!(receipt["ledger"]["cmd"]
        .as_str()
        .unwrap()
        .contains("run-compiler-oracles"));
    assert!(is_sha(&receipt["binary_sha256"]));
    for (field, bytes) in [
        (
            "oracle_source_sha256",
            &include_bytes!("../examples/cdna_ds_or_hw_oracle.rs")[..],
        ),
        (
            "interpreter_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/sim/mem.rs")[..],
        ),
        (
            "codec_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/enc/ds.rs")[..],
        ),
    ] {
        assert_eq!(oracle[field], sha(bytes), "{field}");
    }
    // These shared sources are stamped by the companion compare executable from the same build.
    for (field, bytes) in [
        (
            "interpreter_mod_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/sim/mod.rs")[..],
        ),
        (
            "codec_inst_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/inst.rs")[..],
        ),
    ] {
        assert_eq!(receipt[field], compare[field], "shared source {field}");
        assert_eq!(receipt[field], sha(bytes), "{field}");
    }
    assert_eq!(oracle["status"], "PASS");
    assert_eq!(oracle["oracle"], "cdna_ds_or_hw_oracle");
    assert_eq!(oracle["arch"], "gfx942");
    assert_eq!(oracle["kernels"], 21);
    assert_eq!(oracle["cases_per_kernel_poison"], 96);
    assert_eq!(oracle["cases"], 4032);
    assert_eq!(oracle["empty_exec_cases"], 672);
    assert_eq!(oracle["controls_detected"], 504);
    assert_eq!(oracle["offsets"], serde_json::json!(OFFSETS));
    assert_eq!(oracle["poisons"], serde_json::json!([90, 165]));
    for field in [
        "exact_return_lanes",
        "inactive_lanes_checked",
        "contended_groups_linearized",
        "contended_lanes",
    ] {
        assert!(oracle[field].as_u64().unwrap() > 0, "{field}");
    }
    for field in [
        "hip_source_sha256",
        "code_object_sha256",
        "device_elf_sha256",
        "disassembly_sha256",
        "hardware_output_sha256",
    ] {
        assert!(is_sha(&oracle[field]), "{field}");
    }
}

#[test]
fn both_atomic_forms_and_data_alias_cover_every_offset_poison_and_wait() {
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    let bindings = oracle["kernel_bindings"].as_array().unwrap();
    assert_eq!(bindings.len(), 21);
    let mut names = BTreeSet::new();
    let mut forms = BTreeMap::<u16, [usize; 3]>::new();
    for binding in bindings {
        let name = binding["kernel"].as_str().unwrap();
        assert!(names.insert(name));
        assert_eq!(binding["lgkm0_wait_follows"], true);
        let words: Vec<u32> = binding["codec_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_u64().unwrap().try_into().unwrap())
            .collect();
        let (inst, consumed) = Ds::decode(Arch::Gfx942, &words).unwrap();
        assert_eq!(consumed, words.len());
        let mut encoded = Vec::new();
        inst.encode(Arch::Gfx942, &mut encoded).unwrap();
        assert_eq!(encoded, words);
        assert!(!inst.gds);
        assert!(inst.data1.is_none());
        let offset: u16 = binding["offset"].as_u64().unwrap().try_into().unwrap();
        assert_eq!(inst.offset(), offset);
        assert!(OFFSETS.contains(&offset));
        let regs = binding["regs"].as_array().unwrap();
        let form = match inst.op {
            DsOp::OrB32 => {
                assert_eq!(regs.len(), 2);
                assert!(inst.vdst.is_none());
                assert_eq!(inst.addr, Some(vreg(&regs[0])));
                assert_eq!(inst.data0, Some(vreg(&regs[1])));
                0
            }
            DsOp::OrRtnB32 => {
                assert_eq!(regs.len(), 3);
                assert_eq!(inst.vdst, Some(vreg(&regs[0])));
                assert_eq!(inst.addr, Some(vreg(&regs[1])));
                assert_eq!(inst.data0, Some(vreg(&regs[2])));
                if regs[0] == regs[2] {
                    2
                } else {
                    assert_ne!(regs[0], regs[1]);
                    1
                }
            }
            other => panic!("unexpected atomic {other:?}"),
        };
        forms.entry(offset).or_default()[form] += 1;
    }
    assert_eq!(forms.len(), OFFSETS.len());
    for offset in OFFSETS {
        assert_eq!(forms[&offset], [1, 1, 1]);
    }
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 42);
    let mut seen = BTreeSet::new();
    let mut cases = 0;
    for row in rows {
        let name = row["kernel"].as_str().unwrap();
        assert!(names.contains(name));
        let poison = row["poison"].as_u64().unwrap();
        assert!([90, 165].contains(&poison));
        assert!(seen.insert((name, poison)));
        assert_eq!(row["cases"], 96);
        assert_eq!(row["cases_verified"], 96);
        assert_eq!(row["verified"], true);
        assert_eq!(row["guards_ok"], true);
        assert!(is_sha(&row["hardware_sha256"]));
        cases += row["cases_verified"].as_u64().unwrap();
    }
    assert_eq!(cases, 4032);
    for name in names {
        for poison in [90, 165] {
            assert!(seen.contains(&(name, poison)));
        }
    }
}
