// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Complete baseline opcode audit of frozen wide-verify images #6–#10. The
//! fixture records every unsupported instance, not merely each seal's first
//! error. GPR-index controls remain deliberately negative: enabling their rows
//! without the following VALU's indexed effects would be unsound.
use peacemaker_cdna::inst::RowStatus;
use peacemaker_cdna::{Arch, Inst};
use serde_json::Value;
use std::collections::BTreeSet;

#[test]
fn all_frozen_non_indexed_opcode_gaps_have_real_typed_models() {
    let audit: Value = serde_json::from_str(include_str!(
        "fixtures/cdna/gfx942_wide_queue_opcode_audit.json"
    ))
    .unwrap();
    assert_eq!(audit["baseline_compiler"], "00818bfb54");
    assert_eq!(audit["source_tip"], "8231c29205");
    let images = audit["images"].as_array().unwrap();
    assert_eq!(images.len(), 5);
    let mut instructions = 0;
    let mut promoted = 0;
    let mut deferred = 0;
    let mut names = BTreeSet::new();
    for image in images {
        assert_eq!(image["native_elf_sha256"].as_str().unwrap().len(), 64);
        instructions += image["instructions"].as_u64().unwrap();
        for row in image["unmodelled_instances"].as_array().unwrap() {
            assert!(!row["baseline_refusal"].as_str().unwrap().is_empty());
            let words: Vec<u32> = row["words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|word| u32::try_from(word.as_u64().unwrap()).unwrap())
                .collect();
            let (inst, consumed) = Inst::decode(Arch::Gfx942, &words).unwrap();
            assert_eq!(consumed, words.len(), "{row}");
            assert_eq!(inst.words(Arch::Gfx942).unwrap(), words, "{row}");
            let mnemonic = row["assembly"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap();
            if matches!(mnemonic, "s_set_gpr_idx_on" | "s_set_gpr_idx_off") {
                assert!(
                    inst.unsupported().is_some(),
                    "indexed controls must not be silent no-ops: {row}"
                );
                deferred += 1;
            } else {
                assert!(inst.unsupported().is_none(), "{row}: {inst:?}");
                assert!(
                    matches!(inst.verification(Arch::Gfx942), RowStatus::Verified),
                    "{row}"
                );
                names.insert(mnemonic.to_owned());
                promoted += 1;
            }
        }
    }
    assert_eq!(instructions, 20_938);
    assert_eq!((promoted, deferred), (12, 8));
    assert_eq!(
        names,
        BTreeSet::from([
            "ds_or_b32".to_owned(),
            "v_cmp_ge_u64_e32".to_owned(),
            "v_cmp_nlg_f32_e64".to_owned(),
        ])
    );
}
