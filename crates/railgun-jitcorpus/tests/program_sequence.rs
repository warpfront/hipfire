// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Raon plain-AR decode program in the JIT corpus inventory: its node set
//! comes from a recorded launch sequence (`--program-sequence`), under its own
//! program id, with the exact loaded object digest per node. CPU only.
use railgun_jitcorpus::{
    archs, current_program_keys, inventory, plan, raon_ar_program_id, Inventory, NodeKey,
};

/// First launch of each distinct node of the real Raon decode body (probe inventory phase).
const REAL_NODES: &str = include_str!("fixtures/raon-ar-decode-nodes.jsonl");

const SHA_A: &str = "737484c8eae52e4e09fdfb7752e0b3449a93856bb500ae52946f47ffe386ca07";
const SHA_B: &str = "ac0bebbb956d9b4d178a4cce600ba2d2c46e994ab47e6885ed669d67286ddf4f";

fn line(kind: &str, symbol: &str, module: &str, sha: &str) -> String {
    format!(
        r#"{{"i":0,"kind":"{kind}","symbol":"{symbol}","module":"{module}","sha256":"{sha}","grid":[1,1,1],"block":[256,1,1],"lds":0,"kernarg_hex":"00"}}"#
    )
}

fn add(jsonl: &str) -> Result<Inventory, String> {
    let mut inv = inventory()?;
    inv.add_program_sequence("raon-ar-decode-gfx1201", "gfx1201", jsonl)?;
    Ok(inv)
}

#[test]
fn program_id_is_raon_and_never_the_dflash_window_label() {
    assert_eq!(raon_ar_program_id("gfx1201", 0), "raon-ar-decode-gfx1201");
    assert_eq!(raon_ar_program_id("gfx1201", 1), "raon-ar-decode-gfx1201-2");
    assert_eq!(raon_ar_program_id("gfx1100", 2), "raon-ar-decode-gfx1100-3");
    assert!(!raon_ar_program_id("gfx1201", 0).contains("dflash"));
}

#[test]
fn real_raon_sequence_parses_to_its_distinct_nodes_with_their_loaded_digests() {
    let inv = add(REAL_NODES).unwrap();
    let set = &inv.sequence_sets[0];
    assert_eq!(set.id, "raon-ar-decode-gfx1201");
    assert_eq!(set.arch, "gfx1201");
    // 32 distinct (symbol, module) nodes in 27 objects: modules share objects.
    assert_eq!(set.nodes.len(), 32);
    assert_eq!(set.loaded_sha256.len(), 32);
    assert_eq!(set.launches, 32);
    assert_eq!(
        set.nodes[0],
        ("embedding_q8_batched".into(), "embedding_q8_batched".into())
    );
    assert_eq!(set.loaded_sha256[0], SHA_A);
    let objects: std::collections::BTreeSet<&String> = set.loaded_sha256.iter().collect();
    assert_eq!(objects.len(), 27, "27 distinct loaded objects");
    // The sequence is its own node set; it is not a DFlash window set.
    assert!(inv.window_sets.is_empty());
    let sets = inv.node_sets();
    let raon = sets
        .iter()
        .find(|s| s.id == "raon-ar-decode-gfx1201")
        .expect("raon node set");
    assert_eq!(raon.loaded_sha256.map(<[String]>::len), Some(32));
    assert!(sets
        .iter()
        .filter(|s| s.id != "raon-ar-decode-gfx1201")
        .all(|s| s.loaded_sha256.is_none()));
}

/// Every module the real Raon decode body loads resolves in the runtime's own registry on gfx1201
/// (source expression + recipe under the product defaults), so `build` can compile and certify it.
#[test]
fn every_raon_decode_module_resolves_in_the_registry_and_gets_a_current_key() {
    let inv = add(REAL_NODES).unwrap();
    let plan = plan(&inv, "gfx1201").unwrap();
    for (_, module) in &inv.sequence_sets[0].nodes {
        assert!(
            plan.entries.iter().any(|e| e.module == module)
                || plan.embedded.iter().any(|n| n.module == module),
            "module {module} missing from the plan"
        );
    }
    // The RDA recurrence and causal conv run from the sealed embedded image, not a hipcc module.
    assert_eq!(
        plan.embedded.iter().map(|n| n.module).collect::<Vec<_>>(),
        ["gated_delta_vector_pm_gfx1201"]
    );
    let keys = current_program_keys(&inv).unwrap();
    let raon = keys
        .iter()
        .find(|k| k.id == "raon-ar-decode-gfx1201")
        .unwrap();
    assert_eq!(raon.nodes.len(), 32);
    assert_eq!(raon.loaded_sha256.as_ref().map(Vec::len), Some(32));
    // Embedded-image nodes key by the image digest, the rest by the packaging key.
    let native: Vec<&str> = raon
        .nodes
        .iter()
        .filter(|(_, _, k)| matches!(k, NodeKey::Native { .. }))
        .map(|(s, _, _)| s.as_str())
        .collect();
    assert_eq!(
        native,
        [
            "causal_conv_depthwise_silu_rows_f32_pm_gfx1201",
            "gated_delta_net_vector_serial_f32_pm_gfx1201"
        ]
    );
    assert_eq!(raon.nodes.len() - native.len(), 30);
    assert!(archs(&inv).iter().any(|a| a == "gfx1201"));
}

#[test]
fn fills_have_no_object_and_duplicates_collapse() {
    let seq = [
        line(
            "kernel",
            "embedding_q8_batched",
            "embedding_q8_batched",
            SHA_A,
        ),
        r#"{"i":1,"kind":"fill","dst":1,"value":0,"words":4}"#.to_owned(),
        line(
            "kernel",
            "embedding_q8_batched",
            "embedding_q8_batched",
            SHA_A,
        ),
        line(
            "kernel",
            "stream_mix_broadcast_bf16",
            "stream_mix_broadcast",
            SHA_B,
        ),
        String::new(),
        "# a comment".to_owned(),
    ]
    .join("\n");
    let inv = add(&seq).unwrap();
    let set = &inv.sequence_sets[0];
    assert_eq!(set.nodes.len(), 2);
    assert_eq!((set.launches, set.fills), (3, 1));
    assert_eq!(set.loaded_sha256, vec![SHA_A.to_owned(), SHA_B.to_owned()]);
}

#[test]
fn malformed_sequences_are_refused_with_the_line() {
    let err = |text: &str| add(text).unwrap_err();
    assert!(err("").contains("names no kernel node"));
    assert!(err(&line("fill", "x", "y", SHA_A)).contains("no kernel node"));
    assert!(err("not json").contains("line 1"));
    assert!(
        err(&line("launch", "a", "a", SHA_A)).contains("unknown node kind `launch`"),
        "unknown kind"
    );
    let no_sha = r#"{"kind":"kernel","symbol":"a","module":"b"}"#;
    assert!(err(no_sha).contains("line 1") && err(no_sha).contains("no sha256"));
    let no_module = r#"{"kind":"kernel","symbol":"a","sha256":"00"}"#;
    assert!(err(no_module).contains("no module"));
    assert!(err(&line("kernel", "a", "b", "abc")).contains("not 64 lowercase hex"));
    assert!(err(&line("kernel", "a", "b", &SHA_A.to_uppercase())).contains("not 64 lowercase hex"));
    // One node loading two different objects has no single certified object.
    let two = [
        line("kernel", "a", "b", SHA_A),
        line("kernel", "a", "b", SHA_B),
    ]
    .join("\n");
    let e = err(&two);
    assert!(
        e.contains("line 2") && e.contains("earlier launch loaded"),
        "{e}"
    );
    // One symbol under two modules is ambiguous for the registry lookup.
    let two = [
        line("kernel", "a", "b", SHA_A),
        line("kernel", "a", "c", SHA_A),
    ]
    .join("\n");
    assert!(err(&two).contains("appears under modules b and c"));
}

#[test]
fn a_sequence_id_cannot_be_added_twice() {
    let mut inv = add(&line("kernel", "a", "b", SHA_A)).unwrap();
    let e = inv
        .add_program_sequence(
            "raon-ar-decode-gfx1201",
            "gfx1201",
            &line("kernel", "a", "b", SHA_A),
        )
        .unwrap_err();
    assert!(e.contains("already in the inventory"), "{e}");
}

/// An unknown module is not silently dropped: `plan` names the node and its module.
#[test]
fn a_module_missing_from_the_registry_fails_the_plan_naming_the_node() {
    let inv = add(&line("kernel", "mystery_kernel", "mystery_module", SHA_A)).unwrap();
    let e = match plan(&inv, "gfx1201") {
        Ok(_) => panic!("plan accepted a module that is not in the registry"),
        Err(e) => e,
    };
    assert!(
        e.contains("raon-ar-decode-gfx1201")
            && e.contains("mystery_kernel")
            && e.contains("mystery_module"),
        "{e}"
    );
}

#[test]
fn retain_programs_keeps_the_named_set_and_the_railgun_kernels() {
    let mut inv = add(REAL_NODES).unwrap();
    inv.retain_programs(&["raon-ar-decode-gfx1201".to_owned()])
        .unwrap();
    assert!(inv.programs.is_empty());
    let ids: Vec<String> = inv.node_sets().into_iter().map(|s| s.id).collect();
    assert!(ids.contains(&"raon-ar-decode-gfx1201".to_owned()));
    assert!(ids.iter().any(|i| i.starts_with("railgun-kernels-")));
    assert!(inv
        .retain_programs(&["no-such-program".to_owned()])
        .unwrap_err()
        .contains("no such program"));
}

/// First launch of each distinct node of the real gfx1100 (7900 XTX) Raon decode body.
const REAL_NODES_GFX1100: &str = include_str!("fixtures/raon-ar-decode-nodes-gfx1100.jsonl");

fn add_gfx1100(jsonl: &str) -> Result<Inventory, String> {
    let mut inv = inventory()?;
    inv.add_program_sequence(&raon_ar_program_id("gfx1100", 0), "gfx1100", jsonl)?;
    Ok(inv)
}

/// `--program-arch gfx1100` adds `raon-ar-decode-gfx1100` under arch gfx1100 and every module of the
/// real gfx1100 decode body resolves in the gfx1100 registry (hipcc module or sealed embedded
/// image), with the arch's own `_pm_gfx1100` RDA twins and no gfx1201 name.
#[test]
fn the_gfx1100_raon_decode_sequence_plans_on_its_own_arch() {
    let inv = add_gfx1100(REAL_NODES_GFX1100).unwrap();
    let set = &inv.sequence_sets[0];
    assert_eq!(set.id, "raon-ar-decode-gfx1100");
    assert_eq!(set.arch, "gfx1100");
    assert_eq!(set.nodes.len(), 33);
    assert_eq!(set.loaded_sha256.len(), 33);
    assert!(set
        .nodes
        .iter()
        .all(|(s, m)| !s.contains("gfx1201") && !m.contains("gfx1201")));
    assert!(archs(&inv).iter().any(|a| a == "gfx1100"));
    let plan = plan(&inv, "gfx1100").unwrap_or_else(|e| panic!("{e}"));
    for (_, module) in &set.nodes {
        assert!(
            plan.entries.iter().any(|e| e.module == module)
                || plan.embedded.iter().any(|n| n.module == module)
                || plan.twins.iter().any(|t| t.module == module),
            "module {module} missing from the gfx1100 plan"
        );
    }
    let keys = current_program_keys(&inv).unwrap();
    let raon = keys
        .iter()
        .find(|k| k.id == "raon-ar-decode-gfx1100")
        .expect("raon gfx1100 program");
    assert_eq!(raon.arch, "gfx1100");
    assert_eq!(raon.nodes.len(), 33);
    assert_eq!(raon.loaded_sha256.as_ref().map(Vec::len), Some(33));
}

/// The gfx1100 program is selected alone (plus railgun's own kernels) by `--only-program`, and a
/// gfx1201 sequence cannot be relabelled as gfx1100 by id.
#[test]
fn only_program_selects_the_gfx1100_decode_set() {
    let mut inv = add_gfx1100(REAL_NODES_GFX1100).unwrap();
    inv.retain_programs(&["raon-ar-decode-gfx1100".to_owned()])
        .unwrap();
    assert!(inv.programs.is_empty());
    let sets = inv.node_sets();
    assert!(sets
        .iter()
        .any(|s| s.id == "raon-ar-decode-gfx1100" && s.arch == "gfx1100"));
    assert!(sets
        .iter()
        .filter(|s| s.id != "raon-ar-decode-gfx1100")
        .all(|s| s.id.starts_with("railgun-kernels-")));
}
