// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Corpus lookup and the CI check over a one-object corpus built from a real
//! code object and a real railgun-cert receipt. CPU only.
use std::path::{Path, PathBuf};

use railgun_corpus::{
    sha256_hex, ArtifactKey, BuildHost, CorpusIndex, CorpusRecord, JitCorpus, Lookup, NodeRecord,
    ObjectOrigin, ProgramNode, ProgramRecord, RejectedObject, ToolchainPin, CORPUS_INDEX,
    CORPUS_SCHEMA, CORPUS_VERSION,
};
use railgun_jitcorpus::{check, NodeKey, ProgramKeys};

const OBJECT: &[u8] = include_bytes!("../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
const ARCH: &str = "gfx1201";

struct Fixture {
    root: PathBuf,
    sha: String,
    symbol: String,
    index: CorpusIndex,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("railgun-jitcorpus-{name}"));
        let _ = std::fs::remove_dir_all(&root);
        let receipt = railgun_cert::certify(OBJECT, None).expect("fixture certifies");
        let sha = sha256_hex(OBJECT);
        let symbol = receipt.railgun.kernels[0].symbol.clone();
        let object = format!("pack/{ARCH}/kt48.hsaco");
        let receipt_rel = format!("receipts/{ARCH}/{sha}.receipt.json");
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        for (rel, bytes) in [(&object, OBJECT), (&receipt_rel, receipt_bytes.as_slice())] {
            std::fs::create_dir_all(root.join(rel).parent().unwrap()).unwrap();
            std::fs::write(root.join(rel), bytes).unwrap();
        }
        let id = "HIP version: test".to_owned();
        let index = CorpusIndex {
            schema: CORPUS_SCHEMA.into(),
            version: CORPUS_VERSION,
            source_commit: "test".into(),
            build_host: BuildHost {
                key: "k".repeat(64),
                hostname: "test".into(),
                hipcc: "hipcc".into(),
                rocm_env_root: None,
                clang_sha256: String::new(),
                header_set_sha256: String::new(),
                build_dir: "/build".into(),
            },
            toolchain: ToolchainPin {
                pin: sha256_hex(id.as_bytes()),
                id,
                version_sha256: String::new(),
            },
            records: vec![CorpusRecord {
                arch: ARCH.into(),
                artifact_sha256: sha.clone(),
                module: "kt48".into(),
                origin: ObjectOrigin::Hipcc,
                symbols: vec![symbol.clone()],
                source_sha256: String::new(),
                extra_flags: String::new(),
                flags: Vec::new(),
                scheduler_profile: "default".into(),
                packaging_key: "key-now".into(),
                jit_cache_key: String::new(),
                argv: Vec::new(),
                object,
                receipt: receipt_rel,
                receipt_sha256: sha256_hex(&receipt_bytes),
                kernels: Vec::new(),
            }],
            rejected: Vec::new(),
            programs: vec![ProgramRecord {
                id: "p".into(),
                arch: ARCH.into(),
                nodes: vec![NodeRecord {
                    symbol: symbol.clone(),
                    module: "kt48".into(),
                    artifact_sha256: sha.clone(),
                }],
            }],
        };
        let fixture = Self {
            root,
            sha,
            symbol,
            index,
        };
        fixture.write_index();
        fixture
    }

    fn write_index(&self) {
        std::fs::write(
            self.root.join(CORPUS_INDEX),
            serde_json::to_vec_pretty(&self.index).unwrap(),
        )
        .unwrap();
    }

    fn program(&self, key: &str) -> Vec<ProgramKeys> {
        vec![ProgramKeys {
            id: "p".into(),
            arch: ARCH.into(),
            nodes: vec![(
                self.symbol.clone(),
                "kt48".into(),
                NodeKey::Hipcc(key.into()),
            )],
            loaded_sha256: None,
        }]
    }

    fn native_program(&self) -> Vec<ProgramKeys> {
        let key = NodeKey::Native {
            artifact_sha256: self.sha.clone(),
        };
        vec![ProgramKeys {
            id: "p".into(),
            arch: ARCH.into(),
            nodes: vec![(self.symbol.clone(), "kt48".into(), key)],
            loaded_sha256: None,
        }]
    }

    fn key<'a>(&'a self, pin: &'a str) -> ArtifactKey<'a> {
        ArtifactKey {
            arch: ARCH,
            toolchain_pin: pin,
            artifact_sha256: &self.sha,
        }
    }
}

#[test]
fn complete_corpus_covers_the_program_and_lookup_returns_the_verified_receipt() {
    let f = Fixture::new("complete");
    let corpus = JitCorpus::open(&f.root).unwrap();
    let coverage = check(&corpus, &f.program("key-now"));
    assert_eq!(
        (
            coverage[0].nodes,
            coverage[0].covered,
            coverage[0].missing.len()
        ),
        (1, 1, 0),
        "{coverage:?}"
    );
    let pin = corpus.toolchain_pin().to_owned();
    match corpus.lookup(&f.key(&pin)).unwrap() {
        Lookup::Hit(hit) => assert_eq!(
            (
                hit.receipt.object_sha256.as_str(),
                hit.record.module.as_str()
            ),
            (f.sha.as_str(), "kt48")
        ),
        Lookup::Missing => panic!("recorded artifact missed"),
    }
    // Another toolchain's bytes, or bytes never certified, are a miss.
    assert!(matches!(
        corpus.lookup(&f.key(&"0".repeat(64))).unwrap(),
        Lookup::Missing
    ));
    let other = ArtifactKey {
        arch: ARCH,
        toolchain_pin: &pin,
        artifact_sha256: &"1".repeat(64),
    };
    assert!(matches!(corpus.lookup(&other).unwrap(), Lookup::Missing));
}

#[test]
fn deliberately_removed_receipt_fails_the_check_loudly_naming_the_node() {
    let f = Fixture::new("removed-receipt");
    std::fs::remove_file(f.root.join(&f.index.records[0].receipt)).unwrap();
    let corpus = JitCorpus::open(&f.root).unwrap();
    let coverage = check(&corpus, &f.program("key-now"));
    assert_eq!(coverage[0].covered, 0);
    assert_eq!(coverage[0].missing[0].symbol, f.symbol);
    let node = ProgramNode {
        symbol: &f.symbol,
        artifact_sha256: &f.sha,
    };
    let error = corpus
        .check_program("p", ARCH, corpus.toolchain_pin(), &[node])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("receipt_missing=[") && error.contains(&f.symbol),
        "{error}"
    );
}

#[test]
fn deliberately_removed_index_record_refuses_the_corpus() {
    let mut f = Fixture::new("removed-record");
    f.index.records.clear();
    f.write_index();
    let error = JitCorpus::open(&f.root).unwrap_err().to_string();
    assert!(
        error.contains("program p node") && error.contains(&f.symbol),
        "{error}"
    );
}

#[test]
fn altered_receipt_is_an_error_not_a_miss() {
    let f = Fixture::new("altered");
    let path = f.root.join(&f.index.records[0].receipt);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&path, bytes).unwrap();
    let corpus = JitCorpus::open(&f.root).unwrap();
    let error = corpus
        .lookup(&f.key(corpus.toolchain_pin()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("SHA-256"), "{error}");
}

#[test]
fn node_whose_jit_inputs_changed_since_the_build_has_no_record() {
    let f = Fixture::new("stale");
    let coverage = check(
        &JitCorpus::open(&f.root).unwrap(),
        &f.program("key-after-a-source-edit"),
    );
    assert!(
        coverage[0].missing[0]
            .reason
            .contains("current key key-after-a-source-edit"),
        "{:?}",
        coverage[0].missing
    );
}

#[test]
fn object_the_certifier_rejected_is_missing_with_the_reason() {
    let mut f = Fixture::new("rejected");
    let record = f.index.records.remove(0);
    f.index.rejected.push(RejectedObject {
        arch: ARCH.into(),
        artifact_sha256: record.artifact_sha256,
        module: record.module,
        origin: ObjectOrigin::Hipcc,
        symbols: record.symbols,
        source_sha256: String::new(),
        packaging_key: record.packaging_key,
        object: record.object,
        reason: "lift: Decode at 0x10".into(),
    });
    f.write_index();
    let corpus = JitCorpus::open(&f.root).unwrap();
    assert!(matches!(
        corpus.lookup(&f.key(corpus.toolchain_pin())).unwrap(),
        Lookup::Missing
    ));
    let coverage = check(&corpus, &f.program("key-now"));
    assert!(
        coverage[0].missing[0]
            .reason
            .contains("rejected kt48: lift: Decode at 0x10"),
        "{:?}",
        coverage[0].missing
    );
}

#[test]
fn pm_native_record_covers_a_native_keyed_node() {
    let mut f = Fixture::new("native-covered");
    f.index.records[0].origin = ObjectOrigin::PmNative;
    f.write_index();
    let corpus = JitCorpus::open(&f.root).unwrap();
    let coverage = check(&corpus, &f.native_program());
    assert_eq!(
        (
            coverage[0].nodes,
            coverage[0].covered,
            coverage[0].missing.len()
        ),
        (1, 1, 0),
        "{coverage:?}"
    );
}

#[test]
fn native_key_whose_artifact_is_only_recorded_as_hipcc_is_missing() {
    let f = Fixture::new("native-wrong-origin");
    let coverage = check(&JitCorpus::open(&f.root).unwrap(), &f.native_program());
    assert_eq!(coverage[0].covered, 0);
    assert!(
        coverage[0].missing[0]
            .reason
            .contains("no pm_native record for twin kt48 image"),
        "{:?}",
        coverage[0].missing
    );
}

#[test]
fn native_key_whose_twin_the_certifier_rejected_is_missing_with_the_reason() {
    let mut f = Fixture::new("native-rejected");
    let record = f.index.records.remove(0);
    f.index.rejected.push(RejectedObject {
        arch: ARCH.into(),
        artifact_sha256: record.artifact_sha256,
        module: record.module,
        origin: ObjectOrigin::PmNative,
        symbols: record.symbols,
        source_sha256: String::new(),
        packaging_key: String::new(),
        object: record.object,
        reason: "radiowave: differential fail".into(),
    });
    f.write_index();
    let coverage = check(&JitCorpus::open(&f.root).unwrap(), &f.native_program());
    assert!(
        coverage[0].missing[0]
            .reason
            .contains("rejected pm_native twin kt48: radiowave: differential fail"),
        "{:?}",
        coverage[0].missing
    );
}

#[test]
fn record_for_module_ignores_pm_native_records() {
    let mut f = Fixture::new("module-lookup-origin");
    let corpus = JitCorpus::open(&f.root).unwrap();
    assert!(corpus.record_for_module(ARCH, "kt48", "key-now").is_some());
    f.index.records[0].origin = ObjectOrigin::PmNative;
    f.write_index();
    let corpus = JitCorpus::open(&f.root).unwrap();
    assert_eq!(corpus.index().records[0].packaging_key, "key-now");
    assert!(corpus.record_for_module(ARCH, "kt48", "key-now").is_none());
}

fn with_loaded(mut programs: Vec<ProgramKeys>, sha: &str) -> Vec<ProgramKeys> {
    programs[0].loaded_sha256 = Some(vec![sha.to_owned()]);
    programs
}

#[test]
fn recorded_sequence_node_resolves_when_the_record_is_for_the_loaded_object() {
    let f = Fixture::new("sequence-exact");
    let corpus = JitCorpus::open(&f.root).unwrap();
    let coverage = check(&corpus, &with_loaded(f.program("key-now"), &f.sha));
    assert_eq!(
        (coverage[0].covered, coverage[0].missing.len()),
        (1, 0),
        "{coverage:?}"
    );
}

#[test]
fn recorded_sequence_node_whose_loaded_object_differs_from_the_record_is_missing_loudly() {
    let f = Fixture::new("sequence-other-bytes");
    let corpus = JitCorpus::open(&f.root).unwrap();
    let other = "7".repeat(64);
    let coverage = check(&corpus, &with_loaded(f.program("key-now"), &other));
    assert_eq!(coverage[0].covered, 0);
    let missing = &coverage[0].missing[0];
    assert_eq!(missing.symbol, f.symbol);
    assert!(
        missing.reason.contains(&other) && missing.reason.contains(&f.sha),
        "{missing:?}"
    );
    // A twin-keyed node is held to the same rule.
    let mut native = f.native_program();
    native[0].loaded_sha256 = Some(vec![other.clone()]);
    let mut f2 = Fixture::new("sequence-other-bytes-native");
    f2.index.records[0].origin = ObjectOrigin::PmNative;
    f2.write_index();
    let coverage = check(&JitCorpus::open(&f2.root).unwrap(), &native);
    assert_eq!(coverage[0].covered, 0, "{coverage:?}");
    assert!(coverage[0].missing[0].reason.contains(&other));
}

#[test]
fn recorded_sequence_with_a_digest_count_mismatch_is_an_error() {
    let f = Fixture::new("sequence-count");
    let mut p = f.program("key-now");
    p[0].loaded_sha256 = Some(vec![f.sha.clone(), f.sha.clone()]);
    let coverage = check(&JitCorpus::open(&f.root).unwrap(), &p);
    assert!(
        coverage[0]
            .missing
            .iter()
            .any(|m| m.reason.contains("1 nodes but 2 recorded object digests")),
        "{coverage:?}"
    );
}

#[test]
fn conditional_node_is_reported_not_an_error_and_unknown_is_listed() {
    for (status, proven, unknown, conditional) in [
        ("proven", 1, 0, 0),
        ("conditional", 0, 0, 1),
        ("unknown", 0, 1, 0),
    ] {
        let mut f = Fixture::new(&format!("verdict-{status}"));
        f.index.records[0].kernels = vec![railgun_corpus::KernelVerdict {
            symbol: f.symbol.clone(),
            status: status.into(),
            differential: "pass".into(),
        }];
        f.write_index();
        let corpus = JitCorpus::open(&f.root).unwrap();
        let c = &check(&corpus, &f.program("key-now"))[0];
        assert_eq!(
            (
                c.covered,
                c.missing.len(),
                c.proven,
                c.unknown,
                c.conditional
            ),
            (1, 0, proven, unknown, conditional),
            "{status}: {c:?}"
        );
        assert_eq!(c.unproven.len(), usize::from(status != "proven"));
        if status != "proven" {
            assert_eq!(
                (c.unproven[0].symbol.as_str(), c.unproven[0].status.as_str()),
                (f.symbol.as_str(), status)
            );
        }
    }
}
