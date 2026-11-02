// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WorkloadCaseV1`: one real model workload (model, prompt, arch, phase,
//! execution mode, ordered kernel chain with launch geometry and object
//! identity, state requirements, and the denominator it is measured against)
//! as a content-addressed record, and `WorkloadCorpusV1`, an order-independent
//! set of them (PM/railgun bedrock R2).
//!
//! This module is layer 0: it names kernel ABI contracts only by
//! [`AbiRefV1`] (id, version, digest strings) and depends on no kernel-ABI or
//! peacemaker crate. Extraction from real routes lives in the `pm-census`
//! developer tool.
//!
//! Identity rules (frozen):
//!
//! * `case_id` is the lowercase hex SHA-256 of `serde_json::to_vec` of the
//!   case with `case_id` set to `""`.
//! * The corpus digest is the SHA-256 of the sorted `case_id`s joined by
//!   `\n`.
//! * A JSONL corpus is a header line `{schema, digest, cases}` followed by one
//!   case per line.
//!
//! No identity field is ever filled with a guess: unknown values are `None`
//! or, for [`ObjectIdentityV1::join`], an `unresolved: <reason>` string.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const WORKLOAD_CASE_SCHEMA: &str = "hipfire.workload_case.v1";
pub const WORKLOAD_CORPUS_SCHEMA: &str = "hipfire.workload_corpus.v1";

/// The model a case ran.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRefV1 {
    pub name: String,
    pub family: String,
    pub weight_format: Option<String>,
    pub file_sha256: Option<String>,
    pub file_bytes: Option<u64>,
    pub sidecar_sha256: Option<String>,
}

/// The prompt a case ran (identity only; the text is never stored).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptRefV1 {
    pub sha256: Option<String>,
    pub tokens: Option<u64>,
    pub token_ids_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseV1 {
    Prefill,
    Ar,
    Spec,
}

/// Row geometry of the workload; every field is `None`/empty when the source
/// does not state it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowsV1 {
    pub tokens: Option<u64>,
    pub context: Option<u64>,
    pub chunk: Option<u64>,
    pub ragged: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecModeV1 {
    Eager,
    HipGraph,
    RedlinePm4,
    Railgun,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementV1 {
    Resident,
    HostMapped,
    Mixed,
    Unknown,
}

/// How a node's `params` were obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamsBasisV1 {
    /// Decoded from the launch's kernarg bytes through a kernel ABI contract.
    KernargDecode,
    /// Inverted exactly from the launch grid through a contract's grid
    /// expression.
    GridBound,
    /// Taken from the model's geometry, not from the launch.
    ModelGeometry,
    /// `params` are not launch-derived (empty, or recorded verbatim from a
    /// source that does not state how they relate to the kernarg).
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginV1 {
    /// A PeaceMaker-emitted object.
    Pm,
    /// Compiled from HIP source by hipcc.
    Hipcc,
    Unknown,
}

/// Identity of the code object a node launched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectIdentityV1 {
    pub origin: OriginV1,
    pub module: Option<String>,
    pub artifact: Option<String>,
    pub artifact_sha256: Option<String>,
    /// How the identity above was established, or `unresolved: <reason>`.
    pub join: String,
}

/// Reference to a kernel ABI contract (`hipfire_kernel_abi::KernelAbiV1`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiRefV1 {
    pub id: String,
    pub version: u32,
    pub digest: String,
}

/// One distinct launch of the chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainNodeV1 {
    pub symbol: String,
    pub object: ObjectIdentityV1,
    pub abi: Option<AbiRefV1>,
    pub params: BTreeMap<String, u64>,
    pub params_basis: ParamsBasisV1,
    /// Grid in workgroups.
    pub grid: Option<[u32; 3]>,
    /// Workgroup size in threads.
    pub block: Option<[u32; 3]>,
    pub dynamic_lds: Option<u32>,
    pub kernarg_sha256: Option<String>,
    /// Launches of this node in the ROI (0 = not known).
    pub calls: u64,
    pub time_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureV1 {
    pub name: String,
    /// `synthetic` or `captured`.
    pub kind: String,
    pub path: Option<String>,
    pub sha256: Option<String>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateReqV1 {
    pub rollback: bool,
    pub prefix_commit: bool,
    pub graph_replay: bool,
    pub notes: Vec<String>,
}

/// What the chain's time is a fraction of.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DenominatorV1 {
    Measured {
        wall_ns: u64,
        roi: String,
        source: String,
    },
    Unmeasured {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceV1 {
    pub source_kind: String,
    pub source_path: String,
    pub source_sha256: String,
    pub extracted_by: String,
    pub commit: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadCaseV1 {
    pub schema: String,
    pub case_id: String,
    pub model: ModelRefV1,
    pub prompt: Option<PromptRefV1>,
    pub arch: String,
    pub device: Option<String>,
    pub config: BTreeMap<String, String>,
    pub phase: PhaseV1,
    pub rows: RowsV1,
    pub mode: ExecModeV1,
    pub placement: PlacementV1,
    pub chain: Vec<ChainNodeV1>,
    pub fixtures: Vec<FixtureV1>,
    pub state: StateReqV1,
    pub denominator: DenominatorV1,
    pub provenance: ProvenanceV1,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl WorkloadCaseV1 {
    /// Lowercase hex SHA-256 of the case serialized with `case_id` = `""`.
    pub fn compute_id(&self) -> String {
        let mut c = self.clone();
        c.case_id = String::new();
        let bytes = serde_json::to_vec(&c).expect("WorkloadCaseV1 serializes");
        sha256_hex(&bytes)
    }

    /// Set `schema` and `case_id`.
    pub fn seal(mut self) -> Self {
        self.schema = WORKLOAD_CASE_SCHEMA.to_string();
        self.case_id = self.compute_id();
        self
    }

    /// The schema is this one and `case_id` is the id of the content.
    pub fn verify_id(&self) -> Result<(), String> {
        if self.schema != WORKLOAD_CASE_SCHEMA {
            return Err(format!(
                "case {}: schema {:?} is not {WORKLOAD_CASE_SCHEMA:?}",
                self.case_id, self.schema
            ));
        }
        let want = self.compute_id();
        if self.case_id != want {
            return Err(format!(
                "case_id {:?} does not match its content (computed {want})",
                self.case_id
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadCorpusV1 {
    pub schema: String,
    pub digest: String,
    pub cases: Vec<WorkloadCaseV1>,
}

fn corpus_digest(cases: &[WorkloadCaseV1]) -> String {
    let mut ids: Vec<&str> = cases.iter().map(|c| c.case_id.as_str()).collect();
    ids.sort_unstable();
    sha256_hex(ids.join("\n").as_bytes())
}

#[derive(Serialize, Deserialize)]
struct CorpusHeader {
    schema: String,
    digest: String,
    cases: u64,
}

impl WorkloadCorpusV1 {
    /// Verify every case, drop exact duplicates (same `case_id`), sort by
    /// `case_id`, and compute the digest. Input order never matters.
    pub fn new(mut cases: Vec<WorkloadCaseV1>) -> Result<Self, String> {
        for c in &cases {
            c.verify_id()?;
        }
        cases.sort_by(|a, b| a.case_id.cmp(&b.case_id));
        cases.dedup_by(|a, b| a.case_id == b.case_id);
        let digest = corpus_digest(&cases);
        Ok(Self {
            schema: WORKLOAD_CORPUS_SCHEMA.to_string(),
            digest,
            cases,
        })
    }

    /// Schema, every case id, uniqueness of ids and the digest.
    pub fn verify(&self) -> Result<(), String> {
        if self.schema != WORKLOAD_CORPUS_SCHEMA {
            return Err(format!(
                "corpus schema {:?} is not {WORKLOAD_CORPUS_SCHEMA:?}",
                self.schema
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for c in &self.cases {
            c.verify_id()?;
            if !seen.insert(c.case_id.as_str()) {
                return Err(format!("duplicate case_id {}", c.case_id));
            }
        }
        let want = corpus_digest(&self.cases);
        if self.digest != want {
            return Err(format!(
                "corpus digest {:?} does not match its cases (computed {want})",
                self.digest
            ));
        }
        Ok(())
    }

    /// Header line `{schema, digest, cases}`, then one case per line.
    pub fn write_jsonl(&self, path: &Path) -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        let header = CorpusHeader {
            schema: self.schema.clone(),
            digest: self.digest.clone(),
            cases: self.cases.len() as u64,
        };
        let line = |v: Result<String, serde_json::Error>| {
            v.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        };
        w.write_all(line(serde_json::to_string(&header))?.as_bytes())?;
        w.write_all(b"\n")?;
        for c in &self.cases {
            w.write_all(line(serde_json::to_string(c))?.as_bytes())?;
            w.write_all(b"\n")?;
        }
        w.flush()
    }

    /// Read and [`verify`](Self::verify) a JSONL corpus.
    pub fn read_jsonl(path: &Path) -> Result<Self, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lines = BufReader::new(f).lines();
        let first = lines
            .next()
            .ok_or_else(|| format!("{}: empty corpus file", path.display()))?
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let header: CorpusHeader = serde_json::from_str(&first)
            .map_err(|e| format!("{}: header line: {e}", path.display()))?;
        let mut cases = Vec::new();
        for (i, l) in lines.enumerate() {
            let l = l.map_err(|e| format!("{}: {e}", path.display()))?;
            if l.trim().is_empty() {
                continue;
            }
            let c: WorkloadCaseV1 = serde_json::from_str(&l)
                .map_err(|e| format!("{}: case line {}: {e}", path.display(), i + 2))?;
            cases.push(c);
        }
        if cases.len() as u64 != header.cases {
            return Err(format!(
                "{}: header says {} cases, file has {}",
                path.display(),
                header.cases,
                cases.len()
            ));
        }
        let corpus = Self {
            schema: header.schema,
            digest: header.digest,
            cases,
        };
        corpus
            .verify()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(corpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(symbol: &str, calls: u64) -> ChainNodeV1 {
        ChainNodeV1 {
            symbol: symbol.into(),
            object: ObjectIdentityV1 {
                origin: OriginV1::Hipcc,
                module: Some("m".into()),
                artifact: Some("/a/m.hsaco".into()),
                artifact_sha256: Some("00".repeat(32)),
                join: "test".into(),
            },
            abi: Some(AbiRefV1 {
                id: "fam".into(),
                version: 1,
                digest: "11".repeat(32),
            }),
            params: BTreeMap::from([("m".to_string(), 4u64)]),
            params_basis: ParamsBasisV1::KernargDecode,
            grid: Some([4, 1, 1]),
            block: Some([32, 1, 1]),
            dynamic_lds: Some(0),
            kernarg_sha256: Some("22".repeat(32)),
            calls,
            time_ns: Some(10),
        }
    }

    fn case(tag: &str) -> WorkloadCaseV1 {
        WorkloadCaseV1 {
            schema: String::new(),
            case_id: String::new(),
            model: ModelRefV1 {
                name: format!("model-{tag}"),
                family: "fam".into(),
                weight_format: Some("mq6".into()),
                file_sha256: Some("33".repeat(32)),
                file_bytes: Some(123),
                sidecar_sha256: None,
            },
            prompt: Some(PromptRefV1 {
                sha256: Some("44".repeat(32)),
                tokens: Some(512),
                token_ids_sha256: None,
            }),
            arch: "gfx1201".into(),
            device: Some("card".into()),
            config: BTreeMap::from([("kv".to_string(), "fp8".to_string())]),
            phase: PhaseV1::Ar,
            rows: RowsV1 {
                tokens: Some(1),
                context: Some(512),
                chunk: None,
                ragged: vec![1, 2],
            },
            mode: ExecModeV1::RedlinePm4,
            placement: PlacementV1::Resident,
            chain: vec![node("k0", 2), node("k1", 1)],
            fixtures: vec![FixtureV1 {
                name: "f".into(),
                kind: "synthetic".into(),
                path: None,
                sha256: None,
                seed: Some(7),
            }],
            state: StateReqV1 {
                rollback: false,
                prefix_commit: true,
                graph_replay: false,
                notes: vec!["n".into()],
            },
            denominator: DenominatorV1::Measured {
                wall_ns: 1000,
                roi: "roi".into(),
                source: "src".into(),
            },
            provenance: ProvenanceV1 {
                source_kind: "test".into(),
                source_path: "/x".into(),
                source_sha256: "55".repeat(32),
                extracted_by: "test".into(),
                commit: None,
            },
        }
    }

    #[test]
    fn seal_and_verify_id_round_trip() {
        let c = case("a").seal();
        assert_eq!(c.schema, WORKLOAD_CASE_SCHEMA);
        assert_eq!(c.case_id.len(), 64);
        c.verify_id().unwrap();
        assert_eq!(c.case_id, c.compute_id());
        // Sealing is idempotent.
        assert_eq!(c.clone().seal(), c);
        // JSON round trip keeps the id valid.
        let back: WorkloadCaseV1 =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
        back.verify_id().unwrap();
        // An unsealed case does not verify.
        assert!(case("a").verify_id().is_err());
    }

    #[test]
    fn id_changes_on_any_field_change() {
        let base = case("a").seal();
        fn boxed<F: Fn(&mut WorkloadCaseV1) + 'static>(f: F) -> Box<dyn Fn(&mut WorkloadCaseV1)> {
            Box::new(f)
        }
        let mutations: Vec<(&str, Box<dyn Fn(&mut WorkloadCaseV1)>)> = vec![
            ("model.name", boxed(|c| c.model.name.push('x'))),
            (
                "model.file_bytes",
                boxed(|c| c.model.file_bytes = Some(124)),
            ),
            ("prompt", boxed(|c| c.prompt = None)),
            ("arch", boxed(|c| c.arch = "gfx1100".into())),
            ("device", boxed(|c| c.device = None)),
            (
                "config",
                boxed(|c| {
                    c.config.insert("k".into(), "v".into());
                }),
            ),
            ("phase", boxed(|c| c.phase = PhaseV1::Prefill)),
            ("rows.ragged", boxed(|c| c.rows.ragged.push(3))),
            ("mode", boxed(|c| c.mode = ExecModeV1::Railgun)),
            ("placement", boxed(|c| c.placement = PlacementV1::Mixed)),
            ("chain.calls", boxed(|c| c.chain[0].calls += 1)),
            (
                "chain.params",
                boxed(|c| {
                    c.chain[1].params.insert("n".into(), 1);
                }),
            ),
            (
                "chain.object.join",
                boxed(|c| c.chain[0].object.join.push('!')),
            ),
            ("chain.order", boxed(|c| c.chain.reverse())),
            (
                "chain.params_basis",
                boxed(|c| c.chain[0].params_basis = ParamsBasisV1::None),
            ),
            ("fixtures", boxed(|c| c.fixtures[0].seed = Some(8))),
            ("state", boxed(|c| c.state.rollback = true)),
            (
                "denominator",
                boxed(|c| c.denominator = DenominatorV1::Unmeasured { reason: "r".into() }),
            ),
            (
                "provenance.commit",
                boxed(|c| c.provenance.commit = Some("abc".into())),
            ),
        ];
        for (name, m) in mutations {
            let mut c = base.clone();
            m(&mut c);
            assert_ne!(c.compute_id(), base.case_id, "{name} did not change the id");
            assert!(c.verify_id().is_err(), "{name}: stale id still verifies");
        }
    }

    #[test]
    fn corpus_digest_is_order_independent_and_dedupes() {
        let (a, b, c) = (case("a").seal(), case("b").seal(), case("c").seal());
        let one = WorkloadCorpusV1::new(vec![a.clone(), b.clone(), c.clone()]).unwrap();
        let two = WorkloadCorpusV1::new(vec![c.clone(), a.clone(), b.clone(), a.clone()]).unwrap();
        assert_eq!(one.cases.len(), 3);
        assert_eq!(two.cases.len(), 3, "exact duplicate is dropped");
        assert_eq!(one.digest, two.digest);
        assert_eq!(one, two);
        one.verify().unwrap();
        let other = WorkloadCorpusV1::new(vec![a.clone(), b.clone()]).unwrap();
        assert_ne!(other.digest, one.digest);
        // Digest = SHA-256 of sorted ids joined by \n.
        let mut ids = vec![a.case_id, b.case_id, c.case_id];
        ids.sort();
        assert_eq!(one.digest, sha256_hex(ids.join("\n").as_bytes()));
        // Unsealed or tampered cases are refused.
        assert!(WorkloadCorpusV1::new(vec![case("x")]).is_err());
        let mut bad = one.clone();
        bad.digest = "0".repeat(64);
        assert!(bad.verify().is_err());
        let mut dup = one;
        dup.cases.push(dup.cases[0].clone());
        assert!(dup.verify().is_err());
    }

    #[test]
    fn jsonl_round_trip() {
        let corpus = WorkloadCorpusV1::new(vec![case("a").seal(), case("b").seal()]).unwrap();
        let path = std::env::temp_dir().join(format!(
            "hipfire-atlas-workload-{}-{}.jsonl",
            std::process::id(),
            corpus.digest
        ));
        corpus.write_jsonl(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(header["schema"], WORKLOAD_CORPUS_SCHEMA);
        assert_eq!(header["digest"], corpus.digest);
        assert_eq!(header["cases"], 2);
        assert_eq!(lines.count(), 2);
        let back = WorkloadCorpusV1::read_jsonl(&path).unwrap();
        assert_eq!(back, corpus);

        // A truncated file (header count mismatch) is refused.
        let first_two: String = text.lines().take(2).map(|l| format!("{l}\n")).collect();
        std::fs::write(&path, first_two).unwrap();
        assert!(WorkloadCorpusV1::read_jsonl(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn denominator_serde_tag_shape() {
        let m = DenominatorV1::Measured {
            wall_ns: 5,
            roi: "r".into(),
            source: "s".into(),
        };
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({"status": "measured", "wall_ns": 5, "roi": "r", "source": "s"})
        );
        let u = DenominatorV1::Unmeasured {
            reason: "why".into(),
        };
        assert_eq!(
            serde_json::to_value(&u).unwrap(),
            serde_json::json!({"status": "unmeasured", "reason": "why"})
        );
        let back: DenominatorV1 =
            serde_json::from_value(serde_json::json!({"status": "unmeasured", "reason": "why"}))
                .unwrap();
        assert_eq!(back, u);
        // Plain enums are snake_case.
        assert_eq!(
            serde_json::to_value(ExecModeV1::RedlinePm4).unwrap(),
            "redline_pm4"
        );
        assert_eq!(
            serde_json::to_value(ExecModeV1::HipGraph).unwrap(),
            "hip_graph"
        );
        assert_eq!(
            serde_json::to_value(ParamsBasisV1::KernargDecode).unwrap(),
            "kernarg_decode"
        );
        assert_eq!(
            serde_json::to_value(PlacementV1::HostMapped).unwrap(),
            "host_mapped"
        );
    }
}
