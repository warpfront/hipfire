//! railgun-corpus: the runtime's read-only view of railgun fact records
//! (railgun design §2.5, D1, D12).
//!
//! `railgun-jit-corpus` builds a corpus offline: every runtime-JIT module an
//! admitted default program loads, compiled with the pinned toolchain on one
//! build host and certified by `railgun-cert`. The runtime never links
//! peacemaker; it opens the corpus index through this crate and looks a node's
//! artifact up by `(arch, toolchain pin, artifact_sha256)`. A hit returns the
//! verified receipt; anything else is [`Lookup::Missing`], and a node of an
//! admitted default program without a record is a loud refusal
//! ([`JitCorpus::check_program`]), never a silent fallback.
//!
//! hipcc output is not reproducible across hosts or even across the runtime's
//! own compiles (the `__hip_cuid_*` marker is named after the input path and
//! argv), so a corpus is keyed per build host: its records bind the exact
//! object bytes built there, which ship in the corpus pack.
pub mod receipt;

use std::collections::{BTreeMap, HashMap};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use receipt::*;

pub const CORPUS_SCHEMA: &str = "railgun-jit-corpus";
pub const CORPUS_VERSION: u32 = 1;
/// The index file at the root of a corpus directory.
pub const CORPUS_INDEX: &str = "index.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusIndex {
    pub schema: String,
    pub version: u32,
    /// Git revision whose generated sources and recipes were compiled.
    pub source_commit: String,
    pub build_host: BuildHost,
    pub toolchain: ToolchainPin,
    pub records: Vec<CorpusRecord>,
    /// Objects the certifier rejected: shipped, but they have no fact record.
    pub rejected: Vec<RejectedObject>,
    pub programs: Vec<ProgramRecord>,
}

/// The host a corpus was built on. Object bytes depend on it (resolved
/// headers, compiler binaries, input paths), so records are only claimed for
/// objects built here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildHost {
    /// SHA-256 over the other identity fields; names the corpus directory.
    pub key: String,
    pub hostname: String,
    pub hipcc: String,
    pub rocm_env_root: Option<String>,
    pub clang_sha256: String,
    /// SHA-256 over `path\tsha256` of every header a probe preprocess read.
    pub header_set_sha256: String,
    /// Fixed directory the sources were compiled from (part of the cuid input).
    pub build_dir: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolchainPin {
    /// SHA-256 of `id`: the runtime pack index `toolchain_sha256`.
    pub pin: String,
    /// First line of `hipcc --version` (`KernelCompiler::toolchain_id`).
    pub id: String,
    /// SHA-256 of the complete `hipcc --version` output.
    pub version_sha256: String,
}

/// One certified object: the runtime's JIT cache-key inputs for the module,
/// the object they produced on the build host, and its receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusRecord {
    pub arch: String,
    pub artifact_sha256: String,
    pub module: String,
    pub symbols: Vec<String>,
    pub source_sha256: String,
    pub extra_flags: String,
    /// `KernelCompiler::recipe_for_source` flags.
    pub flags: Vec<String>,
    pub scheduler_profile: String,
    /// `KernelCompiler::packaging_hash`: the toolchain-free recipe key.
    pub packaging_key: String,
    /// `KernelCompiler::jit_cache_key`: the hot-cache key under this toolchain.
    pub jit_cache_key: String,
    pub argv: Vec<String>,
    /// Object path relative to the corpus root (`pack/<arch>/<module>.hsaco`).
    pub object: String,
    /// Receipt path relative to the corpus root.
    pub receipt: String,
    pub receipt_sha256: String,
    pub kernels: Vec<KernelVerdict>,
}

/// A built object `railgun-cert` could not certify (e.g. the lift rejected
/// an instruction). Lookups of its artifact are [`Lookup::Missing`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedObject {
    pub arch: String,
    pub artifact_sha256: String,
    pub module: String,
    pub symbols: Vec<String>,
    pub source_sha256: String,
    pub packaging_key: String,
    pub object: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelVerdict {
    pub symbol: String,
    /// A1 status: `proven` or `unknown`.
    pub status: String,
    /// G8 differential: `pass`, `fail` or `not_applicable`.
    pub differential: String,
}

/// An admitted default program as built: every node resolved to its record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramRecord {
    pub id: String,
    pub arch: String,
    pub nodes: Vec<NodeRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeRecord {
    pub symbol: String,
    pub module: String,
    pub artifact_sha256: String,
}

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("{}: {source}", path.display())]
    Io { path: PathBuf, source: std::io::Error },
    #[error("{}: {source}", path.display())]
    Json { path: PathBuf, source: serde_json::Error },
    #[error("{}: {detail}", path.display())]
    Invalid { path: PathBuf, detail: String },
}

/// The lookup key: an artifact is only ever matched on its own arch and
/// under the toolchain pin the corpus was built with.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactKey<'a> {
    pub arch: &'a str,
    pub toolchain_pin: &'a str,
    pub artifact_sha256: &'a str,
}

#[derive(Debug)]
pub enum Lookup {
    Hit(Box<Hit>),
    Missing,
}

#[derive(Debug)]
pub struct Hit {
    pub record: CorpusRecord,
    pub receipt: Receipt,
}

/// One launch of a program: the symbol it runs and the SHA-256 of the object
/// the runtime loaded it from.
#[derive(Clone, Copy, Debug)]
pub struct ProgramNode<'a> {
    pub symbol: &'a str,
    pub artifact_sha256: &'a str,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingNode {
    pub symbol: String,
    pub artifact_sha256: String,
    pub reason: String,
}

/// A node of an admitted default program has no usable fact record: the
/// railgun default is refused for the process (design §2.3, G5
/// `receipt_missing`).
#[derive(Debug, thiserror::Error)]
#[error("railgun JIT corpus: {} node(s) of admitted default program {program} ({arch}) have no fact record: receipt_missing=[{}]",
    missing.len(), missing.iter().map(|m| format!("{}@{} ({})", m.symbol, short(&m.artifact_sha256), m.reason)).collect::<Vec<_>>().join(", "))]
pub struct ProgramCheckError {
    pub program: String,
    pub arch: String,
    pub missing: Vec<MissingNode>,
}

fn short(sha: &str) -> &str {
    sha.get(..16).unwrap_or(sha)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A path inside the corpus: relative, no `..`, no root.
fn contained(rel: &str) -> bool {
    !rel.is_empty() && Path::new(rel).components().all(|c| matches!(c, Component::Normal(_)))
}

/// An opened corpus directory. Opening validates the whole index; receipts
/// are read and verified on lookup.
#[derive(Debug)]
pub struct JitCorpus {
    root: PathBuf,
    index: CorpusIndex,
    by_artifact: HashMap<(String, String), usize>,
}

impl JitCorpus {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, CorpusError> {
        let root = root.as_ref().to_path_buf();
        let path = root.join(CORPUS_INDEX);
        let bytes = std::fs::read(&path).map_err(|source| CorpusError::Io { path: path.clone(), source })?;
        let index: CorpusIndex = serde_json::from_slice(&bytes).map_err(|source| CorpusError::Json { path: path.clone(), source })?;
        let invalid = |detail: String| CorpusError::Invalid { path: path.clone(), detail };
        if index.schema != CORPUS_SCHEMA || index.version != CORPUS_VERSION {
            return Err(invalid(format!("schema {} v{} is not {CORPUS_SCHEMA} v{CORPUS_VERSION}", index.schema, index.version)));
        }
        if !is_sha256(&index.toolchain.pin) || index.toolchain.pin != sha256_hex(index.toolchain.id.as_bytes()) {
            return Err(invalid("toolchain pin is not the SHA-256 of the toolchain id".into()));
        }
        let mut by_artifact = HashMap::new();
        for (i, r) in index.records.iter().enumerate() {
            if !is_sha256(&r.artifact_sha256) || !is_sha256(&r.receipt_sha256) {
                return Err(invalid(format!("record {} {}: malformed digest", r.arch, r.module)));
            }
            if !contained(&r.object) || !contained(&r.receipt) {
                return Err(invalid(format!("record {} {}: path escapes the corpus", r.arch, r.module)));
            }
            // Two registry modules with the same source compile to the same
            // object (beta's `gemv_mq4g256` / `mq_rotate_x`). One object has one
            // receipt, so such rows must agree on it; the first row is the
            // artifact's record.
            match by_artifact.entry((r.arch.clone(), r.artifact_sha256.clone())) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(i);
                }
                std::collections::hash_map::Entry::Occupied(o) => {
                    if index.records[*o.get()].receipt_sha256 != r.receipt_sha256 {
                        return Err(invalid(format!("duplicate record {} {} with a different receipt", r.arch, r.artifact_sha256)));
                    }
                }
            }
        }
        for r in &index.rejected {
            if !is_sha256(&r.artifact_sha256) || !contained(&r.object) || by_artifact.contains_key(&(r.arch.clone(), r.artifact_sha256.clone())) {
                return Err(invalid(format!("rejected object {} {} is malformed or also a record", r.arch, r.module)));
            }
        }
        for program in &index.programs {
            for node in &program.nodes {
                let exports = match by_artifact.get(&(program.arch.clone(), node.artifact_sha256.clone())) {
                    Some(&i) => &index.records[i].symbols,
                    None => match index.rejected.iter().find(|r| r.arch == program.arch && r.artifact_sha256 == node.artifact_sha256) {
                        Some(r) => &r.symbols,
                        None => return Err(invalid(format!("program {} node {} names artifact {} with no record",
                            program.id, node.symbol, node.artifact_sha256))),
                    },
                };
                if !exports.iter().any(|s| s == &node.symbol) {
                    return Err(invalid(format!("program {} node {}: its object does not export it", program.id, node.symbol)));
                }
            }
        }
        Ok(Self { root, index, by_artifact })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index(&self) -> &CorpusIndex {
        &self.index
    }

    pub fn toolchain_pin(&self) -> &str {
        &self.index.toolchain.pin
    }

    /// The index row for `key`, without touching the receipt.
    pub fn record(&self, key: &ArtifactKey) -> Option<&CorpusRecord> {
        if key.toolchain_pin != self.index.toolchain.pin {
            return None;
        }
        self.by_artifact.get(&(key.arch.to_owned(), key.artifact_sha256.to_owned())).map(|&i| &self.index.records[i])
    }

    /// Why `(arch, artifact_sha256)` has no record, when the certifier
    /// rejected that object.
    pub fn rejection(&self, arch: &str, artifact_sha256: &str) -> Option<&RejectedObject> {
        self.index.rejected.iter().find(|r| r.arch == arch && r.artifact_sha256 == artifact_sha256)
    }

    /// The first record for `(arch, module, packaging_key)`: the runtime's
    /// toolchain-free JIT cache key for the module's generated source.
    pub fn record_for_module(&self, arch: &str, module: &str, packaging_key: &str) -> Option<&CorpusRecord> {
        self.index.records.iter().find(|r| r.arch == arch && r.module == module && r.packaging_key == packaging_key)
    }

    /// Look an artifact up and return its verified receipt. A receipt that is
    /// absent, altered, or names other bytes is an error, not a miss.
    pub fn lookup(&self, key: &ArtifactKey) -> Result<Lookup, CorpusError> {
        let Some(record) = self.record(key) else { return Ok(Lookup::Missing) };
        let path = self.root.join(&record.receipt);
        let bytes = std::fs::read(&path).map_err(|source| CorpusError::Io { path: path.clone(), source })?;
        let invalid = |detail: String| CorpusError::Invalid { path: path.clone(), detail };
        if sha256_hex(&bytes) != record.receipt_sha256 {
            return Err(invalid("receipt SHA-256 does not match the corpus index".into()));
        }
        let receipt: Receipt = serde_json::from_slice(&bytes).map_err(|source| CorpusError::Json { path: path.clone(), source })?;
        if receipt.schema != RECEIPT_SCHEMA || receipt.schema_version != RECEIPT_SCHEMA_VERSION
            || receipt.railgun.section_version != RAILGUN_SECTION_VERSION
        {
            return Err(invalid(format!("receipt schema {} v{} section v{} is not {RECEIPT_SCHEMA} v{RECEIPT_SCHEMA_VERSION} section v{RAILGUN_SECTION_VERSION}",
                receipt.schema, receipt.schema_version, receipt.railgun.section_version)));
        }
        if receipt.object_sha256 != key.artifact_sha256 || receipt.arch != key.arch {
            return Err(invalid(format!("receipt certifies {} {}, not {} {}", receipt.arch, receipt.object_sha256, key.arch, key.artifact_sha256)));
        }
        Ok(Lookup::Hit(Box::new(Hit { record: record.clone(), receipt })))
    }

    /// Every node of `program` must resolve to a verified receipt that
    /// carries a record for the node's symbol.
    pub fn check_program(&self, program: &str, arch: &str, toolchain_pin: &str, nodes: &[ProgramNode]) -> Result<(), ProgramCheckError> {
        let mut receipts: BTreeMap<&str, Result<Receipt, String>> = BTreeMap::new();
        let mut missing = Vec::new();
        for node in nodes {
            let receipt = receipts.entry(node.artifact_sha256).or_insert_with(|| {
                let key = ArtifactKey { arch, toolchain_pin, artifact_sha256: node.artifact_sha256 };
                match self.lookup(&key) {
                    Ok(Lookup::Hit(hit)) => Ok(hit.receipt),
                    Ok(Lookup::Missing) if toolchain_pin != self.toolchain_pin() => {
                        Err(format!("corpus toolchain pin {} is not {}", short(self.toolchain_pin()), short(toolchain_pin)))
                    }
                    Ok(Lookup::Missing) => Err(match self.rejection(arch, node.artifact_sha256) {
                        Some(r) => format!("railgun-cert rejected the object: {}", r.reason),
                        None => "no record".into(),
                    }),
                    Err(e) => Err(e.to_string()),
                }
            });
            let reason = match receipt {
                Ok(r) if r.railgun.kernels.iter().any(|k| k.symbol == node.symbol) => continue,
                Ok(_) => format!("receipt has no kernel {}", node.symbol),
                Err(e) => e.clone(),
            };
            missing.push(MissingNode { symbol: node.symbol.to_owned(), artifact_sha256: node.artifact_sha256.to_owned(), reason });
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(ProgramCheckError { program: program.to_owned(), arch: arch.to_owned(), missing })
        }
    }
}
