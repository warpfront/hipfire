// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Serialized identity records for the HIP / Vulkan executable union.
//!
//! This module is **provenance only**. It owns the JSON schema, the typed
//! identity domains and CPU-side consistency checks. It never loads a code
//! object, never talks to a driver, never links Mesa and never proves a
//! program correct: the PM static certificate, the actual loaded-code receipt
//! and the oracle bytes are produced elsewhere and *bound* here by digest.
//! A record that passes validation is internally complete and consistent; it
//! is not evidence that the described GPU run happened.
//!
//! # Separation from the compile manifest
//!
//! [`crate::CompileManifest`] is untouched. Union records live in their own
//! serde schema (`schema_version` = [`UNION_SCHEMA_VERSION`]), reject unknown
//! fields and reject any other version, so an existing manifest can neither be
//! mistaken for a union record nor change shape because of one.
//!
//! # Identity domains
//!
//! Every digest is a distinct Rust type ([`SourceDigest`], [`RawCodeDigest`],
//! [`HsaImageDigest`], [`RadvBlobDigest`], ...). A digest of one domain cannot
//! be passed where another is expected, which is the point: HSA ELF bytes are
//! not a Vulkan blob, a raw ACO body is not the adapted envelope, a Windows
//! driver key is not a RADV key, and a file hash is not a loaded executable.
//!
//! | Domain | Record |
//! |---|---|
//! | semantic recipe | [`SemanticRecipe`] |
//! | producer (source/SPIR-V/compiler/driver/options/Mesa pins) | [`ProducerRecord`] |
//! | arch / wave | [`UnionTarget`] |
//! | raw code/data/config/entry-ABI/sidecar | [`RawArtifact`] |
//! | static certificate | [`StaticCertificate`] |
//! | adapter prologue/body | [`AdapterRecord`] |
//! | execution envelope | [`EnvelopeRecord`] |
//! | loaded executable | [`LoadedExecutableReceipt`] |
//! | four-arm oracle pair | [`PairOracleReceipt`] |
//!
//! # Frozen interfaces
//!
//! * U-F1 [`CertifiedExecutableRecord`], [`LoadedExecutableReceipt`],
//!   [`PairOracleReceipt`].
//! * U-F2 [`VulkanExecutableRecord`].
//! * U-F3 [`SharedBufferViewRecord`], [`BoundaryContractRecord`].
//! * U-F4 [`UnionPlanRecord`].
//!
//! U-F3 and U-F4 are serialized descriptions only. They carry no ownership,
//! handles, pointers or runtime behavior; a runtime that implements them owns
//! the proof that its actual state matches the record.
//!
//! All records are immutable values. Dependencies between them are digests
//! (one-way: a loaded receipt or oracle receipt names the certified record, never
//! the reverse) so no shared mutable program graph is needed.

use crate::Wavefront;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use thiserror::Error;

/// Schema version of every record in this module.
pub const UNION_SCHEMA_VERSION: u32 = 1;

const PRIVATE_CACHE_KEY_DOMAIN: &str = "radiowave.union.private-cache-key.v1";
const RECORD_DIGEST_DOMAIN: &str = "radiowave.union.record.v1";
const STATE_SURFACE_DOMAIN: &str = "radiowave.union.state-surface.v1";
const PLAN_DIGEST_DOMAIN: &str = "radiowave.union.runtime-plan.v1";

#[derive(Debug, Error)]
pub enum UnionError {
    #[error("union record JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{kind} record has no schema_version")]
    MissingSchemaVersion { kind: &'static str },
    #[error("{kind} schema_version {found} is unsupported (this build reads {expected})")]
    UnsupportedSchemaVersion {
        kind: &'static str,
        found: String,
        expected: u32,
    },
    #[error("malformed {field}: {reason}")]
    Malformed { field: String, reason: String },
    #[error("incomplete evidence for {field}: {reason}")]
    Incomplete { field: String, reason: String },
    #[error("{what} mismatch: expected {expected}, found {actual}")]
    Mismatch {
        what: String,
        expected: String,
        actual: String,
    },
    #[error("envelope incompatible: {0}")]
    EnvelopeIncompatible(String),
    #[error("pair oracle binding failed: {0}")]
    PairBinding(String),
}

pub type UnionResult<T> = std::result::Result<T, UnionError>;

fn malformed(field: &str, reason: impl Into<String>) -> UnionError {
    UnionError::Malformed {
        field: field.to_owned(),
        reason: reason.into(),
    }
}

fn incomplete(field: &str, reason: impl Into<String>) -> UnionError {
    UnionError::Incomplete {
        field: field.to_owned(),
        reason: reason.into(),
    }
}

fn mismatch(what: &str, expected: &str, actual: &str) -> UnionError {
    UnionError::Mismatch {
        what: what.to_owned(),
        expected: expected.to_owned(),
        actual: actual.to_owned(),
    }
}

fn pair_binding(reason: impl Into<String>) -> UnionError {
    UnionError::PairBinding(reason.into())
}

fn ensure_version(kind: &'static str, found: u32) -> UnionResult<()> {
    if found == UNION_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(UnionError::UnsupportedSchemaVersion {
            kind,
            found: found.to_string(),
            expected: UNION_SCHEMA_VERSION,
        })
    }
}

fn ensure_text(field: &str, value: &str) -> UnionResult<()> {
    if value.trim().is_empty() {
        return Err(incomplete(field, "must not be empty"));
    }
    if value.chars().any(char::is_control) {
        return Err(malformed(field, "must not contain control characters"));
    }
    Ok(())
}

fn ensure_equal(what: &str, expected: &str, actual: &str) -> UnionResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(mismatch(what, expected, actual))
    }
}

fn checked_end(field: &str, offset: u64, length: u64) -> UnionResult<u64> {
    offset
        .checked_add(length)
        .ok_or_else(|| malformed(field, "offset + length overflows u64"))
}

fn ranges_overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 over a domain tag followed by length-prefixed parts, so two
/// identities of different domains can never collide by concatenation.
fn domain_sha256_hex(domain: &str, parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0u8]);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

fn canonical_json<T: Serialize + ?Sized>(value: &T) -> UnionResult<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn slice_of<'a>(raw: &'a [u8], field: &str, offset: u64, length: u64) -> UnionResult<&'a [u8]> {
    let start =
        usize::try_from(offset).map_err(|_| malformed(field, "offset does not fit usize"))?;
    let end = usize::try_from(checked_end(field, offset, length)?)
        .map_err(|_| malformed(field, "end does not fit usize"))?;
    raw.get(start..end)
        .ok_or_else(|| malformed(field, "range lies outside the raw stream"))
}

fn parse_versioned<T: serde::de::DeserializeOwned>(
    kind: &'static str,
    json: &str,
) -> UnionResult<T> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let found = value
        .get("schema_version")
        .ok_or(UnionError::MissingSchemaVersion { kind })?;
    if found.as_u64() != Some(u64::from(UNION_SCHEMA_VERSION)) {
        return Err(UnionError::UnsupportedSchemaVersion {
            kind,
            found: found.to_string(),
            expected: UNION_SCHEMA_VERSION,
        });
    }
    Ok(serde_json::from_value(value)?)
}

macro_rules! versioned_json {
    ($ty:ident, $kind:literal) => {
        impl $ty {
            /// Parse, reject unknown fields and any other schema version, then
            /// run [`Self::validate`].
            pub fn from_json(json: &str) -> UnionResult<Self> {
                let record: Self = parse_versioned($kind, json)?;
                record.validate()?;
                Ok(record)
            }

            /// Validate, then render stable pretty JSON.
            pub fn to_json(&self) -> UnionResult<String> {
                self.validate()?;
                Ok(serde_json::to_string_pretty(self)?)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Typed hex identities
// ---------------------------------------------------------------------------

/// A fixed-width lowercase-hex identity belonging to exactly one domain.
pub trait HexIdentity: Sized {
    /// Required number of lowercase hex characters.
    const HEX_LEN: usize;
    fn parse_hex(value: &str) -> UnionResult<Self>;
    fn as_hex(&self) -> &str;
}

fn validate_hex(domain: &str, value: &str, len: usize) -> UnionResult<()> {
    if value.len() != len {
        return Err(malformed(
            domain,
            format!(
                "expected {len} lowercase hex characters, found {}",
                value.len()
            ),
        ));
    }
    if !value
        .bytes()
        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(malformed(domain, "must be lowercase hexadecimal"));
    }
    if value.bytes().all(|byte| byte == b'0') {
        return Err(malformed(
            domain,
            "all-zero value is a placeholder, not an identity",
        ));
    }
    Ok(())
}

macro_rules! hex_identity {
    ($(#[$doc:meta])* $name:ident, $len:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            /// Parse and validate an identity of this domain.
            pub fn parse(value: &str) -> UnionResult<Self> {
                validate_hex(stringify!($name), value, $len)?;
                Ok(Self(value.to_owned()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl HexIdentity for $name {
            const HEX_LEN: usize = $len;

            fn parse_hex(value: &str) -> UnionResult<Self> {
                Self::parse(value)
            }

            fn as_hex(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = String::deserialize(deserializer)?;
                Self::parse(&value).map_err(serde::de::Error::custom)
            }
        }
    };
}

macro_rules! sha256_identity {
    ($(#[$doc:meta])* $name:ident) => {
        hex_identity!($(#[$doc])* $name, 64);

        impl $name {
            /// SHA-256 of `bytes` in this domain.
            pub fn of_bytes(bytes: &[u8]) -> Self {
                Self(sha256_hex(bytes))
            }
        }
    };
}

hex_identity!(
    /// A 40-character git commit id (a Mesa pin).
    GitCommit,
    40
);

sha256_identity!(
    /// Identity of a semantic recipe (what is computed, independent of any producer).
    SemanticRecipeDigest
);
sha256_identity!(
    /// Exact shape/input domain contract of a recipe.
    DomainContractDigest
);
sha256_identity!(
    /// Logical layout contract of a recipe.
    LayoutContractDigest
);
sha256_identity!(
    /// Floating-point conversion, rounding, fold-order and epilogue contract.
    FpContractDigest
);
sha256_identity!(
    /// Read/write extent contract of a recipe.
    AccessContractDigest
);

sha256_identity!(
    /// Shader/kernel source text.
    SourceDigest
);
sha256_identity!(
    /// SPIR-V module handed to a Vulkan driver.
    SpirvDigest
);
sha256_identity!(
    /// A compiler or driver build (binary or source tree) pin.
    BuildDigest
);
sha256_identity!(
    /// Canonical compiler / driver option set.
    OptionsDigest
);
sha256_identity!(
    /// A source patch applied on top of a pinned upstream commit.
    PatchDigest
);

sha256_identity!(
    /// Original executable instruction bytes exactly as produced.
    RawCodeDigest
);
sha256_identity!(
    /// Original non-executable data bytes exactly as produced.
    RawDataDigest
);
sha256_identity!(
    /// Final resource-programming configuration (RSRC words and friends).
    RawConfigDigest
);
sha256_identity!(
    /// Entry ABI facts: user SGPRs, local-id packing, aperture, dispatch mode.
    EntryAbiDigest
);
sha256_identity!(
    /// The producer's sidecar artifact (canonical `pm-aco-artifact` style JSON).
    SidecarDigest
);

sha256_identity!(
    /// The PM static certificate.
    StaticCertificateDigest
);
sha256_identity!(
    /// The ABI adapter definition (prologue + entry mapping + proofs).
    AdapterDigest
);
sha256_identity!(
    /// Adapter prologue bytes (new code, never part of the preserved body).
    PrologueDigest
);
sha256_identity!(
    /// A preserved instruction body: the same domain on the raw and adapted side
    /// so "unchanged" is an equality check.
    BodyDigest
);
sha256_identity!(
    /// Executable code exactly as uploaded to the device.
    ExecutableCodeDigest
);

sha256_identity!(
    /// An entire HSA ELF or offload bundle.
    HsaImageDigest
);
sha256_identity!(
    /// RADV `VkPipelineBinaryKeyKHR`.
    RadvBinaryKeyDigest
);
sha256_identity!(
    /// RADV `VkPipelineBinaryDataKHR` blob, original and unedited.
    RadvBlobDigest
);
sha256_identity!(
    /// RADV global pipeline key (cache UUID + device cache hash).
    RadvGlobalKeyDigest
);
sha256_identity!(
    /// Versioned canonical PM compute envelope for the pinned private RADV import.
    PrivateEnvelopeDigest
);
sha256_identity!(
    /// The isolated private-ICD build.
    PrivateIcdBuildDigest
);
sha256_identity!(
    /// Content-derived private cache key (see [`derive_private_cache_key`]).
    PrivateCacheKeyDigest
);
sha256_identity!(
    /// Windows driver pipeline binary key.
    WindowsBinaryKeyDigest
);
sha256_identity!(
    /// Windows driver pipeline binary blob, original and unedited.
    WindowsBlobDigest
);
sha256_identity!(
    /// Windows driver global compatibility key.
    WindowsGlobalKeyDigest
);

sha256_identity!(
    /// Canonical digest of a whole U-F1 record.
    RecordDigest
);
sha256_identity!(
    /// The image or pipeline the runtime actually admitted and loaded.
    LoadedExecutableDigest
);
sha256_identity!(
    /// Entry register/state facts observed at the loaded entry.
    EntryStateDigest
);
sha256_identity!(
    /// The relocations applied while loading.
    RelocationDigest
);
sha256_identity!(
    /// A loaded-executable receipt.
    LoadedReceiptDigest
);

sha256_identity!(
    /// The exact logical input / pre-image fixture.
    FixtureDigest
);
sha256_identity!(
    /// The ordered list of compared state surfaces.
    StateSurfaceDigest
);
sha256_identity!(
    /// Post-run bytes of every bound state surface.
    PostStateDigest
);
sha256_identity!(
    /// A runtime execution plan (backend/queue/graph/argument layout).
    RuntimePlanDigest
);

sha256_identity!(
    /// Vulkan pipeline layout.
    PipelineLayoutDigest
);
sha256_identity!(
    /// Specialization data block.
    SpecializationDigest
);
sha256_identity!(
    /// The blob-to-raw-code extraction mapping.
    ExtractionMappingDigest
);

sha256_identity!(
    /// HIP view of a shared allocation range.
    HipViewDigest
);
sha256_identity!(
    /// Vulkan view of a shared allocation range.
    VulkanViewDigest
);
sha256_identity!(
    /// A recorded capability census/receipt for a synchronization route.
    CapabilityReceiptDigest
);
sha256_identity!(
    /// Exact shape descriptor of a route-selection cell.
    ShapeDigest
);

/// Private cache keys are content-derived and namespaced: they bind the
/// private envelope and the static certificate and can never equal a stock
/// driver key.
pub fn derive_private_cache_key(
    envelope: &PrivateEnvelopeDigest,
    certificate: &StaticCertificateDigest,
) -> PrivateCacheKeyDigest {
    PrivateCacheKeyDigest(domain_sha256_hex(
        PRIVATE_CACHE_KEY_DOMAIN,
        &[
            envelope.as_str().as_bytes(),
            certificate.as_str().as_bytes(),
        ],
    ))
}

/// A byte range plus the digest of exactly those bytes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extent<D> {
    pub offset: u64,
    pub length: u64,
    pub sha256: D,
}

impl<D: HexIdentity> Extent<D> {
    /// Exclusive end offset; rejects empty ranges and overflow.
    pub fn end(&self, field: &str) -> UnionResult<u64> {
        if self.length == 0 {
            return Err(malformed(field, "zero-length range"));
        }
        checked_end(field, self.offset, self.length)
    }

    /// Exclusive end offset after checking the range lies inside `total`.
    pub fn within(&self, field: &str, total: u64) -> UnionResult<u64> {
        let end = self.end(field)?;
        if end > total {
            return Err(malformed(
                field,
                format!("range {}..{} exceeds {total} bytes", self.offset, end),
            ));
        }
        Ok(end)
    }

    fn span(&self) -> (u64, u64) {
        (self.offset, self.offset.saturating_add(self.length))
    }
}

// ---------------------------------------------------------------------------
// Target, semantic recipe, producer
// ---------------------------------------------------------------------------

/// Exact architecture and wave size. Prefixes ("gfx11") and generic targets
/// are rejected: a candidate certified for one target is not another's.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnionTarget {
    pub architecture: String,
    pub wave: Wavefront,
}

impl UnionTarget {
    pub fn validate(&self) -> UnionResult<()> {
        let exact = self.architecture.strip_prefix("gfx").is_some_and(|rest| {
            (3..=4).contains(&rest.len())
                && rest
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        });
        if !exact {
            return Err(malformed(
                "target.architecture",
                format!("`{}` is not an exact gfx target", self.architecture),
            ));
        }
        Ok(())
    }

    /// Require `other` to be the identical architecture and wave size.
    pub fn require_same(&self, what: &str, other: &UnionTarget) -> UnionResult<()> {
        if self.architecture != other.architecture {
            return Err(mismatch(
                &format!("{what} architecture"),
                &self.architecture,
                &other.architecture,
            ));
        }
        if self.wave != other.wave {
            return Err(mismatch(
                &format!("{what} wave size"),
                &self.wave.width().to_string(),
                &other.wave.width().to_string(),
            ));
        }
        Ok(())
    }
}

/// Semantic recipe identity: the exact numerical/layout contract, independent
/// of which compiler produced the executable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticRecipe {
    pub recipe_id: SemanticRecipeDigest,
    pub name: String,
    pub domain: DomainContractDigest,
    pub layout: LayoutContractDigest,
    pub fp: FpContractDigest,
    pub access: AccessContractDigest,
}

impl SemanticRecipe {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_text("semantic.name", &self.name)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProducerKind {
    /// hipcc / LLVM HSA code object.
    HipccLlvm,
    /// SPIR-V compiled by RADV's ACO branch.
    RadvAco,
    /// SPIR-V compiled by RADV's LLVM branch.
    RadvLlvm,
    /// hipfire's native builder.
    NativeBuilder,
    /// A Windows Vulkan driver. Never labelled as ACO or hipcc by convenience.
    WindowsDriver,
}

impl ProducerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HipccLlvm => "hipcc_llvm",
            Self::RadvAco => "radv_aco",
            Self::RadvLlvm => "radv_llvm",
            Self::NativeBuilder => "native_builder",
            Self::WindowsDriver => "windows_driver",
        }
    }

    /// Producers whose raw code follows a foreign (non-HSA) entry ABI.
    pub const fn is_foreign_abi(self) -> bool {
        matches!(self, Self::RadvAco | Self::RadvLlvm | Self::WindowsDriver)
    }

    const fn is_radv(self) -> bool {
        matches!(self, Self::RadvAco | Self::RadvLlvm)
    }
}

/// A named compiler or driver build.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildPin {
    pub version: String,
    pub build_sha256: BuildDigest,
}

impl BuildPin {
    fn validate(&self, field: &str) -> UnionResult<()> {
        ensure_text(&format!("{field}.version"), &self.version)
    }
}

/// A pinned Mesa tree. `patched` and `patch_sha256` agree: a patched tree
/// names its patch, an unpatched one carries none.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MesaPin {
    pub commit: GitCommit,
    pub patched: bool,
    pub patch_sha256: Option<PatchDigest>,
}

impl MesaPin {
    fn validate(&self) -> UnionResult<()> {
        match (self.patched, &self.patch_sha256) {
            (true, None) => Err(incomplete(
                "producer.mesa.patch_sha256",
                "a patched Mesa pin must name its patch",
            )),
            (false, Some(_)) => Err(malformed(
                "producer.mesa.patch_sha256",
                "an unpatched Mesa pin cannot carry a patch digest",
            )),
            _ => Ok(()),
        }
    }
}

/// Source, compiler, driver, option and Mesa pins of a producer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerRecord {
    pub kind: ProducerKind,
    pub source_sha256: SourceDigest,
    pub spirv_sha256: Option<SpirvDigest>,
    pub compiler: BuildPin,
    pub driver: Option<BuildPin>,
    pub options_sha256: OptionsDigest,
    pub mesa: Option<MesaPin>,
}

impl ProducerRecord {
    pub fn validate(&self) -> UnionResult<()> {
        self.compiler.validate("producer.compiler")?;
        let kind = self.kind.as_str();
        let spirv = matches!(
            self.kind,
            ProducerKind::RadvAco | ProducerKind::RadvLlvm | ProducerKind::WindowsDriver
        );
        require_presence(
            "producer.spirv_sha256",
            kind,
            spirv,
            self.spirv_sha256.is_some(),
        )?;
        require_presence("producer.driver", kind, spirv, self.driver.is_some())?;
        require_presence(
            "producer.mesa",
            kind,
            self.kind.is_radv(),
            self.mesa.is_some(),
        )?;
        if let Some(driver) = &self.driver {
            driver.validate("producer.driver")?;
        }
        if let Some(mesa) = &self.mesa {
            mesa.validate()?;
        }
        Ok(())
    }
}

fn require_presence(field: &str, kind: &str, required: bool, present: bool) -> UnionResult<()> {
    match (required, present) {
        (true, false) => Err(incomplete(field, format!("required for producer `{kind}`"))),
        (false, true) => Err(malformed(
            field,
            format!("not applicable to producer `{kind}`"),
        )),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Raw artifact, static certificate, adapter
// ---------------------------------------------------------------------------

/// The producer's raw artifact: executable bytes, data, config, entry ABI and
/// sidecar, with the byte ranges they occupy in the raw stream.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawArtifact {
    /// Length of the raw stream the extents index into.
    pub total_length: u64,
    pub code: Extent<RawCodeDigest>,
    pub data: Option<Extent<RawDataDigest>>,
    pub config_sha256: RawConfigDigest,
    pub entry_abi_sha256: EntryAbiDigest,
    pub sidecar_sha256: SidecarDigest,
    pub entry_name: String,
    /// Entry offset inside the code extent.
    pub entry_offset: u64,
}

impl RawArtifact {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_text("raw.entry_name", &self.entry_name)?;
        if self.total_length == 0 {
            return Err(incomplete("raw.total_length", "must be non-zero"));
        }
        self.code.within("raw.code", self.total_length)?;
        if self.entry_offset >= self.code.length {
            return Err(malformed(
                "raw.entry_offset",
                format!(
                    "entry offset {} is outside the {}-byte code extent",
                    self.entry_offset, self.code.length
                ),
            ));
        }
        if let Some(data) = &self.data {
            data.within("raw.data", self.total_length)?;
            if ranges_overlap(self.code.span(), data.span()) {
                return Err(malformed("raw.data", "overlaps the code extent"));
            }
        }
        Ok(())
    }

    /// Recompute the raw digests from the actual bytes, independent of any
    /// stored provenance. `raw` is the raw stream, `sidecar` the sidecar file.
    pub fn verify_bytes(&self, raw: &[u8], sidecar: &[u8]) -> UnionResult<()> {
        self.validate()?;
        ensure_equal(
            "raw stream length",
            &self.total_length.to_string(),
            &raw.len().to_string(),
        )?;
        let slice = |field: &str, offset: u64, length: u64| slice_of(raw, field, offset, length);
        let code = RawCodeDigest::of_bytes(slice("raw.code", self.code.offset, self.code.length)?);
        ensure_equal("raw code digest", self.code.sha256.as_str(), code.as_str())?;
        if let Some(data) = &self.data {
            let actual = RawDataDigest::of_bytes(slice("raw.data", data.offset, data.length)?);
            ensure_equal("raw data digest", data.sha256.as_str(), actual.as_str())?;
        }
        let actual = SidecarDigest::of_bytes(sidecar);
        ensure_equal(
            "sidecar digest",
            self.sidecar_sha256.as_str(),
            actual.as_str(),
        )
    }
}

/// Reference to a PM static certificate. The certificate itself is produced by
/// the PM checker; this record only binds it to the exact raw code and entry
/// ABI it covered.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticCertificate {
    pub certificate_sha256: StaticCertificateDigest,
    pub certified_code_sha256: RawCodeDigest,
    pub certified_entry_abi_sha256: EntryAbiDigest,
}

impl StaticCertificate {
    fn validate_against(&self, raw: &RawArtifact) -> UnionResult<()> {
        ensure_equal(
            "certificate code binding",
            raw.code.sha256.as_str(),
            self.certified_code_sha256.as_str(),
        )?;
        ensure_equal(
            "certificate entry ABI binding",
            raw.entry_abi_sha256.as_str(),
            self.certified_entry_abi_sha256.as_str(),
        )
    }
}

/// ABI adapter: a new prologue followed by the original body, unchanged.
///
/// `raw_body` indexes the raw code extent; `prologue` and `adapted_body` index
/// the adapted code. The body digest on both sides is the same domain and must
/// match, which is the "unchanged body" claim. The prologue and envelope are
/// new bytes and are not called byte-identical to the original shader.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterRecord {
    pub adapter_sha256: AdapterDigest,
    pub raw_body: Extent<BodyDigest>,
    pub adapted_code_length: u64,
    pub adapted_code_sha256: ExecutableCodeDigest,
    pub prologue: Extent<PrologueDigest>,
    pub adapted_body: Extent<BodyDigest>,
    /// Entry offset inside the adapted code; must lie in the prologue.
    pub entry_offset: u64,
}

impl AdapterRecord {
    fn validate(&self, raw: &RawArtifact) -> UnionResult<()> {
        if self.adapted_code_length == 0 {
            return Err(incomplete(
                "adapter.adapted_code_length",
                "must be non-zero",
            ));
        }
        self.raw_body.within("adapter.raw_body", raw.code.length)?;
        let prologue_end = self
            .prologue
            .within("adapter.prologue", self.adapted_code_length)?;
        self.adapted_body
            .within("adapter.adapted_body", self.adapted_code_length)?;
        if self.adapted_body.length != self.raw_body.length {
            return Err(mismatch(
                "adapter body length",
                &self.raw_body.length.to_string(),
                &self.adapted_body.length.to_string(),
            ));
        }
        ensure_equal(
            "adapter body digest (raw body vs adapted body)",
            self.raw_body.sha256.as_str(),
            self.adapted_body.sha256.as_str(),
        )?;
        if prologue_end > self.adapted_body.offset {
            return Err(malformed(
                "adapter.prologue",
                "must end before the preserved body begins",
            ));
        }
        if self.entry_offset < self.prologue.offset || self.entry_offset >= prologue_end {
            return Err(malformed(
                "adapter.entry_offset",
                "adapted entry must lie inside the prologue",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Execution envelopes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HsaImageKind {
    Elf,
    OffloadBundle,
}

/// Whole HSA ELF or offload bundle identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HsaEnvelope {
    pub image_kind: HsaImageKind,
    pub image_sha256: HsaImageDigest,
    pub image_length: u64,
    /// Executable code inside the image, as uploaded.
    pub code: Extent<ExecutableCodeDigest>,
    pub symbol: String,
}

/// Original, unedited RADV pipeline binary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadvOriginalEnvelope {
    pub binary_key_sha256: RadvBinaryKeyDigest,
    pub blob_sha256: RadvBlobDigest,
    pub blob_length: u64,
    pub global_key_sha256: RadvGlobalKeyDigest,
}

/// Private pinned-RADV import envelope. Deliberately has no stock key fields:
/// the original shader hash must not survive a code change.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadvPrivateEnvelope {
    pub private_envelope_sha256: PrivateEnvelopeDigest,
    pub private_envelope_length: u64,
    pub private_icd_build_sha256: PrivateIcdBuildDigest,
    pub cache_key_sha256: PrivateCacheKeyDigest,
}

/// Original, unedited Windows driver pipeline binary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowsOriginalEnvelope {
    pub binary_key_sha256: WindowsBinaryKeyDigest,
    pub blob_sha256: WindowsBlobDigest,
    pub blob_length: u64,
    pub global_key_sha256: WindowsGlobalKeyDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeKind {
    Hsa,
    RadvOriginalBinary,
    RadvPrivatePm,
    WindowsDriverOriginalBinary,
}

impl EnvelopeKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hsa => "hsa",
            Self::RadvOriginalBinary => "radv_original_binary",
            Self::RadvPrivatePm => "radv_private_pm",
            Self::WindowsDriverOriginalBinary => "windows_driver_original_binary",
        }
    }
}

/// The container that carries the executable. Kinds are distinct namespaces:
/// none of their digests is comparable with another's.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvelopeRecord {
    Hsa(HsaEnvelope),
    RadvOriginalBinary(RadvOriginalEnvelope),
    RadvPrivatePm(RadvPrivateEnvelope),
    WindowsDriverOriginalBinary(WindowsOriginalEnvelope),
}

impl EnvelopeRecord {
    pub fn kind(&self) -> EnvelopeKind {
        match self {
            Self::Hsa(_) => EnvelopeKind::Hsa,
            Self::RadvOriginalBinary(_) => EnvelopeKind::RadvOriginalBinary,
            Self::RadvPrivatePm(_) => EnvelopeKind::RadvPrivatePm,
            Self::WindowsDriverOriginalBinary(_) => EnvelopeKind::WindowsDriverOriginalBinary,
        }
    }

    /// Hex identity of the whole container; this is the value a loaded
    /// receipt must reproduce.
    pub fn identity_hex(&self) -> &str {
        match self {
            Self::Hsa(envelope) => envelope.image_sha256.as_str(),
            Self::RadvOriginalBinary(envelope) => envelope.blob_sha256.as_str(),
            Self::RadvPrivatePm(envelope) => envelope.private_envelope_sha256.as_str(),
            Self::WindowsDriverOriginalBinary(envelope) => envelope.blob_sha256.as_str(),
        }
    }

    /// Byte length of the whole container.
    pub fn container_length(&self) -> u64 {
        match self {
            Self::Hsa(envelope) => envelope.image_length,
            Self::RadvOriginalBinary(envelope) => envelope.blob_length,
            Self::RadvPrivatePm(envelope) => envelope.private_envelope_length,
            Self::WindowsDriverOriginalBinary(envelope) => envelope.blob_length,
        }
    }

    fn validate(&self) -> UnionResult<()> {
        if self.container_length() == 0 {
            return Err(incomplete(
                "envelope.length",
                "container length must be non-zero",
            ));
        }
        if let Self::Hsa(envelope) = self {
            ensure_text("envelope.symbol", &envelope.symbol)?;
            envelope
                .code
                .within("envelope.code", envelope.image_length)?;
        }
        Ok(())
    }
}

/// What a host stack accepts. Checked before any driver import.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnvelopeExpectation {
    Hsa,
    RadvOriginalBinary {
        global_key_sha256: RadvGlobalKeyDigest,
    },
    RadvPrivatePm {
        private_icd_build_sha256: PrivateIcdBuildDigest,
    },
    WindowsDriverOriginalBinary {
        global_key_sha256: WindowsGlobalKeyDigest,
    },
}

impl EnvelopeExpectation {
    pub const fn kind(&self) -> EnvelopeKind {
        match self {
            Self::Hsa => EnvelopeKind::Hsa,
            Self::RadvOriginalBinary { .. } => EnvelopeKind::RadvOriginalBinary,
            Self::RadvPrivatePm { .. } => EnvelopeKind::RadvPrivatePm,
            Self::WindowsDriverOriginalBinary { .. } => EnvelopeKind::WindowsDriverOriginalBinary,
        }
    }
}

// ---------------------------------------------------------------------------
// U-F1: certified executable
// ---------------------------------------------------------------------------

/// U-F1: the sealed identity of one certified executable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertifiedExecutableRecord {
    pub schema_version: u32,
    pub semantic: SemanticRecipe,
    pub producer: ProducerRecord,
    pub target: UnionTarget,
    pub raw: RawArtifact,
    pub certificate: StaticCertificate,
    /// Present exactly when the code was adapted to a different entry ABI.
    pub adapter: Option<AdapterRecord>,
    pub envelope: EnvelopeRecord,
}

versioned_json!(CertifiedExecutableRecord, "certified_executable");

impl CertifiedExecutableRecord {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("certified_executable", self.schema_version)?;
        self.semantic.validate()?;
        self.producer.validate()?;
        self.target.validate()?;
        self.raw.validate()?;
        self.certificate.validate_against(&self.raw)?;
        self.envelope.validate()?;
        self.validate_adapter_and_envelope()
    }

    fn validate_adapter_and_envelope(&self) -> UnionResult<()> {
        let kind = self.producer.kind;
        let adapter = self.adapter.as_ref();
        if let Some(adapter) = adapter {
            adapter.validate(&self.raw)?;
        }
        match &self.envelope {
            EnvelopeRecord::Hsa(hsa) => {
                if kind.is_foreign_abi() {
                    require_adapter(adapter.is_some(), kind, "an HSA envelope")?;
                } else {
                    forbid_adapter(adapter.is_some(), kind, "an HSA envelope")?;
                }
                let (code_sha, code_length) = self.executable_code();
                ensure_equal(
                    "HSA envelope code digest",
                    code_sha,
                    hsa.code.sha256.as_str(),
                )?;
                ensure_equal(
                    "HSA envelope code length",
                    &code_length.to_string(),
                    &hsa.code.length.to_string(),
                )
            }
            EnvelopeRecord::RadvOriginalBinary(_) => {
                if !kind.is_radv() {
                    return Err(UnionError::EnvelopeIncompatible(format!(
                        "an original RADV binary cannot come from producer `{}`",
                        kind.as_str()
                    )));
                }
                forbid_adapter(adapter.is_some(), kind, "an original RADV binary")
            }
            EnvelopeRecord::RadvPrivatePm(private) => {
                require_adapter(adapter.is_some(), kind, "a private RADV PM envelope")?;
                let derived = derive_private_cache_key(
                    &private.private_envelope_sha256,
                    &self.certificate.certificate_sha256,
                );
                ensure_equal(
                    "private cache key (must derive from envelope + certificate)",
                    derived.as_str(),
                    private.cache_key_sha256.as_str(),
                )
            }
            EnvelopeRecord::WindowsDriverOriginalBinary(_) => {
                if kind != ProducerKind::WindowsDriver {
                    return Err(UnionError::EnvelopeIncompatible(format!(
                        "a Windows driver binary cannot come from producer `{}`",
                        kind.as_str()
                    )));
                }
                forbid_adapter(adapter.is_some(), kind, "an original Windows driver binary")
            }
        }
    }

    /// Digest and length of the executable code exactly as uploaded: the
    /// adapted code when an adapter exists, otherwise the raw code.
    pub fn executable_code(&self) -> (&str, u64) {
        match &self.adapter {
            Some(adapter) => (
                adapter.adapted_code_sha256.as_str(),
                adapter.adapted_code_length,
            ),
            None => (self.raw.code.sha256.as_str(), self.raw.code.length),
        }
    }

    /// Entry offset inside the uploaded executable code.
    pub fn executable_entry_offset(&self) -> u64 {
        match &self.adapter {
            Some(adapter) => adapter.entry_offset,
            None => self.raw.entry_offset,
        }
    }

    /// Canonical digest of the whole record (compact JSON in declared field
    /// order, domain separated).
    pub fn record_digest(&self) -> UnionResult<RecordDigest> {
        let json = canonical_json(self)?;
        Ok(RecordDigest(domain_sha256_hex(
            RECORD_DIGEST_DOMAIN,
            &[json.as_slice()],
        )))
    }

    /// Check the record against what a host stack can accept, before import.
    pub fn check_envelope_compat(&self, expected: &EnvelopeExpectation) -> UnionResult<()> {
        let actual = self.envelope.kind();
        if actual != expected.kind() {
            return Err(UnionError::EnvelopeIncompatible(format!(
                "record envelope is `{}` but the host expects `{}`",
                actual.as_str(),
                expected.kind().as_str()
            )));
        }
        match (&self.envelope, expected) {
            (
                EnvelopeRecord::RadvOriginalBinary(envelope),
                EnvelopeExpectation::RadvOriginalBinary { global_key_sha256 },
            ) if &envelope.global_key_sha256 != global_key_sha256 => {
                Err(UnionError::EnvelopeIncompatible(
                    "RADV global key differs from the host's (blob must be rejected)".to_owned(),
                ))
            }
            (
                EnvelopeRecord::RadvPrivatePm(envelope),
                EnvelopeExpectation::RadvPrivatePm {
                    private_icd_build_sha256,
                },
            ) if &envelope.private_icd_build_sha256 != private_icd_build_sha256 => {
                Err(UnionError::EnvelopeIncompatible(
                    "private ICD build differs from the host's".to_owned(),
                ))
            }
            (
                EnvelopeRecord::WindowsDriverOriginalBinary(envelope),
                EnvelopeExpectation::WindowsDriverOriginalBinary { global_key_sha256 },
            ) if &envelope.global_key_sha256 != global_key_sha256 => {
                Err(UnionError::EnvelopeIncompatible(
                    "Windows driver global key differs from the host's".to_owned(),
                ))
            }
            _ => Ok(()),
        }
    }

    /// Require the actual device target to be the certified one.
    pub fn check_target(&self, actual: &UnionTarget) -> UnionResult<()> {
        self.target.require_same("device vs certified", actual)
    }
}

fn require_adapter(present: bool, kind: ProducerKind, container: &str) -> UnionResult<()> {
    if present {
        Ok(())
    } else {
        Err(incomplete(
            "adapter",
            format!(
                "{container} from producer `{}` needs an adapter record",
                kind.as_str()
            ),
        ))
    }
}

fn forbid_adapter(present: bool, kind: ProducerKind, container: &str) -> UnionResult<()> {
    if present {
        Err(malformed(
            "adapter",
            format!(
                "{container} from producer `{}` carries unchanged code; no adapter allowed",
                kind.as_str()
            ),
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Loaded executable
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadPath {
    /// Code copied verbatim into executable memory.
    VerbatimCopy,
    /// Code passed through a relocating loader; bytes may differ from the
    /// certified code and are not covered by the static certificate.
    RelocatedLoad,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadRecord {
    pub path: UploadPath,
    pub uploaded_length: u64,
    pub uploaded_sha256: ExecutableCodeDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelocationRecord {
    pub applied_count: u32,
    pub applied_sha256: Option<RelocationDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadedEntry {
    pub name: String,
    pub offset: u64,
    pub state_sha256: EntryStateDigest,
}

/// Evidence produced by a runtime that actually admitted a module/pipeline.
/// This module can bind and cross-check it; it never fabricates one.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadedExecutableReceipt {
    pub schema_version: u32,
    pub record_sha256: RecordDigest,
    pub target: UnionTarget,
    pub executable_sha256: LoadedExecutableDigest,
    pub upload: UploadRecord,
    pub relocation: RelocationRecord,
    pub entry: LoadedEntry,
}

versioned_json!(LoadedExecutableReceipt, "loaded_executable_receipt");

impl LoadedExecutableReceipt {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("loaded_executable_receipt", self.schema_version)?;
        self.target.validate()?;
        ensure_text("loaded.entry.name", &self.entry.name)?;
        if self.upload.uploaded_length == 0 {
            return Err(incomplete(
                "loaded.upload.uploaded_length",
                "must be non-zero",
            ));
        }
        match (
            self.relocation.applied_count,
            &self.relocation.applied_sha256,
        ) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(malformed(
                    "loaded.relocation",
                    "a relocation digest requires a non-zero applied_count",
                ));
            }
            (_, None) => {
                return Err(incomplete(
                    "loaded.relocation.applied_sha256",
                    "applied relocations need their digest",
                ));
            }
            (_, Some(_)) => {}
        }
        if self.upload.path == UploadPath::RelocatedLoad && self.relocation.applied_count == 0 {
            return Err(incomplete(
                "loaded.relocation",
                "a relocated load must record its relocations",
            ));
        }
        Ok(())
    }

    /// Bind this receipt to the certified record it claims to have loaded.
    pub fn bind_to(&self, record: &CertifiedExecutableRecord) -> UnionResult<()> {
        self.validate()?;
        record.validate()?;
        let digest = record.record_digest()?;
        ensure_equal(
            "loaded receipt record digest",
            digest.as_str(),
            self.record_sha256.as_str(),
        )?;
        record.check_target(&self.target)?;
        ensure_equal(
            "loaded executable identity (envelope)",
            record.envelope.identity_hex(),
            self.executable_sha256.as_str(),
        )?;
        if self.upload.path != UploadPath::VerbatimCopy || self.relocation.applied_count != 0 {
            return Err(incomplete(
                "loaded.upload",
                "relocated uploads are not covered by the static certificate",
            ));
        }
        let (code_sha, code_length) = record.executable_code();
        ensure_equal(
            "uploaded code digest",
            code_sha,
            self.upload.uploaded_sha256.as_str(),
        )?;
        ensure_equal(
            "uploaded code length",
            &code_length.to_string(),
            &self.upload.uploaded_length.to_string(),
        )?;
        ensure_equal(
            "loaded entry symbol",
            &record.raw.entry_name,
            &self.entry.name,
        )?;
        ensure_equal(
            "loaded entry offset",
            &record.executable_entry_offset().to_string(),
            &self.entry.offset.to_string(),
        )
    }

    /// Canonical digest of the receipt, referenced by route plans.
    pub fn receipt_digest(&self) -> UnionResult<LoadedReceiptDigest> {
        let json = canonical_json(self)?;
        Ok(LoadedReceiptDigest(domain_sha256_hex(
            RECORD_DIGEST_DOMAIN,
            &[json.as_slice()],
        )))
    }
}

// ---------------------------------------------------------------------------
// Four-arm pair oracle receipt
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleArm {
    /// Original SPIR-V under the producing Vulkan driver.
    V0,
    /// PM-lifted ACO body executed through HIP via the HSA adapter.
    Hpm,
    /// PM-certified original Vulkan binary replay (or a named private import).
    Vpm,
    /// The tuned hipcc/LLVM twin.
    Hllvm,
}

impl OracleArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V0 => "v0",
            Self::Hpm => "hpm",
            Self::Vpm => "vpm",
            Self::Hllvm => "hllvm",
        }
    }
}

/// The four claims: foreign-body preservation, Vulkan re-entry, and exact
/// replacement of the incumbent by each PM route.
pub const REQUIRED_CLAIMS: [(OracleArm, OracleArm); 4] = [
    (OracleArm::V0, OracleArm::Hpm),
    (OracleArm::V0, OracleArm::Vpm),
    (OracleArm::Hpm, OracleArm::Hllvm),
    (OracleArm::Vpm, OracleArm::Hllvm),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Hip,
    Vulkan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Eager,
    HipGraph,
    Railgun,
    VulkanCommandBuffer,
    Mixed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalSpirvArm {
    pub spirv_sha256: SpirvDigest,
    pub driver_build_sha256: BuildDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertifiedArm {
    pub record_sha256: RecordDigest,
    pub loaded_executable_sha256: LoadedExecutableDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HipccTwinArm {
    pub source_sha256: SourceDigest,
    pub code_object_sha256: HsaImageDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ArmIdentity {
    OriginalSpirv(OriginalSpirvArm),
    Certified(CertifiedArm),
    HipccTwin(HipccTwinArm),
}

impl ArmIdentity {
    const fn name(&self) -> &'static str {
        match self {
            Self::OriginalSpirv(_) => "original_spirv",
            Self::Certified(_) => "certified",
            Self::HipccTwin(_) => "hipcc_twin",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmBinding {
    pub arm: OracleArm,
    pub backend: Backend,
    pub identity: ArmIdentity,
    pub runtime_plan_sha256: RuntimePlanDigest,
    /// Independently restored pre-image; must equal the receipt fixture.
    pub pre_image_sha256: FixtureDigest,
    pub post_state_sha256: PostStateDigest,
}

impl ArmBinding {
    fn validate(&self, arm: OracleArm, fixture: &FixtureDigest) -> UnionResult<()> {
        if self.arm != arm {
            return Err(pair_binding(format!(
                "arm slot `{}` holds a binding for `{}`",
                arm.as_str(),
                self.arm.as_str()
            )));
        }
        let (backend, identity) = match arm {
            OracleArm::V0 => (Backend::Vulkan, "original_spirv"),
            OracleArm::Hpm => (Backend::Hip, "certified"),
            OracleArm::Vpm => (Backend::Vulkan, "certified"),
            OracleArm::Hllvm => (Backend::Hip, "hipcc_twin"),
        };
        if self.backend != backend {
            return Err(pair_binding(format!(
                "arm `{}` must run on {backend:?}, not {:?}",
                arm.as_str(),
                self.backend
            )));
        }
        if self.identity.name() != identity {
            return Err(pair_binding(format!(
                "arm `{}` needs a `{identity}` identity, found `{}`",
                arm.as_str(),
                self.identity.name()
            )));
        }
        if &self.pre_image_sha256 != fixture {
            return Err(pair_binding(format!(
                "arm `{}` pre-image differs from the receipt fixture",
                arm.as_str()
            )));
        }
        Ok(())
    }
}

/// Exactly four arms. A missing, duplicated or extra arm cannot be expressed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FourArms {
    pub v0: ArmBinding,
    pub hpm: ArmBinding,
    pub vpm: ArmBinding,
    pub hllvm: ArmBinding,
}

impl FourArms {
    pub fn get(&self, arm: OracleArm) -> &ArmBinding {
        match arm {
            OracleArm::V0 => &self.v0,
            OracleArm::Hpm => &self.hpm,
            OracleArm::Vpm => &self.vpm,
            OracleArm::Hllvm => &self.hllvm,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    Input,
    Output,
    State,
    Guard,
}

/// One compared byte surface (output, state, untouched input, guard region).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateSurface {
    pub name: String,
    pub kind: SurfaceKind,
}

/// Identity of an ordered surface list.
pub fn state_surface_digest(surfaces: &[StateSurface]) -> UnionResult<StateSurfaceDigest> {
    let json = canonical_json(surfaces)?;
    Ok(StateSurfaceDigest(domain_sha256_hex(
        STATE_SURFACE_DOMAIN,
        &[json.as_slice()],
    )))
}

/// Result of comparing two arms over every bound surface.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairClaim {
    pub lhs: OracleArm,
    pub rhs: OracleArm,
    pub compared_bytes: u64,
    pub differing_bytes: u64,
}

/// A deliberate one-byte corruption that the comparison had to catch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NegativeControl {
    pub mutated_surface: String,
    pub detected: bool,
}

/// Four-arm oracle receipt bound to exact artifacts, fixture, state surfaces,
/// execution mode and plans. It records an outcome measured elsewhere.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairOracleReceipt {
    pub schema_version: u32,
    pub semantic_recipe_id: SemanticRecipeDigest,
    pub target: UnionTarget,
    pub device_uuid: String,
    pub mode: ExecutionMode,
    pub fixture_sha256: FixtureDigest,
    pub state_surface_sha256: StateSurfaceDigest,
    pub surfaces: Vec<StateSurface>,
    pub arms: FourArms,
    pub claims: Vec<PairClaim>,
    pub negative_control: NegativeControl,
}

versioned_json!(PairOracleReceipt, "pair_oracle_receipt");

impl PairOracleReceipt {
    /// Structural completeness and internal consistency.
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("pair_oracle_receipt", self.schema_version)?;
        self.target.validate()?;
        ensure_text("device_uuid", &self.device_uuid)?;
        self.validate_surfaces()?;
        for arm in [
            OracleArm::V0,
            OracleArm::Hpm,
            OracleArm::Vpm,
            OracleArm::Hllvm,
        ] {
            self.arms.get(arm).validate(arm, &self.fixture_sha256)?;
        }
        self.validate_claims()?;
        ensure_text(
            "negative_control.mutated_surface",
            &self.negative_control.mutated_surface,
        )?;
        if !self
            .surfaces
            .iter()
            .any(|surface| surface.name == self.negative_control.mutated_surface)
        {
            return Err(pair_binding(
                "negative control mutated a surface that is not bound",
            ));
        }
        Ok(())
    }

    fn validate_surfaces(&self) -> UnionResult<()> {
        if !self
            .surfaces
            .iter()
            .any(|surface| surface.kind == SurfaceKind::Output)
        {
            return Err(incomplete(
                "surfaces",
                "at least one output surface is required",
            ));
        }
        let mut names = BTreeSet::new();
        for surface in &self.surfaces {
            ensure_text("surfaces.name", &surface.name)?;
            if !names.insert(surface.name.as_str()) {
                return Err(malformed(
                    "surfaces",
                    format!("duplicate surface `{}`", surface.name),
                ));
            }
        }
        let derived = state_surface_digest(&self.surfaces)?;
        ensure_equal(
            "state surface digest",
            derived.as_str(),
            self.state_surface_sha256.as_str(),
        )
    }

    fn validate_claims(&self) -> UnionResult<()> {
        if self.claims.len() != REQUIRED_CLAIMS.len() {
            return Err(incomplete(
                "claims",
                format!(
                    "exactly {} pair claims are required, found {}",
                    REQUIRED_CLAIMS.len(),
                    self.claims.len()
                ),
            ));
        }
        for (lhs, rhs) in REQUIRED_CLAIMS {
            let mut matches = self
                .claims
                .iter()
                .filter(|claim| claim.lhs == lhs && claim.rhs == rhs);
            let claim = matches.next().ok_or_else(|| {
                incomplete(
                    "claims",
                    format!("missing claim {} == {}", lhs.as_str(), rhs.as_str()),
                )
            })?;
            if matches.next().is_some() {
                return Err(malformed(
                    "claims",
                    format!("duplicate claim {} == {}", lhs.as_str(), rhs.as_str()),
                ));
            }
            if claim.compared_bytes == 0 {
                return Err(incomplete("claims.compared_bytes", "must be non-zero"));
            }
            if claim.differing_bytes > claim.compared_bytes {
                return Err(malformed(
                    "claims.differing_bytes",
                    "exceeds compared_bytes",
                ));
            }
            let same_state =
                self.arms.get(lhs).post_state_sha256 == self.arms.get(rhs).post_state_sha256;
            if same_state != (claim.differing_bytes == 0) {
                return Err(pair_binding(format!(
                    "claim {} == {} disagrees with the arms' post-state digests",
                    lhs.as_str(),
                    rhs.as_str()
                )));
            }
        }
        Ok(())
    }

    /// Whether every pair is byte-identical. Does not imply the run happened.
    pub fn all_pairs_exact(&self) -> bool {
        self.claims.iter().all(|claim| claim.differing_bytes == 0)
    }

    /// `validate` plus the promotion preconditions: zero differing bytes in
    /// every pair and a caught negative control.
    pub fn validate_exact(&self) -> UnionResult<()> {
        self.validate()?;
        if !self.all_pairs_exact() {
            return Err(pair_binding("at least one pair has differing bytes"));
        }
        if !self.negative_control.detected {
            return Err(pair_binding("the negative control was not detected"));
        }
        Ok(())
    }

    /// Bind the two certified arms to their U-F1 records and the receipt.
    pub fn bind_certified(
        &self,
        hpm: &CertifiedExecutableRecord,
        vpm: &CertifiedExecutableRecord,
    ) -> UnionResult<()> {
        self.validate()?;
        hpm.validate()?;
        vpm.validate()?;
        if hpm.envelope.kind() != EnvelopeKind::Hsa {
            return Err(pair_binding("Hpm must be an HSA-envelope executable"));
        }
        if !matches!(
            vpm.envelope.kind(),
            EnvelopeKind::RadvOriginalBinary
                | EnvelopeKind::RadvPrivatePm
                | EnvelopeKind::WindowsDriverOriginalBinary
        ) {
            return Err(pair_binding("Vpm must be a Vulkan-envelope executable"));
        }
        for (arm, record) in [(OracleArm::Hpm, hpm), (OracleArm::Vpm, vpm)] {
            let ArmIdentity::Certified(bound) = &self.arms.get(arm).identity else {
                return Err(pair_binding(format!(
                    "arm `{}` is not a certified arm",
                    arm.as_str()
                )));
            };
            ensure_equal(
                &format!("{} record digest", arm.as_str()),
                record.record_digest()?.as_str(),
                bound.record_sha256.as_str(),
            )?;
            ensure_equal(
                &format!("{} loaded executable", arm.as_str()),
                record.envelope.identity_hex(),
                bound.loaded_executable_sha256.as_str(),
            )?;
            ensure_equal(
                &format!("{} semantic recipe", arm.as_str()),
                self.semantic_recipe_id.as_str(),
                record.semantic.recipe_id.as_str(),
            )?;
            self.target.require_same(
                &format!("{} certified vs receipt", arm.as_str()),
                &record.target,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// U-F2: Vulkan executable record
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorType {
    StorageBuffer,
    UniformBuffer,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescriptorBinding {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescriptorSetLayout {
    pub set: u32,
    pub bindings: Vec<DescriptorBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineLayoutRecord {
    pub layout_sha256: PipelineLayoutDigest,
    pub push_constant_bytes: u32,
    pub sets: Vec<DescriptorSetLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecializationEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecializationRecord {
    pub data_sha256: SpecializationDigest,
    pub data_length: u32,
    pub entries: Vec<SpecializationEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkgroupContract {
    pub size: [u32; 3],
    pub shared_memory_bytes: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubgroupContract {
    pub size: u32,
    pub full_subgroups_required: bool,
}

/// Where the certified code (and optional data) sits inside the container blob.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionMapping {
    pub mapping_sha256: ExtractionMappingDigest,
    pub code: Extent<ExecutableCodeDigest>,
    pub data: Option<Extent<RawDataDigest>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Robustness {
    Disabled,
    RobustBufferAccess,
    RobustBufferAccess2,
}

/// What a bound buffer must satisfy at a layout binding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferBindingRequirement {
    pub set: u32,
    pub binding: u32,
    pub access: BufferAccess,
    pub min_range_bytes: u64,
    pub offset_alignment: u32,
    pub robustness: Robustness,
    pub logical_resource: String,
}

/// U-F2: certified executable plus the Vulkan contract needed to import and
/// bind it. Original-binary kinds never edit the blob.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VulkanExecutableRecord {
    pub schema_version: u32,
    pub certified: CertifiedExecutableRecord,
    pub pipeline_layout: PipelineLayoutRecord,
    pub specialization: SpecializationRecord,
    pub workgroup: WorkgroupContract,
    pub subgroup: SubgroupContract,
    pub extraction: ExtractionMapping,
    pub buffer_bindings: Vec<BufferBindingRequirement>,
}

versioned_json!(VulkanExecutableRecord, "vulkan_executable");

impl VulkanExecutableRecord {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("vulkan_executable", self.schema_version)?;
        self.certified.validate()?;
        if self.certified.envelope.kind() == EnvelopeKind::Hsa {
            return Err(UnionError::EnvelopeIncompatible(
                "a Vulkan executable record cannot carry an HSA envelope".to_owned(),
            ));
        }
        let layout = self.validate_layout()?;
        self.validate_specialization()?;
        self.validate_workgroup_and_subgroup()?;
        self.validate_extraction()?;
        self.validate_bindings(&layout)
    }

    /// Returns `(set, binding)` pairs defined by the pipeline layout.
    fn validate_layout(&self) -> UnionResult<BTreeSet<(u32, u32)>> {
        let mut defined = BTreeSet::new();
        let mut sets = BTreeSet::new();
        for set in &self.pipeline_layout.sets {
            if !sets.insert(set.set) {
                return Err(malformed(
                    "pipeline_layout.sets",
                    format!("duplicate set {}", set.set),
                ));
            }
            for binding in &set.bindings {
                if binding.count == 0 {
                    return Err(malformed(
                        "pipeline_layout.bindings.count",
                        "descriptor count must be non-zero",
                    ));
                }
                if !defined.insert((set.set, binding.binding)) {
                    return Err(malformed(
                        "pipeline_layout.bindings",
                        format!("duplicate binding {}:{}", set.set, binding.binding),
                    ));
                }
            }
        }
        Ok(defined)
    }

    fn validate_specialization(&self) -> UnionResult<()> {
        let mut ids = BTreeSet::new();
        for entry in &self.specialization.entries {
            if !ids.insert(entry.constant_id) {
                return Err(malformed(
                    "specialization.entries",
                    format!("duplicate constant_id {}", entry.constant_id),
                ));
            }
            let end = u64::from(entry.offset) + u64::from(entry.size);
            if entry.size == 0 || end > u64::from(self.specialization.data_length) {
                return Err(malformed(
                    "specialization.entries",
                    format!("constant {} lies outside the data block", entry.constant_id),
                ));
            }
        }
        Ok(())
    }

    fn validate_workgroup_and_subgroup(&self) -> UnionResult<()> {
        if self.workgroup.size.contains(&0) {
            return Err(malformed(
                "workgroup.size",
                "every dimension must be non-zero",
            ));
        }
        let invocations: u64 = self.workgroup.size.iter().map(|&n| u64::from(n)).product();
        if invocations > 1024 {
            return Err(malformed("workgroup.size", "exceeds 1024 invocations"));
        }
        let wave = self.certified.target.wave.width();
        if self.subgroup.size != wave {
            return Err(mismatch(
                "subgroup size vs certified wave size",
                &wave.to_string(),
                &self.subgroup.size.to_string(),
            ));
        }
        if self.subgroup.full_subgroups_required && invocations % u64::from(wave) != 0 {
            return Err(malformed(
                "workgroup.size",
                "full subgroups required but invocations are not a wave multiple",
            ));
        }
        Ok(())
    }

    fn validate_extraction(&self) -> UnionResult<()> {
        let blob_length = self.certified.envelope.container_length();
        self.extraction
            .code
            .within("extraction.code", blob_length)?;
        let (code_sha, code_length) = self.certified.executable_code();
        ensure_equal(
            "extraction code digest",
            code_sha,
            self.extraction.code.sha256.as_str(),
        )?;
        ensure_equal(
            "extraction code length",
            &code_length.to_string(),
            &self.extraction.code.length.to_string(),
        )?;
        match (&self.certified.raw.data, &self.extraction.data) {
            (None, None) => {}
            (Some(raw), Some(mapped)) => {
                mapped.within("extraction.data", blob_length)?;
                ensure_equal(
                    "extraction data digest",
                    raw.sha256.as_str(),
                    mapped.sha256.as_str(),
                )?;
                ensure_equal(
                    "extraction data length",
                    &raw.length.to_string(),
                    &mapped.length.to_string(),
                )?;
                if ranges_overlap(self.extraction.code.span(), mapped.span()) {
                    return Err(malformed("extraction.data", "overlaps the code mapping"));
                }
            }
            _ => {
                return Err(incomplete(
                    "extraction.data",
                    "raw data and its extraction mapping must both be present or both absent",
                ));
            }
        }
        Ok(())
    }

    fn validate_bindings(&self, layout: &BTreeSet<(u32, u32)>) -> UnionResult<()> {
        let mut bound = BTreeSet::new();
        for requirement in &self.buffer_bindings {
            let key = (requirement.set, requirement.binding);
            ensure_text(
                "buffer_bindings.logical_resource",
                &requirement.logical_resource,
            )?;
            if !bound.insert(key) {
                return Err(malformed(
                    "buffer_bindings",
                    format!("duplicate requirement for {}:{}", key.0, key.1),
                ));
            }
            if !layout.contains(&key) {
                return Err(malformed(
                    "buffer_bindings",
                    format!("{}:{} is not in the pipeline layout", key.0, key.1),
                ));
            }
            if requirement.min_range_bytes == 0 {
                return Err(incomplete(
                    "buffer_bindings.min_range_bytes",
                    "must be non-zero",
                ));
            }
            if !requirement.offset_alignment.is_power_of_two() {
                return Err(malformed(
                    "buffer_bindings.offset_alignment",
                    "must be a non-zero power of two",
                ));
            }
        }
        if let Some(missing) = layout.difference(&bound).next() {
            return Err(incomplete(
                "buffer_bindings",
                format!(
                    "layout binding {}:{} has no requirement",
                    missing.0, missing.1
                ),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// U-F3: shared resource and completion boundary (serialized contracts only)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Api {
    Hip,
    Vulkan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    DeviceLocal,
    HostVisibleCoherent,
    HostVisibleCached,
}

/// A range of one physical allocation. Alias analysis keys on this, never on a
/// virtual pointer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalRange {
    pub allocation_id: u64,
    pub offset: u64,
    pub length: u64,
}

impl PhysicalRange {
    fn end(&self, field: &str) -> UnionResult<u64> {
        if self.length == 0 {
            return Err(malformed(field, "zero-length range"));
        }
        checked_end(field, self.offset, self.length)
    }

    fn contains(&self, field: &str, inner: &PhysicalRange) -> UnionResult<bool> {
        Ok(self.allocation_id == inner.allocation_id
            && inner.offset >= self.offset
            && inner.end(field)? <= self.end(field)?)
    }
}

/// A borrowed view over an allocator-owned resource at one generation. It
/// records identity only: no handle, no pointer and no ownership of the
/// backing.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedBufferViewRecord {
    pub schema_version: u32,
    pub resource_id: u64,
    pub physical_range: PhysicalRange,
    pub generation: u64,
    pub placement: Placement,
    pub hip_view_sha256: HipViewDigest,
    pub vulkan_view_sha256: VulkanViewDigest,
    pub layout_id: LayoutContractDigest,
}

versioned_json!(SharedBufferViewRecord, "shared_buffer_view");

impl SharedBufferViewRecord {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("shared_buffer_view", self.schema_version)?;
        self.physical_range
            .end("shared_buffer_view.physical_range")?;
        if self.generation == 0 {
            return Err(incomplete(
                "shared_buffer_view.generation",
                "generations start at 1",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub api: Api,
    pub queue_family: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeAccess {
    pub resource_id: u64,
    pub generation: u64,
    pub range: PhysicalRange,
    pub producer: AccessKind,
    pub consumer: AccessKind,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisibilityOp {
    ProducerRelease,
    ConsumerAcquire,
    QueueFamilyOwnershipRelease,
    QueueFamilyOwnershipAcquire,
    HostFenceWait,
    SemaphoreSignal,
    SemaphoreWait,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMode {
    HostFenced,
    ExternalSemaphore,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionKind {
    HostFence,
    BinarySemaphore,
    TimelineSemaphore,
}

/// Backend-typed, epoch-bound completion. It names the producing API and
/// cannot describe another API's handle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionTokenRecord {
    pub api: Api,
    pub epoch: u64,
    pub kind: CompletionKind,
}

/// A handoff between two APIs: exact ranges/accesses, required visibility
/// operations, synchronization mode and completion.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryContractRecord {
    pub schema_version: u32,
    pub boundary_id: u32,
    pub epoch: u64,
    pub producer: Endpoint,
    pub consumer: Endpoint,
    pub accesses: Vec<RangeAccess>,
    pub visibility: Vec<VisibilityOp>,
    pub sync_mode: SyncMode,
    pub completion: CompletionTokenRecord,
    /// Required for `ExternalSemaphore`: the recorded capability census.
    pub capability_receipt_sha256: Option<CapabilityReceiptDigest>,
}

versioned_json!(BoundaryContractRecord, "boundary_contract");

impl BoundaryContractRecord {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("boundary_contract", self.schema_version)?;
        if self.epoch == 0 {
            return Err(incomplete("boundary.epoch", "epochs start at 1"));
        }
        if self.producer.api == self.consumer.api {
            return Err(malformed(
                "boundary",
                "producer and consumer must be different APIs",
            ));
        }
        if self.accesses.is_empty() {
            return Err(incomplete(
                "boundary.accesses",
                "at least one range access is required",
            ));
        }
        for access in &self.accesses {
            access.range.end("boundary.accesses.range")?;
            if access.generation == 0 {
                return Err(incomplete(
                    "boundary.accesses.generation",
                    "generations start at 1",
                ));
            }
        }
        let ops: BTreeSet<VisibilityOp> = self.visibility.iter().copied().collect();
        if ops.len() != self.visibility.len() {
            return Err(malformed("boundary.visibility", "duplicate operation"));
        }
        let host = ops.contains(&VisibilityOp::HostFenceWait);
        let signal = ops.contains(&VisibilityOp::SemaphoreSignal);
        let wait = ops.contains(&VisibilityOp::SemaphoreWait);
        match self.sync_mode {
            SyncMode::HostFenced => {
                if !host || signal || wait {
                    return Err(malformed(
                        "boundary.visibility",
                        "host-fenced mode needs a host fence wait and no semaphore operations",
                    ));
                }
                if self.completion.kind != CompletionKind::HostFence {
                    return Err(malformed(
                        "boundary.completion",
                        "host-fenced mode needs a host-fence completion",
                    ));
                }
            }
            SyncMode::ExternalSemaphore => {
                if host || !signal || !wait {
                    return Err(malformed(
                        "boundary.visibility",
                        "external-semaphore mode needs signal and wait and no host fence wait",
                    ));
                }
                if self.completion.kind == CompletionKind::HostFence {
                    return Err(malformed(
                        "boundary.completion",
                        "external-semaphore mode needs a semaphore completion",
                    ));
                }
                if self.capability_receipt_sha256.is_none() {
                    return Err(incomplete(
                        "boundary.capability_receipt_sha256",
                        "async requires an explicit capability receipt",
                    ));
                }
            }
        }
        if self.completion.api != self.producer.api {
            return Err(malformed(
                "boundary.completion",
                "completion token must belong to the producing API",
            ));
        }
        if self.completion.epoch != self.epoch {
            return Err(mismatch(
                "completion epoch",
                &self.epoch.to_string(),
                &self.completion.epoch.to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// U-F4: route plan (serialized contract only)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnionMode {
    Off,
    HostFenced,
    Async,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IslandBackend {
    Hip,
    Railgun,
    Vulkan,
}

impl IslandBackend {
    const fn api(self) -> Api {
        match self {
            Self::Hip | Self::Railgun => Api::Hip,
            Self::Vulkan => Api::Vulkan,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    pub grid: [u32; 3],
    pub workgroup: [u32; 3],
}

/// Exact route-selection cell. Selection is per cell, never inferred from a
/// neighbouring architecture, shape or driver.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellKey {
    pub semantic_recipe_id: SemanticRecipeDigest,
    pub target: UnionTarget,
    pub driver_build_sha256: Option<BuildDigest>,
    pub shape_sha256: ShapeDigest,
    pub dtype_qtype: String,
    pub geometry: Geometry,
    pub placement: Placement,
    pub execution_mode: ExecutionMode,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IslandKernel {
    pub record_sha256: RecordDigest,
    pub loaded_receipt_sha256: LoadedReceiptDigest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IslandRecord {
    pub island_id: u32,
    pub backend: IslandBackend,
    pub queue_family: u32,
    pub kernels: Vec<IslandKernel>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IslandEdge {
    pub from: u32,
    pub to: u32,
    /// Required exactly when the edge crosses APIs.
    pub boundary_id: Option<u32>,
}

/// U-F4: immutable assignment of admitted kernels to islands, with boundary
/// contracts and resource generations. A description of a plan, never a
/// scheduler.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnionPlanRecord {
    pub schema_version: u32,
    pub mode: UnionMode,
    pub cell: CellKey,
    pub islands: Vec<IslandRecord>,
    pub edges: Vec<IslandEdge>,
    pub boundaries: Vec<BoundaryContractRecord>,
    pub resources: Vec<SharedBufferViewRecord>,
}

versioned_json!(UnionPlanRecord, "union_plan");

impl UnionPlanRecord {
    pub fn validate(&self) -> UnionResult<()> {
        ensure_version("union_plan", self.schema_version)?;
        self.cell.target.validate()?;
        ensure_text("cell.dtype_qtype", &self.cell.dtype_qtype)?;
        if self
            .cell
            .geometry
            .grid
            .iter()
            .chain(&self.cell.geometry.workgroup)
            .any(|&n| n == 0)
        {
            return Err(malformed(
                "cell.geometry",
                "every dimension must be non-zero",
            ));
        }
        if self.mode == UnionMode::Off {
            if !(self.islands.is_empty()
                && self.edges.is_empty()
                && self.boundaries.is_empty()
                && self.resources.is_empty())
            {
                return Err(malformed(
                    "union_plan",
                    "mode `off` retains current routes and carries no islands or boundaries",
                ));
            }
            return Ok(());
        }
        if self.islands.is_empty() {
            return Err(incomplete(
                "islands",
                "an enabled plan needs at least one island",
            ));
        }
        let resources = self.validate_resources()?;
        let apis = self.validate_islands()?;
        let boundaries = self.validate_boundaries(&resources)?;
        self.validate_edges(&apis, &boundaries)
    }

    fn validate_resources(&self) -> UnionResult<BTreeMap<(u64, u64), &SharedBufferViewRecord>> {
        let mut resources = BTreeMap::new();
        for view in &self.resources {
            view.validate()?;
            if resources
                .insert((view.resource_id, view.generation), view)
                .is_some()
            {
                return Err(malformed(
                    "resources",
                    format!(
                        "duplicate resource {} generation {}",
                        view.resource_id, view.generation
                    ),
                ));
            }
        }
        Ok(resources)
    }

    fn validate_islands(&self) -> UnionResult<BTreeMap<u32, Api>> {
        let mut apis = BTreeMap::new();
        for island in &self.islands {
            if island.kernels.is_empty() {
                return Err(incomplete(
                    "islands.kernels",
                    format!("island {} has no admitted kernel", island.island_id),
                ));
            }
            if apis
                .insert(island.island_id, island.backend.api())
                .is_some()
            {
                return Err(malformed(
                    "islands",
                    format!("duplicate island id {}", island.island_id),
                ));
            }
        }
        Ok(apis)
    }

    fn validate_boundaries(
        &self,
        resources: &BTreeMap<(u64, u64), &SharedBufferViewRecord>,
    ) -> UnionResult<BTreeMap<u32, &BoundaryContractRecord>> {
        let mut boundaries = BTreeMap::new();
        for boundary in &self.boundaries {
            boundary.validate()?;
            let required = match self.mode {
                UnionMode::HostFenced => SyncMode::HostFenced,
                _ => SyncMode::ExternalSemaphore,
            };
            if boundary.sync_mode != required {
                return Err(malformed(
                    "boundaries.sync_mode",
                    format!(
                        "boundary {} sync mode disagrees with plan mode {:?}",
                        boundary.boundary_id, self.mode
                    ),
                ));
            }
            for access in &boundary.accesses {
                let view = resources
                    .get(&(access.resource_id, access.generation))
                    .ok_or_else(|| {
                        incomplete(
                            "boundaries.accesses",
                            format!(
                                "resource {} generation {} is not in the plan",
                                access.resource_id, access.generation
                            ),
                        )
                    })?;
                if !view
                    .physical_range
                    .contains("boundaries.accesses.range", &access.range)?
                {
                    return Err(malformed(
                        "boundaries.accesses.range",
                        format!(
                            "range lies outside resource {} generation {}",
                            access.resource_id, access.generation
                        ),
                    ));
                }
            }
            if boundaries.insert(boundary.boundary_id, boundary).is_some() {
                return Err(malformed(
                    "boundaries",
                    format!("duplicate boundary id {}", boundary.boundary_id),
                ));
            }
        }
        Ok(boundaries)
    }

    fn validate_edges(
        &self,
        apis: &BTreeMap<u32, Api>,
        boundaries: &BTreeMap<u32, &BoundaryContractRecord>,
    ) -> UnionResult<()> {
        let mut referenced = BTreeSet::new();
        let mut indegree: BTreeMap<u32, usize> = apis.keys().map(|&id| (id, 0)).collect();
        let mut successors: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for edge in &self.edges {
            let (Some(&from_api), Some(&to_api)) = (apis.get(&edge.from), apis.get(&edge.to))
            else {
                return Err(malformed(
                    "edges",
                    format!("edge {} -> {} names an unknown island", edge.from, edge.to),
                ));
            };
            if edge.from == edge.to {
                return Err(malformed("edges", "self edge"));
            }
            match (from_api == to_api, edge.boundary_id) {
                (true, None) => {}
                (true, Some(_)) => {
                    return Err(malformed(
                        "edges.boundary_id",
                        "an edge inside one API must not carry a boundary",
                    ));
                }
                (false, None) => {
                    return Err(incomplete(
                        "edges.boundary_id",
                        "an edge crossing APIs needs a boundary contract",
                    ));
                }
                (false, Some(id)) => {
                    let boundary = boundaries.get(&id).ok_or_else(|| {
                        incomplete("edges.boundary_id", format!("unknown boundary {id}"))
                    })?;
                    if boundary.producer.api != from_api || boundary.consumer.api != to_api {
                        return Err(malformed(
                            "edges.boundary_id",
                            format!("boundary {id} APIs disagree with the edge direction"),
                        ));
                    }
                    referenced.insert(id);
                }
            }
            if let Some(count) = indegree.get_mut(&edge.to) {
                *count += 1;
            }
            successors.entry(edge.from).or_default().push(edge.to);
        }
        if let Some(unused) = boundaries.keys().find(|id| !referenced.contains(*id)) {
            return Err(incomplete(
                "boundaries",
                format!("boundary {unused} is not used by any edge"),
            ));
        }
        let mut ready: Vec<u32> = indegree
            .iter()
            .filter(|entry| *entry.1 == 0)
            .map(|entry| *entry.0)
            .collect();
        let mut visited = 0usize;
        while let Some(id) = ready.pop() {
            visited += 1;
            for next in successors.get(&id).into_iter().flatten() {
                if let Some(count) = indegree.get_mut(next) {
                    *count -= 1;
                    if *count == 0 {
                        ready.push(*next);
                    }
                }
            }
        }
        if visited != apis.len() {
            return Err(malformed("edges", "dependency graph contains a cycle"));
        }
        Ok(())
    }

    /// The plan identity quoted by oracle receipts and loaded kernels.
    pub fn runtime_plan_digest(&self) -> UnionResult<RuntimePlanDigest> {
        let json = canonical_json(self)?;
        Ok(RuntimePlanDigest(domain_sha256_hex(
            PLAN_DIGEST_DOMAIN,
            &[json.as_slice()],
        )))
    }
}
