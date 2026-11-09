// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed model-registry loading and model identity resolution.
//!
//! The checked-in v1 registry is compiled into every binary as the offline
//! floor. Dynamic loading preserves the established fallback order: fresh
//! cache, network, stale cache, then bundled. A malformed registry is rejected
//! wholesale; model code never consumes a partially validated catalog.

use hipfire_config::{
    load_catalog, load_global, resolve, ConfigFormat, ConfigLayer, ConfigPaths, ConfigSource,
    ConfigValue, NamedLayer, ResolvedConfig,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
pub mod local_files;
pub use local_files::{
    is_expert_hotset_name, is_hc8_sidecar_name, is_hot_units_sidecar_name, is_listable_model_file,
    is_model_file, is_ple_sidecar_name, is_standalone_model, is_trunk_overlay_name,
    names_registry_model, probe_hfq_arch_id, registry_sidecar_files, MODEL_SUFFIXES,
    PLE_AUX_SUFFIXES, SERVE_TRUNK_ARCH_IDS,
};
pub mod xdna;
pub use xdna::{XdnaManifest, XdnaProfile, XdnaSidecarDescriptor};
pub mod receipts;

pub const REGISTRY_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/warpfront/hipfire/master/registry/v1.json";
pub const REGISTRY_CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub const REGISTRY_FETCH_TIMEOUT: Duration = Duration::from_millis(3500);
const BUNDLED_REGISTRY: &str = include_str!("../../../registry/v1.json");

pub const FLASH_NEXT_XTS_LEGACY_FILE: &str = "qwen3.8-flash-next-gptq3.mq4";
pub const FLASH_NEXT_XTS_CANONICAL_FILE: &str = "qwen3.8-flash-next.mq4-xts";
const FLASH_NEXT_XTS_REPO: &str = "hipfire-models/qwen3.8-flash-next";
const FLASH_NEXT_XTS_SIZE: u64 = 125288544792;
const FLASH_NEXT_XTS_SHA256: &str =
    "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";

/// The Flash-Next MQ4 XTS split base every stock MQ4 tag publishes: the
/// PLE-less base of `hipfire ple split` over the monolithic
/// [`FLASH_NEXT_XTS_CANONICAL_FILE`].
pub const FLASH_NEXT_XTS_SPLIT_BASE_FILE: &str = "qwen3.8-flash-next-base.mq4-xts";
const FLASH_NEXT_XTS_SPLIT_BASE_SIZE: u64 = 70_888_270_360;
const FLASH_NEXT_XTS_SPLIT_BASE_SHA256: &str =
    "eafaff94fb9c6bb51282942e0ea6441cbdc266b634bb003b5913cfe2ccb90f45";

/// PLE format key of the exact Q8F16 rows: the reference every lossy format is
/// derived from. A grandfathered monolithic embeds it, and it is the fallback
/// when a lossy format a client wants is unpublished.
pub const PLE_EXACT_FORMAT: &str = "q8";

/// True for a PLE format token (a key of [`ModelEntry::ple`]): non-empty,
/// lowercase ASCII letters and digits.
pub fn is_ple_format_token(key: &str) -> bool {
    !key.is_empty()
        && key
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
}

/// Suffix appended to a split base's file name for its PLE rows of format
/// `key` (`.ple-q4`).
pub fn ple_sidecar_suffix(key: &str) -> String {
    format!(".ple-{key}")
}

/// Marker inserted before the extension of a split base's file name
/// (`qwen3.8-flash-next.mq4-xts` -> `qwen3.8-flash-next-base.mq4-xts`); the
/// mirror of `hipfire_runtime::ple_convert::default_split_base_path`.
const SPLIT_BASE_STEM_SUFFIX: &str = "-base";

/// True when `file` is a bare split-base file name (`<name>-base.<ext>`).
pub fn is_split_base_file(file: &str) -> bool {
    monolithic_file_for_split_base(file).is_some()
}

/// The monolithic sibling file name of a split-base file name: drops the
/// `-base` marker from the stem. `None` when `file` is not a bare split-base
/// name (a path, an empty stem, or no `-base` marker).
pub fn monolithic_file_for_split_base(file: &str) -> Option<String> {
    if file.contains('/') || file.contains('\\') {
        return None;
    }
    let path = Path::new(file);
    let stem = path.file_stem()?.to_str()?;
    let prefix = stem.strip_suffix(SPLIT_BASE_STEM_SUFFIX)?;
    if prefix.is_empty() {
        return None;
    }
    Some(
        match path.extension().and_then(|extension| extension.to_str()) {
            Some(extension) => format!("{prefix}.{extension}"),
            None => prefix.to_owned(),
        },
    )
}

/// Stem (before `-base`) of the one split-base family that shares its PLE row
/// files between bases (`qwen3.8-flash-next-base.<ext>`). Every other file
/// keeps strictly per-base sidecar names.
const SHARED_PLE_FAMILY: &str = "qwen3.8-flash-next";

/// The family-wide PLE row file a split base may name instead of its own
/// `<file>.ple-<key>` (`qwen3.8-flash-next-base.mq3-xts` + `q4` ->
/// `qwen3.8-flash-next.ple-q4`). `None` unless `base_file` is a Flash-Next
/// split base and `key` a [`is_ple_format_token`].
///
/// The rows are bound to the base by the row table, not by the file name, so
/// one shared set serves every split base of the family; consumers install
/// `<base file>.ple-<key>` as a link to it for the runtime loader.
pub fn shared_ple_file(base_file: &str, key: &str) -> Option<String> {
    monolithic_file_for_split_base(base_file)?;
    if !is_ple_format_token(key) {
        return None;
    }
    let stem = Path::new(base_file).file_stem()?.to_str()?;
    (stem.strip_suffix(SPLIT_BASE_STEM_SUFFIX)? == SHARED_PLE_FAMILY)
        .then(|| format!("{SHARED_PLE_FAMILY}{}", ple_sidecar_suffix(key)))
}

/// Local file names that serve a split `entry`, best first: the split base,
/// then its grandfathered monolithic (both pinned spellings), from the pin
/// table `pins`. This is what lets a user who never migrated keep using the
/// local monolithic under the split tag instead of re-downloading.
pub fn split_model_files_in(entry: &ModelEntry, pins: &[GrandfatheredMonolithic]) -> Vec<String> {
    let mut files = vec![entry.file.clone()];
    if let Some(pin) = grandfathered_monolithic_in(entry, pins) {
        files.push(pin.canonical_file.to_owned());
        files.push(pin.legacy_file.to_owned());
    }
    files
}

/// [`split_model_files_in`] over the compiled-in [`GRANDFATHERED_MONOLITHICS`].
pub fn split_model_files(entry: &ModelEntry) -> Vec<String> {
    split_model_files_in(entry, GRANDFATHERED_MONOLITHICS)
}

/// Native Qwen4 (Flash-Next) `model_arch` id.
const FLASH_NEXT_ARCH_ID: u32 = 16;

/// Which installed Flash-Next models have their q8 PLE rows (the local facts
/// the tuning plan needs, spec §2.2). A Flash-Next model is a registry entry
/// with `arch_id` 16 whose file (or grandfathered monolithic) is on disk. A
/// monolithic file embeds the q8 rows; a split base (`<name>-base.<ext>`)
/// needs `<base>.ple-q8` beside it, the entry's published q8 file, or the
/// family-shared q8 file.
pub fn tuning_plan_env(
    models_dir: &Path,
    registry: &RegistryV1,
) -> hipfire_config::tuning::PlanEnv {
    use hipfire_config::memory::{ple_sidecar_path, PleFormat};
    let mut env = hipfire_config::tuning::PlanEnv::default();
    let mut seen = std::collections::BTreeSet::new();
    for (tag, entry) in &registry.models {
        if entry.arch_id != Some(FLASH_NEXT_ARCH_ID) {
            continue;
        }
        let Some(file) = split_model_files(entry)
            .into_iter()
            .find(|file| models_dir.join(file).is_file())
        else {
            continue;
        };
        if !seen.insert(file.clone()) {
            continue;
        }
        env.flash_next_installed = true;
        let model = models_dir.join(&file);
        let split_base = monolithic_file_for_split_base(&file).is_some();
        let present = !split_base
            || ple_sidecar_path(&model, PleFormat::Q8).is_file()
            || entry
                .ple_sidecar("q8")
                .is_some_and(|sidecar| models_dir.join(&sidecar.file).is_file())
            || shared_ple_file(&file, "q8").is_some_and(|shared| models_dir.join(shared).is_file());
        if !present {
            env.ple_q8_missing.push(file);
            env.pull_tag.get_or_insert_with(|| tag.clone());
        }
    }
    env.ple_q8_present = env.flash_next_installed && env.ple_q8_missing.is_empty();
    env
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse registry from {source_name}: {message}")]
    Parse {
        source_name: String,
        message: String,
    },
    #[error("invalid registry from {source_name}: {message}")]
    Invalid {
        source_name: String,
        message: String,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid registry model defaults: {0}")]
    ModelDefaults(String),
    #[error(transparent)]
    Config(#[from] hipfire_config::ConfigError),
}

pub type Result<T> = std::result::Result<T, RegistryError>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sidecar {
    pub file: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

/// Content identity of one trunk overlay sidecar and of the exact base
/// artifact it is bound to.
///
/// The overlay is an HFQM container of pre-encoded QT44 trunk tensors
/// (`<base file>.trunk-overlay`) that the Qwen4 loader applies by default on
/// gfx1151 (`HIPFIRE_QWEN4_TRUNK_OVERLAY=0` is the kill switch). The pin lives
/// in the client instead of a registry field so that older clients' strict
/// `deny_unknown_fields` v1 parser keeps accepting the published registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkOverlayPin {
    /// SHA-256 of the base artifact (the registry entry's `sha256`).
    pub base_sha256: &'static str,
    /// Size in bytes of the base artifact (the registry entry's `size_bytes`).
    pub base_size: u64,
    /// Hugging Face repo that hosts the overlay file.
    pub repo: &'static str,
    /// Overlay file name: the canonical base file name plus
    /// `hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX`.
    pub file: &'static str,
    /// SHA-256 of the overlay file.
    pub sha256: &'static str,
    /// Size in bytes of the overlay file.
    pub size_bytes: u64,
}

/// Trunk overlays known to this client, keyed by exact base artifact identity.
pub const TRUNK_OVERLAY_PINS: &[TrunkOverlayPin] = &[
    // The grandfathered monolithic (a verified local install keeps its overlay).
    TrunkOverlayPin {
        base_sha256: FLASH_NEXT_XTS_SHA256,
        base_size: FLASH_NEXT_XTS_SIZE,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next.mq4-xts.trunk-overlay",
        sha256: "df49bc0aa601c2e26556261effd680f0be5d216804a1c5976a8a07ccd66840fd",
        size_bytes: 305_543_168,
    },
    // The split base (the overlay payload is re-bound to its tensor index).
    TrunkOverlayPin {
        base_sha256: FLASH_NEXT_XTS_SPLIT_BASE_SHA256,
        base_size: FLASH_NEXT_XTS_SPLIT_BASE_SIZE,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next-base.mq4-xts.trunk-overlay",
        sha256: "e3e352ca7fdcb0890fb0c6ddc07903cbbd9ceab65ad24ed5af0637e7dd92e4a6",
        size_bytes: 305_543_168,
    },
];

/// A resolved trunk overlay: the repo that hosts it and its sidecar identity.
#[derive(Clone, Debug, PartialEq)]
pub struct TrunkOverlay {
    pub repo: &'static str,
    pub sidecar: Sidecar,
}

/// The pin in `pins` whose base identity equals the entry's `sha256` and
/// `size_bytes`, as a [`TrunkOverlay`]. An entry without a pinned digest and
/// size never matches.
pub fn trunk_overlay_in(entry: &ModelEntry, pins: &[TrunkOverlayPin]) -> Option<TrunkOverlay> {
    let sha256 = entry.sha256.as_deref()?;
    let size_bytes = entry.size_bytes?;
    pins.iter()
        .find(|pin| pin.base_size == size_bytes && pin.base_sha256.eq_ignore_ascii_case(sha256))
        .map(|pin| TrunkOverlay {
            repo: pin.repo,
            sidecar: Sidecar {
                file: pin.file.to_owned(),
                sha256: Some(pin.sha256.to_owned()),
                size_bytes: Some(pin.size_bytes),
            },
        })
}

/// Content identity of one HC8 sidecar and of the exact base artifact it is
/// bound to.
///
/// The HC8 sidecar (`<base file>.hc8`) is an HFQM container of the base's
/// hyper-connection mixer weights requantized to per-row-scaled E4M3
/// (`hipfire_runtime::hc8_sidecar`); the Qwen4 loader applies it by default on
/// exact gfx1201 (`HIPFIRE_QWEN4_HC8=0` is the kill switch) and binds it to the
/// base's file length and index sha256. Like [`TrunkOverlayPin`], the pin lives
/// in the client rather than in a registry field, keyed by the base identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hc8SidecarPin {
    /// SHA-256 of the base artifact (the registry entry's `sha256`).
    pub base_sha256: &'static str,
    /// Size in bytes of the base artifact (the registry entry's `size_bytes`).
    pub base_size: u64,
    /// Hugging Face repo that hosts the sidecar file.
    pub repo: &'static str,
    /// Sidecar file name: the base file name plus
    /// `hipfire_config::memory::HC8_SIDECAR_SUFFIX`.
    pub file: &'static str,
    /// SHA-256 of the sidecar file.
    pub sha256: &'static str,
    /// Size in bytes of the sidecar file.
    pub size_bytes: u64,
}

/// HC8 sidecars known to this client, keyed by exact base artifact identity.
pub const HC8_SIDECAR_PINS: &[Hc8SidecarPin] = &[
    // The Flash-Next MQ4 XTS symmetric-trunk base.
    Hc8SidecarPin {
        base_sha256: "27dfff6692eba3385655959d134771fcff154c6279ab0a6c43cb8c7d81a7fa31",
        base_size: 70_888_270_360,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next-base.mq4-xts-symtrunk.hc8",
        sha256: "2adf32e564f09efbe185c7ebf57483dc58103f21341569e716173098f0e2ee67",
        size_bytes: 659_649_536,
    },
];

/// A resolved HC8 sidecar: the repo that hosts it and its sidecar identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Hc8Sidecar {
    pub repo: &'static str,
    pub sidecar: Sidecar,
}

/// The pin in `pins` whose base identity equals the entry's `sha256` and
/// `size_bytes`, as a [`Hc8Sidecar`]. An entry without a pinned digest and
/// size never matches.
pub fn hc8_sidecar_in(entry: &ModelEntry, pins: &[Hc8SidecarPin]) -> Option<Hc8Sidecar> {
    let sha256 = entry.sha256.as_deref()?;
    let size_bytes = entry.size_bytes?;
    pins.iter()
        .find(|pin| pin.base_size == size_bytes && pin.base_sha256.eq_ignore_ascii_case(sha256))
        .map(|pin| Hc8Sidecar {
            repo: pin.repo,
            sidecar: Sidecar {
                file: pin.file.to_owned(),
                sha256: Some(pin.sha256.to_owned()),
                size_bytes: Some(pin.size_bytes),
            },
        })
}

/// Content identity of one EP2 hot-unit loan sidecar (`<base file>.ep.hot-units`) and of the
/// exact base artifact and EP expert-ranking sidecar it was calibrated under.
///
/// The file is closed JSON (`hipfire.qwen4.ep_hot_units`); the Qwen4 EP2 loader binds it to the
/// base (`model_sha256`, `base_size`, `base_index_sha256`) and to the sha256 of the
/// `.ep.expert-hotset` it ran under (`base_ranking_sha256`), and declines the loan on any
/// mismatch. Like [`Hc8SidecarPin`], the pin lives in the client, keyed by the base identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HotUnitsSidecarPin {
    /// SHA-256 of the base artifact (the registry entry's `sha256`).
    pub base_sha256: &'static str,
    /// Size in bytes of the base artifact (the registry entry's `size_bytes`).
    pub base_size: u64,
    /// SHA-256 of the `.ep.expert-hotset` sidecar the calibration ran under (the file's
    /// `base_ranking_sha256`).
    pub ranking_sha256: &'static str,
    /// Hugging Face repo that will host the sidecar file.
    pub repo: &'static str,
    /// Sidecar file name: the base file name plus
    /// `hipfire_config::memory::EP_HOT_UNITS_SIDECAR_SUFFIX`.
    pub file: &'static str,
    /// SHA-256 of the sidecar file.
    pub sha256: &'static str,
    /// Size in bytes of the sidecar file.
    pub size_bytes: u64,
    /// Whether the file is on the Hugging Face repo. While `false`, `hipfire pull` never
    /// requests it (no 404, no warning); a local copy beside the model is still verified and
    /// honoured by the loader.
    pub published: bool,
}

/// Hot-unit loan sidecars known to this client, keyed by exact base artifact identity.
pub const HOT_UNITS_SIDECAR_PINS: &[HotUnitsSidecarPin] = &[
    // The Flash-Next MQ4 XTS symmetric-trunk base. Pinned ahead of the Hugging Face upload,
    // which happens at release (v0.4.5 is embargoed).
    HotUnitsSidecarPin {
        base_sha256: "27dfff6692eba3385655959d134771fcff154c6279ab0a6c43cb8c7d81a7fa31",
        base_size: 70_888_270_360,
        ranking_sha256: "c27a8ebff3d560e6c73d9b1e8b52fb27878a13735301660159109e857b0fb53c",
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.hot-units",
        sha256: "a50317cf2014f88d306d88366131212757d875c4c6e65572d659d361ec17db13",
        size_bytes: 67_526,
        published: false,
    },
];

/// A resolved hot-unit loan sidecar: the repo that hosts it, its sidecar identity and whether it
/// is published.
#[derive(Clone, Debug, PartialEq)]
pub struct HotUnitsSidecar {
    pub repo: &'static str,
    pub sidecar: Sidecar,
    /// The ranking sidecar sha256 the calibration is bound to.
    pub ranking_sha256: &'static str,
    pub published: bool,
}

/// The pin in `pins` whose base identity equals the entry's `sha256` and `size_bytes`, as a
/// [`HotUnitsSidecar`]. An entry without a pinned digest and size never matches.
pub fn hot_units_sidecar_in(
    entry: &ModelEntry,
    pins: &[HotUnitsSidecarPin],
) -> Option<HotUnitsSidecar> {
    let sha256 = entry.sha256.as_deref()?;
    let size_bytes = entry.size_bytes?;
    pins.iter()
        .find(|pin| pin.base_size == size_bytes && pin.base_sha256.eq_ignore_ascii_case(sha256))
        .map(|pin| HotUnitsSidecar {
            repo: pin.repo,
            sidecar: Sidecar {
                file: pin.file.to_owned(),
                sha256: Some(pin.sha256.to_owned()),
                size_bytes: Some(pin.size_bytes),
            },
            ranking_sha256: pin.ranking_sha256,
            published: pin.published,
        })
}

/// A replacement artifact this client pulls in place of a published one.
///
/// The published registry entry (repo, `file`, `sha256`, `size_bytes`) stays
/// untouched so older clients keep pulling the artifact they can run, and so
/// their strict `deny_unknown_fields` v1 parser keeps accepting the registry
/// (the same reason as [`TrunkOverlayPin`]). This client applies the pin at
/// parse time ([`RegistryV1::apply_artifact_upgrades`]): the entry takes the
/// replacement's `sha256` / `size_bytes`, keeps its tag and local file name,
/// and [`artifact_remote_file`] fetches the replacement from a versioned path
/// in the same repo, so the published file is never overwritten.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactUpgradePin {
    /// Registry tag the pin applies to.
    pub tag: &'static str,
    /// SHA-256 of the published artifact being replaced (the entry's `sha256`).
    pub base_sha256: &'static str,
    /// Size in bytes of the published artifact (the entry's `size_bytes`).
    pub base_size: u64,
    /// Path of the replacement inside the entry's repo.
    pub remote_file: &'static str,
    /// SHA-256 of the replacement.
    pub sha256: &'static str,
    /// Size in bytes of the replacement.
    pub size_bytes: u64,
}

/// Artifact replacements known to this client.
pub const ARTIFACT_UPGRADE_PINS: &[ArtifactUpgradePin] = &[
    // Raon-1-mini-preview: the attention-class projections MQ8G256 (FWHT256
    // int8, one scale per 256) in place of Q8F16; routed experts, shared /
    // dense FFN and lm_head byte-identical to the published file. Needs the
    // MQ8G256 Raon route, which builds before it lack, hence a client pin.
    ArtifactUpgradePin {
        tag: "raon-1-mini-preview",
        base_sha256: "e136b14c2ab7f31a8951c3ffb74439ca4299e9c835fe77fa1ed533f5dfe86688",
        base_size: 17_865_468_284,
        remote_file: "v2/raon-1-mini-preview.mq4-xts",
        sha256: "fa6d150bcb2af4600aad29aa96bb8fdaf894f0bf1f6bd706af17bc65c8f8690e",
        size_bytes: 17_772_056_444,
    },
];

/// The repo path to fetch `file` of `entry` from: an applied
/// [`ArtifactUpgradePin`]'s `remote_file` when `file` is the entry's own
/// artifact and the entry carries that pin's replacement identity, else `file`.
pub fn artifact_remote_file<'a>(entry: &ModelEntry, file: &'a str) -> &'a str {
    artifact_remote_file_in(entry, file, ARTIFACT_UPGRADE_PINS)
}

fn artifact_remote_file_in<'a>(
    entry: &ModelEntry,
    file: &'a str,
    pins: &'static [ArtifactUpgradePin],
) -> &'a str {
    if file != entry.file {
        return file;
    }
    let (Some(sha256), Some(size_bytes)) = (entry.sha256.as_deref(), entry.size_bytes) else {
        return file;
    };
    pins.iter()
        .find(|pin| pin.size_bytes == size_bytes && pin.sha256.eq_ignore_ascii_case(sha256))
        .map_or(file, |pin| pin.remote_file)
}

/// Content identity of one expert-hotset sidecar and of the exact base
/// artifact it is bound to.
///
/// The pin lives in the client instead of a registry field, like
/// [`TrunkOverlayPin`], so that older clients' strict `deny_unknown_fields` v1
/// parser keeps accepting the published registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertHotsetPin {
    /// SHA-256 of the base artifact (the registry entry's `sha256`).
    pub base_sha256: &'static str,
    /// Size in bytes of the base artifact (the registry entry's `size_bytes`).
    pub base_size: u64,
    /// Hugging Face repo that hosts the hotset file.
    pub repo: &'static str,
    /// Hotset file name: the canonical base file name plus `.expert-hotset`.
    pub file: &'static str,
    /// SHA-256 of the hotset file.
    pub sha256: &'static str,
    /// Size in bytes of the hotset file.
    pub size_bytes: u64,
}

/// Expert hotsets known to this client, keyed by exact base artifact identity.
pub const EXPERT_HOTSET_PINS: &[ExpertHotsetPin] = &[
    // The grandfathered monolithic (a verified local install keeps its hot set).
    ExpertHotsetPin {
        base_sha256: FLASH_NEXT_XTS_SHA256,
        base_size: FLASH_NEXT_XTS_SIZE,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next.mq4-xts.expert-hotset",
        sha256: "7f6760e11a86e4ffe0b05c050ce64e6eb772ab748b6b11c8b2e6dbc566e57086",
        size_bytes: 217_676,
    },
    // The split base (the hot set is re-bound to its tensor index).
    ExpertHotsetPin {
        base_sha256: FLASH_NEXT_XTS_SPLIT_BASE_SHA256,
        base_size: FLASH_NEXT_XTS_SPLIT_BASE_SIZE,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next-base.mq4-xts.expert-hotset",
        sha256: "911430746607f0a15f388594302304d1ce0283f1787e3f3d6aa32005d0a067a1",
        size_bytes: 217_675,
    },
];

/// EP2 expert-ranking sidecars, distinct from [`EXPERT_HOTSET_PINS`]: the name
/// is the canonical base file plus
/// `hipfire_config::memory::EP_EXPERT_HOTSET_SIDECAR_SUFFIX`. The split base and the
/// symmetric-trunk base are pinned; each ranking is bound to its base's tensor index.
pub const EP_EXPERT_HOTSET_PINS: &[ExpertHotsetPin] = &[
    ExpertHotsetPin {
        base_sha256: FLASH_NEXT_XTS_SPLIT_BASE_SHA256,
        base_size: FLASH_NEXT_XTS_SPLIT_BASE_SIZE,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next-base.mq4-xts.ep.expert-hotset",
        sha256: "77540499adecb1cbca1186051d24aaa329b986dddd52f6d950c887088e61d7b6",
        size_bytes: 312_036,
    },
    // The symmetric-trunk base (the ranking `HOT_UNITS_SIDECAR_PINS` is calibrated under).
    // Pinned ahead of the Hugging Face upload; see [`UNPUBLISHED_PIN_FILES`].
    ExpertHotsetPin {
        base_sha256: "27dfff6692eba3385655959d134771fcff154c6279ab0a6c43cb8c7d81a7fa31",
        base_size: 70_888_270_360,
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.expert-hotset",
        sha256: "c27a8ebff3d560e6c73d9b1e8b52fb27878a13735301660159109e857b0fb53c",
        size_bytes: 312_036,
    },
];

/// Pinned sidecar files that are not on Hugging Face yet (both upload at the release).
/// `hipfire pull` never requests them and never touches a local copy; a local copy beside the
/// model is still size- and sha256-checked against its pin and reported.
pub const UNPUBLISHED_PIN_FILES: &[&str] =
    &["qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.expert-hotset"];

/// A resolved expert hotset: the repo that hosts it and its sidecar identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertHotset {
    pub repo: &'static str,
    pub sidecar: Sidecar,
    /// Whether the file is on the repo (`false` for [`UNPUBLISHED_PIN_FILES`]).
    pub published: bool,
}

/// The pin in `pins` whose base identity equals the entry's `sha256` and
/// `size_bytes`, as an [`ExpertHotset`]. An entry without a pinned digest and
/// size never matches.
pub fn expert_hotset_in(entry: &ModelEntry, pins: &[ExpertHotsetPin]) -> Option<ExpertHotset> {
    let sha256 = entry.sha256.as_deref()?;
    let size_bytes = entry.size_bytes?;
    pins.iter()
        .find(|pin| pin.base_size == size_bytes && pin.base_sha256.eq_ignore_ascii_case(sha256))
        .map(|pin| ExpertHotset {
            repo: pin.repo,
            sidecar: Sidecar {
                file: pin.file.to_owned(),
                sha256: Some(pin.sha256.to_owned()),
                size_bytes: Some(pin.size_bytes),
            },
            published: !UNPUBLISHED_PIN_FILES.contains(&pin.file),
        })
}

/// [`expert_hotset_in`] over the EP2 ranking pins, `None` while `gate` is off.
pub fn ep_expert_hotset_in(
    entry: &ModelEntry,
    pins: &[ExpertHotsetPin],
    gate: bool,
) -> Option<ExpertHotset> {
    if !gate {
        return None;
    }
    expert_hotset_in(entry, pins)
}

/// A monolithic artifact a published split base was cut from, pinned in the
/// client so a verified local copy keeps serving the split tags without the
/// base download.
///
/// The published split base is a byte-exact `hipfire ple split` of the
/// monolithic: everything but the 128 PLE shards is byte-identical, and the
/// split base's PLE rows are bound to the monolithic's by source rows. A local
/// monolithic whose bytes verify against [`Self::sha256`] / [`Self::size_bytes`]
/// is therefore the same model and serves the split tags directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrandfatheredMonolithic {
    /// Hugging Face repo of both the split base and the monolithic.
    pub repo: &'static str,
    /// File name of the split base this monolithic serves.
    pub split_file: &'static str,
    /// SHA-256 of the split base (the registry entry's `sha256`).
    pub split_sha256: &'static str,
    /// Size in bytes of the split base (the registry entry's `size_bytes`).
    pub split_size_bytes: u64,
    /// Canonical monolithic file name.
    pub canonical_file: &'static str,
    /// Legacy monolithic file name (the same bytes under the old spelling).
    pub legacy_file: &'static str,
    /// SHA-256 of the monolithic.
    pub sha256: &'static str,
    /// Size in bytes of the monolithic.
    pub size_bytes: u64,
}

/// Monolithics the compiled client keeps serving under their split tags.
pub const GRANDFATHERED_MONOLITHICS: &[GrandfatheredMonolithic] = &[GrandfatheredMonolithic {
    repo: FLASH_NEXT_XTS_REPO,
    split_file: FLASH_NEXT_XTS_SPLIT_BASE_FILE,
    split_sha256: FLASH_NEXT_XTS_SPLIT_BASE_SHA256,
    split_size_bytes: FLASH_NEXT_XTS_SPLIT_BASE_SIZE,
    canonical_file: FLASH_NEXT_XTS_CANONICAL_FILE,
    legacy_file: FLASH_NEXT_XTS_LEGACY_FILE,
    sha256: FLASH_NEXT_XTS_SHA256,
    size_bytes: FLASH_NEXT_XTS_SIZE,
}];

/// The pin in `pins` whose split-base identity equals `entry` (repo, file,
/// `sha256`, `size_bytes`) while the entry publishes PLE slots. An entry
/// without a pinned digest and size never matches.
pub fn grandfathered_monolithic_in<'a>(
    entry: &ModelEntry,
    pins: &'a [GrandfatheredMonolithic],
) -> Option<&'a GrandfatheredMonolithic> {
    if !entry.has_ple_sidecars() {
        return None;
    }
    let sha256 = entry.sha256.as_deref()?;
    let size_bytes = entry.size_bytes?;
    pins.iter().find(|pin| {
        entry.repo == pin.repo
            && entry.file == pin.split_file
            && pin.split_size_bytes == size_bytes
            && pin.split_sha256.eq_ignore_ascii_case(sha256)
    })
}

/// A model the registry no longer publishes but this client still runs from a
/// verified local copy.
///
/// The pin is the retired entry's exact artifact identity (repo, file, SHA-256,
/// size) plus the card fields a load needs ([`RetiredModel::entry`]), compiled
/// into the client like [`GrandfatheredMonolithic`]. `hipfire pull` refuses a
/// retired tag and no resolver downloads one; a local file serves the tag, its
/// file name, or its installed path only after its bytes verify against
/// `sha256` / `size_bytes`. The Hugging Face files stay published for older
/// clients; this client never fetches them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetiredModel {
    /// The retired registry tag.
    pub tag: &'static str,
    /// Hugging Face repo that still hosts the file.
    pub repo: &'static str,
    /// Artifact file name in the models directory.
    pub file: &'static str,
    /// SHA-256 of the artifact.
    pub sha256: &'static str,
    /// Size in bytes of the artifact.
    pub size_bytes: u64,
    pub size_gb: f64,
    pub min_vram_gb: f64,
    pub desc: &'static str,
    pub arch_id: u32,
    pub quant: &'static str,
    /// Tag a user moves to.
    pub replacement: &'static str,
}

impl RetiredModel {
    /// The registry card the retired entry carried, so a verified local copy
    /// loads with the same identity, sampling defaults and sidecar pins (the
    /// sha256/size keyed [`TRUNK_OVERLAY_PINS`], [`HC8_SIDECAR_PINS`] and [`EXPERT_HOTSET_PINS`]) it
    /// had while published. It declares no sidecar slots.
    pub fn entry(&self) -> ModelEntry {
        ModelEntry {
            repo: self.repo.to_owned(),
            file: self.file.to_owned(),
            size_gb: self.size_gb,
            min_vram_gb: self.min_vram_gb,
            desc: self.desc.to_owned(),
            recommended_settings: Some(RecommendedSettings {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(20),
                ..RecommendedSettings::default()
            }),
            sha256: Some(self.sha256.to_owned()),
            size_bytes: Some(self.size_bytes),
            arch_id: Some(self.arch_id),
            quant: Some(self.quant.to_owned()),
            ..ModelEntry::default()
        }
    }

    /// [`RetiredModel::entry`] without rebuilding it per query: the compiled
    /// pins' cards are built once and borrowed. A pin that is not exactly one
    /// of [`RETIRED_MODELS`] (a test fixture) gets an owned card.
    pub fn card(&self) -> std::borrow::Cow<'static, ModelEntry> {
        static CARDS: std::sync::LazyLock<Vec<ModelEntry>> =
            std::sync::LazyLock::new(|| RETIRED_MODELS.iter().map(RetiredModel::entry).collect());
        match RETIRED_MODELS.iter().position(|pin| pin == self) {
            Some(index) => std::borrow::Cow::Borrowed(&CARDS[index]),
            None => std::borrow::Cow::Owned(self.entry()),
        }
    }

    /// The one-line refusal `hipfire pull` prints for this tag.
    pub fn refusal(&self) -> String {
        format!(
            "{} is retired; the file still runs if you have it; replacement {}",
            self.tag, self.replacement
        )
    }
}

/// Models the compiled client keeps running from a verified local copy after
/// the registry dropped them.
pub const RETIRED_MODELS: &[RetiredModel] = &[
    // 2026-09-28 to 2026-10-02: RTN asymmetric experts, MQ6G256V2 language head.
    RetiredModel {
        tag: "qwen3.8:flash-next-rtn-asym",
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next.mq4",
        sha256: "8aa01cf41bf2a90b319b9a1c70837baf51a2f92811af551f59b59d418518f650",
        size_bytes: 125_288_540_696,
        size_gb: 125.3,
        min_vram_gb: 128.0,
        desc: "Qwen3.8-Flash-Next / Qwen4 HFQM (arch 16), retired RTN-asymmetric MQ4 rung \
               (2026-09-28 to 2026-10-02): MQ6G256V2 trunk and language head, MQ4 routed \
               experts, Q8F16 PLE rows embedded. Replaced by qwen3.8:flash-next.",
        arch_id: 16,
        quant: "mq4",
        replacement: "qwen3.8:flash-next",
    },
    // 2026-09-22 to 2026-09-28: the same rung with a Q8F16 language head.
    RetiredModel {
        tag: "qwen3.8:flash-next-mq6q8-pleq8",
        repo: FLASH_NEXT_XTS_REPO,
        file: "qwen3.8-flash-next.mq6q8-pleq8",
        sha256: "c0628b848077f02afed9ce5a4daa0598c379aa1d1e5ca0c0b24ddd04b49773a9",
        size_bytes: 125_467_331_096,
        size_gb: 125.5,
        min_vram_gb: 128.0,
        desc: "Qwen3.8-Flash-Next / Qwen4 HFQM (arch 16), retired MQ6/Q8 rung (2026-09-22 to \
               2026-09-28): MQ6G256V2 trunk, Q8F16 language head, MQ4 routed experts, Q8F16 PLE \
               rows embedded. Replaced by qwen3.8:flash-next.",
        arch_id: 16,
        quant: "mq6q8-pleq8",
        replacement: "qwen3.8:flash-next",
    },
];

/// The pin in `pins` named by `input`: its exact tag or its bare file name. A
/// path never matches (callers compare canonical installed paths instead).
pub fn retired_model_named<'a>(input: &str, pins: &'a [RetiredModel]) -> Option<&'a RetiredModel> {
    if input.contains('/') || input.contains('\\') {
        return None;
    }
    pins.iter()
        .find(|pin| input == pin.tag || input == pin.file)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecommendedSettings {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Parent-model reasoning effort. This is prompt semantics, independent of
    /// any explicit token cap selected by `thinking_budget`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Optional named cap policy for reasoning tokens
    /// (`off|low|med|high|xhigh|max|uncapped`). Absence means uncapped.
    /// Effort-native families (Qwen3.8, DeepSeek4, Muse Glimmer, Ornith) omit this field.
    #[serde(default)]
    pub thinking_budget: Option<String>,
}

impl RecommendedSettings {
    pub fn config_layer(&self) -> std::result::Result<ConfigLayer, String> {
        let mut layer = ConfigLayer::default();
        if let Some(value) = self.temperature {
            layer
                .set("generation.temperature", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.top_p {
            layer
                .set("generation.top_p", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.top_k {
            let value = i64::try_from(value).map_err(|_| "top_k is too large".to_owned())?;
            layer
                .set("generation.top_k", ConfigValue::Integer(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.min_p {
            layer
                .set("generation.min_p", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.presence_penalty {
            layer
                .set("generation.presence_penalty", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.repeat_penalty {
            layer
                .set("generation.repeat_penalty", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.system_prompt {
            layer
                .set("prompt.system", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.reasoning_effort {
            layer
                .set("reasoning.effort", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.thinking_budget {
            layer
                .set("reasoning.budget", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        Ok(layer)
    }
}

/// Per-mode sampling profiles for a model, mirroring the model card's
/// documented modes. Each is a full [`RecommendedSettings`] blob, including
/// optional reasoning effort and optional legacy budget policy. `general` is the thinking-mode
/// default (equals the entry's `recommended_settings`); `coding` is the precise
/// thinking-coding profile; `instruct` is the non-thinking profile. Profiles
/// are entry-level metadata and are selected client-side (e.g. serve_harness
/// `--sampling registry:<profile>`), lowering through the typed config/request
/// path rather than architecture-specific environment variables.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingProfiles {
    #[serde(default)]
    pub general: Option<RecommendedSettings>,
    #[serde(default)]
    pub coding: Option<RecommendedSettings>,
    #[serde(default)]
    pub instruct: Option<RecommendedSettings>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub repo: String,
    pub file: String,
    pub size_gb: f64,
    pub min_vram_gb: f64,
    pub desc: String,
    #[serde(default)]
    pub triattn: Option<Sidecar>,
    #[serde(default)]
    pub mtp: Option<Sidecar>,
    #[serde(default)]
    pub dspark: Option<Sidecar>,
    // Image-generation component sidecars (arch 40+ trunks, see
    // docs/architecture-ids.md — ids provisional until the HFQM pack ships).
    // `t5` is the T5-XXL text-encoder component (sidecar arch 41), `clip` the
    // CLIP-L pooled-text encoder (arch 42), `vae` the VAE decoder (arch 43).
    // Shared across FLUX.1 entries; a FLUX.2 Klein entry uses `qwen3` (its
    // Qwen3 text encoder, arch 45) plus the shared `vae` instead of
    // `t5`/`clip`.
    #[serde(default)]
    pub t5: Option<Sidecar>,
    #[serde(default)]
    pub clip: Option<Sidecar>,
    #[serde(default)]
    pub qwen3: Option<Sidecar>,
    #[serde(default)]
    pub vae: Option<Sidecar>,
    #[serde(default)]
    pub dflash: Option<Sidecar>,
    /// Opt-in gfx1151 XDNA NPU spillover sidecar (`.xdna.zip` with manifest
    /// v1). Capability data only: `kernel.npu_spillover` must also be on, the
    /// archive must verify, and the host arch must be exactly `gfx1151`.
    /// No certified entries declare one yet; the first certified model binds
    /// its payload hashes here via `sidecar.sha256`.
    #[serde(default)]
    pub xdna: Option<Sidecar>,
    /// Shared Qwen3.8-27B vision-tower sidecar (`qwen3.8-27b-vision.hfq`,
    /// llm.cpp mmproj-style). Every `qwen3.8:27b*` tier declares the same
    /// file so each text quant tier serves images without requantizing the
    /// trunk; sha256/size_bytes stay absent until the pack ships.
    #[serde(default)]
    pub vision: Option<Sidecar>,
    /// PLE n-gram row sidecars of a Flash-Next split base, keyed by format
    /// token (lowercase `[a-z0-9]+`: `q4` = 4-bit Lloyd-Max rows, `q8` = exact
    /// Q8F16 rows; future formats such as `q2` parse and validate too). Only
    /// valid on an entry whose `file` is a split base (`<name>-base.<ext>`);
    /// the file name is `<file>.ple-<key>`, or, for the Flash-Next family
    /// only, the shared `qwen3.8-flash-next.ple-<key>` pinned identically by
    /// every entry that names it ([`shared_ple_file`]). A client plans only
    /// the formats it can load and ignores the rest.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ple: BTreeMap<String, Sidecar>,
    /// Alternative `lm_head` carriers, keyed by short name (`q4k`, `bf16`).
    ///
    /// Each is a single-tensor `.hfq` from `hipfire-quantize --head-only` that
    /// shadows the base's `lm_head.weight` at load time. The BASE ships the
    /// recommended head and runs standalone; these only exist so a different
    /// carrier costs a 188-635 MB download instead of a near-identical 6.5 GB
    /// model. Three full variants would be 19.63 GB; base plus two overlays is
    /// 7.30 GB.
    #[serde(default)]
    pub heads: std::collections::BTreeMap<String, Sidecar>,
    #[serde(default)]
    pub default_tool_format: Option<String>,
    #[serde(default)]
    pub default_kv_mode: Option<String>,
    #[serde(default)]
    pub quant_recipe: Option<String>,
    #[serde(default)]
    pub sampling: Option<SamplingDefaults>,
    #[serde(default)]
    pub recommended_settings: Option<RecommendedSettings>,
    #[serde(default)]
    pub sampling_profiles: Option<SamplingProfiles>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub arch_id: Option<u32>,
    #[serde(default)]
    pub quant: Option<String>,
}

impl ModelEntry {
    /// Exact XTS artifact identity, including the two admitted filename spellings.
    /// This selection has no filesystem side effects; CLI migration verifies bytes.
    pub fn is_flash_next_xts(&self) -> bool {
        self.repo == FLASH_NEXT_XTS_REPO
            && self.size_bytes == Some(FLASH_NEXT_XTS_SIZE)
            && self.sha256.as_deref() == Some(FLASH_NEXT_XTS_SHA256)
            && matches!(
                self.file.as_str(),
                FLASH_NEXT_XTS_LEGACY_FILE | FLASH_NEXT_XTS_CANONICAL_FILE
            )
    }

    /// The pinned monolithic this split base was cut from, from the compiled-in
    /// [`GRANDFATHERED_MONOLITHICS`]; `Some` only for the published MQ4 XTS
    /// split base. A verified local copy of it serves this entry's tags without
    /// the base download. `None` for every other entry.
    pub fn grandfathered_monolithic(&self) -> Option<&'static GrandfatheredMonolithic> {
        grandfathered_monolithic_in(self, GRANDFATHERED_MONOLITHICS)
    }

    /// The Flash-Next trunk overlay sidecar of this entry's base artifact, from
    /// the compiled-in [`TRUNK_OVERLAY_PINS`]; `None` for every other entry.
    pub fn trunk_overlay(&self) -> Option<TrunkOverlay> {
        trunk_overlay_in(self, TRUNK_OVERLAY_PINS)
    }

    /// The Flash-Next HC8 sidecar of this entry's base artifact, from the
    /// compiled-in [`HC8_SIDECAR_PINS`]; `None` for every other entry.
    pub fn hc8_sidecar(&self) -> Option<Hc8Sidecar> {
        hc8_sidecar_in(self, HC8_SIDECAR_PINS)
    }

    /// The Flash-Next hot-unit loan sidecar of this entry's base artifact, from the
    /// compiled-in [`HOT_UNITS_SIDECAR_PINS`]; `None` for every other entry.
    pub fn hot_units_sidecar(&self) -> Option<HotUnitsSidecar> {
        hot_units_sidecar_in(self, HOT_UNITS_SIDECAR_PINS)
    }

    /// The Flash-Next expert-hotset sidecar of this entry's base artifact, from
    /// the compiled-in [`EXPERT_HOTSET_PINS`]; `None` for every other entry.
    pub fn expert_hotset(&self) -> Option<ExpertHotset> {
        expert_hotset_in(self, EXPERT_HOTSET_PINS)
    }

    /// The EP2 expert-ranking sidecar of this entry's base artifact; `None`
    /// while `hipfire_config::memory::EP_EXPERT_HOTSET_DEFAULT_ON` is off.
    pub fn ep_expert_hotset(&self) -> Option<ExpertHotset> {
        ep_expert_hotset_in(
            self,
            EP_EXPERT_HOTSET_PINS,
            hipfire_config::memory::EP_EXPERT_HOTSET_DEFAULT_ON,
        )
    }

    /// True when this entry is a split base that publishes PLE row sidecars.
    pub fn has_ple_sidecars(&self) -> bool {
        !self.ple.is_empty()
    }

    /// The PLE row sidecar of format `key` (`q4`, `q8`, ...), if published.
    pub fn ple_sidecar(&self, key: &str) -> Option<&Sidecar> {
        self.ple.get(key)
    }

    /// True when this split base is published split-only: there is no
    /// monolithic artifact it is a view of, so no monolithic can be
    /// grandfathered for it.
    /// Derived from the entry's shape, not a flag: every published PLE sidecar
    /// names the Flash-Next family's shared rows-bound file
    /// ([`shared_ple_file`]), and the name without `-base` is not the
    /// published MQ4 XTS monolithic, whose split views must keep a compiled-in
    /// pin ([`ModelEntry::grandfathered_monolithic`]). A split base naming its
    /// own `<file>.ple-<key>` rows is a view of a monolithic.
    pub fn is_split_only(&self) -> bool {
        self.has_ple_sidecars()
            && monolithic_file_for_split_base(&self.file)
                .is_some_and(|monolithic| monolithic != FLASH_NEXT_XTS_CANONICAL_FILE)
            && self.ple.iter().all(|(key, sidecar)| {
                shared_ple_file(&self.file, key).as_deref() == Some(sidecar.file.as_str())
            })
    }

    /// Resolve a named sampling profile (`general` | `coding` | `instruct`).
    /// `general` falls back to `recommended_settings` (the default profile)
    /// when no explicit profile map is present. Unknown names return `None`.
    pub fn sampling_profile(&self, name: &str) -> Option<&RecommendedSettings> {
        let profiles = self.sampling_profiles.as_ref();
        match name {
            "general" => profiles
                .and_then(|p| p.general.as_ref())
                .or(self.recommended_settings.as_ref()),
            "coding" => profiles.and_then(|p| p.coding.as_ref()),
            "instruct" => profiles.and_then(|p| p.instruct.as_ref()),
            _ => None,
        }
    }

    /// Lower this entry's load/sampling defaults into a sparse config layer.
    ///
    /// Starts from [`RecommendedSettings::config_layer`] when present (sampling
    /// and reasoning only), then overlays model-specific load defaults onto
    /// their canonical keys. `default_kv_mode = "q8"` is the universal fallback,
    /// not a model-specific opinion, so it is left for `auto` resolution rather
    /// than lowered here. Callers merge this under `RegistryModel` precedence so
    /// global/model/one-shot user config still wins.
    pub fn config_layer(&self) -> std::result::Result<ConfigLayer, String> {
        let mut layer = match &self.recommended_settings {
            Some(settings) => settings.config_layer()?,
            None => ConfigLayer::default(),
        };
        if let Some(mode) = &self.default_kv_mode {
            if mode != "q8" {
                layer
                    .set("memory.kv_cache", ConfigValue::String(mode.clone()))
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(layer)
    }
}

/// Registry tags (besides the Qwen3.8-27B tiers) whose declared `dflash`
/// sidecar lowers `speculation.dflash = "auto"`.
const RAON_DFLASH_AUTO_TAG: &str = "raon-1-mini-preview";

/// Tag-aware registry config layer. Starts from [`ModelEntry::config_layer`]
/// then applies automatic load policy by canonical tag. This is the single
/// canonical place for tag policy so validation and CLI stay in sync.
///
/// Policy (static; no registry/v1 wire fields): Qwen3.5/3.6/3.8 and
/// DeepSeek V4 Flash tags supply generation max_tokens only — never
/// `memory.max_seq` (resolved at load time) and never a KV backend.
/// Backend selection is source-independent at admission. Qwen3.8 entries
/// with `arch_id=16` (Qwen4) and Muse Glimmer tags write neither length.
///
/// Qwen3.8-27B trunk tiers (`qwen3.8:27b` and `qwen3.8:27b-*`, never a
/// longer lookalike such as `27boops`) that declare a paired `dflash` sidecar
/// also lower `speculation.dflash = "auto"`. The load resolver wires the draft
/// only when that registry-declared file is present; without it MTP `auto`
/// stays in effect. The schema/BuiltIn default stays `off` for every other
/// model; `arch_id=16`, draft/dflash sidecars, and tiers with no `dflash`
/// declaration get no speculation policy.
///
/// `raon-1-mini-preview` (arch 19) that declares its DFlash2 `dflash` sidecar
/// lowers the same `auto`: the Raon carrier builds the DFlash2 speculator when
/// the pulled draft is wired and runs plain AR (Raon has no MTP) when it is not;
/// `speculation.dflash = off` opts out.
///
/// Explicit global/model/one-shot user config (including `dflash=off`) remains
/// higher precedence than this registry layer.
pub fn config_layer_for_tag(
    tag: &str,
    entry: &ModelEntry,
) -> std::result::Result<ConfigLayer, String> {
    let mut layer = entry.config_layer()?;
    let family = tag.split(':').next().unwrap_or(tag);
    let is_sidecar = tag.contains("draft") || tag.contains("dflash");
    let is_qwen_tag_policy = matches!(family, "qwen3.5" | "qwen3.6" | "qwen3.8")
        && !is_sidecar
        && entry.arch_id != Some(16);
    if is_qwen_tag_policy {
        layer
            .set("generation.max_tokens", ConfigValue::Integer(81920))
            .map_err(|error| error.to_string())?;
    }
    let is_qwen38_27b_tier = tag == "qwen3.8:27b" || tag.starts_with("qwen3.8:27b-");
    if is_qwen_tag_policy && is_qwen38_27b_tier && entry.dflash.is_some() {
        layer
            .set("speculation.dflash", ConfigValue::String("auto".into()))
            .map_err(|error| error.to_string())?;
    }
    // Raon-1-mini-preview: the explicit allow-list entry for the arch-19 target
    // that declares its DFlash2 sidecar (`raon-1-mini-preview:draft` and any
    // other draft/dflash tag never match).
    if tag == RAON_DFLASH_AUTO_TAG && !is_sidecar && entry.dflash.is_some() {
        layer
            .set("speculation.dflash", ConfigValue::String("auto".into()))
            .map_err(|error| error.to_string())?;
    }
    let is_deepseek_tag_policy =
        matches!(family, "deepseek-v4-flash" | "deepseek-v4-flash-preview") && !is_sidecar;
    if is_deepseek_tag_policy {
        layer
            .set("generation.max_tokens", ConfigValue::Integer(393216))
            .map_err(|error| error.to_string())?;
    }
    Ok(layer)
}

/// Result of [`resolve_for_model`].
#[derive(Clone, Debug)]
pub struct ResolvedForModel {
    /// Layers merged built-in → registry card → global → model → env.
    pub resolved: ResolvedConfig,
    /// `config.toml` warnings (retired keys the load ignored). Empty unless the
    /// global config is TOML; legacy-JSON migration notes stay with `config`.
    pub warnings: Vec<String>,
    /// `models.toml` identity whose overrides matched the model, if any.
    pub model_id: Option<String>,
}

/// Resolve the effective configuration for one model.
///
/// Layer order, lowest first: registry card defaults (`card` is the registry
/// tag with its entry), global `config.toml`, the `models.toml` entry (found by
/// `model_name`, then the card's tag, then the card's file), and `env` when it
/// holds any value.
pub fn resolve_for_model(
    paths: &ConfigPaths,
    model_name: &str,
    card: Option<(&str, &ModelEntry)>,
    env: ConfigLayer,
) -> Result<ResolvedForModel> {
    let loaded = load_global(paths)?;
    let warnings = if loaded.format == ConfigFormat::Toml {
        loaded.warnings
    } else {
        Vec::new()
    };
    let mut layers = Vec::new();
    if let Some((tag, entry)) = card {
        layers.push(NamedLayer {
            source: ConfigSource::RegistryModel {
                tag: tag.to_owned(),
                revision: "v1".into(),
            },
            layer: config_layer_for_tag(tag, entry).map_err(RegistryError::ModelDefaults)?,
        });
    }
    layers.push(NamedLayer {
        source: ConfigSource::GlobalUser { path: loaded.path },
        layer: loaded.layer,
    });
    let catalog = load_catalog(paths)?;
    let model_override = catalog
        .catalog
        .model(model_name)
        .or_else(|| card.and_then(|(tag, _)| catalog.catalog.model(tag)))
        .or_else(|| card.and_then(|(_, entry)| catalog.catalog.model(entry.file.as_str())));
    let mut model_id = None;
    if let Some((id, model)) = model_override {
        model_id = Some(id.to_owned());
        if !model.overrides.values.is_empty() {
            layers.push(NamedLayer {
                source: ConfigSource::ModelUser {
                    model: id.to_owned(),
                    path: catalog.path.clone(),
                },
                layer: model.overrides.clone(),
            });
        }
    }
    if !env.values.is_empty() {
        layers.push(NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: env,
        });
    }
    Ok(ResolvedForModel {
        resolved: resolve(layers)?,
        warnings,
        model_id,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryV1 {
    pub schema_version: u32,
    pub generated_at: String,
    #[serde(default)]
    pub _comment: Option<String>,
    pub models: BTreeMap<String, ModelEntry>,
    pub aliases: BTreeMap<String, String>,
}

impl RegistryV1 {
    pub fn parse(raw: &str, source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        let mut registry: Self =
            serde_json::from_str(raw).map_err(|error| RegistryError::Parse {
                source_name: source_name.clone(),
                message: error.to_string(),
            })?;
        registry.validate(&source_name)?;
        registry.normalize_flash_next_xts();
        registry.apply_artifact_upgrades(ARTIFACT_UPGRADE_PINS);
        registry.strip_retired(RETIRED_MODELS);
        // An alias is non-authoritative convenience data. Match v1 behavior by
        // dropping dangling redirects instead of rejecting an otherwise valid
        // registry.
        registry
            .aliases
            .retain(|_, target| registry.models.contains_key(target));
        Ok(registry)
    }

    /// Drop the retired tags in `pins` from the available catalog, whatever
    /// delivered them (an older remote payload, a cache written before the
    /// retirement). The compiled pins, not the delivered entries, define a
    /// retired model; the stripped tag stays runnable from a verified local
    /// copy through [`RETIRED_MODELS`] but is never offered or pulled.
    pub fn strip_retired(&mut self, pins: &[RetiredModel]) {
        self.models
            .retain(|tag, _| !pins.iter().any(|pin| pin.tag == tag));
    }

    /// Point every entry whose artifact identity (`sha256`, `size_bytes`)
    /// equals a pin's base identity at the pin's replacement: the entry's
    /// `sha256` and `size_bytes` become the replacement's, and
    /// [`artifact_remote_file`] fetches it from the pin's repo path. The local
    /// file name (`file`) and the tag are unchanged, so an installed copy of
    /// the old artifact fails verification and is refreshed by the next pull.
    pub fn apply_artifact_upgrades(&mut self, pins: &[ArtifactUpgradePin]) {
        for (tag, entry) in self.models.iter_mut() {
            let (Some(sha256), Some(size_bytes)) = (entry.sha256.as_deref(), entry.size_bytes)
            else {
                continue;
            };
            if let Some(pin) = pins.iter().find(|pin| {
                pin.tag == tag
                    && pin.base_size == size_bytes
                    && pin.base_sha256.eq_ignore_ascii_case(sha256)
            }) {
                entry.sha256 = Some(pin.sha256.to_owned());
                entry.size_bytes = Some(pin.size_bytes);
            }
        }
    }

    fn normalize_flash_next_xts(&mut self) {
        for entry in self.models.values_mut() {
            if entry.is_flash_next_xts() && entry.file != FLASH_NEXT_XTS_CANONICAL_FILE {
                entry.file.clear();
                entry.file.push_str(FLASH_NEXT_XTS_CANONICAL_FILE);
            }
        }
    }

    pub fn validate(&self, source_name: &str) -> Result<()> {
        let fail = |message: String| RegistryError::Invalid {
            source_name: source_name.to_owned(),
            message,
        };
        if self.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err(fail(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        validate_generated_at(&self.generated_at).map_err(fail)?;
        if self.models.is_empty() {
            return Err(fail("model catalog is empty".into()));
        }
        let mut shared_pins: BTreeMap<String, (Option<&str>, Option<u64>)> = BTreeMap::new();
        for (tag, entry) in &self.models {
            if tag.trim().is_empty() || entry.file.trim().is_empty() {
                return Err(fail(format!("model '{tag}' has an empty tag or file")));
            }
            if !entry.size_gb.is_finite()
                || entry.size_gb < 0.0
                || !entry.min_vram_gb.is_finite()
                || entry.min_vram_gb < 0.0
            {
                return Err(fail(format!("model '{tag}' has invalid size metadata")));
            }
            validate_digest(entry.sha256.as_deref(), tag).map_err(fail)?;
            for sidecar in [
                &entry.triattn,
                &entry.mtp,
                &entry.dspark,
                &entry.dflash,
                &entry.xdna,
                &entry.vision,
                &entry.t5,
                &entry.clip,
                &entry.qwen3,
                &entry.vae,
            ]
            .into_iter()
            .flatten()
            .chain(entry.heads.values())
            .chain(entry.ple.values())
            {
                if sidecar.file.trim().is_empty() {
                    return Err(fail(format!("model '{tag}' has an empty sidecar file")));
                }
                validate_digest(sidecar.sha256.as_deref(), tag).map_err(fail)?;
            }
            for (key, sidecar) in &entry.ple {
                let slot = format!("ple.{key}");
                if !is_ple_format_token(key) {
                    return Err(fail(format!(
                        "model '{tag}' has invalid PLE format key '{key}' (expected lowercase \
                         letters and digits, e.g. q4)"
                    )));
                }
                if !is_split_base_file(&entry.file) {
                    return Err(fail(format!(
                        "model '{tag}' declares {slot} but its file '{}' is not a split base \
                         (expected <name>-base.<ext>)",
                        entry.file
                    )));
                }
                let own = format!("{}{}", entry.file, ple_sidecar_suffix(key));
                if sidecar.file == own {
                    continue;
                }
                let Some(shared) =
                    shared_ple_file(&entry.file, key).filter(|shared| *shared == sidecar.file)
                else {
                    return Err(fail(format!(
                        "model '{tag}' {slot} file '{}' must be '{own}'",
                        sidecar.file
                    )));
                };
                // Every split base of the family shares one file, so each one
                // must pin the same bytes.
                let pin = (sidecar.sha256.as_deref(), sidecar.size_bytes);
                match shared_pins.get(&shared) {
                    Some(first) if *first != pin => {
                        return Err(fail(format!(
                            "model '{tag}' {slot} shared file '{shared}' is pinned differently \
                             by another model"
                        )));
                    }
                    Some(_) => {}
                    None => {
                        shared_pins.insert(shared, pin);
                    }
                }
            }
            if let Some(format) = entry.default_tool_format.as_deref() {
                if !matches!(format, "hermes" | "qwen_xml") {
                    return Err(fail(format!(
                        "model '{tag}' has invalid default_tool_format '{format}'"
                    )));
                }
            }
            if let Some(settings) = &entry.recommended_settings {
                validate_recommendations(tag, settings).map_err(fail)?;
            }
            config_layer_for_tag(tag, entry)
                .map_err(|error| fail(format!("model '{tag}': {error}")))?;
            if let Some(profiles) = &entry.sampling_profiles {
                for (name, settings) in [
                    ("general", &profiles.general),
                    ("coding", &profiles.coding),
                    ("instruct", &profiles.instruct),
                ] {
                    if let Some(settings) = settings {
                        validate_recommendations(tag, settings).map_err(|error| {
                            fail(format!("model '{tag}' profile '{name}': {error}"))
                        })?;
                        settings.config_layer().map_err(|error| {
                            fail(format!("model '{tag}' profile '{name}': {error}"))
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn resolve_tag(&self, input: &str) -> String {
        let normalized = input
            .replace("-hfq4", "-hf4")
            .replace("-hfq6", "-hf6")
            .strip_suffix(".hfq")
            .map(|prefix| format!("{prefix}.hf4"))
            .unwrap_or_else(|| input.replace("-hfq4", "-hf4").replace("-hfq6", "-hf6"));
        if self.models.contains_key(&normalized) {
            return normalized;
        }
        if let Some(tag) = self.aliases.get(&normalized) {
            return tag.clone();
        }
        let qwen = format!("qwen3.5:{normalized}");
        if self.models.contains_key(&qwen) {
            return qwen;
        }
        // A bare file name matches its entry. A path never does: matching on
        // `file_name()` let a lookalike file outside the models directory
        // inherit the installed artifact's identity (sidecars, kv/max_seq
        // policy, rm targets). Callers that know the models directory resolve
        // paths with canonical comparison instead (`registry_entry_for_path`
        // in hipfire-cli); a bare file name carries no directory to confuse.
        self.models
            .iter()
            .find_map(|(tag, entry)| {
                (entry.file == normalized
                    || entry.file == input
                    || (input == FLASH_NEXT_XTS_LEGACY_FILE && entry.is_flash_next_xts()))
                .then(|| tag.clone())
            })
            .unwrap_or(normalized)
    }

    /// Exact bare filename match for callers that already established the input
    /// is the installed artifact (e.g. via canonical path comparison against
    /// the models directory). The pinned XTS legacy basename also matches its
    /// normalized entry. Unlike [`RegistryV1::model`], no tag/alias normalization
    /// is applied.
    pub fn entry_for_file(&self, file_name: &str) -> Option<(&str, &ModelEntry)> {
        self.models
            .iter()
            .find(|(_, entry)| {
                entry.file == file_name
                    || (file_name == FLASH_NEXT_XTS_LEGACY_FILE && entry.is_flash_next_xts())
            })
            .map(|(tag, entry)| (tag.as_str(), entry))
    }

    pub fn model(&self, input: &str) -> Option<(&str, &ModelEntry)> {
        let tag = self.resolve_tag(input);
        self.models
            .get_key_value(&tag)
            .map(|(tag, entry)| (tag.as_str(), entry))
    }
}

fn validate_digest(digest: Option<&str>, label: &str) -> std::result::Result<(), String> {
    if let Some(digest) = digest {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("'{label}' has an invalid SHA-256"));
        }
    }
    Ok(())
}

fn validate_generated_at(ts: &str) -> std::result::Result<(), String> {
    // Strict normalized RFC3339 UTC: YYYY-MM-DDTHH:MM:SSZ (20 bytes).
    // Lexical order equals chronological order only in this normalized form,
    // which is what `prefer_bundled_if_newer` relies on. Reject any
    // non-normalized representation (offsets, fractional seconds, whitespace,
    // lowercase, etc.) and validate calendar ranges so malformed strings
    // cannot invert precedence via lexical comparison.
    if ts.len() != 20 {
        return Err(format!(
            "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
            ts
        ));
    }
    let b = ts.as_bytes();
    if b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return Err(format!(
            "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
            ts
        ));
    }
    for i in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !b[i].is_ascii_digit() {
            return Err(format!(
                "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
                ts
            ));
        }
    }
    let year = (b[0] - b'0') as u16 * 1000
        + (b[1] - b'0') as u16 * 100
        + (b[2] - b'0') as u16 * 10
        + (b[3] - b'0') as u16;
    let month = (b[5] - b'0') * 10 + (b[6] - b'0');
    let day = (b[8] - b'0') * 10 + (b[9] - b'0');
    let hour = (b[11] - b'0') * 10 + (b[12] - b'0');
    let minute = (b[14] - b'0') * 10 + (b[15] - b'0');
    let second = (b[17] - b'0') * 10 + (b[18] - b'0');
    if month == 0 || month > 12 {
        return Err(format!("generated_at '{}' has invalid month", ts));
    }
    if day == 0 || day > days_in_month(year, month) {
        return Err(format!("generated_at '{}' has invalid day", ts));
    }
    if hour > 23 {
        return Err(format!("generated_at '{}' has invalid hour", ts));
    }
    if minute > 59 {
        return Err(format!("generated_at '{}' has invalid minute", ts));
    }
    if second > 59 {
        return Err(format!("generated_at '{}' has invalid second", ts));
    }
    Ok(())
}

fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn is_leap_year(year: u16) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn is_effort_native_tag(tag: &str) -> bool {
    // Mirrors scripts/registry_gen.py:_effort_native_tag and the family/tag
    // conventions already used by config_layer_for_tag. Effort-native families
    // (Qwen3.8, DeepSeek V4 Flash/preview, Muse Glimmer product SKUs, Ornith
    // 1.5/legacy) have no registry thinking_budget; absence means uncapped.
    // Draft/dflash sidecars are excluded.
    let base = tag.split(" sampling_profiles.").next().unwrap_or(tag);
    if base.contains("draft") || base.contains("dflash") {
        return false;
    }
    let family = base.split(':').next().unwrap_or(base);
    if family == "qwen3.8" {
        return true;
    }
    if matches!(family, "deepseek-v4-flash" | "deepseek-v4-flash-preview") {
        return true;
    }
    if base == "muse-glimmer" || base == "muse-glimmer:fast" {
        return true;
    }
    if matches!(family, "ornith-1.5" | "ornith1.5" | "ornith") {
        return true;
    }
    false
}

fn validate_recommendations(
    tag: &str,
    value: &RecommendedSettings,
) -> std::result::Result<(), String> {
    let ranged = |name: &str, value: Option<f64>, min: f64, max: f64| {
        if value.is_some_and(|value| !value.is_finite() || value < min || value > max) {
            Err(format!("model '{tag}' has invalid {name}"))
        } else {
            Ok(())
        }
    };
    ranged("temperature", value.temperature, 0.0, 2.0)?;
    ranged("top_p", value.top_p, 0.0, 1.0)?;
    ranged("min_p", value.min_p, 0.0, 1.0)?;
    ranged("presence_penalty", value.presence_penalty, 0.0, 2.0)?;
    ranged("repeat_penalty", value.repeat_penalty, 0.5, 2.0)?;
    if value
        .top_k
        .is_some_and(|value| value == 0 || value > 100_000)
    {
        return Err(format!("model '{tag}' has invalid top_k"));
    }
    const REASONING_EFFORTS: &[&str] = &["auto", "none", "low", "medium", "high", "xhigh", "max"];
    const THINKING_BUDGETS: &[&str] = &["off", "low", "med", "high", "xhigh", "max", "uncapped"];
    if let Some(effort) = &value.reasoning_effort {
        if !REASONING_EFFORTS.contains(&effort.as_str()) {
            return Err(format!(
                "model '{tag}' has invalid reasoning_effort '{effort}'"
            ));
        }
    }
    if let Some(budget) = &value.thinking_budget {
        if is_effort_native_tag(tag) {
            return Err(format!(
                "model '{tag}' has unsupported thinking_budget '{budget}' on effort-native model (omit the field; absence means uncapped)"
            ));
        }
        if !THINKING_BUDGETS.contains(&budget.as_str()) {
            return Err(format!(
                "model '{tag}' has invalid thinking_budget '{budget}'"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegistrySource {
    Cache,
    Network,
    StaleCache,
    Bundled,
}

#[derive(Clone, Debug)]
pub struct LoadedRegistry {
    pub registry: RegistryV1,
    pub source: RegistrySource,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RegistryPaths {
    pub cache: PathBuf,
}

impl RegistryPaths {
    /// Infallible form of [`RegistryPaths::try_discover`]; panics with the
    /// actionable message when no usable Windows home exists.
    pub fn discover() -> Self {
        Self::try_discover().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_discover() -> std::result::Result<Self, hipfire_config::ConfigError> {
        Ok(Self {
            cache: hipfire_config::hipfire_home()?.join("registry.cache.json"),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryCache {
    fetched_at: u64,
    url: String,
    registry: RegistryV1,
}

pub fn bundled() -> Result<RegistryV1> {
    RegistryV1::parse(BUNDLED_REGISTRY, "bundled registry/v1.json")
}

/// Prefer the BUNDLED registry when it is newer than whatever was fetched.
///
/// `registry/v1.json` is compiled into the binary, so on a branch the bundled
/// copy IS that branch's registry — while the fetch targets master. Without
/// this, editing the registry on a branch changes nothing for a locally built
/// binary: the 24h cache or a master fetch silently wins, and nothing says so.
/// That cost a real debugging detour (a branch's `heads` map read as empty).
///
/// `generated_at` is the existing signal and needs no new configuration:
/// `scripts/registry_gen.py` stamps it on every regeneration, and its
/// `%Y-%m-%dT%H:%M:%SZ` form compares correctly as a plain string.
///
/// This does NOT break distribution. A released binary's bundled registry is
/// older than master's by construction, so the fetch keeps winning there and
/// users still get new models without upgrading. Only a freshly regenerated
/// local registry — i.e. someone editing it — takes precedence.
fn prefer_bundled_if_newer(
    loaded: LoadedRegistry,
    bundled: RegistryV1,
    warnings: &mut Vec<String>,
) -> LoadedRegistry {
    if loaded.source == RegistrySource::Bundled
        || bundled.generated_at <= loaded.registry.generated_at
    {
        return loaded;
    }
    warnings.push(format!(
        "using the bundled registry ({}), which is newer than the fetched one ({})",
        bundled.generated_at, loaded.registry.generated_at
    ));
    LoadedRegistry {
        registry: bundled,
        source: RegistrySource::Bundled,
        warnings: std::mem::take(warnings),
    }
}

pub fn load(paths: &RegistryPaths) -> LoadedRegistry {
    let url = env::var("HIPFIRE_REGISTRY_URL").unwrap_or_else(|_| DEFAULT_REGISTRY_URL.into());
    let fetch = env::var_os("HIPFIRE_NO_REGISTRY_FETCH").as_deref() != Some("1".as_ref());
    load_with_options(paths, fetch.then_some(url.as_str()))
}

fn load_with_options(paths: &RegistryPaths, url: Option<&str>) -> LoadedRegistry {
    let mut warnings = Vec::new();
    let bundled = bundled().expect("checked-in registry/v1.json must validate");
    let Some(url) = url else {
        return LoadedRegistry {
            registry: bundled,
            source: RegistrySource::Bundled,
            warnings,
        };
    };
    let now = epoch_millis();
    let cache = read_cache(&paths.cache, url, &mut warnings);
    if cache
        .as_ref()
        .is_some_and(|cache| cache_is_fresh(cache, now, REGISTRY_CACHE_TTL))
    {
        let loaded = LoadedRegistry {
            registry: cache.expect("checked above").registry,
            source: RegistrySource::Cache,
            warnings: std::mem::take(&mut warnings),
        };
        return prefer_bundled_if_newer(loaded, bundled, &mut warnings);
    }

    let loaded = match fetch_registry(&url) {
        Ok(registry) => {
            let cache_file = RegistryCache {
                fetched_at: now,
                url: url.to_owned(),
                registry: registry.clone(),
            };
            if let Err(error) = write_cache(&paths.cache, &cache_file) {
                warnings.push(error.to_string());
            }
            LoadedRegistry {
                registry,
                source: RegistrySource::Network,
                warnings: std::mem::take(&mut warnings),
            }
        }
        Err(error) => {
            warnings.push(error);
            if let Some(cache) = cache {
                LoadedRegistry {
                    registry: cache.registry,
                    source: RegistrySource::StaleCache,
                    warnings: std::mem::take(&mut warnings),
                }
            } else {
                LoadedRegistry {
                    registry: bundled.clone(),
                    source: RegistrySource::Bundled,
                    warnings: std::mem::take(&mut warnings),
                }
            }
        }
    };
    prefer_bundled_if_newer(loaded, bundled, &mut warnings)
}

fn read_cache(path: &Path, url: &str, warnings: &mut Vec<String>) -> Option<RegistryCache> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warnings.push(format!("registry cache read failed: {error}"));
            return None;
        }
    };
    let mut cache: RegistryCache = match serde_json::from_str(&raw) {
        Ok(cache) => cache,
        Err(error) => {
            warnings.push(format!("registry cache parse failed: {error}"));
            return None;
        }
    };
    if cache.url != url {
        return None;
    }
    if let Err(error) = cache.registry.validate("registry cache") {
        warnings.push(error.to_string());
        return None;
    }
    cache.registry.normalize_flash_next_xts();
    cache.registry.strip_retired(RETIRED_MODELS);
    cache
        .registry
        .aliases
        .retain(|_, target| cache.registry.models.contains_key(target));
    Some(cache)
}

fn cache_is_fresh(cache: &RegistryCache, now_ms: u64, ttl: Duration) -> bool {
    cache.fetched_at <= now_ms && now_ms.saturating_sub(cache.fetched_at) < ttl.as_millis() as u64
}

fn fetch_registry(url: &str) -> std::result::Result<RegistryV1, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(REGISTRY_FETCH_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|error| format!("registry fetch failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "registry fetch returned HTTP {}",
            response.status()
        ));
    }
    let raw = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("registry response read failed: {error}"))?;
    RegistryV1::parse(&raw, url).map_err(|error| error.to_string())
}

fn write_cache(path: &Path, cache: &RegistryCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| RegistryError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }
    let bytes = serde_json::to_vec(cache).map_err(|error| RegistryError::Parse {
        source_name: "registry cache serialization".into(),
        message: error.to_string(),
    })?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|source| RegistryError::Write {
        path: tmp.clone(),
        source,
    })?;
    fs::rename(&tmp, path).map_err(|source| RegistryError::Write {
        path: path.to_owned(),
        source,
    })
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLASH_NEXT_FIXTURE_TAGS: [&str; 4] = [
        "qwen3.8:flash-next",
        "qwen3.8:flash-next-mq4",
        "qwen3.8:flash-next-gptq3",
        "qwen3.8:flash-next-mq4-xts",
    ];

    fn pinned_flash_next_entry(file: &str) -> ModelEntry {
        ModelEntry {
            repo: FLASH_NEXT_XTS_REPO.into(),
            file: file.into(),
            size_gb: 125.29,
            min_vram_gb: 8.0,
            desc: "Qwen3.8-Flash-Next MQ4 XTS fixture".into(),
            sha256: Some(FLASH_NEXT_XTS_SHA256.into()),
            size_bytes: Some(FLASH_NEXT_XTS_SIZE),
            arch_id: Some(16),
            quant: Some("mq4".into()),
            ..Default::default()
        }
    }

    /// A registry as the pre-split wire format spelled it: three tags on the
    /// legacy filename and the canonical pinned monolithic under its own tag.
    fn flash_next_wire_registry(stamp: &str) -> RegistryV1 {
        let mut models = BTreeMap::new();
        for tag in &FLASH_NEXT_FIXTURE_TAGS[..3] {
            models.insert(
                (*tag).to_owned(),
                pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE),
            );
        }
        models.insert(
            "qwen3.8:flash-next-mq4-xts".into(),
            pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE),
        );
        RegistryV1 {
            schema_version: REGISTRY_SCHEMA_VERSION,
            generated_at: stamp.into(),
            _comment: None,
            models,
            aliases: BTreeMap::from([("fixture".into(), "qwen3.8:flash-next".into())]),
        }
    }

    fn assert_flash_next_xts(registry: &RegistryV1, tags: &[&str]) {
        for tag in tags {
            let (_, entry) = registry.model(tag).unwrap();
            assert!(entry.is_flash_next_xts(), "{tag}");
            assert_eq!(entry.file, FLASH_NEXT_XTS_CANONICAL_FILE, "{tag}");
            assert_eq!(entry.arch_id, Some(16));
            assert_eq!(entry.quant.as_deref(), Some("mq4"));
        }
    }

    const STOCK_PLE_Q4: (&str, &str, u64) = (
        "qwen3.8-flash-next.ple-q4",
        "fc78d30b9d2376db2b713732e80de2069f1cd1d6e1e40b50c716927dcc18ee1e",
        29_120_146_432,
    );
    const STOCK_PLE_Q8: (&str, &str, u64) = (
        "qwen3.8-flash-next.ple-q8",
        "a655edc25402d0dfbef276cb3c5ca57724fa1074976d8a9029371a3a52eca89c",
        54_400_269_312,
    );

    /// The published MQ4 XTS split base with the shared PLE pair: the entry
    /// every stock MQ4 tag carries.
    fn stock_split_base_entry() -> ModelEntry {
        let slot = |(file, sha256, size_bytes): (&str, &str, u64)| Sidecar {
            file: file.into(),
            sha256: Some(sha256.into()),
            size_bytes: Some(size_bytes),
        };
        ModelEntry {
            repo: FLASH_NEXT_XTS_REPO.into(),
            file: FLASH_NEXT_XTS_SPLIT_BASE_FILE.into(),
            size_gb: 70.89,
            min_vram_gb: 8.0,
            desc: "Qwen3.8-Flash-Next MQ4 XTS split base".into(),
            sha256: Some(FLASH_NEXT_XTS_SPLIT_BASE_SHA256.into()),
            size_bytes: Some(FLASH_NEXT_XTS_SPLIT_BASE_SIZE),
            arch_id: Some(16),
            quant: Some("mq4".into()),
            ple: BTreeMap::from([
                ("q4".to_owned(), slot(STOCK_PLE_Q4)),
                ("q8".to_owned(), slot(STOCK_PLE_Q8)),
            ]),
            ..Default::default()
        }
    }

    /// The four stock MQ4 tags are one artifact: the split base on the shared
    /// PLE pair, never a monolithic.
    fn assert_stock_split_tags(registry: &RegistryV1) {
        let first = &registry.models[FLASH_NEXT_FIXTURE_TAGS[0]];
        for tag in FLASH_NEXT_FIXTURE_TAGS {
            let (_, entry) = registry.model(tag).unwrap();
            assert_eq!(entry, first, "{tag}: identical metadata");
            assert_eq!(entry.repo, FLASH_NEXT_XTS_REPO, "{tag}");
            assert_eq!(entry.file, FLASH_NEXT_XTS_SPLIT_BASE_FILE, "{tag}");
            assert_eq!(
                entry.sha256.as_deref(),
                Some(FLASH_NEXT_XTS_SPLIT_BASE_SHA256),
                "{tag}"
            );
            assert_eq!(entry.size_bytes, Some(FLASH_NEXT_XTS_SPLIT_BASE_SIZE));
            assert_eq!(entry.arch_id, Some(16));
            assert_eq!(entry.quant.as_deref(), Some("mq4"));
            assert!(entry.desc.contains("MQ4 XTS"), "{tag}");
            assert_eq!(
                entry.ple.keys().map(String::as_str).collect::<Vec<_>>(),
                ["q4", "q8"],
                "{tag}"
            );
            for (key, (file, sha256, size)) in [("q4", STOCK_PLE_Q4), ("q8", STOCK_PLE_Q8)] {
                let sidecar = entry
                    .ple_sidecar(key)
                    .unwrap_or_else(|| panic!("{tag}: ple.{key}"));
                assert_eq!(sidecar.file, file, "{tag}");
                assert_eq!(sidecar.sha256.as_deref(), Some(sha256), "{tag}");
                assert_eq!(sidecar.size_bytes, Some(size), "{tag}");
            }
            assert!(!entry.is_flash_next_xts(), "{tag}");
            assert!(!entry.is_split_only(), "{tag}");
            assert!(entry.grandfathered_monolithic().is_some(), "{tag}");
        }
    }

    #[test]
    fn bundled_registry_publishes_only_the_split_mq4_xts_base() {
        let wire: RegistryV1 = serde_json::from_str(BUNDLED_REGISTRY).unwrap();
        let parsed = bundled().unwrap();
        assert_stock_split_tags(&parsed);
        assert_eq!(wire.aliases["qwen3.8:flash"], "qwen3.8:flash-next");
        let (short_tag, _) = parsed.model("qwen3.8:flash").unwrap();
        assert_eq!(short_tag, "qwen3.8:flash-next");
        for (tag, entry) in &wire.models {
            assert_ne!(entry.file, FLASH_NEXT_XTS_CANONICAL_FILE, "{tag}");
            assert_ne!(entry.file, FLASH_NEXT_XTS_LEGACY_FILE, "{tag}");
            assert!(!entry.is_flash_next_xts(), "{tag}");
            // The recipe is named MQ4 XTS; the old marketing name stays out of descs
            // (a desc may still cite the legacy FILE names in lower case).
            assert!(!entry.desc.contains("GPTQ3"), "{tag}");
            // An artifact upgrade pin is the one intended parse-time rewrite (identity only).
            let mut expected = entry.clone();
            if let Some(pin) = ARTIFACT_UPGRADE_PINS.iter().find(|pin| pin.tag == tag) {
                expected.sha256 = Some(pin.sha256.to_owned());
                expected.size_bytes = Some(pin.size_bytes);
            }
            assert_eq!(
                &parsed.models[tag], &expected,
                "{tag}: nothing else to normalize"
            );
            assert!(
                entry.ple.keys().all(|key| is_ple_format_token(key)),
                "{tag}"
            );
            // Only the MQ4 XTS base is a view of a pinned monolithic.
            assert_eq!(
                entry.grandfathered_monolithic().is_some(),
                FLASH_NEXT_FIXTURE_TAGS.contains(&tag.as_str()),
                "{tag}"
            );
        }
        // The MQ3 and MQ2 bases are split-only on the same shared pair.
        let split_tags = [
            "qwen3.8:flash-next-mq3-xts",
            "qwen3.8:flash-next-mq2-xts",
            "qwen3.8:flash-next-mq2-xts-4500",
        ];
        for tag in split_tags {
            let split = &parsed.models[tag];
            assert!(split.has_ple_sidecars(), "{tag}");
            assert!(split.is_split_only(), "{tag}");
            assert_eq!(
                split.ple.keys().map(String::as_str).collect::<Vec<_>>(),
                ["q4", "q8"],
                "{tag}"
            );
            assert_eq!(split.ple["q4"].file, "qwen3.8-flash-next.ple-q4", "{tag}");
            assert_eq!(split.ple["q8"].file, "qwen3.8-flash-next.ple-q8", "{tag}");
        }
        for (tag, entry) in &parsed.models {
            if !split_tags.contains(&tag.as_str())
                && !FLASH_NEXT_FIXTURE_TAGS.contains(&tag.as_str())
            {
                assert!(
                    !entry.has_ple_sidecars(),
                    "{tag}: only the MQ4, MQ3 and MQ2 bases are split"
                );
            }
        }
        for pin in RETIRED_MODELS {
            assert!(
                !parsed.models.contains_key(pin.tag),
                "{}: a retired tag is not in the available catalog",
                pin.tag
            );
            assert!(
                !parsed.models.values().any(|entry| entry.file == pin.file),
                "{}: no available entry names the retired file",
                pin.tag
            );
        }
    }

    const TRUNK_OVERLAY_FILE: &str = "qwen3.8-flash-next.mq4-xts.trunk-overlay";

    #[test]
    fn trunk_overlay_pin_names_the_canonical_base_file_plus_the_suffix() {
        assert_eq!(TRUNK_OVERLAY_PINS.len(), 2);
        let pin = &TRUNK_OVERLAY_PINS[0];
        assert_eq!(
            pin.file,
            format!(
                "{FLASH_NEXT_XTS_CANONICAL_FILE}{}",
                hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX
            )
        );
        assert_eq!(pin.file, TRUNK_OVERLAY_FILE);
        assert_eq!(pin.base_sha256, FLASH_NEXT_XTS_SHA256);
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(pin.size_bytes, 305_543_168);
        for digest in [pin.base_sha256, pin.sha256] {
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert!(!pin.file.contains('/') && !pin.file.contains('\\'));
    }

    #[test]
    fn trunk_overlay_lookup_matches_the_base_identity_only() {
        let entry = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        let overlay = entry.trunk_overlay().unwrap();
        assert_eq!(overlay.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            overlay.sidecar,
            Sidecar {
                file: TRUNK_OVERLAY_FILE.into(),
                sha256: Some(TRUNK_OVERLAY_PINS[0].sha256.into()),
                size_bytes: Some(305_543_168),
            }
        );
        // The legacy spelling of the same pinned bytes resolves to the canonical overlay name.
        let legacy = pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE);
        assert_eq!(legacy.trunk_overlay(), Some(overlay));

        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.sha256 = Some(digest('a'));
        assert!(other.trunk_overlay().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.size_bytes = Some(FLASH_NEXT_XTS_SIZE - 1);
        assert!(other.trunk_overlay().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.sha256 = None;
        assert!(other.trunk_overlay().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.size_bytes = None;
        assert!(other.trunk_overlay().is_none());
    }

    #[test]
    fn trunk_overlay_in_uses_the_supplied_pin_table() {
        let mut entry = pinned_flash_next_entry("custom.mq4");
        entry.sha256 = Some(digest('b'));
        entry.size_bytes = Some(123);
        assert!(trunk_overlay_in(&entry, TRUNK_OVERLAY_PINS).is_none());
        let pins = [TrunkOverlayPin {
            base_sha256: "b".repeat(64).leak(),
            base_size: 123,
            repo: "test/repo",
            file: "custom.mq4.trunk-overlay",
            sha256: "c".repeat(64).leak(),
            size_bytes: 7,
        }];
        let overlay = trunk_overlay_in(&entry, &pins).unwrap();
        assert_eq!(overlay.repo, "test/repo");
        assert_eq!(overlay.sidecar.file, "custom.mq4.trunk-overlay");
        assert_eq!(overlay.sidecar.sha256, Some(digest('c')));
        assert_eq!(overlay.sidecar.size_bytes, Some(7));
        assert!(trunk_overlay_in(&entry, &[]).is_none());
    }

    #[test]
    fn bundled_registry_resolves_the_trunk_overlay_on_exactly_the_stock_xts_tags() {
        let parsed = bundled().unwrap();
        let mut resolved: Vec<&str> = parsed
            .models
            .iter()
            .filter(|(_, entry)| entry.trunk_overlay().is_some())
            .map(|(tag, _)| tag.as_str())
            .collect();
        resolved.sort_unstable();
        let mut expected = FLASH_NEXT_FIXTURE_TAGS.to_vec();
        expected.sort_unstable();
        assert_eq!(resolved, expected);
        for pin in RETIRED_MODELS {
            assert!(pin.entry().trunk_overlay().is_none(), "{}", pin.tag);
        }
        let first = parsed.models["qwen3.8:flash-next"].trunk_overlay().unwrap();
        for tag in FLASH_NEXT_FIXTURE_TAGS {
            assert_eq!(
                parsed.models[tag].trunk_overlay().as_ref(),
                Some(&first),
                "{tag}"
            );
        }
        let (alias_tag, aliased) = parsed.model("qwen3.8:flash").unwrap();
        assert_eq!(alias_tag, "qwen3.8:flash-next");
        assert_eq!(aliased.trunk_overlay().as_ref(), Some(&first));
    }

    const SPLIT_TRUNK_OVERLAY_FILE: &str = "qwen3.8-flash-next-base.mq4-xts.trunk-overlay";

    #[test]
    fn split_base_trunk_overlay_pin_is_rebound_and_keyed_by_the_base_identity() {
        let pin = &TRUNK_OVERLAY_PINS[1];
        assert_eq!(
            pin.file,
            format!(
                "{FLASH_NEXT_XTS_SPLIT_BASE_FILE}{}",
                hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX
            )
        );
        assert_eq!(pin.file, SPLIT_TRUNK_OVERLAY_FILE);
        assert_eq!(pin.base_sha256, FLASH_NEXT_XTS_SPLIT_BASE_SHA256);
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SPLIT_BASE_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            pin.sha256,
            "e3e352ca7fdcb0890fb0c6ddc07903cbbd9ceab65ad24ed5af0637e7dd92e4a6"
        );
        assert_eq!(pin.size_bytes, 305_543_168);
        assert_ne!(pin.base_sha256, TRUNK_OVERLAY_PINS[0].base_sha256);
        assert_ne!(pin.sha256, TRUNK_OVERLAY_PINS[0].sha256);

        let split = stock_split_base_entry();
        let overlay = split.trunk_overlay().unwrap();
        assert_eq!(overlay.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            overlay.sidecar,
            Sidecar {
                file: SPLIT_TRUNK_OVERLAY_FILE.into(),
                sha256: Some(pin.sha256.into()),
                size_bytes: Some(305_543_168),
            }
        );
        // The monolithic keeps its own overlay; neither pin binds the other base.
        let monolithic = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE)
            .trunk_overlay()
            .unwrap();
        assert_eq!(monolithic.sidecar.file, TRUNK_OVERLAY_FILE);
        assert_eq!(
            monolithic.sidecar.sha256.as_deref(),
            Some(TRUNK_OVERLAY_PINS[0].sha256)
        );
        let mut other = split.clone();
        other.size_bytes = Some(FLASH_NEXT_XTS_SPLIT_BASE_SIZE - 1);
        assert!(other.trunk_overlay().is_none());
        let mut other = split;
        other.sha256 = Some(digest('a'));
        assert!(other.trunk_overlay().is_none());
    }

    const SYMTRUNK_BASE_FILE: &str = "qwen3.8-flash-next-base.mq4-xts-symtrunk";

    /// The published symmetric-trunk base the HC8 pin binds to.
    fn symtrunk_base_entry() -> ModelEntry {
        let pin = &HC8_SIDECAR_PINS[0];
        ModelEntry {
            file: SYMTRUNK_BASE_FILE.into(),
            sha256: Some(pin.base_sha256.into()),
            size_bytes: Some(pin.base_size),
            ..stock_split_base_entry()
        }
    }

    #[test]
    fn hc8_sidecar_pin_names_the_base_file_plus_the_suffix() {
        assert_eq!(HC8_SIDECAR_PINS.len(), 1);
        let pin = &HC8_SIDECAR_PINS[0];
        assert_eq!(
            pin.file,
            format!(
                "{SYMTRUNK_BASE_FILE}{}",
                hipfire_config::memory::HC8_SIDECAR_SUFFIX
            )
        );
        assert_eq!(
            pin.base_sha256,
            "27dfff6692eba3385655959d134771fcff154c6279ab0a6c43cb8c7d81a7fa31"
        );
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SPLIT_BASE_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            pin.sha256,
            "2adf32e564f09efbe185c7ebf57483dc58103f21341569e716173098f0e2ee67"
        );
        assert_eq!(pin.size_bytes, 659_649_536);
        for digest in [pin.base_sha256, pin.sha256] {
            assert_eq!(digest.len(), 64);
            assert!(digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        }
        assert!(!pin.file.contains('/') && !pin.file.contains('\\'));
    }

    #[test]
    fn hc8_sidecar_lookup_matches_the_symtrunk_base_identity_only() {
        let entry = symtrunk_base_entry();
        let hc8 = entry.hc8_sidecar().unwrap();
        let pin = &HC8_SIDECAR_PINS[0];
        assert_eq!(hc8.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            hc8.sidecar,
            Sidecar {
                file: pin.file.into(),
                sha256: Some(pin.sha256.into()),
                size_bytes: Some(pin.size_bytes),
            }
        );
        // Upper-case digests of the same base still match.
        let mut upper = entry.clone();
        upper.sha256 = upper.sha256.map(|sha| sha.to_ascii_uppercase());
        assert!(upper.hc8_sidecar().is_some());
        // The stock split base (a different artifact of the same size) has no pin.
        assert!(stock_split_base_entry().hc8_sidecar().is_none());
        assert_eq!(
            stock_split_base_entry().size_bytes,
            Some(HC8_SIDECAR_PINS[0].base_size)
        );
        // A different size or sha, or no pinned identity, never matches.
        let mut other = entry.clone();
        other.size_bytes = Some(pin.base_size - 1);
        assert!(other.hc8_sidecar().is_none());
        let mut other = entry.clone();
        other.sha256 = Some(digest('a'));
        assert!(other.hc8_sidecar().is_none());
        let mut other = entry.clone();
        other.sha256 = None;
        assert!(other.hc8_sidecar().is_none());
        let mut other = entry;
        other.size_bytes = None;
        assert!(other.hc8_sidecar().is_none());
    }

    #[test]
    fn hc8_pin_does_not_bind_either_overlay_or_hotset_base() {
        for pin in TRUNK_OVERLAY_PINS {
            assert_ne!(pin.base_sha256, HC8_SIDECAR_PINS[0].base_sha256);
        }
        assert!(symtrunk_base_entry().trunk_overlay().is_none());
        assert!(symtrunk_base_entry().expert_hotset().is_none());
        assert!(hc8_sidecar_in(&symtrunk_base_entry(), &[]).is_none());
    }

    #[test]
    fn hot_units_pin_names_the_base_file_plus_the_suffix_and_binds_its_calibration() {
        assert_eq!(HOT_UNITS_SIDECAR_PINS.len(), 1);
        let pin = &HOT_UNITS_SIDECAR_PINS[0];
        assert_eq!(
            pin.file,
            format!(
                "{SYMTRUNK_BASE_FILE}{}",
                hipfire_config::memory::EP_HOT_UNITS_SIDECAR_SUFFIX
            )
        );
        assert_eq!(
            pin.file,
            "qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.hot-units"
        );
        assert_eq!(pin.base_sha256, HC8_SIDECAR_PINS[0].base_sha256);
        assert_eq!(pin.base_size, HC8_SIDECAR_PINS[0].base_size);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            pin.ranking_sha256,
            "c27a8ebff3d560e6c73d9b1e8b52fb27878a13735301660159109e857b0fb53c"
        );
        assert_eq!(
            pin.sha256,
            "a50317cf2014f88d306d88366131212757d875c4c6e65572d659d361ec17db13"
        );
        assert_eq!(pin.size_bytes, 67_526);
        assert!(!pin.published, "pinned ahead of the release upload");
        for digest in [pin.base_sha256, pin.ranking_sha256, pin.sha256] {
            assert_eq!(digest.len(), 64);
            assert!(digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        }
        assert!(!pin.file.contains('/') && !pin.file.contains('\\'));
    }

    #[test]
    fn hot_units_lookup_matches_the_symtrunk_base_identity_only() {
        let entry = symtrunk_base_entry();
        let units = entry.hot_units_sidecar().unwrap();
        let pin = &HOT_UNITS_SIDECAR_PINS[0];
        assert_eq!(units.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            units.sidecar,
            Sidecar {
                file: pin.file.into(),
                sha256: Some(pin.sha256.into()),
                size_bytes: Some(pin.size_bytes),
            }
        );
        assert_eq!(units.ranking_sha256, pin.ranking_sha256);
        assert!(!units.published);
        let mut upper = entry.clone();
        upper.sha256 = upper.sha256.map(|sha| sha.to_ascii_uppercase());
        assert!(upper.hot_units_sidecar().is_some());
        // The stock split base (same size, different artifact) has no pin.
        assert!(stock_split_base_entry().hot_units_sidecar().is_none());
        let mut other = entry.clone();
        other.size_bytes = Some(pin.base_size - 1);
        assert!(other.hot_units_sidecar().is_none());
        let mut other = entry.clone();
        other.sha256 = Some(digest('a'));
        assert!(other.hot_units_sidecar().is_none());
        let mut other = entry.clone();
        other.sha256 = None;
        assert!(other.hot_units_sidecar().is_none());
        let mut other = entry;
        other.size_bytes = None;
        assert!(other.hot_units_sidecar().is_none());
        assert!(hot_units_sidecar_in(&symtrunk_base_entry(), &[]).is_none());
    }

    #[test]
    fn hot_units_pin_matches_the_calibration_file_it_was_cut_from() {
        // The sidecar's own header binds the pinned base and ranking; the pin must say the same
        // when the calibration file is at hand (it is not in the repo, so skip when absent).
        let path = std::path::Path::new(
            "/home/kaden/qcal/release-0.4.5/dynres/statichot/sidecar/qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.hot-units",
        );
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let pin = &HOT_UNITS_SIDECAR_PINS[0];
        assert_eq!(bytes.len() as u64, pin.size_bytes);
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["model_sha256"], pin.base_sha256);
        assert_eq!(value["base_size"], pin.base_size);
        assert_eq!(value["base_ranking_sha256"], pin.ranking_sha256);
    }

    #[test]
    fn hot_units_pin_binds_only_the_symtrunk_base() {
        for pin in TRUNK_OVERLAY_PINS {
            assert_ne!(pin.base_sha256, HOT_UNITS_SIDECAR_PINS[0].base_sha256);
        }
        assert!(symtrunk_base_entry().hc8_sidecar().is_some());
        assert!(symtrunk_base_entry().expert_hotset().is_none());
        let registry = RegistryV1::parse(BUNDLED_REGISTRY, "bundled").unwrap();
        assert!(registry_sidecar_files(&registry).contains(HOT_UNITS_SIDECAR_PINS[0].file));
    }

    #[test]
    fn hot_units_ranking_binding_resolves_to_a_pinned_hotset_of_the_same_base() {
        let units = HOT_UNITS_SIDECAR_PINS[0];
        let ranking = EP_EXPERT_HOTSET_PINS
            .iter()
            .find(|pin| pin.sha256 == units.ranking_sha256)
            .expect("the ranking the hot units were calibrated under is pinned");
        assert_eq!(ranking.base_sha256, units.base_sha256);
        assert_eq!(ranking.base_size, units.base_size);
        assert_eq!(ranking.repo, units.repo);
        assert_eq!(ranking.size_bytes, 312_036);
        assert_eq!(
            ranking.file,
            format!(
                "{SYMTRUNK_BASE_FILE}{}",
                hipfire_config::memory::EP_EXPERT_HOTSET_SIDECAR_SUFFIX
            )
        );
        // Through the entry lookup, with the gate on.
        let resolved = ep_expert_hotset_in(&symtrunk_base_entry(), EP_EXPERT_HOTSET_PINS, true)
            .expect("the symtrunk base resolves its ranking");
        assert_eq!(resolved.sidecar.sha256.as_deref(), Some(units.ranking_sha256));
        // Both upload together at the release.
        assert!(!resolved.published);
        assert!(!units.published);
        assert!(UNPUBLISHED_PIN_FILES.contains(&ranking.file));
        // The split base's ranking stays published and is not the hot units' ranking.
        let split = ep_expert_hotset_in(&stock_split_base_entry(), EP_EXPERT_HOTSET_PINS, true)
            .unwrap();
        assert!(split.published);
        assert_ne!(split.sidecar.sha256.as_deref(), Some(units.ranking_sha256));
        // The local calibration copy, when at hand, is the pinned bytes.
        let path = std::path::Path::new(
            "/home/kaden/qcal/release-0.4.5/symtrunk/model/qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.expert-hotset",
        );
        if let Ok(bytes) = std::fs::read(path) {
            assert_eq!(bytes.len() as u64, ranking.size_bytes);
        }
    }

    const EXPERT_HOTSET_FILE: &str = "qwen3.8-flash-next.mq4-xts.expert-hotset";

    #[test]
    fn expert_hotset_pin_has_exact_metadata_and_canonical_name() {
        assert_eq!(EXPERT_HOTSET_PINS.len(), 2);
        let pin = &EXPERT_HOTSET_PINS[0];
        assert_eq!(
            pin.file,
            format!("{FLASH_NEXT_XTS_CANONICAL_FILE}.expert-hotset")
        );
        assert_eq!(pin.file, EXPERT_HOTSET_FILE);
        assert_eq!(pin.base_sha256, FLASH_NEXT_XTS_SHA256);
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            pin.sha256,
            "7f6760e11a86e4ffe0b05c050ce64e6eb772ab748b6b11c8b2e6dbc566e57086"
        );
        assert_eq!(pin.size_bytes, 217_676);
        for digest in [pin.base_sha256, pin.sha256] {
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert!(!pin.file.contains('/') && !pin.file.contains('\\'));
    }

    #[test]
    fn expert_hotset_lookup_matches_the_base_identity_only() {
        let entry = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        let hotset = entry.expert_hotset().unwrap();
        assert_eq!(hotset.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            hotset.sidecar,
            Sidecar {
                file: EXPERT_HOTSET_FILE.into(),
                sha256: Some(EXPERT_HOTSET_PINS[0].sha256.into()),
                size_bytes: Some(217_676),
            }
        );
        // The legacy spelling of the same pinned bytes resolves to the canonical hotset name.
        let legacy = pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE);
        assert_eq!(legacy.expert_hotset(), Some(hotset));

        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.sha256 = Some(digest('a'));
        assert!(other.expert_hotset().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.size_bytes = Some(FLASH_NEXT_XTS_SIZE - 1);
        assert!(other.expert_hotset().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.sha256 = None;
        assert!(other.expert_hotset().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.size_bytes = None;
        assert!(other.expert_hotset().is_none());
    }

    #[test]
    fn expert_hotset_in_uses_the_supplied_pin_table() {
        let mut entry = pinned_flash_next_entry("custom.mq4");
        entry.sha256 = Some(digest('b'));
        entry.size_bytes = Some(123);
        assert!(expert_hotset_in(&entry, EXPERT_HOTSET_PINS).is_none());
        let pins = [ExpertHotsetPin {
            base_sha256: "b".repeat(64).leak(),
            base_size: 123,
            repo: "test/repo",
            file: "custom.mq4.expert-hotset",
            sha256: "c".repeat(64).leak(),
            size_bytes: 7,
        }];
        let hotset = expert_hotset_in(&entry, &pins).unwrap();
        assert_eq!(hotset.repo, "test/repo");
        assert_eq!(hotset.sidecar.file, "custom.mq4.expert-hotset");
        assert_eq!(hotset.sidecar.sha256, Some(digest('c')));
        assert_eq!(hotset.sidecar.size_bytes, Some(7));
        assert!(expert_hotset_in(&entry, &[]).is_none());
    }

    #[test]
    fn bundled_registry_resolves_the_expert_hotset_on_exactly_the_stock_xts_tags() {
        let parsed = bundled().unwrap();
        let mut resolved: Vec<&str> = parsed
            .models
            .iter()
            .filter(|(_, entry)| entry.expert_hotset().is_some())
            .map(|(tag, _)| tag.as_str())
            .collect();
        resolved.sort_unstable();
        let mut expected = FLASH_NEXT_FIXTURE_TAGS.to_vec();
        expected.sort_unstable();
        assert_eq!(resolved, expected);
        for pin in RETIRED_MODELS {
            assert!(pin.entry().expert_hotset().is_none(), "{}", pin.tag);
        }
        let first = parsed.models["qwen3.8:flash-next"].expert_hotset().unwrap();
        for tag in FLASH_NEXT_FIXTURE_TAGS {
            assert_eq!(
                parsed.models[tag].expert_hotset().as_ref(),
                Some(&first),
                "{tag}"
            );
        }
        let (alias_tag, aliased) = parsed.model("qwen3.8:flash").unwrap();
        assert_eq!(alias_tag, "qwen3.8:flash-next");
        assert_eq!(aliased.expert_hotset().as_ref(), Some(&first));
        assert!(!BUNDLED_REGISTRY.contains("expert_hotset"));
        assert!(!include_str!("../../../registry/models.json").contains("expert_hotset"));
    }

    #[test]
    fn split_base_expert_hotset_pin_is_rebound_and_keyed_by_the_base_identity() {
        let pin = &EXPERT_HOTSET_PINS[1];
        assert_eq!(
            pin.file,
            format!("{FLASH_NEXT_XTS_SPLIT_BASE_FILE}.expert-hotset")
        );
        assert_eq!(pin.base_sha256, FLASH_NEXT_XTS_SPLIT_BASE_SHA256);
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SPLIT_BASE_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            pin.sha256,
            "911430746607f0a15f388594302304d1ce0283f1787e3f3d6aa32005d0a067a1"
        );
        assert_eq!(pin.size_bytes, 217_675);
        assert_ne!(pin.sha256, EXPERT_HOTSET_PINS[0].sha256);

        let split = stock_split_base_entry();
        let hotset = split.expert_hotset().unwrap();
        assert_eq!(hotset.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            hotset.sidecar,
            Sidecar {
                file: "qwen3.8-flash-next-base.mq4-xts.expert-hotset".into(),
                sha256: Some(pin.sha256.into()),
                size_bytes: Some(217_675),
            }
        );
        let monolithic = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE)
            .expert_hotset()
            .unwrap();
        assert_eq!(monolithic.sidecar.file, EXPERT_HOTSET_FILE);
        assert_eq!(
            monolithic.sidecar.sha256.as_deref(),
            Some(EXPERT_HOTSET_PINS[0].sha256)
        );
        let mut other = split.clone();
        other.size_bytes = Some(FLASH_NEXT_XTS_SPLIT_BASE_SIZE - 1);
        assert!(other.expert_hotset().is_none());
        let mut other = split;
        other.sha256 = None;
        assert!(other.expert_hotset().is_none());
    }

    #[test]
    fn ep_expert_hotset_pin_has_exact_metadata_and_is_gated() {
        assert_eq!(EP_EXPERT_HOTSET_PINS.len(), 2);
        let pin = &EP_EXPERT_HOTSET_PINS[0];
        let split_hotset = &EXPERT_HOTSET_PINS[1];
        assert_eq!(
            pin.file,
            format!(
                "{FLASH_NEXT_XTS_SPLIT_BASE_FILE}{}",
                hipfire_config::memory::EP_EXPERT_HOTSET_SIDECAR_SUFFIX
            )
        );
        assert_eq!(pin.file, "qwen3.8-flash-next-base.mq4-xts.ep.expert-hotset");
        assert_ne!(pin.file, split_hotset.file);
        assert_eq!(pin.base_sha256, split_hotset.base_sha256);
        assert_eq!(pin.base_size, split_hotset.base_size);
        assert_eq!(pin.base_sha256, FLASH_NEXT_XTS_SPLIT_BASE_SHA256);
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SPLIT_BASE_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            pin.sha256,
            "77540499adecb1cbca1186051d24aaa329b986dddd52f6d950c887088e61d7b6"
        );
        assert_eq!(pin.size_bytes, 312_036);

        let split = stock_split_base_entry();
        assert!(ep_expert_hotset_in(&split, EP_EXPERT_HOTSET_PINS, false).is_none());
        let ep = ep_expert_hotset_in(&split, EP_EXPERT_HOTSET_PINS, true).unwrap();
        assert_eq!(ep.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            ep.sidecar,
            Sidecar {
                file: pin.file.into(),
                sha256: Some(pin.sha256.into()),
                size_bytes: Some(312_036),
            }
        );
        // The method follows the compiled-in gate.
        assert_eq!(
            split.ep_expert_hotset().is_some(),
            hipfire_config::memory::EP_EXPERT_HOTSET_DEFAULT_ON
        );

        let monolithic = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        assert!(ep_expert_hotset_in(&monolithic, EP_EXPERT_HOTSET_PINS, true).is_none());
        let mut other = split.clone();
        other.size_bytes = Some(FLASH_NEXT_XTS_SPLIT_BASE_SIZE - 1);
        assert!(ep_expert_hotset_in(&other, EP_EXPERT_HOTSET_PINS, true).is_none());
        let mut other = split.clone();
        other.sha256 = Some(digest('c'));
        assert!(ep_expert_hotset_in(&other, EP_EXPERT_HOTSET_PINS, true).is_none());
        let mut unpinned = split;
        unpinned.sha256 = None;
        assert!(ep_expert_hotset_in(&unpinned, EP_EXPERT_HOTSET_PINS, true).is_none());
    }

    #[test]
    fn registry_sidecar_files_contains_the_ep_expert_hotset_file() {
        let registry = RegistryV1::parse(BUNDLED_REGISTRY, "bundled").unwrap();
        assert!(registry_sidecar_files(&registry).contains(EP_EXPERT_HOTSET_PINS[0].file));
    }

    #[test]
    fn the_wire_registry_has_no_trunk_overlay_field() {
        // The overlay pin is compiled into the client; the registry schema is
        // unchanged so older clients' strict v1 parser keeps accepting it.
        assert!(!BUNDLED_REGISTRY.contains("trunk_overlay"));
        assert!(!include_str!("../../../registry/models.json").contains("trunk_overlay"));
        let mut value: serde_json::Value = serde_json::from_str(BUNDLED_REGISTRY).unwrap();
        value["models"]["qwen3.8:flash-next"]["trunk_overlay"] = serde_json::json!({
            "file": TRUNK_OVERLAY_FILE
        });
        assert!(
            RegistryV1::parse(&value.to_string(), "fixture").is_err(),
            "ModelEntry stays deny_unknown_fields"
        );
    }

    #[test]
    fn flash_next_legacy_spelled_wire_entries_normalize_locally() {
        let wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        for tag in &FLASH_NEXT_FIXTURE_TAGS[..3] {
            assert_eq!(wire.models[*tag].file, FLASH_NEXT_XTS_LEGACY_FILE);
        }
        let parsed = RegistryV1::parse(&serde_json::to_string(&wire).unwrap(), "fixture").unwrap();
        assert_flash_next_xts(&parsed, &FLASH_NEXT_FIXTURE_TAGS);
        assert_eq!(parsed.model("fixture").unwrap().0, "qwen3.8:flash-next");
    }

    const SPLIT_BASE_FILE: &str = "qwen3.8-flash-next-base.mq4-xts";

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    /// The registry after the split switch: `qwen3.8:flash-next` is the split
    /// base with both PLE slots; the pinned monolithic keeps its own tags.
    fn split_registry_json() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": {
                    "repo": FLASH_NEXT_XTS_REPO,
                    "file": SPLIT_BASE_FILE,
                    "size_gb": 70.9,
                    "min_vram_gb": 8.0,
                    "desc": "Qwen3.8-Flash-Next MQ4 XTS split base",
                    "arch_id": 16,
                    "quant": "mq4",
                    "sha256": FLASH_NEXT_XTS_SPLIT_BASE_SHA256,
                    "size_bytes": FLASH_NEXT_XTS_SPLIT_BASE_SIZE,
                    "ple": {
                        "q8": {
                            "file": format!("{SPLIT_BASE_FILE}.ple-q8"),
                            "sha256": digest('b'),
                            "size_bytes": 54_400_000_000_u64
                        },
                        "q4": {
                            "file": format!("{SPLIT_BASE_FILE}.ple-q4"),
                            "sha256": digest('c'),
                            "size_bytes": 28_800_000_000_u64
                        }
                    }
                },
                "qwen3.8:flash-next-mq4": serde_json::to_value(
                    pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE)).unwrap(),
                "qwen3.8:flash-next-mq4-xts": serde_json::to_value(
                    pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE)).unwrap()
            },
            "aliases": { "qwen3.8:flash": "qwen3.8:flash-next" }
        })
    }

    fn parse_split(value: &serde_json::Value) -> Result<RegistryV1> {
        RegistryV1::parse(&value.to_string(), "split fixture")
    }

    #[test]
    fn split_entry_parses_and_keeps_the_pinned_monolithic_distinct() {
        let registry = parse_split(&split_registry_json()).unwrap();
        let (tag, split) = registry.model("qwen3.8:flash").unwrap();
        assert_eq!(tag, "qwen3.8:flash-next");
        assert_eq!(split.file, SPLIT_BASE_FILE);
        assert!(split.has_ple_sidecars());
        assert!(
            !split.is_flash_next_xts(),
            "the split base is not the pinned monolithic"
        );
        assert_eq!(
            split.ple["q8"].file,
            format!("{SPLIT_BASE_FILE}{}", ple_sidecar_suffix("q8"))
        );
        assert_eq!(
            split.ple["q4"].file,
            format!("{SPLIT_BASE_FILE}{}", ple_sidecar_suffix("q4"))
        );
        // The pinned monolithic still normalizes to the canonical file under its tags.
        assert_flash_next_xts(
            &registry,
            &["qwen3.8:flash-next-mq4", "qwen3.8:flash-next-mq4-xts"],
        );
        // Either legacy spelling of the pin resolves to a pinned entry, never the split.
        let (legacy_tag, legacy) = registry.model(FLASH_NEXT_XTS_LEGACY_FILE).unwrap();
        assert!(legacy.is_flash_next_xts(), "{legacy_tag}");
        assert_eq!(
            registry.entry_for_file(SPLIT_BASE_FILE).unwrap().0,
            "qwen3.8:flash-next"
        );
        let reparsed = parse_split(&serde_json::to_value(&registry).unwrap()).unwrap();
        assert_eq!(registry, reparsed);
    }

    #[test]
    fn split_base_without_slots_or_with_one_slot_is_valid() {
        for keep in [None, Some("q8"), Some("q4")] {
            let mut value = split_registry_json();
            let ple = value["models"]["qwen3.8:flash-next"]["ple"]
                .as_object_mut()
                .unwrap();
            for slot in ["q8", "q4"] {
                if Some(slot) != keep {
                    ple.remove(slot);
                }
            }
            if keep.is_none() {
                value["models"]["qwen3.8:flash-next"]
                    .as_object_mut()
                    .unwrap()
                    .remove("ple");
            }
            let registry = parse_split(&value).unwrap();
            assert_eq!(
                registry.models["qwen3.8:flash-next"].has_ple_sidecars(),
                keep.is_some()
            );
            // An empty map is never written back.
            let written = serde_json::to_value(&registry).unwrap();
            assert_eq!(
                written["models"]["qwen3.8:flash-next"].get("ple").is_some(),
                keep.is_some()
            );
        }
    }

    #[test]
    fn ple_slot_validation_rejects_misplaced_or_malformed_sidecars() {
        let cases: [(&str, Box<dyn Fn(&mut serde_json::Value)>, &str); 7] = [
            (
                "slot on a monolithic file",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["file"] = "qwen3.8-flash-next.mq4-xts".into();
                    entry["ple"]["q8"]["file"] = "qwen3.8-flash-next.mq4-xts.ple-q8".into();
                    entry["ple"]["q4"]["file"] = "qwen3.8-flash-next.mq4-xts.ple-q4".into();
                }),
                "not a split base",
            ),
            (
                "q8 file does not follow the base",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q8"]["file"] = "other.ple-q8".into()
                }),
                "must be",
            ),
            (
                "q4 slot names the q8 file",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q4"]["file"] = format!("{SPLIT_BASE_FILE}.ple-q8").into();
                }),
                "must be",
            ),
            (
                "sidecar in a subdirectory",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q4"]["file"] = format!("sub/{SPLIT_BASE_FILE}.ple-q4").into();
                }),
                "must be",
            ),
            (
                "bad q8 digest",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q8"]["sha256"] = "8b15b6".into()
                }),
                "invalid SHA-256",
            ),
            (
                "empty q4 file",
                Box::new(|entry: &mut serde_json::Value| entry["ple"]["q4"]["file"] = "  ".into()),
                "empty sidecar file",
            ),
            (
                "empty stem before -base",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["file"] = "-base.mq4-xts".into();
                    entry["ple"]["q8"]["file"] = "-base.mq4-xts.ple-q8".into();
                    entry["ple"]["q4"]["file"] = "-base.mq4-xts.ple-q4".into();
                }),
                "not a split base",
            ),
        ];
        for (name, mutate, needle) in cases {
            let mut value = split_registry_json();
            mutate(&mut value["models"]["qwen3.8:flash-next"]);
            let error = parse_split(&value).expect_err(name).to_string();
            assert!(error.contains(needle), "{name}: {error}");
        }
        let mut value = split_registry_json();
        value["models"]["qwen3.8:flash-next"]["ple"]["q8"]["unknown"] = 1.into();
        assert!(
            parse_split(&value).is_err(),
            "sidecar slots stay deny_unknown_fields"
        );
    }

    const SHARED_Q4: &str = "qwen3.8-flash-next.ple-q4";
    const SHARED_Q8: &str = "qwen3.8-flash-next.ple-q8";

    /// The split registry with the shared Flash-Next pair on the MQ4 split base
    /// and on a second (MQ3) split base of the same family.
    fn shared_pair_registry_json() -> serde_json::Value {
        let mut value = split_registry_json();
        let models = value["models"].as_object_mut().unwrap();
        let mut mq3 = models["qwen3.8:flash-next"].clone();
        mq3["file"] = "qwen3.8-flash-next-base.mq3-xts".into();
        models.insert("qwen3.8:flash-next-mq3".into(), mq3);
        for tag in ["qwen3.8:flash-next", "qwen3.8:flash-next-mq3"] {
            models[tag]["ple"]["q4"]["file"] = SHARED_Q4.into();
            models[tag]["ple"]["q8"]["file"] = SHARED_Q8.into();
        }
        value
    }

    #[test]
    fn flash_next_split_bases_may_share_one_ple_pair() {
        let registry = parse_split(&shared_pair_registry_json()).unwrap();
        for tag in ["qwen3.8:flash-next", "qwen3.8:flash-next-mq3"] {
            let entry = &registry.models[tag];
            assert_eq!(entry.ple["q4"].file, SHARED_Q4, "{tag}");
            assert_eq!(entry.ple["q8"].file, SHARED_Q8, "{tag}");
        }
        // Own per-base names stay valid next to the shared ones (mixed slots).
        let mut mixed = shared_pair_registry_json();
        mixed["models"]["qwen3.8:flash-next-mq3"]["ple"]["q8"]["file"] =
            "qwen3.8-flash-next-base.mq3-xts.ple-q8".into();
        parse_split(&mixed).unwrap();
        assert_eq!(
            shared_ple_file("qwen3.8-flash-next-base.mq3-xts", "q4").as_deref(),
            Some(SHARED_Q4)
        );
        assert_eq!(
            shared_ple_file("qwen3.8-flash-next-base.mq4-xts", "q8").as_deref(),
            Some(SHARED_Q8)
        );
        for (file, why) in [
            ("qwen3.8-flash-next.mq4-xts", "monolithic"),
            ("model-base.mq4", "other family"),
            ("Qwen3.8-Flash-Next-base.mq4-xts", "case"),
            ("dir/qwen3.8-flash-next-base.mq4-xts", "path"),
        ] {
            assert_eq!(shared_ple_file(file, "q4"), None, "{file}: {why}");
        }
    }

    #[test]
    fn shared_ple_names_are_refused_outside_the_flash_next_split_family() {
        let cases: [(&str, Box<dyn Fn(&mut serde_json::Value)>, &str); 6] = [
            (
                "shared file on the monolithic mq4 entry",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["file"] = "qwen3.8-flash-next.mq4-xts".into();
                }),
                "not a split base",
            ),
            (
                "another family cannot borrow the Flash-Next pair",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["file"] = "other-model-base.mq4".into();
                }),
                "must be 'other-model-base.mq4.ple-q",
            ),
            (
                "q4 slot names the shared q8",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q4"]["file"] = SHARED_Q8.into();
                }),
                "must be",
            ),
            (
                "arbitrary family file",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q4"]["file"] = "qwen3.8-flash-next.other-q4".into();
                }),
                "must be",
            ),
            (
                "shared name in a subdirectory",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q8"]["file"] = format!("sub/{SHARED_Q8}").into();
                }),
                "must be",
            ),
            (
                "different case",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple"]["q8"]["file"] = "Qwen3.8-Flash-Next.ple-q8".into();
                }),
                "must be",
            ),
        ];
        for (name, mutate, needle) in cases {
            let mut value = shared_pair_registry_json();
            mutate(&mut value["models"]["qwen3.8:flash-next"]);
            let error = parse_split(&value).expect_err(name).to_string();
            assert!(error.contains(needle), "{name}: {error}");
        }
    }

    #[test]
    fn bases_naming_the_same_shared_file_must_pin_the_same_bytes() {
        for (field, value) in [
            ("sha256", serde_json::Value::from(digest('d'))),
            ("size_bytes", serde_json::Value::from(1_u64)),
        ] {
            let mut value_json = shared_pair_registry_json();
            value_json["models"]["qwen3.8:flash-next-mq3"]["ple"]["q4"][field] = value;
            let error = parse_split(&value_json).unwrap_err().to_string();
            assert!(error.contains("pinned differently"), "{field}: {error}");
        }
    }

    #[test]
    fn ple_map_keys_are_format_tokens_and_unknown_formats_parse_and_validate() {
        for (key, ok) in [
            ("q4", true),
            ("q8", true),
            ("q2", true),
            ("nf4", true),
            ("q10", true),
            ("", false),
            ("Q4", false),
            ("q-4", false),
            ("q_4", false),
            ("q4 ", false),
            ("q4.x", false),
        ] {
            assert_eq!(is_ple_format_token(key), ok, "{key:?}");
        }
        assert_eq!(ple_sidecar_suffix("q2"), ".ple-q2");
        assert_eq!(
            shared_ple_file("qwen3.8-flash-next-base.mq4-xts", "q2").as_deref(),
            Some("qwen3.8-flash-next.ple-q2")
        );
        assert_eq!(
            shared_ple_file("qwen3.8-flash-next-base.mq4-xts", "Q2"),
            None
        );
        assert_eq!(shared_ple_file("qwen3.8-flash-next-base.mq4-xts", ""), None);

        // A key this client cannot load (q2) is valid, own-named or shared.
        let mut own = shared_pair_registry_json();
        own["models"]["qwen3.8:flash-next"]["ple"]["q2"] = serde_json::json!({
            "file": format!("{SPLIT_BASE_FILE}.ple-q2"),
            "sha256": digest('e'),
            "size_bytes": 14_000_000_000_u64,
        });
        let registry = parse_split(&own).unwrap();
        let entry = &registry.models["qwen3.8:flash-next"];
        assert_eq!(
            entry.ple.keys().map(String::as_str).collect::<Vec<_>>(),
            ["q2", "q4", "q8"]
        );
        assert_eq!(
            entry.ple_sidecar("q2").unwrap().size_bytes,
            Some(14_000_000_000)
        );
        assert!(entry.ple_sidecar("q3").is_none());
        assert_eq!(PLE_EXACT_FORMAT, "q8");
        // The MQ4 XTS split base is a view of the published monolithic (and its
        // own-named q2 rows are not shared), so it needs a pin.
        assert!(!entry.is_split_only());
        let reparsed = parse_split(&serde_json::to_value(&registry).unwrap()).unwrap();
        assert_eq!(registry, reparsed);

        // The shared q2 file is pinned identically across entries, like q4/q8.
        let mut shared = shared_pair_registry_json();
        for (tag, sha) in [
            ("qwen3.8:flash-next", digest('e')),
            ("qwen3.8:flash-next-mq3", digest('e')),
        ] {
            shared["models"][tag]["ple"]["q2"] = serde_json::json!({
                "file": "qwen3.8-flash-next.ple-q2",
                "sha256": sha,
                "size_bytes": 14_000_000_000_u64,
            });
        }
        let registry = parse_split(&shared).unwrap();
        assert!(registry.models["qwen3.8:flash-next-mq3"].is_split_only());
        shared["models"]["qwen3.8:flash-next-mq3"]["ple"]["q2"]["sha256"] = digest('f').into();
        let error = parse_split(&shared).unwrap_err().to_string();
        assert!(error.contains("pinned differently"), "{error}");
        assert!(error.contains("ple.q2"), "{error}");

        // Invalid key spellings are refused with a clear message.
        for key in ["Q4", "q-4", "q_4", ""] {
            let mut bad = split_registry_json();
            bad["models"]["qwen3.8:flash-next"]["ple"][key] = serde_json::json!({
                "file": format!("{SPLIT_BASE_FILE}.ple-{key}"),
            });
            let error = parse_split(&bad).unwrap_err().to_string();
            assert!(error.contains("invalid PLE format key"), "{key:?}: {error}");
        }

        // A shared unknown-key file outside the Flash-Next family stays refused.
        let mut foreign = split_registry_json();
        foreign["models"]["qwen3.8:flash-next"]["file"] = "other-model-base.mq4".into();
        foreign["models"]["qwen3.8:flash-next"]["ple"]["q2"] = serde_json::json!({
            "file": "qwen3.8-flash-next.ple-q2",
        });
        let error = parse_split(&foreign).unwrap_err().to_string();
        assert!(error.contains("must be"), "{error}");
    }

    #[test]
    fn split_base_file_helpers_mirror_the_converter_naming() {
        for (base, monolithic) in [
            (
                "qwen3.8-flash-next-base.mq4-xts",
                Some("qwen3.8-flash-next.mq4-xts"),
            ),
            ("model-base.mq4", Some("model.mq4")),
            ("model-base", Some("model")),
            ("model.mq4", None),
            ("-base.mq4", None),
            ("model-base.mq4.ple-q8", None),
            ("dir/model-base.mq4", None),
            ("", None),
        ] {
            assert_eq!(
                monolithic_file_for_split_base(base).as_deref(),
                monolithic,
                "{base}"
            );
            assert_eq!(is_split_base_file(base), monolithic.is_some(), "{base}");
        }
    }

    #[test]
    fn grandfather_pin_binds_exactly_the_published_mq4_xts_split_base() {
        let registry = parse_split(&split_registry_json()).unwrap();
        let split = &registry.models["qwen3.8:flash-next"];
        let pin = split.grandfathered_monolithic().unwrap();
        assert_eq!(pin.canonical_file, FLASH_NEXT_XTS_CANONICAL_FILE);
        assert_eq!(pin.legacy_file, FLASH_NEXT_XTS_LEGACY_FILE);
        assert_eq!(pin.sha256, FLASH_NEXT_XTS_SHA256);
        assert_eq!(pin.size_bytes, FLASH_NEXT_XTS_SIZE);
        assert_eq!(pin.size_bytes, 125_288_544_792);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(pin.split_file, FLASH_NEXT_XTS_SPLIT_BASE_FILE);
        assert_eq!(pin.split_sha256, FLASH_NEXT_XTS_SPLIT_BASE_SHA256);
        assert_eq!(pin.split_size_bytes, 70_888_270_360);
        assert_eq!(
            Some(pin.canonical_file.to_owned()),
            monolithic_file_for_split_base(pin.split_file)
        );
        // The monolithic itself (no PLE slots) grandfathers nothing.
        assert!(pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE)
            .grandfathered_monolithic()
            .is_none());
        assert!(pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE)
            .grandfathered_monolithic()
            .is_none());

        // Every part of the split-base identity is required.
        for mismatch in 0..7 {
            let mut entry = stock_split_base_entry();
            match mismatch {
                0 => entry.repo = "other/qwen3.8-flash-next".into(),
                1 => entry.file = "qwen3.8-flash-next-base.mq3-xts".into(),
                2 => entry.size_bytes = Some(FLASH_NEXT_XTS_SPLIT_BASE_SIZE - 1),
                3 => entry.sha256 = Some(digest('a')),
                4 => entry.sha256 = None,
                5 => entry.size_bytes = None,
                6 => {
                    entry.ple.clear();
                }
                _ => unreachable!(),
            }
            assert!(entry.grandfathered_monolithic().is_none(), "{mismatch}");
        }
        assert!(stock_split_base_entry()
            .grandfathered_monolithic()
            .is_some());
    }

    #[test]
    fn grandfathered_monolithic_in_uses_the_supplied_pin_table() {
        let mut entry = stock_split_base_entry();
        entry.file = "custom-base.mq4".into();
        entry.sha256 = Some(digest('b'));
        entry.size_bytes = Some(123);
        assert!(grandfathered_monolithic_in(&entry, GRANDFATHERED_MONOLITHICS).is_none());
        let pins = [GrandfatheredMonolithic {
            repo: FLASH_NEXT_XTS_REPO,
            split_file: "custom-base.mq4",
            split_sha256: "B".repeat(64).leak(),
            split_size_bytes: 123,
            canonical_file: "custom.mq4",
            legacy_file: "custom-old.mq4",
            sha256: "c".repeat(64).leak(),
            size_bytes: 7,
        }];
        let found = grandfathered_monolithic_in(&entry, &pins).unwrap();
        assert_eq!(found.canonical_file, "custom.mq4");
        assert!(grandfathered_monolithic_in(&entry, &[]).is_none());
    }

    /// Split tags that are views of a monolithic (not split-only) without a
    /// compiled-in grandfather pin.
    fn missing_grandfather_pins(registry: &RegistryV1) -> Vec<&str> {
        registry
            .models
            .iter()
            .filter(|(_, entry)| entry.has_ple_sidecars() && !entry.is_split_only())
            .filter(|(_, entry)| entry.grandfathered_monolithic().is_none())
            .map(|(tag, _)| tag.as_str())
            .collect()
    }

    #[test]
    fn bundled_split_entries_that_are_views_of_a_monolithic_have_a_grandfather_pin() {
        let registry = bundled().unwrap();
        assert_eq!(
            missing_grandfather_pins(&registry),
            Vec::<&str>::new(),
            "re-publishing a split view of a monolithic must update the compiled-in pin"
        );
    }

    #[test]
    fn only_a_shared_pair_split_base_without_a_published_monolithic_is_split_only() {
        let mut value = shared_pair_registry_json();
        for tag in ["qwen3.8:flash-next-mq4", "qwen3.8:flash-next-mq4-xts"] {
            value["models"].as_object_mut().unwrap().remove(tag);
        }
        let registry = parse_split(&value).unwrap();
        // Both slots on the shared rows-bound pair and no published monolithic.
        assert!(registry.models["qwen3.8:flash-next-mq3"].is_split_only());
        // The MQ4 XTS split base is a view of the published monolithic even on
        // the shared pair; its compiled-in pin keeps that monolithic serving it.
        assert!(!registry.models["qwen3.8:flash-next"].is_split_only());
        assert_eq!(missing_grandfather_pins(&registry), Vec::<&str>::new());

        // Re-publishing the MQ4 XTS base (new bytes) without updating the pin fails.
        let mut republished = value.clone();
        republished["models"]["qwen3.8:flash-next"]["sha256"] = digest('d').into();
        let republished = parse_split(&republished).unwrap();
        assert_eq!(
            missing_grandfather_pins(&republished),
            ["qwen3.8:flash-next"]
        );

        // Without the marker (one slot naming its own per-base rows) the MQ3
        // split base is a view of an unpinned monolithic: it fails too.
        value["models"]["qwen3.8:flash-next-mq3"]["ple"]["q8"]["file"] =
            "qwen3.8-flash-next-base.mq3-xts.ple-q8".into();
        let unmarked = parse_split(&value).unwrap();
        assert!(!unmarked.models["qwen3.8:flash-next-mq3"].is_split_only());
        assert_eq!(
            missing_grandfather_pins(&unmarked),
            ["qwen3.8:flash-next-mq3"]
        );
        // A monolithic entry is never split-only.
        let split = parse_split(&split_registry_json()).unwrap();
        assert!(!split.models["qwen3.8:flash-next-mq4-xts"].is_split_only());
    }

    #[test]
    fn flash_next_xts_normalization_requires_the_complete_pin_and_exact_filename() {
        let wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        let original = wire.models["qwen3.8:flash-next"].clone();
        assert!(original.is_flash_next_xts());
        for mismatch in 0..7 {
            let mut entry = original.clone();
            match mismatch {
                0 => entry.repo = "other/qwen3.8-flash-next".into(),
                1 => entry.size_bytes = Some(FLASH_NEXT_XTS_SIZE - 1),
                2 => entry.sha256 = Some("a".repeat(64)),
                3 => entry.file = "qwen3.8-flash-next.mq4".into(),
                4 => entry.file = format!("subdir/{FLASH_NEXT_XTS_LEGACY_FILE}"),
                5 => entry.sha256 = None,
                6 => entry.size_bytes = None,
                _ => unreachable!(),
            }
            assert!(!entry.is_flash_next_xts());
            let mut changed = wire.clone();
            changed
                .models
                .insert("qwen3.8:flash-next".into(), entry.clone());
            let parsed =
                RegistryV1::parse(&serde_json::to_string(&changed).unwrap(), "fixture").unwrap();
            assert_eq!(parsed.models["qwen3.8:flash-next"], entry);
        }
        let parsed = RegistryV1::parse(&serde_json::to_string(&wire).unwrap(), "fixture").unwrap();
        assert_eq!(
            parsed.model("fixture").unwrap().1.file,
            FLASH_NEXT_XTS_CANONICAL_FILE
        );
        let reparsed =
            RegistryV1::parse(&serde_json::to_string(&parsed).unwrap(), "fixture").unwrap();
        assert_eq!(parsed, reparsed);
        assert!(parsed
            .model(FLASH_NEXT_XTS_LEGACY_FILE)
            .unwrap()
            .1
            .is_flash_next_xts());
        assert!(parsed
            .entry_for_file(FLASH_NEXT_XTS_LEGACY_FILE)
            .unwrap()
            .1
            .is_flash_next_xts());
        for path in [
            format!("./{FLASH_NEXT_XTS_LEGACY_FILE}"),
            format!("/tmp/{FLASH_NEXT_XTS_LEGACY_FILE}"),
        ] {
            assert!(parsed.model(&path).is_none());
            assert!(parsed.entry_for_file(&path).is_none());
        }
        let mut vintage = wire.clone();
        vintage.models.remove("qwen3.8:flash-next-mq4-xts");
        let parsed =
            RegistryV1::parse(&serde_json::to_string(&vintage).unwrap(), "v0.4.1 remote").unwrap();
        for entry in parsed.models.values() {
            assert!(entry.is_flash_next_xts());
            assert_eq!(entry.file, FLASH_NEXT_XTS_CANONICAL_FILE);
        }
        let mut invalid = wire;
        invalid.models.get_mut("qwen3.8:flash-next").unwrap().sha256 = Some("8b15b6".into());
        assert!(RegistryV1::parse(&serde_json::to_string(&invalid).unwrap(), "fixture").is_err());
    }

    struct RegistryTestDir(PathBuf);

    impl RegistryTestDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = env::temp_dir().join(format!(
                "hipfire-registry-{}-{}-{}",
                std::process::id(),
                epoch_millis(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for RegistryTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn serve_registry_once(registry: &RegistryV1) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!(
            "http://{}/alternate-registry.json",
            listener.local_addr().unwrap()
        );
        let body = serde_json::to_string(registry).unwrap();
        let handle = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = [0; 4096];
            let read = socket.read(&mut request).unwrap();
            assert!(read > 0, "client sent no request");
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            ).unwrap();
        });
        (url, handle)
    }

    #[test]
    fn flash_next_xts_normalizes_bundle_remote_fresh_and_stale_cache_and_alternate_url() {
        let dir = RegistryTestDir::new();
        let paths = RegistryPaths {
            cache: dir.0.join("registry.cache.json"),
        };
        let offline = load_with_options(&paths, None);
        assert_eq!(offline.source, RegistrySource::Bundled);
        assert_stock_split_tags(&offline.registry);
        assert!(
            !paths.cache.exists(),
            "bundle-only must not fetch or write cache"
        );

        let wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        let (url, server) = serve_registry_once(&wire);
        let remote = load_with_options(&paths, Some(&url));
        server.join().unwrap();
        assert_eq!(remote.source, RegistrySource::Network);
        assert_flash_next_xts(&remote.registry, &FLASH_NEXT_FIXTURE_TAGS);
        let cached = load_with_options(&paths, Some(&url));
        assert_eq!(cached.source, RegistrySource::Cache);
        assert_flash_next_xts(&cached.registry, &FLASH_NEXT_FIXTURE_TAGS);

        // Simulate a pre-upgrade cache: deserialization alone has not normalized it.
        let legacy_cache = RegistryCache {
            fetched_at: epoch_millis(),
            url: url.clone(),
            registry: wire.clone(),
        };
        write_cache(&paths.cache, &legacy_cache).unwrap();
        let cached = load_with_options(&paths, Some(&url));
        assert_eq!(cached.source, RegistrySource::Cache);
        assert_flash_next_xts(&cached.registry, &FLASH_NEXT_FIXTURE_TAGS);
        assert!(cached
            .registry
            .model("fixture")
            .unwrap()
            .1
            .is_flash_next_xts());

        let mut stale = legacy_cache;
        stale.fetched_at = 0;
        write_cache(&paths.cache, &stale).unwrap();
        let loaded = load_with_options(&paths, Some(&url));
        assert_eq!(loaded.source, RegistrySource::StaleCache);
        assert_flash_next_xts(&loaded.registry, &FLASH_NEXT_FIXTURE_TAGS);
        assert!(!loaded.warnings.is_empty());

        let (alternate, server) = serve_registry_once(&wire);
        assert_ne!(url, alternate);
        let loaded = load_with_options(&paths, Some(&alternate));
        server.join().unwrap();
        assert_eq!(loaded.source, RegistrySource::Network, "cache is URL-bound");
        assert_flash_next_xts(&loaded.registry, &FLASH_NEXT_FIXTURE_TAGS);
        let offline = load_with_options(&paths, None);
        assert_eq!(
            offline.source,
            RegistrySource::Bundled,
            "disabled fetch ignores cache"
        );

        let old_wire = flash_next_wire_registry("2000-01-01T00:00:00Z");
        let (old_url, server) = serve_registry_once(&old_wire);
        let loaded = load_with_options(&paths, Some(&old_url));
        server.join().unwrap();
        assert_eq!(loaded.source, RegistrySource::Bundled, "newer bundle wins");
        assert_stock_split_tags(&loaded.registry);
        assert!(loaded
            .warnings
            .iter()
            .any(|warning| warning.contains("newer")));
    }

    #[test]
    fn retired_pins_are_the_dropped_flash_next_artifacts_with_their_original_identity() {
        let tags: Vec<&str> = RETIRED_MODELS.iter().map(|pin| pin.tag).collect();
        assert_eq!(
            tags,
            [
                "qwen3.8:flash-next-rtn-asym",
                "qwen3.8:flash-next-mq6q8-pleq8"
            ]
        );
        let rtn = &RETIRED_MODELS[0];
        assert_eq!(rtn.file, "qwen3.8-flash-next.mq4");
        assert_eq!(rtn.size_bytes, 125_288_540_696);
        assert_eq!(
            rtn.sha256,
            "8aa01cf41bf2a90b319b9a1c70837baf51a2f92811af551f59b59d418518f650"
        );
        let pleq8 = &RETIRED_MODELS[1];
        assert_eq!(pleq8.file, "qwen3.8-flash-next.mq6q8-pleq8");
        assert_eq!(pleq8.size_bytes, 125_467_331_096);
        assert_eq!(
            pleq8.sha256,
            "c0628b848077f02afed9ce5a4daa0598c379aa1d1e5ca0c0b24ddd04b49773a9"
        );
        for pin in RETIRED_MODELS {
            assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO, "{}", pin.tag);
            assert_eq!(pin.replacement, "qwen3.8:flash-next", "{}", pin.tag);
            let entry = pin.entry();
            assert_eq!(entry.file, pin.file);
            assert_eq!(entry.sha256.as_deref(), Some(pin.sha256));
            assert_eq!(entry.size_bytes, Some(pin.size_bytes));
            assert_eq!(entry.arch_id, Some(16));
            assert!(!entry.has_ple_sidecars());
            assert!(!entry.is_flash_next_xts(), "{}", pin.tag);
            assert!(entry.grandfathered_monolithic().is_none());
            // The card lowers to the same sampling defaults it published.
            config_layer_for_tag(pin.tag, &entry).unwrap();
            let settings = entry.recommended_settings.as_ref().unwrap();
            assert_eq!(settings.temperature, Some(1.0));
            assert_eq!(settings.top_p, Some(0.95));
            assert_eq!(settings.top_k, Some(20));
            // No two pins share a tag or a file, and none collides with the
            // grandfathered monolithic spellings.
            assert_eq!(
                RETIRED_MODELS
                    .iter()
                    .filter(|other| other.tag == pin.tag || other.file == pin.file)
                    .count(),
                1
            );
            assert_ne!(pin.file, FLASH_NEXT_XTS_CANONICAL_FILE);
            assert_ne!(pin.file, FLASH_NEXT_XTS_LEGACY_FILE);
            assert_ne!(pin.sha256, FLASH_NEXT_XTS_SHA256);
        }
    }

    #[test]
    fn retired_refusal_is_one_line_naming_the_replacement() {
        for pin in RETIRED_MODELS {
            let message = pin.refusal();
            assert!(!message.contains('\n'));
            assert_eq!(
                message,
                format!(
                    "{} is retired; the file still runs if you have it; replacement \
                     qwen3.8:flash-next",
                    pin.tag
                )
            );
        }
    }

    #[test]
    fn retired_cards_are_cached_for_compiled_pins_and_owned_for_any_other_identity() {
        for pin in RETIRED_MODELS {
            let card = pin.card();
            assert!(matches!(card, std::borrow::Cow::Borrowed(_)), "{}", pin.tag);
            assert_eq!(*card, pin.entry());
            // Every query borrows the same cached card.
            let again = pin.card();
            assert!(std::ptr::eq(&*card, &*again));
        }
        // A fixture with a different digest is not a compiled identity: it gets
        // its own card, never another pin's.
        let fixture = RetiredModel {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            size_bytes: 7,
            ..RETIRED_MODELS[0]
        };
        let card = fixture.card();
        assert!(matches!(card, std::borrow::Cow::Owned(_)));
        assert_eq!(card.size_bytes, Some(7));
        assert_eq!(card.sha256.as_deref(), Some(fixture.sha256));
    }

    #[test]
    fn retired_model_named_matches_a_tag_or_bare_file_but_never_a_path() {
        let rtn = &RETIRED_MODELS[0];
        assert_eq!(retired_model_named(rtn.tag, RETIRED_MODELS), Some(rtn));
        assert_eq!(retired_model_named(rtn.file, RETIRED_MODELS), Some(rtn));
        for lookalike in [
            format!("./{}", rtn.file),
            format!("/tmp/{}", rtn.file),
            format!("models\\{}", rtn.file),
            format!("{}x", rtn.file),
            "qwen3.8:flash-next".to_owned(),
            "qwen3.8:flash-next-mq4".to_owned(),
            FLASH_NEXT_XTS_CANONICAL_FILE.to_owned(),
            FLASH_NEXT_XTS_LEGACY_FILE.to_owned(),
        ] {
            assert!(
                retired_model_named(&lookalike, RETIRED_MODELS).is_none(),
                "{lookalike}"
            );
        }
        assert!(retired_model_named(rtn.tag, &[]).is_none());
    }

    #[test]
    fn retired_files_stay_listable_though_the_catalog_no_longer_names_them() {
        let registry = bundled().unwrap();
        for pin in RETIRED_MODELS {
            assert!(
                names_registry_model(pin.file, &registry, GRANDFATHERED_MONOLITHICS),
                "{}",
                pin.file
            );
            assert!(is_listable_model_file(
                pin.file,
                &registry,
                GRANDFATHERED_MONOLITHICS
            ));
        }
        // Sidecars of a retired file are still never listed as models.
        assert!(!is_listable_model_file(
            "qwen3.8-flash-next.mq6q8-pleq8.ple-q4",
            &registry,
            GRANDFATHERED_MONOLITHICS
        ));
    }

    #[test]
    fn retired_tags_are_stripped_from_every_delivered_catalog_view() {
        let mut wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        for pin in RETIRED_MODELS {
            wire.models.insert(pin.tag.to_owned(), pin.entry());
            wire.aliases
                .insert(format!("{}-alias", pin.tag), pin.tag.to_owned());
        }
        // A strict parse of an older payload that still carries the retired
        // entries drops them (and the aliases that pointed at them).
        let parsed =
            RegistryV1::parse(&serde_json::to_string(&wire).unwrap(), "old remote").unwrap();
        for pin in RETIRED_MODELS {
            assert!(parsed.model(pin.tag).is_none(), "{}", pin.tag);
            assert!(parsed.model(pin.file).is_none(), "{}", pin.file);
            assert!(
                !parsed.aliases.contains_key(&format!("{}-alias", pin.tag)),
                "{}",
                pin.tag
            );
        }
        assert_flash_next_xts(&parsed, &FLASH_NEXT_FIXTURE_TAGS);

        let dir = RegistryTestDir::new();
        let paths = RegistryPaths {
            cache: dir.0.join("registry.cache.json"),
        };
        let (url, server) = serve_registry_once(&wire);
        let remote = load_with_options(&paths, Some(&url));
        server.join().unwrap();
        assert_eq!(remote.source, RegistrySource::Network);
        for pin in RETIRED_MODELS {
            assert!(!remote.registry.models.contains_key(pin.tag), "network");
        }
        // A cache written before the retirement still carries the tags.
        let legacy_cache = RegistryCache {
            fetched_at: epoch_millis(),
            url: url.clone(),
            registry: wire.clone(),
        };
        write_cache(&paths.cache, &legacy_cache).unwrap();
        let cached = load_with_options(&paths, Some(&url));
        assert_eq!(cached.source, RegistrySource::Cache);
        let mut stale = legacy_cache;
        stale.fetched_at = 0;
        write_cache(&paths.cache, &stale).unwrap();
        let stale_loaded = load_with_options(&paths, Some(&url));
        assert_eq!(stale_loaded.source, RegistrySource::StaleCache);
        for loaded in [&cached, &stale_loaded] {
            for pin in RETIRED_MODELS {
                assert!(
                    !loaded.registry.models.contains_key(pin.tag),
                    "{:?}",
                    loaded.source
                );
            }
        }
        // The bundled offline floor never lists them either.
        let bundled = bundled().unwrap();
        for pin in RETIRED_MODELS {
            assert!(!bundled.models.contains_key(pin.tag));
        }
    }

    #[test]
    fn config_layer_omits_universal_q8_fallback_but_lowers_model_opinions() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "fallback":{"repo":"r","file":"fallback.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","default_kv_mode":"q8"},
                "opinion":{"repo":"r","file":"opinion.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","default_kv_mode":"fwht3"}
            },
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        let fallback = registry
            .model("fallback")
            .unwrap()
            .1
            .config_layer()
            .unwrap();
        assert!(fallback.get("memory.kv_cache").is_none());

        let opinion = registry.model("opinion").unwrap().1.config_layer().unwrap();
        assert_eq!(
            opinion.get("memory.kv_cache"),
            Some(&ConfigValue::String("fwht3".into()))
        );
    }

    /// The `heads` map must be VALIDATED, not merely parsed.
    ///
    /// Adding a field to the struct makes it round-trip; it does not make the
    /// validator look at it. This asserts the negative directly: a head with a
    /// malformed digest is REJECTED. Without the `.chain(entry.heads.values())`
    /// in `validate`, this test fails and the bundled-registry check would
    /// happily ship an unverifiable head overlay.
    #[test]
    fn heads_sidecars_are_digest_validated() {
        let with_bad_head = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{"repo":"r","file":"f.hfq","size_gb":1,"min_vram_gb":1,"desc":"d",
              "heads":{"q4k":{"file":"h.hfq","sha256":"not-a-sha"}}}},
            "aliases":{}
        }"#;
        let err = RegistryV1::parse(with_bad_head, "test")
            .expect_err("a malformed head digest must be rejected");
        assert!(
            format!("{err}").contains("invalid SHA-256"),
            "expected a digest complaint, got: {err}"
        );

        // Control: the SAME registry with a well-formed digest parses, so the
        // rejection above is about the digest and not about `heads` being
        // unparseable.
        let good = with_bad_head.replace("not-a-sha", &"a".repeat(64));
        let reg = RegistryV1::parse(&good, "test").expect("valid head must parse");
        let (_, entry) = reg.model("m").unwrap();
        assert_eq!(entry.heads.len(), 1);
        assert_eq!(entry.heads["q4k"].file, "h.hfq");
    }
    /// The `xdna` sidecar must be VALIDATED like every other sidecar slot:
    /// a malformed digest is REJECTED and a well-formed one parses onto the
    /// entry. Without `&entry.xdna` in the `validate` walk, an unverifiable
    /// `.xdna.zip` declaration would ship silently.
    #[test]
    fn xdna_sidecar_is_digest_validated() {
        let with_bad_xdna = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{"repo":"r","file":"f.hfq","size_gb":1,"min_vram_gb":1,"desc":"d",
              "xdna":{"file":"m.xdna.zip","sha256":"not-a-sha"}}},
            "aliases":{}
        }"#;
        let err = RegistryV1::parse(with_bad_xdna, "test")
            .expect_err("a malformed xdna digest must be rejected");
        assert!(
            format!("{err}").contains("invalid SHA-256"),
            "expected a digest complaint, got: {err}"
        );
        let good = with_bad_xdna.replace("not-a-sha", &"b".repeat(64));
        let reg = RegistryV1::parse(&good, "test").expect("valid xdna must parse");
        let (_, entry) = reg.model("m").unwrap();
        assert_eq!(entry.xdna.as_ref().unwrap().file, "m.xdna.zip");
    }

    fn reg_at(stamp: &str) -> RegistryV1 {
        RegistryV1::parse(
            &format!(
                r#"{{"schema_version":1,"generated_at":"{stamp}",
                   "models":{{"m":{{"repo":"r","file":"f","size_gb":1,"min_vram_gb":1,"desc":"d"}}}},
                   "aliases":{{}}}}"#
            ),
            "test",
        )
        .unwrap()
    }

    /// A NEWER bundled registry wins — this is what makes a branch's registry
    /// edits visible to a locally built binary instead of being silently
    /// overridden by the 24h cache or a master fetch.
    #[test]
    fn newer_bundled_registry_beats_a_stale_fetch() {
        let mut w = Vec::new();
        let fetched = LoadedRegistry {
            registry: reg_at("2026-08-31T05:32:38Z"),
            source: RegistrySource::Cache,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at("2026-09-01T13:14:39Z"), &mut w);
        assert_eq!(out.source, RegistrySource::Bundled);
        assert_eq!(out.registry.generated_at, "2026-09-01T13:14:39Z");
        assert!(
            out.warnings.iter().any(|x| x.contains("newer")),
            "the override must be reported, not silent: {:?}",
            out.warnings
        );
    }

    /// The other direction is what keeps DISTRIBUTION working: a released
    /// binary's bundled registry is older than master's, so the fetch must
    /// still win and users get new models without upgrading. Without this the
    /// change above would freeze every client at its build-time registry.
    #[test]
    fn older_bundled_registry_defers_to_the_fetch() {
        let mut w = Vec::new();
        let fetched = LoadedRegistry {
            registry: reg_at("2026-09-01T13:14:39Z"),
            source: RegistrySource::Network,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at("2026-08-28T08:31:54Z"), &mut w);
        assert_eq!(out.source, RegistrySource::Network);
        assert_eq!(out.registry.generated_at, "2026-09-01T13:14:39Z");
        assert!(out.warnings.is_empty(), "no override, so nothing to report");
    }

    /// Equal stamps must not flap between sources.
    #[test]
    fn equal_timestamps_keep_the_fetched_registry() {
        let mut w = Vec::new();
        let same = "2026-09-01T13:14:39Z";
        let fetched = LoadedRegistry {
            registry: reg_at(same),
            source: RegistrySource::Cache,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at(same), &mut w);
        assert_eq!(out.source, RegistrySource::Cache);
    }

    #[test]
    fn bundled_registry_is_strictly_valid() {
        let registry = bundled().unwrap();
        assert!(registry.models.len() > 20);
        let (tag, model) = registry.model("qwen3.6:35b-a3b-mq4r").unwrap();
        assert_eq!(tag, "qwen3.6:35b-a3b-mq4r");
        assert_eq!(model.quant.as_deref(), Some("mq4r"));
        assert_eq!(
            model.sha256.as_deref(),
            Some("4685c140c46b1a6f31a0fd9053bf09d5faf1d2529d715b84794249b66cde0428")
        );

        let (tag, model) = registry.model("ornith-1.5:35b-a3b-mq4r").unwrap();
        assert_eq!(tag, "ornith-1.5:35b-a3b-mq4r");
        for alias in ["ornith-1.5:fast", "ornith-1.5:35b-a3b-fast"] {
            let (resolved, fast) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(
                resolved, "ornith-1.5:35b-a3b-mq4r",
                "{alias} is the MQ4R speed SKU"
            );
            assert_eq!(fast.file, "ornith-1.5-35b-a3b.mq4r");
        }
        assert_eq!(model.file, "ornith-1.5-35b-a3b.mq4r");
        assert_eq!(model.quant.as_deref(), Some("mq4r"));
        assert_eq!(model.size_bytes, Some(18_700_570_368));
        assert_eq!(
            model.quant_recipe.as_deref(),
            Some("mq4v2-uniform-no-q8-router@10fbf86f")
        );
        assert_eq!(
            model.sha256.as_deref(),
            Some("84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6")
        );
    }

    /// Pins Muse Glimmer's sampling contract to its model card's "Best Practices"
    /// section (`meta-models/Muse-Glimmer-30B`): temperature 1.0, top_p 0.95, top_k 64.
    ///
    /// `top_k` is the one that matters. Before this entry existed the tag resolved to
    /// nothing and callers silently fell back to `recipe(general)`, which uses the Qwen
    /// family's `top_k = 20` — a different model's sampling contract applied to Glimmer
    /// with only a warning on stderr.
    #[test]
    fn bundled_muse_glimmer_matches_the_model_card_sampling_contract() {
        let registry = bundled().unwrap();
        // Only one Glimmer size exists, so the canonical tag carries no size
        // suffix. `muse-glimmer:30b` stays a back-compat alias because
        // scripts/serve_harness.py infers it from the artifact filename.
        let (tag, model) = registry.model("muse-glimmer").unwrap();
        assert_eq!(tag, "muse-glimmer");
        assert_eq!(model.arch_id, Some(14));
        assert_eq!(model.file, "muse-glimmer-30b.mq4");
        for alias in [
            "muse-glimmer:30b",
            "muse-glimmer:latest",
            "muse-glimmer:quality",
        ] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(resolved, "muse-glimmer", "{alias} resolves to the trunk");
        }

        // The speed SKU is a distinct entry on the MQ4-attention artifact, not an
        // alias of the trunk.
        let (fast_tag, fast) = registry.model("muse-glimmer:fast").unwrap();
        assert_eq!(fast_tag, "muse-glimmer:fast");
        assert_eq!(fast.file, "muse-glimmer-30b.mq4r");
        assert_eq!(fast.arch_id, Some(14));
        assert_ne!(
            fast.file, model.file,
            "the two SKUs are different artifacts"
        );

        // The drafter is arch 23 and pairs with either SKU.
        let (draft_tag, draft) = registry.model("muse-glimmer:draft").unwrap();
        assert_eq!(draft_tag, "muse-glimmer:draft");
        assert_eq!(draft.arch_id, Some(23));

        let settings = model
            .recommended_settings
            .as_ref()
            .expect("muse-glimmer must carry the card's recommended_settings");
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(64));
        assert_eq!(settings.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.top_k, Some(64));
        assert_eq!(general.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(
            general.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // The card specifies no separate coding/instruct sampling, so those are
        // deliberately absent rather than invented.
        assert!(model.sampling_profile("coding").is_none());
        assert!(model.sampling_profile("instruct").is_none());
    }

    /// Pins Qwen3.8-27B's release contract: aliases, artifact identity, arch/quant,
    /// default KV, and the upstream sampling card (effort-native; no thinking_budget).
    #[test]
    fn bundled_qwen38_matches_the_release_contract() {
        let registry = bundled().unwrap();
        let (tag, model) = registry.model("qwen3.8:27b").unwrap();
        assert_eq!(tag, "qwen3.8:27b");
        assert_eq!(model.file, "qwen3.8-27b.mq4");
        assert_eq!(model.arch_id, Some(5));
        assert_eq!(model.quant.as_deref(), Some("mq4"));
        assert_eq!(model.default_kv_mode.as_deref(), Some("q8"));
        // Pin the canonical MQ4V2 body by content as well as filename: MQ4V2
        // supersedes the former MQ4V1/MQ4R Qwen3.8 products.
        assert_eq!(
            model.sha256.as_deref(),
            Some("5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507")
        );
        assert_eq!(model.size_bytes, Some(15662615552));

        for alias in ["qwen3.8", "qwen3.8:latest", "qwen3.8:27b-mq4"] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(
                resolved, "qwen3.8:27b",
                "{alias} resolves to the canonical tag"
            );
        }

        // The speed tier is MQ4V2 XT. Both former `fast` tags migrate to it;
        // the superseded `.mq4r` artifact is no longer in the registry.
        let (fast_tag, fast) = registry.model("qwen3.8:27b-fast").unwrap();
        assert_eq!(fast_tag, "qwen3.8:27b-mq4-xt");
        assert_eq!(fast.file, "qwen3.8-27b.mq4-xt");
        assert_eq!(fast.arch_id, Some(5));
        assert_eq!(fast.quant.as_deref(), Some("mq4"));
        assert_eq!(fast.default_kv_mode.as_deref(), Some("q8"));
        assert_eq!(
            fast.sha256.as_deref(),
            Some("80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3")
        );
        assert_eq!(fast.size_bytes, Some(14987185152));
        assert_ne!(
            fast.sha256, model.sha256,
            "the two tiers must not share a content digest"
        );

        for alias in ["qwen3.8:fast", "qwen3.8:27b-fast"] {
            let (resolved, _) = registry.model(alias).expect("fast alias must resolve");
            assert_eq!(resolved, "qwen3.8:27b-mq4-xt");
        }
        assert!(
            registry.model("qwen3.8-27b.mq4r").is_none(),
            "superseded MQ4R filename must not remain addressable"
        );

        // The symmetric XT checkpoint (H2) is the canonical dense fixture. It has
        // exactly the asymmetric XT's size, so only the digest tells them apart;
        // it shares the MQ4 DFlash draft.
        let (xts_tag, xts) = registry.model("qwen3.8:27b-mq4-xts").unwrap();
        assert_eq!(xts_tag, "qwen3.8:27b-mq4-xts");
        assert_eq!(xts.file, "qwen3.8-27b.mq4-xts");
        assert_eq!(xts.arch_id, Some(5));
        assert_eq!(xts.quant.as_deref(), Some("mq4"));
        assert_eq!(xts.default_kv_mode.as_deref(), Some("q8"));
        assert_eq!(
            xts.sha256.as_deref(),
            Some("3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae")
        );
        assert_eq!(xts.size_bytes, Some(14987185152));
        assert_ne!(xts.sha256, fast.sha256, "XTS and XT are distinct artifacts");
        assert_eq!(
            xts.dflash.as_ref().map(|draft| &draft.file),
            fast.dflash.as_ref().map(|draft| &draft.file)
        );

        // The native Qwen3.8-27B MTP head (public checkpoint's `mtp.*` layer,
        // `mtp_extract --quant mq4`) ships as one trunk-independent sidecar:
        // it reuses the trunk's embed/lm_head, so every trunk tier pulls the
        // same bytes, pinned by content.
        let expected_mtp = Sidecar {
            file: "qwen3.8-27b.mtp".into(),
            sha256: Some("f0d46d07ded75abc095ebfdbccc167c68418a6515b187bf810395faa460bd32e".into()),
            size_bytes: Some(225716224),
        };
        for (tag, entry) in [(tag, model), (fast_tag, fast), (xts_tag, xts)] {
            assert_eq!(entry.mtp.as_ref(), Some(&expected_mtp), "{tag} MTP head");
        }
        let mut trunk_tiers = 0;
        for (tag, entry) in &registry.models {
            if !tag.starts_with("qwen3.8:27b") {
                continue;
            }
            if tag.contains("draft") {
                assert!(
                    entry.mtp.is_none(),
                    "{tag}: a DFlash draft carries no MTP head"
                );
            } else {
                assert_eq!(entry.mtp.as_ref(), Some(&expected_mtp), "{tag} MTP head");
                trunk_tiers += 1;
            }
        }
        assert!(
            trunk_tiers >= 13,
            "every Qwen3.8-27B trunk tier declares the head"
        );

        let settings = model
            .recommended_settings
            .as_ref()
            .expect("qwen3.8:27b must carry recommended_settings");
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(20));
        assert_eq!(settings.min_p, Some(0.0));
        assert_eq!(settings.presence_penalty, Some(0.0));
        assert_eq!(settings.repeat_penalty, Some(1.0));
        assert_eq!(settings.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.temperature, Some(1.0));
        assert_eq!(general.top_p, Some(0.95));
        assert_eq!(general.top_k, Some(20));
        assert_eq!(general.min_p, Some(0.0));
        assert_eq!(general.presence_penalty, Some(0.0));
        assert_eq!(general.repeat_penalty, Some(1.0));
        assert_eq!(general.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            general.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // Tag policy pins generation.max_tokens only for Qwen3.8; never max_seq or a KV backend.
        let layer =
            config_layer_for_tag(tag, model).expect("qwen3.8:27b tag policy lowers cleanly");
        assert!(layer.get("memory.kv_cache").is_none());
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(
            layer.get("memory.max_seq").is_none(),
            "tag layer must not write memory.max_seq"
        );
        assert_eq!(
            layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );
        // Tag policy keys off the family before ':' and excludes only draft/dflash
        // tags, so the fast SKU receives the same max_tokens pin (no max_seq/backend).
        let fast_layer = config_layer_for_tag(fast_tag, fast)
            .expect("qwen3.8:27b-fast tag policy lowers cleanly");
        assert!(fast_layer.get("memory.kv_backend").is_none());
        assert!(
            fast_layer.get("memory.max_seq").is_none(),
            "tag layer must not write memory.max_seq"
        );
        assert_eq!(
            fast_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );
        // Flash-Next carries arch_id=16 inside the qwen3.8 naming family;
        // do not inherit the ordinary Qwen3.8 VMM/262K policy.
        let (flash_tag, flash) = registry.model("qwen3.8:flash-next").unwrap();
        assert_eq!(flash.arch_id, Some(16));
        let flash_layer = config_layer_for_tag(flash_tag, flash)
            .expect("qwen3.8:flash-next must lower without a family policy");
        assert!(flash_layer.get("memory.kv_backend").is_none());
        assert!(flash_layer.get("memory.max_seq").is_none());
        assert!(flash_layer.get("generation.max_tokens").is_none());
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("medium".into()))
        );
        // Original Qwen3 family (without .5/.6/.8) receives no automatic policy.
        let qwen3_raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"}},
            "aliases":{}
        }"#;
        let qwen3_registry = RegistryV1::parse(qwen3_raw, "test").unwrap();
        let (q3_tag, q3_entry) = qwen3_registry.model("qwen3:8b").unwrap();
        let q3_layer = config_layer_for_tag(q3_tag, q3_entry).unwrap();
        assert!(q3_layer.get("memory.kv_cache").is_none());
        assert!(q3_layer.get("memory.kv_backend").is_none());
        assert!(q3_layer.get("memory.max_seq").is_none());
        assert!(q3_layer.get("generation.max_tokens").is_none());
    }

    /// Qwen3.8 dense tiers and both Ornith 1.5 artifacts share the effort-native
    /// Qwen3.8 contract: the registry default and general profile are medium.
    /// Flash-Next (Qwen4, arch_id=16), sidecar drafts, and the intentional
    /// Ornith coding/instruct profiles are unchanged.
    #[test]
    fn bundled_qwen38_family_effort_defaults_medium() {
        let registry = bundled().unwrap();
        let mut tiers = 0;
        for (tag, entry) in &registry.models {
            let is_dense_trunk = tag.starts_with("qwen3.8:27b") && !tag.contains("-draft");
            let is_ornith = tag.starts_with("ornith-1.5:");
            if !is_dense_trunk && !is_ornith {
                continue;
            }
            let settings = entry
                .recommended_settings
                .as_ref()
                .unwrap_or_else(|| panic!("{tag} must carry recommended_settings"));
            assert_eq!(
                settings.reasoning_effort.as_deref(),
                Some("medium"),
                "{tag} recommended_settings effort"
            );
            let general = entry
                .sampling_profile("general")
                .unwrap_or_else(|| panic!("{tag} must carry a general profile"));
            assert_eq!(
                general.reasoning_effort.as_deref(),
                Some("medium"),
                "{tag} general profile effort"
            );
            tiers += 1;
        }
        assert!(tiers >= 15, "13 Qwen3.8-27B tiers plus 2 Ornith artifacts");

        // Flash-Next (arch_id=16) never carried an effort default.
        let (_, flash) = registry.model("qwen3.8:flash-next").unwrap();
        assert_eq!(flash.arch_id, Some(16));
        assert!(!flash
            .recommended_settings
            .as_ref()
            .is_some_and(|settings| settings.reasoning_effort.is_some()));
    }

    #[test]
    fn bundled_0731_mq2r_is_default_and_mq2lloyd_stays_addressable() {
        let registry = bundled().unwrap();
        // MQ2R became the default on 2026-08-14 on the DeepSeek V4 contributor's
        // recommendation. The bare tag and the explicit `:mq2r` name must serve
        // the SAME artifact, while `:mq2lloyd` stays a distinct, still-pullable
        // identity — a silent collapse of the two would ship 86 GB under an 82 GB
        // name or vice versa.
        let (_, default_sku) = registry.model("deepseek-v4-flash").unwrap();
        let (_, mq2r) = registry.model("deepseek-v4-flash:mq2r").unwrap();
        let (_, mq2lloyd) = registry.model("deepseek-v4-flash:mq2lloyd").unwrap();

        assert_eq!(default_sku.file, "deepseek-v4-flash-0731.mq2r");
        assert_eq!(mq2r.file, "deepseek-v4-flash-0731.mq2r");
        assert_eq!(mq2lloyd.file, "deepseek-v4-flash-0731.mq2lloyd");
        assert_ne!(mq2r.sha256, mq2lloyd.sha256);

        assert_eq!(default_sku.default_kv_mode.as_deref(), Some("f32"));
        assert_eq!(mq2lloyd.default_kv_mode.as_deref(), Some("f32"));
        assert_eq!(
            mq2r.sha256.as_deref(),
            Some("cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce")
        );
        assert_eq!(
            mq2lloyd.sha256.as_deref(),
            Some("521c9687a3f5c12fb3d89bde4a3ed202698b95ae5a102d0b5ba7f3abb87982d0")
        );
        assert_eq!(
            mq2r.quant_recipe.as_deref(),
            Some("deepseek4-mq2r-e8-p3-v1")
        );

        assert_eq!(
            mq2r.dspark
                .as_ref()
                .and_then(|sidecar| sidecar.sha256.as_deref()),
            Some("bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e")
        );
        assert!(registry.model("deepseek4:preview-mq2r").is_none());

        // Effort-native DeepSeek defaults: main low / coding max / instruct none;
        // no registry thinking_budget (absence = uncapped). Preview defaults high.
        let settings = default_sku.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.reasoning_effort.as_deref(), Some("low"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
        let general = default_sku.sampling_profile("general").unwrap();
        assert_eq!(general.reasoning_effort.as_deref(), Some("low"));
        assert!(general.thinking_budget.is_none());
        let coding = default_sku.sampling_profile("coding").unwrap();
        assert_eq!(coding.reasoning_effort.as_deref(), Some("max"));
        assert!(coding.thinking_budget.is_none());
        let instruct = default_sku.sampling_profile("instruct").unwrap();
        assert_eq!(instruct.reasoning_effort.as_deref(), Some("none"));
        assert!(instruct.thinking_budget.is_none());
        let (_, preview) = registry.model("deepseek-v4-flash-preview").unwrap();
        let preview_rs = preview.recommended_settings.as_ref().unwrap();
        assert_eq!(preview_rs.reasoning_effort.as_deref(), Some("high"));
        assert!(
            preview_rs.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
    }
    #[test]
    fn bundled_raon_takes_the_mq8_upgrade_pin_and_keeps_its_published_entry() {
        // The checked-in (published) entry is unchanged: old clients pull the
        // Q8F16-attention file they can run.
        let raw: serde_json::Value = serde_json::from_str(BUNDLED_REGISTRY).unwrap();
        let published = &raw["models"]["raon-1-mini-preview"];
        assert_eq!(published["file"], "raon-1-mini-preview.mq4-xts");
        assert_eq!(
            published["sha256"],
            "e136b14c2ab7f31a8951c3ffb74439ca4299e9c835fe77fa1ed533f5dfe86688"
        );
        assert_eq!(published["size_bytes"], 17_865_468_284u64);
        // This client: same tag and local file name, the replacement identity,
        // fetched from the versioned repo path; sidecars and drafts unchanged.
        let registry = bundled().unwrap();
        let (_, entry) = registry.model("raon-1-mini-preview").unwrap();
        assert_eq!(entry.file, "raon-1-mini-preview.mq4-xts");
        assert_eq!(
            entry.sha256.as_deref(),
            Some("fa6d150bcb2af4600aad29aa96bb8fdaf894f0bf1f6bd706af17bc65c8f8690e")
        );
        assert_eq!(entry.size_bytes, Some(17_772_056_444));
        assert_eq!(
            artifact_remote_file(entry, &entry.file),
            "v2/raon-1-mini-preview.mq4-xts"
        );
        let sidecar = entry.dflash.as_ref().unwrap();
        assert_eq!(artifact_remote_file(entry, &sidecar.file), sidecar.file);
        let (_, draft) = registry.model("raon-1-mini-preview:draft").unwrap();
        assert_eq!(artifact_remote_file(draft, &draft.file), draft.file);
        // Re-parsing an already-upgraded registry (e.g. the fetch cache) is a no-op.
        let again = RegistryV1::parse(&serde_json::to_string(&registry).unwrap(), "x").unwrap();
        assert_eq!(again.models["raon-1-mini-preview"], *entry);
    }

    #[test]
    fn artifact_upgrade_pins_match_tag_and_exact_base_identity_only() {
        let pin = ARTIFACT_UPGRADE_PINS[0];
        let entry = |tag_sha: &str, size: u64| ModelEntry {
            repo: "hipfire-models/raon-1-mini-preview".into(),
            file: "raon-1-mini-preview.mq4-xts".into(),
            sha256: Some(tag_sha.into()),
            size_bytes: Some(size),
            ..ModelEntry::default()
        };
        let mut registry = bundled().unwrap();
        registry.models.clear();
        registry
            .models
            .insert(pin.tag.into(), entry(pin.base_sha256, pin.base_size));
        registry
            .models
            .insert("other".into(), entry(pin.base_sha256, pin.base_size));
        registry
            .models
            .insert("raon-wrong-size".into(), entry(pin.base_sha256, 1));
        registry.apply_artifact_upgrades(ARTIFACT_UPGRADE_PINS);
        assert_eq!(registry.models[pin.tag].sha256.as_deref(), Some(pin.sha256));
        assert_eq!(
            registry.models["other"].sha256.as_deref(),
            Some(pin.base_sha256)
        );
        let unchanged = &registry.models["other"];
        assert_eq!(
            artifact_remote_file(unchanged, &unchanged.file),
            unchanged.file
        );
        let mut fresh = bundled().unwrap();
        fresh.models.clear();
        let mut wrong = entry(pin.base_sha256, pin.base_size);
        wrong.size_bytes = Some(pin.base_size + 1);
        fresh.models.insert(pin.tag.into(), wrong);
        fresh.apply_artifact_upgrades(ARTIFACT_UPGRADE_PINS);
        assert_eq!(
            fresh.models[pin.tag].sha256.as_deref(),
            Some(pin.base_sha256)
        );
    }

    #[test]
    fn bundled_raon_dflash2_sidecar_is_declared_and_lowers_auto_on_the_target_only() {
        let registry = bundled().unwrap();
        let (tag, target) = registry.model("raon-1-mini-preview").unwrap();
        let sidecar = target.dflash.as_ref().expect("raon declares its draft");
        assert_eq!(sidecar.file, "raon-1-mini-preview-dflash2.mq4");
        assert_eq!(sidecar.size_bytes, Some(748_250_112));
        assert_eq!(
            sidecar.sha256.as_deref(),
            Some("f46b4257773afdfaf25370df47d558429e950b8825561a8f1c3ad5a0010758ee")
        );
        let (draft_tag, draft) = registry.model("raon-1-mini-preview:draft").unwrap();
        assert_eq!(draft.file, sidecar.file);
        assert_eq!(draft.repo, target.repo);
        assert_eq!(draft.arch_id, Some(20));
        assert_eq!(draft.sha256, sidecar.sha256);
        assert_eq!(draft.size_bytes, sidecar.size_bytes);
        let auto = ConfigValue::String("auto".into());
        let layer = config_layer_for_tag(tag, target).unwrap();
        assert_eq!(layer.get("speculation.dflash"), Some(&auto));
        let mut bare = target.clone();
        bare.dflash = None;
        assert!(config_layer_for_tag(tag, &bare)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
        assert!(config_layer_for_tag(draft_tag, draft)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
    }

    #[test]
    fn bundled_dflash_sidecars_name_pullable_files() {
        // Every `dflash.file` must name a file that some registry entry's
        // `file` also names, so `hipfire pull <target>` fetching the sidecar
        // always lands a file that `hipfire pull <tag>-draft` could fetch too.
        let registry = bundled().unwrap();
        let files: std::collections::BTreeSet<&str> = registry
            .models
            .values()
            .map(|entry| entry.file.as_str())
            .collect();
        let mut paired = 0;
        for (tag, entry) in &registry.models {
            if let Some(sidecar) = entry.dflash.as_ref() {
                assert!(
                    files.contains(sidecar.file.as_str()),
                    "model '{tag}' declares dflash sidecar '{}' with no matching entry file",
                    sidecar.file
                );
                paired += 1;
            }
        }
        assert!(
            paired > 0,
            "bundled registry should pair at least one dflash sidecar"
        );
    }

    #[test]
    fn bundled_vision_sidecar_is_shared_across_qwen38_tiers() {
        // Every non-draft `qwen3.8:27b*` target tier declares the same
        // vision-tower sidecar (`qwen3.8-27b-vision.hfq`) so each text quant
        // tier serves images without requantizing the trunk. `-draft*` tags
        // are text-only DFlash controller sidecars and carry no vision. Unlike
        // dflash, the vision file is standalone: no registry entry's `file`
        // names it.
        let registry = bundled().unwrap();
        let mut declared = 0;
        for (tag, entry) in &registry.models {
            if !tag.starts_with("qwen3.8:27b") || tag.contains("-draft") {
                continue;
            }
            let sidecar = entry
                .vision
                .as_ref()
                .unwrap_or_else(|| panic!("model '{tag}' must declare the shared vision sidecar"));
            assert_eq!(
                sidecar.file, "qwen3.8-27b-vision.hfq",
                "model '{tag}' must share qwen3.8-27b-vision.hfq"
            );
            declared += 1;
        }
        assert!(
            declared > 0,
            "bundled registry should declare a vision sidecar"
        );
    }

    #[test]
    fn aliases_and_filenames_resolve_to_canonical_tags() {
        let registry = bundled().unwrap();
        assert_eq!(registry.resolve_tag("qwen3.6"), "qwen3.6:35b-a3b");
        assert_eq!(
            registry.resolve_tag("qwen3.6-35b-a3b.mq4r"),
            "qwen3.6:35b-a3b-mq4r"
        );
        assert_eq!(registry.resolve_tag("qwen3.8-27b.mq4"), "qwen3.8:27b");
        assert_eq!(
            registry.resolve_tag("qwen3.8-27b.mq4-xt"),
            "qwen3.8:27b-mq4-xt"
        );
        assert_eq!(registry.resolve_tag("qwen3.8:fast"), "qwen3.8:27b-mq4-xt");
        // A path is not a tag: even a path into the models directory resolves
        // to itself here. Callers that know the models directory establish
        // identity with canonical path comparison (`registry_entry_for_path`
        // in hipfire-cli), so `serve --model ~/.hipfire/models/<file>` still
        // sees the entry's sidecars without letting a same-basename lookalike
        // elsewhere inherit them (PR #686 hw-gate regression).
        assert_eq!(
            registry.resolve_tag("/home/u/.hipfire/models/qwen3.8-27b.mq5"),
            "/home/u/.hipfire/models/qwen3.8-27b.mq5"
        );
        assert_eq!(
            registry.resolve_tag("/home/u/.hipfire/models/qwen3.8-27b.mq4"),
            "/home/u/.hipfire/models/qwen3.8-27b.mq4"
        );
        assert!(registry
            .model("/home/u/.hipfire/models/qwen3.8-27b.mq4")
            .is_none());

        assert_eq!(registry.resolve_tag("deepseek4"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("deepseek4:0731"), "deepseek-v4-flash");
        // `:mq2r` names now land on the default tag, since MQ2R *is* the default.
        assert_eq!(
            registry.resolve_tag("deepseek4:0731-mq2r"),
            "deepseek-v4-flash"
        );
        assert_eq!(
            registry.resolve_tag("deepseek-v4-flash-0731.mq2r"),
            "deepseek-v4-flash"
        );
        // The superseded artifact must still be reachable by its own filename.
        assert_eq!(
            registry.resolve_tag("deepseek-v4-flash-0731.mq2lloyd"),
            "deepseek-v4-flash:mq2lloyd"
        );
        assert_eq!(
            registry.resolve_tag("deepseek4:preview"),
            "deepseek-v4-flash-preview"
        );
        assert!(registry.model("deepseek4:preview-mq2r").is_none());

        // `ds4` is the short name used throughout the tree (ds4-adapter-r128.bin,
        // ds4_length_sweep.sh, hipfire-arch-deepseek4), so it resolves too, and
        // mirrors the full `deepseek4:*` surface.
        assert_eq!(registry.resolve_tag("ds4"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("ds4:0731"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("ds4:mq2r"), "deepseek-v4-flash");
        assert_eq!(
            registry.resolve_tag("ds4:preview"),
            "deepseek-v4-flash-preview"
        );
        // Every SKU reachable by a short `:mq2r` name must also be reachable by
        // the matching `:mq2lloyd` one — otherwise demoting MQ2-Lloyd from the
        // default silently strands it behind long-form names only.
        for name in [
            "deepseek4:mq2lloyd",
            "deepseek-v4:mq2lloyd",
            "deepseek4:0731-mq2lloyd",
            "ds4:mq2lloyd",
            "ds4:0731-mq2lloyd",
        ] {
            assert_eq!(
                registry.resolve_tag(name),
                "deepseek-v4-flash:mq2lloyd",
                "{name} must resolve to the MQ2-Lloyd tag"
            );
        }
        let (_, default_sku) = registry.model("deepseek4:0731").unwrap();
        let settings = default_sku.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.reasoning_effort.as_deref(), Some("low"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
    }

    #[test]
    fn path_inputs_never_resolve_to_registry_tags() {
        // PR #686 hw-gate regression: `hipfire rm /elsewhere/qwen3.6-27b.mq4`
        // basename-matched the `qwen3.6:27b` entry and deleted the installed
        // sidecars. A path — anywhere, including the models directory itself —
        // is not a tag and resolves to itself.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-10T00:00:00Z",
            "models":{"qwen3.6:27b":{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        assert_eq!(
            registry.resolve_tag("/elsewhere/qwen3.6-27b.mq4"),
            "/elsewhere/qwen3.6-27b.mq4"
        );
        assert!(registry.model("/elsewhere/qwen3.6-27b.mq4").is_none());
        // The bare file name still resolves: it carries no directory to confuse
        // with the installed artifact.
        assert_eq!(registry.resolve_tag("qwen3.6-27b.mq4"), "qwen3.6:27b");
        assert!(registry.model("qwen3.6-27b.mq4").is_some());
    }

    #[test]
    fn entry_for_file_matches_exact_bare_names_only() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-10T00:00:00Z",
            "models":{"qwen3.6:27b":{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{"qwen36":"qwen3.6:27b"}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, _) = registry.entry_for_file("qwen3.6-27b.mq4").unwrap();
        assert_eq!(tag, "qwen3.6:27b");
        assert!(registry
            .entry_for_file("/elsewhere/qwen3.6-27b.mq4")
            .is_none());
        assert!(registry.entry_for_file("qwen3.6:27b").is_none());
        assert!(registry.entry_for_file("qwen36").is_none());
        assert!(registry.entry_for_file("other.mq4").is_none());
    }

    #[test]
    fn recommended_settings_lower_the_full_sampling_contract_to_config() {
        let settings = RecommendedSettings {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(40),
            min_p: Some(0.05),
            presence_penalty: Some(1.5),
            repeat_penalty: Some(1.05),
            system_prompt: Some("You are MiniMax.".into()),
            reasoning_effort: Some("high".into()),
            thinking_budget: Some("xhigh".into()),
        };
        let layer = settings.config_layer().unwrap();
        assert_eq!(
            layer.get("generation.temperature"),
            Some(&ConfigValue::Float(1.0))
        );
        assert_eq!(
            layer.get("generation.top_p"),
            Some(&ConfigValue::Float(0.95))
        );
        assert_eq!(
            layer.get("generation.top_k"),
            Some(&ConfigValue::Integer(40))
        );
        assert_eq!(
            layer.get("generation.min_p"),
            Some(&ConfigValue::Float(0.05))
        );
        assert_eq!(
            layer.get("generation.presence_penalty"),
            Some(&ConfigValue::Float(1.5))
        );
        assert_eq!(
            layer.get("generation.repeat_penalty"),
            Some(&ConfigValue::Float(1.05))
        );
        assert_eq!(
            layer.get("prompt.system"),
            Some(&ConfigValue::String("You are MiniMax.".into()))
        );
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("high".into()))
        );
        assert_eq!(
            layer.get("reasoning.budget"),
            Some(&ConfigValue::String("xhigh".into()))
        );
    }

    #[test]
    fn malformed_entry_rejects_the_whole_registry() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"magic4"}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn tag_policy_pins_qwen_deepseek_and_glimmer_targets() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:4b":{"repo":"x","file":"qwen3.5-4b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.6:35b-a3b":{"repo":"x","file":"qwen3.6-35b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.5:27b-draft":{"repo":"x","file":"qwen35-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.5:27b-dflash":{"repo":"x","file":"qwen35-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash":{"repo":"x","file":"deepseek-v4-flash-0731.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"deepseek-v4-flash-0731.mq2lloyd","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash-preview":{"repo":"x","file":"deepseek-v4-flash-preview.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:draft":{"repo":"x","file":"deepseek-v4-flash-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:fast":{"repo":"x","file":"muse-glimmer-30b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:draft":{"repo":"x","file":"muse-glimmer-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "other:model":{"repo":"x","file":"other.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{
                "deepseek4":"deepseek-v4-flash",
                "ds4":"deepseek-v4-flash",
                "deepseek4:preview":"deepseek-v4-flash-preview",
                "muse-glimmer:quality":"muse-glimmer"
            }
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Exact Qwen families get max_tokens 81920 only; never max_seq or a KV backend.
        for tag in ["qwen3.5:4b", "qwen3.6:35b-a3b", "qwen3.8:27b"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{tag} must not write memory.max_seq"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(81920)),
                "{tag} should get 81920"
            );
        }

        // Original Qwen3 (no dot) receives none
        let (_, entry) = registry.model("qwen3:8b").unwrap();
        let layer = config_layer_for_tag("qwen3:8b", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Draft/dflash sidecars do not get the Qwen policy
        for tag in ["qwen3.5:27b-draft", "qwen3.5:27b-dflash"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} sidecar must not write a KV backend"
            );
            assert!(layer.get("memory.max_seq").is_none());
            assert!(layer.get("generation.max_tokens").is_none());
        }

        // DeepSeek official / MQ2Lloyd / preview targets get max_tokens 393216 only.
        for tag in [
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(resolved, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{tag} must not write memory.max_seq"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(393216)),
                "{tag} should get 393216"
            );
        }
        // Aliases resolve to canonical families that carry the same policy.
        for alias in ["deepseek4", "ds4", "deepseek4:preview"] {
            let (resolved, entry) = registry.model(alias).unwrap();
            let layer = config_layer_for_tag(resolved, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{alias}->{resolved} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{alias}->{resolved} must not write memory.max_seq"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(393216)),
                "{alias}->{resolved} should get 393216"
            );
        }
        // DeepSeek draft sidecar receives none
        let (_, entry) = registry.model("deepseek-v4-flash:draft").unwrap();
        let layer = config_layer_for_tag("deepseek-v4-flash:draft", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Muse Glimmer quality and fast targets write nothing (no max_seq/max_tokens/backend).
        for tag in ["muse-glimmer", "muse-glimmer:fast"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{tag} must not write memory.max_seq"
            );
            assert!(
                layer.get("generation.max_tokens").is_none(),
                "{tag} must not get max_tokens"
            );
        }
        // quality alias lands on trunk and inherits the same (empty) policy.
        let (resolved, entry) = registry.model("muse-glimmer:quality").unwrap();
        assert_eq!(resolved, "muse-glimmer");
        let layer = config_layer_for_tag(resolved, entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Muse Glimmer draft receives none
        let (_, entry) = registry.model("muse-glimmer:draft").unwrap();
        let layer = config_layer_for_tag("muse-glimmer:draft", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Absent policy: unrelated models get no automatic policy
        let (_, entry) = registry.model("other:model").unwrap();
        let layer = config_layer_for_tag("other:model", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());
    }

    #[test]
    fn tag_policy_does_not_expand_wire_schema() {
        // Old v1 JSON without the invented fields must still parse (deny_unknown_fields).
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"ok":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("ok").unwrap();
        let layer = entry.config_layer().unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());
        // Tag policy also absent for non-target.
        let tagged = config_layer_for_tag("ok", entry).unwrap();
        assert!(tagged.get("memory.kv_backend").is_none());

        // Invented wire fields must be rejected (no schema expansion).
        let bad = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_backend":"vmm"}},
            "aliases":{}
        }"#;
        assert!(
            RegistryV1::parse(bad, "test").is_err(),
            "default_kv_backend must be rejected as unknown field"
        );
        let bad2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_seq":262144}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad2, "test").is_err());
        let bad3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_tokens":81920}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad3, "test").is_err());
    }

    #[test]
    fn qwen38_27b_dflash_auto_is_scoped_to_paired_trunk_tiers() {
        // `speculation.dflash = "auto"` is lowered only for canonical Qwen3.8-27B
        // trunk tags (`qwen3.8:27b`, `qwen3.8:27b-*`) that declare a `dflash`
        // sidecar, are not draft/dflash sidecars, and are not arch_id=16.
        let dflash = r#""dflash":{"file":"qwen38-27b-dflash-mq4.hfq"}"#;
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{{
                "qwen3.8:27b":{{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq3":{{"repo":"x","file":"qwen3.8-27b.mq3","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq4-pro":{{"repo":"x","file":"qwen3.8-27b.mq4-pro","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq4-xt":{{"repo":"x","file":"qwen3.8-27b.mq4-xt","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq5":{{"repo":"x","file":"qwen3.8-27b.mq5","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5}},
                "qwen3.8:27boops":{{"repo":"x","file":"qwen3.8-27boops.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:flash-next":{{"repo":"x","file":"qwen3.8-flash-next.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":16,{dflash}}},
                "qwen3.8:27b-draft-mq4":{{"repo":"x","file":"qwen38-27b-draft.hfq","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":20,{dflash}}},
                "qwen3.8:27b-dflash-mq4":{{"repo":"x","file":"qwen38-27b-dflash.hfq","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":20,{dflash}}},
                "qwen3.6:27b":{{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3.5:27b":{{"repo":"x","file":"qwen3.5-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3:8b":{{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3.8:9b":{{"repo":"x","file":"qwen3.8-9b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}}
            }},
            "aliases":{{
                "qwen3.8":"qwen3.8:27b",
                "qwen3.8:latest":"qwen3.8:27b",
                "qwen3.8:27b-mq4":"qwen3.8:27b",
                "qwen3.8:fast":"qwen3.8:27b-mq4-xt",
                "qwen3.8:27b-fast":"qwen3.8:27b-mq4-xt"
            }}
        }}"#
        );
        let registry = RegistryV1::parse(&raw, "test").unwrap();
        let auto_value = ConfigValue::String("auto".into());
        let auto = Some(&auto_value);

        // Canonical tags and every alias that resolves to an eligible tag.
        for input in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq3",
            "qwen3.8:27b-mq4-pro",
            "qwen3.8:27b-mq4-xt",
            "qwen3.8",
            "qwen3.8:latest",
            "qwen3.8:27b-mq4",
            "qwen3.8:fast",
            "qwen3.8:27b-fast",
        ] {
            let (tag, entry) = registry.model(input).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert_eq!(layer.get("speculation.dflash"), auto, "{input}->{tag}");
            // Only the dflash mode is added; MTP and the ngram/mode knobs are untouched.
            assert!(layer.get("speculation.mtp").is_none(), "{input}->{tag}");
            assert!(layer.get("speculation.mode").is_none(), "{input}->{tag}");
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(81920)),
                "{input}->{tag} keeps the existing max_tokens policy"
            );
        }

        // Same family/tier without a registry dflash declaration, a longer
        // lookalike tier, arch_id=16, draft/dflash sidecar tags, and other
        // families/sizes never get the policy.
        for tag in [
            "qwen3.8:27b-mq5",
            "qwen3.8:27boops",
            "qwen3.8:flash-next",
            "qwen3.8:27b-draft-mq4",
            "qwen3.8:27b-dflash-mq4",
            "qwen3.6:27b",
            "qwen3.5:27b",
            "qwen3:8b",
            "qwen3.8:9b",
        ] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("speculation.dflash").is_none(),
                "{tag} must not default DFlash"
            );
        }

        // arch_id and the dflash declaration are both required, not just the tag.
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let mut no_dflash = entry.clone();
        no_dflash.dflash = None;
        assert!(config_layer_for_tag(tag, &no_dflash)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
        let mut qwen4 = entry.clone();
        qwen4.arch_id = Some(16);
        assert!(config_layer_for_tag(tag, &qwen4)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
        // A different tag with the same dflash-less entry stays off too.
        assert!(config_layer_for_tag("qwen3.8:27b-mq4-pro", &no_dflash)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
    }

    #[test]
    fn bundled_dflash_auto_matches_only_declared_qwen38_27b_tiers() {
        // Source-grounded against checked-in registry/v1.json: DFlash-paired
        // Qwen3.8-27B trunk tiers lower `auto`; mq5/mq6 tiers (no dflash
        // declaration), Flash-Next, drafts, and other families do not; the
        // explicit `raon-1-mini-preview` allow-list entry lowers it too.
        let registry = bundled().unwrap();
        let mut auto_tiers = 0;
        for (tag, entry) in &registry.models {
            let layer = config_layer_for_tag(tag, entry).unwrap();
            let expected = ((tag == "qwen3.8:27b" || tag.starts_with("qwen3.8:27b-"))
                && !tag.contains("draft")
                && !tag.contains("dflash")
                && entry.arch_id != Some(16)
                && entry.dflash.is_some())
                || (tag == "raon-1-mini-preview" && entry.dflash.is_some());
            assert_eq!(
                layer.get("speculation.dflash") == Some(&ConfigValue::String("auto".into())),
                expected,
                "{tag}"
            );
            if expected {
                auto_tiers += 1;
            } else {
                assert!(layer.get("speculation.dflash").is_none(), "{tag}");
            }
        }
        assert!(
            auto_tiers >= 1,
            "at least one bundled Qwen3.8-27B tier declares a DFlash draft"
        );
        for tag in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq3",
            "qwen3.8:27b-mq3-xt",
            "qwen3.8:27b-mq3-pro",
            "qwen3.8:27b-mq4-xt",
            "qwen3.8:27b-mq4-xts",
            "qwen3.8:27b-mq4-pro",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            assert!(entry.dflash.is_some(), "{tag} declares a DFlash draft");
            assert_eq!(
                config_layer_for_tag(resolved, entry)
                    .unwrap()
                    .get("speculation.dflash"),
                Some(&ConfigValue::String("auto".into())),
                "{tag}"
            );
        }
        for tag in [
            "qwen3.8:27b-mq5",
            "qwen3.8:27b-mq6-pro",
            "qwen3.8:flash-next",
            "qwen3.8:27b-draft-mq4",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            assert!(
                config_layer_for_tag(resolved, entry)
                    .unwrap()
                    .get("speculation.dflash")
                    .is_none(),
                "{tag}"
            );
        }
        for alias in [
            "qwen3.8",
            "qwen3.8:latest",
            "qwen3.8:fast",
            "qwen3.8:27b-fast",
        ] {
            let (resolved, entry) = registry.model(alias).unwrap();
            assert_eq!(
                config_layer_for_tag(resolved, entry)
                    .unwrap()
                    .get("speculation.dflash"),
                Some(&ConfigValue::String("auto".into())),
                "{alias}->{resolved}"
            );
        }
    }

    #[test]
    fn qwen38_27b_dflash_auto_yields_to_explicit_user_off() {
        // Registry layer is lowest; global/model/one-shot dflash=off wins.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,"dflash":{"file":"qwen38-27b-dflash-mq4.hfq"}}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let registry_named = || hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::RegistryModel {
                tag: tag.to_owned(),
                revision: "v1".into(),
            },
            layer: config_layer_for_tag(tag, entry).unwrap(),
        };
        let dflash_of = |layers: Vec<hipfire_config::NamedLayer>| {
            hipfire_config::resolve(layers)
                .unwrap()
                .get("speculation.dflash")
                .map(|item| item.value.clone())
        };
        // Registry alone: auto beats the BuiltIn off default.
        assert_eq!(
            dflash_of(vec![registry_named()]),
            Some(ConfigValue::String("auto".into()))
        );
        // Explicit global-user off wins.
        let mut global_off = ConfigLayer::default();
        global_off
            .set("speculation.dflash", ConfigValue::String("off".into()))
            .unwrap();
        assert_eq!(
            dflash_of(vec![
                registry_named(),
                hipfire_config::NamedLayer {
                    source: hipfire_config::ConfigSource::GlobalUser {
                        path: std::path::PathBuf::from("/tmp/global.toml"),
                    },
                    layer: global_off,
                },
            ]),
            Some(ConfigValue::String("off".into()))
        );
        // Explicit per-model-user off wins too.
        let mut model_off = ConfigLayer::default();
        model_off
            .set("speculation.dflash", ConfigValue::String("off".into()))
            .unwrap();
        assert_eq!(
            dflash_of(vec![
                registry_named(),
                hipfire_config::NamedLayer {
                    source: hipfire_config::ConfigSource::ModelUser {
                        model: tag.to_owned(),
                        path: std::path::PathBuf::from("/tmp/model.toml"),
                    },
                    layer: model_off,
                },
            ]),
            Some(ConfigValue::String("off".into()))
        );
    }

    #[test]
    fn tag_policy_explicit_override_wins_over_registry() {
        // Registry tag policy is a low-precedence layer; global/model/one-shot user config wins.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let registry_layer = config_layer_for_tag(tag, entry).unwrap();
        assert!(registry_layer.get("memory.kv_backend").is_none());
        assert!(
            registry_layer.get("memory.max_seq").is_none(),
            "tag layer must not write memory.max_seq"
        );
        assert_eq!(
            registry_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );

        // Simulate user global override to legacy + different max_seq/max_tokens.
        let mut user_layer = ConfigLayer::default();
        user_layer
            .set("memory.kv_backend", ConfigValue::String("legacy".into()))
            .unwrap();
        user_layer
            .set("memory.max_seq", ConfigValue::Integer(32768))
            .unwrap();
        user_layer
            .set("generation.max_tokens", ConfigValue::Integer(1024))
            .unwrap();

        let resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test.toml"),
                },
                layer: user_layer,
            },
        ])
        .unwrap();
        // User wins.
        assert_eq!(
            resolved.get("memory.kv_backend").map(|value| &value.value),
            Some(&ConfigValue::String("legacy".into()))
        );
        assert_eq!(
            resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(32768))
        );
        assert_eq!(
            resolved
                .get("generation.max_tokens")
                .map(|value| &value.value),
            Some(&ConfigValue::Integer(1024))
        );

        // Glimmer target override likewise wins (backend + max_seq); tag writes neither.
        let raw2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry2 = RegistryV1::parse(raw2, "test").unwrap();
        let (g_tag, g_entry) = registry2.model("muse-glimmer").unwrap();
        let g_registry_layer = config_layer_for_tag(g_tag, g_entry).unwrap();
        assert!(g_registry_layer.get("memory.kv_backend").is_none());
        assert!(g_registry_layer.get("memory.max_seq").is_none());
        assert!(g_registry_layer.get("generation.max_tokens").is_none());
        let mut g_user = ConfigLayer::default();
        g_user
            .set("memory.kv_backend", ConfigValue::String("legacy".into()))
            .unwrap();
        g_user
            .set("memory.max_seq", ConfigValue::Integer(8192))
            .unwrap();
        let g_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: g_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: g_registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test2.toml"),
                },
                layer: g_user,
            },
        ])
        .unwrap();
        assert_eq!(
            g_resolved
                .get("memory.kv_backend")
                .map(|value| &value.value),
            Some(&ConfigValue::String("legacy".into()))
        );
        assert_eq!(
            g_resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(8192))
        );

        // DeepSeek target override wins over max_tokens policy; tag writes no max_seq.
        let raw3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"deepseek-v4-flash":{"repo":"x","file":"ds4.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry3 = RegistryV1::parse(raw3, "test").unwrap();
        let (d_tag, d_entry) = registry3.model("deepseek-v4-flash").unwrap();
        let d_registry_layer = config_layer_for_tag(d_tag, d_entry).unwrap();
        assert!(d_registry_layer.get("memory.max_seq").is_none());
        assert_eq!(
            d_registry_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(393216))
        );
        let mut d_user = ConfigLayer::default();
        d_user
            .set("memory.max_seq", ConfigValue::Integer(65536))
            .unwrap();
        d_user
            .set("generation.max_tokens", ConfigValue::Integer(2048))
            .unwrap();
        let d_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: d_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: d_registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test3.toml"),
                },
                layer: d_user,
            },
        ])
        .unwrap();
        assert_eq!(
            d_resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(65536))
        );
        assert_eq!(
            d_resolved
                .get("generation.max_tokens")
                .map(|value| &value.value),
            Some(&ConfigValue::Integer(2048))
        );
    }

    #[test]
    fn dangling_aliases_are_dropped() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"ok":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{"good":"ok","bad":"missing"}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        assert_eq!(registry.aliases.len(), 1);
        assert_eq!(registry.resolve_tag("good"), "ok");
    }

    #[test]
    fn future_cache_timestamps_are_stale() {
        let cache = RegistryCache {
            fetched_at: 101,
            url: "x".into(),
            registry: bundled().unwrap(),
        };
        assert!(!cache_is_fresh(&cache, 100, REGISTRY_CACHE_TTL));
    }

    #[test]
    fn sampling_profiles_resolve_per_mode_with_general_fallback() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "recommended_settings":{"temperature":1.0,"presence_penalty":1.5},
                "sampling_profiles":{
                    "coding":{"temperature":0.6,"presence_penalty":0.0},
                    "instruct":{"temperature":0.7,"top_p":0.8}
                }
            }},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("m").unwrap();
        assert_eq!(
            entry.sampling_profile("coding").unwrap().temperature,
            Some(0.6)
        );
        assert_eq!(entry.sampling_profile("instruct").unwrap().top_p, Some(0.8));
        // general has no explicit profile → falls back to recommended_settings.
        assert_eq!(
            entry.sampling_profile("general").unwrap().presence_penalty,
            Some(1.5)
        );
        assert!(entry.sampling_profile("nope").is_none());
    }

    #[test]
    fn out_of_range_sampling_profile_rejects_the_whole_registry() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "sampling_profiles":{"coding":{"temperature":9.0}}
            }},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn bundled_registry_passes_reasoning_validation() {
        // Bundled v1.json must validate under the mirrored generator invariants:
        // effort-native families have no thinking_budget (absence = uncapped)
        // and all present enums are recognized.
        let registry = bundled().unwrap();
        // Spot-check that effort-native bundled entries indeed omit budget
        for tag in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq4-xt",
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
            "muse-glimmer",
            "muse-glimmer:fast",
            "ornith-1.5:35b-a3b",
            "ornith-1.5:35b-a3b-mq4r",
        ] {
            let (_, entry) = registry
                .model(tag)
                .unwrap_or_else(|| panic!("{tag} must exist"));
            if let Some(rs) = &entry.recommended_settings {
                assert!(
                    rs.thinking_budget.is_none(),
                    "{tag} bundled thinking_budget must be absent"
                );
            }
            if let Some(profiles) = &entry.sampling_profiles {
                for (_, settings) in [
                    ("general", &profiles.general),
                    ("coding", &profiles.coding),
                    ("instruct", &profiles.instruct),
                ] {
                    if let Some(settings) = settings {
                        assert!(
                            settings.thinking_budget.is_none(),
                            "{tag} profile thinking_budget must be absent"
                        );
                    }
                }
            }
        }

        // Both Ornith 1.5 artifacts share the Qwen3.8 effort-native contract:
        // default/general medium (hipfire default; xhigh stays selectable),
        // explicit coding profile xhigh, instruct none, and no named
        // thinking_budget (absence = uncapped).
        for tag in ["ornith-1.5:35b-a3b", "ornith-1.5:35b-a3b-mq4r"] {
            let (_, ornith) = registry
                .model(tag)
                .unwrap_or_else(|| panic!("{tag} must exist"));
            let ornith_rs = ornith.recommended_settings.as_ref().unwrap();
            assert_eq!(ornith_rs.reasoning_effort.as_deref(), Some("medium"));
            assert!(
                ornith_rs.thinking_budget.is_none(),
                "{tag} effort-native: absence means uncapped"
            );
            let ornith_general = ornith.sampling_profile("general").unwrap();
            assert_eq!(ornith_general.reasoning_effort.as_deref(), Some("medium"));
            assert!(ornith_general.thinking_budget.is_none());
            let ornith_coding = ornith.sampling_profile("coding").unwrap();
            assert_eq!(ornith_coding.reasoning_effort.as_deref(), Some("xhigh"));
            assert!(ornith_coding.thinking_budget.is_none());
            let ornith_instruct = ornith.sampling_profile("instruct").unwrap();
            assert_eq!(ornith_instruct.reasoning_effort.as_deref(), Some("none"));
            assert!(ornith_instruct.thinking_budget.is_none());
        }
    }

    #[test]
    fn effort_native_thinking_budget_rejected_for_fetched_or_cached_registry() {
        // Mirrors scripts/registry_gen.py:_effort_native_tag. A fetched/cache
        // registry that reintroduces a named thinking_budget on effort-native
        // families must fail validation wholesale so load() falls back to
        // bundled (network) or discards the cache entry (fresh/stale).
        let cases = [
            // Qwen3.8 product SKUs
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"high"}}},"aliases":{}}"#,
            // DeepSeek V4 Flash (also covers :mq2lloyd via family)
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"low","thinking_budget":"uncapped"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash-preview":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"high","thinking_budget":"med"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"low"}}},"aliases":{}}"#,
            // Muse Glimmer product SKUs
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"xhigh"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer:fast":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"max"}}},"aliases":{}}"#,
            // Ornith 1.5 product + legacy family spellings
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith-1.5:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"high"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith1.5:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"uncapped"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"med"}}},"aliases":{}}"#,
            // Effort-native sampling_profiles also rejected
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"coding":{"reasoning_effort":"xhigh","thinking_budget":"high"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"low"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"uncapped"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith-1.5:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"high"}}}},"aliases":{}}"#,
        ];
        for raw in cases {
            let err = RegistryV1::parse(raw, "network/cache")
                .expect_err("must reject effort-native thinking_budget");
            let msg = err.to_string();
            assert!(
                msg.contains("thinking_budget") && msg.contains("effort-native"),
                "unexpected error for effort-native budget rejection: {msg}"
            );
        }
        // A cache entry that violates the invariant is also rejected via
        // validate(), causing read_cache to return None and load() to fall back.
        let stale_raw = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}}},"aliases":{}}"#;
        let stale: RegistryV1 = serde_json::from_str(stale_raw).unwrap();
        assert!(
            stale.validate("registry cache").is_err(),
            "stale cache with effort-native thinking_budget must fail validate()"
        );
    }

    #[test]
    fn invalid_reasoning_enums_rejected() {
        // Mirrors hipfire-config/registry_gen enum allowlists: recognizable
        // invalid values are rejected with a clear error; malformed types
        // already fail via surrounding validation and are not re-tested here.
        let invalid_effort = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"m":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"turbo"}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_effort, "test")
            .expect_err("invalid reasoning_effort must be rejected");
        assert!(err.to_string().contains("reasoning_effort"));

        let invalid_effort_profile = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"m":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"coding":{"reasoning_effort":"ultra"}}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_effort_profile, "test")
            .expect_err("invalid profile effort must be rejected");
        assert!(err.to_string().contains("reasoning_effort"));

        // thinking_budget invalid on legacy (non-effort-native) model
        let invalid_budget = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.5:9b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"yolo"}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_budget, "test")
            .expect_err("invalid thinking_budget must be rejected");
        assert!(err.to_string().contains("thinking_budget"));

        let invalid_budget_profile = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.6:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"superhigh"}}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_budget_profile, "test")
            .expect_err("invalid profile budget must be rejected");
        assert!(err.to_string().contains("thinking_budget"));
    }

    #[test]
    fn legacy_qwen35_qwen36_thinking_budget_accepted() {
        // Legacy Qwen3.5/3.6 families are not effort-native; named budgets
        // remain valid and pass validation for both top-level and profiles.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:9b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"high","thinking_budget":"high"}},
                "qwen3.5:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"med"},"coding":{"reasoning_effort":"max","thinking_budget":"max"}}},
                "qwen3.6:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"uncapped"}},
                "qwen3.6:35b-a3b-mq4r":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"instruct":{"thinking_budget":"off","reasoning_effort":"none"}}}
            },
            "aliases":{}
        }"#;
        RegistryV1::parse(raw, "test").expect("legacy thinking_budget must remain valid");

        // Draft/dflash sidecars are excluded from effort-native gating, so they
        // also accept thinking_budget even when family would otherwise be native.
        let sidecars = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.8:27b-draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}},
                "qwen3.8:27b-dflash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"low"}},
                "deepseek-v4-flash:draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"xhigh"}},
                "muse-glimmer:draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"med"}},
                "ornith-1.5:35b-a3b-draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}}
            },
            "aliases":{}
        }"#;
        RegistryV1::parse(sidecars, "test")
            .expect("draft/dflash sidecars must allow thinking_budget");
    }

    fn resolve_fixture_registry() -> RegistryV1 {
        RegistryV1::parse(
            r#"{
                "schema_version":1,
                "generated_at":"2026-09-01T00:00:00Z",
                "models":{"qwen3.5:4b":{"repo":"x","file":"qwen3.5-4b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
                "aliases":{}
            }"#,
            "test",
        )
        .unwrap()
    }

    fn max_tokens(resolved: &ResolvedForModel) -> (ConfigValue, ConfigSource) {
        let item = resolved.resolved.get("generation.max_tokens").unwrap();
        (item.value.clone(), item.source.clone())
    }

    #[test]
    fn resolve_for_model_layers_registry_global_model_env_in_order() {
        let dir = RegistryTestDir::new();
        let paths = ConfigPaths::under(&dir.0);
        let registry = resolve_fixture_registry();
        let (tag, entry) = registry.model("qwen3.5:4b").unwrap();
        let card = Some((tag, entry));

        let resolved = resolve_for_model(&paths, tag, card, ConfigLayer::default()).unwrap();
        let (value, source) = max_tokens(&resolved);
        assert_eq!(value, ConfigValue::Integer(81920));
        assert!(matches!(source, ConfigSource::RegistryModel { .. }));
        assert_eq!(resolved.model_id, None);
        assert!(resolved.warnings.is_empty());

        fs::write(&paths.config_toml, "[generation]\nmax_tokens = 1234\n").unwrap();
        let resolved = resolve_for_model(&paths, tag, card, ConfigLayer::default()).unwrap();
        let (value, source) = max_tokens(&resolved);
        assert_eq!(value, ConfigValue::Integer(1234));
        assert!(matches!(source, ConfigSource::GlobalUser { .. }));

        hipfire_config::set_model_override(
            &paths,
            Some(tag),
            None,
            tag,
            "generation.max_tokens",
            "777",
        )
        .unwrap();
        let resolved = resolve_for_model(&paths, tag, card, ConfigLayer::default()).unwrap();
        let (value, source) = max_tokens(&resolved);
        assert_eq!(value, ConfigValue::Integer(777));
        assert!(matches!(source, ConfigSource::ModelUser { .. }));
        assert_eq!(resolved.model_id.as_deref(), Some(tag));

        let mut env_layer = ConfigLayer::default();
        env_layer
            .set("generation.max_tokens", ConfigValue::Integer(555))
            .unwrap();
        let resolved = resolve_for_model(&paths, tag, card, env_layer).unwrap();
        let (value, source) = max_tokens(&resolved);
        assert_eq!(value, ConfigValue::Integer(555));
        assert!(matches!(source, ConfigSource::LegacyEnv { .. }));

        // Without the card the registry layer is absent and the model entry is
        // still found by name.
        let resolved = resolve_for_model(&paths, tag, None, ConfigLayer::default()).unwrap();
        assert_eq!(max_tokens(&resolved).0, ConfigValue::Integer(777));
    }

    #[test]
    fn resolve_for_model_finds_the_model_entry_by_tag_and_by_entry_file() {
        let registry = resolve_fixture_registry();
        let (tag, entry) = registry.model("qwen3.5:4b").unwrap();

        // By registry tag: the caller named the model some other way.
        let dir = RegistryTestDir::new();
        let paths = ConfigPaths::under(&dir.0);
        hipfire_config::set_model_override(&paths, None, None, tag, "generation.max_tokens", "777")
            .unwrap();
        let resolved = resolve_for_model(
            &paths,
            "my-local-name",
            Some((tag, entry)),
            ConfigLayer::default(),
        )
        .unwrap();
        assert_eq!(resolved.model_id.as_deref(), Some(tag));
        assert_eq!(max_tokens(&resolved).0, ConfigValue::Integer(777));
        let unmatched =
            resolve_for_model(&paths, "my-local-name", None, ConfigLayer::default()).unwrap();
        assert_eq!(unmatched.model_id, None);

        // By the entry's file name.
        let dir = RegistryTestDir::new();
        let paths = ConfigPaths::under(&dir.0);
        hipfire_config::set_model_override(
            &paths,
            None,
            None,
            &entry.file,
            "generation.max_tokens",
            "888",
        )
        .unwrap();
        let resolved = resolve_for_model(
            &paths,
            "my-local-name",
            Some((tag, entry)),
            ConfigLayer::default(),
        )
        .unwrap();
        assert_eq!(resolved.model_id.as_deref(), Some(entry.file.as_str()));
        assert_eq!(max_tokens(&resolved).0, ConfigValue::Integer(888));
    }

    fn flash_next_plan_registry() -> RegistryV1 {
        let sidecar =
            |key: &str| serde_json::json!({ "file": format!("qwen3.8-flash-next.ple-{key}") });
        let value = serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": {
                    "repo": "x/y", "file": "qwen3.8-flash-next-gptq3.mq4",
                    "size_gb": 1, "min_vram_gb": 1, "desc": "mono", "arch_id": 16
                },
                "qwen3.8:flash-next-mq3-xts": {
                    "repo": "x/y", "file": "qwen3.8-flash-next-base.mq3-xts",
                    "size_gb": 1, "min_vram_gb": 1, "desc": "split", "arch_id": 16,
                    "ple": { "q4": sidecar("q4"), "q8": sidecar("q8") }
                }
            },
            "aliases": {}
        });
        RegistryV1::parse(&value.to_string(), "tuning test").unwrap()
    }

    #[test]
    fn tuning_plan_env_reads_installed_flash_next_and_its_q8_files() {
        use hipfire_config::tuning::PlanEnv;
        let dir = env::temp_dir().join(format!(
            "hipfire-registry-plan-env-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let registry = flash_next_plan_registry();
        assert_eq!(tuning_plan_env(&dir, &registry), PlanEnv::default());

        // A monolithic embeds the q8 rows.
        fs::write(dir.join("qwen3.8-flash-next-gptq3.mq4"), b"x").unwrap();
        let mono = tuning_plan_env(&dir, &registry);
        assert!(mono.flash_next_installed && mono.ple_q8_present, "{mono:?}");

        // A split base with only the q4 rows has no q8.
        fs::write(dir.join("qwen3.8-flash-next-base.mq3-xts"), b"x").unwrap();
        fs::write(dir.join("qwen3.8-flash-next.ple-q4"), b"x").unwrap();
        let split = tuning_plan_env(&dir, &registry);
        assert!(
            split.flash_next_installed && !split.ple_q8_present,
            "{split:?}"
        );
        assert_eq!(
            split.ple_q8_missing,
            vec!["qwen3.8-flash-next-base.mq3-xts"]
        );
        assert_eq!(
            split.pull_tag.as_deref(),
            Some("qwen3.8:flash-next-mq3-xts")
        );

        // The shared q8 file satisfies it.
        fs::write(dir.join("qwen3.8-flash-next.ple-q8"), b"x").unwrap();
        let shared = tuning_plan_env(&dir, &registry);
        assert!(shared.ple_q8_present, "{shared:?}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn split_model_files_lists_base_then_grandfathered_monolithics() {
        let registry = flash_next_plan_registry();
        // An entry without a pin serves only its own file.
        assert_eq!(
            split_model_files(&registry.models["qwen3.8:flash-next-mq3-xts"]),
            ["qwen3.8-flash-next-base.mq3-xts"]
        );
        let pinned = stock_split_base_entry();
        assert_eq!(
            split_model_files(&pinned),
            [
                FLASH_NEXT_XTS_SPLIT_BASE_FILE,
                FLASH_NEXT_XTS_CANONICAL_FILE,
                FLASH_NEXT_XTS_LEGACY_FILE
            ]
        );
    }
}
