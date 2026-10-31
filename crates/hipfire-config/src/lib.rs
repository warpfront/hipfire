// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed configuration schema and deterministic layer resolution for hipfire.
//!
//! This crate is deliberately independent of GPU and model crates. It owns the
//! public configuration vocabulary, legacy-key compatibility, TOML/JSON
//! persistence, validation, and field-level provenance. Runtime crates consume
//! resolved values; they do not parse user configuration in hot paths.

pub mod calibration;
pub mod devices;
pub mod effective;
pub mod emu;
pub mod emu_budget;
pub mod emu_model;
pub mod env_audit;
pub mod evidence;
pub mod host_memory;
pub mod http_auth;
pub mod intent;
pub mod load_summary;
pub mod rocm;
pub mod secret;
pub mod tuning;

pub use host_memory::{host_memory, HostMemory};

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    env, fmt, fs,
    path::{Path, PathBuf},
    sync::{LazyLock, OnceLock},
};
use thiserror::Error;

pub const CONFIG_SCHEMA_VERSION: i64 = 1;

/// Largest `serve.continuous_batch_size` the schema, CLI and serve precheck
/// accept. It is a configuration ceiling only: runtime memory admission may
/// still allow fewer live lanes, and per-chunk verify row limits bound rows
/// per chunk, not the total lane count.
pub const MAX_CONTINUOUS_BATCH_SIZE: u64 = 256;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("unknown configuration key '{0}'")]
    UnknownKey(String),
    #[error(
        "configuration key '{key}' was removed in {removed_in} ({reason}); it has no effect — \
         delete it from config.toml if it is still set"
    )]
    RetiredKey {
        key: &'static str,
        removed_in: &'static str,
        reason: &'static str,
    },
    #[error("invalid value for {key}: {message}")]
    InvalidValue { key: String, message: String },
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{0}")]
    Home(String),
    /// Catalog state that cannot be edited as requested.
    #[error("{0}")]
    Catalog(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Null,
}

/// Stable device identity used by model-specific placement policies. Logical
/// HIP ordinals are deliberately absent: they are re-numbered by visibility
/// masks and across reboot/hotplug.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum DeviceSelector {
    PciBdf(String),
    Uuid(String),
    /// Permitted only when exactly one visible device matches at resolution.
    ExactArch(String),
}

impl fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PciBdf(value) => write!(f, "pci:{value}"),
            Self::Uuid(value) => write!(f, "uuid:{value}"),
            Self::ExactArch(value) => write!(f, "arch:{value}"),
        }
    }
}

impl std::str::FromStr for DeviceSelector {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        let (kind, value) = raw.split_once(':').ok_or_else(|| {
            format!("device selector '{raw}' must start with pci:, uuid:, or arch:")
        })?;
        if value.is_empty() || value.chars().any(char::is_whitespace) {
            return Err(format!(
                "device selector '{raw}' has an empty or whitespace value"
            ));
        }
        match kind {
            "pci" if valid_pci_bdf(value) => Ok(Self::PciBdf(value.to_ascii_lowercase())),
            "pci" => Err(format!(
                "PCI selector '{raw}' must use domain:bus:device.function (for example 0000:03:00.0)"
            )),
            "uuid" => Ok(Self::Uuid(value.to_owned())),
            "arch" if value.starts_with("gfx") => Ok(Self::ExactArch(value.to_ascii_lowercase())),
            "arch" => Err(format!(
                "architecture selector '{raw}' must name an exact gfx target"
            )),
            _ => Err(format!("unknown device selector kind '{kind}'")),
        }
    }
}

fn valid_pci_bdf(value: &str) -> bool {
    let Some((domain, rest)) = value.split_once(':') else {
        return false;
    };
    let Some((bus, rest)) = rest.split_once(':') else {
        return false;
    };
    let Some((device, function)) = rest.split_once('.') else {
        return false;
    };
    let hex = |part: &str, width: usize| {
        part.len() == width && part.bytes().all(|byte| byte.is_ascii_hexdigit())
    };
    hex(domain, 4) && hex(bus, 2) && hex(device, 2) && hex(function, 1)
}

/// User-facing DeepSeek placement. This is intentionally model-specific so a
/// DS4 split can never alter Qwen or the process-wide mixed-arch policy.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum Deepseek4ComputePlacement {
    #[default]
    Single,
    DenseExpertSplit {
        dense: DeviceSelector,
        experts: DeviceSelector,
    },
}

/// Storage policy for DeepSeek V4's long-lived compressor caches. This is
/// deliberately separate from the ordinary KV-cache mode: it affects only
/// the model's compressed main/indexer memory and never Qwen or other models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Deepseek4CompressorCache {
    #[default]
    F32,
    F16,
}

impl fmt::Display for Deepseek4CompressorCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        })
    }
}

impl std::str::FromStr for Deepseek4CompressorCache {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            _ => Err("DeepSeek V4 compressor cache must be f32 or f16".to_string()),
        }
    }
}

impl fmt::Display for Deepseek4ComputePlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single => f.write_str("single"),
            Self::DenseExpertSplit { dense, experts } => {
                write!(f, "dense-expert-split(dense={dense},experts={experts})")
            }
        }
    }
}

impl std::str::FromStr for Deepseek4ComputePlacement {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        if raw == "single" {
            return Ok(Self::Single);
        }
        let body = raw
            .strip_prefix("dense-expert-split(dense=")
            .and_then(|value| value.strip_suffix(')'))
            .ok_or_else(|| {
                "expected single or dense-expert-split(dense=<selector>,experts=<selector>)"
                    .to_string()
            })?;
        let (dense, experts) = body
            .split_once(",experts=")
            .ok_or_else(|| "dense-expert-split requires dense and experts selectors".to_string())?;
        let dense = dense.parse()?;
        let experts = experts.parse()?;
        if dense == experts {
            return Err("dense and experts selectors must identify distinct devices".into());
        }
        Ok(Self::DenseExpertSplit { dense, experts })
    }
}

impl ConfigValue {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Integer(_) => "integer",
            Self::Float(_) => "number",
            Self::String(_) => "string",
            Self::Null => "null",
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    fn into_toml(self) -> Option<toml::Value> {
        match self {
            Self::Bool(v) => Some(toml::Value::Boolean(v)),
            Self::Integer(v) => Some(toml::Value::Integer(v)),
            Self::Float(v) => Some(toml::Value::Float(v)),
            Self::String(v) => Some(toml::Value::String(v)),
            Self::Null => None,
        }
    }

    /// Render as a TOML scalar (`"q8"`, `4096`, `true`, `0.42`), the form a
    /// user would see in `config.toml`. `Null` has no TOML form and renders as
    /// `null`.
    pub fn to_toml_string(&self) -> String {
        match self.clone().into_toml() {
            Some(value) => value.to_string(),
            None => "null".to_owned(),
        }
    }

    fn from_toml(value: &toml::Value) -> Option<Self> {
        match value {
            toml::Value::Boolean(v) => Some(Self::Bool(*v)),
            toml::Value::Integer(v) => Some(Self::Integer(*v)),
            toml::Value::Float(v) => Some(Self::Float(*v)),
            toml::Value::String(v) => Some(Self::String(v.clone())),
            _ => None,
        }
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Integer(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::String(v) => f.write_str(v),
            Self::Null => f.write_str("null"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigCategory {
    Generation,
    Reasoning,
    Hardware,
    Kernel,
    Memory,
    Attention,
    Speculation,
    Vision,
    Replay,
    Fusions,
    Prompt,
    Serve,
    Diagnostic,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigScope {
    Process,
    ModelLoad,
    Session,
    Request,
    Diagnostic,
}

#[derive(Clone, Copy, Debug)]
pub enum DefaultValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(&'static str),
    Null,
}

impl DefaultValue {
    fn to_value(self) -> ConfigValue {
        match self {
            Self::Bool(v) => ConfigValue::Bool(v),
            Self::Integer(v) => ConfigValue::Integer(v),
            Self::Float(v) => ConfigValue::Float(v),
            Self::String(v) => ConfigValue::String(v.to_owned()),
            Self::Null => ConfigValue::Null,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ValueRule {
    Bool,
    Integer {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
        min_inclusive: bool,
    },
    String,
    NonEmptyString,
    Host,
    PathOrEmpty,
    AbsolutePathOrEmpty,
    Enum(&'static [&'static str]),
    AutoBool,
    NullableString,
    NullableEnum(&'static [&'static str]),
    NullableInteger {
        min: i64,
        max: i64,
    },
    /// The string `auto` or an integer in `min..=max`.
    AutoInteger {
        min: i64,
        max: i64,
    },
    /// Like [`ValueRule::NullableInteger`], but also accepts the literal
    /// `auto` (case-insensitive) — the spelling the TUI rows and the help text
    /// use for these budget keys, and the one an operator is likely to type.
    /// `-1` remains the wire spelling that `auto` is equivalent to.
    NullableIntegerOrAuto {
        min: i64,
        max: i64,
    },
    NullableFloat {
        min: f64,
        max: f64,
    },
    KvAdaptive,
    /// `auto` | a GPU share in `[0.0, 0.5]`. The schedule divides a spilled step's
    /// output rows between the two engines (GPU = the `pcie` path, CPU = the `cpu`
    /// path); `0.0` collapses it to the CPU engine alone, i.e. the step runs exactly
    /// as `memory.offload_exec=cpu`. The upper bound is 0.5 because a share above
    /// half the rows is better expressed by switching `memory.offload_exec` to `pcie`.
    PassbackShare,
    /// `legacy` | `vmm`. Rejects the pre-rename `contiguous` spelling with a
    /// migration message rather than a generic enum failure.
    KvBackend,
    Deepseek4Placement,
}

#[derive(Clone, Copy, Debug)]
pub struct ConfigField {
    pub key: &'static str,
    pub legacy_key: &'static str,
    pub category: ConfigCategory,
    pub scope: ConfigScope,
    pub default: DefaultValue,
    pub rule: ValueRule,
    pub registry_allowed: bool,
    pub experimental: bool,
    pub env_compat: Option<&'static str>,
    pub include_builtin_in_process_config: bool,
    pub help: &'static str,
}

impl ConfigField {
    pub fn validate(&self, value: &ConfigValue) -> Result<()> {
        let valid = match self.rule {
            ValueRule::Bool => matches!(value, ConfigValue::Bool(_)),
            ValueRule::Integer { min, max } => {
                matches!(value, ConfigValue::Integer(v) if *v >= min && *v <= max)
            }
            ValueRule::Float {
                min,
                max,
                min_inclusive,
            } => value
                .as_f64()
                .is_some_and(|v| (if min_inclusive { v >= min } else { v > min }) && v <= max),
            ValueRule::String => matches!(value, ConfigValue::String(_)),
            ValueRule::NonEmptyString => {
                matches!(value, ConfigValue::String(v) if !v.trim().is_empty())
            }
            ValueRule::Host => matches!(value, ConfigValue::String(v)
                if !v.is_empty() && v.len() <= 255 && v.trim() == v && !v.chars().any(char::is_whitespace)),
            ValueRule::PathOrEmpty => matches!(value, ConfigValue::String(v) if {
                if v.is_empty() {
                    true
                } else {
                    expand_tilde(v).is_file()
                }
            }),
            ValueRule::AbsolutePathOrEmpty => matches!(value, ConfigValue::String(v)
                if v.is_empty() || (v.trim() == v && Path::new(v).is_absolute())),
            ValueRule::Enum(values) => {
                matches!(value, ConfigValue::String(v) if values.contains(&v.as_str()))
            }
            ValueRule::AutoBool => {
                matches!(value, ConfigValue::Bool(_))
                    || matches!(value, ConfigValue::String(v) if v == "auto")
            }
            ValueRule::AutoInteger { min, max } => {
                matches!(value, ConfigValue::String(v) if v == "auto")
                    || matches!(value, ConfigValue::Integer(v) if *v >= min && *v <= max)
            }
            ValueRule::NullableString => {
                matches!(value, ConfigValue::Null | ConfigValue::String(_))
            }
            ValueRule::NullableEnum(values) => {
                matches!(value, ConfigValue::Null)
                    || matches!(value, ConfigValue::String(v) if values.contains(&v.as_str()))
            }
            ValueRule::NullableInteger { min, max } => {
                matches!(value, ConfigValue::Null)
                    || matches!(value, ConfigValue::Integer(v) if *v >= min && *v <= max)
            }
            ValueRule::NullableIntegerOrAuto { min, max } => {
                matches!(value, ConfigValue::Null)
                    || matches!(value, ConfigValue::Integer(v) if *v >= min && *v <= max)
                    || matches!(value, ConfigValue::String(v) if v.trim().eq_ignore_ascii_case("auto"))
            }
            ValueRule::NullableFloat { min, max } => {
                matches!(value, ConfigValue::Null)
                    || value.as_f64().is_some_and(|v| v >= min && v <= max)
            }
            ValueRule::KvAdaptive => matches!(value, ConfigValue::String(v) if {
                matches!(v.as_str(), "off" | "conservative" | "balanced" | "aggressive")
                    || valid_advanced_kv(v)
            }),
            ValueRule::KvBackend => {
                matches!(value, ConfigValue::String(v) if matches!(v.as_str(), "legacy" | "vmm"))
            }
            ValueRule::PassbackShare => matches!(value, ConfigValue::String(v) if {
                v == "auto" || v.parse::<f64>().is_ok_and(|f| (0.0..=0.5).contains(&f))
            }),
            ValueRule::Deepseek4Placement => matches!(value, ConfigValue::String(v)
                if v.parse::<Deepseek4ComputePlacement>().is_ok()),
        };

        if valid {
            Ok(())
        } else if matches!(self.rule, ValueRule::KvBackend)
            && matches!(value, ConfigValue::String(v) if v == "contiguous")
        {
            Err(ConfigError::InvalidValue {
                key: self.key.to_owned(),
                message: "KV backend 'contiguous' was renamed to 'legacy'; use --kv-backend legacy or memory.kv_backend = \"legacy\"".into(),
            })
        } else {
            Err(ConfigError::InvalidValue {
                key: self.key.to_owned(),
                message: format!("{} does not satisfy {:?}", value.kind(), self.rule),
            })
        }
    }

    pub fn parse_cli(&self, raw: &str) -> Result<ConfigValue> {
        let value = match self.rule {
            ValueRule::Bool => parse_bool(raw).map(ConfigValue::Bool),
            ValueRule::Integer { .. } => raw.parse::<i64>().ok().map(ConfigValue::Integer),
            ValueRule::Float { .. } => raw.parse::<f64>().ok().map(ConfigValue::Float),
            ValueRule::AutoBool if raw == "auto" => Some(ConfigValue::String(raw.to_owned())),
            ValueRule::AutoBool => parse_bool(raw).map(ConfigValue::Bool),
            ValueRule::AutoInteger { .. } if raw.eq_ignore_ascii_case("auto") => {
                Some(ConfigValue::String("auto".to_owned()))
            }
            ValueRule::AutoInteger { .. } => raw.parse::<i64>().ok().map(ConfigValue::Integer),
            ValueRule::NullableString if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableEnum(_) if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableInteger { .. } if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableInteger { .. } => raw.parse::<i64>().ok().map(ConfigValue::Integer),
            ValueRule::NullableIntegerOrAuto { .. } if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableIntegerOrAuto { .. } if raw.trim().eq_ignore_ascii_case("auto") => {
                Some(ConfigValue::String(raw.to_owned()))
            }
            ValueRule::NullableIntegerOrAuto { .. } => {
                raw.parse::<i64>().ok().map(ConfigValue::Integer)
            }
            ValueRule::NullableFloat { .. } if raw.eq_ignore_ascii_case("null") => {
                Some(ConfigValue::Null)
            }
            ValueRule::NullableFloat { .. } => raw.parse::<f64>().ok().map(ConfigValue::Float),
            _ => Some(ConfigValue::String(raw.to_owned())),
        }
        .ok_or_else(|| ConfigError::InvalidValue {
            key: self.key.to_owned(),
            message: format!("could not parse '{raw}'"),
        })?;
        self.validate(&value)?;
        Ok(value)
    }
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn valid_advanced_kv(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("advanced:k=") else {
        return false;
    };
    let Some((k, v)) = rest.split_once(",v=") else {
        return false;
    };
    matches!(k, "fwht4" | "fwht3" | "fwht2") && matches!(v, "lloyd4" | "lloyd3" | "lloyd2")
}

fn expand_tilde(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(value)
}

// Qwen-only K/V axis overrides. Empty is the unset default (not an authored
// format); consumers omit BuiltIn empty values from IPC.
const KV_K_NAMES: &[&str] = &[
    "",
    "q8",
    "fwht2",
    "fwht3",
    "fwht4",
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV and the asymN/turboN aliases are superseded by fwht3
    "asym2",
    "asym3",
    "asym4",
    "turbo",
    "turbo2",
    "turbo3",
    "turbo4",
    "legacy-asym2",
    "legacy-asym3",
    "legacy-asym4",
];
const KV_V_NAMES: &[&str] = &["", "q8", "lloyd2", "lloyd3", "lloyd4"];

// The union of every KV-mode name any SITE accepts. This is the config
// schema's allow-list only — it is NOT a promise that a given model supports a
// mode. Per-site acceptance lives in `hipfire_runtime::kv_mode`'s policies,
// which warn and fall back for anything they cannot allocate. `bf16` is
// maple's default plus the Qwen quality-control arm; `fp8` is admitted at
// the single-GPU Qwen sites under the carrier's exact gfx1201/geometry guards.
// `legacy-asym3` is the Gemma4 lowered full-tier opt-out back to Givens asym3
// (on Qwen it names the same legacy K as `--kv-k legacy-asym3`).
const KV_MODES: &[&str] = &[
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV and the asymN/turboN aliases are superseded by fwht3 (asymN / turbo*)
    "auto",
    "f32",
    "f16",
    "bf16",
    "q8",
    "asym4",
    "asym3",
    "asym2",
    "fwht4",
    "fwht3",
    "fwht2",
    "turbo",
    "turbo4",
    "turbo3",
    "turbo2",
    "fp8",
    "legacy-asym3",
];
// GDN/DeltaNet recurrent-state storage names accepted by `memory.state_quant`.
// A schema allow-list only: the Qwen3.5 (`parse_state_quant`) and Qwen4
// (`resolve_state_format`) loaders each honor or refuse what they cannot serve
// (`q4`/`int4` are Qwen3.5-only).
const STATE_QUANT_MODES: &[&str] = &["auto", "q8", "int8", "q8_ef", "fp32", "f32", "q4", "int4"];
const AUTO_ON_OFF: &[&str] = &["auto", "on", "off"];
const VISION_MODES: &[&str] = &["auto", "resident", "on", "off"];
/// VL image decode path: `cpu` (default) / `vcn` / `auto` (VCN when probed).
const IMAGE_DECODE_MODES: &[&str] = &["cpu", "vcn", "auto"];
// `off` disables thinking outright. It resolves to a cap of 1, the engine's
// established "no thinking" sentinel (the daemon reads
// `enable_thinking: max_think_tokens != 1`) and the same value the OpenAI
// `enable_thinking=false` / `reasoning_effort="none"` paths already send. It is
// NOT 0 — 0 means `uncapped` (think until the model closes the block itself).
const THINKING_BUDGETS: &[&str] = &["off", "low", "med", "high", "xhigh", "max", "uncapped"];
// Qwen3.8's published effort ladder is `low|medium|xhigh` (default xhigh).
// Keep generic OpenAI-style values (`auto|none|high|max`) alongside it so
// non-Qwen3.8 parents still validate. Values pass through as request strings;
// model-specific mapping lives downstream of config validation.
pub const REASONING_EFFORTS: &[&str] = &["auto", "none", "low", "medium", "high", "xhigh", "max"];
// Which engine executes a spilled layer's weight-reading GEMVs. Exported for the
// same reason as `REASONING_EFFORTS`: the TUI's option list must be this list, or
// a value the schema accepts becomes unselectable (and an unselectable value
// cycles from the wrong place).
pub const OFFLOAD_EXECS: &[&str] = &["pcie", "cpu", "passback"];
/// Values of `memory.ple_format`; exported so the TUI offers exactly this list.
pub const PLE_FORMATS: &[&str] = &["auto", "q4", "q8"];
/// Values of `speculation.mode`; exported so the TUI offers exactly this list.
pub const SPECULATION_MODES: &[&str] = &["off", "auto", "ngram", "dflash", "mtp", "dspark"];

macro_rules! field {
    ($key:literal, $legacy:literal, $category:ident, $scope:ident, $default:expr, $rule:expr, $registry:expr, $experimental:expr, $env:expr, $help:literal) => {
        ConfigField {
            key: $key,
            legacy_key: $legacy,
            category: ConfigCategory::$category,
            scope: ConfigScope::$scope,
            default: $default,
            rule: $rule,
            registry_allowed: $registry,
            experimental: $experimental,
            env_compat: $env,
            include_builtin_in_process_config: true,
            help: $help,
        }
    };
}

macro_rules! bridge_field {
    ($key:literal, $legacy:literal, $category:ident, $scope:ident, $default:expr, $rule:expr, $experimental:expr, $env:literal, $help:literal) => {
        ConfigField {
            key: $key,
            legacy_key: $legacy,
            category: ConfigCategory::$category,
            scope: ConfigScope::$scope,
            default: $default,
            rule: $rule,
            registry_allowed: false,
            experimental: $experimental,
            env_compat: Some($env),
            include_builtin_in_process_config: false,
            help: $help,
        }
    };
}

// These fields preserve legacy environment spellings while exposing stable
// TOML policy to the typed ProcessConfig handoff consumed by hipfire-runtime
// and rdna-compute. They are process-scoped because those crates snapshot the
// values once; advertising per-model overrides would be dishonest for a
// long-lived serve process.
macro_rules! process_bool_field {
    ($key:literal, $legacy:literal, $category:ident, $default:expr, $experimental:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            $category,
            Process,
            DefaultValue::Bool($default),
            ValueRule::Bool,
            $experimental,
            $env,
            $help
        )
    };
}

macro_rules! process_auto_bool_field {
    ($key:literal, $legacy:literal, $category:ident, $experimental:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            $category,
            Process,
            DefaultValue::String("auto"),
            ValueRule::AutoBool,
            $experimental,
            $env,
            $help
        )
    };
}

macro_rules! diagnostic_bool_field {
    ($key:literal, $legacy:literal, $default:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            Diagnostic,
            Diagnostic,
            DefaultValue::Bool($default),
            ValueRule::Bool,
            true,
            $env,
            $help
        )
    };
}

macro_rules! process_field {
    ($key:literal, $legacy:literal, $category:ident, $default:expr, $rule:expr, $experimental:expr, $env:literal, $help:literal) => {
        bridge_field!(
            $key,
            $legacy,
            $category,
            Process,
            $default,
            $rule,
            $experimental,
            $env,
            $help
        )
    };
}

macro_rules! diagnostic_field {
    ($key:literal, $legacy:literal, $default:expr, $rule:expr, $env:literal, $help:literal) => {
        bridge_field!($key, $legacy, Diagnostic, Diagnostic, $default, $rule, true, $env, $help)
    };
}

/// The public schema. Canonical dotted keys are the TOML surface; `legacy_key`
/// accepts the current flat JSON/CLI spelling during migration.
pub static FIELDS: &[ConfigField] = &[
    field!(
        "memory.kv_cache",
        "kv_cache",
        Memory,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(KV_MODES),
        true,
        false,
        Some("HIPFIRE_KV_MODE"),
        "KV cache format; auto inherits the registry recommendation, then q8 — except single-GPU Qwen on exact gfx1201, where auto means native fp8 (stage-b FA2 arithmetic). DeepSeek V4 currently supports f32 and f16. asymN and turbo* are deprecated (removal in 0.5.0); use fwhtN or q8."
    ),
    field!(
        "memory.kv_adaptive",
        "kv_adaptive",
        Memory,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::KvAdaptive,
        true,
        false,
        Some("HIPFIRE_KV_ADAPTIVE"),
        "Runtime VRAM-fit KV precision policy."
    ),
    field!(
        "memory.state_quant",
        "state_quant",
        Memory,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(STATE_QUANT_MODES),
        true,
        false,
        Some("HIPFIRE_STATE_QUANT"),
        "GDN/DeltaNet recurrent-state storage; auto = per-model/arch default (Qwen3.8-Flash-Next: F32 on gfx1151, Q8 elsewhere; Qwen3.5: Q8). Explicit values are honored or refused, never rewritten; set q8 to keep Q8 state on gfx1151. q8_ef (Qwen3.8-Flash-Next only, explicit opt-in) is Q8 plus a live f16 error-feedback residual that resets on rollback/restore. q4/int4 are Qwen3.5-only."
    ),
    field!(
        "memory.gpu_layer_budget",
        "gpu_layer_budget",
        Memory,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableIntegerOrAuto { min: -1, max: 65536 },
        true,
        false,
        Some("HIPFIRE_GPU_LAYER_BUDGET"),
        "Resident-layer budget for partial GPU offload: N keeps the last N layers on the GPU and spills the rest to system RAM; unset/null keeps every layer on the GPU on every model, including MoE. The number counts layers ON the GPU, not layers offloaded — 3 on a 64-layer model spills 61. Explicit 'auto' (-1) opts into placement by fit: the smallest spill that loads, routed experts first, and a model that already fits is left fully resident."
    ),
    field!(
        "memory.moe_expert_budget",
        "moe_expert_budget",
        Memory,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableIntegerOrAuto { min: -1, max: 65536 },
        true,
        false,
        Some("HIPFIRE_MOE_EXPERT_BUDGET"),
        "Resident routed-expert budget for a Mixture-of-Experts model whose experts exceed VRAM: N keeps the last N layers' routed experts on the GPU and spills the rest to system RAM; unset/null keeps every routed expert on the GPU. The number counts layers whose expert weights stay ON the GPU — llama.cpp's `--n-cpu-moe K` is this value written as `n_layers - K`. Only routed experts move: attention, router, shared expert and KV stay on the card. Explicit 'auto' (-1) opts into placement by fit, spilling routed experts before whole layers. No effect on a dense model, and none on an architecture that cannot spill experts (a configured spill fails that arch's load rather than being ignored)."
    ),
    // Sparse in the process config (like `replay.backend`): when the user,
    // registry or a compatibility env never set it, the key is absent, so the
    // loader can tell "unset" (a Flash-Next load then picks the engine from a
    // measured decode-cost comparison, see `memory::offload_exec_explicit`) from an
    // explicit `pcie`. The built-in `pcie` stays the raw value
    // ([`memory::offload_exec`]) every other consumer reads.
    ConfigField {
        key: "memory.offload_exec",
        legacy_key: "offload_exec",
        category: ConfigCategory::Memory,
        scope: ConfigScope::ModelLoad,
        default: DefaultValue::String("pcie"),
        rule: ValueRule::Enum(OFFLOAD_EXECS),
        registry_allowed: true,
        experimental: false,
        env_compat: Some("HIPFIRE_OFFLOAD_EXEC"),
        include_builtin_in_process_config: false,
        help: "Which engine executes the ops that read a spilled layer's weights: 'pcie' runs the GPU kernels against host-mapped weights over the link; 'cpu' executes those GEMVs on the CPU instead; 'passback' is scheduled co-inference — both engines run the same spilled step concurrently (the GPU arm is the 'pcie' path, the CPU arm is the 'cpu' path), divided by output rows per 'memory.offload_passback_share'. Not a third engine and not a fallback: it decides who multiplies, never what is spilled — placement stays memory.gpu_layer_budget, and the KV cache stays in VRAM either way. Unset behaves as 'pcie' everywhere except one measured default: a Flash-Next (qwen4) load whose host-placed routed experts the D2 CPU split can serve (hot-set tier layers, or un-tiered aggregate layers whose allocations are host-mapped QT44/QT53) runs only that D2 split (decode and <=4-row verify of the host-mapped experts) when its predicted per-layer decode cost is at least 15% below PCIe's: the production CPU arm timed on this model's real host experts plus the copy-transport handoff, against the live zero-copy host-link rate plus a launch floor, compared pessimistically (the slowest of 5 CPU arm samples against the fastest of 5 link samples); unified-memory devices, a failed or inconclusive measurement and links the CPU does not beat by that margin stay on the link, and prefill, placement and host allocations never change. The typed developer key developer.qwen4_d2_default forces or vetoes that comparison. An explicit value, 'pcie' included, always wins. Empty and unknown values fall back to 'pcie'.",
    },
    field!(
        "memory.offload_passback_share",
        "offload_passback_share",
        Memory,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::PassbackShare,
        true,
        false,
        Some("HIPFIRE_OFFLOAD_PASSBACK_SHARE"),
        "How a spilled step's output rows are divided between the two engines under scheduled co-inference (memory.offload_exec=passback only): the GPU takes the first rows (the 'pcie' arm) while the CPU runs the rest (the 'cpu' arm). 'auto' (default) schedules the split from this host's measured engine rates and refines it online; a number in (0, 0.5] pins it; 0 collapses the split onto the CPU engine alone, so the mode is then byte-identical to memory.offload_exec=cpu."
    ),
    // Process-scoped: the preflight guards snapshot this once at startup, and
    // a mid-serve flip would make the refusal policy depend on which load ran
    // last — dishonest for a long-lived daemon.
    process_auto_bool_field!(
        "memory.oom_guard",
        "oom_guard",
        Memory,
        false,
        "HIPFIRE_OOM_GUARD",
        "Memory preflight OOM guard. Default auto: on for unified-memory APU architectures (GPU allocations come out of system RAM, so an overshoot can globally OOM the desktop), off for discrete GPUs, and for GPU-less processes decided by host swap state. Set true to force on, false to force off (HIPFIRE_OOM_GUARD)."
    ),
    field!(
        "model.deepseek4_experts_per_token",
        "deepseek4_experts_per_token",
        Experimental,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 1, max: 6 },
        true,
        false,
        None,
        "DeepSeek V4 routed experts per token; null preserves the checkpoint default."
    ),
    field!(
        "hardware.deepseek4_compute_placement",
        "deepseek4_compute_placement",
        Hardware,
        ModelLoad,
        DefaultValue::String("single"),
        ValueRule::Deepseek4Placement,
        false,
        false,
        None,
        "DeepSeek V4 compute placement; single or a typed dense/expert device split."
    ),
    field!(
        "attention.flash",
        "flash_mode",
        Attention,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "always", "never"]),
        true,
        false,
        Some("HIPFIRE_ATTN_FLASH"),
        "Flash-attention admission preference."
    ),
    field!(
        "attention.yarn_factor",
        "yarn_factor",
        Attention,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableFloat { min: 1.0, max: 64.0 },
        true,
        false,
        Some("HIPFIRE_YARN_FACTOR"),
        "YaRN RoPE scaling factor override. Default null = the checkpoint's own rope_parameters; above 1 enables YaRN over the checkpoint's original context (factor x original positions); 1 disables a checkpoint's YaRN."
    ),
    field!(
        "serve.default_model",
        "default_model",
        Serve,
        Process,
        DefaultValue::String("qwen3.5:9b"),
        ValueRule::NonEmptyString,
        false,
        false,
        Some("HIPFIRE_MODEL"),
        "Model pre-warmed by serve."
    ),
    process_bool_field!(
        "serve.local",
        "local",
        Serve,
        false,
        false,
        "HIPFIRE_LOCAL",
        "Force the current command to use a locally spawned daemon."
    ),
    process_bool_field!(
        "serve.multi_slot",
        "multi_slot",
        Serve,
        false,
        false,
        "HIPFIRE_SERVE_MULTI_SLOT",
        "Serve requests concurrently on the multi-slot engine instead of one at a time. \
         Concurrency is per arch: the qwen3.5 generation engine needs memory.kv_cache=q8 \
         (it refuses every other value, including the default auto), while the arch-17 \
         embedding encoder has no KV cache at all and merges concurrent encodes into one \
         batched forward."
    ),
    field!(
        "serve.multi_slot_slots",
        "multi_slot_slots",
        Serve,
        Process,
        DefaultValue::Integer(4),
        ValueRule::Integer { min: 1, max: 64 },
        false,
        false,
        Some("HIPFIRE_SERVE_MULTI_SLOT_SLOTS"),
        "Concurrent slots for the multi-slot engine."
    ),
    field!(
        "serve.multi_slot_ctx",
        "multi_slot_ctx",
        Serve,
        Process,
        DefaultValue::Integer(8192),
        ValueRule::Integer {
            min: 512,
            max: 1048576
        },
        false,
        false,
        Some("HIPFIRE_SERVE_MULTI_SLOT_CTX"),
        "Per-slot context capacity (tokens) for the multi-slot engine."
    ),
    field!(
        "serve.multi_slot_prefill_chunk",
        "multi_slot_prefill_chunk",
        Serve,
        Process,
        DefaultValue::Integer(1024),
        ValueRule::Integer {
            min: 1,
            max: 1048576
        },
        false,
        false,
        Some("HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK"),
        "Prefill tokens taken from one slot per multi-slot step; batch scratch is sized n_slots x this."
    ),
    field!(
        "generation.temperature",
        "temperature",
        Generation,
        Request,
        DefaultValue::Float(0.3),
        ValueRule::Float {
            min: 0.0,
            max: 2.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Sampling temperature."
    ),
    field!(
        "decide.calibration.choice",
        "decide_calibration_choice",
        Generation,
        Request,
        DefaultValue::Float(1.0),
        ValueRule::Float {
            min: 0.05,
            max: 20.0,
            min_inclusive: true
        },
        false,
        false,
        None,
        "Decide (/v1/systemone) temperature T for choice answers: probabilities are softmax(label logits / T); 1 = raw. Set per model from a fit (docs/specs/2026-09-28-jev-decide-design.md section 13)."
    ),
    field!(
        "decide.calibration.score",
        "decide_calibration_score",
        Generation,
        Request,
        DefaultValue::Float(1.0),
        ValueRule::Float {
            min: 0.05,
            max: 20.0,
            min_inclusive: true
        },
        false,
        false,
        None,
        "Decide (/v1/systemone) temperature T for score answers: probabilities are softmax(label logits / T); 1 = raw. Set per model from a fit (docs/specs/2026-09-28-jev-decide-design.md section 13)."
    ),
    field!(
        "decide.calibration.noul",
        "decide_calibration_noul",
        Generation,
        Request,
        DefaultValue::Float(1.0),
        ValueRule::Float {
            min: 0.05,
            max: 20.0,
            min_inclusive: true
        },
        false,
        false,
        None,
        "Decide (/v1/systemone) temperature T for noul (yes/no) answers: p(yes) = sigmoid(log-odds / T); 1 = raw. Set per model from a fit (docs/specs/2026-09-28-jev-decide-design.md section 13)."
    ),
    field!(
        "generation.top_p",
        "top_p",
        Generation,
        Request,
        DefaultValue::Float(0.8),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: false
        },
        true,
        false,
        None,
        "Nucleus sampling probability."
    ),
    field!(
        "generation.top_k",
        "top_k",
        Generation,
        Request,
        DefaultValue::Integer(20),
        ValueRule::Integer {
            min: 1,
            max: 100000
        },
        true,
        false,
        None,
        "Maximum number of highest-probability tokens retained for sampling."
    ),
    field!(
        "generation.min_p",
        "min_p",
        Generation,
        Request,
        DefaultValue::Float(0.0),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Minimum probability relative to the highest-probability token; zero disables."
    ),
    field!(
        "generation.presence_penalty",
        "presence_penalty",
        Generation,
        Request,
        DefaultValue::Float(0.0),
        ValueRule::Float {
            min: 0.0,
            max: 2.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "OpenAI-style flat penalty for tokens already present in the repeat window."
    ),
    field!(
        "generation.repeat_penalty",
        "repeat_penalty",
        Generation,
        Request,
        DefaultValue::Float(1.05),
        ValueRule::Float {
            min: 1.0,
            max: 3.0,
            min_inclusive: true
        },
        true,
        false,
        None,
        "Token repetition penalty."
    ),
    field!(
        "generation.max_tokens",
        "max_tokens",
        Generation,
        Request,
        DefaultValue::Integer(4096),
        ValueRule::Integer {
            min: 1,
            max: 393216
        },
        true,
        false,
        None,
        "Per-turn generation cap."
    ),
    field!(
        "memory.max_seq",
        "max_seq",
        Memory,
        ModelLoad,
        DefaultValue::Integer(32768),
        ValueRule::Integer {
            min: 512,
            max: 1048576
        },
        true,
        false,
        None,
        "Logical KV context capacity."
    ),
    field!(
        "memory.kv_backend",
        "kv_backend",
        Memory,
        ModelLoad,
        DefaultValue::String("vmm"),
        ValueRule::KvBackend,
        true,
        false,
        None,
        "KV storage backend: legacy (physically contiguous) or vmm (reserves the logical context window and commits physical pages on demand)."
    ),
    field!(
        "memory.kv_k",
        "kv_k",
        Memory,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::Enum(KV_K_NAMES),
        true,
        false,
        None,
        "Qwen-only K-axis override; empty leaves the mode-preset K unchanged. asymN, turboN and legacy-asymN are deprecated (removal in 0.5.0); use fwhtN or q8."
    ),
    field!(
        "memory.kv_v",
        "kv_v",
        Memory,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::Enum(KV_V_NAMES),
        true,
        false,
        None,
        "Qwen-only V-axis override; empty leaves the mode-preset V unchanged."
    ),
    field!(
        "reasoning.mode",
        "thinking",
        Reasoning,
        Session,
        DefaultValue::String("on"),
        ValueRule::Enum(&["on", "off"]),
        true,
        false,
        None,
        "Visible reasoning mode."
    ),
    field!(
        "reasoning.effort",
        "reasoning_effort",
        Reasoning,
        Request,
        DefaultValue::String("auto"),
        ValueRule::Enum(REASONING_EFFORTS),
        true,
        false,
        None,
        "Model-specific reasoning effort; auto preserves the carrier fallback."
    ),
    field!(
        "reasoning.budget",
        "thinking_budget",
        Reasoning,
        Request,
        DefaultValue::String("med"),
        ValueRule::Enum(THINKING_BUDGETS),
        true,
        false,
        None,
        "Named per-turn reasoning budget."
    ),
    field!(
        "reasoning.max_tokens",
        "max_think_tokens",
        Reasoning,
        Request,
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: 393216
        },
        true,
        false,
        None,
        "Raw reasoning-token override; absent means use the named budget."
    ),
    field!(
        "reasoning.max_total_tokens",
        "max_total_think_tokens",
        Reasoning,
        Request,
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: 1000000
        },
        true,
        false,
        Some("HIPFIRE_MAX_TOTAL_THINK_TOKENS"),
        "Cross-reopen reasoning budget; zero disables it."
    ),
    field!(
        "serve.host",
        "host",
        Serve,
        Process,
        DefaultValue::String("127.0.0.1"),
        ValueRule::Host,
        false,
        false,
        Some("HIPFIRE_HOST"),
        "Serve bind address."
    ),
    field!(
        "serve.port",
        "port",
        Serve,
        Process,
        DefaultValue::Integer(11435),
        ValueRule::Integer { min: 1, max: 65535 },
        false,
        false,
        Some("HIPFIRE_PORT"),
        "Serve TCP port."
    ),
    field!(
        "serve.idle_timeout_seconds",
        "idle_timeout",
        Serve,
        Process,
        DefaultValue::Integer(300),
        ValueRule::Integer { min: 0, max: 86400 },
        false,
        false,
        Some("HIPFIRE_IDLE_TIMEOUT"),
        "Seconds before idle model unload; zero disables."
    ),
    field!(
        "serve.max_request_bytes",
        "max_request_bytes",
        Serve,
        Process,
        DefaultValue::Integer(67108864),
        ValueRule::Integer {
            min: 4096,
            max: 4294967296
        },
        false,
        false,
        Some("HIPFIRE_MAX_REQUEST_BYTES"),
        "Maximum request-body bytes."
    ),
    process_bool_field!(
        "serve.ui",
        "serve_ui",
        Serve,
        false,
        false,
        "HIPFIRE_SERVE_UI",
        "Serve the embedded chat UI at /ui on the serve listener; default off."
    ),
    field!(
        "serve.max_queue",
        "serve_max_queue",
        Serve,
        Process,
        DefaultValue::Integer(64),
        ValueRule::Integer {
            min: 0,
            max: 100000
        },
        false,
        false,
        Some("HIPFIRE_SERVE_MAX_QUEUE"),
        "Maximum queued serve requests; zero is uncapped."
    ),
    field!(
        "serve.continuous_batch_size",
        "continuous_batch_size",
        Serve,
        Process,
        DefaultValue::Integer(1),
        ValueRule::Integer {
            min: 1,
            max: MAX_CONTINUOUS_BATCH_SIZE as i64
        },
        false,
        false,
        Some("HIPFIRE_CONTINUOUS_BATCH_SIZE"),
        "Maximum coexisting eligible batch lanes for serve; 1 preserves sequential behavior."
    ),
    process_bool_field!(
        "serve.vmm_batch",
        "serve_vmm_batch",
        Serve,
        false,
        true,
        "HIPFIRE_SERVE_VMM_BATCH",
        "Route eligible concurrent requests through the VMM continuous-batching executor (per-request VMM KV, one resident target). Off = existing VMM singleton route; never the fixed-slot legacy route. Developer default off until its route gates pass."
    ),
    process_bool_field!(
        "serve.batch_spec",
        "serve_batch_spec",
        Serve,
        true,
        true,
        "HIPFIRE_SERVE_BATCH_SPEC",
        "Allow cross-request speculative verify rows on the VMM batch route; 0 disables batched speculation only and keeps singleton speculation unchanged."
    ),
    process_bool_field!(
        "serve.batch_nonexact",
        "serve_batch_nonexact",
        Serve,
        false,
        true,
        "HIPFIRE_SERVE_BATCH_NONEXACT",
        "Permit explicitly labeled non-bit-exact batched arithmetic on the VMM batch route. Always default off."
    ),
    process_bool_field!(
        "serve.batch_sampled",
        "serve_batch_sampled",
        Serve,
        true,
        true,
        "HIPFIRE_SERVE_BATCH_SAMPLED",
        "Default on (applies only with serve.vmm_batch). Admit sampled (temperature>0) requests to the VMM batch route only in the lane mode the singleton route selects for them, each proven byte-identical to that singleton (AR, native MTP, FAST_SAMPLE selector-chain DFlash). 0 is the kill switch: sampled requests stay on the serialized singleton route."
    ),
    field!(
        "serve.queue_timeout_ms",
        "serve_queue_timeout_ms",
        Serve,
        Process,
        DefaultValue::Integer(600000),
        ValueRule::Integer {
            min: 0,
            max: 3600000
        },
        false,
        false,
        Some("HIPFIRE_SERVE_QUEUE_TIMEOUT_MS"),
        "Maximum admission-queue wait before a queued request gets 503. Serve runs one generation at a time by default, so this must cover one full generation; zero waits forever."
    ),
    process_bool_field!(
        "serve.retry_enabled",
        "retry_enabled",
        Serve,
        false,
        true,
        "HIPFIRE_SERVE_RETRY_ENABLED",
        "Server-owned single retry on typed transient daemon failures; promoted only after merged-path GPU parity."
    ),
    process_field!(
        "serve.retry_backoff_ms",
        "retry_backoff_ms",
        Serve,
        DefaultValue::Integer(50),
        ValueRule::Integer { min: 0, max: 60000 },
        true,
        "HIPFIRE_SERVE_RETRY_BACKOFF_MS",
        "Backoff before the single serve retry; slept outside runtime and admission locks."
    ),
    process_bool_field!(
        "serve.allow_request_pull",
        "allow_request_pull",
        Serve,
        false,
        false,
        "HIPFIRE_SERVE_ALLOW_REQUEST_PULL",
        "Let a chat request that names a registry model not on disk download it. Off: the request gets 404 and the operator runs `hipfire pull`."
    ),
    process_bool_field!(
        "serve.allow_request_paths",
        "allow_request_paths",
        Serve,
        false,
        false,
        "HIPFIRE_SERVE_ALLOW_REQUEST_PATHS",
        "Let a chat request load any readable file it names. Off: requests may name only installed models (models directory, catalog, pre-warm model)."
    ),
    field!(
        "serve.api_key_file",
        "api_key_file",
        Serve,
        Process,
        DefaultValue::String(""),
        ValueRule::AbsolutePathOrEmpty,
        false,
        false,
        Some("HIPFIRE_SERVE_API_KEY_FILE"),
        "Absolute path of a file holding the serve bearer token (at least 32 bytes after trimming trailing whitespace; Unix mode must grant no group/other access). Empty (unset) keeps serve keyless; the token itself is never stored in configuration."
    ),
    // --- Serving cache/scheduler contract keys (spec §9.1) -----------------
    // All experimental: the cache/scheduler route is planned and has not
    // passed release gates (spec §9.1). Defaults are conservative
    // planner-derived constants, not universal GPU tuning constants; the
    // memory planner resolves final per-route values later (spec §9.1).
    field!(
        "serve.prefix_cache",
        "prefix_cache",
        Serve,
        Process,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        true,
        Some("HIPFIRE_SERVE_PREFIX_CACHE"),
        "Experimental: select cross-session shared-prefix cache reuse; off until the exact route passes release gates (spec §4.1, §9.1)."
    ),
    field!(
        "serve.prefix_cache_max_bytes",
        "prefix_cache_max_bytes",
        Serve,
        Process,
        DefaultValue::Integer(0),
        ValueRule::Integer { min: 0, max: 1099511627776 },
        true,
        true,
        Some("HIPFIRE_SERVE_PREFIX_CACHE_MAX_BYTES"),
        "Experimental: retained shared-prefix cache ceiling including hybrid snapshots; 0 means no retained cache, never unlimited (spec §4.4, §9.1)."
    ),
    field!(
        "serve.swap_host_bytes",
        "swap_host_bytes",
        Serve,
        Process,
        DefaultValue::Integer(17179869184),
        ValueRule::Integer { min: 0, max: 1099511627776 },
        true,
        true,
        Some("HIPFIRE_SWAP_HOST_BYTES"),
        "Multi-slot serve: host-RAM ceiling for parked session swap snapshots, in bytes; 0 disables the host tier so snapshots spill to disk."
    ),
    field!(
        "serve.swap_disk_bytes",
        "swap_disk_bytes",
        Serve,
        Process,
        DefaultValue::Integer(8589934592),
        ValueRule::Integer { min: 0, max: 1099511627776 },
        true,
        true,
        Some("HIPFIRE_SWAP_DISK_BYTES"),
        "Multi-slot serve: disk ceiling for parked session swap snapshots, in bytes; 0 disables disk spill. A snapshot larger than the ceiling is refused without evicting the cache; otherwise the least-recently-used snapshot is evicted and its session re-prefills."
    ),
    field!(
        "serve.swap_dir",
        "swap_dir",
        Serve,
        Process,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        true,
        Some("HIPFIRE_SWAP_DIR"),
        "Multi-slot serve: directory for swap spill files. Empty uses $XDG_RUNTIME_DIR/hipfire/swap/<pid>, falling back to $TMPDIR/hipfire-swap-<pid>. The directory is created 0700, must be owned by this user, is locked to one daemon, and may not be a symlink; stale *.snap files are removed at start and other files are left untouched."
    ),
    field!(
        "serve.multi_slot_session_ttl_secs",
        "multi_slot_session_ttl_secs",
        Serve,
        Process,
        DefaultValue::Integer(3600),
        ValueRule::Integer { min: 0, max: 86400 },
        true,
        true,
        Some("HIPFIRE_SERVE_MULTI_SLOT_SESSION_TTL_SECS"),
        "Multi-slot route: close a parked (swapped or cold) session after this many seconds idle; 0 disables the idle sweep (the hard session cap still applies)."
    ),
    field!(
        "serve.multi_slot_max_sessions",
        "multi_slot_max_sessions",
        Serve,
        Process,
        DefaultValue::Integer(0),
        ValueRule::Integer { min: 0, max: 4096 },
        true,
        true,
        Some("HIPFIRE_SERVE_MULTI_SLOT_MAX_SESSIONS"),
        "Multi-slot route: hard cap on session-table entries; 0 means auto max(64, 16 x serve.multi_slot_slots). Only parked sessions are closed to enforce it."
    ),
    field!(
        "serve.max_batch_tokens",
        "max_batch_tokens",
        Serve,
        Process,
        DefaultValue::Integer(4096),
        ValueRule::Integer { min: 1, max: 1048576 },
        true,
        true,
        Some("HIPFIRE_SERVE_MAX_BATCH_TOKENS"),
        "Experimental: global trunk-row budget enforced per step (sum of prefill + decode + verify + forced rows); 4096 is a conservative planner-derived default, validate against slot count and scratch limits (spec §5.2, §9.1)."
    ),
    field!(
        "serve.prefill_min_tokens",
        "prefill_min_tokens",
        Serve,
        Process,
        DefaultValue::Integer(1),
        ValueRule::Integer { min: 1, max: 1048576 },
        true,
        true,
        Some("HIPFIRE_SERVE_PREFILL_MIN_TOKENS"),
        "Experimental: minimum service quantum when prefill is runnable; positive in mixed mode so an endless decode stream cannot starve prefill (spec §5.3, §9.1)."
    ),
    field!(
        "serve.max_queue_bytes",
        "max_queue_bytes",
        Serve,
        Process,
        DefaultValue::Integer(268435456),
        ValueRule::Integer { min: 1, max: 1099511627776 },
        true,
        true,
        Some("HIPFIRE_SERVE_MAX_QUEUE_BYTES"),
        "Experimental: total canonical pending-input byte budget; positive and finite so queue count alone cannot admit unbounded bytes (spec §5.3, §9.1)."
    ),
    field!(
        "serve.stream_stall_timeout_ms",
        "stream_stall_timeout_ms",
        Serve,
        Process,
        DefaultValue::Integer(30000),
        ValueRule::Integer { min: 0, max: 3600000 },
        true,
        true,
        Some("HIPFIRE_SERVE_STREAM_STALL_TIMEOUT_MS"),
        "Experimental (multi-slot route): a streaming client that leaves its response channel full, or its final frame unflushed, this long is aborted and its queue permit released (spec §5.4, §9.1)."
    ),
    field!(
        "serve.structured_jump_forward",
        "structured_jump_forward",
        Serve,
        Process,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        true,
        Some("HIPFIRE_SERVE_STRUCTURED_JUMP_FORWARD"),
        "Experimental: token-safe jump-forward for provably forced output; off until scalar constrained-equivalence gates pass (spec §1, §9.1)."
    ),
    field!(
        "experimental.budget_alert",
        "experimental_budget_alert",
        Experimental,
        Request,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        false,
        true,
        Some("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT"),
        "Research-only in-band reasoning alert gate."
    ),
    field!(
        "speculation.dflash_adaptive_b",
        "dflash_adaptive_b",
        Speculation,
        ModelLoad,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        false,
        None,
        "Opt in to the DFlash trailing-τ adaptive verify-block width (default fixed block)."
    ),
    field!(
        "speculation.dflash_block",
        "dflash_block",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::AutoInteger { min: 4, max: 16 },
        true,
        false,
        Some("HIPFIRE_DFLASH_BLOCK"),
        "DFlash2 proposal width per serving configuration, never per live batch (batched lanes stay byte-identical to the singleton). Default `auto` = 8 on a `serve.vmm_batch` serve with `continuous_batch_size` >= 4, else 16; `16` restores the previous fixed width. Legacy DFlash drafts keep their declared width."
    ),
    field!(
        "speculation.dflash_lookup",
        "dflash_lookup",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "on", "off", "1", "0"]),
        true,
        false,
        Some("HIPFIRE_DFLASH_LOOKUP"),
        "Request-local verbatim-copy lookup composed into the dense Qwen3.5-class DFlash2 singleton verify window: a Copy window (lookup candidates replace the learned candidates) or a Tail window (first K learned candidates plus a lookup tail, at most 16 verify rows), taken only when measured acceptance per measured window price pays; a request with no lookup proposal runs the incumbent window unchanged. auto (default) and on/1 arm it on a singleton DFlash2 selector chain (no DDTree, no retained-PM4 verify, no CACTUS); off/0 disables it. Developer `HIPFIRE_SPEC_LOOKUP_TAIL=0` disables only the Tail windows."
    ),
    field!(
        "speculation.dflash",
        "dflash_mode",
        Speculation,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::Enum(AUTO_ON_OFF),
        true,
        false,
        Some("HIPFIRE_DFLASH_MODE"),
        "DFlash eligibility policy."
    ),
    field!(
        "vision.mode",
        "vision_mode",
        Vision,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(VISION_MODES),
        true,
        false,
        Some("HIPFIRE_VISION_MODE"),
        "Vision-tower policy: auto (default; Qwen3.5/3.8-VL tower streamed from host/disk per block, no resident tower VRAM; text-only without a sidecar), resident (whole tower in VRAM at load), on (like auto, but the load fails without a tower), off (text-only)."
    ),
    field!(
        "image.decode",
        "image_decode",
        Vision,
        ModelLoad,
        DefaultValue::String("cpu"),
        ValueRule::Enum(IMAGE_DECODE_MODES),
        true,
        false,
        Some("HIPFIRE_IMAGE_DECODE"),
        "VL image JPEG decode path: cpu (default), vcn, or auto (VCN when probed, else cpu)."
    ),
    field!(
        "speculation.dflash_ngram_block",
        "dflash_ngram_block",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::AutoBool,
        true,
        false,
        Some("HIPFIRE_DFLASH_NGRAM_BLOCK"),
        "Verify-path n-gram defense."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.sidecar",
        "cask_sidecar",
        Memory,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        true,
        Some("HIPFIRE_CASK_SIDECAR"),
        "Deprecated CASK (removal in 0.5.0): TriAttention sidecar path; empty disables eviction."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.enabled",
        "cask",
        Memory,
        ModelLoad,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): enable core-aware m-folding."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.budget",
        "cask_budget",
        Memory,
        ModelLoad,
        DefaultValue::Integer(512),
        ValueRule::Integer {
            min: 64,
            max: 65536
        },
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): active-token target after eviction."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.beta",
        "cask_beta",
        Memory,
        ModelLoad,
        DefaultValue::Integer(128),
        ValueRule::Integer { min: 0, max: 65536 },
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): eviction hysteresis."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.handoff_tokens",
        "cask_handoff_tokens",
        Memory,
        ModelLoad,
        DefaultValue::Integer(0),
        ValueRule::Integer { min: 0, max: 1048576 },
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): one-way kv_adaptive to plain TriAttention handoff position; zero disables it."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.core_fraction",
        "cask_core_frac",
        Memory,
        ModelLoad,
        DefaultValue::Float(0.5),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: true
        },
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): fraction of the budget retained as core."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.fold",
        "cask_fold_m",
        Memory,
        ModelLoad,
        DefaultValue::Integer(2),
        ValueRule::Integer { min: 1, max: 16 },
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): merge factor."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CASK/TriAttention KV eviction is unsupported research; use compact kv_cache modes
    field!(
        "memory.cask.auto_attach",
        "cask_auto_attach",
        Memory,
        ModelLoad,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        true,
        true,
        None,
        "Deprecated CASK (removal in 0.5.0): discover a matching TriAttention sidecar when explicitly enabled."
    ),
    field!(
        "prompt.normalize",
        "prompt_normalize",
        Prompt,
        Request,
        DefaultValue::Bool(true),
        ValueRule::Bool,
        true,
        false,
        Some("HIPFIRE_NORMALIZE_PROMPT"),
        "Collapse runs of three or more newlines before tokenization."
    ),
    field!(
        "prompt.system",
        "system_prompt",
        Prompt,
        Request,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        false,
        None,
        "Default system prompt used only when a request supplies no system or developer message."
    ),
    field!(
        "memory.mmq.screen",
        "mmq_screen",
        Memory,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(AUTO_ON_OFF),
        true,
        false,
        Some("HIPFIRE_MMQ_SCREEN"),
        "Per-weight MMQ correctness screening policy."
    ),
    field!(
        "memory.mmq.screen_threshold",
        "mmq_screen_threshold",
        Memory,
        ModelLoad,
        DefaultValue::Float(0.1),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: false
        },
        true,
        false,
        Some("HIPFIRE_MMQ_SCREEN_THRESHOLD"),
        "MMQ screening absolute-error threshold."
    ),
    process_field!(
        "memory.prompt_cache_capacity",
        "prompt_cache_capacity",
        Memory,
        DefaultValue::Integer(32),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        false,
        "HIPFIRE_PROMPT_CACHE_CAP",
        "Maximum cached assistant-turn tokenizations; zero keeps no entries."
    ),
    process_field!(
        "memory.session_cache_bytes",
        "session_cache_bytes",
        Memory,
        DefaultValue::Integer(8589934592),
        ValueRule::Integer {
            min: 0,
            max: 1099511627776
        },
        false,
        "HIPFIRE_SESSION_CACHE_BYTES",
        "Byte budget for reusable prefill snapshots (session cache; Flash-Next). 0 disables the cache."
    ),
    process_field!(
        "memory.ple_hot_store_bytes",
        "ple_hot_store_bytes",
        Memory,
        DefaultValue::Integer(0),
        ValueRule::AutoInteger {
            min: 0,
            max: 1099511627776
        },
        false,
        "HIPFIRE_PLE_HOT_STORE_BYTES",
        "Prewarm budget of the Qwen4 PLE hot-row store (bytes of the frequency-ranked `<model>.plehotrows` hot-row pack read into the page cache after load; rows the pack holds are read from it, everything else from the model file; unified-memory GPUs only, needs the `<model>.plehot` sidecar and the pack from `hipfire-quantize ple-hot-pack`). Default 0 (off). `auto` prewarms MemAvailable minus the CMA region (the larger of CmaFree and the CMA pageblocks: CmaTotal or /proc/zoneinfo cma; only movable allocations can use them) minus 4 GiB at ready, capped at the pack. The pack's pages are ordinary reclaimable page cache: they are dropped, never swapped, under memory pressure."
    ),
    process_field!(
        "memory.ple_format",
        "ple_format",
        Memory,
        DefaultValue::String("auto"),
        ValueRule::Enum(PLE_FORMATS),
        false,
        "HIPFIRE_PLE_FORMAT",
        "Which PLE row table a Flash-Next (Qwen4) load reads: `q8` = the exact Q8 rows (embedded in a monolithic model, or the `<model>.ple-q8` sidecar of a split base), `q4` = the 4-bit Lloyd-Max rows (`<model>.ple-q4`, created ahead of time with `hipfire ple derive --format q4 <model>` or `hipfire pull`; a load never derives it and refuses when it is missing), `auto` = q4 on gfx1151 (unified-memory Strix Halo) and q8 elsewhere, using the other format when only that one is on disk (auto never derives or downloads)."
    ),
    process_field!(
        "memory.ple_prewarm_bytes",
        "ple_prewarm_bytes",
        Memory,
        DefaultValue::String("auto"),
        ValueRule::AutoInteger {
            min: 0,
            max: 1099511627776
        },
        false,
        "HIPFIRE_PLE_PREWARM_BYTES",
        "Prewarm budget of the q4 PLE table (bytes of `<model>.ple-q4` read into the page cache after the model is ready, while no request runs; a rank-ordered table hottest tier first, then tier 0 again). Default `auto` = MemAvailable minus the CMA region minus 8 GiB, sampled once after the pinned experts and the model are loaded, capped at the table's data bytes; on a unified-memory GPU and on a discrete GPU with every expert resident in VRAM it is also capped at 15 GiB (the smaller of the two), and only a discrete GPU with host-placed experts takes the uncapped figure; an unreadable MemAvailable gives 0. `0` turns the prewarm off; an explicit N is that many bytes, is never capped at 15 GiB, and overrides the derivation (the table still caps it). The pages are ordinary reclaimable page cache: dropped, never swapped, under memory pressure, and not charged in admission."
    ),
    process_field!(
        "memory.ple_load_fill_bytes",
        "ple_load_fill_bytes",
        Memory,
        DefaultValue::String("auto"),
        ValueRule::AutoInteger {
            min: 0,
            max: 1099511627776
        },
        false,
        "HIPFIRE_PLE_LOAD_FILL_BYTES",
        "Bytes of a rank-ordered q4 PLE table (`<model>.ple-q4`) read into the page cache while the model loads, hottest rows first, at no more than 0.2 GB/s so the weight upload keeps the drive, re-reading pages the upload's cache traffic evicted (hottest first) until ready; the prewarm after ready (`memory.ple_prewarm_bytes`) continues at the end of the resident prefix. Ready never waits for it: what is resident by then serves the first request. Default `auto` = 14 GiB, capped at the table, at an explicit `memory.ple_prewarm_bytes`, and at the host RAM available at load start (MemAvailable minus the CMA region, plus TTM's page pool) minus the weights the load puts in host RAM minus 8 GiB. `0` turns it off; an explicit N is that many bytes (same caps). Ordinary reclaimable page cache, not charged in admission."
    ),
    process_field!(
        "memory.session_cache_disk_bytes",
        "session_cache_disk_bytes",
        Memory,
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: 70368744177664
        },
        false,
        "HIPFIRE_SESSION_CACHE_DISK_BYTES",
        "Byte budget of the session cache's disk tier: snapshots evicted from memory are written to memory.session_cache_dir and restored on demand, surviving restarts. 0 disables it."
    ),
    process_field!(
        "memory.session_cache_dir",
        "session_cache_dir",
        Memory,
        DefaultValue::String(""),
        ValueRule::String,
        false,
        "HIPFIRE_SESSION_CACHE_DIR",
        "Directory of the session cache's disk tier; empty = $HIPFIRE_HOME/session-cache. One process uses it at a time."
    ),
    process_bool_field!(
        "memory.prompt_cache_unbounded",
        "prompt_cache_unbounded",
        Memory,
        false,
        true,
        "HIPFIRE_PROMPT_CACHE_UNBOUNDED",
        "Remove the assistant-turn cache capacity bound."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.mode",
        "prefill_compression",
        Speculation,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::Enum(&["off", "auto", "always"]),
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): speculative-prefill policy."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.threshold",
        "prefill_threshold",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(32768),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): auto-mode token threshold."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.keep_ratio",
        "prefill_keep_ratio",
        Speculation,
        ModelLoad,
        DefaultValue::Float(0.05),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: false
        },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): retained-token ratio."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.alpha",
        "prefill_alpha",
        Speculation,
        ModelLoad,
        DefaultValue::Float(0.85),
        ValueRule::Float {
            min: 0.0,
            max: 1.0,
            min_inclusive: true
        },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): block-selection strictness."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.min_keep",
        "prefill_min_keep",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(2048),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): retained-token floor."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.sink",
        "prefill_sink",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(256),
        ValueRule::Integer { min: 0, max: 65536 },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): always-retained prompt prefix."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.recent",
        "prefill_recent",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(1024),
        ValueRule::Integer { min: 0, max: 65536 },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): always-retained prompt tail."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.block",
        "prefill_block",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(128),
        ValueRule::Integer { min: 1, max: 4096 },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): scoring block size."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.drafter",
        "prefill_drafter",
        Speculation,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::String,
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): drafter path."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.drafter_device",
        "prefill_drafter_device",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(-1),
        ValueRule::Integer { min: -1, max: 15 },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): drafter device; -1 uses the target device."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.profile",
        "prefill_profile",
        Speculation,
        Diagnostic,
        DefaultValue::Bool(false),
        ValueRule::Bool,
        false,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): emit PFlash stage timings."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.sparse_threshold",
        "prefill_sparse_threshold",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(32768),
        ValueRule::Integer {
            min: 0,
            max: 1048576
        },
        true,
        true,
        None,
        "Deprecated PFlash (removal in 0.5.0): sparse-attention threshold."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    field!(
        "speculation.prefill.drafter_kv",
        "prefill_drafter_kv",
        Speculation,
        ModelLoad,
        DefaultValue::String("q8"),
        ValueRule::Enum(&["q8", "fwht4", "fwht3", "fwht2"]),
        true,
        true,
        Some("HIPFIRE_PFLASH_DRAFTER_KV"),
        "Deprecated PFlash (removal in 0.5.0): KV quantization used by the PFlash drafter scorer."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — PFlash is unsupported research; prefix caching supersedes it
    diagnostic_field!(
        "diagnostic.pflash.score_layer",
        "pflash_score_layer",
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 0, max: 65535 },
        "HIPFIRE_PFLASH_SCORE_LAYER",
        "Deprecated PFlash (removal in 0.5.0): override the PFlash scoring layer; null uses model policy."
    ),
    field!(
        "speculation.mtp",
        "mtp_mode",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(AUTO_ON_OFF),
        true,
        false,
        Some("HIPFIRE_MTP_MODE"),
        "MTP eligibility policy. `auto` (default) arms the native MTP head only for a producer that declares a measured autoregressive floor (native Qwen4 MTP, Flash-Next); every other producer (Qwen3.5/3.6, DeepSeek V4) keeps plain AR and does not load the head. `on` arms the head wherever one exists (also over a host-mapped expert spill); `off` never arms it."
    ),
    field!(
        "speculation.mtp_k",
        "mtp_k",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::AutoInteger { min: 1, max: 63 },
        true,
        false,
        Some("HIPFIRE_MTP_K"),
        "MTP draft window. Default `auto` = one below the qualified row-exact verify rows for native Qwen4 MTP (Flash-Next, arch 16), narrowed to the CPU-owned expert window after placement; else 3. Explicit 1-63 overrides the default, not executor/qualified-row/CPU constraints. Other producers retain their own limits."
    ),
    field!(
        "speculation.mtp_ngram",
        "mtp_ngram",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "on", "off", "1", "0"]),
        true,
        false,
        Some("HIPFIRE_MTP_NGRAM"),
        "N-gram takeover windows inside native MTP (greedy and sampled, thinking on or off). auto (default) arms it for native Qwen4 MTP (arch 16) on gfx1151 and gfx1201 and leaves other producers off; on/1 arms every MTP producer that supports it; off/0 disables it."
    ),
    field!(
        "speculation.mtp_ngram_copy",
        "mtp_ngram_copy",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "on", "off", "1", "0"]),
        true,
        false,
        Some("HIPFIRE_MTP_NGRAM_COPY"),
        "Native Qwen4 MTP n-gram copy lane (verbatim-copy matcher, adaptive take of up to the row-exact verify rows minus one, row-0 confirmation only until the copy source is established, kept on floor-retired requests). on/1 arms it wherever the native Qwen4 n-gram source is armed (triple 5/1/63); off/0 keeps the 5/3/3 pool with row-0 confirmation on every takeover (kill switch); auto (default) arms it for native Qwen4 MTP (arch 16) on gfx1151 and gfx1201 and leaves every other GPU off."
    ),
    field!(
        "speculation.mtp_sampled",
        "mtp_sampled",
        Speculation,
        ModelLoad,
        DefaultValue::Bool(true),
        ValueRule::Bool,
        false,
        false,
        Some("HIPFIRE_MTP_SAMPLED"),
        "Qwen4 (Flash-Next) native MTP for sampled (temperature > 0) requests: lossless speculative rejection sampling against the AR sampler's distribution. On by default; off routes sampled requests to AR."
    ),
    field!(
        "speculation.mode",
        "speculation",
        Speculation,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(SPECULATION_MODES),
        true,
        false,
        Some("HIPFIRE_SPECULATION"),
        "Canonical speculative-decoding selector."
    ),
    field!(
        "speculation.dspark_confidence",
        "dspark_conf_threshold",
        Speculation,
        ModelLoad,
        DefaultValue::Null,
        ValueRule::NullableFloat { min: 0.0, max: 1.0 },
        true,
        true,
        None,
        "DSpark confidence truncation threshold."
    ),
    field!(
        "speculation.ngram",
        "ngram_mode",
        Speculation,
        ModelLoad,
        DefaultValue::String("off"),
        ValueRule::Enum(&["auto", "on", "off", "1", "0", "true", "false"]),
        true,
        false,
        Some("HIPFIRE_NGRAM_DRAFT"),
        "Model-free n-gram speculation policy: on/1/true arms it, off/0/false (default) disables it, auto arms it wherever the loader has an n-gram speculator for the arch; any other value is a config error."
    ),
    field!(
        "speculation.ngram_k",
        "ngram_k",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(12),
        ValueRule::Integer { min: 2, max: 32 },
        true,
        false,
        Some("HIPFIRE_NGRAM_DRAFT_K"),
        "N-gram draft window."
    ),
    field!(
        "speculation.ngram_min_count",
        "ngram_min_count",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(2),
        ValueRule::Integer { min: 1, max: 10 },
        true,
        false,
        Some("HIPFIRE_NGRAM_MIN_COUNT"),
        "Minimum n-gram match count."
    ),
    field!(
        "speculation.ddtree_budget",
        "ddtree_budget",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(0),
        ValueRule::Integer { min: 0, max: 64 },
        true,
        true,
        Some("HIPFIRE_DDTREE_BUDGET"),
        "DFlash verify-tree node budget; zero selects chain mode."
    ),
    field!(
        "speculation.ddtree_topk",
        "ddtree_topk",
        Speculation,
        ModelLoad,
        DefaultValue::Integer(4),
        ValueRule::Integer { min: 1, max: 8 },
        true,
        true,
        Some("HIPFIRE_DDTREE_TOPK"),
        "DFlash verify-tree fanout."
    ),
    field!(
        "prompt.chat_template",
        "chat_template",
        Prompt,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::PathOrEmpty,
        true,
        false,
        Some("HIPFIRE_CHAT_TEMPLATE_FILE"),
        "Optional Jinja chat-template path."
    ),
    field!(
        "prompt.default_chatml",
        "default_chatml",
        Prompt,
        ModelLoad,
        DefaultValue::Bool(true),
        ValueRule::Bool,
        true,
        false,
        Some("HIPFIRE_DEFAULT_CHATML"),
        "Allow the fallback ChatML frame when no template resolves."
    ),
    process_bool_field!(
        "prompt.jinja_chat",
        "jinja_chat",
        Prompt,
        true,
        false,
        "HIPFIRE_JINJA_CHAT",
        "Render chat prompts through the model Jinja template; set HIPFIRE_JINJA_CHAT=0 for the hand-rolled ChatScaffold path."
    ),
    process_field!(
        "hardware.devices",
        "devices",
        Hardware,
        DefaultValue::Null,
        ValueRule::NullableString,
        false,
        "HIPFIRE_DEVICES",
        "Device list for this process, in logical order: BDF-order index, gfxNNNN (first free card of that arch), GPU-<uuid>, or PCI address; resolved from the KFD topology and lowered to ROCr selectors plus HIP logical 0..N-1 before GPU initialization."
    ),
    field!(
        "topology.ep",
        "ep",
        Hardware,
        ModelLoad,
        DefaultValue::Integer(1),
        ValueRule::Integer { min: 1, max: 8 },
        false,
        true,
        Some("HIPFIRE_TOPOLOGY_EP"),
        "Expert-parallel degree for Qwen4 (Flash-Next) on gfx1100 or gfx1201: 1 (default, single device) or 2..=8. Experts shard e % N across N visible homogeneous gfx1100 or gfx1201 cards that can access each other as peers; every card runs the unchanged single-device program. Opt-in; HIPFIRE_QWEN4_EP=0 forces 1."
    ),
    field!(
        "topology.pp",
        "pp",
        Hardware,
        ModelLoad,
        DefaultValue::Integer(1),
        ValueRule::Integer { min: 1, max: 8 },
        false,
        true,
        Some("HIPFIRE_TOPOLOGY_PP"),
        "Pipeline-parallel degree: 1 (default, single device) or 2..=8. Layers split into N contiguous stages across the first N visible devices (hardware.devices / HIPFIRE_DEVICES). An explicit load request with pp > 1 overrides this value; a request of 1 or no request uses it. Cannot combine with tensor parallelism (tp) or Qwen4 expert parallelism."
    ),
    process_field!(
        "hardware.uniform_vram_tolerance_gb",
        "uniform_vram_tolerance_gb",
        Hardware,
        DefaultValue::Null,
        ValueRule::NullableFloat {
            min: f64::MIN,
            max: f64::MAX
        },
        false,
        "HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB",
        "Allowed free-VRAM spread across a uniform multi-GPU topology."
    ),
    process_field!(
        "generation.loop_guard_threshold",
        "ngram_loop_threshold",
        Generation,
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        false,
        "HIPFIRE_NGRAM_LOOP_THRESHOLD",
        "Repeated 4-gram count that forces EOS; zero disables the guard."
    ),
    process_field!(
        "generation.loop_guard_window",
        "ngram_window",
        Generation,
        DefaultValue::Integer(256),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        false,
        "HIPFIRE_NGRAM_WINDOW",
        "Token window inspected by the repeated 4-gram loop guard."
    ),
    process_field!(
        "kernel.flash_partials_batch",
        "flash_partials_batch",
        Kernel,
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 0, max: 65536 },
        true,
        "HIPFIRE_FLASH_PARTIALS_BATCH",
        "Override the prefill flash-attention partial-scratch batch multiplier."
    ),
    process_field!(
        "kernel.lm_head_f16",
        "lm_head_f16",
        Kernel,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "native", "f16", "1", "f32", "fp32", "legacy", "0"]),
        false,
        "HIPFIRE_LM_HEAD_F16",
        "Storage policy for native FP16 LM-head weights."
    ),
    diagnostic_field!(
        "diagnostic.prompt_heat_limit",
        "prompt_heat_limit",
        DefaultValue::Integer(64),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_PROMPT_HEAT_LIMIT",
        "Maximum rows emitted by prompt token-heat diagnostics."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gemv_rows",
        "gemv_rows",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "4", "8"]),
        "HIPFIRE_GEMV_ROWS",
        "Override the architecture-selected GEMV rows per workgroup."
    ),
    diagnostic_field!(
        "diagnostic.kernel.fp16_layer_min",
        "fp16_layer_min",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_FP16_LAYER_MIN",
        "First layer included in the FP16 route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.fp16_layer_max",
        "fp16_layer_max",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_FP16_LAYER_MAX",
        "Last layer included in the FP16 route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.hfq3_mmq_layer_min",
        "hfq3_mmq_layer_min",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_HFQ3_MMQ_LAYER_MIN",
        "First layer included in the HFQ3 MMQ route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.hfq3_mmq_layer_max",
        "hfq3_mmq_layer_max",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_HFQ3_MMQ_LAYER_MAX",
        "Last layer included in the HFQ3 MMQ route override."
    ),
    diagnostic_field!(
        "diagnostic.kernel.mmq_min_batch",
        "mmq_min_batch",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_MMQ_MIN_BATCH",
        "Minimum batch size for MMQ dispatch."
    ),
    diagnostic_field!(
        "diagnostic.kernel.rocblas_min_batch",
        "rocblas_min_batch",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_ROCBLAS_MIN_BATCH",
        "Minimum batch size for rocBLAS dispatch."
    ),
    diagnostic_field!(
        "diagnostic.kernel.ddtree_logw_cutoff",
        "ddtree_logw_cutoff",
        DefaultValue::Null,
        ValueRule::NullableFloat {
            min: 0.0,
            max: f64::MAX
        },
        "HIPFIRE_DDTREE_LOGW_CUTOFF",
        "Positive DDTree cumulative-log-weight expansion cutoff."
    ),
    diagnostic_field!(
        "diagnostic.kernel.lloyd_mb4",
        "lloyd_mb4",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "4"]),
        "HIPFIRE_LLOYD_MB4",
        "Lloyd kernel rows packed per MB4 work item."
    ),
    diagnostic_field!(
        "diagnostic.kernel.mq3_mb4",
        "mq3_mb4",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "4"]),
        "HIPFIRE_MQ3_MB4",
        "MQ3 kernel rows packed per MB4 work item."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gate_up_variant",
        "gate_up_variant",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["ldsx", "k4", "ldscoop", "2tile"]),
        "HIPFIRE_GATE_UP_VARIANT",
        "Select a gate/up WMMA experiment variant."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gfx11_weight_load_policy",
        "gfx11_weight_load_policy",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["buffer", "global", "flat-buffer"]),
        "HIPFIRE_GFX11_WEIGHT_LOAD_POLICY",
        "Select the gfx11 compiler weight-load policy."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gfx12_weight_load_policy",
        "gfx12_weight_load_policy",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["rt", "global", "ht", "nt-rt", "nt-ht"]),
        "HIPFIRE_GFX12_WEIGHT_LOAD_POLICY",
        "Select the gfx12 compiler weight-load policy."
    ),
    diagnostic_field!(
        "diagnostic.kernel.gfx942_mfma_prefill",
        "gfx942_mfma_prefill",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["1", "2", "3", "4"]),
        "HIPFIRE_GFX942_MFMA_PREFILL",
        "Select the gfx942 direct-MFMA prefill experiment."
    ),
    diagnostic_field!(
        "diagnostic.kernel.rdna2_variant",
        "rdna2_variant",
        DefaultValue::Null,
        ValueRule::NullableInteger { min: 1, max: 5 },
        "HIPFIRE_RDNA2_VARIANT",
        "Select the gfx1030/gfx1031 HFQ4-G256 GEMV variant."
    ),
    diagnostic_field!(
        "diagnostic.kernel.wo_wmma_variant",
        "wo_wmma_variant",
        DefaultValue::Null,
        ValueRule::NullableEnum(&["ksplit", "ksplit_det", "k2", "k2x32", "k4", "wmma", "wmma2"]),
        "HIPFIRE_WO_WMMA_VARIANT",
        "Select a weight-only residual WMMA variant."
    ),
    diagnostic_field!(
        "diagnostic.compiler.hipcc_extra_flags",
        "hipcc_extra_flags",
        DefaultValue::String(""),
        ValueRule::String,
        "HIPFIRE_HIPCC_EXTRA_FLAGS",
        "Append advanced local flags to HIP kernel compilation."
    ),
    process_bool_field!(
        "hardware.allow_mixed_arch",
        "allow_mixed_arch",
        Hardware,
        false,
        false,
        "HIPFIRE_ALLOW_MIXED_ARCH",
        "Allow a multi-GPU topology containing different GPU architectures."
    ),
    process_bool_field!(
        "hardware.tp_use_rccl",
        "tp_use_rccl",
        Hardware,
        true,
        false,
        "HIPFIRE_TP_USE_RCCL",
        "Use RCCL for tensor-parallel all-reduce."
    ),
    process_bool_field!(
        "kernel.prefill_batched",
        "prefill_batched",
        Kernel,
        true,
        false,
        "HIPFIRE_PREFILL_BATCHED",
        "Use batched prefill kernels when eligible."
    ),
    process_field!(
        "prefill.chunk_rows",
        "prefill_chunk_rows",
        Kernel,
        DefaultValue::Integer(512),
        ValueRule::Integer { min: 2, max: 1048576 },
        false,
        "HIPFIRE_PREFILL_CHUNK_ROWS",
        "Widened ordinary-prefill chunk ceiling in rows (arch default 8192 on exact gfx1100/gfx1151/gfx1201, 512 elsewhere; HIPFIRE_PREFILL_MAX_BATCH overrides; per-device VRAM admission may select a smaller rung). Resident Qwen3.5/3.6 MoE models (fully resident, single device, grouped MQ4 producers) take one widened rung instead: unset requests 4096 rows, admitted only when the projected prefill batch scratch plus grouped/projection workspace plus the 1 GiB margin fit the free VRAM, else the incumbent 512 (arch default) ceiling with no intermediate rung; an explicit value below 4096 or HIPFIRE_PREFILL_MAX_BATCH keeps the incumbent, one at or above 4096 requests the admitted rung (the actual admitted chunk is logged as `prefill_chunk: requested=… admitted=…`). Dense Qwen3.5/3.6 keeps the 8192 ladder above. Qwen4 (Flash-Next) rounds it down to 256 rows; its auto expert placement is sized from the same chunk: the explicit value, else 8192 on exact gfx1151, 4096 on exact gfx1201, 1536 elsewhere; exact gfx1100 on a measured narrow discrete host link (below 16 GB/s, probed before placement is sized) defaults to 8192, ceding resident expert layers for the wider chunk (an explicit value never probes and is authoritative for placement; 1536 opts out)."
    ),
    process_bool_field!(
        "speculation.draft_f16",
        "draft_f16",
        Speculation,
        true,
        false,
        "HIPFIRE_DRAFT_F16",
        "Keep DFlash draft activations in FP16."
    ),
    diagnostic_bool_field!(
        "diagnostic.prompt_token_heat",
        "prompt_token_heat",
        false,
        "HIPFIRE_PROMPT_TOKEN_HEAT",
        "Emit prompt token-heat diagnostics."
    ),
    diagnostic_bool_field!(
        "diagnostic.prompt_heat_json",
        "prompt_heat_json",
        false,
        "HIPFIRE_PROMPT_HEAT_JSON",
        "Render prompt token-heat diagnostics as JSON."
    ),
    diagnostic_bool_field!(
        "diagnostic.draft_gemm_dump",
        "draft_gemm_dump",
        false,
        "HIPFIRE_DRAFT_GEMM_DUMP",
        "Dump DFlash draft GEMM diagnostics."
    ),
    diagnostic_bool_field!(
        "diagnostic.draft_subphase",
        "draft_subphase",
        false,
        "HIPFIRE_DRAFT_SUBPHASE",
        "Emit DFlash draft subphase timings."
    ),
    process_auto_bool_field!(
        "kernel.gemv_dp4a",
        "gemv_dp4a",
        Kernel,
        false,
        "HIPFIRE_GEMV_DP4A",
        "Override the architecture-selected DP4A GEMV route."
    ),
    process_auto_bool_field!(
        "kernel.gemv_prefetch",
        "gemv_prefetch",
        Kernel,
        false,
        "HIPFIRE_GEMV_PREFETCH",
        "Override the architecture-selected GEMV prefetch route."
    ),
    process_auto_bool_field!(
        "kernel.gfx942_lds_gemv",
        "gfx942_lds_gemv",
        Kernel,
        true,
        "HIPFIRE_GFX942_LDS_GEMV",
        "Override the gfx942 LDS GEMV experiment."
    ),
    process_auto_bool_field!(
        "kernel.hfq3_dp4a",
        "hfq3_dp4a",
        Kernel,
        true,
        "HIPFIRE_HFQ3_DP4A",
        "Override HFQ3 DP4A dispatch."
    ),
    process_auto_bool_field!(
        "kernel.hfq3_mmq",
        "hfq3_mmq",
        Kernel,
        true,
        "HIPFIRE_HFQ3_MMQ",
        "Override HFQ3 MMQ dispatch."
    ),
    process_auto_bool_field!(
        "kernel.hfq4_mmq_rdna2",
        "hfq4_mmq_rdna2",
        Kernel,
        true,
        "HIPFIRE_HFQ4_MMQ_RDNA2",
        "Override HFQ4 MMQ dispatch on RDNA2."
    ),
    process_auto_bool_field!(
        "kernel.gcn5_wave64_hybrid",
        "gcn5_wave64_hybrid",
        Kernel,
        true,
        "HIPFIRE_GCN5_WAVE64_HYBRID",
        "Override the GCN5 wave64 hybrid route."
    ),
    process_auto_bool_field!(
        "kernel.mmq",
        "mmq",
        Kernel,
        false,
        "HIPFIRE_MMQ",
        "Override architecture-selected MMQ dispatch."
    ),
    process_auto_bool_field!(
        "kernel.gfx942_gemv_v2",
        "gfx942_gemv_v2",
        Kernel,
        true,
        "HIPFIRE_GFX942_GEMV_V2",
        "Override gfx942 GEMV v2 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.moe_grouped_i8",
        "moe_grouped_i8",
        Kernel,
        true,
        "HIPFIRE_MOE_GROUPED_I8",
        "Override grouped MoE i8 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.moe_paro_i8",
        "moe_paro_i8",
        Kernel,
        true,
        "HIPFIRE_MOE_PARO_I8",
        "Override architecture-selected Paro grouped-GEMM i8 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.moe_paro_i8_k8",
        "moe_paro_i8_k8",
        Kernel,
        true,
        "HIPFIRE_MOE_PARO_I8_K8",
        "Override architecture-selected Paro grouped-GEMM i8 K8 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_hfq4_qkvza_k2048",
        "rdna3_hfq4_qkvza_k2048",
        Kernel,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_K2048",
        "Override the certified gfx1100 fixed-K QKVZA kernel."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_hfq4_residual_stage_x32",
        "rdna3_hfq4_residual_stage_x32",
        Kernel,
        true,
        "HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32",
        "Override the certified gfx1100 residual activation-staging kernel."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_hfq4_sigmoid_buffer",
        "rdna3_hfq4_sigmoid_buffer",
        Kernel,
        true,
        "HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER",
        "Override the certified gfx1100 sigmoid buffer-load kernel."
    ),
    process_auto_bool_field!(
        "kernel.rdna3_rmsnorm_vecsum",
        "rdna3_rmsnorm_vecsum",
        Kernel,
        true,
        "HIPFIRE_RDNA3_RMSNORM_VECSUM",
        "Override the certified gfx1100 RMSNorm vecsum kernel."
    ),
    process_bool_field!(
        "kernel.fp8_wmma",
        "fp8_wmma",
        Kernel,
        false,
        true,
        "HIPFIRE_FP8_WMMA",
        "Enable the experimental FP8 WMMA route."
    ),
    process_bool_field!(
        "kernel.mq4v2_gateup_k5120",
        "mq4v2_gateup_k5120",
        Kernel,
        true,
        false,
        "HIPFIRE_MQ4V2_GATEUP_K5120",
        "Run exact gfx1100 MQ4V2 K5120 gate/up decode (17408 rows per output) on its fixed-group-count kernel (default on; byte-identical to the generic kernel; set to false or HIPFIRE_MQ4V2_GATEUP_K5120=0 to restore the generic route)."
    ),
    process_bool_field!(
        "kernel.gfx12_mq4v2_gateup_k5120",
        "gfx12_mq4v2_gateup_k5120",
        Kernel,
        false,
        true,
        "HIPFIRE_GFX12_MQ4V2_GATEUP_K5120",
        "Enable experimental exact gfx1201 MQ4V2 K5120 gate/up decode (17408 rows per output); preserves the existing DEV/RT weight-load policy."
    ),
    process_bool_field!(
        "kernel.gfx12_mq4v2_fp8_gateup",
        "gfx12_mq4v2_fp8_gateup",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_MQ4V2_FP8_GATEUP",
        "Enable the gfx1201 MQ4v2 FP8 WMMA gate/up prefill route (default on exact gfx1201; set to false or HIPFIRE_GFX12_MQ4V2_FP8_GATEUP=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.gfx12_mq4v2_fp8_resid",
        "gfx12_mq4v2_fp8_resid",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_MQ4V2_FP8_RESID",
        "Enable the gfx1201 MQ4v2 FP8 WMMA residual prefill route (default on exact gfx1201; set to false or HIPFIRE_GFX12_MQ4V2_FP8_RESID=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.gfx12_mq4v2_fp8_qkvza",
        "gfx12_mq4v2_fp8_qkvza",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_MQ4V2_FP8_QKVZA",
        "Enable the gfx1201 MQ4v2 FP8 WMMA 4-way QKVZA prefill route (default on exact gfx1201; set to false or HIPFIRE_GFX12_MQ4V2_FP8_QKVZA=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.gfx12_mq4v2_fp8_qkv",
        "gfx12_mq4v2_fp8_qkv",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_MQ4V2_FP8_QKV",
        "Enable the gfx1201 MQ4v2 FP8 WMMA 3-way QKV prefill route (default on exact gfx1201; set to false or HIPFIRE_GFX12_MQ4V2_FP8_QKV=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.gfx12_mq4v2_fp8_v2",
        "gfx12_mq4v2_fp8_v2",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_MQ4V2_FP8_V2",
        "Enable the gfx1201 MQ4v2 FP8 WMMA staged-tile v2 prefill route (default on exact gfx1201; set to false or HIPFIRE_GFX12_MQ4V2_FP8_V2=0 to opt out; selects the _v2_gfx1201 symbols at N>=256 with 128x128 geometry and 2 slabs unless HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM/_SLABS override, the four family flags remain prerequisites)."
    ),
    process_bool_field!(
        "kernel.gfx12_gdn_pre_fused",
        "gfx12_gdn_pre_fused",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_GDN_PRE_FUSED",
        "Enable the gfx1201 batched prefill GDN preamble fusion (default on exact gfx1201; set to false or HIPFIRE_GFX12_GDN_PRE_FUSED=0 to opt out; fuses sigmoid+conv+qknorm into gdn_pre_batched_gfx1201, byte-exact)."
    ),
    process_bool_field!(
        "kernel.gfx12_gdn_chunk_scan",
        "gfx12_gdn_chunk_scan",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_GDN_CHUNK_SCAN",
        "Enable the fused chunked gated-delta-net prefill scan on gfx1100/gfx1151/gfx1201 (default on; set to false or HIPFIRE_GFX12_GDN_CHUNK_SCAN=0 to opt out; exact-shape ordinary prefill only)."
    ),
    process_bool_field!(
        "kernel.gfx12_silu_quant_fused",
        "gfx12_silu_quant_fused",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_SILU_QUANT_FUSED",
        "Fuse the int4 activation quantiser into fused_silu_mul_mq_rotate_awq on gfx1201 (default on exact gfx1201; set to false or HIPFIRE_GFX12_SILU_QUANT_FUSED=0 to opt out; emits block_i4_128 from the down-proj SwiGLU/FWHT producer so the standalone quantize_int4_mmq_ds128 launch disappears, bit-identical)."
    ),
    process_bool_field!(
        "kernel.gfx12_producer_quant_fused",
        "gfx12_producer_quant_fused",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_PRODUCER_QUANT_FUSED",
        "Fuse the activation quantiser into the gfx1201 producers (default on exact gfx1201; set to false or HIPFIRE_GFX12_PRODUCER_QUANT_FUSED=0 to opt out). IU4: emits block_i4_128 from the RMSNorm/rotate/gated-norm _gfx12 producer twins so the standalone quantize_int4_mmq_ds128 launch disappears at each admitted site. A8 (grouped MoE / gated-FFN down, W4A8 tier): one launch forms the SwiGLU/SiTU activation, casts, pads, divides by the AWQ scale, rotates and emits block_i8_128, replacing the activation + mq_rotate_x + quantize_int8_mmq_ds128 chain (the F32 activation/rotation planes are never written). Bit-identical."
    ),
    process_bool_field!(
        "kernel.g12_norm",
        "g12_norm",
        Kernel,
        true,
        false,
        "HIPFIRE_G12_NORM",
        "Launch the gfx1201 _v2 RMSNorm and gated-norm int4 producers (default on exact gfx1201; set to false or HIPFIRE_G12_NORM=0 to restore the incumbent _gfx12 symbols; batched sum-of-squares loads, one-reciprocal RTN codes and one wave per gated-norm group, bit-identical)."
    ),
    process_bool_field!(
        "kernel.g12_dec_norm",
        "g12_dec_norm",
        Kernel,
        true,
        false,
        "HIPFIRE_G12_DEC_NORM",
        "Launch the gfx1201 and gfx1151 decode norms as multi-workgroup grids (default on exact gfx1201 and exact gfx1151; set to false or HIPFIRE_G12_DEC_NORM=0 to restore the single-workgroup launches; f32 AWQ RMSNorm+FWHT K/256 workgroups, out-of-place rmsnorm_f32 n/256 workgroups per row, half-split partial RoPE one workgroup per head, bit-identical)."
    ),
    process_bool_field!(
        "kernel.gfx1100_dec_norm",
        "gfx1100_dec_norm",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX1100_DEC_NORM",
        "Launch the gfx1100 decode norms as multi-workgroup grids (default on exact gfx1100; set to false or HIPFIRE_GFX1100_DEC_NORM=0 to restore the single-workgroup launches; f32 AWQ RMSNorm+FWHT K/256 workgroups, out-of-place rmsnorm_f32 n/256 workgroups per row, bit-identical)."
    ),
    process_bool_field!(
        "kernel.g12_a4c2",
        "g12_a4c2",
        Kernel,
        true,
        false,
        "HIPFIRE_G12_A4C2",
        "Search two activation-quantization scales ({5,7}, as on gfx11) inside the gfx1201 int4 producers (default on exact gfx1201; set to false or HIPFIRE_G12_A4C2=0 to restore round-to-nearest d = amax/7; compiles the gfx1201 JIT with -DIU4_A4_CANDIDATES=2, whose one-pass producer-layout search is bit-identical to the generic candidate loop)."
    ),
    process_bool_field!(
        "kernel.gfx11_producer_quant_fused",
        "gfx11_producer_quant_fused",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX11_PRODUCER_QUANT_FUSED",
        "Fuse the int4 activation quantiser into the gfx11 sigmoid/gated-norm producers (default on gfx1100/gfx1151; set to false or HIPFIRE_GFX11_PRODUCER_QUANT_FUSED=0 to opt out; emits block_i4_128 from the _gfx11 producer twins so the standalone quantize_int4_mmq_ds128 launch disappears at each admitted site, bit-identical)."
    ),
    process_bool_field!(
        "kernel.gfx11_lean_pbs",
        "gfx11_lean_pbs",
        Kernel,
        true,
        true,
        "HIPFIRE_GFX11_LEAN_PBS",
        "Size fallback-only ordinary gfx11 MQ4V2 prefill scratch for at most 64 rows (default on for the admitted fused gfx1100/gfx1151 route); verify, non-fused, and other architectures retain full scratch. Set false or HIPFIRE_GFX11_LEAN_PBS=0 to opt out."
    ),
    process_field!(
        "kernel.gfx11_a4_candidates",
        "gfx11_a4_candidates",
        Kernel,
        DefaultValue::String("2"),
        ValueRule::Enum(&["1", "2", "4", "8"]),
        true,
        "HIPFIRE_GFX11_A4_CANDIDATES",
        "Number of activation-quantization candidates searched inside fused gfx1100/gfx1151 IU4 producers (default 2, selecting {5,7}); 8 restores the full-grid search. Other architectures and the standalone quantizer are unchanged."
    ),
    process_bool_field!(
        "kernel.gfx12_fp8_stream",
        "gfx12_fp8_stream",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_FP8_STREAM",
        "Fuse the MQ4v2 FP8 activation pre-pass into the gfx1201 RMSNorm/rotate producers (default on exact gfx1201; set to false or HIPFIRE_GFX12_FP8_STREAM=0 to opt out; emits byte-identical prepare_mq4v2_fp8_x_f32 outputs for the qkvza/gate_up/qkv inputs so the standalone pack_f32_to_fp8_mq4v2_gfx12 launch disappears at each admitted site)."
    ),
    process_bool_field!(
        "kernel.gfx12_fa2_prefill",
        "gfx12_fa2_prefill",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_FA2_PREFILL",
        "Enable the gfx1201 GQA-fused FA2 prefill attention route (default on exact gfx1201; set to false or HIPFIRE_GFX12_FA2_PREFILL=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.gfx12_fa_packet",
        "gfx12_fa_packet",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_FA_PACKET",
        "Enable the gfx1201 packet-minimal Q128 FA2 attention body (default on exact gfx1201; set to false or HIPFIRE_GFX12_FA_PACKET=0 to opt out to route-N; exact stage-b arithmetic)."
    ),
    process_bool_field!(
        "kernel.attn_qresident",
        "attn_qresident",
        Kernel,
        true,
        false,
        "HIPFIRE_ATTN_QRESIDENT",
        "Enable the gfx1201 register-resident-Q wide-workgroup FA2 prefill route (default on exact gfx1201; set to false or HIPFIRE_ATTN_QRESIDENT=0 to opt out; exact H24/KV4/D256 native-fp8-KV shapes only)."
    ),
    process_bool_field!(
        "kernel.attn_qresident_v2",
        "attn_qresident_v2",
        Kernel,
        true,
        false,
        "HIPFIRE_ATTN_QRESIDENT_V2",
        "Use the bit-exact v2 schedule of the gfx1201 Q-resident FA2 prefill kernel (default on exact gfx1201; set to false or HIPFIRE_ATTN_QRESIDENT_V2=0 to restore the v1 Q-resident kernel; only applies where kernel.attn_qresident selects the Q-resident route)."
    ),
    process_bool_field!(
        "kernel.verify_attn",
        "verify_attn",
        Kernel,
        true,
        false,
        "HIPFIRE_VERIFY_ATTN",
        "Run speculative-verify attention (1..=32 rows, non-tree) through VerifyAttn: on gfx1201 and gfx1100 the GQA-shared split-K twin of the batched flash tile + reduce (on gfx1100 also of the multi-row R4/R8 Q8 tile), on gfx1151 the context-parallel twin of the single-slot WMMA flash prefill (default on exact gfx1201, gfx1100 and gfx1151; byte-identical output; set to false or HIPFIRE_VERIFY_ATTN=0 to opt out to attention_flash_*_tile_batched / attention_flash_q8_0_rows{4,8}_d8 / attention_q8_0_flash_prefill_wmma)."
    ),
    process_bool_field!(
        "kernel.gfx1100_fa2_split_verify",
        "gfx1100_fa2_split_verify",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX1100_FA2_SPLIT_VERIFY",
        "Use the Q8 FA2 split-KV VerifyAttn route by default on exact gfx1100 inside the existing dense H24/KV4/HD256 long-context VMM-KV envelope (false or HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=0 restores the established VerifyAttn/R4-R8/batched route). The independent parent kernel.verify_attn=false / HIPFIRE_VERIFY_ATTN=0 also opts out. Other architectures remain off."
    ),
    process_bool_field!(
        "kernel.gfx12_fa_prep_fused",
        "gfx12_fa_prep_fused",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_FA_PREP_FUSED",
        "Fuse gfx1201 prefill Q/K norm and RoPE (default on exact gfx1201; false restores the separate launches)."
    ),
    process_bool_field!(
        "kernel.gfx12_fa_prep_fp8q",
        "gfx12_fa_prep_fp8q",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX12_FA_PREP_FP8Q",
        "Emit preconverted fp8 Q for gfx1201 Q-resident v2 attention (default on exact gfx1201; false retains F32 Q)."
    ),
    process_auto_bool_field!(
        "kernel.gfx11_q8_fa2_wide",
        "gfx11_q8_fa2_wide",
        Kernel,
        true,
        "HIPFIRE_GFX11_Q8_FA2_WIDE",
        "Whole-chunk Q8/Q8 FA2 prefill: auto enables on exact gfx1100 and gfx1151; other arches default off. Explicit false opts out. Requires kernel.gfx11_fa2_prefill, H24/KV4/D256, 64..8192 rows (above 512 aligned to 512), context 64..262144; explicit flash-off, CK and alternate variants retain precedence."
    ),
    process_bool_field!(
        "kernel.gfx11_fa2_prefill",
        "gfx11_fa2_prefill",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX11_FA2_PREFILL",
        "Enable the gfx11 GQA-fused FA2 prefill attention route (default on gfx1100/gfx1151; set to false or HIPFIRE_GFX11_FA2_PREFILL=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.iu4_prefill",
        "iu4_prefill",
        Kernel,
        true,
        false,
        "HIPFIRE_IU4_PREFILL",
        "Enable the W4A4 iu4-direct MMQ prefill route (default on exact gfx1100/gfx1151/gfx1201, other arches keep their incumbent route; set to false or HIPFIRE_IU4_PREFILL=0 to opt out)."
    ),
    process_auto_bool_field!(
        "kernel.pm_decode",
        "pm_decode",
        Kernel,
        true,
        "HIPFIRE_PM_DECODE",
        "Load the oracle-accepted builder (PeaceMaker) twins of the W1 decode projection modules instead of compiling them with hipcc: auto enables on exact gfx1201 only; false or HIPFIRE_PM_DECODE=0 restores the hipcc modules. Other architectures ignore it."
    ),
    process_bool_field!(
        "kernel.gfx11_iu4_gridspec",
        "gfx11_iu4_gridspec",
        Kernel,
        true,
        false,
        "HIPFIRE_GFX11_IU4_GRIDSPEC",
        "Split partial-N gfx11 IU4 GEMMs so full interior tiles use the unchecked specialization and only the final tile uses guarded loads/stores (default on; set to false or HIPFIRE_GFX11_IU4_GRIDSPEC=0 to opt out)."
    ),
    process_bool_field!(
        "kernel.gfx11_iu4_shape",
        "gfx11_iu4_shape",
        Kernel,
        true,
        true,
        "HIPFIRE_GFX11_IU4_SHAPE",
        "Use the 16-wave low-footprint IU4 full-tile entry on gfx1100 (default on); ignored on other architectures. Set false or HIPFIRE_GFX11_IU4_SHAPE=0 to restore the prior gfx1100 shape."
    ),
    process_bool_field!(
        "kernel.gfx11_iu4_symfold",
        "gfx11_iu4_symfold",
        Kernel,
        true,
        true,
        "HIPFIRE_IU4_SYMFOLD",
        "Exploit symmetric MQ4V2 weights on exact gfx1100/gfx1151 by rebiasing q to q-8 and using signed-weight IU4 WMMA (default on for symmetric artifacts only); set false or HIPFIRE_IU4_SYMFOLD=0 to opt out. Asymmetric artifacts and other architectures are unchanged."
    ),
    process_bool_field!(
        "kernel.gfx11_a8_prefill",
        "gfx11_a8_prefill",
        Kernel,
        false,
        true,
        "HIPFIRE_GFX11_A8_PREFILL",
        "Opt in to the W4A8 MQ4V2 prefill route on exact gfx1100/gfx1151 (symmetric MQ4V2, eager, batch >= 64, K % 256 == 0): int8 per-128 activations from the fused producers and an iu8-WMMA GEMM: about 43% lower KLD than IU4 W4A4 on Strix Halo, prompts about 1.5-1.6x slower. Default off on every architecture (IU4 W4A4 stays the default); other architectures ignore it."
    ),
    process_bool_field!(
        "kernel.npu_spillover",
        "npu_spillover",
        Kernel,
        false,
        true,
        "HIPFIRE_NPU_SPILLOVER",
        "Enable the opt-in gfx1151-only XDNA NPU spillover sidecar (requires a verified .xdna.zip registry sidecar; default off, never active on other architectures)."
    ),
    process_bool_field!(
        "kernel.dot2_gemv",
        "dot2_gemv",
        Kernel,
        false,
        true,
        "HIPFIRE_DOT2_GEMV",
        "Enable dot2 GEMV dispatch."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkv_wave64",
        "rdna3_hfq4_qkv_wave64",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKV_WAVE64",
        "Enable the RDNA3 wave64 QKV experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_2wave",
        "rdna3_hfq4_qkvza_2wave",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE",
        "Enable the RDNA3 two-wave QKVZA experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_wavepack4",
        "rdna3_hfq4_qkvza_wavepack4",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4",
        "Enable the RDNA3 four-wave QKVZA packing experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_ldsx8",
        "rdna3_hfq4_qkvza_ldsx8",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8",
        "Enable the RDNA3 LDS-staged QKVZA experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_reduce_chain",
        "rdna3_hfq4_qkvza_reduce_chain",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN",
        "Enable the RDNA3 explicit QKVZA reduction chain."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_qkvza_hoist_x32",
        "rdna3_hfq4_qkvza_hoist_x32",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32",
        "Enable the RDNA3 QKVZA activation-hoist experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_residual_k2048",
        "rdna3_hfq4_residual_k2048",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048",
        "Enable the RDNA3 fixed-K residual experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_sigmoid_tight_grid",
        "rdna3_hfq4_sigmoid_tight_grid",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID",
        "Enable the RDNA3 tight-grid sigmoid kernel."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_sigmoid_rows4",
        "rdna3_hfq4_sigmoid_rows4",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4",
        "Enable the RDNA3 four-row sigmoid kernel."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_lm_head_k2048",
        "rdna3_hfq4_lm_head_k2048",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048",
        "Enable the RDNA3 fixed-K LM-head experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_hfq4_moe_gate_up_k2048",
        "rdna3_hfq4_moe_gate_up_k2048",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048",
        "Enable the RDNA3 fixed-K MoE gate/up experiment."
    ),
    process_bool_field!(
        "kernel.fp16",
        "fp16",
        Kernel,
        true,
        false,
        "HIPFIRE_FP16",
        "Allow FP16 kernel routes."
    ),
    process_bool_field!(
        "kernel.wo_mmq",
        "wo_mmq",
        Kernel,
        false,
        true,
        "HIPFIRE_WO_MMQ",
        "Enable weight-only MMQ dispatch."
    ),
    process_bool_field!(
        "kernel.lm_head_wmma",
        "lm_head_wmma",
        Kernel,
        true,
        false,
        "HIPFIRE_LM_HEAD_WMMA",
        "Allow WMMA LM-head dispatch."
    ),
    process_bool_field!(
        "kernel.lm_head_overwrite",
        "lm_head_overwrite",
        Kernel,
        false,
        true,
        "HIPFIRE_LM_HEAD_OVERWRITE",
        "Enable the LM-head overwrite experiment."
    ),
    diagnostic_bool_field!(
        "diagnostic.mmq_quantize_only",
        "mmq_diag_quantize_only",
        false,
        "HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY",
        "Stop MMQ screening after quantization diagnostics."
    ),
    process_bool_field!(
        "kernel.hfq4g128_mmq",
        "hfq4g128_mmq",
        Kernel,
        true,
        false,
        "HIPFIRE_HFQ4G128_MMQ",
        "Allow HFQ4G128 MMQ dispatch."
    ),
    process_bool_field!(
        "kernel.hfq4_mmq_gfx906_y64",
        "hfq4_mmq_gfx906_y64",
        Kernel,
        false,
        true,
        "HIPFIRE_HFQ4_MMQ_GFX906_Y64",
        "Enable the gfx906 HFQ4 MMQ Y64 experiment."
    ),
    process_bool_field!(
        "kernel.gate_up_nosync",
        "gate_up_nosync",
        Kernel,
        false,
        true,
        "HIPFIRE_GATE_UP_NOSYNC",
        "Enable the no-sync gate/up experiment."
    ),
    process_bool_field!(
        "kernel.calib_force_bf16",
        "calib_force_bf16",
        Kernel,
        false,
        true,
        "HIPFIRE_CALIB_BF16",
        "Keep native-BF16 calibration teachers in BF16 instead of widening to F32 (calibration only; shipped inference is unaffected)."
    ),
    process_bool_field!(
        "kernel.draft_shared_qkv_rotation",
        "draft_shared_qkv_rotation",
        Kernel,
        true,
        false,
        "HIPFIRE_DRAFT_SHARED_QKV_ROTATION",
        "Rotate the DFlash draft's MQ4V2 noise Q/K/V input once and reuse it for all three projections (default on exact gfx1100, ignored elsewhere; set to false or HIPFIRE_DRAFT_SHARED_QKV_ROTATION=0 to rotate once per projection; same GEMMs, bit-identical)."
    ),
    process_bool_field!(
        "kernel.draft_shared_ffn_rotation",
        "draft_shared_ffn_rotation",
        Kernel,
        true,
        false,
        "HIPFIRE_DRAFT_SHARED_FFN_ROTATION",
        "Rotate the DFlash draft's MQ4V2 FFN input once and reuse it for gate and up (default on exact gfx1100, ignored elsewhere; set to false or HIPFIRE_DRAFT_SHARED_FFN_ROTATION=0 to rotate once per projection; same GEMMs, bit-identical)."
    ),
    process_bool_field!(
        "kernel.draft_shared_ctx_kv_rotation",
        "draft_shared_ctx_kv_rotation",
        Kernel,
        true,
        false,
        "HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION",
        "Rotate the DFlash draft's MQ4V2 incremental context input once and reuse it for K and V (default on exact gfx1100, ignored elsewhere; set to false or HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION=0 to rotate once per projection; same GEMMs, bit-identical)."
    ),
    process_bool_field!(
        "kernel.qkvza_split_tail",
        "qkvza_split_tail",
        Kernel,
        false,
        true,
        "HIPFIRE_QKVZA_SPLIT_TAIL",
        "Enable the RDNA3 QKVZA split-tail prefill route."
    ),
    process_field!(
        "attention.ck_runtime_lib",
        "ck_runtime_lib",
        Attention,
        DefaultValue::Null,
        ValueRule::NullableString,
        true,
        "HIPFIRE_FLASH_ATTN_CK_LIB",
        "Optional exact-architecture CK runtime sidecar path; load or capability failure retains native attention."
    ),
    process_field!(
        "attention.ck_workspace_bytes",
        "ck_workspace_bytes",
        Attention,
        DefaultValue::Integer(0),
        ValueRule::Integer { min: 0, max: 17_179_869_184 },
        true,
        "HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES",
        "Preallocated caller-owned bytes for the optional CK staged attention path."
    ),
    process_bool_field!(
        "kernel.gfx942_gemv_v3",
        "gfx942_gemv_v3",
        Kernel,
        false,
        true,
        "HIPFIRE_GFX942_GEMV_V3",
        "Enable gfx942 GEMV v3 dispatch."
    ),
    process_auto_bool_field!(
        "kernel.gfx942_rmsnorm_split",
        "gfx942_rmsnorm_split",
        Kernel,
        true,
        "HIPFIRE_GFX942_RMSNORM_SPLIT",
        "Override the architecture-selected gfx942 split RMSNorm route."
    ),
    process_bool_field!(
        "kernel.rmsnorm_mq_tight_lds",
        "rmsnorm_mq_tight_lds",
        Kernel,
        false,
        true,
        "HIPFIRE_RMSNORM_MQ_TIGHT_LDS",
        "Enable the tight-LDS fused RMSNorm MQ route."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_wavegrid",
        "rdna3_rmsnorm_wavegrid",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_WAVEGRID",
        "Enable the RDNA3 wave-grid RMSNorm experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_split",
        "rdna3_rmsnorm_split",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_SPLIT",
        "Enable the RDNA3 split RMSNorm experiment."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_sign_lds",
        "rdna3_rmsnorm_sign_lds",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_SIGN_LDS",
        "Enable LDS-staged MQ sign tables for RDNA3 RMSNorm."
    ),
    process_bool_field!(
        "kernel.rdna3_rmsnorm_sign_const",
        "rdna3_rmsnorm_sign_const",
        Kernel,
        false,
        true,
        "HIPFIRE_RDNA3_RMSNORM_SIGN_CONST",
        "Enable packed-constant MQ signs for RDNA3 RMSNorm."
    ),
    process_bool_field!(
        "kernel.moe_grouped_i8_k8",
        "moe_grouped_i8_k8",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_I8_K8",
        "Enable grouped MoE i8 K8 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_i8_k4",
        "moe_grouped_i8_k4",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_I8_K4",
        "Enable grouped MoE i8 K4 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_i8_k4_gfx12",
        "moe_grouped_i8_k4_gfx12",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_I8_K4_GFX12",
        "Enable grouped MoE i8 K4 dispatch on gfx12."
    ),
    process_bool_field!(
        "kernel.moe_grouped_m2",
        "moe_grouped_m2",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_M2",
        "Enable grouped two-row MoE dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_4w",
        "moe_grouped_4w",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_GROUPED_4W",
        "Enable grouped four-wave MoE dispatch."
    ),
    process_bool_field!(
        "kernel.moe_down_combine_vec4",
        "moe_down_combine_vec4",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_DOWN_COMBINE_VEC4",
        "Enable four-row MoE down-projection combining."
    ),
    process_bool_field!(
        "kernel.moe_hfq6_i8",
        "moe_hfq6_i8",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_HFQ6_I8",
        "Enable HFQ6 grouped MoE i8 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_hfq6_v2",
        "moe_hfq6_v2",
        Kernel,
        false,
        true,
        "HIPFIRE_MOE_HFQ6_V2",
        "Enable HFQ6 grouped MoE v2 dispatch."
    ),
    process_bool_field!(
        "kernel.moe_grouped_gemm",
        "moe_grouped_gemm",
        Kernel,
        true,
        false,
        "HIPFIRE_MOE_GROUPED_GEMM",
        "Allow grouped-GEMM MoE prefill."
    ),
    diagnostic_bool_field!(
        "diagnostic.blob_force",
        "blob_force",
        false,
        "HIPFIRE_BLOB_FORCE",
        "Force the retained kernarg-blob launch path."
    ),
    diagnostic_bool_field!(
        "diagnostic.gemm_dump",
        "gemm_dump",
        false,
        "HIPFIRE_GEMM_DUMP",
        "Dump GEMM routing diagnostics."
    ),
    process_auto_bool_field!(
        "experimental.graph.forward",
        "graph_forward",
        Experimental,
        true,
        "HIPFIRE_GRAPH",
        "Enable hipGraph forward capture; auto follows architecture policy."
    ),
    process_bool_field!(
        "experimental.graph.ar",
        "graph_ar",
        Experimental,
        true,
        true,
        "HIPFIRE_AR_GRAPH",
        "Allow autoregressive forward graph capture when otherwise eligible. Default on, except Windows gfx1100, where it defaults off (opt-in) because graph replay silently produced wrong output (#655); an explicit true/false or HIPFIRE_AR_GRAPH always wins."
    ),
    process_bool_field!(
        "experimental.graph.moe",
        "graph_moe",
        Experimental,
        true,
        true,
        "HIPFIRE_GRAPH_MOE",
        "Allow graph capture for supported MoE forward paths."
    ),
    process_bool_field!(
        "kernel.deterministic",
        "deterministic",
        Kernel,
        false,
        true,
        "HIPFIRE_DETERMINISTIC",
        "Select deterministic kernel variants where available."
    ),
    process_bool_field!(
        "kernel.q8_batched_legacy",
        "q8_batched_legacy",
        Kernel,
        false,
        true,
        "HIPFIRE_Q8_BATCHED_LEGACY",
        "Use the legacy batched Q8 route."
    ),
    process_bool_field!(
        "kernel.gemma4_q8_fused_prefill",
        "gemma4_q8_fused_prefill",
        Kernel,
        false,
        true,
        "HIPFIRE_GEMMA4_Q8_FUSED_PREFILL",
        "Fuse eligible Gemma 4 Q8 prefill projections on gfx1100."
    ),
    process_auto_bool_field!(
        "kernel.gemma4_batched_embedding_prefill",
        "gemma4_batched_embedding_prefill",
        Kernel,
        true,
        "HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL",
        "Batch eligible Gemma 4 embedding lookups; auto enables validated gfx1100/gfx1201."
    ),
    process_bool_field!(
        "kernel.gemma4_ple_batched_prefill",
        "gemma4_ple_batched_prefill",
        Kernel,
        false,
        true,
        "HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL",
        "Batch the Gemma 4 E-series PLE model projection on gfx1100."
    ),
    process_auto_bool_field!(
        "kernel.gemma4_ple_branch_batched_prefill",
        "gemma4_ple_branch_batched_prefill",
        Kernel,
        true,
        "HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL",
        "Exact-arithmetic Gemma 4 E-series PLE branch batching; auto-enabled on validated gfx1100/gfx1201 paths."
    ),
    process_auto_bool_field!(
        "kernel.gemma4_ple_activation_fused_prefill",
        "gemma4_ple_activation_fused_prefill",
        Kernel,
        true,
        "HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL",
        "Fuse the Gemma 4 E-series PLE activation and strided multiply; auto enables validated gfx1100/gfx1201."
    ),
    process_bool_field!(
        "kernel.deepseek4_q8_wmma",
        "deepseek4_q8_wmma",
        Kernel,
        true,
        false,
        "HIPFIRE_DEEPSEEK4_Q8_WMMA",
        "Allow DeepSeek4 Q8 WMMA prefill."
    ),
    process_bool_field!(
        "kernel.deepseek4_q8_4w",
        "deepseek4_q8_4w",
        Kernel,
        true,
        false,
        "HIPFIRE_DEEPSEEK4_Q8_4W",
        "Allow the DeepSeek4 four-wave Q8 WMMA tile."
    ),
    process_bool_field!(
        "kernel.rope_interleaved_legacy",
        "rope_interleaved_legacy",
        Kernel,
        false,
        true,
        "HIPFIRE_ROPE_INTERLEAVED_LEGACY",
        "Use the legacy interleaved RoPE route."
    ),
    process_bool_field!(
        "kernel.rocblas_all_archs",
        "rocblas_all_archs",
        Kernel,
        false,
        true,
        "HIPFIRE_ROCBLAS_ALL_ARCHS",
        "Allow rocBLAS dispatch on all architectures."
    ),
    process_bool_field!(
        "kernel.rocblas_off",
        "rocblas_off",
        Kernel,
        false,
        false,
        "HIPFIRE_ROCBLAS_OFF",
        "Disable rocBLAS dispatch."
    ),
    process_bool_field!(
        "kernel.lloyd_force_baseline",
        "lloyd_force_baseline",
        Kernel,
        false,
        true,
        "HIPFIRE_LLOYD_FORCE_BASELINE",
        "Force the Lloyd baseline kernel route."
    ),
    process_bool_field!(
        "fusions.force_unfused",
        "force_unfused",
        Fusions,
        false,
        true,
        "HIPFIRE_FORCE_UNFUSED",
        "Force supported projection paths to remain unfused."
    ),
    process_bool_field!(
        "speculation.dflash_tree",
        "dflash_tree",
        Speculation,
        false,
        true,
        "HIPFIRE_DFLASH_TREE",
        "Enable DDTree tree-SWOR verification."
    ),
    process_bool_field!(
        "speculation.ddtree_tree_la",
        "ddtree_tree_la",
        Speculation,
        true,
        true,
        "HIPFIRE_DDTREE_TREE_LA",
        "Allow DDTree linearized-ancestor tape replay."
    ),
    process_bool_field!(
        "speculation.dflash_fast_sample",
        "dflash_fast_sample",
        Speculation,
        true,
        true,
        "HIPFIRE_DFLASH_FAST_SAMPLE",
        "Allow DFlash GPU sampling on sampled verification."
    ),
    process_bool_field!(
        "speculation.dflash_q8_lmhead_wmma",
        "dflash_q8_lmhead_wmma",
        Speculation,
        true,
        true,
        "HIPFIRE_DFLASH_Q8_LMHEAD_WMMA",
        "Allow Q8 WMMA LM-head dispatch during DFlash verification."
    ),
    process_bool_field!(
        "fusions.qkv_bias",
        "fuse_qkv_bias",
        Fusions,
        true,
        false,
        "HIPFIRE_FUSE_QKV_BIAS",
        "Fold supported QKV bias additions into fused QKV decode."
    ),
    diagnostic_bool_field!(
        "diagnostic.qkv_bias",
        "fuse_qkv_bias_debug",
        false,
        "HIPFIRE_FUSE_QKV_BIAS_DEBUG",
        "Log each QKV-bias fold."
    ),
    ConfigField {
        key: "replay.backend",
        legacy_key: "replay_backend",
        category: ConfigCategory::Replay,
        scope: ConfigScope::Process,
        default: DefaultValue::String("auto"),
        rule: ValueRule::Enum(&["auto", "hip", "redline", "shadow", "off"]),
        registry_allowed: false,
        experimental: false,
        env_compat: Some("HIPFIRE_REPLAY_BACKEND"),
        include_builtin_in_process_config: false,
        help: "Preferred launch backend; runtime default selection is distinct from certification/admission.",
    },
    process_field!(
        "replay.transport",
        "replay_transport",
        Replay,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["auto", "aql", "pm4", "pm4_ib", "ib"]),
        true,
        "HIPFIRE_REPLAY_TRANSPORT",
        "Retained replay transport; auto follows the runtime default route predicate, not certification/admission."
    ),
    process_bool_field!(
        "replay.unsafe_wsl_redline",
        "unsafe_wsl_redline",
        Replay,
        false,
        true,
        "HIPFIRE_UNSAFE_WSL_REDLINE",
        "UNSAFE until certified: allow Redline PM4/retained replay under WSL/ROCDXG (/dev/dxg without /dev/kfd). Default off: the retained default falls back to the HIP graph and an explicit replay.backend=redline/shadow is refused."
    ),
    process_bool_field!(
        "memory.unsafe_wsl_vmm_kv",
        "unsafe_wsl_vmm_kv",
        Memory,
        false,
        true,
        "HIPFIRE_UNSAFE_WSL_VMM_KV",
        "UNSAFE until certified: allow the VMM KV backend under WSL/ROCDXG (/dev/dxg without /dev/kfd), where WDDM VA growth may alias earlier KV pages. Default off: automatic KV selects legacy and an explicit kv_backend=vmm is refused."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.route_proof_log",
        "replay_route_proof_log",
        false,
        "HIPFIRE_REPLAY_ROUTE_PROOF_LOG",
        "When enabled, the daemon emits one post-generate retained-route proof marker per successful request (fields: transport, position, request_id, replays) so product coherence can prove route identity without enabling capture."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.manual_capture",
        "replay_manual_capture",
        false,
        "HIPFIRE_REPLAY_MANUAL_CAPTURE",
        "Arm replay recording manually instead of using the model lifecycle."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pool_debug",
        "replay_pool_debug",
        false,
        "HIPFIRE_REDLINE_POOL_DEBUG",
        "Report which memory pool backs the retained indirect buffer."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.dispatch_profile",
        "replay_dispatch_profile",
        false,
        "HIPFIRE_REDLINE_DISPATCH_PROFILE",
        "Emit a GPU-clock write per dispatch and report the span distribution. \
         Changes the tape identity, so an instrumented run cannot satisfy a golden fixture."
    ),
    diagnostic_bool_field!(
        "diagnostic.compiler.no_device_compiler",
        "no_device_compiler",
        false,
        "HIPFIRE_NO_DEVICE_COMPILER",
        "Treat the device compiler as absent so pre-compiled code objects are used verbatim."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_min_parallel_width",
        "replay_pm4_min_parallel_width",
        DefaultValue::Integer(2),
        ValueRule::Integer { min: 2, max: 1024 },
        "HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH",
        "Minimum independent launches required to form a parallel PM4 phase."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_min_parallel_workgroups",
        "replay_pm4_min_parallel_workgroups",
        DefaultValue::Integer(0),
        ValueRule::Integer {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS",
        "Minimum aggregate workgroups required to form a parallel PM4 phase."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_max_parallel_phases",
        "replay_pm4_max_parallel_phases",
        DefaultValue::Null,
        ValueRule::NullableInteger {
            min: 0,
            max: i64::MAX
        },
        "HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES",
        "Maximum parallel PM4 phases; null leaves the count unlimited."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pm4_native_phases",
        "replay_pm4_native_phases",
        false,
        "HIPFIRE_REPLAY_PM4_NATIVE_PHASES",
        "Use native cross-queue synchronization for parallel PM4 phases."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_queues",
        "replay_pm4_queues",
        DefaultValue::String("1"),
        ValueRule::Enum(&["1", "2", "4", "auto"]),
        "HIPFIRE_REPLAY_PM4_QUEUES",
        "Number of PM4 queues used for retained replay."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_register_policy",
        "replay_pm4_stateful",
        DefaultValue::String("static"),
        ValueRule::Enum(&["legacy", "static", "stateful"]),
        "HIPFIRE_REPLAY_PM4_STATEFUL",
        "PM4 register emission policy; static is the gfx12-safe product default and caches only queue-global invariants."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_kernarg_pool",
        "replay_pm4_kernarg_pool",
        DefaultValue::String("vram"),
        ValueRule::Enum(&["vram", "host"]),
        "HIPFIRE_PM4_KERNARG_POOL",
        "Retained PM4 kernarg placement: vram (host-writable GPU-agent pool; host pool on small-BAR systems) or host (CPU-agent fine-grained pool)."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_wait_policy",
        "replay_pm4_wait_policy",
        DefaultValue::String("resource"),
        ValueRule::Enum(&["allowlist", "resource-audit", "resource"]),
        "HIPFIRE_REPLAY_PM4_WAIT_POLICY",
        "Dependency policy for waits between retained PM4 launches."
    ),
    diagnostic_field!(
        "diagnostic.replay.pm4_acquire_policy",
        "replay_pm4_acquire_policy",
        DefaultValue::String("required-only"),
        ValueRule::Enum(&[
            "conservative",
            "entry-only",
            "required-only",
            "without-repeat-interleave",
            "without-fused-silu-rotate",
            "without-mq-rotate",
            "without-rope"
        ]),
        "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY",
        "Cache-acquire policy between retained PM4 launches."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pm4_gcr_trim",
        "replay_pm4_gcr_trim",
        true,
        "HIPFIRE_REPLAY_PM4_GCR_TRIM",
        "Trim redundant GCR operations on supported retained PM4 routes."
    ),
    process_auto_bool_field!(
        "replay.pm4_gfx11_vmem_acquire",
        "replay_pm4_gfx11_vmem_acquire",
        Replay,
        true,
        "HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE",
        "Enable Radiowave-classified VMEM acquires on gfx11; auto selects gfx1151."
    ),
    process_field!(
        "replay.gfx1201_pm4_pacing",
        "gfx1201_pm4_pacing",
        Replay,
        DefaultValue::String("auto"),
        ValueRule::String,
        false,
        "HIPFIRE_GFX1201_PM4_PACING",
        "NOP pacing of the retained gfx1201 Qwen3.5-dense decode PM4 tape: auto (default: a 64-dword NOP after every dispatch), off/0, or nop:N (N-dword NOP after every dispatch). NOPs write no register or memory; decode is bit-identical."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_initiator",
        "gfx1151_pm4_initiator",
        DefaultValue::String("legacy"),
        ValueRule::Enum(&["legacy", "order", "radv"]),
        "HIPFIRE_GFX1151_PM4_INITIATOR",
        "gfx1151 PM4 dispatch-initiator policy."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_interleave",
        "gfx1151_pm4_interleave",
        DefaultValue::String("inherit"),
        ValueRule::Enum(&["inherit", "off", "64", "128", "256", "512"]),
        "HIPFIRE_GFX1151_PM4_INTERLEAVE",
        "gfx1151 dispatch-interleave threads per shader engine."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_resource_limits",
        "gfx1151_pm4_resource_limits",
        DefaultValue::String("legacy"),
        ValueRule::Enum(&["legacy", "simd-always", "radv"]),
        "HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS",
        "gfx1151 compute-resource-limits packet policy."
    ),
    // lifecycle: deprecated since 0.4.0, removal 0.5.0 — CU masking on RDNA hung the R9700 MES (2026-10-14); refused
    diagnostic_field!(
        "diagnostic.replay.gfx1151_cu_count",
        "gfx1151_redline_cu_count",
        DefaultValue::String("all"),
        ValueRule::String,
        "HIPFIRE_GFX1151_REDLINE_CU_COUNT",
        "Refused: any value other than all fails PM4 replay preparation (CU masks on RDNA hung the R9700 MES on 2026-10-14). Default all."
    ),
    diagnostic_field!(
        "diagnostic.replay.gfx1151_entry_acquire",
        "gfx1151_pm4_entry_acquire",
        DefaultValue::String("system"),
        ValueRule::Enum(&["system", "agent", "vmem", "none"]),
        "HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE",
        "gfx1151 PM4 entry cache-acquire scope."
    ),
    diagnostic_bool_field!(
        "diagnostic.replay.pm4_dynamic_grid",
        "replay_pm4_dynamic_grid",
        false,
        "HIPFIRE_REPLAY_PM4_DYNAMIC_GRID",
        "Patch retained PM4 grid dimensions at replay time."
    ),
    field!(
        "fusions.npu_trunk",
        "npu_trunk",
        Fusions,
        ModelLoad,
        DefaultValue::String("auto"),
        ValueRule::Enum(&["off", "auto", "on", "emulate"]),
        false,
        true,
        Some("HIPFIRE_QWEN4_NPU_TRUNK"),
        "Qwen4 trunk projection engine; auto admits calibrated gfx1151 NPU prefill, off is the GPU-only kill switch."
    ),
    field!(
        "fusions.npu_trunk_calibration",
        "npu_trunk_calibration",
        Fusions,
        ModelLoad,
        DefaultValue::String(""),
        ValueRule::String,
        false,
        true,
        Some("HIPFIRE_QWEN4_NPU_TRUNK_CAL"),
        "NPU trunk calibration path; empty resolves to <hipfire home>/npu-trunk/calibration.json; absent file uses the keyed builtin; never read when effectively off."
    ),
    field!(
        "fusions.policy",
        "fusion_policy",
        Fusions,
        ModelLoad,
        DefaultValue::String("safe"),
        ValueRule::Enum(&["safe", "off"]),
        true,
        false,
        None,
        "Certified fusion policy; individual kernel selection remains compiled."
    ),
];

pub fn fields() -> &'static [ConfigField] {
    FIELDS
}

pub fn field(key: &str) -> Option<&'static ConfigField> {
    FIELDS
        .iter()
        .find(|candidate| candidate.key == key || candidate.legacy_key == key)
}

/// The schema default of `vision.mode` (`auto`/`resident`/`on`/`off`), for a load
/// request that carries no `params.vision_mode` (a daemon-protocol client that
/// does not project the config ladder). Every consumer that needs a fallback
/// reads it here so no caller hardcodes a mode of its own.
pub fn vision_mode_default() -> &'static str {
    match field("vision.mode").map(|schema| schema.default) {
        Some(DefaultValue::String(mode)) => mode,
        _ => unreachable!("vision.mode is a string-valued schema field"),
    }
}

const DEVELOPER_PREFIX: &str = "developer.";
const BOOTSTRAP_ENV: &[&str] = &[
    "HIPFIRE_HOME",
    "HIPFIRE_MODELS_DIR",
    "HIPFIRE_DAEMON_BIN",
    "HIPFIRE_TUI_BIN",
    "HIPFIRE_CLI_BIN",
    "HIPFIRE_HF_BASE",
    "HF_ENDPOINT",
    "HIPFIRE_REGISTRY_URL",
    "HIPFIRE_NO_REGISTRY_FETCH",
    "HIPFIRE_KERNEL_CACHE",
    "HIPFIRE_SPILL_DIR",
    "HIPFIRE_QUANT_DIAG_PATH",
];

pub fn is_developer_key(key: &str) -> bool {
    key.strip_prefix(DEVELOPER_PREFIX)
        .is_some_and(valid_developer_suffix)
}

/// `developer.*` keys with a typed boolean value: `true`, `false`, `1` or `0`
/// in the `HIPFIRE_*` environment variable, `config set`, or TOML (a native
/// boolean, a string spelling, or the integers 0/1). Anything else is
/// rejected when the layer is built, never silently ignored. Every other
/// developer key keeps its free-form legacy spelling.
const TYPED_BOOL_DEVELOPER_KEYS: &[&str] = &[
    "developer.qwen4_moe_sym_iu4",
    "developer.qwen4_d2_default",
    "developer.qwen4_mtp_head_d2",
    "developer.qwen4_sp",
    "developer.qwen4_sp_touched",
    "developer.moe_sym_iu4",
    "developer.qwen4_slowlink_prefill",
    "developer.gdn_qk_indexed",
    "developer.gdn_moe_pre_fused",
    "developer.gdn_moe_chunk_scan",
    "developer.moe_router_fused_prefill",
    "developer.moe_shared_prepared_iu4",
    "developer.fa_prefill_pm",
    "developer.moe_router_topk_multirow",
    "developer.moe_combine_v2",
    "developer.moe_iu4_wide_tiles",
    "developer.prefill_iu4_producer_fused",
    "developer.fa_prefill_pm_qprep",
    "developer.moe_rotate_i4_pm",
    "developer.prefill_iu4_c8_pm",
    "developer.gdn_chunk_prep_pm",
    "developer.moe_shared_down_sigmoid_epi",
];

pub fn is_typed_bool_developer_key(key: &str) -> bool {
    TYPED_BOOL_DEVELOPER_KEYS.contains(&key)
}

/// The typed-boolean spellings: `1`/`true` and `0`/`false` (`true`/`false`
/// ASCII case-insensitive). No other word (`on`, `yes`, `2`, empty) parses.
pub fn parse_typed_bool(raw: &str) -> Option<bool> {
    match raw {
        "1" => Some(true),
        "0" => Some(false),
        _ if raw.eq_ignore_ascii_case("true") => Some(true),
        _ if raw.eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    }
}

/// Normalise `value` for the typed-boolean developer key `key` to
/// [`ConfigValue::Bool`] (rendered `1`/`0` to every `HIPFIRE_*` reader), or
/// fail with the accepted spellings.
fn typed_bool_developer_value(key: &str, value: ConfigValue) -> Result<ConfigValue> {
    let parsed = match &value {
        ConfigValue::Bool(value) => Some(*value),
        ConfigValue::Integer(0) => Some(false),
        ConfigValue::Integer(1) => Some(true),
        ConfigValue::String(raw) => parse_typed_bool(raw),
        _ => None,
    };
    parsed
        .map(ConfigValue::Bool)
        .ok_or_else(|| ConfigError::InvalidValue {
            key: key.to_owned(),
            message: format!(
                "expected a boolean: true, false, 1 or 0 (got {})",
                match &value {
                    ConfigValue::String(raw) => format!("'{raw}'"),
                    ConfigValue::Integer(raw) => raw.to_string(),
                    ConfigValue::Float(raw) => raw.to_string(),
                    ConfigValue::Bool(raw) => raw.to_string(),
                    ConfigValue::Null => "null".to_owned(),
                }
            ),
        })
}

pub fn developer_key_for_env(name: &str) -> Option<String> {
    let suffix = name.strip_prefix("HIPFIRE_")?.to_ascii_lowercase();
    valid_developer_suffix(&suffix).then(|| format!("{DEVELOPER_PREFIX}{suffix}"))
}

fn valid_developer_suffix(suffix: &str) -> bool {
    !suffix.is_empty()
        && suffix
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// Key prefixes (canonical names) of the CASK and PFlash research features,
/// deprecated since 0.4.0 with removal in 0.5.0.
const DEPRECATED_KEY_PREFIXES: &[&str] =
    &["memory.cask.", "speculation.prefill.", "diagnostic.pflash."];

/// True when `key` (canonical or legacy flat name) belongs to a deprecated
/// CASK/PFlash family: `memory.cask.*`, `speculation.prefill.*` or
/// `diagnostic.pflash.*`.
pub fn is_deprecated_key(key: &str) -> bool {
    let canonical = field(key).map_or(key, |schema| schema.key);
    DEPRECATED_KEY_PREFIXES
        .iter()
        .any(|prefix| canonical.starts_with(prefix))
}

pub fn canonical_config_key(key: &str) -> Option<String> {
    field(key)
        .map(|schema| schema.key.to_owned())
        .or_else(|| is_developer_key(key).then(|| key.to_owned()))
}

/// A config key that shipped in a release and was later removed from the
/// schema. Through its deprecation window (until `ignored_until`) a config
/// file that still sets it loads with a warning and the value is ignored;
/// keys that never shipped stay unknown and fail the load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetiredConfigKey {
    pub key: &'static str,
    pub legacy_key: &'static str,
    pub shipped_in: &'static str,
    pub removed_in: &'static str,
    pub ignored_until: &'static str,
    pub reason: &'static str,
}

pub static RETIRED_CONFIG_KEYS: &[RetiredConfigKey] = &[RetiredConfigKey {
    key: "kernel.mw16",
    legacy_key: "mw16",
    shipped_in: "0.3.1",
    removed_in: "0.4.1",
    ignored_until: "0.5.0",
    reason: "the MW16 GEMM experiment was removed",
}];

pub fn retired_config_key(key: &str) -> Option<&'static RetiredConfigKey> {
    RETIRED_CONFIG_KEYS
        .iter()
        .find(|retired| retired.key == key || retired.legacy_key == key)
}

/// The error for a key that is not in the schema: the removal message for a
/// [`RETIRED_CONFIG_KEYS`] entry, otherwise [`ConfigError::UnknownKey`].
pub fn unknown_key(key: &str) -> ConfigError {
    match retired_config_key(key) {
        Some(retired) => ConfigError::RetiredKey {
            key: retired.key,
            removed_in: retired.removed_in,
            reason: retired.reason,
        },
        None => ConfigError::UnknownKey(key.to_owned()),
    }
}

impl RetiredConfigKey {
    fn warning(&self, path: &Path) -> String {
        format!(
            "ignored {} in {}: removed in {} ({}); the setting has no effect and is \
             accepted with this warning until {} — delete it from the file",
            self.key,
            path.display(),
            self.removed_in,
            self.reason,
            self.ignored_until
        )
    }
}

pub fn developer_env_for_key(key: &str) -> Option<String> {
    key.strip_prefix(DEVELOPER_PREFIX)
        .filter(|suffix| valid_developer_suffix(suffix))
        .map(|suffix| format!("HIPFIRE_{}", suffix.to_ascii_uppercase()))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfigLayer {
    pub values: BTreeMap<String, ConfigValue>,
}

impl ConfigLayer {
    pub fn set(&mut self, key: &str, value: ConfigValue) -> Result<()> {
        if let Some(field) = field(key) {
            field.validate(&value)?;
            self.values.insert(field.key.to_owned(), value);
        } else if is_developer_key(key) {
            let value = if is_typed_bool_developer_key(key) {
                typed_bool_developer_value(key, value)?
            } else {
                value
            };
            self.values.insert(key.to_owned(), value);
        } else {
            return Err(unknown_key(key));
        }
        Ok(())
    }

    pub fn set_cli(&mut self, key: &str, raw: &str) -> Result<()> {
        if let Some(field) = field(key) {
            self.set(field.key, field.parse_cli(raw)?)
        } else if is_developer_key(key) {
            // Preserve the exact legacy spelling. Experimental consumers may
            // distinguish enum-like values such as `on` from boolean `1`.
            self.set(key, ConfigValue::String(raw.to_owned()))
        } else {
            Err(unknown_key(key))
        }
    }

    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.values.get(&canonical_config_key(key)?)
    }

    pub fn remove(&mut self, key: &str) -> Result<Option<ConfigValue>> {
        let key = canonical_config_key(key).ok_or_else(|| unknown_key(key))?;
        Ok(self.values.remove(&key))
    }

    pub fn validate(&self) -> Result<()> {
        for (key, value) in &self.values {
            if let Some(field) = field(key) {
                field.validate(value)?;
            } else if is_developer_key(key) {
                if is_typed_bool_developer_key(key) {
                    typed_bool_developer_value(key, value.clone())?;
                }
            } else {
                return Err(ConfigError::UnknownKey(key.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ConfigSource {
    BuiltIn,
    RegistryModel {
        tag: String,
        revision: String,
    },
    RegistryTarget {
        tag: String,
        arch: String,
        revision: String,
    },
    GlobalUser {
        path: PathBuf,
    },
    ModelUser {
        model: String,
        path: PathBuf,
    },
    LegacyEnv {
        name: String,
    },
    OneShot {
        argument: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConfigCandidate {
    pub value: ConfigValue,
    pub source: ConfigSource,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResolvedValue {
    pub value: ConfigValue,
    pub source: ConfigSource,
    pub shadowed: Vec<ConfigCandidate>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ResolvedConfig {
    pub values: BTreeMap<String, ResolvedValue>,
}

impl ResolvedConfig {
    pub fn get(&self, key: &str) -> Option<&ResolvedValue> {
        self.values.get(&canonical_config_key(key)?)
    }

    pub fn legacy_values(&self) -> BTreeMap<String, ConfigValue> {
        FIELDS
            .iter()
            .filter_map(|field| {
                self.values
                    .get(field.key)
                    .map(|resolved| (field.legacy_key.to_owned(), resolved.value.clone()))
            })
            .collect()
    }
}

/// Versioned process-start policy sent from the native CLI to the engine
/// daemon. Values remain sparse where absence selects an architecture-specific
/// default, but every included value is validated against [`FIELDS`] before it
/// crosses the process boundary and again after deserialization.
///
/// The daemon lowers this envelope into compact runtime-specific structs; GPU
/// and model hot paths never parse TOML or inspect this generic value map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessConfig {
    pub schema_version: i64,
    pub values: ConfigLayer,
}

static ACTIVE_PROCESS_CONFIG: OnceLock<ProcessConfig> = OnceLock::new();

impl ProcessConfig {
    pub fn from_resolved(resolved: &ResolvedConfig) -> Result<Self> {
        let mut values = ConfigLayer::default();
        for schema in FIELDS {
            if schema.env_compat.is_none() {
                continue;
            }
            let Some(resolved_value) = resolved.get(schema.key) else {
                continue;
            };
            // Architecture-sensitive defaults are represented by absence.
            // Once a user, registry, one-shot argument, or compatibility env
            // explicitly resolves the field, carry its concrete value.
            if matches!(resolved_value.source, ConfigSource::BuiltIn)
                && !schema.include_builtin_in_process_config
            {
                continue;
            }
            values.set(schema.key, resolved_value.value.clone())?;
        }
        for (key, resolved_value) in &resolved.values {
            if is_developer_key(key) {
                values.set(key, resolved_value.value.clone())?;
            }
        }
        let config = Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            values,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::InvalidValue {
                key: "schema_version".into(),
                message: format!(
                    "unsupported process config schema {}; expected {}",
                    self.schema_version, CONFIG_SCHEMA_VERSION
                ),
            });
        }
        self.values.validate()?;
        for key in self.values.values.keys() {
            if is_developer_key(key) {
                continue;
            }
            let Some(schema) = field(key) else {
                return Err(ConfigError::UnknownKey(key.clone()));
            };
            if schema.env_compat.is_none() {
                return Err(ConfigError::InvalidValue {
                    key: key.clone(),
                    message: "field is not part of process-start policy".into(),
                });
            }
        }
        Ok(())
    }

    /// Render the compatibility spelling expected by an internal snapshot
    /// parser. This is an in-memory adapter only; it does not mutate or inspect
    /// the ambient process environment.
    pub fn legacy_value(&self, name: &str) -> Option<String> {
        if let Some(schema) = FIELDS.iter().find(|schema| schema.env_compat == Some(name)) {
            return self.values.get(schema.key).and_then(render_compat_value);
        }
        let key = developer_key_for_env(name)?;
        self.values.get(&key).and_then(render_compat_value)
    }
}

pub fn install_process_config(config: ProcessConfig) -> std::result::Result<(), ProcessConfig> {
    ACTIVE_PROCESS_CONFIG.set(config)
}

pub fn active_process_config() -> Option<&'static ProcessConfig> {
    ACTIVE_PROCESS_CONFIG.get()
}

/// Resolve global TOML plus the legacy compatibility layer for direct daemon
/// and developer-tool invocation. Native CLI launches install their already
/// resolved policy explicitly and never take this fallback.
pub fn load_local_process_config() -> Result<ProcessConfig> {
    let paths = ConfigPaths::try_discover()?;
    let loaded = load_global(&paths)?;
    let mut layers = vec![NamedLayer {
        source: ConfigSource::GlobalUser { path: loaded.path },
        layer: loaded.layer,
    }];
    let environment = load_env_layer()?;
    if !environment.values.is_empty() {
        layers.push(NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: environment,
        });
    }
    ProcessConfig::from_resolved(&resolve(layers)?)
}

pub fn active_or_local_process_config() -> &'static ProcessConfig {
    ACTIVE_PROCESS_CONFIG.get_or_init(|| {
        load_local_process_config()
            .unwrap_or_else(|error| panic!("invalid hipfire process configuration: {error}"))
    })
}

/// Read one process-start value from the validated in-memory policy. The
/// argument is the temporary compatibility spelling used by compact runtime
/// parsers; this function never reads or mutates the ambient environment.
/// One hash lookup into a table rendered once from the active policy (it is
/// read on per-request and per-kernel paths); same answers as
/// [`ProcessConfig::legacy_value`].
pub fn process_value(name: &str) -> Option<String> {
    static TABLE: LazyLock<HashMap<String, String>> =
        LazyLock::new(|| legacy_table(active_or_local_process_config()));
    match TABLE.get(name) {
        Some(value) => Some(value.clone()),
        // The table holds canonical upper-case spellings; legacy_value also
        // folds the case of developer names, so defer to it for those.
        None if name.bytes().any(|b| b.is_ascii_lowercase()) => {
            active_or_local_process_config().legacy_value(name)
        }
        None => None,
    }
}

/// Every compatibility name `config` answers, rendered: schema fields under
/// their `env_compat` spelling, then developer keys under their `HIPFIRE_*`
/// name unless a schema field owns that spelling (which `legacy_value`
/// consults first, present or not).
fn legacy_table(config: &ProcessConfig) -> HashMap<String, String> {
    let mut table = HashMap::new();
    for schema in FIELDS {
        let Some(name) = schema.env_compat else {
            continue;
        };
        if let Some(value) = config.values.get(schema.key).and_then(render_compat_value) {
            table.insert(name.to_owned(), value);
        }
    }
    for (key, value) in &config.values.values {
        let Some(name) = developer_env_for_key(key) else {
            continue;
        };
        if FIELDS
            .iter()
            .any(|schema| schema.env_compat == Some(name.as_str()))
        {
            continue;
        }
        if let Some(value) = render_compat_value(value) {
            table.insert(name, value);
        }
    }
    table
}

/// Resolve the memory preflight OOM guard (`memory.oom_guard`, compat
/// `HIPFIRE_OOM_GUARD`). The guard exists because on unified-memory APUs
/// (Strix Halo) GPU allocations come out of system RAM with no swap, so a
/// bad admission takes the desktop down with a global OOM rather than
/// failing one request; on a discrete GPU an overshoot is a plain failed
/// `hipMalloc`. Default `auto` resolves per deployment class — see
/// [`oom_guard_effective`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OomGuardMode {
    /// Decide by deployment class (unified-memory APU vs discrete GPU).
    Auto,
    /// Always refuse oversized allocations before they are made.
    On,
    /// Never refuse (the operator's informed trade).
    Off,
}

/// Read the configured mode: `auto` (also unset or unparseable — validated
/// layers should not produce anything else), or an on/off spelling.
fn oom_guard_mode_for(value: Option<&str>) -> OomGuardMode {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "0" || v == "false" || v == "off" || v == "no" => OomGuardMode::Off,
        Some(v) if v == "1" || v == "true" || v == "on" || v == "yes" => OomGuardMode::On,
        _ => OomGuardMode::Auto,
    }
}

/// The configured mode of the memory preflight OOM guard.
pub fn oom_guard_mode() -> OomGuardMode {
    oom_guard_mode_for(process_value("HIPFIRE_OOM_GUARD").as_deref())
}

/// GPU architectures whose allocations land in system RAM: the GPU has no
/// private VRAM (or only a small carve-out), so model weights and KV eat the
/// same physical memory as the desktop. An overshoot here is a global OOM,
/// not a failed hipMalloc.
pub const UNIFIED_MEMORY_ARCHS: &[&str] = &[
    "gfx1035", "gfx1036", // RDNA2 APU (Van Gogh / Steam Deck class)
    "gfx1103", // RDNA3 APU (Phoenix orphan)
    "gfx1150", "gfx1151", "gfx1152", // RDNA3.5 APU (Strix Point / Strix Halo)
];

/// GPU architectures with private VRAM: allocations that exceed it fail
/// that one allocation instead of the machine.
pub const DISCRETE_MEMORY_ARCHS: &[&str] = &[
    "gfx906", "gfx908", "gfx940", "gfx941", "gfx942", // CDNA (HBM)
    "gfx1010", "gfx1011", "gfx1012", // RDNA1
    "gfx1030", "gfx1031", "gfx1032", // RDNA2 dGPU
    "gfx1100", "gfx1101", "gfx1102", // RDNA3 dGPU
    "gfx1200", "gfx1201", // RDNA4
];

/// Whether `arch` is a unified-memory APU (GPU memory is system RAM).
pub fn is_unified_memory_arch(arch: &str) -> bool {
    UNIFIED_MEMORY_ARCHS
        .iter()
        .any(|known| arch.eq_ignore_ascii_case(known))
}

/// Whether `arch` is a recognized discrete-VRAM GPU.
pub fn is_discrete_memory_arch(arch: &str) -> bool {
    DISCRETE_MEMORY_ARCHS
        .iter()
        .any(|known| arch.eq_ignore_ascii_case(known))
}

/// Whether the host has swap; `None` when the host memory snapshot or its
/// swap size is unavailable. On Linux this is `SwapTotal > 0`; on Windows the
/// swap size is the pagefile approximation from [`host_memory::host_memory`].
fn host_has_swap() -> Option<bool> {
    Some(host_memory::host_memory()?.swap_total_bytes? > 0)
}

/// Pure auto decision, testable without pinning host state. With a known GPU
/// arch the deployment class decides (unified-memory APU → on, discrete →
/// off, unrecognized → on, failing safe). Without one (no GPU has been
/// initialized in this process) the host's own lethality decides: with swap
/// an overcommit degrades instead of killing, so the guard stands down;
/// without (or with unreadable) swap, it stays up.
fn oom_guard_auto_for(arch: Option<&str>, has_swap: Option<bool>) -> bool {
    match arch {
        Some(arch) if is_unified_memory_arch(arch) => true,
        Some(arch) if is_discrete_memory_arch(arch) => false,
        Some(_) => true,
        None => !matches!(has_swap, Some(true)),
    }
}

/// Resolve whether the memory preflight guard should refuse allocations in
/// this process.
///
/// `arch` is the GPU arch this process initialized (see
/// `rdna_compute::arch_caps::process_gpu_arch`), or `None` when no GPU is
/// (yet) known — e.g. a CLI process that only supervises the daemon. The
/// `auto` decision is logged once to stderr with its reason so a refusal (or
/// a skipped refusal) in a daemon log explains itself.
pub fn oom_guard_effective(arch: Option<&str>) -> bool {
    match oom_guard_mode() {
        OomGuardMode::On => true,
        OomGuardMode::Off => false,
        OomGuardMode::Auto => {
            static DECISION_NOTE: std::sync::Once = std::sync::Once::new();
            let has_swap = host_has_swap();
            let enabled = oom_guard_auto_for(arch, has_swap);
            DECISION_NOTE.call_once(|| {
                let why = match (arch, has_swap) {
                    (Some(a), _) if is_unified_memory_arch(a) => {
                        format!("{a}: unified-memory APU; GPU allocations come from system RAM")
                    }
                    (Some(a), _) if is_discrete_memory_arch(a) => {
                        format!("{a}: discrete GPU; an overshoot is a failed hipMalloc, not an OOM")
                    }
                    (Some(a), _) => format!("{a}: unrecognized arch; failing safe"),
                    (None, Some(true)) => {
                        "no GPU arch known; host has swap, so an overcommit degrades rather than kills"
                            .to_string()
                    }
                    (None, _) => {
                        "no GPU arch known; host has no readable swap; failing safe".to_string()
                    }
                };
                eprintln!(
                    "[oom_guard] auto: {why} → guard {}",
                    if enabled { "on" } else { "off" }
                );
            });
            enabled
        }
    }
}

/// Compatibility-shaped access for experimental code while its public policy
/// is being consolidated. Values come exclusively from the process snapshot.
pub fn developer_var(name: &str) -> std::result::Result<String, std::env::VarError> {
    process_value(name).ok_or(std::env::VarError::NotPresent)
}

pub fn developer_var_os(name: &str) -> Option<std::ffi::OsString> {
    process_value(name).map(Into::into)
}
/// Strict process-snapshot boolean for the `HIPFIRE_*` long tail.
/// Reads exactly one snapshot value: `"1"` is true, `"0"` is false, and
/// anything else (absent, empty, or any other spelling) falls back to
/// `default`. The strict table matters: it exactly preserves both ambient
/// idioms it replaces — `var == "1"` (default off) and `var != "0"`
/// (default on) — while routing the read through the versioned
/// [`ProcessConfig`] snapshot so TOML `[developer]` overrides and the
/// env-beats-config-beats-default precedence apply. Deliberately narrower
/// than `FeatureFlags::from_lookup`'s `parse_bool`, which also accepts
/// `true`/`on`/`yes`: call sites that accept those spellings keep their
/// own predicate.
pub fn developer_bool(name: &str, default: bool) -> bool {
    parse_developer_bool(process_value(name).as_deref(), default)
}
/// Typed-boolean developer read for the keys in `TYPED_BOOL_DEVELOPER_KEYS`:
/// `None` when unset, else the value of one of the spellings
/// [`parse_typed_bool`] accepts (`true`/`false`/`1`/`0`). The process snapshot
/// rejects any other spelling when it is built, so reaching one here is a bug
/// in the caller (a name not declared typed) and fails loudly instead of being
/// read as unset.
pub fn developer_typed_bool(name: &str) -> Option<bool> {
    typed_developer_bool_value(name, process_value(name).as_deref())
}
/// Pure body of [`developer_typed_bool`] for unit tests.
fn typed_developer_bool_value(name: &str, raw: Option<&str>) -> Option<bool> {
    let raw = raw?;
    match parse_typed_bool(raw) {
        Some(value) => Some(value),
        None => panic!("{name}={raw:?} is not a boolean: expected true, false, 1 or 0"),
    }
}
/// Pure truth table behind [`developer_bool`], split out so unit tests can
/// pin the contract without touching the process-global snapshot.
fn parse_developer_bool(raw: Option<&str>, default: bool) -> bool {
    match raw {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}
/// The `on`/`off`/`auto` mode knobs the CLI lowers into load params
/// (`dspark_mode`, `dflash_mode`, `mtp_mode`): `"on"` forces the feature
/// (`Some(true)`), `"off"` skips it (`Some(false)`), and `"auto"` or any
/// other value defers to the loader default (`None`). Exact, lower-case.
pub fn parse_tri_state(mode: &str) -> Option<bool> {
    match mode {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}
/// Strict form of [`parse_tri_state`] for the speculation selectors the schema
/// accepts with boolean aliases (`speculation.ngram`, `mtp_ngram`,
/// `mtp_ngram_copy`): `on`/`1`/`true` → `Some(true)`, `off`/`0`/`false` →
/// `Some(false)`, `auto` → `None` (defer). Any other spelling is an error, never
/// a silent off — a typed consumer that matched only `"on"` once turned
/// `HIPFIRE_NGRAM_DRAFT=1` into off. Exact, lower-case.
pub fn parse_tri_state_alias(mode: &str) -> std::result::Result<Option<bool>, String> {
    match mode {
        "1" | "true" => Ok(Some(true)),
        "0" | "false" => Ok(Some(false)),
        "auto" => Ok(None),
        other => match parse_tri_state(other) {
            Some(value) => Ok(Some(value)),
            None => Err(format!(
                "invalid value {other:?}: expected on, 1, true, off, 0, false or auto"
            )),
        },
    }
}
/// The `on`/`off`/`auto` MTP selector the CLI lowers into `params.mtp_mode`.
/// `Off` keeps AR, `On` arms the head wherever one exists, `Auto` arms it only
/// for a producer that declared a measured AR floor ([`mtp_mode_admits`]) and,
/// even then, not over a host-mapped expert spill (the shared
/// [`mtp_host_spill_policy`] decline). `Default = Off`: the slots `mtp_mode`
/// param defaults to `"off"` unless the operator opted in, and the daemon
/// parse keeps that default
/// verbatim — only the `auto`-vs-`on` distinction now survives beside `mtp_k`.
/// Exact, lower-case; anything unrecognized (including `auto` and the empty
/// string, which the slots `mtp_k` rule already treats as off) resolves as
/// documented on [`MtpMode::parse`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpMode {
    /// Never arm the head (today's behaviour, unchanged).
    #[default]
    Off,
    /// Arm only for a producer with a measured AR floor, and not over a
    /// host-mapped expert spill the host-spill policy declines.
    Auto,
    /// Arm even over a host-mapped expert spill (explicit opt-in).
    On,
}

impl MtpMode {
    /// Parse a `params.mtp_mode` value. Absent normalizes to `"off"` at the
    /// call site (so the default stays `Off`); an explicit empty string maps
    /// to `Off` too, matching the slots `mtp_k` rule that already treats an
    /// empty name as off. Any other unrecognized spelling defers to `Auto`,
    /// matching the schema default the CLI sends.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "off" | "" => Self::Off,
            "on" => Self::On,
            _ => Self::Auto,
        }
    }
}

impl MtpMode {
    /// Lift the typed load tri-state (`SpecLoadCfg::mtp`: `Some(true)` = `on`,
    /// `Some(false)` = `off`, `None` = `auto`) into the mode every MTP arming
    /// site shares, so no consumer re-spells the three-way match.
    pub fn from_tri_state(mtp: Option<bool>) -> Self {
        match mtp {
            Some(true) => Self::On,
            Some(false) => Self::Off,
            None => Self::Auto,
        }
    }

    /// The exact lower-case spelling [`MtpMode::parse`] reads back.
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
        }
    }
}

/// The one producer-capability gate every MTP consumer applies before it loads,
/// attaches or routes a native MTP head: `measured_ar_floor` is the producer's
/// declaration that its drafter ships a measured autoregressive floor (a
/// per-request AR-vs-draft pricing that retires drafting when it loses to AR).
/// `On` is an explicit opt-in and always admits; `Off` never does; `Auto`
/// admits only a producer that declared the floor, so an unmeasured drafter
/// never speculates by default and the model keeps its plain AR route. Pure
/// and arch-blind: the capability, not the model tag or the card, decides.
/// Host-spill and head-presence policies ([`mtp_host_spill_policy`]) apply on
/// top of this verdict.
pub fn mtp_mode_admits(mode: MtpMode, measured_ar_floor: bool) -> bool {
    match mode {
        MtpMode::Off => false,
        MtpMode::On => true,
        MtpMode::Auto => measured_ar_floor,
    }
}

/// `mode` after the producer-capability gate: `Auto` on a producer without a
/// measured AR floor resolves to `Off` (consumers that read the mode see AR,
/// never a rearm from head or weights presence); `On`, `Off` and an `Auto`
/// with a declared floor are unchanged.
pub fn mtp_effective_mode(mode: MtpMode, measured_ar_floor: bool) -> MtpMode {
    match mode {
        MtpMode::Auto if !measured_ar_floor => MtpMode::Off,
        other => other,
    }
}

/// [`mtp_effective_mode`] over the raw `params.mtp_mode` spelling, as the
/// lower-case name string consumers compare against. Spelling follows
/// [`MtpMode::parse`] (an unrecognized value defers to `auto`, an empty one is
/// `off`).
pub fn mtp_effective_mode_name(raw: &str, measured_ar_floor: bool) -> &'static str {
    mtp_effective_mode(MtpMode::parse(raw), measured_ar_floor).name()
}

/// The shared log line for an `auto` MTP that a producer without a measured AR
/// floor declines, worded once for every route (sequential, slots, daemon).
pub fn mtp_no_floor_decline_line(model: &str) -> String {
    format!("  {model} MTP: off by default (no measured AR floor); opt in with --spec mtp")
}

/// One MTP-over-host-spill verdict, shared by the qwen35 and qwen4 arming
/// sites. `attach` decides whether the head loads; `reason` is the shared
/// static sentence both arches print inside their one decline line (the
/// per-arch line adds its own counts via [`mtp_host_spill_decline_line`]).
pub struct MtpHostSpillPolicy {
    pub attach: bool,
    pub reason: &'static str,
}

/// Whether an MTP head stays attached once `host_mapped_expert_blobs` routed
/// experts sit in host RAM. The arch's admitted list is data: an arch that
/// keeps `auto` armed on a measured GPU names it (qwen4's `&["gfx1100", "gfx1201"]`),
/// one with no such measurement passes none (qwen35's `&[]`). `On` always
/// attaches (explicit opt-in); `Off` never does; a fully resident placement
/// (`blobs == 0`) always attaches.
///
/// Folds in the qwen4 spelling it replaces
/// (`host == 0 || mtp == Some(true) || arch == "gfx1201"`): same table, with
/// the `gfx1201` clause as the admitted list instead of a second function.
/// Worst-case pre-placement use passes the largest count the placement can
/// still produce (never the literal `1` the old pre-copy pinned).
pub fn mtp_host_spill_policy(
    mode: MtpMode,
    host_mapped_expert_blobs: usize,
    arch: &str,
    admitted_arm_arches: &[&str],
) -> MtpHostSpillPolicy {
    if host_mapped_expert_blobs == 0 {
        return MtpHostSpillPolicy {
            attach: !matches!(mode, MtpMode::Off),
            reason: "",
        };
    }
    if matches!(mode, MtpMode::Off) {
        return MtpHostSpillPolicy {
            attach: false,
            reason: "",
        };
    }
    if matches!(mode, MtpMode::On) || admitted_arm_arches.contains(&arch) {
        return MtpHostSpillPolicy {
            attach: true,
            reason: "",
        };
    }
    MtpHostSpillPolicy {
        attach: false,
        reason: "MTP: off by default with host-mapped experts; opt in with --spec mtp",
    }
}

/// The one shared MTP-over-spill decline line, worded once here so the
/// sequential and slots routes report the same decline. `model` is the arch
/// label (`"qwen35"`, `"qwen4 native"`); `detail` is the placement fact that
/// disambiguates the spill (`"offload_exec=cpu"` / `"offload_exec=pcie"` on
/// qwen35, `"on <arch>"` on qwen4, which names the admitted-arch clause).
pub fn mtp_host_spill_decline_line(
    model: &str,
    host_mapped_expert_blobs: usize,
    host_mapped_expert_bytes: u64,
    detail: &str,
) -> String {
    format!(
        "  {model} {} ({host_mapped_expert_blobs} expert blobs, {:.2} GiB, {detail})",
        mtp_host_spill_reason(),
        host_mapped_expert_bytes as f64 / (1u64 << 30) as f64,
    )
}

/// The shared static decline sentence inside [`mtp_host_spill_decline_line`]
/// (and [`MtpHostSpillPolicy::reason`]).
pub fn mtp_host_spill_reason() -> &'static str {
    "MTP: off by default with host-mapped experts; opt in with --spec mtp"
}
/// MTP divergent-render checkpoint + strict-prefix terminal-repair policy.
///
/// Resolved once at drafter construction from the documented process-snapshot
/// keys (see `docs/env-vars.md`): `HIPFIRE_DFLASH_CKPT_RESUME` (shared with the
/// DFlash divergent-render path — there is no MTP-only twin),
/// `HIPFIRE_CACHE_CKPT_INTERVAL` / `HIPFIRE_CACHE_CKPT_MAX`, and
/// `HIPFIRE_SPEC_WINDOW_ROLLBACK`. Defaults match the MTP PRs that introduced
/// the consumers: resume on, rollback on, interval 2048 (floor 256), cap 8
/// (floor 1). Reads go through the versioned [`ProcessConfig`] snapshot so
/// TOML `[developer]` overrides and the env-beats-config-beats-default
/// precedence apply; arch crates receive these resolved values and perform
/// no policy reads of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpCachePolicy {
    pub checkpoint_resume: bool,
    pub checkpoint_interval: usize,
    pub checkpoint_cap: usize,
    pub window_rollback: bool,
}
/// Resolve the [`MtpCachePolicy`] from the process snapshot. Called once at
/// load (mirrors `build_dflash_speculator`'s one-shot resolution); never
/// per-repair or per-step.
pub fn mtp_cache_policy() -> MtpCachePolicy {
    MtpCachePolicy {
        checkpoint_resume: developer_bool("HIPFIRE_DFLASH_CKPT_RESUME", true),
        checkpoint_interval: developer_var("HIPFIRE_CACHE_CKPT_INTERVAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048usize)
            .max(256),
        checkpoint_cap: developer_var("HIPFIRE_CACHE_CKPT_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8usize)
            .max(1),
        window_rollback: developer_bool("HIPFIRE_SPEC_WINDOW_ROLLBACK", true),
    }
}
/// MTP n-gram-modifier explicit-arm check (`speculation.mtp_ngram`, env
/// override `HIPFIRE_MTP_NGRAM`). True only for an explicit `on`/`1`; `auto`
/// (the schema default), `off`/`0`, and unset all return false. This is the
/// legacy no-arch answer (Qwen3.5 verify-capacity sizing). Arch-aware arming
/// decisions use [`mtp_ngram_enabled_for_arch`].
pub fn mtp_ngram_enabled() -> bool {
    mtp_ngram_enabled_for(process_value("HIPFIRE_MTP_NGRAM").as_deref())
}
fn mtp_ngram_enabled_for(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "on"))
}
/// Arch-aware MTP + n-gram-modifier arming. `on`/`1` arms every MTP producer
/// that supports it, `off`/`0` disables it, and `auto` (the schema default;
/// an unset key also resolves to `auto`) arms it only for native Qwen4 MTP
/// (`model_arch == 16`) on gfx1151 and gfx1201.
pub fn mtp_ngram_enabled_for_arch(model_arch: u32, gpu_arch: &str) -> bool {
    mtp_ngram_enabled_for_arch_value(
        process_value("HIPFIRE_MTP_NGRAM").as_deref(),
        model_arch,
        gpu_arch,
    )
}
fn mtp_ngram_enabled_for_arch_value(value: Option<&str>, model_arch: u32, gpu_arch: &str) -> bool {
    match value {
        Some("1" | "on") => true,
        Some("0" | "off") => false,
        _ => ngram_mod_arch_admitted(model_arch, gpu_arch),
    }
}
/// Native Qwen4 MTP (model arch 16) on the two GPUs the n-gram composition is
/// admitted for.
fn ngram_mod_arch_admitted(model_arch: u32, gpu_arch: &str) -> bool {
    model_arch == 16 && matches!(gpu_arch, "gfx1151" | "gfx1201")
}
/// Fallback MTP draft cap (`speculation.mtp_k` = `auto`): every model that
/// does not report a row-exact verify capability (see [`mtp_k_default`]).
pub const MTP_K_DEFAULT: usize = 3;
/// Physical capacity of a Qwen4 verify window, including its seed row.
///
/// This is a storage/executor bound, not a row-exactness certificate or a
/// shipping admission limit. The proven per-architecture tables below remain
/// authoritative until byte, state, ID and cost qualification permits widening.
pub const QWEN4_VERIFY_MAX_ROWS: usize = 64;
/// Verify rows a Qwen4 forward runs bitwise like the single-row decode on a
/// GPU whose row-exactness oracle has not proven a wider block: the historic
/// few-row route limit (`rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS`,
/// which re-exports this value).
pub const QWEN4_FEW_ROW_ROUTE_ROWS: usize = 8;
/// gfx1151 row-exact verify rows; raised only after the row-exactness oracle
/// proves the arch at that many rows (`rdna_compute::prefill_invariant`
/// re-exports this value).
pub const QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1151: usize = 8;
/// gfx1201 row-exact verify rows; raised only after the row-exactness oracle
/// proves the arch at that many rows (`rdna_compute::prefill_invariant`
/// re-exports this value).
pub const QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1201: usize = 8;
/// Per-arch proven row-exact Qwen4 verify rows without the developer/oracle
/// overrides (other arches [`QWEN4_FEW_ROW_ROUTE_ROWS`]). The runtime value is
/// `rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows`, which layers
/// those overrides over this table; the config view of `speculation.mtp_k`
/// reads the table because `rdna-compute` depends on this crate.
pub fn qwen4_row_exact_verify_rows_table(gpu_arch: &str) -> usize {
    match gpu_arch {
        "gfx1151" => QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1151,
        "gfx1201" => QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1201,
        _ => QWEN4_FEW_ROW_ROUTE_ROWS,
    }
}
/// `auto` MTP draft cap. Native Qwen4 MTP (`model_arch == 16`, Flash-Next)
/// drafts up to one below the GPU's row-exact verify rows
/// (`row_exact_verify_rows`, the rows a verify forward runs bitwise like the
/// single-row decode): the adaptive depth chooser picks each window's depth
/// from this request's measured window costs and agreement, so the cap is the
/// proven capability, not a per-GPU tuning guess. Other models keep
/// [`MTP_K_DEFAULT`]. An explicit `HIPFIRE_MTP_K` / `speculation.mtp_k` /
/// `mtp_k` load param (including `3`) overrides this; callers apply it only
/// when no explicit value is present. Callers pass
/// `rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(gpu_arch)`;
/// the config view passes [`qwen4_row_exact_verify_rows_table`].
pub fn mtp_k_default(model_arch: u32, row_exact_verify_rows: usize) -> usize {
    if model_arch == 16 {
        row_exact_verify_rows.saturating_sub(1).max(1)
    } else {
        MTP_K_DEFAULT
    }
}
/// Sampled native-MTP enablement (`speculation.mtp_sampled`, env override
/// `HIPFIRE_MTP_SAMPLED`, default on; `0` opts out). On, the Qwen4 MTP
/// drafter verifies temperature > 0 requests by speculative rejection
/// sampling; off leaves them to AR.
pub fn mtp_sampled_enabled() -> bool {
    developer_bool("HIPFIRE_MTP_SAMPLED", true)
}
/// MTP prompt-fill route opt-out (`HIPFIRE_MTP_OWN_PREFILL=1`). Strict
/// snapshot boolean, default off: the MTP prompt fill prefills the trunk
/// through AR's ordinary route. `1` restores MTP's own route (512-row
/// speculative-verify capture chunks, sequential GDN recurrence).
pub fn mtp_own_prefill() -> bool {
    developer_bool("HIPFIRE_MTP_OWN_PREFILL", false)
}
/// Raw `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}` triple with production
/// defaults (24/48/64). Validation (max <= 64, min <= max, …) stays with the
/// arch consumer; this only resolves the snapshot values.
pub fn ngram_mod_triple() -> (usize, usize, usize) {
    ngram_mod_triple_with_defaults((24, 48, 64))
}
/// Arch-aware `(n_match, n_min, n_max)` triple. `5/3/3` is the approved
/// arch-16 (native Qwen4 MTP) default on gfx1151 and gfx1201; with the copy
/// lane armed ([`mtp_ngram_copy_enabled_for_arch`]) arch 16 resolves
/// [`NGRAM_MOD_COPY_TRIPLE`] instead. Every other family keeps the
/// production `24/48/64`. A `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}`
/// override wins per component. Validation stays with the consumer.
pub fn ngram_mod_triple_for_arch(model_arch: u32, gpu_arch: &str) -> (usize, usize, usize) {
    if model_arch == 16 && mtp_ngram_copy_enabled_for_arch(model_arch, gpu_arch) {
        return ngram_mod_copy_triple();
    }
    ngram_mod_triple_with_defaults(ngram_mod_default_triple(model_arch, gpu_arch))
}
fn ngram_mod_default_triple(model_arch: u32, gpu_arch: &str) -> (usize, usize, usize) {
    if ngram_mod_arch_admitted(model_arch, gpu_arch) {
        (5, 3, 3)
    } else {
        (24, 48, 64)
    }
}
/// The native Qwen4 MTP n-gram copy lane's default triple: `n_max` is the
/// proposal and shadow-observation length, not the takeover cap (the drafter
/// takes at most `min(n_max, row-exact verify rows - 1)` candidates, chosen
/// per window from the request's own measured acceptance and costs), and a
/// one-candidate take is allowed.
pub const NGRAM_MOD_COPY_TRIPLE: (usize, usize, usize) = (5, 1, 63);
/// [`NGRAM_MOD_COPY_TRIPLE`] with the `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,
/// N_MAX}` overrides applied per component: the triple of an armed copy lane.
pub fn ngram_mod_copy_triple() -> (usize, usize, usize) {
    ngram_mod_triple_with_defaults(NGRAM_MOD_COPY_TRIPLE)
}
/// Native Qwen4 MTP n-gram copy lane arming (`speculation.mtp_ngram_copy`,
/// env override `HIPFIRE_MTP_NGRAM_COPY`). `on`/`1` arms it wherever the
/// native Qwen4 n-gram source is armed ([`mtp_ngram_enabled_for_arch`]),
/// `off`/`0` keeps the 5/3/3 pool with row-0 confirmation on every takeover
/// (the kill switch), and `auto` (the schema default; an unset key also
/// resolves to `auto`) arms it for native Qwen4 MTP (`model_arch == 16`) on
/// gfx1151 and gfx1201, the GPUs whose greedy-identity and timing receipts
/// flipped it; every other GPU stays off until its own receipts say otherwise.
pub fn mtp_ngram_copy_enabled_for_arch(model_arch: u32, gpu_arch: &str) -> bool {
    mtp_ngram_copy_enabled_for_arch_value(
        process_value("HIPFIRE_MTP_NGRAM_COPY").as_deref(),
        model_arch,
        gpu_arch,
    )
}
fn mtp_ngram_copy_enabled_for_arch_value(
    value: Option<&str>,
    model_arch: u32,
    gpu_arch: &str,
) -> bool {
    match value {
        Some("1" | "on") => true,
        Some("0" | "off") => false,
        _ => copy_lane_arch_admitted(model_arch, gpu_arch),
    }
}
/// Native Qwen4 MTP (model arch 16) on the GPUs the copy lane's `auto` arms.
fn copy_lane_arch_admitted(model_arch: u32, gpu_arch: &str) -> bool {
    model_arch == 16 && matches!(gpu_arch, "gfx1151" | "gfx1201")
}
/// Dense Qwen3.5-class DFlash2 verbatim-copy lookup arming
/// (`speculation.dflash_lookup`, env override `HIPFIRE_DFLASH_LOOKUP`).
/// `auto` (the schema default; an unset key also resolves to `auto`) and
/// `on`/`1` arm it; only `off`/`0` disables it (CHANGELOG 0.4.3: +39.8%
/// copy-heavy decode on R9700; non-copy requests run the incumbent window).
/// This only resolves the user policy; the consumer (`DflashSpeculator`)
/// additionally requires a singleton DFlash2 selector chain and reports the
/// reason it declined.
pub fn dflash_lookup_enabled() -> bool {
    dflash_lookup_enabled_value(process_value("HIPFIRE_DFLASH_LOOKUP").as_deref())
}
fn dflash_lookup_enabled_value(value: Option<&str>) -> bool {
    !matches!(value, Some("0" | "off"))
}
/// Developer kill switch for the Tail windows of the DFlash lookup
/// (`HIPFIRE_SPEC_LOOKUP_TAIL=0`, read through the `[developer]` namespace):
/// an exact `0` disables only the Tail windows (Copy windows stay); anything
/// else, including unset, leaves them on.
pub fn dflash_lookup_tail_enabled() -> bool {
    dflash_lookup_tail_enabled_value(process_value("HIPFIRE_SPEC_LOOKUP_TAIL").as_deref())
}
fn dflash_lookup_tail_enabled_value(value: Option<&str>) -> bool {
    value != Some("0")
}
fn ngram_mod_triple_with_defaults(defaults: (usize, usize, usize)) -> (usize, usize, usize) {
    ngram_mod_triple_resolve(defaults, |name| developer_var(name).ok())
}
fn ngram_mod_triple_resolve(
    defaults: (usize, usize, usize),
    get: impl Fn(&str) -> Option<String>,
) -> (usize, usize, usize) {
    let pick =
        |name: &str, default: usize| get(name).and_then(|s| s.parse().ok()).unwrap_or(default);
    (
        pick("HIPFIRE_NGRAM_MOD_N_MATCH", defaults.0),
        pick("HIPFIRE_NGRAM_MOD_N_MIN", defaults.1),
        pick("HIPFIRE_NGRAM_MOD_N_MAX", defaults.2),
    )
}

fn render_compat_value(value: &ConfigValue) -> Option<String> {
    Some(match value {
        ConfigValue::Bool(value) => {
            if *value {
                "1".into()
            } else {
                "0".into()
            }
        }
        ConfigValue::Integer(value) => value.to_string(),
        ConfigValue::Float(value) => value.to_string(),
        ConfigValue::String(value) => value.clone(),
        ConfigValue::Null => return None,
    })
}

#[derive(Clone, Debug)]
pub struct NamedLayer {
    pub source: ConfigSource,
    pub layer: ConfigLayer,
}

/// Resolve layers supplied from lowest to highest priority.
pub fn resolve(layers: impl IntoIterator<Item = NamedLayer>) -> Result<ResolvedConfig> {
    let mut out = ResolvedConfig::default();
    for field in FIELDS {
        out.values.insert(
            field.key.to_owned(),
            ResolvedValue {
                value: field.default.to_value(),
                source: ConfigSource::BuiltIn,
                shadowed: Vec::new(),
            },
        );
    }

    for named in layers {
        named.layer.validate()?;
        for (key, value) in named.layer.values {
            if let Some(current) = out.values.get_mut(&key) {
                current.shadowed.push(ConfigCandidate {
                    value: current.value.clone(),
                    source: current.source.clone(),
                });
                current.value = value;
                current.source = named.source.clone();
            } else if is_developer_key(&key) {
                out.values.insert(
                    key,
                    ResolvedValue {
                        value,
                        source: named.source.clone(),
                        shadowed: Vec::new(),
                    },
                );
            } else {
                return Err(ConfigError::UnknownKey(key));
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigFormat {
    Toml,
    LegacyJson,
    Empty,
}

#[derive(Clone, Debug)]
pub struct LoadedConfig {
    pub layer: ConfigLayer,
    pub path: PathBuf,
    pub format: ConfigFormat,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogFormat {
    Toml,
    LegacyJson,
    Empty,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalModelConfig {
    pub path: Option<PathBuf>,
    pub registry_tag: Option<String>,
    pub overrides: ConfigLayer,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelCatalog {
    /// Human-friendly name to canonical local model identity.
    pub aliases: BTreeMap<String, String>,
    /// Canonical local identity to path, registry identity, and sparse overrides.
    pub models: BTreeMap<String, LocalModelConfig>,
}

impl ModelCatalog {
    pub fn model_id(&self, name: &str) -> Option<&str> {
        if let Some((id, _)) = self.models.get_key_value(name) {
            return Some(id);
        }
        if let Some(id) = self.aliases.get(name) {
            return Some(id);
        }
        self.models.iter().find_map(|(id, model)| {
            let path_match = model.path.as_ref().is_some_and(|path| {
                path == Path::new(name)
                    || path.file_name().and_then(|file| file.to_str()) == Some(name)
            });
            (model.registry_tag.as_deref() == Some(name) || path_match).then_some(id.as_str())
        })
    }

    pub fn model(&self, name: &str) -> Option<(&str, &LocalModelConfig)> {
        let id = self.model_id(name)?;
        Some((id, self.models.get(id)?))
    }

    pub fn validate(&self) -> Result<()> {
        let mut paths = BTreeSet::new();
        for (id, model) in &self.models {
            if id.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: "models".into(),
                    message: "model identity cannot be empty".into(),
                });
            }
            if let Some(path) = &model.path {
                if path.as_os_str().is_empty() {
                    return Err(ConfigError::InvalidValue {
                        key: format!("models.{id}.path"),
                        message: "path cannot be empty".into(),
                    });
                }
                if !paths.insert(path.clone()) {
                    return Err(ConfigError::InvalidValue {
                        key: format!("models.{id}.path"),
                        message: format!("duplicate catalog path {}", path.display()),
                    });
                }
            }
            validate_model_layer(&model.overrides)?;
        }
        for (alias, target) in &self.aliases {
            if alias.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: "aliases".into(),
                    message: "alias cannot be empty".into(),
                });
            }
            if alias == target {
                return Err(ConfigError::InvalidValue {
                    key: format!("aliases.{alias}"),
                    message: "alias cannot point to itself".into(),
                });
            }
            if !self.models.contains_key(target) {
                return Err(ConfigError::InvalidValue {
                    key: format!("aliases.{alias}"),
                    message: format!("unknown local model identity {target}"),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LoadedCatalog {
    pub catalog: ModelCatalog,
    pub path: PathBuf,
    pub format: CatalogFormat,
    pub warnings: Vec<String>,
}

/// Resolve the opt-in trunk calibration path without reading the file.
pub fn npu_trunk_calibration_path() -> PathBuf {
    developer_var("HIPFIRE_QWEN4_NPU_TRUNK_CAL")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            ConfigPaths::discover()
                .root
                .join("npu-trunk/calibration.json")
        })
}

#[derive(Clone, Debug)]
pub struct ConfigPaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub profiles: PathBuf,
    pub config_toml: PathBuf,
    pub config_json: PathBuf,
    pub models_toml: PathBuf,
    pub models_json: PathBuf,
    pub legacy_per_model_json: PathBuf,
    /// Tuning-preset ownership and undo state (`tuning.toml`), a control-plane
    /// file beside `config.toml`; see [`tuning`].
    pub tuning_toml: PathBuf,
}

/// The hipfire state root every crate shares (`ConfigPaths::root`,
/// `RegistryPaths::cache`'s parent). Selection order:
///
/// - Unix: `HIPFIRE_HOME`, else `$HOME/.hipfire`, else the relative `.hipfire`.
/// - Windows: `HIPFIRE_HOME`, else `%USERPROFILE%\.hipfire` (the root
///   `scripts/install.ps1` installs into), else `%HOME%\.hipfire`. A Windows
///   root is never CWD-relative: a missing home or a non-absolute value is an
///   actionable [`ConfigError::Home`].
pub fn hipfire_home() -> Result<PathBuf> {
    hipfire_home_from(cfg!(windows), |name| env::var_os(name))
}

/// Pure core of [`hipfire_home`]: `windows` selects the rule set and `get`
/// supplies environment values, so tests inject both without touching the
/// process environment.
fn hipfire_home_from(
    windows: bool,
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<PathBuf> {
    if !windows {
        return Ok(get("HIPFIRE_HOME")
            .map(PathBuf::from)
            .or_else(|| get("HOME").map(|home| PathBuf::from(home).join(".hipfire")))
            .unwrap_or_else(|| PathBuf::from(".hipfire")));
    }
    if let Some(raw) = get("HIPFIRE_HOME") {
        return require_windows_absolute("HIPFIRE_HOME", PathBuf::from(raw));
    }
    for var in ["USERPROFILE", "HOME"] {
        if let Some(profile) = get(var).filter(|value| !value.is_empty()) {
            return require_windows_absolute(var, PathBuf::from(profile))
                .map(|profile| profile.join(".hipfire"));
        }
    }
    Err(ConfigError::Home(
        "cannot locate the hipfire home: neither HIPFIRE_HOME nor USERPROFILE is set \
         (and HOME is unset); set HIPFIRE_HOME to an absolute directory such as \
         C:\\Users\\<you>\\.hipfire"
            .to_string(),
    ))
}

fn require_windows_absolute(var: &str, path: PathBuf) -> Result<PathBuf> {
    if is_windows_absolute(&path) {
        Ok(path)
    } else {
        Err(ConfigError::Home(format!(
            "{var}={} is not an absolute Windows path; hipfire will not use a \
             current-directory-relative home. Set {var} to a full path such as \
             C:\\Users\\<you>\\.hipfire",
            path.display()
        )))
    }
}

/// Windows absoluteness decided textually so it is the same on every host:
/// `C:\` / `C:/` drive roots, or `\\server\share` / `\\?\` style prefixes.
/// Drive-relative (`C:foo`) and rooted-without-drive (`\foo`) paths are not
/// absolute.
fn is_windows_absolute(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let bytes = text.as_bytes();
    let is_sep = |b: u8| b == b'\\' || b == b'/';
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && is_sep(bytes[2]) {
        return true;
    }
    bytes.len() >= 3 && is_sep(bytes[0]) && is_sep(bytes[1]) && !is_sep(bytes[2])
}

impl ConfigPaths {
    /// Infallible form of [`ConfigPaths::try_discover`]; panics with the
    /// actionable message when no usable Windows home exists. Fallible callers
    /// use `try_discover`.
    pub fn discover() -> Self {
        Self::try_discover().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_discover() -> Result<Self> {
        let mut paths = Self::under(hipfire_home()?);
        if let Some(models) = env::var_os("HIPFIRE_MODELS_DIR") {
            paths.models = PathBuf::from(models);
        }
        Ok(paths)
    }

    pub fn under(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            models: root.join("models"),
            profiles: root.join("profiles"),
            config_toml: root.join("config.toml"),
            config_json: root.join("config.json"),
            models_toml: root.join("models.toml"),
            models_json: root.join("models.json"),
            legacy_per_model_json: root.join("per_model_config.json"),
            tuning_toml: root.join("tuning.toml"),
            root,
        }
    }
}

pub fn load_global(paths: &ConfigPaths) -> Result<LoadedConfig> {
    if paths.config_toml.exists() {
        let (layer, warnings) = load_toml_layer_with_warnings(&paths.config_toml)?;
        return Ok(LoadedConfig {
            layer,
            path: paths.config_toml.clone(),
            format: ConfigFormat::Toml,
            warnings,
        });
    }
    if paths.config_json.exists() {
        let (layer, warnings) = load_legacy_json_layer(&paths.config_json)?;
        return Ok(LoadedConfig {
            layer,
            path: paths.config_json.clone(),
            format: ConfigFormat::LegacyJson,
            warnings,
        });
    }
    Ok(LoadedConfig {
        layer: ConfigLayer::default(),
        path: paths.config_toml.clone(),
        format: ConfigFormat::Empty,
        warnings: Vec::new(),
    })
}

/// Load the local model catalog. TOML is authoritative once present. Legacy
/// JSON inputs are merged in memory so the first native write can preserve
/// aliases and per-model overrides without deleting the rollback files.
pub fn load_catalog(paths: &ConfigPaths) -> Result<LoadedCatalog> {
    if paths.models_toml.exists() {
        return Ok(LoadedCatalog {
            catalog: load_catalog_toml(&paths.models_toml)?,
            path: paths.models_toml.clone(),
            format: CatalogFormat::Toml,
            warnings: Vec::new(),
        });
    }

    if paths.models_json.exists() || paths.legacy_per_model_json.exists() {
        let (catalog, warnings) = load_legacy_catalog(paths)?;
        return Ok(LoadedCatalog {
            catalog,
            path: paths.models_json.clone(),
            format: CatalogFormat::LegacyJson,
            warnings,
        });
    }

    Ok(LoadedCatalog {
        catalog: ModelCatalog::default(),
        path: paths.models_toml.clone(),
        format: CatalogFormat::Empty,
        warnings: Vec::new(),
    })
}

pub fn load_catalog_toml(path: &Path) -> Result<ModelCatalog> {
    let raw = read_string(path)?;
    let mut root = toml::from_str::<toml::Table>(&raw).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        message: source.to_string(),
    })?;
    let version = root
        .remove("schema_version")
        .and_then(|value| value.as_integer())
        .unwrap_or(CONFIG_SCHEMA_VERSION);
    if version != CONFIG_SCHEMA_VERSION {
        return Err(ConfigError::Parse {
            path: path.to_owned(),
            message: format!("unsupported schema_version {version}"),
        });
    }

    let aliases_value = root
        .remove("aliases")
        .unwrap_or_else(|| toml::Value::Table(toml::Table::new()));
    let aliases_table = aliases_value.as_table().ok_or_else(|| ConfigError::Parse {
        path: path.to_owned(),
        message: "aliases must be a table".into(),
    })?;
    let mut aliases = BTreeMap::new();
    for (alias, target) in aliases_table {
        let target = target.as_str().ok_or_else(|| ConfigError::Parse {
            path: path.to_owned(),
            message: format!("aliases.{alias} must be a string"),
        })?;
        aliases.insert(alias.clone(), target.to_owned());
    }

    let models_value = root
        .remove("models")
        .unwrap_or_else(|| toml::Value::Table(toml::Table::new()));
    let models_table = models_value.as_table().ok_or_else(|| ConfigError::Parse {
        path: path.to_owned(),
        message: "models must be a table".into(),
    })?;
    let mut models = BTreeMap::new();
    for (id, value) in models_table {
        let record = value.as_table().ok_or_else(|| ConfigError::Parse {
            path: path.to_owned(),
            message: format!("models.{id} must be a table"),
        })?;
        let unknown = record
            .keys()
            .filter(|key| !matches!(key.as_str(), "path" | "registry_tag" | "overrides"))
            .cloned()
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(ConfigError::Parse {
                path: path.to_owned(),
                message: format!("unknown fields in models.{id}: {}", unknown.join(", ")),
            });
        }
        let path_value = record
            .get("path")
            .map(|value| {
                value
                    .as_str()
                    .map(PathBuf::from)
                    .ok_or_else(|| ConfigError::Parse {
                        path: path.to_owned(),
                        message: format!("models.{id}.path must be a string"),
                    })
            })
            .transpose()?;
        let registry_tag = record
            .get("registry_tag")
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ConfigError::Parse {
                        path: path.to_owned(),
                        message: format!("models.{id}.registry_tag must be a string"),
                    })
            })
            .transpose()?;
        let mut overrides = ConfigLayer::default();
        if let Some(value) = record.get("overrides") {
            let table = value.as_table().ok_or_else(|| ConfigError::Parse {
                path: path.to_owned(),
                message: format!("models.{id}.overrides must be a table"),
            })?;
            let mut flat = BTreeMap::new();
            flatten_toml("", table, &mut flat, path)?;
            for (key, value) in flat {
                overrides.set(&key, value)?;
            }
        }
        validate_model_layer(&overrides)?;
        models.insert(
            id.clone(),
            LocalModelConfig {
                path: path_value,
                registry_tag,
                overrides,
            },
        );
    }
    if !root.is_empty() {
        return Err(ConfigError::Parse {
            path: path.to_owned(),
            message: format!(
                "unknown top-level fields: {}",
                root.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        });
    }

    let catalog = ModelCatalog { aliases, models };
    catalog.validate()?;
    Ok(catalog)
}

pub fn write_catalog_toml(paths: &ConfigPaths, catalog: &ModelCatalog) -> Result<()> {
    catalog.validate()?;
    let mut root = toml::Table::new();
    root.insert(
        "schema_version".into(),
        toml::Value::Integer(CONFIG_SCHEMA_VERSION),
    );
    if !catalog.aliases.is_empty() {
        root.insert(
            "aliases".into(),
            toml::Value::Table(
                catalog
                    .aliases
                    .iter()
                    .map(|(alias, target)| (alias.clone(), toml::Value::String(target.clone())))
                    .collect(),
            ),
        );
    }
    let mut models = toml::Table::new();
    for (id, model) in &catalog.models {
        let mut record = toml::Table::new();
        if let Some(path) = &model.path {
            record.insert(
                "path".into(),
                toml::Value::String(path.display().to_string()),
            );
        }
        if let Some(tag) = &model.registry_tag {
            record.insert("registry_tag".into(), toml::Value::String(tag.clone()));
        }
        if !model.overrides.values.is_empty() {
            let mut overrides = toml::Table::new();
            for (key, value) in &model.overrides.values {
                let Some(value) = value.clone().into_toml() else {
                    continue;
                };
                insert_toml_path(&mut overrides, key, value)?;
            }
            record.insert("overrides".into(), toml::Value::Table(overrides));
        }
        models.insert(id.clone(), toml::Value::Table(record));
    }
    if !models.is_empty() {
        root.insert("models".into(), toml::Value::Table(models));
    }
    let rendered =
        toml::to_string_pretty(&toml::Value::Table(root)).map_err(|source| ConfigError::Parse {
            path: paths.models_toml.clone(),
            message: source.to_string(),
        })?;
    atomic_write(&paths.models_toml, rendered.as_bytes())
}

/// Whether `key` may appear in a per-model override layer.
///
/// Developer keys are global process policy; process/diagnostic scoped fields
/// are not valid per model. This is the per-key rule `models.toml` validation
/// applies to every override.
pub fn model_override_allowed(key: &str) -> Result<()> {
    if is_developer_key(key) {
        return Err(ConfigError::InvalidValue {
            key: key.to_owned(),
            message: "developer fields are global process policy and not valid per-model".into(),
        });
    }
    let Some(schema) = field(key) else {
        return Err(unknown_key(key));
    };
    if matches!(schema.scope, ConfigScope::Process | ConfigScope::Diagnostic) {
        return Err(ConfigError::InvalidValue {
            key: key.to_owned(),
            message: "field is not valid in a per-model override".into(),
        });
    }
    Ok(())
}

fn validate_model_layer(layer: &ConfigLayer) -> Result<()> {
    layer.validate()?;
    for key in layer.values.keys() {
        model_override_allowed(key)?;
    }
    Ok(())
}

/// What [`set_model_override`] saved, for the caller to report.
#[derive(Clone, Debug, PartialEq)]
pub struct SetModelOverride {
    /// Catalog identity the override was written under.
    pub id: String,
    /// Canonical key that was set.
    pub key: &'static str,
    /// Value as stored in the override layer.
    pub value: ConfigValue,
    /// The catalog was read from legacy JSON and has now been written as TOML.
    pub migrated_legacy_json: bool,
}

/// Outcome of [`remove_model_override`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoveModelOverride {
    /// The catalog has no entry for the model; nothing was written.
    NoOverrides,
    /// The key's override was removed.
    KeyRemoved { id: String, key: &'static str },
    /// The key had no override; the catalog was rewritten unchanged.
    KeyInherited { id: String, key: &'static str },
    /// Every override of the model was removed.
    AllRemoved { id: String },
}

/// Set one per-model override and write `models.toml`.
///
/// The catalog id is the existing identity for `model`, else `registry_tag`,
/// else `model` itself. `local_path` and `registry_tag` backfill the record
/// only when it has none.
pub fn set_model_override(
    paths: &ConfigPaths,
    registry_tag: Option<&str>,
    local_path: Option<PathBuf>,
    model: &str,
    key: &str,
    raw: &str,
) -> Result<SetModelOverride> {
    let mut loaded = load_catalog(paths)?;
    let id = loaded
        .catalog
        .model_id(model)
        .map(str::to_owned)
        .unwrap_or_else(|| registry_tag.unwrap_or(model).to_owned());
    let (canonical, saved) = {
        let record = loaded.catalog.models.entry(id.clone()).or_default();
        if record.path.is_none() {
            record.path = local_path;
        }
        if record.registry_tag.is_none() {
            record.registry_tag = registry_tag.map(str::to_owned);
        }
        record.overrides.set_cli(key, raw)?;
        let schema = field(key).ok_or_else(|| unknown_key(key))?;
        let saved = record
            .overrides
            .get(schema.key)
            .ok_or_else(|| {
                ConfigError::Catalog(format!("configuration key '{}' is not set", schema.key))
            })?
            .clone();
        (schema.key, saved)
    };
    write_catalog_toml(paths, &loaded.catalog)?;
    Ok(SetModelOverride {
        id,
        key: canonical,
        value: saved,
        migrated_legacy_json: loaded.format == CatalogFormat::LegacyJson,
    })
}

/// Remove one per-model override (`key`) or all of them (`None`) and write
/// `models.toml`. A model with no catalog entry writes nothing.
pub fn remove_model_override(
    paths: &ConfigPaths,
    model: &str,
    key: Option<&str>,
) -> Result<RemoveModelOverride> {
    let mut loaded = load_catalog(paths)?;
    let Some(id) = loaded.catalog.model_id(model).map(str::to_owned) else {
        return Ok(RemoveModelOverride::NoOverrides);
    };
    let record = loaded.catalog.models.get_mut(&id).ok_or_else(|| {
        ConfigError::Catalog(format!(
            "model '{model}' has no catalog entry for id '{id}'"
        ))
    })?;
    let outcome = if let Some(key) = key {
        let schema = field(key).ok_or_else(|| unknown_key(key))?;
        if record.overrides.remove(schema.key)?.is_some() {
            RemoveModelOverride::KeyRemoved {
                id,
                key: schema.key,
            }
        } else {
            RemoveModelOverride::KeyInherited {
                id,
                key: schema.key,
            }
        }
    } else {
        record.overrides = ConfigLayer::default();
        RemoveModelOverride::AllRemoved { id }
    };
    write_catalog_toml(paths, &loaded.catalog)?;
    Ok(outcome)
}

fn load_legacy_catalog(paths: &ConfigPaths) -> Result<(ModelCatalog, Vec<String>)> {
    let mut catalog = ModelCatalog::default();
    let mut warnings = Vec::new();
    let raw = if paths.models_json.exists() {
        Some(read_json_object(&paths.models_json)?)
    } else {
        None
    };

    if let Some(raw) = raw.as_ref() {
        if raw
            .get("schema_version")
            .and_then(serde_json::Value::as_i64)
            == Some(2)
        {
            if let Some(models) = raw.get("models").and_then(serde_json::Value::as_object) {
                for (id, value) in models {
                    let Some(record) = value.as_object() else {
                        warnings.push(format!("ignored invalid legacy model {id}"));
                        continue;
                    };
                    let path_value = record
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .map(PathBuf::from);
                    let registry_tag = record
                        .get("registry_tag")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned);
                    let overrides = legacy_override_layer(
                        record.get("config"),
                        &format!("models.{id}.config"),
                        &mut warnings,
                    );
                    catalog.models.insert(
                        id.clone(),
                        LocalModelConfig {
                            path: path_value,
                            registry_tag,
                            overrides,
                        },
                    );
                    if let Some(aliases) =
                        record.get("aliases").and_then(serde_json::Value::as_array)
                    {
                        for alias in aliases.iter().filter_map(serde_json::Value::as_str) {
                            catalog.aliases.insert(alias.to_owned(), id.clone());
                        }
                    }
                }
            }
            if let Some(configs) = raw.get("configs").and_then(serde_json::Value::as_object) {
                for (key, value) in configs {
                    merge_legacy_override(&mut catalog, key, value, &mut warnings);
                }
            }
            migrate_legacy_aliases(raw.get("aliases"), paths, &mut catalog, &mut warnings);
        } else {
            migrate_legacy_aliases(
                Some(&serde_json::Value::Object(raw.clone())),
                paths,
                &mut catalog,
                &mut warnings,
            );
        }
    }

    if paths.legacy_per_model_json.exists() {
        let legacy = read_json_object(&paths.legacy_per_model_json)?;
        for (key, value) in &legacy {
            merge_legacy_override(&mut catalog, key, value, &mut warnings);
        }
    }

    // Legacy catalogs can contain duplicate derived paths or dangling alias
    // records. Preserve every usable override and report the rest instead of
    // turning migration into a startup failure.
    for (alias, target) in catalog.aliases.clone() {
        if !catalog.models.contains_key(&target) {
            catalog.aliases.remove(&alias);
            warnings.push(format!("ignored dangling legacy alias {alias} -> {target}"));
        }
    }
    Ok((catalog, warnings))
}

fn read_json_object(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let raw = read_string(path)?;
    serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            message: source.to_string(),
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| ConfigError::Parse {
            path: path.to_owned(),
            message: "root must be a JSON object".into(),
        })
}

fn legacy_override_layer(
    value: Option<&serde_json::Value>,
    context: &str,
    warnings: &mut Vec<String>,
) -> ConfigLayer {
    let mut layer = ConfigLayer::default();
    let Some(object) = value.and_then(serde_json::Value::as_object) else {
        return layer;
    };
    for (key, value) in object {
        let Some(schema) = field(key) else {
            warnings.push(format!("ignored unknown legacy field {context}.{key}"));
            continue;
        };
        let Some(value) = from_json_value(value) else {
            warnings.push(format!("ignored unsupported legacy value {context}.{key}"));
            continue;
        };
        if matches!(schema.scope, ConfigScope::Process | ConfigScope::Diagnostic) {
            warnings.push(format!("ignored global-only legacy field {context}.{key}"));
            continue;
        }
        if let Err(error) = layer.set(schema.key, value) {
            warnings.push(format!(
                "ignored invalid legacy field {context}.{key}: {error}"
            ));
        }
    }
    layer
}

fn merge_legacy_override(
    catalog: &mut ModelCatalog,
    key: &str,
    value: &serde_json::Value,
    warnings: &mut Vec<String>,
) {
    let layer = legacy_override_layer(Some(value), key, warnings);
    if layer.values.is_empty() {
        return;
    }
    let model_id = catalog
        .model_id(key)
        .map(str::to_owned)
        .unwrap_or_else(|| key.to_owned());
    let model = catalog
        .models
        .entry(model_id)
        .or_insert_with(|| LocalModelConfig {
            registry_tag: Some(key.to_owned()),
            ..LocalModelConfig::default()
        });
    model.overrides.values.extend(layer.values);
}

fn migrate_legacy_aliases(
    value: Option<&serde_json::Value>,
    paths: &ConfigPaths,
    catalog: &mut ModelCatalog,
    warnings: &mut Vec<String>,
) {
    let Some(aliases) = value.and_then(serde_json::Value::as_object) else {
        return;
    };
    for (alias, value) in aliases {
        let Some(record) = value.as_object() else {
            warnings.push(format!("ignored invalid legacy alias {alias}"));
            continue;
        };
        let local_path = record
            .get("local_path")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from);
        let file = record.get("file").and_then(serde_json::Value::as_str);
        let path = local_path.or_else(|| file.map(|file| paths.models.join(file)));
        let existing = catalog.models.iter().find_map(|(id, model)| {
            let same_path = path
                .as_ref()
                .is_some_and(|path| model.path.as_ref() == Some(path));
            let same_file = file.is_some_and(|file| {
                model
                    .path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|v| v.to_str())
                    == Some(file)
            });
            (same_path || same_file).then_some(id.clone())
        });
        let id = existing.unwrap_or_else(|| format!("local:{alias}"));
        catalog
            .models
            .entry(id.clone())
            .or_insert_with(|| LocalModelConfig {
                path,
                ..LocalModelConfig::default()
            });
        catalog.aliases.insert(alias.clone(), id);
    }
}

/// Whether the legacy environment selects devices (`HIPFIRE_DEVICES` or its
/// singular spelling `HIPFIRE_DEVICE`), without validating any other key.
pub fn legacy_devices_env_set() -> bool {
    env::var_os("HIPFIRE_DEVICES").is_some() || env::var_os("HIPFIRE_DEVICE").is_some()
}

pub fn load_env_layer() -> Result<ConfigLayer> {
    let mut layer = ConfigLayer::default();
    let mut stable_names = BTreeSet::new();
    for field in FIELDS {
        let Some(name) = field.env_compat else {
            continue;
        };
        stable_names.insert(name);
        let Ok(raw) = env::var(name) else {
            continue;
        };
        let value = match (field.legacy_key, raw.as_str()) {
            ("default_chatml", "0") => ConfigValue::Bool(false),
            ("dflash_ngram_block", "1") => ConfigValue::Bool(true),
            ("prompt_normalize", "0" | "false" | "off" | "no") => ConfigValue::Bool(false),
            ("prompt_normalize", _) => ConfigValue::Bool(true),
            _ => field.parse_cli(&raw)?,
        };
        layer.set(field.key, value)?;
    }
    // Singular spelling of HIPFIRE_DEVICES; conflicting values fail closed.
    stable_names.insert("HIPFIRE_DEVICE");
    if let Ok(single) = env::var("HIPFIRE_DEVICE") {
        match env::var("HIPFIRE_DEVICES") {
            Ok(plural) if plural != single => {
                return Err(ConfigError::InvalidValue {
                    key: "hardware.devices".into(),
                    message: format!(
                        "HIPFIRE_DEVICE={single:?} conflicts with HIPFIRE_DEVICES={plural:?}"
                    ),
                })
            }
            Ok(_) => {}
            Err(_) => layer.set_cli("hardware.devices", &single)?,
        }
    }
    // Snapshot the experimental long tail once. These values are quarantined
    // under [developer], excluded from registry/model policy, and read by the
    // engine only through ProcessConfig after this point.
    for (name, raw) in env::vars_os() {
        let Some(name) = name.to_str() else {
            continue;
        };
        if stable_names.contains(name) || BOOTSTRAP_ENV.contains(&name) {
            continue;
        }
        let Some(key) = developer_key_for_env(name) else {
            continue;
        };
        let Ok(raw) = raw.into_string() else {
            continue;
        };
        layer.set(&key, ConfigValue::String(raw))?;
    }
    Ok(layer)
}

/// Load a TOML config layer. Retired keys ([`RETIRED_CONFIG_KEYS`]) are
/// dropped; use [`load_toml_layer_with_warnings`] to report them.
pub fn load_toml_layer(path: &Path) -> Result<ConfigLayer> {
    load_toml_layer_with_warnings(path).map(|(layer, _)| layer)
}

/// Load a TOML config layer, ignoring retired keys with one warning each.
/// Every other unknown key still fails the load.
pub fn load_toml_layer_with_warnings(path: &Path) -> Result<(ConfigLayer, Vec<String>)> {
    let raw = read_string(path)?;
    let table = toml::from_str::<toml::Table>(&raw).map_err(|source| ConfigError::Parse {
        path: path.to_owned(),
        message: source.to_string(),
    })?;
    if let Some(version) = table.get("schema_version") {
        if version.as_integer() != Some(CONFIG_SCHEMA_VERSION) {
            return Err(ConfigError::Parse {
                path: path.to_owned(),
                message: format!("unsupported schema_version {version}"),
            });
        }
    }
    let mut flat = BTreeMap::new();
    flatten_toml("", &table, &mut flat, path)?;
    let mut layer = ConfigLayer::default();
    let mut warnings = Vec::new();
    for (key, value) in flat {
        if field(&key).is_none() {
            if let Some(retired) = retired_config_key(&key) {
                warnings.push(retired.warning(path));
                continue;
            }
        }
        layer.set(&key, value)?;
    }
    Ok((layer, warnings))
}

fn flatten_toml(
    prefix: &str,
    table: &toml::map::Map<String, toml::Value>,
    out: &mut BTreeMap<String, ConfigValue>,
    path: &Path,
) -> Result<()> {
    for (key, value) in table {
        if prefix.is_empty() && (key == "schema_version" || key == calibration::SECTION) {
            continue;
        }
        let dotted = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        if let Some(nested) = value.as_table() {
            flatten_toml(&dotted, nested, out, path)?;
        } else if let Some(value) = ConfigValue::from_toml(value) {
            out.insert(dotted, value);
        } else {
            return Err(ConfigError::Parse {
                path: path.to_owned(),
                message: format!("unsupported value at {dotted}"),
            });
        }
    }
    Ok(())
}

pub fn load_legacy_json_layer(path: &Path) -> Result<(ConfigLayer, Vec<String>)> {
    let raw = read_string(path)?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            message: source.to_string(),
        })?;
    let object = parsed.as_object().ok_or_else(|| ConfigError::Parse {
        path: path.to_owned(),
        message: "root must be a JSON object".to_owned(),
    })?;
    let mut layer = ConfigLayer::default();
    let mut warnings = Vec::new();
    for (key, value) in object {
        let Some(field) = field(key) else {
            warnings.push(format!("ignored unknown legacy key {key}"));
            continue;
        };
        let Some(value) = from_json_value(value) else {
            warnings.push(format!("ignored unsupported legacy value for {key}"));
            continue;
        };
        // 32768 was the TypeScript CLI's built-in default, which it always sent.
        // Carrying it over would pin 32K on carriers that size their own context.
        if field.key == "memory.max_seq" && value == ConfigValue::Integer(32768) {
            warnings.push(format!(
                "dropped legacy {key} = 32768 (the old built-in default); an omitted max_seq sizes the context automatically"
            ));
            continue;
        }
        if let Err(error) = layer.set(field.key, value) {
            warnings.push(format!("ignored invalid legacy {key}: {error}"));
        }
    }
    Ok((layer, warnings))
}

fn from_json_value(value: &serde_json::Value) -> Option<ConfigValue> {
    match value {
        serde_json::Value::Null => Some(ConfigValue::Null),
        serde_json::Value::Bool(v) => Some(ConfigValue::Bool(*v)),
        serde_json::Value::Number(v) => v
            .as_i64()
            .map(ConfigValue::Integer)
            .or_else(|| v.as_f64().map(ConfigValue::Float)),
        serde_json::Value::String(v) => Some(ConfigValue::String(v.clone())),
        _ => None,
    }
}

/// Write the user layer to `config.toml`. The top-level `calibration` table
/// (`hipfire calibrate`'s measured record, see [`calibration`]) is not part of
/// the layer, so it is carried over from the file being overwritten; an
/// unreadable or unparsable existing file contributes nothing.
pub fn write_global_toml(paths: &ConfigPaths, layer: &ConfigLayer) -> Result<()> {
    let existing = fs::read_to_string(&paths.config_toml)
        .ok()
        .and_then(|raw| toml::from_str::<toml::Table>(&raw).ok())
        .and_then(|mut root| root.remove(calibration::SECTION));
    write_global_with_calibration(paths, layer, existing)
}

/// Write `config.toml` as the user layer plus an optional `calibration` table.
pub(crate) fn write_global_with_calibration(
    paths: &ConfigPaths,
    layer: &ConfigLayer,
    table: Option<toml::Value>,
) -> Result<()> {
    let mut root = layer_toml_root(layer)?;
    if let Some(table) = table {
        root.insert(calibration::SECTION.to_owned(), table);
    }
    write_toml_root(&paths.config_toml, root)
}

fn write_layer_toml(path: &Path, layer: &ConfigLayer) -> Result<()> {
    write_toml_root(path, layer_toml_root(layer)?)
}

fn layer_toml_root(layer: &ConfigLayer) -> Result<toml::map::Map<String, toml::Value>> {
    layer.validate()?;
    let mut root = toml::map::Map::new();
    root.insert(
        "schema_version".to_owned(),
        toml::Value::Integer(CONFIG_SCHEMA_VERSION),
    );
    for (key, value) in &layer.values {
        let Some(value) = value.clone().into_toml() else {
            continue;
        };
        insert_toml_path(&mut root, key, value)?;
    }
    Ok(root)
}

fn write_toml_root(path: &Path, root: toml::map::Map<String, toml::Value>) -> Result<()> {
    let rendered =
        toml::to_string_pretty(&toml::Value::Table(root)).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            message: source.to_string(),
        })?;
    atomic_write(path, rendered.as_bytes())
}

fn insert_toml_path(
    table: &mut toml::map::Map<String, toml::Value>,
    dotted: &str,
    value: toml::Value,
) -> Result<()> {
    let mut parts = dotted.splitn(2, '.');
    let head = parts.next().expect("non-empty config key");
    let Some(tail) = parts.next() else {
        table.insert(head.to_owned(), value);
        return Ok(());
    };
    let entry = table
        .entry(head.to_owned())
        .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    let nested = entry
        .as_table_mut()
        .ok_or_else(|| ConfigError::InvalidValue {
            key: dotted.to_owned(),
            message: format!("{head} is already a scalar"),
        })?;
    insert_toml_path(nested, tail, value)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| ConfigError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|source| ConfigError::Write {
        path: tmp.clone(),
        source,
    })?;
    fs::rename(&tmp, path).map_err(|source| ConfigError::Write {
        path: path.to_owned(),
        source,
    })
}

fn read_string(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_owned(),
        source,
    })
}

/// Built-in configuration profile names selectable via `hipfire config profile set`.
///
/// Custom profiles live under `~/.hipfire/profiles/<name>.toml`. Profile names are
/// control-plane identifiers only; they are never persisted inside `config.toml`.
pub const CONFIG_PROFILE_NAMES: &[&str] = &["default", "dev", "hip", "redline"];

/// Origin of a selectable configuration profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigProfileKind {
    Builtin,
    Custom,
}

/// One built-in or on-disk custom profile entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigProfileEntry {
    pub name: String,
    pub kind: ConfigProfileKind,
    pub path: Option<PathBuf>,
}

/// Validate a custom profile name.
///
/// Accepts a single path segment of ASCII letters, digits, `.`, `_`, or `-`.
/// Rejects empty names, built-in names, path separators, and traversal markers.
pub fn validate_config_profile_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: "profile name must not be empty".to_owned(),
        });
    }
    if CONFIG_PROFILE_NAMES.contains(&name) {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!("'{name}' is a built-in profile name and cannot be overwritten"),
        });
    }
    if name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!("invalid profile name '{name}'"),
        });
    }
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
    {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!(
                "invalid profile name '{name}'; use ASCII letters, digits, '.', '_', or '-'"
            ),
        });
    }
    Ok(())
}

/// Path for a custom profile TOML file under `paths.profiles`.
pub fn custom_config_profile_path(paths: &ConfigPaths, name: &str) -> Result<PathBuf> {
    validate_config_profile_name(name)?;
    Ok(paths.profiles.join(format!("{name}.toml")))
}

/// Load the sparse layer for a built-in or custom profile.
pub fn load_config_profile(paths: &ConfigPaths, name: &str) -> Result<ConfigLayer> {
    if let Some(layer) = builtin_config_profile_layer(name) {
        return Ok(layer);
    }
    let path = custom_config_profile_path(paths, name)?;
    if !path.exists() {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!(
                "unknown profile '{name}'; expected one of {} or a custom profile in {}",
                CONFIG_PROFILE_NAMES.join(", "),
                paths.profiles.display()
            ),
        });
    }
    load_toml_layer(&path)
}

/// Merge the selected profile into `layer`.
///
/// Every key the profile sets is written to `layer`, overwriting that key's
/// previous value. Every key the profile does not set is left untouched, so
/// applying a profile never drops unrelated global overrides (devices, ports,
/// default model, kernel flags, ...). Returns the number of keys the profile
/// sets. On error `layer` is unchanged.
pub fn apply_config_profile(
    layer: &mut ConfigLayer,
    paths: &ConfigPaths,
    name: &str,
) -> Result<usize> {
    let profile = load_config_profile(paths, name)?;
    let count = profile.values.len();
    layer.values.extend(profile.values);
    Ok(count)
}

/// Key-level difference between two layers, in key order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConfigLayerDiff {
    /// Keys present only in the new layer: `(key, new)`.
    pub added: Vec<(String, ConfigValue)>,
    /// Keys present in both with different values: `(key, old, new)`.
    pub changed: Vec<(String, ConfigValue, ConfigValue)>,
    /// Keys present only in the old layer: `(key, old)`.
    pub removed: Vec<(String, ConfigValue)>,
}

impl ConfigLayerDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.changed.is_empty() && self.removed.is_empty()
    }

    pub fn len(&self) -> usize {
        self.added.len() + self.changed.len() + self.removed.len()
    }
}

/// Compute what changing `old` into `new` adds, overwrites, and removes.
pub fn diff_config_layers(old: &ConfigLayer, new: &ConfigLayer) -> ConfigLayerDiff {
    let mut diff = ConfigLayerDiff::default();
    for (key, value) in &new.values {
        match old.values.get(key) {
            None => diff.added.push((key.clone(), value.clone())),
            Some(previous) if previous != value => {
                diff.changed
                    .push((key.clone(), previous.clone(), value.clone()))
            }
            Some(_) => {}
        }
    }
    for (key, value) in &old.values {
        if !new.values.contains_key(key) {
            diff.removed.push((key.clone(), value.clone()));
        }
    }
    diff
}

/// Snapshot `layer` as a new custom profile. Fails when the name is a built-in,
/// invalid, or already present on disk.
pub fn create_config_profile(
    paths: &ConfigPaths,
    name: &str,
    layer: &ConfigLayer,
) -> Result<PathBuf> {
    let path = custom_config_profile_path(paths, name)?;
    if path.exists() {
        return Err(ConfigError::InvalidValue {
            key: "profile".to_owned(),
            message: format!("profile '{name}' already exists at {}", path.display()),
        });
    }
    write_layer_toml(&path, layer)?;
    Ok(path)
}

/// List built-in profiles followed by custom on-disk profiles (sorted).
pub fn list_config_profiles(paths: &ConfigPaths) -> Result<Vec<ConfigProfileEntry>> {
    let mut entries: Vec<ConfigProfileEntry> = CONFIG_PROFILE_NAMES
        .iter()
        .map(|name| ConfigProfileEntry {
            name: (*name).to_owned(),
            kind: ConfigProfileKind::Builtin,
            path: None,
        })
        .collect();
    if paths.profiles.is_dir() {
        let mut custom = Vec::new();
        for entry in fs::read_dir(&paths.profiles).map_err(|source| ConfigError::Read {
            path: paths.profiles.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| ConfigError::Read {
                path: paths.profiles.clone(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if validate_config_profile_name(stem).is_err() {
                continue;
            }
            custom.push(ConfigProfileEntry {
                name: stem.to_owned(),
                kind: ConfigProfileKind::Custom,
                path: Some(path),
            });
        }
        custom.sort_by(|left, right| left.name.cmp(&right.name));
        entries.extend(custom);
    }
    Ok(entries)
}

/// Detect which named profile, if any, is currently applied to `layer`.
///
/// Profiles merge into the global layer, so a profile counts as applied when
/// every key it sets is present and equal in `layer` (a subset match; extra
/// keys in `layer` are fine). Empty profiles never match. When several
/// profiles match, the one that sets the most keys wins; ties go to the
/// earlier entry (built-ins first, then custom profiles sorted by name).
pub fn detect_config_profile(paths: &ConfigPaths, layer: &ConfigLayer) -> Option<String> {
    let applied = |profile: &ConfigLayer| {
        !profile.values.is_empty()
            && profile
                .values
                .iter()
                .all(|(key, value)| layer.values.get(key) == Some(value))
    };
    let mut best: Option<(usize, String)> = None;
    let mut consider = |name: &str, profile: &ConfigLayer| {
        if !applied(profile) {
            return;
        }
        let size = profile.values.len();
        if best.as_ref().is_none_or(|(best_size, _)| size > *best_size) {
            best = Some((size, name.to_owned()));
        }
    };
    for name in CONFIG_PROFILE_NAMES {
        if let Some(bundle) = builtin_config_profile_layer(name) {
            consider(name, &bundle);
        }
    }
    if let Ok(entries) = list_config_profiles(paths) {
        for entry in entries {
            if entry.kind != ConfigProfileKind::Custom {
                continue;
            }
            if let Ok(candidate) = load_config_profile(paths, &entry.name) {
                consider(&entry.name, &candidate);
            }
        }
    }
    best.map(|(_, name)| name)
}

fn builtin_config_profile_layer(name: &str) -> Option<ConfigLayer> {
    let pairs = config_profile_bundle(name)?;
    let mut layer = ConfigLayer::default();
    for (key, value) in pairs {
        layer
            .set(key, value)
            .expect("built-in profile values must validate");
    }
    Some(layer)
}

/// Built-in profile contents.
///
/// The "default" bundle pins nothing that has a per-architecture, per-model, or
/// registry-card default. Rule: a key is excluded when its schema field is
/// `registry_allowed` (the registry recommendation may supply the value, e.g.
/// `memory.kv_cache` resolves per arch, `speculation.dflash` per registry
/// card). Pinning such a key here would shadow the card/arch default for every
/// model. Dropped for that reason: `generation.max_tokens`, `reasoning.mode`,
/// `reasoning.budget`, `reasoning.max_total_tokens`, `memory.kv_cache`,
/// `memory.max_seq`, `attention.flash`, `prompt.normalize`,
/// `prompt.default_chatml`, `speculation.mode`, `speculation.dflash`,
/// `speculation.mtp`, `speculation.mtp_k`, `speculation.ngram`. What remains is
/// process-level policy with one fixed default. Enforced by
/// `default_profile_pins_nothing_registry_or_arch_resolved`.
fn config_profile_bundle(name: &str) -> Option<Vec<(&'static str, ConfigValue)>> {
    Some(match name {
        "default" => vec![
            ("generation.loop_guard_threshold", ConfigValue::Integer(0)),
            ("generation.loop_guard_window", ConfigValue::Integer(256)),
            ("memory.prompt_cache_capacity", ConfigValue::Integer(32)),
            ("memory.prompt_cache_unbounded", ConfigValue::Bool(false)),
            ("serve.host", ConfigValue::String("127.0.0.1".to_owned())),
            ("serve.port", ConfigValue::Integer(11435)),
            ("serve.idle_timeout_seconds", ConfigValue::Integer(300)),
            ("serve.local", ConfigValue::Bool(false)),
            ("serve.multi_slot", ConfigValue::Bool(false)),
        ],
        "dev" => vec![
            ("hardware.devices", ConfigValue::String("0".to_owned())),
            ("hardware.allow_mixed_arch", ConfigValue::Bool(false)),
            ("hardware.tp_use_rccl", ConfigValue::Bool(true)),
            ("kernel.mmq", ConfigValue::String("auto".to_owned())),
            ("kernel.prefill_batched", ConfigValue::Bool(true)),
            ("kernel.lm_head_f16", ConfigValue::String("auto".to_owned())),
            (
                "experimental.graph.forward",
                ConfigValue::String("auto".to_owned()),
            ),
            ("experimental.graph.ar", ConfigValue::Bool(true)),
            ("experimental.graph.moe", ConfigValue::Bool(true)),
            ("diagnostic.prompt_token_heat", ConfigValue::Bool(false)),
            ("diagnostic.prompt_heat_json", ConfigValue::Bool(false)),
            ("diagnostic.prompt_heat_limit", ConfigValue::Integer(64)),
            (
                "diagnostic.kernel.gemv_rows",
                ConfigValue::String("4".to_owned()),
            ),
            (
                "diagnostic.kernel.gfx11_weight_load_policy",
                ConfigValue::String("buffer".to_owned()),
            ),
            ("developer.verify_graph", ConfigValue::Bool(false)),
            ("developer.dspark_profile", ConfigValue::Bool(true)),
        ],
        "hip" => vec![("replay.backend", ConfigValue::String("hip".to_owned()))],
        "redline" => vec![
            ("replay.backend", ConfigValue::String("redline".to_owned())),
            ("replay.transport", ConfigValue::String("pm4".to_owned())),
            (
                "replay.pm4_gfx11_vmem_acquire",
                ConfigValue::String("auto".to_owned()),
            ),
            ("diagnostic.replay.manual_capture", ConfigValue::Bool(false)),
            (
                "diagnostic.replay.pm4_min_parallel_width",
                ConfigValue::Integer(2),
            ),
            (
                "diagnostic.replay.pm4_min_parallel_workgroups",
                ConfigValue::Integer(0),
            ),
            (
                "diagnostic.replay.pm4_native_phases",
                ConfigValue::Bool(false),
            ),
            (
                "diagnostic.replay.pm4_queues",
                ConfigValue::String("1".to_owned()),
            ),
            (
                "diagnostic.replay.pm4_register_policy",
                ConfigValue::String("static".to_owned()),
            ),
            (
                "diagnostic.replay.pm4_wait_policy",
                ConfigValue::String("resource".to_owned()),
            ),
            (
                "diagnostic.replay.pm4_acquire_policy",
                ConfigValue::String("required-only".to_owned()),
            ),
            ("diagnostic.replay.pm4_gcr_trim", ConfigValue::Bool(true)),
            (
                "diagnostic.replay.pm4_dynamic_grid",
                ConfigValue::Bool(false),
            ),
            (
                "diagnostic.replay.gfx1151_initiator",
                ConfigValue::String("legacy".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_interleave",
                ConfigValue::String("inherit".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_resource_limits",
                ConfigValue::String("legacy".to_owned()),
            ),
            (
                "diagnostic.replay.gfx1151_entry_acquire",
                ConfigValue::String("system".to_owned()),
            ),
        ],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deprecated_key_covers_cask_and_pflash_families() {
        for key in [
            "memory.cask.enabled",
            "cask",
            "cask_sidecar",
            "speculation.prefill.mode",
            "prefill_compression",
            "prefill_drafter",
            "prefill_threshold",
            "diagnostic.pflash.score_layer",
            "pflash_score_layer",
        ] {
            assert!(is_deprecated_key(key), "{key} must be deprecated");
        }
        for key in [
            "kv_cache",
            "speculation.mode",
            "speculation",
            "ple_format",
            "memory.ple_format",
            "max_seq",
            "nonexistent.key",
        ] {
            assert!(!is_deprecated_key(key), "{key} must not be deprecated");
        }
        for field in fields() {
            let family = ["memory.cask.", "speculation.prefill.", "diagnostic.pflash."]
                .iter()
                .any(|prefix| field.key.starts_with(prefix));
            assert_eq!(is_deprecated_key(field.key), family, "{}", field.key);
            assert_eq!(
                is_deprecated_key(field.legacy_key),
                family,
                "{}",
                field.legacy_key
            );
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        env::temp_dir().join(format!("hipfire-config-{name}-{}", std::process::id()))
    }

    fn home_env<'a>(
        pairs: &'a [(&'a str, &'a str)],
    ) -> impl Fn(&str) -> Option<std::ffi::OsString> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| std::ffi::OsString::from(*value))
        }
    }

    #[test]
    fn unix_home_selection_is_unchanged() {
        let both = [("HIPFIRE_HOME", "/srv/hf"), ("HOME", "/home/u")];
        assert_eq!(
            hipfire_home_from(false, home_env(&both)).unwrap(),
            PathBuf::from("/srv/hf")
        );
        assert_eq!(
            hipfire_home_from(false, home_env(&[("HOME", "/home/u")])).unwrap(),
            PathBuf::from("/home/u/.hipfire")
        );
        // USERPROFILE is a Windows-only input.
        assert_eq!(
            hipfire_home_from(false, home_env(&[("USERPROFILE", "C:\\Users\\u")])).unwrap(),
            PathBuf::from(".hipfire")
        );
        assert_eq!(
            hipfire_home_from(false, home_env(&[])).unwrap(),
            PathBuf::from(".hipfire")
        );
    }

    #[test]
    fn windows_home_selection_order() {
        let all = [
            ("HIPFIRE_HOME", "D:\\hf"),
            ("USERPROFILE", "C:\\Users\\u"),
            ("HOME", "E:\\home"),
        ];
        assert_eq!(
            hipfire_home_from(true, home_env(&all)).unwrap(),
            PathBuf::from("D:\\hf")
        );
        let no_override = [("USERPROFILE", "C:\\Users\\u"), ("HOME", "E:\\home")];
        assert_eq!(
            hipfire_home_from(true, home_env(&no_override)).unwrap(),
            PathBuf::from("C:\\Users\\u").join(".hipfire")
        );
        assert_eq!(
            hipfire_home_from(true, home_env(&[("HOME", "E:\\home")])).unwrap(),
            PathBuf::from("E:\\home").join(".hipfire")
        );
        // An empty USERPROFILE falls through to HOME.
        let empty_profile = [("USERPROFILE", ""), ("HOME", "E:\\home")];
        assert_eq!(
            hipfire_home_from(true, home_env(&empty_profile)).unwrap(),
            PathBuf::from("E:\\home").join(".hipfire")
        );
    }

    #[test]
    fn windows_home_never_falls_back_to_relative() {
        let missing = hipfire_home_from(true, home_env(&[]))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("HIPFIRE_HOME") && missing.contains("USERPROFILE"));
        for bad in [
            ("HIPFIRE_HOME", ".hipfire"),
            ("HIPFIRE_HOME", ""),
            ("HIPFIRE_HOME", "\\rooted"),
            ("HIPFIRE_HOME", "C:drive-relative"),
            ("USERPROFILE", "Users\\u"),
            ("HOME", "/c/Users/u"),
        ] {
            let pairs = [bad];
            let error = hipfire_home_from(true, home_env(&pairs)).unwrap_err();
            assert!(
                error.to_string().contains(bad.0),
                "{bad:?} must name its variable: {error}"
            );
        }
        // A bad HIPFIRE_HOME or USERPROFILE does not silently fall through.
        let bad_override = [("HIPFIRE_HOME", "rel"), ("USERPROFILE", "C:\\Users\\u")];
        assert!(hipfire_home_from(true, home_env(&bad_override)).is_err());
        let bad_profile = [("USERPROFILE", "rel"), ("HOME", "E:\\home")];
        assert!(hipfire_home_from(true, home_env(&bad_profile)).is_err());
    }

    #[test]
    fn windows_absolute_classification() {
        for ok in [
            "C:\\",
            "c:/Users/u",
            "\\\\server\\share\\hf",
            "//server/share",
            "\\\\?\\C:\\Users\\u",
        ] {
            assert!(is_windows_absolute(Path::new(ok)), "{ok}");
        }
        for bad in [
            "", "C:", "C:foo", "\\foo", "/foo", "foo\\bar", "\\\\", "\\\\\\x",
        ] {
            assert!(!is_windows_absolute(Path::new(bad)), "{bad}");
        }
    }

    #[test]
    fn developer_bool_truth_table_is_strict() {
        // "1" is true and "0" is false regardless of the default; every
        // other spelling (including "true"/"on"/empty) falls back to it,
        // which is what makes the helper an exact replacement for both
        // `var == "1"` (default off) and `var != "0"` (default on).
        for default in [false, true] {
            assert!(parse_developer_bool(Some("1"), default));
            assert!(!parse_developer_bool(Some("0"), default));
            assert_eq!(parse_developer_bool(None, default), default);
            assert_eq!(parse_developer_bool(Some("true"), default), default);
            assert_eq!(parse_developer_bool(Some("on"), default), default);
            assert_eq!(parse_developer_bool(Some("yes"), default), default);
            assert_eq!(parse_developer_bool(Some(""), default), default);
            assert_eq!(parse_developer_bool(Some("2"), default), default);
        }
    }

    const SYM_IU4_KEY: &str = "developer.qwen4_moe_sym_iu4";
    const SYM_IU4_ENV: &str = "HIPFIRE_QWEN4_MOE_SYM_IU4";

    #[test]
    fn typed_bool_spellings_parse_and_everything_else_is_rejected() {
        for (raw, want) in [
            ("1", true),
            ("true", true),
            ("TRUE", true),
            ("True", true),
            ("0", false),
            ("false", false),
            ("FALSE", false),
            ("False", false),
        ] {
            assert_eq!(parse_typed_bool(raw), Some(want), "{raw}");
        }
        for bad in [
            "", " ", "2", "-1", "01", "00", "on", "off", "yes", "no", "t", "f", "maybe", " 1",
            "0 ", "truee", "null",
        ] {
            assert_eq!(parse_typed_bool(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn typed_bool_developer_key_set_cli_accepts_each_spelling() {
        assert!(is_typed_bool_developer_key(SYM_IU4_KEY));
        assert!(!is_typed_bool_developer_key("developer.experimental_mode"));
        for (raw, want) in [
            ("true", true),
            ("false", false),
            ("1", true),
            ("0", false),
            ("TRUE", true),
            ("False", false),
        ] {
            let mut layer = ConfigLayer::default();
            layer.set_cli(SYM_IU4_KEY, raw).unwrap();
            assert_eq!(
                layer.get(SYM_IU4_KEY),
                Some(&ConfigValue::Bool(want)),
                "{raw}"
            );
            layer.validate().unwrap();
        }
    }

    /// S1 slow-link prefill selector: `1` lab candidate, `0` incumbent kill
    /// switch, unset automatic. Typed so a stray spelling is a loud load
    /// error, never a silent unset.
    #[test]
    fn developer_slowlink_prefill_is_a_typed_tristate_key() {
        const KEY: &str = "developer.qwen4_slowlink_prefill";
        assert_eq!(
            developer_key_for_env("HIPFIRE_QWEN4_SLOWLINK_PREFILL").as_deref(),
            Some(KEY)
        );
        assert_eq!(
            developer_env_for_key(KEY).as_deref(),
            Some("HIPFIRE_QWEN4_SLOWLINK_PREFILL")
        );
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("False", false)] {
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)), "{raw}");
            layer.validate().unwrap();
            assert_eq!(typed_developer_bool_value(KEY, Some(raw)), Some(want));
        }
        assert_eq!(typed_developer_bool_value(KEY, None), None);
        for bad in ["", "2", "on", "off", "auto", "yes"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
    }

    #[test]
    fn typed_bool_developer_key_rejects_invalid_values_with_a_clear_message() {
        for bad in ["", "2", "on", "off", "yes", "no", "maybe", "-1", "10"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(SYM_IU4_KEY, bad).unwrap_err();
            let text = error.to_string();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == SYM_IU4_KEY),
                "{bad:?}: {text}"
            );
            assert!(text.contains("developer.qwen4_moe_sym_iu4"), "{text}");
            assert!(text.contains("true, false, 1 or 0"), "{text}");
            assert!(text.contains(&format!("'{bad}'")), "{text}");
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
        for bad in [
            ConfigValue::Integer(2),
            ConfigValue::Integer(-1),
            ConfigValue::Float(1.0),
            ConfigValue::Null,
            ConfigValue::String("on".into()),
        ] {
            let mut layer = ConfigLayer::default();
            assert!(layer.set(SYM_IU4_KEY, bad.clone()).is_err(), "{bad:?}");
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    fn typed_bool_developer_key_normalizes_every_input_to_the_legacy_one_zero() {
        for (value, want) in [
            (ConfigValue::Bool(true), "1"),
            (ConfigValue::Bool(false), "0"),
            (ConfigValue::Integer(1), "1"),
            (ConfigValue::Integer(0), "0"),
            (ConfigValue::String("true".into()), "1"),
            (ConfigValue::String("false".into()), "0"),
            (ConfigValue::String("1".into()), "1"),
            (ConfigValue::String("0".into()), "0"),
        ] {
            let mut layer = ConfigLayer::default();
            layer.set(SYM_IU4_KEY, value.clone()).unwrap();
            let process = ProcessConfig {
                schema_version: CONFIG_SCHEMA_VERSION,
                values: layer,
            };
            assert_eq!(
                process.legacy_value(SYM_IU4_ENV).as_deref(),
                Some(want),
                "{value:?}"
            );
        }
    }

    #[test]
    fn typed_bool_developer_key_is_revalidated_for_deserialized_layers() {
        let mut layer = ConfigLayer::default();
        for ok in ["true", "false", "1", "0"] {
            layer
                .values
                .insert(SYM_IU4_KEY.into(), ConfigValue::String(ok.into()));
            layer.validate().unwrap();
        }
        layer
            .values
            .insert(SYM_IU4_KEY.into(), ConfigValue::String("maybe".into()));
        assert!(layer.validate().is_err());
    }

    #[test]
    fn typed_bool_developer_key_toml_accepts_bool_and_string_spellings() {
        let root = temp_root("typed-bool-toml");
        fs::create_dir_all(&root).unwrap();
        for (literal, want) in [
            ("true", true),
            ("false", false),
            ("1", true),
            ("0", false),
            ("\"true\"", true),
            ("\"false\"", false),
            ("\"1\"", true),
            ("\"0\"", false),
        ] {
            let path = root.join("config.toml");
            fs::write(
                &path,
                format!("schema_version = {CONFIG_SCHEMA_VERSION}\n[developer]\nqwen4_moe_sym_iu4 = {literal}\n"),
            )
            .unwrap();
            let layer = load_toml_layer(&path).unwrap();
            assert_eq!(
                layer.get(SYM_IU4_KEY),
                Some(&ConfigValue::Bool(want)),
                "{literal}"
            );
        }
        for literal in ["2", "\"on\"", "\"\"", "1.5"] {
            let path = root.join("config.toml");
            fs::write(
                &path,
                format!("schema_version = {CONFIG_SCHEMA_VERSION}\n[developer]\nqwen4_moe_sym_iu4 = {literal}\n"),
            )
            .unwrap();
            let error = load_toml_layer(&path).unwrap_err().to_string();
            assert!(error.contains("true, false, 1 or 0"), "{literal}: {error}");
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn other_developer_keys_keep_their_free_form_spelling() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("developer.experimental_mode", "on").unwrap();
        layer.set_cli("developer.qwen4_moe_x_lds", "auto").unwrap();
        assert_eq!(
            layer.get("developer.experimental_mode"),
            Some(&ConfigValue::String("on".into()))
        );
        assert_eq!(
            layer.get("developer.qwen4_moe_x_lds"),
            Some(&ConfigValue::String("auto".into()))
        );
    }

    #[test]
    fn developer_typed_bool_reads_each_spelling_and_fails_loudly_on_others() {
        assert_eq!(typed_developer_bool_value(SYM_IU4_ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(SYM_IU4_ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
        }
    }

    #[test]
    fn qwen4_mtp_head_d2_is_a_typed_bool_developer_key_with_the_d2_default_spellings() {
        const KEY: &str = "developer.qwen4_mtp_head_d2";
        const ENV: &str = "HIPFIRE_QWEN4_MTP_HEAD_D2";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
        }
    }

    #[test]
    fn qwen4_sp_is_a_typed_bool_developer_key_unset_follows_the_arch_default() {
        const KEY: &str = "developer.qwen4_sp";
        const ENV: &str = "HIPFIRE_QWEN4_SP";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_eq!(developer_key_for_env(ENV).as_deref(), Some(KEY));
        // Unset is None (the consumer applies the gfx1100 default); each spelling reads as a bool.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)), "{raw}");
            layer.validate().unwrap();
        }
        // Garbage is rejected when the layer is built, never silently ignored.
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
    }

    #[test]
    #[should_panic(expected = "HIPFIRE_QWEN4_SP")]
    fn qwen4_sp_panics_on_an_unparseable_snapshot_value() {
        typed_developer_bool_value("HIPFIRE_QWEN4_SP", Some("garbage"));
    }

    #[test]
    fn qwen4_sp_touched_is_a_typed_bool_developer_key_unset_follows_the_arch_default() {
        const KEY: &str = "developer.qwen4_sp_touched";
        const ENV: &str = "HIPFIRE_QWEN4_SP_TOUCHED";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_eq!(developer_key_for_env(ENV).as_deref(), Some(KEY));
        assert_ne!(KEY, "developer.qwen4_sp");
        // Unset is None: the consumer applies the gfx1100 default (not a config-level on/off).
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)), "{raw}");
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
    }

    const MOE_SYM_IU4_KEY: &str = "developer.moe_sym_iu4";
    const MOE_SYM_IU4_ENV: &str = "HIPFIRE_MOE_SYM_IU4";

    #[test]
    fn moe_sym_iu4_is_a_typed_bool_developer_key_independent_of_the_flash_next_key() {
        assert!(is_developer_key(MOE_SYM_IU4_KEY));
        assert!(is_typed_bool_developer_key(MOE_SYM_IU4_KEY));
        assert_ne!(MOE_SYM_IU4_KEY, SYM_IU4_KEY);
        assert_ne!(MOE_SYM_IU4_ENV, SYM_IU4_ENV);
        // Unset is None (the route's default is OFF); each spelling reads as a bool.
        assert_eq!(typed_developer_bool_value(MOE_SYM_IU4_ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(MOE_SYM_IU4_ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(MOE_SYM_IU4_KEY, raw).unwrap();
            assert_eq!(
                layer.get(MOE_SYM_IU4_KEY),
                Some(&ConfigValue::Bool(want)),
                "{raw}"
            );
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(MOE_SYM_IU4_KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == MOE_SYM_IU4_KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
        // Setting one key never writes the other.
        let mut layer = ConfigLayer::default();
        layer.set_cli(MOE_SYM_IU4_KEY, "1").unwrap();
        assert_eq!(layer.get(SYM_IU4_KEY), None);
    }

    const GDN_QK_INDEXED_KEY: &str = "developer.gdn_qk_indexed";
    const GDN_QK_INDEXED_ENV: &str = "HIPFIRE_GDN_QK_INDEXED";

    #[test]
    fn gdn_qk_indexed_is_a_typed_bool_developer_key_with_its_own_env_name() {
        assert!(is_developer_key(GDN_QK_INDEXED_KEY));
        assert!(is_typed_bool_developer_key(GDN_QK_INDEXED_KEY));
        assert_eq!(
            developer_key_for_env(GDN_QK_INDEXED_ENV).as_deref(),
            Some(GDN_QK_INDEXED_KEY)
        );
        // Unset is None (the typed read; the route's unset default is the
        // `prefill_route_capabilities` qualification); each spelling reads as
        // a bool, `0`/`false` being the kill switch.
        assert_eq!(typed_developer_bool_value(GDN_QK_INDEXED_ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(GDN_QK_INDEXED_ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(GDN_QK_INDEXED_KEY, raw).unwrap();
            assert_eq!(
                layer.get(GDN_QK_INDEXED_KEY),
                Some(&ConfigValue::Bool(want)),
                "{raw}"
            );
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(GDN_QK_INDEXED_KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == GDN_QK_INDEXED_KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
    }

    const GDN_MOE_PRE_FUSED_KEY: &str = "developer.gdn_moe_pre_fused";
    const GDN_MOE_PRE_FUSED_ENV: &str = "HIPFIRE_GDN_MOE_PRE_FUSED";

    #[test]
    fn gdn_moe_pre_fused_is_a_typed_bool_developer_key_with_its_own_env_name() {
        assert!(is_developer_key(GDN_MOE_PRE_FUSED_KEY));
        assert!(is_typed_bool_developer_key(GDN_MOE_PRE_FUSED_KEY));
        assert_eq!(
            developer_key_for_env(GDN_MOE_PRE_FUSED_ENV).as_deref(),
            Some(GDN_MOE_PRE_FUSED_KEY)
        );
        // Distinct from the indexed flag.
        assert_ne!(GDN_MOE_PRE_FUSED_KEY, GDN_QK_INDEXED_KEY);
        assert_ne!(GDN_MOE_PRE_FUSED_ENV, GDN_QK_INDEXED_ENV);
        // Unset is None (the typed read; the route's unset default is the
        // `prefill_route_capabilities` qualification); each spelling reads as
        // a bool, `0`/`false` being the kill switch.
        assert_eq!(
            typed_developer_bool_value(GDN_MOE_PRE_FUSED_ENV, None),
            None
        );
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(GDN_MOE_PRE_FUSED_ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(GDN_MOE_PRE_FUSED_KEY, raw).unwrap();
            assert_eq!(
                layer.get(GDN_MOE_PRE_FUSED_KEY),
                Some(&ConfigValue::Bool(want)),
                "{raw}"
            );
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(GDN_MOE_PRE_FUSED_KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == GDN_MOE_PRE_FUSED_KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
        // Setting one key never writes the other.
        let mut layer = ConfigLayer::default();
        layer.set_cli(GDN_MOE_PRE_FUSED_KEY, "1").unwrap();
        assert_eq!(layer.get(GDN_QK_INDEXED_KEY), None);
    }

    const GDN_MOE_CHUNK_SCAN_KEY: &str = "developer.gdn_moe_chunk_scan";
    const GDN_MOE_CHUNK_SCAN_ENV: &str = "HIPFIRE_GDN_MOE_CHUNK_SCAN";

    #[test]
    fn gdn_moe_chunk_scan_is_a_typed_bool_developer_key_with_its_own_env_name() {
        assert!(is_developer_key(GDN_MOE_CHUNK_SCAN_KEY));
        assert!(is_typed_bool_developer_key(GDN_MOE_CHUNK_SCAN_KEY));
        assert_eq!(
            developer_key_for_env(GDN_MOE_CHUNK_SCAN_ENV).as_deref(),
            Some(GDN_MOE_CHUNK_SCAN_KEY)
        );
        // Distinct from the two MoE GDN preamble flags it supersedes.
        assert_ne!(GDN_MOE_CHUNK_SCAN_KEY, GDN_MOE_PRE_FUSED_KEY);
        assert_ne!(GDN_MOE_CHUNK_SCAN_KEY, GDN_QK_INDEXED_KEY);
        assert_ne!(GDN_MOE_CHUNK_SCAN_ENV, GDN_MOE_PRE_FUSED_ENV);
        assert_ne!(GDN_MOE_CHUNK_SCAN_ENV, GDN_QK_INDEXED_ENV);
        // Unset is None (the typed read; the route's unset default is the
        // `prefill_route_capabilities` qualification); each spelling reads as
        // a bool, `0`/`false` being the kill switch.
        assert_eq!(
            typed_developer_bool_value(GDN_MOE_CHUNK_SCAN_ENV, None),
            None
        );
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(GDN_MOE_CHUNK_SCAN_ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(GDN_MOE_CHUNK_SCAN_KEY, raw).unwrap();
            assert_eq!(
                layer.get(GDN_MOE_CHUNK_SCAN_KEY),
                Some(&ConfigValue::Bool(want)),
                "{raw}"
            );
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(GDN_MOE_CHUNK_SCAN_KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == GDN_MOE_CHUNK_SCAN_KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
        // Setting the chunk-scan key never writes the flags it supersedes.
        let mut layer = ConfigLayer::default();
        layer.set_cli(GDN_MOE_CHUNK_SCAN_KEY, "1").unwrap();
        assert_eq!(layer.get(GDN_MOE_PRE_FUSED_KEY), None);
        assert_eq!(layer.get(GDN_QK_INDEXED_KEY), None);
    }

    #[test]
    fn moe_router_fused_prefill_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.moe_router_fused_prefill";
        const ENV: &str = "HIPFIRE_MOE_ROUTER_FUSED_PREFILL";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // Unset is None (the typed read; the route's unset default is the
        // `prefill_route_capabilities` qualification); each spelling reads as
        // a bool, `0`/`false` being the kill switch.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [
            ("1", true),
            ("true", true),
            ("TRUE", true),
            ("0", false),
            ("false", false),
            ("False", false),
        ] {
            assert_eq!(
                typed_developer_bool_value(ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)), "{raw}");
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
        // Independent of the other MoE typed keys.
        let mut layer = ConfigLayer::default();
        layer.set_cli(KEY, "1").unwrap();
        assert_eq!(layer.get("developer.moe_sym_iu4"), None);
        assert_eq!(layer.get("developer.qwen4_moe_sym_iu4"), None);
    }

    const MOE_SHARED_PREPARED_IU4_KEY: &str = "developer.moe_shared_prepared_iu4";
    const MOE_SHARED_PREPARED_IU4_ENV: &str = "HIPFIRE_MOE_SHARED_PREPARED_IU4";

    #[test]
    fn moe_shared_prepared_iu4_is_a_typed_bool_developer_key() {
        assert!(is_developer_key(MOE_SHARED_PREPARED_IU4_KEY));
        assert!(is_typed_bool_developer_key(MOE_SHARED_PREPARED_IU4_KEY));
        // Independent of the symmetric IU4 keys.
        assert_ne!(MOE_SHARED_PREPARED_IU4_KEY, MOE_SYM_IU4_KEY);
        assert_ne!(MOE_SHARED_PREPARED_IU4_KEY, SYM_IU4_KEY);
        // Unset is None (the typed read; the route's unset default is the
        // `prefill_route_capabilities` qualification); each spelling reads as
        // a bool, `0`/`false` being the kill switch.
        assert_eq!(
            typed_developer_bool_value(MOE_SHARED_PREPARED_IU4_ENV, None),
            None
        );
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(
                typed_developer_bool_value(MOE_SHARED_PREPARED_IU4_ENV, Some(raw)),
                Some(want),
                "{raw}"
            );
            let mut layer = ConfigLayer::default();
            layer.set_cli(MOE_SHARED_PREPARED_IU4_KEY, raw).unwrap();
            assert_eq!(
                layer.get(MOE_SHARED_PREPARED_IU4_KEY),
                Some(&ConfigValue::Bool(want)),
                "{raw}"
            );
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            let error = layer.set_cli(MOE_SHARED_PREPARED_IU4_KEY, bad).unwrap_err();
            assert!(
                matches!(&error, ConfigError::InvalidValue { key, .. } if key == MOE_SHARED_PREPARED_IU4_KEY),
                "{bad:?}: {error}"
            );
            assert!(layer.values.is_empty(), "{bad:?} must not be stored");
        }
        // Setting it never writes the symmetric IU4 keys.
        let mut layer = ConfigLayer::default();
        layer.set_cli(MOE_SHARED_PREPARED_IU4_KEY, "1").unwrap();
        assert_eq!(layer.get(MOE_SYM_IU4_KEY), None);
        assert_eq!(layer.get(SYM_IU4_KEY), None);
    }

    #[test]
    fn fa_prefill_pm_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.fa_prefill_pm";
        const ENV: &str = "HIPFIRE_FA_PREFILL_PM";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    fn moe_combine_v2_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.moe_combine_v2";
        const ENV: &str = "HIPFIRE_MOE_COMBINE_V2";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_ne!(KEY, "developer.moe_sym_iu4");
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }
    #[test]
    fn moe_router_topk_multirow_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.moe_router_topk_multirow";
        const ENV: &str = "HIPFIRE_MOE_ROUTER_TOPK_MULTIROW";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // Distinct from the fused-prefill key it nests under.
        assert_ne!(KEY, "developer.moe_router_fused_prefill");
        // The raw typed lookup of an unset variable is None (the architecture default is
        // resolved later by qualification, not a schema default); explicit `1`/`true` opts in,
        // `0`/`false` kills.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
        // Setting it never writes the fused-prefill key.
        let mut layer = ConfigLayer::default();
        layer.set_cli(KEY, "1").unwrap();
        assert_eq!(layer.get("developer.moe_router_fused_prefill"), None);
    }

    #[test]
    fn moe_iu4_wide_tiles_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.moe_iu4_wide_tiles";
        const ENV: &str = "HIPFIRE_MOE_IU4_WIDE_TILES";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // Independent of the symmetric IU4 route keys and of the FA PM key.
        for other in [MOE_SYM_IU4_KEY, SYM_IU4_KEY, MOE_SHARED_PREPARED_IU4_KEY] {
            assert_ne!(KEY, other);
        }
        assert_ne!(ENV, MOE_SYM_IU4_ENV);
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
        // Setting it never writes the route's own keys.
        let mut layer = ConfigLayer::default();
        layer.set_cli(KEY, "1").unwrap();
        assert_eq!(layer.get(MOE_SYM_IU4_KEY), None);
        assert_eq!(layer.get(SYM_IU4_KEY), None);
    }

    #[test]
    fn prefill_iu4_producer_fused_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.prefill_iu4_producer_fused";
        const ENV: &str = "HIPFIRE_PREFILL_IU4_PRODUCER_FUSED";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // The raw typed lookup of an unset variable is None: qualification resolves the
        // architecture default later, never a schema default.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    fn fa_prefill_pm_qprep_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.fa_prefill_pm_qprep";
        const ENV: &str = "HIPFIRE_FA_PREFILL_PM_QPREP";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    fn moe_rotate_i4_pm_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.moe_rotate_i4_pm";
        const ENV: &str = "HIPFIRE_MOE_ROTATE_I4_PM";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    fn prefill_iu4_c8_pm_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.prefill_iu4_c8_pm";
        const ENV: &str = "HIPFIRE_PREFILL_IU4_C8_PM";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // Unset is None: the capability table (raw inventory, no qualified row) resolves it
        // off; the schema never supplies a default.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
        // Independent of the F3 flag it nests under at the consumer.
        let mut layer = ConfigLayer::default();
        layer.set_cli(KEY, "1").unwrap();
        assert_eq!(layer.get("developer.prefill_iu4_producer_fused"), None);
    }

    #[test]
    fn gdn_chunk_prep_pm_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.gdn_chunk_prep_pm";
        const ENV: &str = "HIPFIRE_GDN_CHUNK_PREP_PM";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // Unset is None: the consumer resolves it with `PrefillRoute::GdnChunkPrepPm`, which
        // has no qualified row, so unset is off.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    fn moe_shared_down_sigmoid_epi_is_a_typed_bool_developer_key() {
        const KEY: &str = "developer.moe_shared_down_sigmoid_epi";
        const ENV: &str = "HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI";
        assert!(is_developer_key(KEY));
        assert!(is_typed_bool_developer_key(KEY));
        // Unset is None and no schema default exists: the capability table has no
        // qualified row, so the route is off everywhere unless explicitly `1`.
        assert_eq!(typed_developer_bool_value(ENV, None), None);
        assert_eq!(developer_env_for_key(KEY).as_deref(), Some(ENV));
        for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
            assert_eq!(typed_developer_bool_value(ENV, Some(raw)), Some(want));
            let mut layer = ConfigLayer::default();
            layer.set_cli(KEY, raw).unwrap();
            assert_eq!(layer.get(KEY), Some(&ConfigValue::Bool(want)));
            layer.validate().unwrap();
        }
        for bad in ["", "2", "on", "off", "yes", "maybe"] {
            let mut layer = ConfigLayer::default();
            assert!(matches!(
                layer.set_cli(KEY, bad),
                Err(ConfigError::InvalidValue { key, .. }) if key == KEY
            ));
            assert!(layer.values.is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "is not a boolean: expected true, false, 1 or 0")]
    fn developer_typed_bool_panics_on_an_unparseable_snapshot_value() {
        typed_developer_bool_value(SYM_IU4_ENV, Some("2"));
    }

    #[test]
    fn iu4_prefill_is_process_wide_and_rejected_in_a_model_layer() {
        let mut overrides = ConfigLayer::default();
        overrides
            .set("kernel.iu4_prefill", ConfigValue::Bool(false))
            .unwrap();
        let error = validate_model_layer(&overrides).unwrap_err().to_string();
        assert!(error.contains("kernel.iu4_prefill"), "{error}");
        assert!(
            error.contains("not valid in a per-model override"),
            "{error}"
        );
    }

    #[test]
    fn model_override_allowed_applies_the_per_model_key_rule() {
        assert!(model_override_allowed("reasoning.effort").is_ok());
        let process = model_override_allowed("kernel.iu4_prefill")
            .unwrap_err()
            .to_string();
        assert!(process.contains("kernel.iu4_prefill"), "{process}");
        assert!(
            process.contains("not valid in a per-model override"),
            "{process}"
        );
        let developer = model_override_allowed("developer.qwen4_moe_sym_iu4")
            .unwrap_err()
            .to_string();
        assert!(
            developer
                .contains("developer fields are global process policy and not valid per-model"),
            "{developer}"
        );
        assert!(matches!(
            model_override_allowed("no.such.key"),
            Err(ConfigError::UnknownKey(_))
        ));
    }

    #[test]
    fn set_and_remove_model_override_round_trip() {
        let root = temp_root("model-override-round-trip");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);

        let saved = set_model_override(
            &paths,
            Some("qwen3:8b"),
            Some(PathBuf::from("/models/qwen3-8b.mq4")),
            "qwen3:8b",
            "reasoning.effort",
            "high",
        )
        .unwrap();
        assert_eq!(saved.id, "qwen3:8b");
        assert_eq!(saved.key, "reasoning.effort");
        assert_eq!(saved.value, ConfigValue::String("high".into()));
        assert!(!saved.migrated_legacy_json);

        let loaded = load_catalog(&paths).unwrap();
        let record = &loaded.catalog.models["qwen3:8b"];
        assert_eq!(record.registry_tag.as_deref(), Some("qwen3:8b"));
        assert_eq!(
            record.path.as_deref(),
            Some(Path::new("/models/qwen3-8b.mq4"))
        );
        assert_eq!(
            record.overrides.get("reasoning.effort"),
            Some(&ConfigValue::String("high".into()))
        );

        assert_eq!(
            remove_model_override(&paths, "qwen3:8b", Some("reasoning.effort")).unwrap(),
            RemoveModelOverride::KeyRemoved {
                id: "qwen3:8b".into(),
                key: "reasoning.effort"
            }
        );
        assert_eq!(
            remove_model_override(&paths, "qwen3:8b", Some("reasoning.effort")).unwrap(),
            RemoveModelOverride::KeyInherited {
                id: "qwen3:8b".into(),
                key: "reasoning.effort"
            }
        );
        set_model_override(&paths, None, None, "qwen3:8b", "reasoning.effort", "low").unwrap();
        assert_eq!(
            remove_model_override(&paths, "qwen3:8b", None).unwrap(),
            RemoveModelOverride::AllRemoved {
                id: "qwen3:8b".into()
            }
        );
        assert!(load_catalog(&paths).unwrap().catalog.models["qwen3:8b"]
            .overrides
            .values
            .is_empty());
        assert_eq!(
            remove_model_override(&paths, "never-seen", None).unwrap(),
            RemoveModelOverride::NoOverrides
        );
        assert!(matches!(
            remove_model_override(&paths, "qwen3:8b", Some("no.such.key")),
            Err(ConfigError::UnknownKey(_))
        ));

        // A process-scoped key is rejected when the catalog is written.
        assert!(
            set_model_override(&paths, None, None, "qwen3:8b", "kernel.iu4_prefill", "0").is_err()
        );
        fs::remove_dir_all(&root).unwrap();
    }

    /// The cached lookup behind `process_value` answers exactly what the
    /// per-call `legacy_value` scan does, for every compatibility name.
    #[test]
    fn legacy_table_matches_legacy_value() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut values = ConfigLayer::default();
        for schema in FIELDS.iter().filter(|schema| schema.env_compat.is_some()) {
            if let Some(resolved_value) = resolved.get(schema.key) {
                values
                    .values
                    .insert(schema.key.to_owned(), resolved_value.value.clone());
            }
        }
        values.values.insert(
            "developer.dspark_q8_wmma".into(),
            ConfigValue::String("1".into()),
        );
        values
            .values
            .insert("developer.ngram_window".into(), ConfigValue::Integer(64));
        let config = ProcessConfig {
            schema_version: CONFIG_SCHEMA_VERSION,
            values,
        };
        let table = legacy_table(&config);
        let mut names: Vec<&str> = FIELDS
            .iter()
            .filter_map(|schema| schema.env_compat)
            .collect();
        names.extend([
            "HIPFIRE_DSPARK_Q8_WMMA",
            "HIPFIRE_NGRAM_WINDOW",
            "NOT_HIPFIRE",
        ]);
        let mut answered = 0;
        for name in names {
            let cached = table.get(name).cloned();
            assert_eq!(cached, config.legacy_value(name), "{name}");
            answered += usize::from(cached.is_some());
        }
        assert!(answered > 2, "the fixture must exercise rendered values");
        // An unset developer-shaped name (built at runtime so the env-docs
        // inventory does not read it as a real knob).
        let unset = ["HIPFIRE", "UNSET", "PROBE"].join("_");
        assert_eq!(table.get(&unset).cloned(), config.legacy_value(&unset));
    }

    #[test]
    fn ngram_schema_accepts_exactly_the_tri_state_alias_spellings() {
        let field = field("speculation.ngram").expect("ngram schema field");
        for raw in ["auto", "on", "off", "1", "0", "true", "false"] {
            assert!(
                field.validate(&ConfigValue::String(raw.into())).is_ok(),
                "{raw}"
            );
            assert!(parse_tri_state_alias(raw).is_ok(), "{raw}");
        }
        for raw in ["", "maybe", "yes", "ON", "2"] {
            assert!(
                field.validate(&ConfigValue::String(raw.into())).is_err(),
                "{raw}"
            );
        }
    }

    #[test]
    fn tri_state_alias_accepts_on_off_auto_spellings_and_rejects_the_rest() {
        for on in ["on", "1", "true"] {
            assert_eq!(parse_tri_state_alias(on), Ok(Some(true)), "{on:?}");
        }
        for off in ["off", "0", "false"] {
            assert_eq!(parse_tri_state_alias(off), Ok(Some(false)), "{off:?}");
        }
        assert_eq!(parse_tri_state_alias("auto"), Ok(None));
        for bad in [
            "", " ", "ON", "Off", "yes", "no", "2", "-1", "01", "1 ", "maybe",
        ] {
            let error = parse_tri_state_alias(bad).unwrap_err();
            assert!(
                error.contains("expected on, 1, true, off, 0, false or auto"),
                "{bad:?}: {error}"
            );
        }
    }
    #[test]
    fn tri_state_truth_table() {
        // Only the exact lower-case modes force; everything else, including
        // "auto" and the boolean spellings other parsers accept, defers.
        assert_eq!(parse_tri_state("on"), Some(true));
        assert_eq!(parse_tri_state("off"), Some(false));
        for deferred in ["auto", "", "ON", "Off", "1", "0", "true", "false", "yes"] {
            assert_eq!(parse_tri_state(deferred), None, "{deferred:?}");
        }
    }
    #[test]
    fn mtp_mode_parse_preserves_tri_state() {
        // The daemon's slots parse must keep auto distinct from on: absent
        // normalizes to "off" at the call site (default Off), while an
        // explicit empty string matches the mtp_k rule (off, not auto).
        assert_eq!(MtpMode::parse("off"), MtpMode::Off);
        assert_eq!(MtpMode::parse(""), MtpMode::Off);
        assert_eq!(MtpMode::parse("auto"), MtpMode::Auto);
        assert_eq!(MtpMode::parse("on"), MtpMode::On);
        assert_eq!(MtpMode::parse("anything-else"), MtpMode::Auto);
        assert_eq!(MtpMode::default(), MtpMode::Off);
    }

    #[test]
    fn mtp_mode_from_tri_state_matches_the_parse_spelling() {
        assert_eq!(MtpMode::from_tri_state(Some(true)), MtpMode::parse("on"));
        assert_eq!(MtpMode::from_tri_state(Some(false)), MtpMode::parse("off"));
        assert_eq!(MtpMode::from_tri_state(None), MtpMode::parse("auto"));
    }

    #[test]
    fn mtp_mode_admits_truth_table_on_off_auto_by_declared_floor() {
        // (mode, producer declared a measured AR floor) -> admitted.
        for floor in [false, true] {
            assert!(!mtp_mode_admits(MtpMode::Off, floor), "off never admits");
            assert!(
                mtp_mode_admits(MtpMode::On, floor),
                "explicit on always admits"
            );
        }
        assert!(
            !mtp_mode_admits(MtpMode::Auto, false),
            "auto without a measured floor stays AR"
        );
        assert!(
            mtp_mode_admits(MtpMode::Auto, true),
            "auto with a measured floor admits"
        );
    }

    #[test]
    fn mtp_effective_mode_name_resolves_auto_by_declared_floor() {
        assert_eq!(mtp_effective_mode_name("auto", false), "off");
        assert_eq!(mtp_effective_mode_name("auto", true), "auto");
        assert_eq!(mtp_effective_mode_name("on", false), "on");
        assert_eq!(mtp_effective_mode_name("on", true), "on");
        for floor in [false, true] {
            assert_eq!(mtp_effective_mode_name("off", floor), "off");
            assert_eq!(mtp_effective_mode_name("", floor), "off");
        }
        // An unrecognized spelling defers to auto, so it follows the floor.
        assert_eq!(mtp_effective_mode_name("bogus", false), "off");
        assert_eq!(mtp_effective_mode_name("bogus", true), "auto");
    }

    #[test]
    fn mtp_effective_mode_and_name_round_trip() {
        for mode in [MtpMode::Off, MtpMode::Auto, MtpMode::On] {
            assert_eq!(MtpMode::parse(mode.name()), mode);
            for floor in [false, true] {
                let effective = mtp_effective_mode(mode, floor);
                // The effective mode admits exactly what the gate admits.
                assert_eq!(effective != MtpMode::Off, mtp_mode_admits(mode, floor));
                assert_eq!(
                    mtp_effective_mode_name(mode.name(), floor),
                    effective.name()
                );
            }
        }
        assert_eq!(mtp_effective_mode(MtpMode::Auto, false), MtpMode::Off);
        assert_eq!(mtp_effective_mode(MtpMode::Auto, true), MtpMode::Auto);
        assert_eq!(mtp_effective_mode(MtpMode::On, false), MtpMode::On);
    }

    #[test]
    fn mtp_floor_gate_composes_with_the_host_spill_policy() {
        // The floor verdict gates first; the spill policy only narrows an
        // admitted mode. Auto + floor + fully resident arms; auto + floor +
        // spill on a non-admitted arch declines; explicit on arms over both.
        let arms = |mode: MtpMode, floor: bool, blobs: usize, arch: &str, admitted: &[&str]| {
            mtp_mode_admits(mode, floor)
                && mtp_host_spill_policy(mode, blobs, arch, admitted).attach
        };
        assert!(arms(MtpMode::Auto, true, 0, "gfx1100", &[]));
        assert!(!arms(MtpMode::Auto, true, 8, "gfx1100", &[]));
        assert!(arms(MtpMode::Auto, true, 8, "gfx1201", &["gfx1201"]));
        assert!(!arms(MtpMode::Auto, false, 0, "gfx1201", &["gfx1201"]));
        assert!(arms(MtpMode::On, false, 8, "gfx1100", &[]));
        assert!(!arms(MtpMode::Off, true, 0, "gfx1201", &["gfx1201"]));
    }

    #[test]
    fn mtp_host_spill_policy_truth_table() {
        // (mode, blobs, arch, admitted) -> attach. qwen35 passes &[] (never
        // attaches on a spill under auto); qwen4 passes &["gfx1100", "gfx1201"].
        let qwen35: &[&str] = &[];
        let qwen4: &[&str] = &["gfx1100", "gfx1201"];
        // Fully resident: attaches unless explicitly off.
        assert!(mtp_host_spill_policy(MtpMode::Auto, 0, "gfx1100", qwen35).attach);
        assert!(mtp_host_spill_policy(MtpMode::On, 0, "gfx1100", qwen35).attach);
        assert!(!mtp_host_spill_policy(MtpMode::Off, 0, "gfx1100", qwen35).attach);
        // Off never attaches, even on a spill.
        assert!(!mtp_host_spill_policy(MtpMode::Off, 21, "gfx1201", qwen4).attach);
        assert!(!mtp_host_spill_policy(MtpMode::Off, 21, "gfx1100", qwen35).attach);
        // On always attaches, even on a spill on an unmeasured arch.
        assert!(mtp_host_spill_policy(MtpMode::On, 21, "gfx1100", qwen35).attach);
        assert!(mtp_host_spill_policy(MtpMode::On, 21, "gfx1030", qwen4).attach);
        // qwen35 under auto: never attaches on a spill, on any arch.
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx1030"] {
            let policy = mtp_host_spill_policy(MtpMode::Auto, 21, arch, qwen35);
            assert!(!policy.attach, "qwen35 auto on {arch}");
            assert_eq!(policy.reason, mtp_host_spill_reason());
        }
        // qwen4 under auto: attaches only on the admitted arch.
        assert!(mtp_host_spill_policy(MtpMode::Auto, 21, "gfx1201", qwen4).attach);
        assert!(mtp_host_spill_policy(MtpMode::Auto, 21, "gfx1100", qwen4).attach);
        assert!(!mtp_host_spill_policy(MtpMode::Auto, 21, "gfx1151", qwen4).attach);
        // Attached verdicts carry no reason; the decline line words it once.
        assert_eq!(
            mtp_host_spill_policy(MtpMode::On, 21, "gfx1100", qwen35).reason,
            ""
        );
        assert_eq!(
            mtp_host_spill_policy(MtpMode::Auto, 0, "gfx1100", qwen35).reason,
            ""
        );
        let line = mtp_host_spill_decline_line("qwen35", 21, 3 << 30, "offload_exec=pcie");
        assert!(line.contains(mtp_host_spill_reason()), "{line}");
        assert!(line.contains("21 expert blobs"), "{line}");
        assert!(line.contains("offload_exec=pcie"), "{line}");
    }

    #[test]
    fn developer_bool_defaults_when_unset() {
        // A probe name nothing sets must resolve to the caller's default
        // through the live snapshot (no install, no TOML, no env).
        assert!(!developer_bool("HIPFIRE_S4_FLAG_PROBE_UNSET_OFF", false));
        assert!(developer_bool("HIPFIRE_S4_FLAG_PROBE_UNSET_ON", true));
    }

    #[test]
    fn mtp_ngram_key_resolves_tri_state_with_env_override() {
        // Legacy no-arch resolver: explicit on/1 only.
        for (raw, want) in [
            (None, false),
            (Some("off"), false),
            (Some("0"), false),
            (Some("on"), true),
            (Some("1"), true),
            (Some("auto"), false),
        ] {
            assert_eq!(mtp_ngram_enabled_for(raw), want, "{raw:?}");
        }
        let field = field("speculation.mtp_ngram").expect("mtp_ngram schema field");
        assert_eq!(field.env_compat, Some("HIPFIRE_MTP_NGRAM"));
        assert!(matches!(field.default, DefaultValue::String("auto")));
        assert!(field.validate(&ConfigValue::String("1".into())).is_ok());
        assert!(field.validate(&ConfigValue::String("auto".into())).is_ok());
        assert!(field.validate(&ConfigValue::String("yes".into())).is_err());
    }

    #[test]
    fn mtp_ngram_arch_resolver_table() {
        let cases: [(u32, &str); 4] = [
            (16, "gfx1151"),
            (16, "gfx1201"),
            (16, "gfx1100"),
            (5, "gfx1151"),
        ];
        // auto/unset: only arch 16 on the two admitted GPUs.
        let auto = [true, true, false, false];
        for (raw, expect) in [
            (Some("on"), [true; 4]),
            (Some("1"), [true; 4]),
            (Some("off"), [false; 4]),
            (Some("0"), [false; 4]),
            (Some("auto"), auto),
            (None, auto),
        ] {
            for (i, (model_arch, gpu)) in cases.iter().enumerate() {
                assert_eq!(
                    mtp_ngram_enabled_for_arch_value(raw, *model_arch, gpu),
                    expect[i],
                    "{raw:?} arch={model_arch} gpu={gpu}"
                );
            }
        }
    }

    #[test]
    fn ngram_mod_triple_defaults_per_arch_with_env_override() {
        assert_eq!(ngram_mod_default_triple(16, "gfx1151"), (5, 3, 3));
        assert_eq!(ngram_mod_default_triple(16, "gfx1201"), (5, 3, 3));
        assert_eq!(ngram_mod_default_triple(16, "gfx1100"), (24, 48, 64));
        assert_eq!(ngram_mod_default_triple(5, "gfx1151"), (24, 48, 64));

        let none = |_: &str| None;
        assert_eq!(ngram_mod_triple_resolve((5, 3, 3), none), (5, 3, 3));
        assert_eq!(ngram_mod_triple_resolve((24, 48, 64), none), (24, 48, 64));
        // Overrides win per component; unparseable values fall back to the default.
        let env = |name: &str| match name {
            "HIPFIRE_NGRAM_MOD_N_MATCH" => Some("7".to_string()),
            "HIPFIRE_NGRAM_MOD_N_MAX" => Some("junk".to_string()),
            _ => None,
        };
        assert_eq!(ngram_mod_triple_resolve((5, 3, 3), env), (7, 3, 3));
        assert_eq!(ngram_mod_triple_resolve((24, 48, 64), env), (7, 48, 64));
    }

    #[test]
    fn mtp_ngram_copy_key_and_arch_resolver() {
        let field = field("speculation.mtp_ngram_copy").expect("mtp_ngram_copy schema field");
        assert_eq!(field.env_compat, Some("HIPFIRE_MTP_NGRAM_COPY"));
        assert!(matches!(field.default, DefaultValue::String("auto")));
        for raw in ["auto", "on", "off", "1", "0"] {
            assert!(
                field.validate(&ConfigValue::String(raw.into())).is_ok(),
                "{raw}"
            );
        }
        assert!(field.validate(&ConfigValue::String("yes".into())).is_err());
        let cases: [(u32, &str); 4] = [
            (16, "gfx1151"),
            (16, "gfx1201"),
            (16, "gfx1100"),
            (5, "gfx1151"),
        ];
        // auto/unset: native Qwen4 MTP (arch 16) on gfx1151 and gfx1201 only.
        let auto = [true, true, false, false];
        for (raw, expect) in [
            (Some("on"), [true; 4]),
            (Some("1"), [true; 4]),
            (Some("off"), [false; 4]),
            (Some("0"), [false; 4]),
            (Some("auto"), auto),
            (None, auto),
        ] {
            for (i, (model_arch, gpu)) in cases.iter().enumerate() {
                assert_eq!(
                    mtp_ngram_copy_enabled_for_arch_value(raw, *model_arch, gpu),
                    expect[i],
                    "{raw:?} arch={model_arch} gpu={gpu}"
                );
            }
        }
        let none = |_: &str| None;
        assert_eq!(
            ngram_mod_triple_resolve(NGRAM_MOD_COPY_TRIPLE, none),
            (5, 1, 63)
        );
    }

    #[test]
    fn dflash_lookup_key_and_resolver() {
        let field = field("speculation.dflash_lookup").expect("dflash_lookup schema field");
        assert_eq!(field.env_compat, Some("HIPFIRE_DFLASH_LOOKUP"));
        assert_eq!(field.legacy_key, "dflash_lookup");
        assert!(matches!(field.default, DefaultValue::String("auto")));
        for raw in ["auto", "on", "off", "1", "0"] {
            assert!(
                field.validate(&ConfigValue::String(raw.into())).is_ok(),
                "{raw}"
            );
        }
        assert!(field.validate(&ConfigValue::String("yes".into())).is_err());
        assert!(field.validate(&ConfigValue::String("true".into())).is_err());
        // Default on: only an explicit off/0 disables; anything else is the default.
        for (raw, expect) in [
            (Some("on"), true),
            (Some("1"), true),
            (Some("off"), false),
            (Some("0"), false),
            (Some("auto"), true),
            (Some(""), true),
            (Some("yes"), true),
            (None, true),
        ] {
            assert_eq!(dflash_lookup_enabled_value(raw), expect, "{raw:?}");
        }
        // The live (unset) snapshot resolves to the auto default: on.
        assert!(dflash_lookup_enabled());
    }

    #[test]
    fn dflash_lookup_tail_kill_switch_is_strict_zero() {
        // Default on (lookup itself is gated by `speculation.dflash_lookup`);
        // only an exact `0` disables the Tail windows.
        assert!(dflash_lookup_tail_enabled_value(None));
        assert!(dflash_lookup_tail_enabled_value(Some("1")));
        assert!(dflash_lookup_tail_enabled_value(Some("")));
        assert!(dflash_lookup_tail_enabled_value(Some("off")));
        assert!(!dflash_lookup_tail_enabled_value(Some("0")));
        assert!(dflash_lookup_tail_enabled());
    }

    #[test]
    fn mtp_k_default_per_arch_and_auto_rule() {
        assert_eq!(mtp_k_default(16, 8), 7);
        assert_eq!(mtp_k_default(16, 16), 15);
        assert_eq!(mtp_k_default(16, 64), 63);
        assert_eq!(mtp_k_default(16, 4), 3);
        assert_eq!(mtp_k_default(16, 1), 1);
        assert_eq!(mtp_k_default(16, 0), 1);
        assert_eq!(mtp_k_default(5, 8), MTP_K_DEFAULT);
        assert_eq!(MTP_K_DEFAULT, 3);
        for gpu in ["gfx1151", "gfx1201", "gfx1100", "gfx1150", ""] {
            assert_eq!(
                mtp_k_default(16, qwen4_row_exact_verify_rows_table(gpu)),
                7,
                "{gpu}"
            );
        }

        let field = field("speculation.mtp_k").expect("speculation.mtp_k field");
        assert!(field.validate(&ConfigValue::String("auto".into())).is_ok());
        for k in 1..=63 {
            assert!(field.validate(&ConfigValue::Integer(k)).is_ok(), "k={k}");
        }
        assert!(field.validate(&ConfigValue::Integer(0)).is_err());
        assert!(field.validate(&ConfigValue::Integer(64)).is_err());
        assert!(field.validate(&ConfigValue::String("fast".into())).is_err());
    }

    #[test]
    fn oom_guard_mode_parses_auto_on_off() {
        // Unset, "auto", and unparseable values all land on Auto — a garbage
        // value must not silently disable a safety guard, nor force it past
        // the deployment-class decision.
        assert_eq!(oom_guard_mode_for(None), OomGuardMode::Auto);
        assert_eq!(oom_guard_mode_for(Some("auto")), OomGuardMode::Auto);
        assert_eq!(oom_guard_mode_for(Some("AUTO")), OomGuardMode::Auto);
        assert_eq!(oom_guard_mode_for(Some("banana")), OomGuardMode::Auto);
        assert_eq!(oom_guard_mode_for(Some("1")), OomGuardMode::On);
        assert_eq!(oom_guard_mode_for(Some("true")), OomGuardMode::On);
        assert_eq!(oom_guard_mode_for(Some("ON")), OomGuardMode::On);
        // The typed bool renders "0"; raw compat spellings also count.
        assert_eq!(oom_guard_mode_for(Some("0")), OomGuardMode::Off);
        assert_eq!(oom_guard_mode_for(Some("false")), OomGuardMode::Off);
        assert_eq!(oom_guard_mode_for(Some("OFF")), OomGuardMode::Off);
        assert_eq!(oom_guard_mode_for(Some("no")), OomGuardMode::Off);
    }

    #[test]
    fn unified_and_discrete_arch_classes_are_disjoint_and_complete() {
        // Every APU arch must resolve to unified, every dGPU/CDNA arch to
        // not-unified, and the two tables must never overlap.
        for arch in UNIFIED_MEMORY_ARCHS {
            assert!(is_unified_memory_arch(arch));
            assert!(
                !DISCRETE_MEMORY_ARCHS.contains(arch),
                "{arch} in both tables"
            );
            // Case-insensitive: arch strings arrive from the HIP runtime.
            assert!(is_unified_memory_arch(&arch.to_uppercase()));
        }
        for arch in DISCRETE_MEMORY_ARCHS {
            assert!(!is_unified_memory_arch(arch));
            assert!(is_discrete_memory_arch(arch));
        }
        assert!(is_unified_memory_arch("gfx1151"));
        assert!(!is_unified_memory_arch("gfx1100"));
    }

    #[test]
    fn oom_guard_auto_decision_matrix() {
        // Known unified-memory APU: guard on regardless of host swap — GPU
        // allocations land in RAM either way.
        assert!(oom_guard_auto_for(Some("gfx1151"), Some(true)));
        assert!(oom_guard_auto_for(Some("gfx1151"), Some(false)));
        assert!(oom_guard_auto_for(Some("gfx1103"), None));
        // Known discrete GPU: overshoot is a failed hipMalloc; stand down.
        assert!(!oom_guard_auto_for(Some("gfx1100"), Some(true)));
        assert!(!oom_guard_auto_for(Some("gfx942"), None));
        assert!(!oom_guard_auto_for(Some("gfx1201"), Some(false)));
        // Unrecognized arch: fail safe.
        assert!(oom_guard_auto_for(Some("gfx9999"), Some(true)));
        // No GPU arch in this process: the host's own lethality decides.
        assert!(!oom_guard_auto_for(None, Some(true)));
        assert!(oom_guard_auto_for(None, Some(false)));
        // Unreadable /proc/meminfo: fail safe.
        assert!(oom_guard_auto_for(None, None));
    }

    #[test]
    fn oom_guard_schema_field_is_process_scoped_with_env_compat() {
        let field = field("memory.oom_guard").expect("oom_guard schema field");
        assert_eq!(field.env_compat, Some("HIPFIRE_OOM_GUARD"));
        // Default is the string "auto": the deployment-class decision, not a
        // blanket on/off.
        assert!(matches!(
            field.default.to_value(),
            ConfigValue::String(v) if v == "auto"
        ));
        assert!(matches!(field.rule, ValueRule::AutoBool));
        assert!(!field.include_builtin_in_process_config);
        // The AutoBool rule must accept all three spellings end to end.
        assert!(field.validate(&ConfigValue::Bool(false)).is_ok());
        assert!(field.validate(&ConfigValue::Bool(true)).is_ok());
        assert!(field
            .validate(&ConfigValue::String("auto".to_string()))
            .is_ok());
        assert!(field
            .validate(&ConfigValue::String("sometimes".to_string()))
            .is_err());
    }

    #[test]
    fn offload_exec_is_absent_from_the_process_config_until_set() {
        // The measured Flash-Next default needs "unset" distinguishable from an
        // explicit `pcie`: the built-in value must not be carried.
        let field = field("memory.offload_exec").expect("offload_exec schema field");
        assert!(!field.include_builtin_in_process_config);
        assert_eq!(field.env_compat, Some("HIPFIRE_OFFLOAD_EXEC"));
        assert!(matches!(
            field.default.to_value(),
            ConfigValue::String(v) if v == "pcie"
        ));
    }

    #[test]
    fn schema_has_unique_keys_and_legacy_keys() {
        let mut canonical = std::collections::BTreeSet::new();
        let mut legacy = std::collections::BTreeSet::new();
        let mut environment = std::collections::BTreeSet::new();
        for field in FIELDS {
            assert!(canonical.insert(field.key), "duplicate {}", field.key);
            assert!(
                legacy.insert(field.legacy_key),
                "duplicate {}",
                field.legacy_key
            );
            if let Some(name) = field.env_compat {
                assert!(
                    environment.insert(name),
                    "duplicate environment alias {name}"
                );
            }
            if !field.include_builtin_in_process_config {
                assert!(field.env_compat.is_some(), "bridge without env alias");
                // `memory.offload_exec` is the one registry-backed, model-load
                // field whose built-in is represented by absence: its unset
                // default is a measured per-device choice, not the value `pcie`.
                if field.key != "memory.offload_exec" {
                    assert!(
                        matches!(field.scope, ConfigScope::Process | ConfigScope::Diagnostic),
                        "non-process bridge {}",
                        field.key
                    );
                    assert!(
                        !field.registry_allowed,
                        "registry-backed bridge {}",
                        field.key
                    );
                }
            }
            field
                .validate(&field.default.to_value())
                .unwrap_or_else(|error| panic!("invalid default {}: {error}", field.key));
        }
    }

    #[test]
    fn decide_calibration_fields_are_per_model_request_temperatures() {
        for kind in ["choice", "score", "noul"] {
            let key = format!("decide.calibration.{kind}");
            let f = field(&key).unwrap_or_else(|| panic!("{key} missing"));
            assert!(matches!(f.scope, ConfigScope::Request), "{key}");
            assert!(f.env_compat.is_none() && !f.registry_allowed, "{key}");
            assert_eq!(f.default.to_value(), ConfigValue::Float(1.0));
            for ok in [
                ConfigValue::Float(0.05),
                ConfigValue::Integer(2),
                ConfigValue::Float(20.0),
            ] {
                assert!(f.validate(&ok).is_ok(), "{key} {ok:?}");
            }
            for bad in [
                ConfigValue::Float(0.0),
                ConfigValue::Float(0.04),
                ConfigValue::Float(20.5),
            ] {
                assert!(f.validate(&bad).is_err(), "{key} {bad:?}");
            }
        }
    }

    #[test]
    fn models_toml_accepts_and_round_trips_decide_calibration() {
        let root = temp_root("decide-calibration");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(
            &paths.models_toml,
            r#"schema_version = 1

[models."qwen3.5-4b.mq4"]
path = "/models/qwen3.5-4b.mq4"
overrides = { decide = { calibration = { choice = 1.3, score = 2, noul = 0.7 } } }
"#,
        )
        .unwrap();
        let loaded = load_catalog(&paths).unwrap();
        let (_, model) = loaded.catalog.model("qwen3.5-4b.mq4").unwrap();
        assert_eq!(
            model.overrides.get("decide.calibration.choice"),
            Some(&ConfigValue::Float(1.3))
        );
        assert_eq!(
            model.overrides.get("decide.calibration.score"),
            Some(&ConfigValue::Integer(2))
        );
        write_catalog_toml(&paths, &loaded.catalog).unwrap();
        let written = fs::read_to_string(&paths.models_toml).unwrap();
        assert!(written.contains("decide.calibration"), "{written}");
        assert_eq!(
            load_catalog_toml(&paths.models_toml).unwrap(),
            loaded.catalog
        );
        fs::write(
            &paths.models_toml,
            "schema_version = 1\n[models.m]\npath = \"/m\"\n\
             [models.m.overrides.decide.calibration]\nchoice = 0\n",
        )
        .unwrap();
        assert!(load_catalog(&paths).is_err(), "T = 0 must be refused");
    }

    #[test]
    fn experimental_memory_and_prefill_features_default_off() {
        assert_eq!(
            field("memory.cask.enabled").unwrap().default.to_value(),
            ConfigValue::Bool(false)
        );
        assert_eq!(
            field("memory.cask.auto_attach").unwrap().default.to_value(),
            ConfigValue::Bool(false)
        );
        assert_eq!(
            field("memory.cask.sidecar").unwrap().default.to_value(),
            ConfigValue::String(String::new())
        );
        assert_eq!(
            field("speculation.prefill.mode")
                .unwrap()
                .default
                .to_value(),
            ConfigValue::String("off".into())
        );
    }

    #[test]
    fn deepseek4_expert_fanout_is_optional_and_bounded() {
        let field = field("model.deepseek4_experts_per_token").unwrap();
        assert_eq!(field.default.to_value(), ConfigValue::Null);
        assert_eq!(field.parse_cli("4").unwrap(), ConfigValue::Integer(4));
        assert!(field.parse_cli("0").is_err());
        assert!(field.parse_cli("7").is_err());
    }
    #[test]
    fn vision_mode_defaults_auto_with_resident_value() {
        let field = field("vision.mode").expect("vision.mode schema field");
        assert_eq!(field.legacy_key, "vision_mode");
        assert_eq!(field.env_compat, Some("HIPFIRE_VISION_MODE"));
        assert_eq!(field.default.to_value(), ConfigValue::String("auto".into()));
        for mode in ["off", "auto", "resident", "on"] {
            assert_eq!(
                field.parse_cli(mode).unwrap(),
                ConfigValue::String(mode.into())
            );
        }
        for bad in ["sometimes", "text-only", "vram-resident"] {
            assert!(field.parse_cli(bad).is_err(), "{bad}");
        }
        assert_eq!(vision_mode_default(), "auto");
    }
    #[test]
    fn npu_trunk_calibration_schema_and_snapshot() {
        let path_field = field("fusions.npu_trunk_calibration").unwrap();
        assert_eq!(path_field.legacy_key, "npu_trunk_calibration");
        assert_eq!(path_field.env_compat, Some("HIPFIRE_QWEN4_NPU_TRUNK_CAL"));
        assert_eq!(
            path_field.default.to_value(),
            ConfigValue::String(String::new())
        );
        assert_eq!(
            path_field.parse_cli("/tmp/cal.json").unwrap(),
            ConfigValue::String("/tmp/cal.json".into())
        );
        let mut layer = ConfigLayer::default();
        layer.values.insert(
            path_field.key.into(),
            ConfigValue::String("/tmp/cal.json".into()),
        );
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::LegacyEnv {
                    name: "HIPFIRE_QWEN4_NPU_TRUNK_CAL".into(),
                },
                layer,
            }])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            process.legacy_value("HIPFIRE_QWEN4_NPU_TRUNK_CAL"),
            Some("/tmp/cal.json".into())
        );
        assert_eq!(
            ConfigPaths::under("/tmp/hipfire")
                .root
                .join("npu-trunk/calibration.json"),
            PathBuf::from("/tmp/hipfire/npu-trunk/calibration.json")
        );
    }
    #[test]
    fn npu_spillover_defaults_off_with_env_parse() {
        let field = field("kernel.npu_spillover").expect("npu_spillover schema field");
        assert_eq!(field.legacy_key, "npu_spillover");
        assert_eq!(field.env_compat, Some("HIPFIRE_NPU_SPILLOVER"));
        assert_eq!(field.default.to_value(), ConfigValue::Bool(false));
        assert_eq!(field.parse_cli("1").unwrap(), ConfigValue::Bool(true));
        assert_eq!(field.parse_cli("0").unwrap(), ConfigValue::Bool(false));
        // Default-off snapshot: an empty resolve leaves the bridge default
        // absent (same as other default-false kernel flags), which the
        // runtime parses as off.
        let process =
            ProcessConfig::from_resolved(&resolve([]).expect("empty resolve")).expect("process");
        assert_eq!(process.legacy_value("HIPFIRE_NPU_SPILLOVER"), None);
        // Env spelling flows through the env layer into the same snapshot.
        // HIPFIRE_NPU_SPILLOVER is unique to this test; no other test reads it.
        std::env::set_var("HIPFIRE_NPU_SPILLOVER", "1");
        let env_layer = load_env_layer().expect("env layer");
        std::env::remove_var("HIPFIRE_NPU_SPILLOVER");
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::LegacyEnv {
                    name: "HIPFIRE_NPU_SPILLOVER".into(),
                },
                layer: env_layer,
            }])
            .expect("env resolve"),
        )
        .expect("process");
        assert_eq!(
            process.legacy_value("HIPFIRE_NPU_SPILLOVER").as_deref(),
            Some("1")
        );
    }
    #[test]
    fn gfx11_a8_prefill_defaults_off_and_round_trips() {
        let field = field("kernel.gfx11_a8_prefill").expect("gfx11_a8_prefill schema field");
        assert_eq!(field.legacy_key, "gfx11_a8_prefill");
        assert_eq!(field.env_compat, Some("HIPFIRE_GFX11_A8_PREFILL"));
        // Off by default on every architecture: the schema default is the
        // only default, there is no per-arch variant.
        assert_eq!(field.default.to_value(), ConfigValue::Bool(false));
        assert_eq!(field.parse_cli("1").unwrap(), ConfigValue::Bool(true));
        assert_eq!(field.parse_cli("true").unwrap(), ConfigValue::Bool(true));
        assert_eq!(field.parse_cli("0").unwrap(), ConfigValue::Bool(false));
        assert!(field.parse_cli("sometimes").is_err());
        // An empty resolve leaves the bridge value absent, which the runtime
        // parses as off.
        let process =
            ProcessConfig::from_resolved(&resolve([]).expect("empty resolve")).expect("process");
        assert_eq!(process.legacy_value("HIPFIRE_GFX11_A8_PREFILL"), None);

        // The key is process-wide: a per-model override is refused.
        let mut overrides = ConfigLayer::default();
        overrides
            .set("kernel.gfx11_a8_prefill", ConfigValue::Bool(true))
            .unwrap();
        let error = validate_model_layer(&overrides).unwrap_err().to_string();
        assert!(error.contains("kernel.gfx11_a8_prefill"), "{error}");
        assert!(
            error.contains("not valid in a per-model override"),
            "{error}"
        );

        // Global TOML round trip under [kernel].
        let root = temp_root("gfx11-a8-prefill");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.gfx11_a8_prefill", "true").unwrap();
        write_global_toml(&paths, &layer).unwrap();
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[kernel]"), "{rendered}");
        assert!(rendered.contains("gfx11_a8_prefill = true"), "{rendered}");
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: paths.config_toml.clone(),
            },
            layer: loaded.layer,
        }])
        .expect("global resolve");
        let process = ProcessConfig::from_resolved(&resolved).expect("process");
        assert_eq!(
            process.legacy_value("HIPFIRE_GFX11_A8_PREFILL").as_deref(),
            Some("1")
        );
        let _ = fs::remove_dir_all(root);

        // Env spelling flows through the env layer into the same snapshot.
        // HIPFIRE_GFX11_A8_PREFILL is unique to this test; no other test reads it.
        std::env::set_var("HIPFIRE_GFX11_A8_PREFILL", "1");
        let env_layer = load_env_layer().expect("env layer");
        std::env::remove_var("HIPFIRE_GFX11_A8_PREFILL");
        let process = ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::LegacyEnv {
                    name: "HIPFIRE_GFX11_A8_PREFILL".into(),
                },
                layer: env_layer,
            }])
            .expect("env resolve"),
        )
        .expect("process");
        assert_eq!(
            process.legacy_value("HIPFIRE_GFX11_A8_PREFILL").as_deref(),
            Some("1")
        );
    }
    #[test]
    fn image_decode_defaults_cpu_with_cpu_vcn_auto_values() {
        let field = field("image.decode").expect("image.decode schema field");
        assert_eq!(field.legacy_key, "image_decode");
        assert_eq!(field.env_compat, Some("HIPFIRE_IMAGE_DECODE"));
        assert_eq!(field.default.to_value(), ConfigValue::String("cpu".into()));
        for mode in ["cpu", "vcn", "auto"] {
            assert_eq!(
                field.parse_cli(mode).unwrap(),
                ConfigValue::String(mode.into())
            );
        }
        assert!(field.parse_cli("sometimes").is_err());
    }

    #[test]
    fn million_context_and_parent_output_limits_validate_without_coupling_effort() {
        let mut layer = ConfigLayer::default();
        layer
            .set_cli("memory.max_seq", "1048576")
            .expect("one-million-token context must validate");
        layer
            .set_cli("generation.max_tokens", "393216")
            .expect("384-Ki-token output must validate");
        layer
            .set_cli("reasoning.max_tokens", "393216")
            .expect("an explicit 384-Ki reasoning cap must validate");
        layer
            .set_cli("reasoning.effort", "max")
            .expect("parent max effort must validate independently");
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("max".into()))
        );
        for effort in ["low", "medium", "xhigh"] {
            layer
                .set_cli("reasoning.effort", effort)
                .unwrap_or_else(|_| panic!("Qwen3.8 effort {effort} must validate"));
            assert_eq!(
                layer.get("reasoning.effort"),
                Some(&ConfigValue::String(effort.into()))
            );
        }
        assert_eq!(
            layer.get("reasoning.max_tokens"),
            Some(&ConfigValue::Integer(393216))
        );
        assert!(layer.set_cli("memory.max_seq", "1048577").is_err());
        assert!(layer.set_cli("generation.max_tokens", "393217").is_err());
    }

    #[test]
    fn legacy_json_maps_to_canonical_keys() {
        let root = temp_root("legacy");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(
            &paths.config_json,
            r#"{"kv_cache":"q8","max_tokens":8192,"prompt_normalize":false,"unknown":1}"#,
        )
        .unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.format, ConfigFormat::LegacyJson);
        assert_eq!(
            loaded.layer.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            loaded.layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192))
        );
        assert_eq!(loaded.warnings.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_json_drops_only_the_old_default_max_seq() {
        for (raw, kept) in [(32768, None), (65536, Some(ConfigValue::Integer(65536)))] {
            let root = temp_root(&format!("legacy-max-seq-{raw}"));
            fs::create_dir_all(&root).unwrap();
            let paths = ConfigPaths::under(&root);
            fs::write(&paths.config_json, format!(r#"{{"max_seq":{raw}}}"#)).unwrap();
            let loaded = load_global(&paths).unwrap();
            assert_eq!(loaded.layer.get("memory.max_seq"), kept.as_ref());
            assert_eq!(loaded.warnings.len(), usize::from(kept.is_none()));
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn retired_keys_warn_and_are_ignored() {
        // kernel.mw16 shipped in 0.3.1/0.4.0 and was removed in 0.4.1. A file
        // written by those releases must still load through the window.
        for (name, body) in [
            (
                "canonical",
                "schema_version = 1\n[kernel]\nmw16 = true\n[memory]\nmax_seq = 8192\n",
            ),
            (
                "legacy",
                "schema_version = 1\nmw16 = true\n[memory]\nmax_seq = 8192\n",
            ),
        ] {
            let root = temp_root(&format!("retired-{name}"));
            fs::create_dir_all(&root).unwrap();
            let paths = ConfigPaths::under(&root);
            fs::write(&paths.config_toml, body).unwrap();
            let loaded = load_global(&paths).unwrap();
            assert_eq!(loaded.format, ConfigFormat::Toml);
            assert_eq!(
                loaded.layer.get("memory.max_seq"),
                Some(&ConfigValue::Integer(8192))
            );
            assert_eq!(
                loaded.layer.values.len(),
                1,
                "{name}: retired key must be dropped"
            );
            assert_eq!(loaded.warnings.len(), 1, "{name}: {:?}", loaded.warnings);
            let warning = &loaded.warnings[0];
            for needle in ["kernel.mw16", "removed in 0.4.1", "until 0.5.0"] {
                assert!(warning.contains(needle), "{name}: {warning}");
            }
            assert!(resolve([NamedLayer {
                source: ConfigSource::GlobalUser { path: loaded.path },
                layer: loaded.layer,
            }])
            .is_ok());
            // Setting a retired key is still refused: it has no effect. The
            // error names the removal instead of calling the key unknown.
            let err = ConfigLayer::default()
                .set_cli("kernel.mw16", "true")
                .unwrap_err();
            assert!(matches!(
                err,
                ConfigError::RetiredKey {
                    key: "kernel.mw16",
                    ..
                }
            ));
            assert!(err.to_string().contains("removed in 0.4.1"), "{err}");
            assert!(matches!(
                ConfigLayer::default().set_cli("mw16", "true"),
                Err(ConfigError::RetiredKey { .. })
            ));
            assert!(matches!(
                ConfigLayer::default().set_cli("kernel.mw17", "true"),
                Err(ConfigError::UnknownKey(_))
            ));
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn unreleased_gfx11_iu4_swizzle_stays_unknown() {
        // kernel.gfx11_iu4_swizzle only existed on the exp/gfx11-iu4-swizzle
        // branch (c34c9ab3bc) and never shipped, so it is an unknown key, not
        // a retired one. This is hipx's config written by that branch.
        let hipx = "schema_version = 1\n\n[diagnostic]\ngemm_dump = false\n\n\
                    [experimental.graph]\nforward = true\n\n[kernel]\n\
                    gfx11_iu4_gridspec = true\ngfx11_iu4_shape = false\n\
                    gfx11_iu4_swizzle = false\n\n[memory]\nmax_seq = 8192\n\n\
                    [speculation]\ndflash = \"off\"\nmode = \"auto\"\nmtp = \"auto\"\n";
        assert!(retired_config_key("kernel.gfx11_iu4_swizzle").is_none());
        let root = temp_root("unreleased-swizzle");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(&paths.config_toml, hipx).unwrap();
        match load_global(&paths) {
            Err(ConfigError::UnknownKey(key)) => assert_eq!(key, "kernel.gfx11_iu4_swizzle"),
            other => panic!("expected UnknownKey, got {other:?}"),
        }
        fs::write(
            &paths.config_toml,
            hipx.replace("gfx11_iu4_swizzle = false\n", ""),
        )
        .unwrap();
        let loaded = load_global(&paths).unwrap();
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn retired_keys_are_not_live_schema_keys() {
        for retired in RETIRED_CONFIG_KEYS {
            assert!(field(retired.key).is_none(), "{} is live", retired.key);
            assert!(
                field(retired.legacy_key).is_none(),
                "{} is live",
                retired.legacy_key
            );
            assert!(!retired.ignored_until.is_empty() && !retired.shipped_in.is_empty());
        }
    }

    #[test]
    fn sparse_toml_roundtrip() {
        let root = temp_root("roundtrip");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("kv_cache", "q8").unwrap();
        layer.set_cli("generation.max_tokens", "8192").unwrap();
        layer.set_cli("prompt_normalize", "false").unwrap();
        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.format, ConfigFormat::Toml);
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[generation]"));
        assert!(rendered.contains("[memory]"));
        assert!(!rendered.contains("temperature"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn full_sampling_surface_roundtrips_through_toml() {
        let root = temp_root("sampling-roundtrip");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("generation.temperature", "1.0").unwrap();
        layer.set_cli("generation.top_p", "0.95").unwrap();
        layer.set_cli("generation.top_k", "40").unwrap();
        layer.set_cli("generation.min_p", "0.05").unwrap();
        layer.set_cli("generation.presence_penalty", "1.5").unwrap();
        layer.set_cli("generation.repeat_penalty", "1.05").unwrap();
        layer
            .set_cli("prompt.system", "Registry-compatible system prompt")
            .unwrap();

        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("top_k = 40"));
        assert!(rendered.contains("min_p = 0.05"));
        assert!(rendered.contains("presence_penalty = 1.5"));
        assert!(rendered.contains("system = \"Registry-compatible system prompt\""));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn developer_namespace_roundtrips_and_preserves_legacy_spelling() {
        let root = temp_root("developer-roundtrip");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer
            .set("developer.gfx1151_gate_up_wave64", ConfigValue::Bool(true))
            .unwrap();
        layer
            .set("developer.pm4_queue_count", ConfigValue::Integer(4))
            .unwrap();
        layer.set_cli("developer.experimental_mode", "on").unwrap();

        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[developer]"));
        assert!(rendered.contains("gfx1151_gate_up_wave64 = true"));
        assert!(rendered.contains("experimental_mode = \"on\""));

        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: paths.config_toml.clone(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(
            process
                .legacy_value("HIPFIRE_GFX1151_GATE_UP_WAVE64")
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_EXPERIMENTAL_MODE").as_deref(),
            Some("on")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn developer_namespace_is_rejected_per_model() {
        let mut overrides = ConfigLayer::default();
        overrides
            .set("developer.gfx1151_gate_up_wave64", ConfigValue::Bool(true))
            .unwrap();
        assert!(validate_model_layer(&overrides).is_err());
    }

    #[test]
    fn typed_runtime_scalars_and_variants_roundtrip() {
        let root = temp_root("runtime-scalars");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("hardware.devices", "2,3").unwrap();
        layer
            .set_cli("hardware.uniform_vram_tolerance_gb", "1.5")
            .unwrap();
        layer
            .set_cli("diagnostic.kernel.gate_up_variant", "k4")
            .unwrap();
        layer
            .set_cli("diagnostic.kernel.rdna2_variant", "5")
            .unwrap();
        layer.set_cli("kernel.lm_head_f16", "f32").unwrap();
        assert!(layer
            .set_cli("diagnostic.kernel.gate_up_variant", "unknown")
            .is_err());
        assert!(layer
            .set_cli("diagnostic.kernel.rdna2_variant", "6")
            .is_err());

        write_global_toml(&paths, &layer).unwrap();
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.layer, layer);
        let rendered = fs::read_to_string(&paths.config_toml).unwrap();
        assert!(rendered.contains("[diagnostic.kernel]"));
        assert!(rendered.contains("gate_up_variant = \"k4\""));
        assert!(rendered.contains("rdna2_variant = 5"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_catalog_migrates_aliases_and_model_overrides_to_toml() {
        let root = temp_root("catalog-migration");
        fs::create_dir_all(&root).unwrap();
        let paths = ConfigPaths::under(&root);
        fs::write(
            &paths.models_json,
            r#"{
              "schema_version": 2,
              "aliases": {
                "my-qwen": {"file":"qwen.mq4r","local_path":"/models/qwen.mq4r"}
              },
              "configs": {"qwen:tag":{"thinking_budget":"xhigh"}},
              "models": {
                "qwen.mq4r": {
                  "path":"/models/qwen.mq4r",
                  "registry_tag":"qwen:tag",
                  "aliases":["q-local"],
                  "config":{"kv_cache":"q8"}
                }
              }
            }"#,
        )
        .unwrap();
        fs::write(
            &paths.legacy_per_model_json,
            r#"{"my-qwen":{"max_tokens":8192}}"#,
        )
        .unwrap();

        let loaded = load_catalog(&paths).unwrap();
        assert_eq!(loaded.format, CatalogFormat::LegacyJson);
        let (id, model) = loaded.catalog.model("my-qwen").unwrap();
        assert_eq!(id, "qwen.mq4r");
        assert_eq!(
            model.overrides.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            model.overrides.get("reasoning.budget"),
            Some(&ConfigValue::String("xhigh".into()))
        );
        assert_eq!(
            model.overrides.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192))
        );

        write_catalog_toml(&paths, &loaded.catalog).unwrap();
        let roundtrip = load_catalog(&paths).unwrap();
        assert_eq!(roundtrip.format, CatalogFormat::Toml);
        assert_eq!(roundtrip.catalog, loaded.catalog);
        assert!(paths.models_json.exists());
        assert!(paths.legacy_per_model_json.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolution_keeps_provenance_and_shadowed_values() {
        let mut registry = ConfigLayer::default();
        registry.set_cli("temperature", "1.0").unwrap();
        let mut global = ConfigLayer::default();
        global.set_cli("generation.temperature", "0.7").unwrap();
        let resolved = resolve([
            NamedLayer {
                source: ConfigSource::RegistryModel {
                    tag: "model".into(),
                    revision: "v1".into(),
                },
                layer: registry,
            },
            NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer: global,
            },
        ])
        .unwrap();
        let temperature = resolved.get("temperature").unwrap();
        assert_eq!(temperature.value, ConfigValue::Float(0.7));
        assert!(matches!(
            temperature.source,
            ConfigSource::GlobalUser { .. }
        ));
        assert_eq!(temperature.shadowed.len(), 2);
    }

    #[test]
    fn process_config_is_sparse_versioned_and_revalidated() {
        let mut global = ConfigLayer::default();
        global.set_cli("kernel.deterministic", "true").unwrap();
        global.set_cli("diagnostic.kernel.gemv_rows", "4").unwrap();
        global
            .set_cli("attention.ck_runtime_lib", "/opt/hipfire/ck.so")
            .unwrap();
        global
            .set_cli("attention.ck_workspace_bytes", "536870912")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();

        assert_eq!(
            process.legacy_value("HIPFIRE_DETERMINISTIC").as_deref(),
            Some("1")
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_FLASH_ATTN_CK_LIB").as_deref(),
            Some("/opt/hipfire/ck.so")
        );
        assert_eq!(
            process
                .legacy_value("HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES")
                .as_deref(),
            Some("536870912")
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_GEMV_ROWS").as_deref(),
            Some("4")
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_RDNA3_HFQ4_QKV_WAVE64"),
            None,
            "architecture-sensitive bridge defaults remain absent"
        );
        assert_eq!(
            process.legacy_value("HIPFIRE_DEVICES"),
            None,
            "nullable TOML fields remain absent instead of becoming empty strings"
        );

        let encoded = serde_json::to_string(&process).unwrap();
        let decoded: ProcessConfig = serde_json::from_str(&encoded).unwrap();
        decoded.validate().unwrap();

        let wrong_version = encoded.replace("\"schema_version\":1", "\"schema_version\":2");
        let decoded: ProcessConfig = serde_json::from_str(&wrong_version).unwrap();
        assert!(decoded.validate().is_err());
        assert!(serde_json::from_str::<ProcessConfig>(
            r#"{"schema_version":1,"values":{"values":{}},"unknown":true}"#
        )
        .is_err());
    }

    #[test]
    fn diagnostic_replay_route_proof_log_lowers_from_toml() {
        let mut global = ConfigLayer::default();
        global
            .set_cli("diagnostic.replay.route_proof_log", "true")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(
            process
                .legacy_value("HIPFIRE_REPLAY_ROUTE_PROOF_LOG")
                .as_deref(),
            Some("1")
        );
        assert_eq!(
            process.values.get("diagnostic.replay.route_proof_log"),
            Some(&ConfigValue::Bool(true))
        );
    }

    #[test]
    fn invalid_values_fail_closed() {
        let mut layer = ConfigLayer::default();
        assert!(layer.set_cli("top_p", "0").is_err());
        assert!(layer.set_cli("port", "70000").is_err());
        assert!(layer.set_cli("kv_cache", "magic4").is_err());
        assert!(layer.set_cli("dflash_ngram_block", "auto").is_ok());
        assert!(layer.set_cli("dflash_ngram_block", "false").is_ok());
    }

    #[test]
    fn kv_backend_accepts_legacy_and_vmm_rejects_contiguous_with_migration() {
        let field = field("memory.kv_backend").expect("memory.kv_backend schema field");
        assert_eq!(field.legacy_key, "kv_backend");
        assert_eq!(field.default.to_value(), ConfigValue::String("vmm".into()));
        assert!(matches!(field.rule, ValueRule::KvBackend));
        assert_eq!(
            field.parse_cli("legacy").unwrap(),
            ConfigValue::String("legacy".into())
        );
        assert_eq!(
            field.parse_cli("vmm").unwrap(),
            ConfigValue::String("vmm".into())
        );

        let message = field.parse_cli("contiguous").unwrap_err().to_string();
        assert!(
            message.contains("contiguous") && message.contains("legacy"),
            "contiguous rejection must name the legacy rename: {message}"
        );
        assert!(
            message.contains("memory.kv_backend") || message.contains("--kv-backend"),
            "contiguous rejection must be actionable: {message}"
        );

        // Both TOML-layer set and config-set CLI paths share field.validate.
        let mut layer = ConfigLayer::default();
        layer
            .set("memory.kv_backend", ConfigValue::String("legacy".into()))
            .expect("legacy must validate on TOML load path");
        assert!(layer
            .set(
                "memory.kv_backend",
                ConfigValue::String("contiguous".into()),
            )
            .is_err());
        let set_cli_err = layer
            .set_cli("memory.kv_backend", "contiguous")
            .unwrap_err();
        let set_cli_msg = set_cli_err.to_string();
        assert!(
            set_cli_msg.contains("legacy"),
            "config set contiguous must name legacy: {set_cli_msg}"
        );
    }

    #[test]
    fn kv_k_and_kv_v_default_empty_and_accept_named_overrides() {
        let k = field("memory.kv_k").expect("memory.kv_k schema field");
        let v = field("memory.kv_v").expect("memory.kv_v schema field");
        assert_eq!(k.legacy_key, "kv_k");
        assert_eq!(v.legacy_key, "kv_v");
        assert_eq!(k.default.to_value(), ConfigValue::String(String::new()));
        assert_eq!(v.default.to_value(), ConfigValue::String(String::new()));

        let resolved = resolve([]).expect("empty resolve");
        let k_resolved = resolved.get("memory.kv_k").expect("built-in kv_k");
        let v_resolved = resolved.get("memory.kv_v").expect("built-in kv_v");
        assert_eq!(k_resolved.value, ConfigValue::String(String::new()));
        assert_eq!(v_resolved.value, ConfigValue::String(String::new()));
        assert_eq!(k_resolved.source, ConfigSource::BuiltIn);
        assert_eq!(v_resolved.source, ConfigSource::BuiltIn);

        for name in [
            "",
            "q8",
            "fwht2",
            "fwht3",
            "fwht4",
            "asym2",
            "asym3",
            "asym4",
            "turbo",
            "turbo2",
            "turbo3",
            "turbo4",
            "legacy-asym2",
            "legacy-asym3",
            "legacy-asym4",
        ] {
            assert_eq!(
                k.parse_cli(name).unwrap(),
                ConfigValue::String(name.into()),
                "kv_k must accept {name:?}"
            );
        }
        for name in ["", "q8", "lloyd2", "lloyd3", "lloyd4"] {
            assert_eq!(
                v.parse_cli(name).unwrap(),
                ConfigValue::String(name.into()),
                "kv_v must accept {name:?}"
            );
        }
        assert!(k.parse_cli("lloyd3").is_err());
        assert!(v.parse_cli("fwht3").is_err());
        assert!(k.parse_cli("auto").is_err());

        let mut layer = ConfigLayer::default();
        layer.set_cli("memory.kv_k", "legacy-asym3").unwrap();
        layer.set_cli("memory.kv_v", "lloyd3").unwrap();
        assert_eq!(
            layer.get("memory.kv_k"),
            Some(&ConfigValue::String("legacy-asym3".into()))
        );
        assert_eq!(
            layer.get("memory.kv_v"),
            Some(&ConfigValue::String("lloyd3".into()))
        );

        // Default profile authors no KV keys at all (kv_cache stays at auto).
        let default =
            load_config_profile(&ConfigPaths::under(temp_root("profile-kv-axes")), "default")
                .unwrap();
        assert!(default.get("memory.kv_cache").is_none());
        assert!(default.get("memory.kv_k").is_none());
        assert!(default.get("memory.kv_v").is_none());
    }

    #[test]
    fn documented_config_profiles_match_the_schema() {
        let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/configs");
        for name in ["user.toml", "developer.toml", "redline-pm4.toml"] {
            let path = docs.join(name);
            load_toml_layer(&path).unwrap_or_else(|error| {
                panic!("{} is not a valid config profile: {error}", path.display())
            });
        }
    }

    #[test]
    fn apply_config_profile_merges_and_keeps_unrelated_keys() {
        let root = temp_root("profile-merge");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("hardware.devices", "1,2").unwrap();
        layer.set_cli("serve.port", "11527").unwrap();
        layer.set_cli("serve.default_model", "qwen3.6:27b").unwrap();
        layer.set_cli("kernel.prefill_batched", "false").unwrap();
        layer.set_cli("memory.max_seq", "262144").unwrap();
        layer
            .set("generation.temperature", ConfigValue::Float(0.42))
            .unwrap();
        layer
            .set(
                "developer.custom_experiment",
                ConfigValue::String("keep-me".into()),
            )
            .unwrap();
        let before = layer.clone();

        // "dev" overwrites kernel.prefill_batched and sets hardware.devices.
        let count = apply_config_profile(&mut layer, &paths, "dev").unwrap();
        let dev = load_config_profile(&paths, "dev").unwrap();
        assert_eq!(count, dev.values.len());

        for (key, value) in &dev.values {
            assert_eq!(layer.values.get(key), Some(value), "{key}");
        }
        for key in [
            "serve.port",
            "serve.default_model",
            "memory.max_seq",
            "generation.temperature",
            "developer.custom_experiment",
        ] {
            assert_eq!(layer.values.get(key), before.values.get(key), "{key}");
        }
        assert_eq!(
            layer.get("hardware.devices"),
            Some(&ConfigValue::String("0".into())),
            "profile key overwrites the old value"
        );
        assert_eq!(
            layer.get("kernel.prefill_batched"),
            Some(&ConfigValue::Bool(true))
        );
        assert_eq!(
            layer.values.len(),
            before.values.len() + dev.values.len() - 2,
            "only hardware.devices and kernel.prefill_batched overlap"
        );
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("dev")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn default_profile_pins_nothing_registry_or_arch_resolved() {
        let bundle = config_profile_bundle("default").unwrap();
        assert!(!bundle.is_empty());
        for (key, _) in &bundle {
            let field = field(key).unwrap_or_else(|| panic!("{key} is not a schema field"));
            assert!(
                !field.registry_allowed,
                "default profile pins registry-overridable key {key}"
            );
        }
        let layer = builtin_config_profile_layer("default").unwrap();
        assert!(layer.get("memory.kv_cache").is_none());
        assert!(layer.get("speculation.dflash").is_none());
    }

    #[test]
    fn detect_config_profile_matches_subsets_and_prefers_largest() {
        let root = temp_root("profile-detect");
        let paths = ConfigPaths::under(&root);

        // Empty layer: nothing applied.
        assert_eq!(detect_config_profile(&paths, &ConfigLayer::default()), None);

        // Extra unrelated keys do not defeat a match.
        let mut layer = ConfigLayer::default();
        apply_config_profile(&mut layer, &paths, "redline").unwrap();
        layer.set_cli("serve.port", "11527").unwrap();
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("redline")
        );

        // A profile whose key has a different value is not applied.
        layer.set_cli("replay.transport", "aql").unwrap();
        assert_eq!(detect_config_profile(&paths, &layer), None);

        // hip and redline both set replay.backend; the layer holds hip's.
        let mut layer = ConfigLayer::default();
        apply_config_profile(&mut layer, &paths, "hip").unwrap();
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("hip")
        );

        // Most keys wins: dev (16 keys) over hip when both are subsets.
        apply_config_profile(&mut layer, &paths, "dev").unwrap();
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("dev")
        );

        // Custom profile with more keys than every built-in subset wins; an
        // empty custom profile never matches; ties keep list order.
        let mut small = ConfigLayer::default();
        small.set_cli("serve.port", "12000").unwrap();
        create_config_profile(&paths, "aaa-empty", &ConfigLayer::default()).unwrap();
        create_config_profile(&paths, "alpha", &small).unwrap();
        create_config_profile(&paths, "beta", &small).unwrap();
        let mut layer = ConfigLayer::default();
        layer.set_cli("serve.port", "12000").unwrap();
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("alpha")
        );
        let mut big = small.clone();
        big.set_cli("serve.host", "0.0.0.0").unwrap();
        create_config_profile(&paths, "big", &big).unwrap();
        layer.set_cli("serve.host", "0.0.0.0").unwrap();
        assert_eq!(
            detect_config_profile(&paths, &layer).as_deref(),
            Some("big")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn diff_config_layers_reports_added_changed_removed_in_toml_form() {
        let mut old = ConfigLayer::default();
        old.set_cli("serve.port", "11527").unwrap();
        old.set_cli("memory.kv_cache", "q8").unwrap();
        old.set_cli("serve.local", "true").unwrap();
        let mut new = old.clone();
        new.set_cli("serve.port", "11435").unwrap();
        new.set_cli("serve.host", "127.0.0.1").unwrap();
        new.remove("serve.local").unwrap();
        let diff = diff_config_layers(&old, &new);
        assert_eq!(
            diff.added,
            vec![("serve.host".into(), ConfigValue::String("127.0.0.1".into()))]
        );
        assert_eq!(
            diff.changed,
            vec![(
                "serve.port".into(),
                ConfigValue::Integer(11527),
                ConfigValue::Integer(11435)
            )]
        );
        assert_eq!(
            diff.removed,
            vec![("serve.local".into(), ConfigValue::Bool(true))]
        );
        assert_eq!(diff.len(), 3);
        assert!(diff_config_layers(&new, &new).is_empty());
        assert_eq!(ConfigValue::String("q8".into()).to_toml_string(), "\"q8\"");
        assert_eq!(ConfigValue::Integer(7).to_toml_string(), "7");
        assert_eq!(ConfigValue::Bool(false).to_toml_string(), "false");
        assert_eq!(ConfigValue::Float(0.42).to_toml_string(), "0.42");
    }

    #[test]
    fn apply_config_profile_materializes_builtin_bundles() {
        let root = temp_root("profile-builtins");
        let paths = ConfigPaths::under(&root);
        for name in CONFIG_PROFILE_NAMES {
            let mut layer = ConfigLayer::default();
            apply_config_profile(&mut layer, &paths, name).unwrap();
            let expected = builtin_config_profile_layer(name).unwrap();
            assert_eq!(layer, expected, "profile {name}");
            assert_eq!(
                detect_config_profile(&paths, &layer).as_deref(),
                Some(*name)
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn hip_profile_is_an_explicit_redline_opt_out() {
        let paths = ConfigPaths::under(temp_root("profile-hip"));
        let layer = load_config_profile(&paths, "hip").unwrap();
        assert_eq!(
            layer.get("replay.backend"),
            Some(&ConfigValue::String("hip".to_owned()))
        );
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn apply_config_profile_rejects_unknown_names() {
        let root = temp_root("profile-unknown");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set("serve.port", ConfigValue::Integer(9)).unwrap();
        let before = layer.clone();
        let err = apply_config_profile(&mut layer, &paths, "staging").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unknown profile 'staging'"), "{message}");
        assert_eq!(layer, before, "failed apply must not mutate layer");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn create_config_profile_snapshots_and_rejects_duplicates() {
        let root = temp_root("profile-create");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("memory.kv_cache", "q8").unwrap();
        layer.set_cli("serve.port", "12000").unwrap();

        let path = create_config_profile(&paths, "lab", &layer).unwrap();
        assert_eq!(path, paths.profiles.join("lab.toml"));
        assert!(path.is_file());
        let loaded = load_config_profile(&paths, "lab").unwrap();
        assert_eq!(loaded, layer);

        let err = create_config_profile(&paths, "lab", &layer).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{}", err);
        let builtin = create_config_profile(&paths, "default", &layer).unwrap_err();
        assert!(builtin.to_string().contains("built-in"), "{}", builtin);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn validate_config_profile_name_rejects_traversal_and_invalid() {
        assert!(validate_config_profile_name("lab").is_ok());
        assert!(validate_config_profile_name("lab-1").is_ok());
        for bad in [
            "",
            "default",
            "dev",
            "redline",
            "hip",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "has space",
            "ü",
        ] {
            assert!(
                validate_config_profile_name(bad).is_err(),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn serve_api_key_file_schema_is_an_absolute_path_defaulting_to_unset() {
        let field = field("serve.api_key_file").unwrap();
        assert_eq!(field.env_compat, Some("HIPFIRE_SERVE_API_KEY_FILE"));
        assert_eq!(field.scope, ConfigScope::Process);
        assert_eq!(field.category, ConfigCategory::Serve);
        assert!(!field.registry_allowed);
        assert_eq!(field.default.to_value(), ConfigValue::String(String::new()));

        let absolute = env::temp_dir().join("serve.key");
        let absolute = absolute.to_str().unwrap();
        let mut layer = ConfigLayer::default();
        layer
            .set("serve.api_key_file", ConfigValue::String(String::new()))
            .unwrap();
        layer.set_cli("serve.api_key_file", absolute).unwrap();
        assert_eq!(
            layer.get("serve.api_key_file"),
            Some(&ConfigValue::String(absolute.to_owned()))
        );
        // Existence is not a schema concern: the token loader owns file checks.
        layer
            .set_cli("serve.api_key_file", &format!("{absolute}.missing"))
            .unwrap();

        let padded_left = format!(" {absolute}");
        let padded_right = format!("{absolute} ");
        for bad in [
            "serve.key",
            "./serve.key",
            "~/serve.key",
            " ",
            padded_left.as_str(),
            padded_right.as_str(),
        ] {
            let err = layer.set_cli("serve.api_key_file", bad).unwrap_err();
            assert!(
                err.to_string().contains("serve.api_key_file"),
                "{bad:?}: {err}"
            );
        }
        assert!(layer
            .set("serve.api_key_file", ConfigValue::Integer(1))
            .is_err());
        assert!(layer.set("serve.api_key_file", ConfigValue::Null).is_err());
    }

    #[test]
    fn list_config_profiles_includes_builtins_and_custom() {
        let root = temp_root("profile-list");
        let paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer.set_cli("serve.host", "127.0.0.1").unwrap();
        create_config_profile(&paths, "zeta", &layer).unwrap();
        create_config_profile(&paths, "alpha", &layer).unwrap();
        let names: Vec<_> = list_config_profiles(&paths)
            .unwrap()
            .into_iter()
            .map(|entry| (entry.name, entry.kind))
            .collect();
        assert_eq!(
            names,
            vec![
                ("default".into(), ConfigProfileKind::Builtin),
                ("dev".into(), ConfigProfileKind::Builtin),
                ("hip".into(), ConfigProfileKind::Builtin),
                ("redline".into(), ConfigProfileKind::Builtin),
                ("alpha".into(), ConfigProfileKind::Custom),
                ("zeta".into(), ConfigProfileKind::Custom),
            ]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn config_profile_bundles_match_documented_examples() {
        let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/configs");
        let paths = ConfigPaths::under(temp_root("profile-docs"));
        let pairs = [
            ("default", "user.toml"),
            ("dev", "developer.toml"),
            ("redline", "redline-pm4.toml"),
        ];
        for (name, file) in pairs {
            let path = docs.join(file);
            let example = load_toml_layer(&path)
                .unwrap_or_else(|error| panic!("{} failed to load: {error}", path.display()));
            let applied = load_config_profile(&paths, name).unwrap();
            assert_eq!(
                applied, example,
                "profile {name} must match docs/configs/{file}"
            );
        }
        let _ = fs::remove_dir_all(paths.root);
    }

    #[test]
    fn deepseek4_placement_round_trips_typed_exact_arch_selectors() {
        let raw = "dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)";
        let placement: Deepseek4ComputePlacement = raw.parse().unwrap();
        assert_eq!(placement.to_string(), raw);
        assert_eq!(
            placement,
            Deepseek4ComputePlacement::DenseExpertSplit {
                dense: DeviceSelector::ExactArch("gfx1100".into()),
                experts: DeviceSelector::ExactArch("gfx1151".into()),
            }
        );
        let mut layer = ConfigLayer::default();
        layer
            .set_cli("hardware.deepseek4_compute_placement", raw)
            .unwrap();
        assert_eq!(
            layer.get("hardware.deepseek4_compute_placement"),
            Some(&ConfigValue::String(raw.into()))
        );
    }

    #[test]
    fn deepseek4_compressor_cache_dtype_is_selected_through_kv_cache() {
        let field = field("memory.kv_cache").unwrap();
        assert_eq!(
            field.parse_cli("f32").unwrap(),
            ConfigValue::String("f32".into())
        );
        assert_eq!(
            field.parse_cli("f16").unwrap(),
            ConfigValue::String("f16".into())
        );
        assert_eq!(
            "f16".parse::<Deepseek4CompressorCache>().unwrap(),
            Deepseek4CompressorCache::F16
        );
        assert_eq!(Deepseek4CompressorCache::F32.to_string(), "f32");
        assert!("auto".parse::<Deepseek4CompressorCache>().is_err());
    }

    #[test]
    fn deepseek4_placement_rejects_logical_ordinals_and_aliasing() {
        for raw in [
            "dense-expert-split(dense=device:0,experts=arch:gfx1151)",
            "dense-expert-split(dense=arch:gfx1151,experts=arch:gfx1151)",
            "dense-expert-split(dense=pci:03:00.0,experts=arch:gfx1151)",
        ] {
            assert!(
                raw.parse::<Deepseek4ComputePlacement>().is_err(),
                "expected rejection for {raw}"
            );
        }
    }
}

/// Device topology policy: expert-parallel degree (`topology.ep`, compat env
/// `HIPFIRE_TOPOLOGY_EP`), the Qwen4 kill switch `HIPFIRE_QWEN4_EP=0`, and
/// pipeline-parallel degree (`topology.pp`, compat env `HIPFIRE_TOPOLOGY_PP`).
/// Pure and GPU-independent; the loader resolves the degree once, before any
/// teardown or allocation.
pub mod topology {
    use super::process_value;

    /// Developer kill switch: `HIPFIRE_QWEN4_EP=0` forces Qwen4 to ep = 1
    /// regardless of `topology.ep`.
    pub const QWEN4_EP_KILL_SWITCH_ENV: &str = "HIPFIRE_QWEN4_EP";
    /// Compatibility spelling of `topology.ep`.
    pub const EP_ENV: &str = "HIPFIRE_TOPOLOGY_EP";
    /// Largest expert-parallel degree the schema admits. Experts shard
    /// `e % N`, so uneven splits (e.g. 512 over 6 ranks) are admitted.
    pub const EP_MAX_DEGREE: usize = 8;
    /// Compatibility spelling of `topology.pp`.
    pub const PP_ENV: &str = "HIPFIRE_TOPOLOGY_PP";
    /// Largest pipeline-parallel degree the schema admits.
    pub const PP_MAX_DEGREE: usize = 8;

    /// Pure parser behind [`ep`]: unset or empty is 1 (the default); an
    /// integer outside `1..=`[`EP_MAX_DEGREE`] is a refusal naming the range.
    pub fn parse_ep(raw: Option<&str>) -> Result<usize, String> {
        let Some(raw) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(1);
        };
        match raw.parse::<usize>() {
            Ok(n) if (1..=EP_MAX_DEGREE).contains(&n) => Ok(n),
            _ => Err(format!(
                "topology.ep = {raw:?} is not a valid expert-parallel degree (allowed: integers 1..=8)"
            )),
        }
    }

    /// True when the raw `HIPFIRE_QWEN4_EP` value is the kill switch (`0`).
    pub fn qwen4_ep_killed(raw: Option<&str>) -> bool {
        raw.map(str::trim) == Some("0")
    }

    /// Qwen4 degree from raw inputs: the kill switch wins over any (even
    /// invalid) configured value.
    pub fn resolve_qwen4_ep(
        configured: Option<&str>,
        kill_switch: Option<&str>,
    ) -> Result<usize, String> {
        if qwen4_ep_killed(kill_switch) {
            return Ok(1);
        }
        parse_ep(configured)
    }

    /// Configured `topology.ep` from the process snapshot.
    pub fn ep() -> Result<usize, String> {
        parse_ep(process_value(EP_ENV).as_deref())
    }

    /// Effective Qwen4 expert-parallel degree: `topology.ep` unless
    /// `HIPFIRE_QWEN4_EP=0` forces 1.
    pub fn qwen4_ep() -> Result<usize, String> {
        resolve_qwen4_ep(
            process_value(EP_ENV).as_deref(),
            process_value(QWEN4_EP_KILL_SWITCH_ENV).as_deref(),
        )
    }

    /// Pure parser behind [`pp`]: unset or empty is 1 (the default); an
    /// integer outside `1..=`[`PP_MAX_DEGREE`] is a refusal naming the range.
    pub fn parse_pp(raw: Option<&str>) -> Result<usize, String> {
        let Some(raw) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(1);
        };
        match raw.parse::<usize>() {
            Ok(n) if (1..=PP_MAX_DEGREE).contains(&n) => Ok(n),
            _ => Err(format!(
                "topology.pp = {raw:?} is not a valid pipeline-parallel degree (allowed: integers 1..=8)"
            )),
        }
    }

    /// Effective pipeline-parallel degree from raw inputs: an explicit
    /// request `requested > 1` wins and the configured value is not read;
    /// otherwise (`requested` 0 or 1) `topology.pp` applies.
    pub fn resolve_pp(requested: usize, configured: Option<&str>) -> Result<usize, String> {
        if requested > 1 {
            return Ok(requested);
        }
        parse_pp(configured)
    }

    /// Configured `topology.pp` from the process snapshot.
    pub fn pp() -> Result<usize, String> {
        parse_pp(process_value(PP_ENV).as_deref())
    }

    /// Effective pipeline-parallel degree for a load request of `requested`:
    /// the request when it is above 1, else `topology.pp` from the process
    /// snapshot.
    pub fn effective_pp(requested: usize) -> Result<usize, String> {
        resolve_pp(requested, process_value(PP_ENV).as_deref())
    }

    /// Pipeline degree of a `load` request: explicit `params.pp` > 1 wins,
    /// else `topology.pp` (`effective`, which also validates the configured
    /// value). The resolved degree is written back into `params.pp` so the
    /// weight-host envelope, slot caps and every later pp gate read one
    /// value. A refusal always names `topology.pp`.
    pub fn resolve_load_pp_with(
        msg: &mut serde_json::Value,
        effective: impl FnOnce(usize) -> Result<usize, String>,
    ) -> Result<usize, String> {
        let requested = msg
            .get("params")
            .and_then(|p| p.get("pp"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let pp = effective(requested).map_err(|error| {
            if error.contains("topology.pp") {
                format!("load refused: {error}")
            } else {
                format!("load refused: invalid topology.pp: {error}")
            }
        })?;
        if pp != requested {
            if let Some(object) = msg.as_object_mut() {
                let params = object
                    .entry("params")
                    .or_insert_with(|| serde_json::json!({}));
                if let Some(params) = params.as_object_mut() {
                    params.insert("pp".to_owned(), serde_json::json!(pp));
                }
            }
        }
        Ok(pp)
    }

    /// [`resolve_load_pp_with`] against the process snapshot.
    pub fn resolve_load_pp(msg: &mut serde_json::Value) -> Result<usize, String> {
        resolve_load_pp_with(msg, effective_pp)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_ep_defaults_and_allowed_set() {
            assert_eq!(parse_ep(None), Ok(1));
            assert_eq!(parse_ep(Some("")), Ok(1));
            assert_eq!(parse_ep(Some("1")), Ok(1));
            assert_eq!(parse_ep(Some(" 2 ")), Ok(2));
            for n in 1..=8usize {
                assert_eq!(parse_ep(Some(&n.to_string())), Ok(n));
            }
            for bad in ["0", "9", "16", "-1", "two", "2.0"] {
                let err = parse_ep(Some(bad)).unwrap_err();
                assert!(
                    err.contains("topology.ep") && err.contains("1..=8"),
                    "{bad}: {err}"
                );
            }
        }

        #[test]
        fn kill_switch_forces_one_regardless_of_config() {
            assert_eq!(resolve_qwen4_ep(Some("4"), Some("0")), Ok(1));
            assert_eq!(resolve_qwen4_ep(Some("2"), Some("0")), Ok(1));
            // Even an invalid configured value is moot under the kill switch.
            assert_eq!(resolve_qwen4_ep(Some("9"), Some("0")), Ok(1));
            // Any other kill-switch spelling leaves the config in charge.
            assert_eq!(resolve_qwen4_ep(Some("4"), Some("1")), Ok(4));
            assert_eq!(resolve_qwen4_ep(Some("4"), Some("")), Ok(4));
            assert_eq!(resolve_qwen4_ep(Some("4"), None), Ok(4));
            assert_eq!(resolve_qwen4_ep(None, None), Ok(1));
            assert_eq!(resolve_qwen4_ep(Some("6"), None), Ok(6));
            assert!(resolve_qwen4_ep(Some("9"), None).is_err());
        }

        #[test]
        fn schema_field_validates_range_and_defaults_to_one() {
            let field = crate::FIELDS
                .iter()
                .find(|field| field.key == "topology.ep")
                .expect("topology.ep is in the schema");
            assert_eq!(field.env_compat, Some(EP_ENV));
            assert!(field.validate(&crate::ConfigValue::Integer(1)).is_ok());
            for n in 1..=8 {
                assert!(field.validate(&crate::ConfigValue::Integer(n)).is_ok());
            }
            assert!(field.validate(&crate::ConfigValue::Integer(0)).is_err());
            assert!(field.validate(&crate::ConfigValue::Integer(9)).is_err());
            assert!(matches!(
                field.parse_cli("2"),
                Ok(crate::ConfigValue::Integer(2))
            ));
            assert!(matches!(field.default, crate::DefaultValue::Integer(1)));
        }

        #[test]
        fn parse_pp_defaults_and_allowed_set() {
            assert_eq!(parse_pp(None), Ok(1));
            assert_eq!(parse_pp(Some("")), Ok(1));
            assert_eq!(parse_pp(Some("   ")), Ok(1));
            assert_eq!(parse_pp(Some("1")), Ok(1));
            assert_eq!(parse_pp(Some(" 2 ")), Ok(2));
            for n in 1..=8usize {
                assert_eq!(parse_pp(Some(&n.to_string())), Ok(n));
            }
            for bad in ["0", "9", "16", "-1", "two", "2.0"] {
                let err = parse_pp(Some(bad)).unwrap_err();
                assert!(
                    err.contains("topology.pp") && err.contains("1..=8"),
                    "{bad}: {err}"
                );
            }
        }

        #[test]
        fn resolve_pp_explicit_request_above_one_wins_over_config() {
            assert_eq!(resolve_pp(2, Some("4")), Ok(2));
            assert_eq!(resolve_pp(3, None), Ok(3));
            assert_eq!(resolve_pp(8, Some("2")), Ok(8));
            // An explicit request makes the configured value moot, even an invalid one.
            assert_eq!(resolve_pp(2, Some("9")), Ok(2));
            assert_eq!(resolve_pp(2, Some("two")), Ok(2));
        }

        #[test]
        fn resolve_pp_request_one_or_zero_defers_to_config() {
            for requested in [0usize, 1] {
                assert_eq!(resolve_pp(requested, None), Ok(1));
                assert_eq!(resolve_pp(requested, Some("")), Ok(1));
                assert_eq!(resolve_pp(requested, Some("4")), Ok(4));
                assert_eq!(resolve_pp(requested, Some(" 8 ")), Ok(8));
                let err = resolve_pp(requested, Some("9")).unwrap_err();
                assert!(
                    err.contains("topology.pp") && err.contains("1..=8"),
                    "{err}"
                );
                assert!(resolve_pp(requested, Some("0")).is_err());
            }
        }

        #[test]
        fn pp_schema_field_validates_range_and_defaults_to_one() {
            let field = crate::FIELDS
                .iter()
                .find(|field| field.key == "topology.pp")
                .expect("topology.pp is in the schema");
            assert_eq!(field.env_compat, Some(PP_ENV));
            assert!(field.validate(&crate::ConfigValue::Integer(1)).is_ok());
            for n in 1..=8 {
                assert!(field.validate(&crate::ConfigValue::Integer(n)).is_ok());
            }
            assert!(field.validate(&crate::ConfigValue::Integer(0)).is_err());
            assert!(field.validate(&crate::ConfigValue::Integer(9)).is_err());
            assert!(matches!(
                field.parse_cli("2"),
                Ok(crate::ConfigValue::Integer(2))
            ));
            assert!(matches!(field.default, crate::DefaultValue::Integer(1)));
        }

        #[test]
        fn load_pp_resolves_config_fallback_and_writes_back() {
            let config_pp = |requested: usize| Ok(if requested > 1 { requested } else { 2 });
            let mut msg = serde_json::json!({"type": "load", "params": {"tp": 1}});
            assert_eq!(resolve_load_pp_with(&mut msg, config_pp), Ok(2));
            assert_eq!(msg["params"]["pp"], 2);
            let mut no_params = serde_json::json!({"type": "load"});
            assert_eq!(resolve_load_pp_with(&mut no_params, config_pp), Ok(2));
            assert_eq!(no_params["params"]["pp"], 2);
            let mut explicit = serde_json::json!({"params": {"pp": 4}});
            assert_eq!(resolve_load_pp_with(&mut explicit, config_pp), Ok(4));
            assert_eq!(explicit["params"]["pp"], 4);
            let mut single = serde_json::json!({"params": {}});
            assert_eq!(resolve_load_pp_with(&mut single, |_| Ok(1)), Ok(1));
            assert!(single["params"].get("pp").is_none());
        }

        #[test]
        fn load_pp_invalid_config_names_topology_pp() {
            let mut msg = serde_json::json!({"params": {}});
            let error = resolve_load_pp_with(&mut msg, |_| {
                Err("\"9\" is not a valid pipeline degree".to_owned())
            })
            .unwrap_err();
            assert_eq!(
                error,
                "load refused: invalid topology.pp: \"9\" is not a valid pipeline degree"
            );
            let named = resolve_load_pp_with(&mut msg, |_| Err("topology.pp = \"9\"".to_owned()))
                .unwrap_err();
            assert_eq!(named, "load refused: topology.pp = \"9\"");
        }
    }
}

/// Placement policy for partial GPU offload — decides which layers stay in
/// device VRAM versus spill to host RAM, resolved once at load so placement is
/// fixed for the model's lifetime and never thrashes per request. This module is
/// intentionally pure and GPU-independent: it owns only the configuration
/// vocabulary ([`OffloadBudget`]); the placement arithmetic and admission live in
/// `hipfire_runtime::offload`, which has the device measurements config cannot
/// take.
pub mod memory {
    use super::{developer_var, process_value};
    use std::path::{Path, PathBuf};

    /// Resident-layer budget for partial GPU offload (`memory.gpu_layer_budget`,
    /// compat env `HIPFIRE_GPU_LAYER_BUDGET`). Resolved once at load.
    ///
    /// * [`OffloadBudget::Full`] is the unset default: keep every layer
    ///   resident, never offload — byte-identical to a pure-VRAM run and the
    ///   regression guard for the whole feature.
    /// * [`OffloadBudget::Auto`] (`-1`) defers placement to the engine, as `auto`
    ///   does for every other key: the loader measures free VRAM, the per-layer
    ///   weight bytes and the KV reservation, then places by fit — routed experts
    ///   first, whole layers only when the expert tier cannot free enough. A model
    ///   that already fits is left fully resident.
    /// * [`OffloadBudget::Layers`] pins exactly this many resident layers; the
    ///   layers before them spill to host RAM.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum OffloadBudget {
        /// Fully resident — never offload (the zero-diff default for every model).
        Full,
        /// Places by fit: the minimum spill that loads, routed experts first,
        /// whole layers only when the expert tier cannot free enough. A model
        /// that already fits stays fully resident.
        Auto,
        /// Pin exactly this many resident layers; spill everything before them.
        Layers(usize),
    }

    impl std::fmt::Display for OffloadBudget {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                OffloadBudget::Full => write!(f, "full"),
                OffloadBudget::Auto => write!(f, "auto"),
                OffloadBudget::Layers(n) => write!(f, "{n}"),
            }
        }
    }

    /// Pure truth table behind [`gpu_layer_budget`], split out so unit tests can
    /// pin the contract without touching the process-global snapshot. Unset,
    /// empty, and unparseable inputs all fail closed to fully resident (so a bad
    /// or absent setting never forces offload); `"auto"`/`"-1"` selects auto; any
    /// other non-negative integer pins that many resident layers. A negative value
    /// other than `-1` is treated as unset rather than an error.
    pub fn parse_gpu_layer_budget(raw: Option<&str>) -> OffloadBudget {
        match raw.map(|v| v.trim().to_ascii_lowercase()) {
            None => OffloadBudget::Full,
            Some(v) if v.is_empty() => OffloadBudget::Full,
            Some(v) if v == "auto" || v == "-1" => OffloadBudget::Auto,
            Some(v) => match v.parse::<i64>() {
                Ok(n) if n >= 0 => OffloadBudget::Layers(n as usize),
                _ => OffloadBudget::Full,
            },
        }
    }

    /// The configured [`OffloadBudget`] from the process snapshot. Reads exactly
    /// one resolved value: `"auto"`/`"-1"` selects auto-fit; a non-negative integer
    /// pins that many resident layers; unset/empty/unparseable fails closed to full
    /// residency (the zero-diff baseline on every model).
    pub fn gpu_layer_budget() -> OffloadBudget {
        parse_gpu_layer_budget(process_value("HIPFIRE_GPU_LAYER_BUDGET").as_deref())
    }

    /// Byte budget of the session cache (`memory.session_cache_bytes`, compat
    /// env `HIPFIRE_SESSION_CACHE_BYTES`); 0 disables it.
    pub fn session_cache_bytes() -> u64 {
        process_value("HIPFIRE_SESSION_CACHE_BYTES")
            .and_then(|value| value.parse().ok())
            .unwrap_or(8589934592)
    }

    /// Prewarm budget request of the PLE hot-row store
    /// (`memory.ple_hot_store_bytes`, compat env `HIPFIRE_PLE_HOT_STORE_BYTES`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PleHotStoreBudget {
        /// `auto`: the carrier picks the budget from `MemAvailable` minus the CMA region when the
        /// device, model, sidecar and pack allow it.
        Auto,
        /// An explicit byte budget; `0` (the default) is off.
        Bytes(u64),
    }

    /// Pure parser behind [`ple_hot_store_bytes`]: unset and empty resolve
    /// to off (`Bytes(0)`, the default), `auto` to [`PleHotStoreBudget::Auto`];
    /// an integer is that budget; an unparseable value (the config layer
    /// rejects those before they get here) fails closed to off.
    pub fn parse_ple_hot_store_bytes(raw: Option<&str>) -> PleHotStoreBudget {
        match raw.map(str::trim) {
            Some(v) if v.eq_ignore_ascii_case("auto") => PleHotStoreBudget::Auto,
            Some(v) if !v.is_empty() => v
                .parse::<u64>()
                .map(PleHotStoreBudget::Bytes)
                .unwrap_or(PleHotStoreBudget::Bytes(0)),
            _ => PleHotStoreBudget::Bytes(0),
        }
    }

    /// Configured PLE hot-store budget from the process snapshot.
    pub fn ple_hot_store_bytes() -> PleHotStoreBudget {
        parse_ple_hot_store_bytes(process_value("HIPFIRE_PLE_HOT_STORE_BYTES").as_deref())
    }

    /// Which PLE row table a Flash-Next (Qwen4) load reads: the exact Q8F16
    /// rows or the 4-bit Lloyd-Max rows derived from them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PleFormat {
        /// 4-bit Lloyd-Max rows (`<model>.ple-q4`).
        Q4,
        /// Exact Q8F16 rows (embedded in a monolithic model, or `<model>.ple-q8`).
        Q8,
    }

    impl PleFormat {
        /// Spelling used in the config value and in messages.
        pub fn name(self) -> &'static str {
            match self {
                PleFormat::Q4 => "q4",
                PleFormat::Q8 => "q8",
            }
        }

        /// Sidecar extension appended to the model file name (without the dot).
        pub fn sidecar_suffix(self) -> &'static str {
            match self {
                PleFormat::Q4 => "ple-q4",
                PleFormat::Q8 => "ple-q8",
            }
        }

        /// Inverse of [`PleFormat::name`].
        pub fn from_name(name: &str) -> Option<Self> {
            match name {
                "q4" => Some(PleFormat::Q4),
                "q8" => Some(PleFormat::Q8),
                _ => None,
            }
        }
    }

    /// The `memory.ple_format` request (compat env `HIPFIRE_PLE_FORMAT`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PleFormatRequest {
        /// q4 on the unified-memory Strix Halo (`gfx1151`), q8 elsewhere; never
        /// derives or downloads a missing sidecar.
        Auto,
        /// 4-bit rows; created ahead of time (`hipfire ple derive`), never derived by a load.
        Q4,
        /// Exact Q8 rows.
        Q8,
    }

    impl PleFormatRequest {
        /// Parse a config spelling (`auto`, `q4`, `q8`; trimmed, case-insensitive).
        pub fn parse(s: &str) -> Option<Self> {
            match s.trim().to_ascii_lowercase().as_str() {
                "auto" => Some(PleFormatRequest::Auto),
                "q4" => Some(PleFormatRequest::Q4),
                "q8" => Some(PleFormatRequest::Q8),
                _ => None,
            }
        }

        /// The config spelling of this request.
        pub fn name(self) -> &'static str {
            match self {
                PleFormatRequest::Auto => "auto",
                PleFormatRequest::Q4 => "q4",
                PleFormatRequest::Q8 => "q8",
            }
        }

        /// The format this request prefers on `gpu_arch`. `Auto` picks q4 only
        /// on `gfx1151` (the unified-memory Strix Halo) and keeps q8 elsewhere
        /// until there is evidence for another arch.
        pub fn resolve(self, gpu_arch: &str) -> PleFormat {
            match self {
                PleFormatRequest::Q4 => PleFormat::Q4,
                PleFormatRequest::Q8 => PleFormat::Q8,
                PleFormatRequest::Auto if gpu_arch == "gfx1151" => PleFormat::Q4,
                PleFormatRequest::Auto => PleFormat::Q8,
            }
        }
    }

    /// Pure parser behind [`ple_format`]: unset, empty and unparseable values
    /// (the config layer rejects those before they get here) resolve to `auto`.
    pub fn parse_ple_format(raw: Option<&str>) -> PleFormatRequest {
        raw.and_then(PleFormatRequest::parse)
            .unwrap_or(PleFormatRequest::Auto)
    }

    /// Configured PLE row format request from the process snapshot.
    pub fn ple_format() -> PleFormatRequest {
        parse_ple_format(process_value("HIPFIRE_PLE_FORMAT").as_deref())
    }

    /// Pure parser behind [`ple_prewarm_bytes`]: like
    /// [`parse_ple_hot_store_bytes`] but unset and empty resolve to `auto`
    /// (the key's default), not off.
    pub fn parse_ple_prewarm_bytes(raw: Option<&str>) -> PleHotStoreBudget {
        match raw.map(str::trim) {
            None | Some("") => PleHotStoreBudget::Auto,
            other => parse_ple_hot_store_bytes(other),
        }
    }

    /// Prewarm budget of the q4 PLE sidecar page-cache prewarm
    /// (`memory.ple_prewarm_bytes`, compat env `HIPFIRE_PLE_PREWARM_BYTES`):
    /// `Auto` (default) = `MemAvailable` minus the CMA region minus 8 GiB, sampled once after
    /// the pinned experts and the model are loaded, capped at the sidecar's data bytes and, on a
    /// unified-memory GPU or a discrete GPU with every expert resident in VRAM, at 15 GiB; only
    /// a discrete GPU with host-placed experts takes the uncapped figure. 0 when `MemAvailable`
    /// is unreadable; `Bytes(0)` = off; `Bytes(n)` = that many bytes (an explicit override of
    /// the derivation, never capped at 15 GiB, still capped at the table).
    pub fn ple_prewarm_bytes() -> PleHotStoreBudget {
        parse_ple_prewarm_bytes(process_value("HIPFIRE_PLE_PREWARM_BYTES").as_deref())
    }

    /// Bytes of the rank-ordered q4 PLE table read during model load
    /// (`memory.ple_load_fill_bytes`, compat env `HIPFIRE_PLE_LOAD_FILL_BYTES`;
    /// parsed like [`parse_ple_prewarm_bytes`]): `Auto` (default) = 14 GiB,
    /// `Bytes(0)` = off, `Bytes(n)` = that many bytes.  The loader caps it at
    /// the table, an explicit prewarm budget and the host memory left after
    /// the weights.
    pub fn ple_load_fill_bytes() -> PleHotStoreBudget {
        parse_ple_prewarm_bytes(process_value("HIPFIRE_PLE_LOAD_FILL_BYTES").as_deref())
    }

    /// Path of the PLE sidecar of `model`: `<model file name>.ple-q4` /
    /// `.ple-q8`, appended to the full file name (extension included).
    pub fn ple_sidecar_path(model: &Path, format: PleFormat) -> PathBuf {
        let mut path = model.as_os_str().to_owned();
        path.push(".");
        path.push(format.sidecar_suffix());
        PathBuf::from(path)
    }

    /// Suffix of a Flash-Next trunk overlay sidecar: pre-encoded QT44 trunk
    /// tensors (an HFQM container bound to its base by size and index
    /// sha256) that the Qwen4 loader substitutes for the base's tensors.
    pub const TRUNK_OVERLAY_SIDECAR_SUFFIX: &str = ".trunk-overlay";

    /// Path of the trunk overlay sidecar of `model`: `<model file
    /// name>.trunk-overlay`, appended to the full file name (extension
    /// included), the same placement as [`ple_sidecar_path`].
    pub fn trunk_overlay_sidecar_path(model: &Path) -> PathBuf {
        let mut path = model.as_os_str().to_owned();
        path.push(TRUNK_OVERLAY_SIDECAR_SUFFIX);
        PathBuf::from(path)
    }

    /// Suffix of a Flash-Next expert hot set sidecar: the decode-ranked
    /// per-expert VRAM placement of a Qwen4 aggregate-MQ4 base (JSON, bound to
    /// its base by size and index sha256).
    pub const EXPERT_HOTSET_SIDECAR_SUFFIX: &str = ".expert-hotset";

    /// Path of the expert hot set sidecar of `model`: `<model file
    /// name>.expert-hotset`, the same placement as [`trunk_overlay_sidecar_path`].
    pub fn expert_hotset_sidecar_path(model: &Path) -> PathBuf {
        let mut path = model.as_os_str().to_owned();
        path.push(EXPERT_HOTSET_SIDECAR_SUFFIX);
        PathBuf::from(path)
    }

    /// Optional override of the PLE hot-rank sidecar path
    /// (`HIPFIRE_PLE_HOT_RANK_FILE`); empty or unset means `<model>.plehot`.
    pub fn ple_hot_rank_file() -> Option<String> {
        developer_var("HIPFIRE_PLE_HOT_RANK_FILE")
            .ok()
            .filter(|v| !v.trim().is_empty())
    }

    /// Optional override of the PLE hot-row pack path
    /// (`HIPFIRE_PLE_HOT_PACK_FILE`); empty or unset means
    /// `<model>.plehotrows`.
    pub fn ple_hot_pack_file() -> Option<String> {
        developer_var("HIPFIRE_PLE_HOT_PACK_FILE")
            .ok()
            .filter(|v| !v.trim().is_empty())
    }

    /// PLE fetch timing on stderr, one line per prefill-chunk fetch
    /// (`HIPFIRE_PLE_HOT_DEBUG=1`, developer variable).
    pub fn ple_hot_debug() -> bool {
        developer_var("HIPFIRE_PLE_HOT_DEBUG").is_ok_and(|v| v == "1")
    }

    /// Byte budget of the session cache's disk tier
    /// (`memory.session_cache_disk_bytes`, compat env
    /// `HIPFIRE_SESSION_CACHE_DISK_BYTES`); 0 disables it.
    pub fn session_cache_disk_bytes() -> u64 {
        process_value("HIPFIRE_SESSION_CACHE_DISK_BYTES")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    }

    /// Directory of the session cache's disk tier (`memory.session_cache_dir`,
    /// compat env `HIPFIRE_SESSION_CACHE_DIR`); empty =
    /// `$HIPFIRE_HOME/session-cache`.
    pub fn session_cache_dir() -> std::path::PathBuf {
        match process_value("HIPFIRE_SESSION_CACHE_DIR") {
            Some(value) if !value.trim().is_empty() => super::expand_tilde(value.trim()),
            _ => crate::ConfigPaths::discover().root.join("session-cache"),
        }
    }

    /// The configured routed-expert [`OffloadBudget`] from the process snapshot
    /// (`memory.moe_expert_budget`). Same parser and spelling as
    /// [`gpu_layer_budget`]: unset/empty/unparseable is [`OffloadBudget::Full`]
    /// (never spill an expert), `"auto"`/`"-1"` is [`OffloadBudget::Auto`], and a
    /// non-negative integer pins that many layers whose routed experts stay on
    /// the device.
    pub fn moe_expert_budget() -> OffloadBudget {
        parse_gpu_layer_budget(process_value("HIPFIRE_MOE_EXPERT_BUDGET").as_deref())
    }

    /// The configured placement budgets. Unset is [`OffloadBudget::Full`] on
    /// both tiers for every model; auto-fit and manual spills require opt-in.
    pub fn effective_offload_budgets() -> (OffloadBudget, OffloadBudget) {
        (gpu_layer_budget(), moe_expert_budget())
    }

    /// Whether an explicit spill was configured on either tier: at least one of
    /// `memory.gpu_layer_budget` / `memory.moe_expert_budget` resolved off
    /// [`OffloadBudget::Full`]. The trigger behind the loader's
    /// `partial_offload_refusal` gate. Reads the process
    /// snapshot (not pure); the pure eligibility table is [`spill_support`].
    pub fn spill_requested() -> bool {
        gpu_layer_budget() != OffloadBudget::Full || moe_expert_budget() != OffloadBudget::Full
    }

    /// Whether a spill may be requested on this path, and whether this path
    /// can honour it: the single policy behind the loader's
    /// `partial_offload_refusal` gate and both Qwen3.5 placement resolvers
    /// (sequential carrier, slots engine). Pure over caller-supplied facts —
    /// GPU-free and dependency-free — so every spelling agrees by
    /// construction. Each refusal has a stable name; callers own the wording.
    ///
    /// Priority matches the loader gate it replaces (arch, tp, pp, EP, VL,
    /// paged, REAP): the first failing dimension wins, so a tp>1 MoE reports
    /// the topology, never the expert pinning behind it.
    /// [`effective_offload_budgets`] decides *what* the budgets resolve to;
    /// this decides whether a resolved spill may *run* here.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SpillSupport {
        /// The path honours a spill; resolve budgets (and auto-fit) normally.
        Supported,
        /// The path cannot honour a spill; the payload names the dimension.
        Refused(SpillRefusal),
    }

    impl SpillSupport {
        /// True for [`SpillSupport::Supported`].
        pub fn is_supported(&self) -> bool {
            matches!(self, SpillSupport::Supported)
        }
    }

    /// Why a path cannot honour a spill. Stable names for the refusal
    /// dimension — callers map these to their own messages (the loader gate
    /// keeps its existing sentences verbatim; the arch resolvers name the
    /// same reason in their fail-closed refusal).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SpillRefusal {
        /// Neither Qwen3.5 dense (5) / MoE (6) nor LFM2.5 (11); carries the
        /// arch id.
        Arch(u32),
        /// Tensor-parallel ranks each load every layer; carries tp.
        Tp(usize),
        /// Pipeline bands own per-device layouts; carries pp.
        Pp(usize),
        /// An expert is pinned to one rank; carries the expert count.
        ExpertParallel(usize),
        /// The Qwen3.5-VL vision tower loads through its own lowered loader.
        VisionTower,
        /// The weight pager owns the expert buffers.
        PagedExperts,
        /// The pruned (REAP keep-map) expert load has no host-placement path.
        ReapKeepMap,
    }

    /// The spill-eligibility truth table. `ep_experts` is `Some(n)` with `n >
    /// 0` on an expert-parallel route (`None`/zero is not expert-parallel);
    /// `reap_keep` is whether a REAP keep-map is attached.
    pub fn spill_support(
        arch_id: u32,
        tp: usize,
        pp: usize,
        ep_experts: Option<usize>,
        paged_experts: bool,
        reap_keep: bool,
        is_vl: bool,
    ) -> SpillSupport {
        use SpillRefusal::*;
        // Only the Qwen3.5 (dense 5 / MoE 6) and LFM2.5 (11) storage seams
        // place a layer or a routed expert in host RAM. Qwen4 has its own
        // expert-residency route and the other arches have none, so either
        // knob is a request this loader cannot honour.
        if !matches!(arch_id, 5 | 6 | 11) {
            return SpillSupport::Refused(Arch(arch_id));
        }
        if tp > 1 {
            return SpillSupport::Refused(Tp(tp));
        }
        if pp > 1 {
            return SpillSupport::Refused(Pp(pp));
        }
        if let Some(experts) = ep_experts {
            if experts > 0 {
                return SpillSupport::Refused(ExpertParallel(experts));
            }
        }
        if is_vl {
            return SpillSupport::Refused(VisionTower);
        }
        if paged_experts {
            return SpillSupport::Refused(PagedExperts);
        }
        if reap_keep {
            return SpillSupport::Refused(ReapKeepMap);
        }
        SpillSupport::Supported
    }

    /// Which engine executes the ops that read a spilled layer's weights
    /// (`memory.offload_exec`, compat env `HIPFIRE_OFFLOAD_EXEC`).
    ///
    /// [`OffloadExec::Pcie`] is the default and byte-for-byte today's behaviour:
    /// the GPU kernels dereference a device alias of the host-mapped weights, so
    /// every spilled byte crosses PCIe once per token. [`OffloadExec::Cpu`]
    /// executes those GEMVs on the CPU instead, which bounds the per-token cost
    /// of a spilled layer by the link rather than by device DRAM.
    ///
    /// This decides *who multiplies*, never *what is spilled* — placement stays
    /// [`gpu_layer_budget`], and the KV cache stays in VRAM either way.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum OffloadExec {
        /// GPU kernels read the host-mapped weights over PCIe (default).
        Pcie,
        /// CPU executes the steps whose weight tensor is host-mapped.
        Cpu,
        /// Every spilled step runs on both: the GPU takes the first
        /// `memory.offload_passback_share` of its output rows, the CPU the rest.
        ///
        /// Not reproducible run to run, by design. [`OffloadExec::Cpu`] and
        /// [`OffloadExec::Pcie`] execute a layer's GEMV on different engines, so
        /// their arithmetic need not agree, and this mode's row split is
        /// scheduled per process (a seeding probe plus the online arm timings) —
        /// two identical greedy runs divide the same step differently. Pin
        /// `memory.offload_passback_share` to a number, or to `0`, which is
        /// byte-identical to [`OffloadExec::Cpu`], before diffing output against
        /// a reference.
        Passback,
    }

    impl std::fmt::Display for OffloadExec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                OffloadExec::Pcie => write!(f, "pcie"),
                OffloadExec::Cpu => write!(f, "cpu"),
                OffloadExec::Passback => write!(f, "passback"),
            }
        }
    }

    /// Pure truth table behind [`offload_exec`], split out so the contract is
    /// unit-testable without the process-global snapshot. Unset, empty and
    /// unknown all fail closed to [`OffloadExec::Pcie`] — a bad value must never
    /// silently move work to the CPU.
    pub fn parse_offload_exec(raw: Option<&str>) -> OffloadExec {
        match raw.map(|v| v.trim().to_ascii_lowercase()) {
            Some(v) if v == "cpu" => OffloadExec::Cpu,
            Some(v) if v == "passback" => OffloadExec::Passback,
            _ => OffloadExec::Pcie,
        }
    }

    /// The configured [`OffloadExec`] from the process snapshot.
    pub fn offload_exec() -> OffloadExec {
        parse_offload_exec(process_value("HIPFIRE_OFFLOAD_EXEC").as_deref())
    }

    /// Pure truth table behind [`offload_exec_explicit`]: `None` when the user
    /// (config, registry or `HIPFIRE_OFFLOAD_EXEC`) never chose a mode — unset
    /// or blank — and the parsed mode otherwise, so an explicit `pcie` is
    /// distinguishable from the unset default and an unknown spelling is an
    /// explicit (fail-closed) `pcie`.
    pub fn parse_offload_exec_explicit(raw: Option<&str>) -> Option<OffloadExec> {
        raw.filter(|v| !v.trim().is_empty())
            .map(|v| parse_offload_exec(Some(v)))
    }

    /// The explicitly configured [`OffloadExec`], `None` when nothing set it
    /// (`memory.offload_exec` is absent from the process snapshot until the
    /// user, registry or env sets it). [`offload_exec`] stays the raw value
    /// with the built-in `pcie` for every consumer that has no measured
    /// default.
    pub fn offload_exec_explicit() -> Option<OffloadExec> {
        parse_offload_exec_explicit(process_value("HIPFIRE_OFFLOAD_EXEC").as_deref())
    }

    /// `developer.qwen4_d2_default` (env `HIPFIRE_QWEN4_D2_DEFAULT`), a typed
    /// boolean: `Some(true)` forces the Flash-Next un-tiered whole-layer D2
    /// split on for a load that has host-placed experts (skipping only the
    /// measured decode-cost comparison), `Some(false)` never arms the default,
    /// `None` (unset) takes the measured selector. An explicit
    /// `memory.offload_exec` still wins over every value.
    pub fn qwen4_d2_default() -> Option<bool> {
        super::developer_typed_bool("HIPFIRE_QWEN4_D2_DEFAULT")
    }

    /// `developer.qwen4_mtp_head_d2` (env `HIPFIRE_QWEN4_MTP_HEAD_D2`), a typed
    /// boolean read once per load by the native MTP head attach: `Some(false)`
    /// never arms the head's rows=1 D2 CPU split executor (the kill); `Some(true)`
    /// and `None` (unset) arm it whenever the trunk's D2 split is armed on the
    /// same load (non-EP), otherwise the head keeps the GPU route.
    pub fn qwen4_mtp_head_d2() -> Option<bool> {
        super::developer_typed_bool("HIPFIRE_QWEN4_MTP_HEAD_D2")
    }

    /// `developer.qwen4_sp` (env `HIPFIRE_QWEN4_SP`), a typed boolean read once
    /// per Qwen4 load by the S1 short-prompt prefill hook (SP: stream the
    /// top-k* experts, run the rest on the CPU, one canonical combine). `None`
    /// (unset) follows the arch default, which the consumer resolves
    /// (`slowlink_sp::hybrid_knob_for`): **on for exact gfx1100**, off elsewhere. An
    /// explicit `false`/`0` is the kill switch; an explicit `true`/`1` asks for it on any arch
    /// (every arch but gfx1100 declines it by name). SP additionally needs an admitted S1
    /// profile, gfx1100 and the other load-time admission rules; a failed rule declines it by
    /// name and, with `developer.qwen4_sp_touched` on, the touched-only mode runs instead.
    pub fn qwen4_sp() -> Option<bool> {
        super::developer_typed_bool("HIPFIRE_QWEN4_SP")
    }

    /// `developer.qwen4_sp_touched` (env `HIPFIRE_QWEN4_SP_TOUCHED`), a typed
    /// boolean read once per Qwen4 load by the S1 short-prompt prefill hook's
    /// touched-only mode (SP `TouchedOnly`: stream exactly the experts a layer
    /// routes to, run S1's own per-panel pipeline over them, no CPU expert, byte-identical
    /// to the stream-all sweep). `None` (unset) follows the arch default, which the
    /// consumer resolves (`slowlink_sp::touched_knob_for`): **on for exact gfx1100**, off
    /// elsewhere. An explicit `false`/`0` is the kill switch; an explicit `true`/`1` asks for it on
    /// any arch (every arch but gfx1100 declines it by name: only gfx1100 is calibrated). It needs
    /// an admitted S1 profile, gfx1100 and an aggregate window route; a failed rule declines it
    /// by name. With `developer.qwen4_sp` also on, the hybrid SP mode wins.
    pub fn qwen4_sp_touched() -> Option<bool> {
        super::developer_typed_bool("HIPFIRE_QWEN4_SP_TOUCHED")
    }

    /// Developer kill switch `HIPFIRE_QWEN4_DIRECT_LOAD` for the discrete-GPU
    /// Qwen4 direct weight reader (parallel `O_DIRECT`/buffered file reads
    /// straight into pinned or staged device memory). Default on; exactly `0`
    /// restores the unchanged mmap + single-thread copy path. Unified-memory
    /// devices never take the direct reader regardless of this value.
    pub fn qwen4_direct_load() -> bool {
        parse_qwen4_direct_load(process_value("HIPFIRE_QWEN4_DIRECT_LOAD").as_deref())
    }

    /// Pure truth table behind [`qwen4_direct_load`]: only a trimmed `0` is
    /// off; unset, empty and every other spelling keep the default (on).
    pub fn parse_qwen4_direct_load(raw: Option<&str>) -> bool {
        raw.map(str::trim) != Some("0")
    }

    /// Developer `HIPFIRE_LOAD_WEIGHT_DIGEST=<path>`: when set and not blank,
    /// the Qwen4 loader appends one JSONL record per fulfilled weight
    /// (`name`, `layer`, `residency`, `dtype`, `logical_bytes`, `sha256` of the
    /// logical bytes: host-owner hash or device download) to that file after a
    /// successful fulfillment. Unset or blank writes nothing.
    pub fn load_weight_digest_path() -> Option<PathBuf> {
        developer_var("HIPFIRE_LOAD_WEIGHT_DIGEST")
            .ok()
            .filter(|path| !path.trim().is_empty())
            .map(PathBuf::from)
    }

    /// Cold-result transport of the D2 CPU split (`developer.qwen4_d2_return`,
    /// env `HIPFIRE_QWEN4_D2_RETURN`).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum D2Return {
        /// The cold slots are copied host-to-device behind the GPU tail
        /// (today's transport, the default and the kill value).
        #[default]
        Copy,
        /// The CPU writes the cold slots into a mapped, coherent host
        /// allocation that the canonical combine reads directly. Honoured only
        /// on all-cold layers; a mixed layer stays on `Copy`.
        Mapped,
    }

    impl D2Return {
        /// The spelling [`parse_d2_return`] accepts for this mode.
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Copy => "copy",
                Self::Mapped => "mapped",
            }
        }
    }

    /// Pure body of [`qwen4_d2_transport`]'s return half: unset is `None` (the
    /// caller applies the arch default, see [`default_d2_transport`]); `copy`
    /// and `mapped` (exact, lower-case) select a mode; every other spelling,
    /// including the empty string, is an error that names the accepted values.
    /// A mistyped transport must refuse the load, never silently run the other
    /// one.
    pub fn parse_d2_return_explicit(
        raw: Option<&str>,
    ) -> std::result::Result<Option<D2Return>, String> {
        match raw {
            None => Ok(None),
            Some("copy") => Ok(Some(D2Return::Copy)),
            Some("mapped") => Ok(Some(D2Return::Mapped)),
            Some(other) => Err(format!(
                "HIPFIRE_QWEN4_D2_RETURN={other:?} is not a D2 return mode: expected `mapped` or `copy`"
            )),
        }
    }

    /// [`parse_d2_return_explicit`] with the arch-independent legacy default:
    /// unset is `Copy`. The arch default lives in [`default_d2_transport`].
    pub fn parse_d2_return(raw: Option<&str>) -> std::result::Result<D2Return, String> {
        parse_d2_return_explicit(raw).map(Option::unwrap_or_default)
    }

    /// The D2 return and join requests a load resolves for one arch.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct D2Transport {
        pub ret: D2Return,
        pub join: D2Join,
        /// `true` when the arch default moved at least one request off the
        /// legacy `copy` / `host` value (no explicit env value was given for
        /// it), so the load log can name the provenance.
        pub arch_default: bool,
    }

    /// Pure default of the D2 transport requests.
    ///
    /// An explicit `HIPFIRE_QWEN4_D2_RETURN` / `HIPFIRE_QWEN4_D2_JOIN` value
    /// always wins. Unset on exact `gfx1100` the request is `mapped` + `cpwait`
    /// (byte-exact transport and scheduling change; the per-layer capability
    /// routing in `hipfire_dispatch::cpu_split` still decides: only all-cold
    /// layers qualify, mixed layers keep `copy` / `host`). A defaulted join
    /// follows the effective return: with `HIPFIRE_QWEN4_D2_RETURN=copy` the
    /// CP-wait join would only be refused per layer, so the default join is
    /// `host` and `copy` alone restores the old transport. Every other arch is
    /// the legacy `copy` / `host`. `HIPFIRE_QWEN4_D2_RETURN=copy
    /// HIPFIRE_QWEN4_D2_JOIN=host` is the kill switch on every arch.
    pub fn default_d2_transport(
        arch: &str,
        explicit_return: Option<D2Return>,
        explicit_join: Option<D2Join>,
    ) -> (D2Return, D2Join) {
        let gfx1100 = arch == "gfx1100";
        let ret = explicit_return.unwrap_or(if gfx1100 {
            D2Return::Mapped
        } else {
            D2Return::Copy
        });
        let join = explicit_join.unwrap_or(if gfx1100 && ret == D2Return::Mapped {
            D2Join::CpWait
        } else {
            D2Join::Host
        });
        (ret, join)
    }

    /// Pure body of [`qwen4_d2_transport`]: parse both raw values and apply
    /// [`default_d2_transport`]. An invalid spelling of either is an error.
    pub fn resolve_d2_transport(
        arch: &str,
        raw_return: Option<&str>,
        raw_join: Option<&str>,
    ) -> std::result::Result<D2Transport, String> {
        let explicit_return = parse_d2_return_explicit(raw_return)?;
        let explicit_join = parse_d2_join_explicit(raw_join)?;
        let (ret, join) = default_d2_transport(arch, explicit_return, explicit_join);
        Ok(D2Transport {
            ret,
            join,
            arch_default: (ret, join)
                != (
                    explicit_return.unwrap_or_default(),
                    explicit_join.unwrap_or_default(),
                ),
        })
    }

    /// `developer.qwen4_d2_return` + `developer.qwen4_d2_join` (env
    /// `HIPFIRE_QWEN4_D2_RETURN` / `HIPFIRE_QWEN4_D2_JOIN`): the D2 cold-result
    /// transport and join requests for a device of `arch`, from the process
    /// snapshot. Resolve once per load; an invalid value is an `Err` the
    /// loader must propagate.
    pub fn qwen4_d2_transport(arch: &str) -> std::result::Result<D2Transport, String> {
        resolve_d2_transport(
            arch,
            process_value("HIPFIRE_QWEN4_D2_RETURN").as_deref(),
            process_value("HIPFIRE_QWEN4_D2_JOIN").as_deref(),
        )
    }

    /// Input-readiness mechanism of the D2 CPU split (`developer.qwen4_d2_ready`,
    /// env `HIPFIRE_QWEN4_D2_READY`).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum D2Ready {
        /// `prepare` blocks on a stream synchronization after the input
        /// downloads (today's boundary, the default and the kill value).
        #[default]
        Sync,
        /// `prepare` records an event after the input downloads and polls it,
        /// falling back to the blocking synchronization after a bounded spin.
        Event,
    }

    impl D2Ready {
        /// The spelling [`parse_d2_ready`] accepts for this mode.
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Sync => "sync",
                Self::Event => "event",
            }
        }
    }

    /// Pure body of [`qwen4_d2_ready`]: unset is `Sync`; `sync` and `event`
    /// (exact, lower-case) select a mode; every other spelling, including the
    /// empty string, is an error that names the accepted values.
    pub fn parse_d2_ready(raw: Option<&str>) -> std::result::Result<D2Ready, String> {
        match raw {
            None => Ok(D2Ready::Sync),
            Some("sync") => Ok(D2Ready::Sync),
            Some("event") => Ok(D2Ready::Event),
            Some(other) => Err(format!(
                "HIPFIRE_QWEN4_D2_READY={other:?} is not a D2 readiness mode: expected `event` or `sync`"
            )),
        }
    }

    /// `developer.qwen4_d2_ready` (env `HIPFIRE_QWEN4_D2_READY`): how the D2 CPU
    /// split learns its inputs reached the host, read from the process snapshot.
    /// Resolve it once per load; an invalid value is an `Err` the loader must
    /// propagate.
    pub fn qwen4_d2_ready() -> std::result::Result<D2Ready, String> {
        parse_d2_ready(process_value("HIPFIRE_QWEN4_D2_READY").as_deref())
    }

    /// Join mechanism of the D2 CPU split (`developer.qwen4_d2_join`, env
    /// `HIPFIRE_QWEN4_D2_JOIN`).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum D2Join {
        /// The canonical combine is enqueued after the CPU arithmetic finished
        /// (today's order, the default and the kill value).
        #[default]
        Host,
        /// The canonical combine is enqueued BEFORE the CPU arithmetic behind a
        /// command-processor `hipStreamWaitValue32` that the CPU releases with a
        /// generation store. Needs the mapped return on an all-cold layer and
        /// the device's stream-wait-value capability; otherwise a logged
        /// refusal keeps `Host`.
        CpWait,
    }

    impl D2Join {
        /// The spelling [`parse_d2_join`] accepts for this mode.
        pub fn as_str(self) -> &'static str {
            match self {
                Self::Host => "host",
                Self::CpWait => "cpwait",
            }
        }
    }

    /// Pure body of [`qwen4_d2_transport`]'s join half: unset is `None` (the
    /// caller applies the arch default, see [`default_d2_transport`]); `host`
    /// and `cpwait` (exact, lower-case) select a mode; every other spelling,
    /// including the empty string, is an error that names the accepted values.
    pub fn parse_d2_join_explicit(
        raw: Option<&str>,
    ) -> std::result::Result<Option<D2Join>, String> {
        match raw {
            None => Ok(None),
            Some("host") => Ok(Some(D2Join::Host)),
            Some("cpwait") => Ok(Some(D2Join::CpWait)),
            Some(other) => Err(format!(
                "HIPFIRE_QWEN4_D2_JOIN={other:?} is not a D2 join mode: expected `cpwait` or `host`"
            )),
        }
    }

    /// [`parse_d2_join_explicit`] with the arch-independent legacy default:
    /// unset is `Host`. The arch default lives in [`default_d2_transport`].
    pub fn parse_d2_join(raw: Option<&str>) -> std::result::Result<D2Join, String> {
        parse_d2_join_explicit(raw).map(Option::unwrap_or_default)
    }

    /// How much of a split step the GPU takes (`memory.offload_passback_share`).
    ///
    /// Read only when [`offload_exec`] is [`OffloadExec::Passback`]; in any other
    /// mode the key is inert (see [`stray_passback_share_notice`]).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum PassbackShare {
        /// Scheduled from this host's own measured engine rates.
        Auto,
        /// Pinned by the user. `0.0` disables the pass-back.
        Share(f64),
    }

    impl std::fmt::Display for PassbackShare {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                PassbackShare::Auto => write!(f, "auto"),
                PassbackShare::Share(v) => write!(f, "{v}"),
            }
        }
    }

    /// Pure truth table behind [`offload_passback_share`]. Unset, empty and
    /// unknown all resolve to `Auto`; the schema ([`super::ValueRule::PassbackShare`])
    /// rejects an out-of-range number before this runs, so the fallback here is
    /// belt-and-braces rather than the primary guard.
    pub fn parse_passback_share(raw: Option<&str>) -> PassbackShare {
        match raw.map(|v| v.trim()) {
            Some("auto") => PassbackShare::Auto,
            Some(v) => match v.parse::<f64>() {
                Ok(f) if (0.0..=0.5).contains(&f) => PassbackShare::Share(f),
                _ => PassbackShare::Auto,
            },
            None => PassbackShare::Auto,
        }
    }

    /// The configured [`PassbackShare`] from the process snapshot.
    pub fn offload_passback_share() -> PassbackShare {
        parse_passback_share(process_value("HIPFIRE_OFFLOAD_PASSBACK_SHARE").as_deref())
    }

    /// `true` when a share is set but the mode cannot read it, so the load can
    /// print one informational line instead of silently ignoring the key.
    ///
    /// The mode is the selector and the share is its modifier, so a share in a
    /// `pcie`/`cpu` run is not a misconfiguration to refuse — just an inert key
    /// worth naming once.
    pub fn stray_passback_share_configured() -> bool {
        let Some(raw) = process_value("HIPFIRE_OFFLOAD_PASSBACK_SHARE") else {
            return false;
        };
        let mode = offload_exec();
        mode != OffloadExec::Passback && raw.trim() != "" && raw.trim() != "auto"
    }

    // Placement arithmetic (`largest_fitting_tail`, the policy→split call) lives
    // in `hipfire_runtime::offload`: it is placement, not configuration
    // vocabulary, and it needs a device measurement this crate cannot take.
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn d2_return_parses_mapped_copy_and_refuses_everything_else() {
            assert_eq!(parse_d2_return(None), Ok(D2Return::Copy));
            assert_eq!(D2Return::default(), D2Return::Copy);
            assert_eq!(parse_d2_return(Some("copy")), Ok(D2Return::Copy));
            assert_eq!(parse_d2_return(Some("mapped")), Ok(D2Return::Mapped));
            for bad in [
                "", "Mapped", "COPY", " mapped", "1", "0", "true", "pinned", "mapped ",
            ] {
                let error = parse_d2_return(Some(bad)).unwrap_err();
                assert!(
                    error.contains("HIPFIRE_QWEN4_D2_RETURN"),
                    "{bad:?}: {error}"
                );
                assert!(error.contains("`mapped` or `copy`"), "{bad:?}: {error}");
            }
            for mode in [D2Return::Copy, D2Return::Mapped] {
                assert_eq!(parse_d2_return(Some(mode.as_str())), Ok(mode));
            }
        }

        #[test]
        fn d2_ready_parses_sync_event_and_refuses_everything_else() {
            assert_eq!(parse_d2_ready(None), Ok(D2Ready::Sync));
            assert_eq!(D2Ready::default(), D2Ready::Sync);
            assert_eq!(parse_d2_ready(Some("sync")), Ok(D2Ready::Sync));
            assert_eq!(parse_d2_ready(Some("event")), Ok(D2Ready::Event));
            for bad in [
                "", "Event", "SYNC", " event", "1", "0", "true", "cpwait", "event ",
            ] {
                let error = parse_d2_ready(Some(bad)).unwrap_err();
                assert!(error.contains("HIPFIRE_QWEN4_D2_READY"), "{bad:?}: {error}");
                assert!(error.contains("`event` or `sync`"), "{bad:?}: {error}");
            }
            for mode in [D2Ready::Sync, D2Ready::Event] {
                assert_eq!(parse_d2_ready(Some(mode.as_str())), Ok(mode));
            }
        }

        #[test]
        fn d2_join_parses_host_cpwait_and_refuses_everything_else() {
            assert_eq!(parse_d2_join(None), Ok(D2Join::Host));
            assert_eq!(D2Join::default(), D2Join::Host);
            assert_eq!(parse_d2_join(Some("host")), Ok(D2Join::Host));
            assert_eq!(parse_d2_join(Some("cpwait")), Ok(D2Join::CpWait));
            for bad in [
                "", "CpWait", "HOST", " cpwait", "1", "0", "true", "event", "cpwait ",
            ] {
                let error = parse_d2_join(Some(bad)).unwrap_err();
                assert!(error.contains("HIPFIRE_QWEN4_D2_JOIN"), "{bad:?}: {error}");
                assert!(error.contains("`cpwait` or `host`"), "{bad:?}: {error}");
            }
            for mode in [D2Join::Host, D2Join::CpWait] {
                assert_eq!(parse_d2_join(Some(mode.as_str())), Ok(mode));
            }
        }

        #[test]
        fn d2_default_transport_table_over_arch_and_explicit_values() {
            use D2Join::{CpWait, Host};
            use D2Return::{Copy, Mapped};
            let rows: &[(&str, Option<D2Return>, Option<D2Join>, (D2Return, D2Join))] = &[
                // gfx1100: unset -> mapped + cpwait; explicit values always win.
                ("gfx1100", None, None, (Mapped, CpWait)),
                ("gfx1100", Some(Copy), Some(Host), (Copy, Host)), // kill switch
                ("gfx1100", Some(Copy), None, (Copy, Host)),       // defaulted join follows return
                ("gfx1100", None, Some(Host), (Mapped, Host)),
                ("gfx1100", Some(Mapped), None, (Mapped, CpWait)),
                ("gfx1100", None, Some(CpWait), (Mapped, CpWait)),
                ("gfx1100", Some(Mapped), Some(Host), (Mapped, Host)),
                ("gfx1100", Some(Copy), Some(CpWait), (Copy, CpWait)), // refused per layer
                ("gfx1100", Some(Mapped), Some(CpWait), (Mapped, CpWait)),
                // Every other arch keeps the legacy copy / host defaults.
                ("gfx1201", None, None, (Copy, Host)),
                ("gfx1151", None, None, (Copy, Host)),
                ("gfx1030", None, None, (Copy, Host)),
                ("gfx1100x", None, None, (Copy, Host)),
                ("", None, None, (Copy, Host)),
                ("gfx1201", Some(Mapped), None, (Mapped, Host)),
                ("gfx1201", None, Some(CpWait), (Copy, CpWait)),
                ("gfx1201", Some(Mapped), Some(CpWait), (Mapped, CpWait)),
                ("gfx1201", Some(Copy), Some(Host), (Copy, Host)),
            ];
            for (arch, ret, join, want) in rows {
                assert_eq!(
                    default_d2_transport(arch, *ret, *join),
                    *want,
                    "{arch} return={ret:?} join={join:?}"
                );
            }
        }

        #[test]
        fn d2_transport_resolution_parses_explicit_values_and_reports_provenance() {
            let resolve = resolve_d2_transport;
            let t = resolve("gfx1100", None, None).unwrap();
            assert_eq!(
                (t.ret, t.join, t.arch_default),
                (D2Return::Mapped, D2Join::CpWait, true)
            );
            let t = resolve("gfx1100", Some("copy"), Some("host")).unwrap();
            assert_eq!(
                (t.ret, t.join, t.arch_default),
                (D2Return::Copy, D2Join::Host, false)
            );
            let t = resolve("gfx1100", Some("copy"), None).unwrap();
            assert_eq!(
                (t.ret, t.join, t.arch_default),
                (D2Return::Copy, D2Join::Host, false)
            );
            let t = resolve("gfx1100", Some("mapped"), Some("cpwait")).unwrap();
            assert_eq!(
                (t.ret, t.join, t.arch_default),
                (D2Return::Mapped, D2Join::CpWait, false)
            );
            let t = resolve("gfx1201", None, None).unwrap();
            assert_eq!(
                (t.ret, t.join, t.arch_default),
                (D2Return::Copy, D2Join::Host, false)
            );
            // An invalid spelling refuses the load on every arch, never defaults.
            for arch in ["gfx1100", "gfx1201"] {
                let error = resolve(arch, Some("Mapped"), None).unwrap_err();
                assert!(error.contains("HIPFIRE_QWEN4_D2_RETURN"), "{error}");
                let error = resolve(arch, None, Some("")).unwrap_err();
                assert!(error.contains("HIPFIRE_QWEN4_D2_JOIN"), "{error}");
            }
        }

        #[test]
        fn offload_exec_roundtrip() {
            // Only "cpu" (trimmed, case-insensitive) selects CPU execution;
            // everything else — including the absent value — is the pcie default,
            // so a typo can never move work to the CPU silently.
            assert_eq!(parse_offload_exec(Some("cpu")), OffloadExec::Cpu);
            assert_eq!(parse_offload_exec(Some(" cpu ")), OffloadExec::Cpu);
            assert_eq!(parse_offload_exec(Some("CPU")), OffloadExec::Cpu);
            assert_eq!(parse_offload_exec(None), OffloadExec::Pcie);
            assert_eq!(parse_offload_exec(Some("")), OffloadExec::Pcie);
            assert_eq!(parse_offload_exec(Some("pcie")), OffloadExec::Pcie);
            assert_eq!(parse_offload_exec(Some("PCIE")), OffloadExec::Pcie);
            assert_eq!(parse_offload_exec(Some("banana")), OffloadExec::Pcie);
            assert_eq!(parse_offload_exec(Some("cpu0")), OffloadExec::Pcie);
            // Pass-back is its own mode; whitespace/case trimmed like "cpu".
            assert_eq!(parse_offload_exec(Some("passback")), OffloadExec::Passback);
            assert_eq!(
                parse_offload_exec(Some(" Passback ")),
                OffloadExec::Passback
            );
            assert_eq!(parse_offload_exec(Some("PASSBACK")), OffloadExec::Passback);
            assert_eq!(parse_offload_exec(Some("passback0")), OffloadExec::Pcie);
            // Display is the wire spelling the registry/TOML round-trips.
            assert_eq!(OffloadExec::Pcie.to_string(), "pcie");
            assert_eq!(OffloadExec::Cpu.to_string(), "cpu");
            assert_eq!(OffloadExec::Passback.to_string(), "passback");
        }

        #[test]
        fn qwen4_direct_load_is_default_on_and_only_zero_is_off() {
            assert!(parse_qwen4_direct_load(None));
            assert!(parse_qwen4_direct_load(Some("")));
            assert!(parse_qwen4_direct_load(Some("1")));
            assert!(parse_qwen4_direct_load(Some(" 1 ")));
            assert!(parse_qwen4_direct_load(Some("true")));
            assert!(!parse_qwen4_direct_load(Some("0")));
            assert!(!parse_qwen4_direct_load(Some(" 0 ")));
        }

        #[test]
        fn offload_exec_explicit_truth_table() {
            // Unset and blank are "not chosen": the loader may pick by
            // measurement. Any spelling the user did type is chosen, and an
            // unknown one still fails closed to `pcie` (never to the CPU).
            assert_eq!(parse_offload_exec_explicit(None), None);
            assert_eq!(parse_offload_exec_explicit(Some("")), None);
            assert_eq!(parse_offload_exec_explicit(Some("  ")), None);
            assert_eq!(
                parse_offload_exec_explicit(Some("pcie")),
                Some(OffloadExec::Pcie)
            );
            assert_eq!(
                parse_offload_exec_explicit(Some(" PCIE ")),
                Some(OffloadExec::Pcie)
            );
            assert_eq!(
                parse_offload_exec_explicit(Some("cpu")),
                Some(OffloadExec::Cpu)
            );
            assert_eq!(
                parse_offload_exec_explicit(Some("passback")),
                Some(OffloadExec::Passback)
            );
            assert_eq!(
                parse_offload_exec_explicit(Some("banana")),
                Some(OffloadExec::Pcie)
            );
        }

        #[test]
        fn passback_share_roundtrip() {
            // Absent/empty/unknown all schedule.
            assert_eq!(parse_passback_share(None), PassbackShare::Auto);
            assert_eq!(parse_passback_share(Some("")), PassbackShare::Auto);
            assert_eq!(parse_passback_share(Some("auto")), PassbackShare::Auto);
            assert_eq!(parse_passback_share(Some(" auto ")), PassbackShare::Auto);
            // Both ends of the legal range are pinned shares; 0 disables.
            assert_eq!(parse_passback_share(Some("0")), PassbackShare::Share(0.0));
            assert_eq!(
                parse_passback_share(Some("0.375")),
                PassbackShare::Share(0.375)
            );
            assert_eq!(parse_passback_share(Some("0.5")), PassbackShare::Share(0.5));
            // Out of range and garbage fail closed to scheduling (the schema
            // rejects them earlier; this is the second guard).
            assert_eq!(parse_passback_share(Some("0.6")), PassbackShare::Auto);
            assert_eq!(parse_passback_share(Some("-0.1")), PassbackShare::Auto);
            assert_eq!(parse_passback_share(Some("banana")), PassbackShare::Auto);
            assert_eq!(PassbackShare::Auto.to_string(), "auto");
            assert_eq!(PassbackShare::Share(0.25).to_string(), "0.25");
        }

        #[test]
        fn gpu_layer_budget_roundtrip() {
            // Unset resolves to full residency — the zero-diff baseline.
            assert_eq!(parse_gpu_layer_budget(None), OffloadBudget::Full);
            // Empty string is unset, not auto.
            assert_eq!(parse_gpu_layer_budget(Some("")), OffloadBudget::Full);
            // "auto" and "-1" both select auto-fit (case-insensitive).
            assert_eq!(parse_gpu_layer_budget(Some("auto")), OffloadBudget::Auto);
            assert_eq!(parse_gpu_layer_budget(Some("-1")), OffloadBudget::Auto);
            assert_eq!(parse_gpu_layer_budget(Some("AUTO")), OffloadBudget::Auto);
            // A non-negative integer pins that many resident layers (trimmed).
            assert_eq!(parse_gpu_layer_budget(Some("3")), OffloadBudget::Layers(3));
            assert_eq!(
                parse_gpu_layer_budget(Some(" 12 ")),
                OffloadBudget::Layers(12)
            );
            // Garbage, negatives other than -1, and anything unparseable fail
            // closed to full residency rather than forcing an offload.
            assert_eq!(parse_gpu_layer_budget(Some("banana")), OffloadBudget::Full);
            assert_eq!(parse_gpu_layer_budget(Some("-2")), OffloadBudget::Full);
        }

        #[test]
        fn ple_hot_store_bytes_parse() {
            assert_eq!(parse_ple_hot_store_bytes(None), PleHotStoreBudget::Bytes(0));
            assert_eq!(
                parse_ple_hot_store_bytes(Some("")),
                PleHotStoreBudget::Bytes(0)
            );
            assert_eq!(
                parse_ple_hot_store_bytes(Some("auto")),
                PleHotStoreBudget::Auto
            );
            assert_eq!(
                parse_ple_hot_store_bytes(Some("0")),
                PleHotStoreBudget::Bytes(0)
            );
            assert_eq!(
                parse_ple_hot_store_bytes(Some(" 16000000000 ")),
                PleHotStoreBudget::Bytes(16_000_000_000)
            );
            assert_eq!(
                parse_ple_hot_store_bytes(Some("-1")),
                PleHotStoreBudget::Bytes(0)
            );
            let field = crate::FIELDS
                .iter()
                .find(|field| field.key == "memory.ple_hot_store_bytes")
                .expect("ple_hot_store_bytes is in the schema");
            assert!(field
                .validate(&crate::ConfigValue::String("auto".into()))
                .is_ok());
            assert!(field.validate(&crate::ConfigValue::Integer(0)).is_ok());
            assert!(field
                .validate(&crate::ConfigValue::String("on".into()))
                .is_err());
            assert!(field.validate(&crate::ConfigValue::Integer(-1)).is_err());
            assert!(matches!(
                field.parse_cli("auto"),
                Ok(crate::ConfigValue::String(_))
            ));
            assert!(matches!(
                field.parse_cli("42"),
                Ok(crate::ConfigValue::Integer(42))
            ));
        }

        #[test]
        fn ple_format_parse_resolve_and_names() {
            assert_eq!(
                PleFormatRequest::parse("auto"),
                Some(PleFormatRequest::Auto)
            );
            assert_eq!(PleFormatRequest::parse("q4"), Some(PleFormatRequest::Q4));
            assert_eq!(PleFormatRequest::parse(" Q8 "), Some(PleFormatRequest::Q8));
            assert_eq!(PleFormatRequest::parse(""), None);
            assert_eq!(PleFormatRequest::parse("q5"), None);
            for request in [
                PleFormatRequest::Auto,
                PleFormatRequest::Q4,
                PleFormatRequest::Q8,
            ] {
                assert_eq!(PleFormatRequest::parse(request.name()), Some(request));
            }
            assert_eq!(PleFormat::from_name("q4"), Some(PleFormat::Q4));
            assert_eq!(PleFormat::from_name("q8"), Some(PleFormat::Q8));
            assert_eq!(PleFormat::from_name("auto"), None);
            assert_eq!(PleFormat::Q4.name(), "q4");
            assert_eq!(PleFormat::Q8.name(), "q8");
            assert_eq!(PleFormat::Q4.sidecar_suffix(), "ple-q4");
            assert_eq!(PleFormat::Q8.sidecar_suffix(), "ple-q8");
            // Explicit requests ignore the arch; auto is q4 only on gfx1151.
            for arch in ["gfx1151", "gfx1100", "gfx1201", ""] {
                assert_eq!(PleFormatRequest::Q4.resolve(arch), PleFormat::Q4);
                assert_eq!(PleFormatRequest::Q8.resolve(arch), PleFormat::Q8);
            }
            assert_eq!(PleFormatRequest::Auto.resolve("gfx1151"), PleFormat::Q4);
            assert_eq!(PleFormatRequest::Auto.resolve("gfx1100"), PleFormat::Q8);
            assert_eq!(PleFormatRequest::Auto.resolve("gfx1201"), PleFormat::Q8);
            assert_eq!(PleFormatRequest::Auto.resolve(""), PleFormat::Q8);
        }

        #[test]
        fn ple_sidecar_path_appends_to_full_file_name() {
            let model = Path::new("/m/qwen3.8-flash-next-base.mq4-xts");
            assert_eq!(
                ple_sidecar_path(model, PleFormat::Q4),
                PathBuf::from("/m/qwen3.8-flash-next-base.mq4-xts.ple-q4")
            );
            assert_eq!(
                ple_sidecar_path(model, PleFormat::Q8),
                PathBuf::from("/m/qwen3.8-flash-next-base.mq4-xts.ple-q8")
            );
            assert_eq!(
                ple_sidecar_path(Path::new("model"), PleFormat::Q4),
                PathBuf::from("model.ple-q4")
            );
        }

        #[test]
        fn trunk_overlay_sidecar_path_appends_to_full_file_name() {
            assert_eq!(
                trunk_overlay_sidecar_path(Path::new("/m/qwen3.8-flash-next.mq4-xts")),
                PathBuf::from("/m/qwen3.8-flash-next.mq4-xts.trunk-overlay")
            );
            assert_eq!(
                trunk_overlay_sidecar_path(Path::new("model")),
                PathBuf::from("model.trunk-overlay")
            );
        }

        #[test]
        fn expert_hotset_sidecar_path_appends_to_full_file_name() {
            assert_eq!(
                expert_hotset_sidecar_path(Path::new("/m/qwen3.8-flash-next.mq4-xts")),
                PathBuf::from("/m/qwen3.8-flash-next.mq4-xts.expert-hotset")
            );
            assert_eq!(
                expert_hotset_sidecar_path(Path::new("model")),
                PathBuf::from("model.expert-hotset")
            );
        }

        #[test]
        fn ple_format_key_defaults_auto_with_env_override() {
            let field = crate::FIELDS
                .iter()
                .find(|field| field.key == "memory.ple_format")
                .expect("ple_format is in the schema");
            assert_eq!(field.legacy_key, "ple_format");
            assert_eq!(field.env_compat, Some("HIPFIRE_PLE_FORMAT"));
            assert_eq!(
                field.default.to_value(),
                crate::ConfigValue::String("auto".into())
            );
            for value in ["auto", "q4", "q8"] {
                assert_eq!(
                    field.parse_cli(value).unwrap(),
                    crate::ConfigValue::String(value.into())
                );
            }
            assert!(field.parse_cli("q5").is_err());
            assert!(field.validate(&crate::ConfigValue::Integer(4)).is_err());
            // HIPFIRE_PLE_FORMAT is unique to this test; no other test reads it.
            std::env::set_var("HIPFIRE_PLE_FORMAT", "q4");
            let env_layer = crate::load_env_layer().expect("env layer");
            std::env::remove_var("HIPFIRE_PLE_FORMAT");
            let process = crate::ProcessConfig::from_resolved(
                &crate::resolve([crate::NamedLayer {
                    source: crate::ConfigSource::LegacyEnv {
                        name: "HIPFIRE_PLE_FORMAT".into(),
                    },
                    layer: env_layer,
                }])
                .expect("env resolve"),
            )
            .expect("process");
            let raw = process.legacy_value("HIPFIRE_PLE_FORMAT");
            assert_eq!(raw.as_deref(), Some("q4"));
            assert_eq!(parse_ple_format(raw.as_deref()), PleFormatRequest::Q4);
            // Unset or unparseable resolves to the default.
            let defaults =
                crate::ProcessConfig::from_resolved(&crate::resolve([]).expect("empty resolve"))
                    .expect("process");
            assert_eq!(
                parse_ple_format(defaults.legacy_value("HIPFIRE_PLE_FORMAT").as_deref()),
                PleFormatRequest::Auto
            );
            assert_eq!(parse_ple_format(None), PleFormatRequest::Auto);
            assert_eq!(parse_ple_format(Some("bogus")), PleFormatRequest::Auto);
        }

        #[test]
        fn ple_prewarm_bytes_defaults_auto_with_env_override() {
            assert_eq!(parse_ple_prewarm_bytes(None), PleHotStoreBudget::Auto);
            assert_eq!(parse_ple_prewarm_bytes(Some("")), PleHotStoreBudget::Auto);
            assert_eq!(parse_ple_prewarm_bytes(Some("  ")), PleHotStoreBudget::Auto);
            assert_eq!(
                parse_ple_prewarm_bytes(Some("auto")),
                PleHotStoreBudget::Auto
            );
            assert_eq!(
                parse_ple_prewarm_bytes(Some("0")),
                PleHotStoreBudget::Bytes(0)
            );
            assert_eq!(
                parse_ple_prewarm_bytes(Some(" 16000000000 ")),
                PleHotStoreBudget::Bytes(16_000_000_000)
            );
            // An unparseable value (the config layer rejects those first) fails
            // closed to off, like the hot-store budget.
            assert_eq!(
                parse_ple_prewarm_bytes(Some("-1")),
                PleHotStoreBudget::Bytes(0)
            );
            let field = crate::FIELDS
                .iter()
                .find(|field| field.key == "memory.ple_prewarm_bytes")
                .expect("ple_prewarm_bytes is in the schema");
            assert_eq!(field.legacy_key, "ple_prewarm_bytes");
            assert_eq!(field.env_compat, Some("HIPFIRE_PLE_PREWARM_BYTES"));
            assert_eq!(
                field.default.to_value(),
                crate::ConfigValue::String("auto".into())
            );
            assert!(field
                .validate(&crate::ConfigValue::String("auto".into()))
                .is_ok());
            assert!(field.validate(&crate::ConfigValue::Integer(0)).is_ok());
            assert!(field
                .validate(&crate::ConfigValue::String("on".into()))
                .is_err());
            assert!(field.validate(&crate::ConfigValue::Integer(-1)).is_err());
            assert!(matches!(
                field.parse_cli("auto"),
                Ok(crate::ConfigValue::String(_))
            ));
            assert!(matches!(
                field.parse_cli("42"),
                Ok(crate::ConfigValue::Integer(42))
            ));
            // HIPFIRE_PLE_PREWARM_BYTES is unique to this test; no other test reads it.
            std::env::set_var("HIPFIRE_PLE_PREWARM_BYTES", "0");
            let env_layer = crate::load_env_layer().expect("env layer");
            std::env::remove_var("HIPFIRE_PLE_PREWARM_BYTES");
            let process = crate::ProcessConfig::from_resolved(
                &crate::resolve([crate::NamedLayer {
                    source: crate::ConfigSource::LegacyEnv {
                        name: "HIPFIRE_PLE_PREWARM_BYTES".into(),
                    },
                    layer: env_layer,
                }])
                .expect("env resolve"),
            )
            .expect("process");
            let raw = process.legacy_value("HIPFIRE_PLE_PREWARM_BYTES");
            assert_eq!(raw.as_deref(), Some("0"));
            assert_eq!(
                parse_ple_prewarm_bytes(raw.as_deref()),
                PleHotStoreBudget::Bytes(0)
            );
            // Without the env the default (auto) resolves.
            let defaults =
                crate::ProcessConfig::from_resolved(&crate::resolve([]).expect("empty resolve"))
                    .expect("process");
            assert_eq!(
                parse_ple_prewarm_bytes(
                    defaults
                        .legacy_value("HIPFIRE_PLE_PREWARM_BYTES")
                        .as_deref()
                ),
                PleHotStoreBudget::Auto
            );
        }

        #[test]
        fn ple_load_fill_bytes_is_an_auto_integer_defaulting_to_auto() {
            let field = crate::FIELDS
                .iter()
                .find(|field| field.key == "memory.ple_load_fill_bytes")
                .expect("ple_load_fill_bytes is in the schema");
            assert_eq!(field.legacy_key, "ple_load_fill_bytes");
            assert_eq!(field.env_compat, Some("HIPFIRE_PLE_LOAD_FILL_BYTES"));
            assert_eq!(
                field.default.to_value(),
                crate::ConfigValue::String("auto".into())
            );
            assert!(field
                .validate(&crate::ConfigValue::String("auto".into()))
                .is_ok());
            assert!(field.validate(&crate::ConfigValue::Integer(0)).is_ok());
            assert!(field.validate(&crate::ConfigValue::Integer(-1)).is_err());
            let defaults =
                crate::ProcessConfig::from_resolved(&crate::resolve([]).expect("empty resolve"))
                    .expect("process");
            assert_eq!(
                parse_ple_prewarm_bytes(
                    defaults
                        .legacy_value("HIPFIRE_PLE_LOAD_FILL_BYTES")
                        .as_deref()
                ),
                PleHotStoreBudget::Auto
            );
        }

        #[test]
        fn budget_keys_default_to_full_and_preserve_explicit_placement() {
            use OffloadBudget::{Auto, Full, Layers};
            let defaults =
                crate::ProcessConfig::from_resolved(&crate::resolve([]).unwrap()).unwrap();
            for (key, env) in [
                ("memory.gpu_layer_budget", "HIPFIRE_GPU_LAYER_BUDGET"),
                ("memory.moe_expert_budget", "HIPFIRE_MOE_EXPERT_BUDGET"),
            ] {
                assert_eq!(
                    crate::field(key).unwrap().default.to_value(),
                    crate::ConfigValue::Null
                );
                assert_eq!(
                    parse_gpu_layer_budget(defaults.legacy_value(env).as_deref()),
                    Full
                );
                for (spelling, expected) in [
                    ("null", Full),
                    ("auto", Auto),
                    ("-1", Auto),
                    ("26", Layers(26)),
                    ("0", Layers(0)),
                ] {
                    let mut layer = crate::ConfigLayer::default();
                    layer.set_cli(key, spelling).unwrap();
                    let resolved = crate::resolve([crate::NamedLayer {
                        source: crate::ConfigSource::OneShot {
                            argument: format!("{key}={spelling}"),
                        },
                        layer,
                    }])
                    .unwrap();
                    let process = crate::ProcessConfig::from_resolved(&resolved).unwrap();
                    for budget_env in ["HIPFIRE_GPU_LAYER_BUDGET", "HIPFIRE_MOE_EXPERT_BUDGET"] {
                        assert_eq!(
                            parse_gpu_layer_budget(process.legacy_value(budget_env).as_deref()),
                            if budget_env == env { expected } else { Full },
                            "{key}={spelling}, {budget_env}"
                        );
                    }
                }
            }
        }

        #[test]
        fn budget_keys_accept_the_auto_spelling() {
            // The TUI rows and the help text both use `auto`; the wire spelling
            // `-1` is equivalent. Both must survive CLI/env validation, or the
            // documented spelling is unusable.
            for key in ["memory.gpu_layer_budget", "memory.moe_expert_budget"] {
                let field = crate::field(key).expect("budget key is in the schema");
                for spelling in ["auto", "AUTO", "-1", "26", "null"] {
                    let parsed = field
                        .parse_cli(spelling)
                        .unwrap_or_else(|e| panic!("{key}={spelling}: {e}"));
                    field.validate(&parsed).unwrap();
                }
                // Garbage still fails closed rather than being read as a count.
                assert!(field.parse_cli("banana").is_err(), "{key}");
            }
        }

        #[test]
        fn gpu_layer_budget_display() {
            assert_eq!(OffloadBudget::Full.to_string(), "full");
            assert_eq!(OffloadBudget::Auto.to_string(), "auto");
            assert_eq!(OffloadBudget::Layers(5).to_string(), "5");
        }

        #[test]
        fn spill_support_truth_table() {
            use SpillRefusal::*;
            use SpillSupport::{Refused, Supported};
            // Clean single-GPU non-VL non-paged non-REAP MoE: the only
            // supported shape (dense included — the policy is MoE-blind).
            for arch_id in [5u32, 6, 11] {
                assert_eq!(
                    spill_support(arch_id, 1, 1, None, false, false, false),
                    Supported,
                    "arch {arch_id}"
                );
            }
            // Each refusal dimension, in gate priority order.
            assert_eq!(
                spill_support(9, 1, 1, None, false, false, false),
                Refused(Arch(9))
            );
            assert_eq!(
                spill_support(6, 2, 1, Some(4), false, false, false),
                Refused(Tp(2)),
                "tp wins over the EP pinning behind it"
            );
            assert_eq!(
                spill_support(6, 1, 2, None, false, false, false),
                Refused(Pp(2))
            );
            assert_eq!(
                spill_support(6, 1, 1, Some(4), false, false, false),
                Refused(ExpertParallel(4))
            );
            assert_eq!(
                spill_support(6, 1, 1, None, false, false, true),
                Refused(VisionTower)
            );
            assert_eq!(
                spill_support(6, 1, 1, None, true, false, false),
                Refused(PagedExperts)
            );
            assert_eq!(
                spill_support(6, 1, 1, None, false, true, false),
                Refused(ReapKeepMap)
            );
            // `None`/zero experts is not expert-parallel — no other
            // dimension failing means supported.
            assert_eq!(
                spill_support(6, 1, 1, Some(0), false, false, false),
                Supported
            );
        }
    }
}
