// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Writable TOML configuration for the TUI Settings tab.
//!
//! Validation and persistence are delegated to `hipfire-config`; this module
//! only retains the TUI's curated cycling order. The TUI and native CLI now
//! mutate the same sparse `~/.hipfire/config.toml` document.

use hipfire_config::{
    effective::Choice, field, is_deprecated_key, is_developer_key, is_typed_bool_developer_key,
    load_global, parse_typed_bool, write_global_toml, ConfigLayer, ConfigPaths, ConfigValue,
    ValueRule,
};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    Enum(&'static [&'static str]),
    Bool,
    /// Numeric kinds carry no bounds: `hipfire-config` is the sole validator,
    /// and [`schema_range`] reads the schema's own range for display.
    Float,
    Int,
    FreeStr {
        require_existing_file: bool,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub key: &'static str,
    pub kind: FieldKind,
}

/// KV cache modes in cycling order. Every entry is in the schema's own list
/// (`memory.kv_cache`); the deprecated asymN/turboN aliases and the
/// `legacy-*` Givens opt-outs are left out. A test pins this list to the schema
/// minus those, so a mode the schema gains cannot silently go unselectable.
const KV_CACHE: &[&str] = &[
    "auto", "q8", "fwht4", "fwht3", "fwht2", "fp8", "bf16", "f16", "f32",
];
const KV_ADAPTIVE: &[&str] = &[
    "off",
    "conservative",
    "balanced",
    "aggressive",
    "advanced:k=fwht4,v=lloyd4",
    "advanced:k=fwht4,v=lloyd3",
    "advanced:k=fwht4,v=lloyd2",
    "advanced:k=fwht3,v=lloyd4",
    "advanced:k=fwht3,v=lloyd3",
    "advanced:k=fwht3,v=lloyd2",
    "advanced:k=fwht2,v=lloyd4",
    "advanced:k=fwht2,v=lloyd3",
    "advanced:k=fwht2,v=lloyd2",
];
const KV_BACKENDS: &[&str] = &["legacy", "vmm"];
const AUTO_ON_OFF: &[&str] = &["off", "on", "auto"];
const THINKING: &[&str] = &["on", "off"];
const THINKING_BUDGET: &[&str] = &["low", "med", "high", "xhigh", "max", "uncapped"];
const FLASH_MODE: &[&str] = &["auto", "always", "never"];

/// The typed-boolean developer key the Settings screen edits (Flash-Next
/// experts). Written as a TOML boolean.
pub const SYM_IU4_KEY: &str = "developer.qwen4_moe_sym_iu4";

/// Overrides of the schema-derived kind, only where the curated list has a
/// better option order than the schema's own. Every other non-deprecated
/// schema field gets its kind from its `ValueRule` ([`field_spec`]).
/// `hipfire-config` is the sole validator; deprecated CASK/PFlash keys are
/// deliberately absent (read-only in the TUI).
pub const EDITABLE_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        key: "kv_cache",
        kind: FieldKind::Enum(KV_CACHE),
    },
    FieldSpec {
        key: "mtp_mode",
        kind: FieldKind::Enum(AUTO_ON_OFF),
    },
    FieldSpec {
        key: "dflash_mode",
        kind: FieldKind::Enum(AUTO_ON_OFF),
    },
    FieldSpec {
        key: "thinking",
        kind: FieldKind::Enum(THINKING),
    },
    FieldSpec {
        key: "thinking_budget",
        kind: FieldKind::Enum(THINKING_BUDGET),
    },
    FieldSpec {
        key: "flash_mode",
        kind: FieldKind::Enum(FLASH_MODE),
    },
    FieldSpec {
        key: "mmq_screen",
        kind: FieldKind::Enum(AUTO_ON_OFF),
    },
];

/// The editing kind a schema rule implies.
fn kind_from_rule(rule: &ValueRule) -> FieldKind {
    match rule {
        ValueRule::Bool => FieldKind::Bool,
        ValueRule::Integer { .. }
        | ValueRule::NullableInteger { .. }
        | ValueRule::AutoInteger { .. }
        | ValueRule::NullableIntegerOrAuto { .. } => FieldKind::Int,
        ValueRule::Float { .. } | ValueRule::NullableFloat { .. } => FieldKind::Float,
        ValueRule::Enum(values) | ValueRule::NullableEnum(values) => FieldKind::Enum(values),
        ValueRule::AutoBool => FieldKind::Enum(AUTO_ON_OFF),
        ValueRule::KvAdaptive => FieldKind::Enum(KV_ADAPTIVE),
        ValueRule::KvBackend => FieldKind::Enum(KV_BACKENDS),
        ValueRule::PathOrEmpty => FieldKind::FreeStr {
            require_existing_file: true,
        },
        // A free string rather than an enum: `PassbackShare` is `auto` or a
        // continuum (0, 0.5] that a fixed list cannot express; the schema rule
        // is the validator, so a bad typed value is rejected by `write_value`.
        ValueRule::String
        | ValueRule::NonEmptyString
        | ValueRule::Host
        | ValueRule::AbsolutePathOrEmpty
        | ValueRule::NullableString
        | ValueRule::PassbackShare
        | ValueRule::Deepseek4Placement => FieldKind::FreeStr {
            require_existing_file: false,
        },
    }
}

/// Every editable key: canonical key → spec. Built once from the schema (each
/// non-deprecated field, kind from its rule) with the curated overrides applied
/// on top, plus the typed-boolean developer key.
fn spec_table() -> &'static [(&'static str, FieldSpec)] {
    static TABLE: std::sync::LazyLock<Vec<(&'static str, FieldSpec)>> =
        std::sync::LazyLock::new(build_spec_table);
    &TABLE
}

fn build_spec_table() -> Vec<(&'static str, FieldSpec)> {
    let mut table: Vec<(&'static str, FieldSpec)> = hipfire_config::fields()
        .iter()
        .filter(|schema| !is_deprecated_key(schema.key))
        .map(|schema| {
            let curated = EDITABLE_FIELDS
                .iter()
                .find(|spec| field(spec.key).is_some_and(|f| f.key == schema.key));
            let spec = match curated {
                Some(spec) => *spec,
                None => FieldSpec {
                    key: schema.key,
                    kind: kind_from_rule(&schema.rule),
                },
            };
            (schema.key, spec)
        })
        .collect();
    table.push((
        SYM_IU4_KEY,
        FieldSpec {
            key: SYM_IU4_KEY,
            kind: FieldKind::Bool,
        },
    ));
    table
}

/// The edit spec of `key` (canonical or legacy spelling); `None` for a
/// deprecated key, an unknown key, and any `developer.*` key but the typed
/// boolean one (see [`field_kind`]).
pub fn field_spec(key: &str) -> Option<&'static FieldSpec> {
    let canonical = match field(key) {
        Some(schema) => schema.key,
        None if key == SYM_IU4_KEY => SYM_IU4_KEY,
        None => return None,
    };
    spec_table()
        .iter()
        .find(|(candidate, _)| *candidate == canonical)
        .map(|(_, spec)| spec)
}

/// [`field_spec`]'s kind, extended to every `developer.*` key (free text:
/// developer keys keep their legacy free-form spelling).
pub fn field_kind(key: &str) -> Option<FieldKind> {
    field_spec(key).map(|spec| spec.kind).or_else(|| {
        is_developer_key(key).then_some(FieldKind::FreeStr {
            require_existing_file: false,
        })
    })
}

/// The accepted numeric range of `key`, read from the schema's `ValueRule` (the
/// sole validator), as display text: `"1..=1048576"`, `"0..=2"`. `None` for keys
/// whose rule has no numeric bounds.
pub fn schema_range(key: &str) -> Option<String> {
    use hipfire_config::ValueRule;
    match field(key)?.rule {
        ValueRule::Integer { min, max }
        | ValueRule::NullableInteger { min, max }
        | ValueRule::AutoInteger { min, max }
        | ValueRule::NullableIntegerOrAuto { min, max } => Some(format!("{min}..={max}")),
        ValueRule::Float {
            min,
            max,
            min_inclusive,
        } => Some(if min_inclusive {
            format!("{min}..={max}")
        } else {
            format!(">{min}..={max}")
        }),
        ValueRule::NullableFloat { min, max } => Some(format!("{min}..={max}")),
        _ => None,
    }
}

pub fn write_value(path: &Path, key: &str, raw: &str) -> Result<Value, WriteError> {
    if field(key).is_none() && is_developer_key(key) {
        return write_developer_value(path, key, raw);
    }
    let schema = field(key).ok_or_else(|| WriteError::Invalid {
        key: key.to_owned(),
        value: raw.to_owned(),
    })?;
    let invalid = || WriteError::Invalid {
        key: key.to_owned(),
        value: raw.to_owned(),
    };
    let config_value = match schema.parse_cli(raw) {
        Ok(value) => value,
        // An empty buffer on a field that cannot hold the empty string is the
        // *unset* spelling, not a typo: the editor opens empty exactly when the
        // stored value is null (`App::current_setting_value` reads "" for an
        // unset key), so Enter on a row the user never typed into must clear the
        // key rather than be rejected. A `FreeStr` field accepts "" directly and
        // never reaches this arm.
        Err(_) if raw.trim().is_empty() => schema.parse_cli("null").map_err(|_| invalid())?,
        Err(_) => return Err(invalid()),
    };
    let json = to_json(&config_value);
    write_config_value(path, schema.key, config_value)?;
    Ok(json)
}

/// `developer.*` keys are not in the schema. The typed-boolean key takes a
/// boolean spelling (`true`/`false`/`1`/`0`) and is written as a TOML
/// boolean; every other developer key keeps its free-form string.
fn write_developer_value(path: &Path, key: &str, raw: &str) -> Result<Value, WriteError> {
    let value = if is_typed_bool_developer_key(key) {
        let parsed = parse_typed_bool(raw.trim()).ok_or_else(|| WriteError::Invalid {
            key: key.to_owned(),
            value: raw.to_owned(),
        })?;
        ConfigValue::Bool(parsed)
    } else {
        ConfigValue::String(raw.to_owned())
    };
    let json = to_json(&value);
    write_config_value(path, key, value)?;
    Ok(json)
}

/// Apply a virtual row's choice (Settings "Prompt processing"): set or remove
/// each key it names, in one write, so the file never holds half a choice.
#[allow(dead_code)] // consumed by the Settings pane rewrite
pub fn write_choice(path: &Path, choice: &Choice) -> Result<(), WriteError> {
    let paths = paths_for(path);
    let mut loaded = load_global(&paths).map_err(config_error)?;
    for (key, value) in &choice.writes {
        match value {
            Some(value) => loaded.layer.set(key, value.clone()).map_err(config_error)?,
            None => {
                loaded.layer.remove(key).map_err(config_error)?;
            }
        }
    }
    write_global_toml(&paths, &loaded.layer).map_err(config_error)
}

pub fn write_raw_value(path: &Path, key: &str, value: Value) -> Result<(), WriteError> {
    let value = from_json(&value).ok_or_else(|| WriteError::Invalid {
        key: key.to_owned(),
        value: value.to_string(),
    })?;
    write_config_value(path, key, value)
}

fn write_config_value(path: &Path, key: &str, value: ConfigValue) -> Result<(), WriteError> {
    let paths = paths_for(path);
    let mut loaded = load_global(&paths).map_err(config_error)?;
    loaded.layer.set(key, value).map_err(config_error)?;
    write_global_toml(&paths, &loaded.layer).map_err(config_error)
}

pub fn delete_key(path: &Path, key: &str) -> Result<bool, WriteError> {
    let paths = paths_for(path);
    let mut loaded = load_global(&paths).map_err(config_error)?;
    let removed = loaded.layer.remove(key).map_err(config_error)?.is_some();
    if removed {
        write_global_toml(&paths, &loaded.layer).map_err(config_error)?;
    }
    Ok(removed)
}

pub fn reset_all(path: &Path) -> Result<(), WriteError> {
    write_global_toml(&paths_for(path), &ConfigLayer::default()).map_err(config_error)
}

pub fn cycle_enum(key: &str, current: &str, forward: bool) -> Option<String> {
    let spec = field_spec(key)?;
    if let FieldKind::Enum(allow) = spec.kind {
        let index = allow
            .iter()
            .position(|value| *value == current)
            .unwrap_or(0);
        let next = if forward {
            (index + 1) % allow.len()
        } else {
            (index + allow.len() - 1) % allow.len()
        };
        Some(allow[next].to_owned())
    } else {
        None
    }
}

fn paths_for(path: &Path) -> ConfigPaths {
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    ConfigPaths::under(root)
}

fn to_json(value: &ConfigValue) -> Value {
    match value {
        ConfigValue::Bool(value) => Value::Bool(*value),
        ConfigValue::Integer(value) => Value::Number((*value).into()),
        ConfigValue::Float(value) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        ConfigValue::String(value) => Value::String(value.clone()),
        ConfigValue::Null => Value::Null,
    }
}

fn from_json(value: &Value) -> Option<ConfigValue> {
    match value {
        Value::Bool(value) => Some(ConfigValue::Bool(*value)),
        Value::Number(value) => value
            .as_i64()
            .map(ConfigValue::Integer)
            .or_else(|| value.as_f64().map(ConfigValue::Float)),
        Value::String(value) => Some(ConfigValue::String(value.clone())),
        Value::Null => Some(ConfigValue::Null),
        Value::Array(_) | Value::Object(_) => None,
    }
}

#[derive(Debug)]
pub enum WriteError {
    Invalid { key: String, value: String },
    Config(String),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid { key, value } => {
                write!(output, "rejected invalid value for {key}: {value:?}")
            }
            Self::Config(message) => output.write_str(message),
        }
    }
}

impl std::error::Error for WriteError {}

fn config_error(error: hipfire_config::ConfigError) -> WriteError {
    WriteError::Config(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_config() -> (std::path::PathBuf, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "hipfire-tui-writer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("config.toml");
        (root, path)
    }

    #[test]
    fn writes_sparse_toml_through_shared_schema() {
        let (root, path) = temp_config();
        write_value(&path, "kv_cache", "q8").unwrap();
        write_value(&path, "max_tokens", "8192").unwrap();
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert_eq!(
            loaded.layer.get("memory.kv_cache"),
            Some(&ConfigValue::String("q8".into()))
        );
        assert_eq!(
            loaded.layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192))
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn shared_validation_rejects_invalid_values() {
        let (root, path) = temp_config();
        assert!(write_value(&path, "top_p", "0").is_err());
        assert!(write_value(&path, "port", "70000").is_err());
        assert!(write_value(&path, "kv_cache", "magic4").is_err());
        assert!(!path.exists());
        let _ = fs::remove_dir_all(root);
    }

    /// An empty buffer is the *unset* spelling on a field that cannot hold the
    /// empty string — the editor opens empty exactly when the stored value is
    /// null, so Enter on a row the user never typed into must clear it (a no-op
    /// when it was already unset) instead of flooding an error. A free-string
    /// field keeps meaning the empty string.
    #[test]
    fn empty_input_clears_a_nullable_field_but_stays_empty_for_a_string() {
        let (root, path) = temp_config();
        assert_eq!(
            write_value(&path, "gpu_layer_budget", "").unwrap(),
            Value::Null,
            "empty on a nullable integer must clear, not be rejected"
        );
        assert_eq!(
            write_value(&path, "gpu_layer_budget", "   ").unwrap(),
            Value::Null,
            "whitespace-only input is the same unset spelling"
        );
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert!(
            loaded.layer.get("gpu_layer_budget").is_none(),
            "cleared key is off disk"
        );

        // A free string keeps the empty string as a value.
        assert_eq!(
            write_value(&path, "prefill_drafter", "").unwrap(),
            Value::String(String::new())
        );
        // And a genuinely bad value is still rejected.
        assert!(write_value(&path, "gpu_layer_budget", "3x").is_err());
        // The empty-means-clear retry must not hand a non-nullable enum a clear
        // spelling it does not have in the schema (`pcie|cpu`).
        assert!(
            write_value(&path, "offload_exec", "").is_err(),
            "empty input must not make a non-nullable enum clearable"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// The option list a user can cycle through must be what the schema accepts:
    /// a value the schema accepts but the TUI omits is unselectable, and one the
    /// TUI offers but the schema rejects is a dead end.
    #[test]
    fn offload_exec_options_are_the_schema_values() {
        let (root, path) = temp_config();
        let spec = field_spec("offload_exec").expect("offload_exec has a spec");
        let FieldKind::Enum(options) = spec.kind else {
            panic!("offload_exec must be an enum row, got {:?}", spec.kind);
        };
        assert_eq!(options, hipfire_config::OFFLOAD_EXECS);
        for value in options {
            write_value(&path, "offload_exec", value)
                .unwrap_or_else(|e| panic!("{value} must be writable: {e:?}"));
        }
        assert!(write_value(&path, "offload_exec", "gpu").is_err());
        let _ = fs::remove_dir_all(root);
    }

    /// `ple_format` is editable and lands under its canonical schema key.
    #[test]
    fn ple_format_round_trips_under_the_schema_key() {
        let (root, path) = temp_config();
        let spec = field_spec("ple_format").expect("ple_format has a spec");
        assert_eq!(spec.kind, FieldKind::Enum(hipfire_config::PLE_FORMATS));
        assert_eq!(
            write_value(&path, "ple_format", "q4").unwrap(),
            Value::String("q4".into())
        );
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert_eq!(
            loaded.layer.get("memory.ple_format"),
            Some(&ConfigValue::String("q4".into()))
        );
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("ple_format"), "{text}");
        assert!(write_value(&path, "ple_format", "q5").is_err());
        let _ = fs::remove_dir_all(root);
    }

    /// The `speculation` row offers exactly the schema's mode list.
    #[test]
    fn speculation_options_are_the_schema_values() {
        let (root, path) = temp_config();
        let spec = field_spec("speculation").expect("speculation has a spec");
        let FieldKind::Enum(options) = spec.kind else {
            panic!("speculation must be an enum row");
        };
        assert_eq!(options, hipfire_config::SPECULATION_MODES);
        for value in options {
            write_value(&path, "speculation", value)
                .unwrap_or_else(|e| panic!("{value} must be writable: {e:?}"));
        }
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert_eq!(
            loaded.layer.get("speculation.mode"),
            Some(&ConfigValue::String("dspark".into()))
        );
        let _ = fs::remove_dir_all(root);
    }

    /// Deprecated CASK/PFlash keys are not editable rows.
    #[test]
    fn deprecated_keys_have_no_field_spec() {
        for spec in EDITABLE_FIELDS {
            assert!(
                !hipfire_config::is_deprecated_key(spec.key),
                "{} is deprecated but editable",
                spec.key
            );
        }
        for key in [
            "cask",
            "prefill_compression",
            "prefill_drafter",
            "prefill_threshold",
        ] {
            assert!(field_spec(key).is_none(), "{key}");
        }
    }

    /// Numeric ranges come from the schema, not a local copy: the old table said
    /// 131072/524288 where the schema allows 393216/1048576.
    #[test]
    fn numeric_rows_report_the_schema_range() {
        for spec in EDITABLE_FIELDS {
            if matches!(spec.kind, FieldKind::Int | FieldKind::Float) {
                assert!(
                    schema_range(spec.key).is_some(),
                    "numeric row {} has no schema range",
                    spec.key
                );
            }
        }
        let rule = field("max_seq").unwrap().rule;
        let hipfire_config::ValueRule::Integer { max, .. } = rule else {
            panic!("max_seq rule is not an integer: {rule:?}");
        };
        assert!(schema_range("max_seq")
            .unwrap()
            .ends_with(&format!("..={max}")));
        let (root, path) = temp_config();
        assert!(write_value(&path, "max_seq", &(max + 1).to_string()).is_err());
        assert!(write_value(&path, "max_seq", &max.to_string()).is_ok());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn reset_removes_only_the_selected_override() {
        let (root, path) = temp_config();
        write_value(&path, "kv_cache", "q8").unwrap();
        write_value(&path, "thinking", "off").unwrap();
        assert!(delete_key(&path, "kv_cache").unwrap());
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert!(loaded.layer.get("kv_cache").is_none());
        assert_eq!(
            loaded.layer.get("thinking"),
            Some(&ConfigValue::String("off".into()))
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cycling_order_remains_curated_for_the_tui() {
        assert_eq!(cycle_enum("kv_cache", "auto", true).as_deref(), Some("q8"));
        assert_eq!(
            cycle_enum("kv_adaptive", "advanced:k=fwht4,v=lloyd4", true).as_deref(),
            Some("advanced:k=fwht4,v=lloyd3")
        );
        assert!(cycle_enum("temperature", "0.3", true).is_none());
    }

    /// KV options are the schema's list minus the deprecated aliases, so a mode
    /// the schema gains cannot be unselectable and fp8/bf16/f16/f32 are offered.
    #[test]
    fn kv_options_are_the_schema_list_minus_deprecated_aliases() {
        let spec = field_spec("kv_cache").expect("kv_cache has a spec");
        let FieldKind::Enum(options) = spec.kind else {
            panic!("kv_cache must be an enum row");
        };
        let ValueRule::Enum(schema_list) = field("memory.kv_cache").unwrap().rule else {
            panic!("memory.kv_cache is not an enum rule");
        };
        let live: std::collections::BTreeSet<&str> = schema_list
            .iter()
            .copied()
            .filter(|mode| {
                !(mode.starts_with("asym")
                    || mode.starts_with("turbo")
                    || mode.starts_with("legacy-"))
            })
            .collect();
        let offered: std::collections::BTreeSet<&str> = options.iter().copied().collect();
        assert_eq!(offered, live);
        for mode in ["fp8", "bf16", "f16", "f32"] {
            assert!(options.contains(&mode), "{mode}");
        }
        let (root, path) = temp_config();
        for mode in options {
            write_value(&path, "memory.kv_cache", mode)
                .unwrap_or_else(|e| panic!("{mode} must be writable: {e:?}"));
        }
        let _ = fs::remove_dir_all(root);
    }

    /// Every non-deprecated schema field is editable and its kind follows its
    /// rule; deprecated and unknown keys stay read-only.
    #[test]
    fn every_live_schema_field_has_a_kind_from_its_rule() {
        for schema in hipfire_config::fields() {
            let spec = field_spec(schema.key);
            if is_deprecated_key(schema.key) {
                assert!(spec.is_none(), "{} is deprecated", schema.key);
                continue;
            }
            let spec = spec.unwrap_or_else(|| panic!("{} has no spec", schema.key));
            assert!(
                spec.key == schema.key || EDITABLE_FIELDS.iter().any(|s| s.key == spec.key),
                "{}",
                schema.key
            );
            if matches!(schema.rule, ValueRule::Bool) {
                assert_eq!(spec.kind, FieldKind::Bool, "{}", schema.key);
            }
        }
        assert_eq!(
            field_spec("memory.kv_backend").map(|s| s.kind),
            Some(FieldKind::Enum(&["legacy", "vmm"]))
        );
        assert!(field_spec("developer.some_future_flag").is_none());
        assert!(matches!(
            field_kind("developer.some_future_flag"),
            Some(FieldKind::FreeStr { .. })
        ));
        assert!(field_spec("no.such.key").is_none());
    }

    /// The typed-boolean developer key is written as a TOML boolean, never the
    /// string spelling.
    #[test]
    fn sym_iu4_is_written_as_a_toml_boolean() {
        let (root, path) = temp_config();
        assert_eq!(
            field_spec(SYM_IU4_KEY).map(|s| s.kind),
            Some(FieldKind::Bool)
        );
        assert_eq!(
            write_value(&path, SYM_IU4_KEY, "false").unwrap(),
            Value::Bool(false)
        );
        let text = fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("qwen4_moe_sym_iu4 = false"),
            "expected a TOML boolean in:\n{text}"
        );
        assert!(
            !text.contains("\"false\"") && !text.contains("\"0\""),
            "{text}"
        );
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert_eq!(
            loaded.layer.get(SYM_IU4_KEY),
            Some(&ConfigValue::Bool(false))
        );
        // `0` is the same boolean; a non-boolean word is rejected.
        assert_eq!(
            write_value(&path, SYM_IU4_KEY, "1").unwrap(),
            Value::Bool(true)
        );
        assert!(write_value(&path, SYM_IU4_KEY, "maybe").is_err());
        assert!(delete_key(&path, SYM_IU4_KEY).unwrap());
        let _ = fs::remove_dir_all(root);
    }

    /// A virtual row's choice sets or removes each of its keys in one write.
    #[test]
    fn write_choice_sets_then_removes_the_iu4_key_on_gfx1201() {
        use hipfire_config::effective::{quality_choices_for, A8Prefill, ArchCtx};
        let ctx = ArchCtx {
            gpu_arch: Some("gfx1201".into()),
            gpu_count: 1,
            model_arch_id: None,
            registry_tag: Some("qwen3.8:27b".into()),
            a8_prefill: A8Prefill::NotLanded,
        };
        let choices = quality_choices_for(&ctx, A8Prefill::NotLanded);
        let (root, path) = temp_config();
        write_value(&path, "max_tokens", "8192").unwrap();
        write_choice(&path, &choices[1]).unwrap();
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert_eq!(
            loaded.layer.get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("iu4_prefill = false"), "{text}");
        write_choice(&path, &choices[0]).unwrap();
        let loaded = load_global(&ConfigPaths::under(&root)).unwrap();
        assert!(loaded.layer.get("kernel.iu4_prefill").is_none());
        assert_eq!(
            loaded.layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(8192)),
            "unrelated keys survive a choice write"
        );
        let _ = fs::remove_dir_all(root);
    }
}
