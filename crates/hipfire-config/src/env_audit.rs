// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Classification of the `HIPFIRE_*` variables set in a process environment
//! (`hipfire config explain`'s `environment` section).
//!
//! The classifier is a pure function over `(name, value)` pairs so tests never
//! touch the process environment. The known-name set is the schema's
//! `env_compat` names, the bootstrap and stable names `load_env_layer` skips,
//! and every name in the generated inventory of `docs/env-vars.md`.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::Serialize;

use crate::{developer_key_for_env, fields, is_deprecated_key, BOOTSTRAP_ENV};

const INVENTORY_DOC: &str = include_str!("../../../docs/env-vars.md");

/// Release a deprecation started in / is removed in.
const DEPRECATED_SINCE: &str = "0.4.0";
const REMOVED_IN: &str = "0.5.0";

/// Names `load_env_layer` handles outside the schema (`HIPFIRE_DEVICE` is the
/// singular spelling of `HIPFIRE_DEVICES`).
const STABLE_ENV: &[&str] = &["HIPFIRE_DEVICES", "HIPFIRE_DEVICE"];

/// Deprecated variables that are not the `env_compat` of a deprecated schema
/// key. `only_value` limits the deprecation to one value (`=0` is the legacy
/// contiguous DFlash opt-in; other window values are supported).
struct DeprecatedEnv {
    name: &'static str,
    only_value: Option<&'static str>,
    replacement: &'static str,
}

const DEPRECATED_ENV: &[DeprecatedEnv] = &[
    DeprecatedEnv {
        name: "HIPFIRE_DFLASH_WINDOW",
        only_value: Some("0"),
        replacement: "unset it (the draft-declared window applies)",
    },
    DeprecatedEnv {
        name: "HIPFIRE_DFLASH_CTX_CAP",
        only_value: None,
        replacement: "unset it (the draft-declared window applies)",
    },
];

const NO_REPLACEMENT: &str = "no replacement (feature removed)";

/// How one set variable relates to hipfire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum EnvStatus {
    /// Consumed by the env layer as this config key (schema `env_compat` or
    /// a `developer.*` mapping).
    Applies { key: String },
    /// Read before config exists (paths, URLs, device selection).
    Bootstrap,
    /// Documented in the env-vars inventory but not a config key.
    Runtime,
    Deprecated {
        since: &'static str,
        removed: &'static str,
        replacement: String,
    },
    /// Not a hipfire variable; `suggestion` is the nearest known name.
    Unknown { suggestion: Option<String> },
}

impl EnvStatus {
    pub fn label(&self) -> String {
        match self {
            Self::Applies { key } => format!("applies {key}"),
            Self::Bootstrap => "bootstrap".into(),
            Self::Runtime => "runtime".into(),
            Self::Deprecated {
                since,
                removed,
                replacement,
            } => format!("DEPRECATED since {since}, removed in {removed}: {replacement}"),
            Self::Unknown {
                suggestion: Some(name),
            } => format!("UNKNOWN: not a hipfire variable (typo? did you mean {name})"),
            Self::Unknown { suggestion: None } => "UNKNOWN: not a hipfire variable (typo?)".into(),
        }
    }
}

/// One set `HIPFIRE_*` variable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EnvEntry {
    pub name: String,
    /// The value, or `(set)` for secret-looking names.
    pub value: String,
    pub status: EnvStatus,
}

/// True for names whose value must never be printed.
pub fn is_secret_name(name: &str) -> bool {
    ["KEY", "TOKEN", "SECRET", "PASS"]
        .iter()
        .any(|pattern| name.contains(pattern))
}

/// Inventory rows of `docs/env-vars.md`: name → status word of the generated
/// table (`stable`, `experimental`, `developer`, `harness`, `deprecated`), when
/// the row has one.
pub fn inventory() -> &'static BTreeMap<&'static str, Option<&'static str>> {
    static INVENTORY: LazyLock<BTreeMap<&'static str, Option<&'static str>>> =
        LazyLock::new(|| parse_inventory(INVENTORY_DOC));
    &INVENTORY
}

fn parse_inventory(doc: &'static str) -> BTreeMap<&'static str, Option<&'static str>> {
    const STATUSES: [&str; 5] = [
        "stable",
        "experimental",
        "developer",
        "harness",
        "deprecated",
    ];
    let mut names: BTreeMap<&'static str, Option<&'static str>> = BTreeMap::new();
    for line in doc.lines() {
        let Some(rest) = line.strip_prefix("| `") else {
            continue;
        };
        let Some((name, rest)) = rest.split_once("` |") else {
            continue;
        };
        let mut chars = name.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_uppercase())
            && chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        if !valid {
            continue;
        }
        let mut columns: Vec<&str> = rest.split('|').map(str::trim).collect();
        if columns.last() == Some(&"") {
            columns.pop();
        }
        let status = (columns.len() == 2)
            .then(|| columns[1].split(';').next().unwrap_or("").trim())
            .and_then(|word| STATUSES.iter().find(|status| **status == word).copied());
        let slot = names.entry(name).or_insert(None);
        if status.is_some() {
            *slot = status;
        }
    }
    names
}

fn schema_env_key(name: &str) -> Option<&'static crate::ConfigField> {
    fields().iter().find(|field| field.env_compat == Some(name))
}

fn is_bootstrap(name: &str) -> bool {
    BOOTSTRAP_ENV.contains(&name) || STABLE_ENV.contains(&name)
}

fn deprecated(name: &str, value: &str) -> Option<EnvStatus> {
    let replacement = if let Some(entry) = DEPRECATED_ENV.iter().find(|entry| entry.name == name) {
        if entry.only_value.is_some_and(|only| only != value.trim()) {
            return None;
        }
        entry.replacement
    } else if schema_env_key(name).is_some_and(|field| is_deprecated_key(field.key))
        || inventory().get(name) == Some(&Some("deprecated"))
    {
        NO_REPLACEMENT
    } else {
        return None;
    };
    Some(EnvStatus::Deprecated {
        since: DEPRECATED_SINCE,
        removed: REMOVED_IN,
        replacement: replacement.to_owned(),
    })
}

/// Every `HIPFIRE_*` name hipfire documents or consumes.
fn known_names() -> impl Iterator<Item = &'static str> {
    fields()
        .iter()
        .filter_map(|field| field.env_compat)
        .chain(BOOTSTRAP_ENV.iter().copied())
        .chain(STABLE_ENV.iter().copied())
        .chain(inventory().keys().copied())
        .filter(|name| name.starts_with("HIPFIRE_"))
}

fn is_known(name: &str) -> bool {
    known_names().any(|known| known == name)
}

/// Levenshtein distance, early-exit above `limit`.
fn edit_distance(a: &str, b: &str, limit: usize) -> Option<usize> {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len().abs_diff(b.len()) > limit {
        return None;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, &cb) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(ca != cb);
            current.push(substitution.min(previous[j + 1] + 1).min(current[j] + 1));
        }
        if current.iter().all(|&d| d > limit) {
            return None;
        }
        previous = current;
    }
    previous.last().copied().filter(|&d| d <= limit)
}

/// The nearest known name within edit distance 3 (ties: alphabetical).
fn suggest(name: &str) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for known in known_names() {
        if let Some(distance) = edit_distance(name, known, 3) {
            if best.is_none_or(|(d, n)| (distance, known) < (d, n)) {
                best = Some((distance, known));
            }
        }
    }
    best.map(|(_, known)| known.to_owned())
}

/// Classify one variable.
pub fn classify(name: &str, value: &str) -> EnvStatus {
    if let Some(status) = deprecated(name, value) {
        return status;
    }
    if is_bootstrap(name) {
        return EnvStatus::Bootstrap;
    }
    if let Some(field) = schema_env_key(name) {
        return EnvStatus::Applies {
            key: field.key.to_owned(),
        };
    }
    if is_known(name) {
        if inventory().get(name) == Some(&Some("developer")) {
            if let Some(key) = developer_key_for_env(name) {
                return EnvStatus::Applies { key };
            }
        }
        return EnvStatus::Runtime;
    }
    EnvStatus::Unknown {
        suggestion: suggest(name),
    }
}

/// Audit every `HIPFIRE_*` variable in `vars`, sorted by name.
pub fn audit_env<I>(vars: I) -> Vec<EnvEntry>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut entries: Vec<EnvEntry> = vars
        .into_iter()
        .filter(|(name, _)| name.starts_with("HIPFIRE_"))
        .map(|(name, value)| {
            let status = classify(&name, &value);
            let value = if is_secret_name(&name) {
                "(set)".to_owned()
            } else {
                value
            };
            EnvEntry {
                name,
                value,
                status,
            }
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

/// Audit the current process environment (non-UTF-8 entries are skipped).
pub fn audit_process_env() -> Vec<EnvEntry> {
    audit_env(std::env::vars())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(name: &str, value: &str) -> EnvEntry {
        audit_env([(name.to_owned(), value.to_owned())])
            .pop()
            .expect("HIPFIRE_ name is audited")
    }

    /// `HIPFIRE_<suffix>`, assembled at runtime so `check-lifecycle.py` does
    /// not read fixture names (typos, secrets) as variables this crate reads.
    fn env_name(suffix: &str) -> String {
        format!("HIPFIRE_{suffix}")
    }

    #[test]
    fn inventory_parses_over_a_thousand_names() {
        let names = inventory();
        assert!(names.len() > 1000, "parsed {} names", names.len());
        assert!(names.contains_key("HIPFIRE_KV_MODE"));
        assert_eq!(names.get("HIPFIRE_CASK_SIDECAR"), Some(&Some("deprecated")));
    }

    #[test]
    fn schema_env_compat_applies_the_config_key() {
        assert_eq!(
            one("HIPFIRE_KV_MODE", "q8").status,
            EnvStatus::Applies {
                key: "memory.kv_cache".into()
            }
        );
    }

    #[test]
    fn bootstrap_and_stable_names() {
        assert_eq!(one("HIPFIRE_HOME", "/x").status, EnvStatus::Bootstrap);
        assert_eq!(one("HIPFIRE_DEVICES", "0").status, EnvStatus::Bootstrap);
        assert_eq!(one("HIPFIRE_DEVICE", "0").status, EnvStatus::Bootstrap);
    }

    #[test]
    fn inventory_developer_names_apply_developer_keys_and_others_are_runtime() {
        assert_eq!(
            one("HIPFIRE_CACHE_CKPT_MAX", "4").status,
            EnvStatus::Applies {
                key: "developer.cache_ckpt_max".into()
            }
        );
        assert_eq!(one(&env_name("B_REF"), "x").status, EnvStatus::Runtime);
    }

    #[test]
    fn deprecated_names_name_since_removal_and_replacement() {
        let status = one("HIPFIRE_DFLASH_CTX_CAP", "4096").status;
        assert_eq!(
            status.label(),
            "DEPRECATED since 0.4.0, removed in 0.5.0: unset it (the draft-declared window applies)"
        );
        // Only `=0` is the deprecated window value.
        assert_eq!(
            one("HIPFIRE_DFLASH_WINDOW", "0")
                .status
                .kind_is_deprecated(),
            true
        );
        assert_ne!(
            one("HIPFIRE_DFLASH_WINDOW", "4096")
                .status
                .kind_is_deprecated(),
            true
        );
        // env_compat of a deprecated schema key.
        let cask = fields()
            .iter()
            .find(|f| f.env_compat.is_some() && is_deprecated_key(f.key))
            .expect("a deprecated key has an env alias");
        assert!(one(cask.env_compat.unwrap(), "1")
            .status
            .kind_is_deprecated());
    }

    #[test]
    fn unknown_names_get_the_nearest_suggestion() {
        let entry = one(&env_name("KVMODE"), "q8");
        assert_eq!(
            entry.status,
            EnvStatus::Unknown {
                suggestion: Some("HIPFIRE_KV_MODE".into())
            }
        );
        assert_eq!(
            entry.status.label(),
            "UNKNOWN: not a hipfire variable (typo? did you mean HIPFIRE_KV_MODE)"
        );
        assert_eq!(
            one(&env_name("ZZZZZZZZZZZZZZZZZZZZ_NOPE"), "1").status,
            EnvStatus::Unknown { suggestion: None }
        );
    }

    #[test]
    fn secret_names_hide_their_value_and_non_hipfire_names_are_skipped() {
        assert!(is_secret_name(&env_name("API_KEY")));
        let entries = audit_env([
            (env_name("SERVE_TOKEN"), "hunter2".to_owned()),
            ("PATH".to_owned(), "/bin".to_owned()),
        ]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value, "(set)");
    }

    impl EnvStatus {
        fn kind_is_deprecated(&self) -> bool {
            matches!(self, Self::Deprecated { .. })
        }
    }
}
