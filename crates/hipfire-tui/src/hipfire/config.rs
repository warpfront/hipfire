// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::collections::BTreeMap;
use std::path::Path;

use hipfire_client::ServiceEndpoint;
use hipfire_config::{
    field, load_catalog, load_global, resolve, secret::BearerToken, ConfigFormat, ConfigPaths,
    ConfigValue, NamedLayer,
};

use super::HipfirePaths;

#[derive(Clone, Debug)]
pub struct ConfigState {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub values: BTreeMap<String, String>,
    pub per_model_count: usize,
    pub loaded_from_disk: bool,
    pub warning: Option<String>,
}

impl ConfigState {
    pub fn load(paths: &HipfirePaths) -> Self {
        Self::load_inner(paths, true)
    }

    /// [`ConfigState::load`] without the `HIPFIRE_SERVE_API_KEY_FILE`
    /// process-environment override: reads only `paths`.
    pub fn load_hermetic(paths: &HipfirePaths) -> Self {
        Self::load_inner(paths, false)
    }

    fn load_inner(paths: &HipfirePaths, read_env: bool) -> Self {
        let mut values = defaults();
        let mut loaded_from_disk = false;
        let mut warning = None;

        match load_global(&ConfigPaths::under(&paths.root)) {
            Ok(loaded) => {
                loaded_from_disk = loaded.format != ConfigFormat::Empty;
                for (canonical, value) in loaded.layer.values {
                    if let Some(schema) = field(&canonical) {
                        values.insert(schema.legacy_key.to_owned(), value_to_string(&value));
                    }
                }
                if !loaded.warnings.is_empty() {
                    warning = Some(loaded.warnings.join("; "));
                }
            }
            Err(error) => warning = Some(format!("config parse error: {error}")),
        }

        // `HIPFIRE_SERVE_API_KEY_FILE` outranks the config file, matching the
        // CLI's resolution order for this key.
        if let Some(schema) = field("serve.api_key_file").filter(|_| read_env) {
            if let Some(name) = schema.env_compat {
                if let Ok(raw) = std::env::var(name) {
                    match schema.parse_cli(&raw) {
                        Ok(value) => {
                            values.insert(schema.legacy_key.to_owned(), value_to_string(&value));
                        }
                        Err(error) => {
                            // Retain the invalid override so HTTP callers fail
                            // closed instead of using the old or absent key.
                            values.insert(schema.legacy_key.to_owned(), raw);
                            let message = format!("{name}: {error}");
                            warning = Some(match warning.take() {
                                Some(previous) => format!("{previous}; {message}"),
                                None => message,
                            });
                        }
                    }
                }
            }
        }

        let per_model_count = load_catalog(&ConfigPaths::under(&paths.root))
            .map(|loaded| loaded.catalog.models.len())
            .unwrap_or(0);

        let host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "0.0.0.0".into());
        let port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);
        let default_model = values
            .get("default_model")
            .cloned()
            .unwrap_or_else(|| "qwen3.5:9b".into());

        Self {
            host,
            port,
            default_model,
            values,
            per_model_count,
            loaded_from_disk,
            warning,
        }
    }

    /// The hardcoded default map, for cross-checking the knob explainer defaults
    /// (5e) against the real defaults so the two can't drift.
    #[cfg(test)]
    pub(crate) fn defaults_for_test() -> BTreeMap<String, String> {
        defaults()
    }

    pub fn probe_host(&self) -> String {
        match self.host.as_str() {
            "0.0.0.0" | "" => "127.0.0.1".into(),
            "::" => "::1".into(),
            other => other.to_string(),
        }
    }

    /// Authenticated endpoint for the local serve process. Loads the bearer
    /// from the resolved `serve.api_key_file` (empty = keyless). Errors name
    /// the key file path and the violated rule, never the token.
    pub fn service_endpoint(&self) -> Result<ServiceEndpoint, String> {
        let key_file = self.values.get("api_key_file").map_or("", String::as_str);
        if !key_file.is_empty()
            && (key_file.trim() != key_file || !Path::new(key_file).is_absolute())
        {
            return Err("serve.api_key_file must be an absolute path".to_owned());
        }
        let bearer = if key_file.is_empty() {
            None
        } else {
            Some(
                BearerToken::load(Path::new(key_file))
                    .map_err(|error| format!("serve.api_key_file: {error}"))?,
            )
        };
        ServiceEndpoint::local(&self.host, self.port, bearer).map_err(|error| error.to_string())
    }
}

/// Plain label for a layer-count budget (`memory.gpu_layer_budget`,
/// `memory.moe_expert_budget`). The editor seeds its buffer from `values`, not
/// from this label.
pub(crate) fn layers_label(raw: &str) -> String {
    match raw {
        // Unset keeps every layer resident on every model. Name the state rather
        // than leaving a blank value column.
        "" => "all on GPU".into(),
        // `-1` is the schema's reserved auto spelling, and the rule also accepts
        // a literal `auto`; both mean fit-to-card and neither is a layer count.
        "-1" | "auto" => "auto (fits to card)".into(),
        v => format!("{v} on GPU"),
    }
}

/// Plain label for `memory.offload_exec`.
pub(crate) fn offload_exec_label(raw: &str) -> String {
    match raw {
        "" | "pcie" => "over PCIe".into(),
        "cpu" => "on CPU".into(),
        "passback" => "passback (CPU + GPU)".into(),
        // Only reachable if the schema's list grows: show it rather than
        // mislabel it.
        other => other.to_string(),
    }
}

/// Plain label for `memory.offload_passback_share`.
pub(crate) fn passback_share_label(raw: &str) -> String {
    match raw {
        "" | "auto" => "auto (scheduled)".into(),
        other => other.to_string(),
    }
}

fn defaults() -> BTreeMap<String, String> {
    default_map().clone()
}

/// The schema defaults, resolved once: the explainer reads them every frame.
fn default_map() -> &'static BTreeMap<String, String> {
    static DEFAULTS: std::sync::OnceLock<BTreeMap<String, String>> = std::sync::OnceLock::new();
    DEFAULTS.get_or_init(|| {
        resolve(Vec::<NamedLayer>::new())
            .expect("shared built-in config schema validates")
            .legacy_values()
            .into_iter()
            .map(|(key, value)| (key, value_to_string(&value)))
            .collect()
    })
}

fn value_to_string(value: &ConfigValue) -> String {
    match value {
        ConfigValue::String(value) => value.clone(),
        ConfigValue::Integer(value) => value.to_string(),
        ConfigValue::Float(value) => value.to_string(),
        ConfigValue::Bool(value) => value.to_string(),
        ConfigValue::Null => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ConfigState seeded from an explicit value map (defaults overlaid).
    fn state_with(pairs: &[(&str, &str)]) -> ConfigState {
        let mut values = defaults();
        for (k, v) in pairs {
            values.insert((*k).to_string(), (*v).to_string());
        }
        ConfigState {
            host: "0.0.0.0".into(),
            port: 11435,
            default_model: "qwen3.5:9b".into(),
            values,
            per_model_count: 0,
            loaded_from_disk: true,
            warning: None,
        }
    }

    #[test]
    fn service_endpoint_keeps_keyless_local_behavior_and_rejects_bad_key_paths() {
        let endpoint = state_with(&[]).service_endpoint().unwrap();
        assert!(endpoint.bearer().is_none());
        assert_eq!(
            hipfire_client::service_url(&endpoint, "/health"),
            "http://127.0.0.1:11435/health"
        );
        assert!(state_with(&[("api_key_file", "relative-key")])
            .service_endpoint()
            .unwrap_err()
            .contains("absolute path"));
        let missing =
            std::env::temp_dir().join(format!("hipfire-tui-key-missing-{}", std::process::id()));
        assert!(state_with(&[("api_key_file", missing.to_str().unwrap())])
            .service_endpoint()
            .is_err());
    }

    /// The layer-count label is the one cell grammar of the GPU-layers rows:
    /// `-1` is the schema's *auto* spelling, not a count of layers, and a
    /// cleared key reads as unset rather than as the string the JSON layer
    /// produces for null.
    #[test]
    fn layer_label_renders_unset_count_and_auto() {
        // Unset never spills, including on a MoE model.
        assert_eq!(layers_label(""), "all on GPU");
        assert_eq!(layers_label("32"), "32 on GPU");
        assert_eq!(
            layers_label("-1"),
            "auto (fits to card)",
            "-1 means auto, not -1 layers on the GPU"
        );
        assert_eq!(
            layers_label("auto"),
            "auto (fits to card)",
            "the literal auto spelling (accepted by the value rule) is not a count"
        );
    }

    /// The exec label's two arms.
    #[test]
    fn offload_exec_label_renders_both_arms() {
        assert_eq!(offload_exec_label(""), "over PCIe");
        assert_eq!(offload_exec_label("pcie"), "over PCIe");
        assert_eq!(offload_exec_label("cpu"), "on CPU");
        assert_eq!(offload_exec_label("passback"), "passback (CPU + GPU)");
    }

    /// Unset and `auto` both mean "the engine schedules it", and a pinned
    /// number renders as itself.
    #[test]
    fn passback_share_label_renders_scheduled_and_pinned() {
        assert_eq!(passback_share_label(""), "auto (scheduled)");
        assert_eq!(passback_share_label("auto"), "auto (scheduled)");
        assert_eq!(passback_share_label("0"), "0");
        assert_eq!(passback_share_label("0.375"), "0.375");
    }
}
