// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Settings intent groups and the shared search predicate.
//!
//! Every config key belongs to exactly one [`Intent`]: a curated promotion
//! first, then the category default. The TUI Settings tab and
//! `hipfire config list` share this table so they cannot disagree about where a
//! key lives. [`field_matches`] / [`key_matches`] are the one search rule both
//! use.

use serde::Serialize;

use crate::{developer_env_for_key, field, ConfigCategory, ConfigField};

/// A Settings group, in display order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Intent {
    Quality,
    Speed,
    MemoryContext,
    Speculation,
    HardwareDevices,
    Advanced,
}

impl Intent {
    /// Display order.
    pub const ALL: [Intent; 6] = [
        Intent::Quality,
        Intent::Speed,
        Intent::MemoryContext,
        Intent::Speculation,
        Intent::HardwareDevices,
        Intent::Advanced,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Intent::Quality => "Quality",
            Intent::Speed => "Speed",
            Intent::MemoryContext => "Memory & context",
            Intent::Speculation => "Speculation",
            Intent::HardwareDevices => "Hardware & devices",
            Intent::Advanced => "Advanced",
        }
    }

    /// The group's one-line "Means" copy.
    pub fn header(self) -> &'static str {
        match self {
            Intent::Quality => "Changes the answers; better is usually slower.",
            Intent::Speed => "Faster or slower; doesn't trade away quality.",
            Intent::MemoryContext => "How much fits: longer chats, bigger models.",
            Intent::Speculation => "Drafting ahead to type faster; designed not to change answers.",
            Intent::HardwareDevices => "Which GPUs, and how they work together.",
            Intent::Advanced => "Everything else, by canonical key.",
        }
    }

    /// Ordered canonical keys of this group's curated rows. The Quality list
    /// names `kernel.gfx11_a8_prefill` unconditionally; it is only a real row
    /// once PR #902 puts the key in the schema, so rendering code uses
    /// [`Intent::curated_present`] and this slice stays `'static`.
    pub fn curated(self) -> &'static [&'static str] {
        match self {
            Intent::Quality => &[
                "kernel.iu4_prefill",
                "kernel.gfx11_a8_prefill",
                "developer.qwen4_moe_sym_iu4",
                "memory.ple_format",
                "reasoning.mode",
                "reasoning.effort",
                "reasoning.budget",
            ],
            Intent::Speed => &[
                "experimental.graph.forward",
                "fusions.npu_trunk",
                "prefill.chunk_rows",
                "serve.multi_slot",
                "attention.flash",
            ],
            Intent::MemoryContext => &[
                "memory.max_seq",
                "memory.kv_cache",
                "memory.kv_adaptive",
                "memory.gpu_layer_budget",
                "memory.moe_expert_budget",
                "memory.offload_exec",
                "memory.offload_passback_share",
                "memory.oom_guard",
                "memory.prompt_cache_capacity",
                "vision.mode",
            ],
            Intent::Speculation => &[
                "speculation.mode",
                "speculation.mtp",
                "speculation.mtp_k",
                "speculation.dflash",
                "speculation.ngram",
            ],
            Intent::HardwareDevices => &[
                "hardware.devices",
                "hardware.allow_mixed_arch",
                "hardware.tp_use_rccl",
            ],
            Intent::Advanced => &[],
        }
    }

    /// [`Intent::curated`] minus keys this build does not have: a curated key
    /// stays if the schema knows it or it is a `developer.*` key.
    pub fn curated_present(self) -> Vec<&'static str> {
        self.curated()
            .iter()
            .copied()
            .filter(|key| field(key).is_some() || crate::is_developer_key(key))
            .collect()
    }

    fn category_default(category: ConfigCategory) -> Intent {
        match category {
            ConfigCategory::Reasoning => Intent::Quality,
            ConfigCategory::Memory => Intent::MemoryContext,
            ConfigCategory::Speculation => Intent::Speculation,
            ConfigCategory::Hardware => Intent::HardwareDevices,
            ConfigCategory::Generation
            | ConfigCategory::Kernel
            | ConfigCategory::Attention
            | ConfigCategory::Vision
            | ConfigCategory::Replay
            | ConfigCategory::Fusions
            | ConfigCategory::Prompt
            | ConfigCategory::Serve
            | ConfigCategory::Diagnostic
            | ConfigCategory::Experimental => Intent::Advanced,
        }
    }
}

/// The group `key` (canonical or legacy spelling) lives in: curated rows
/// first, then the category default. `developer.*` and unknown keys are
/// Advanced.
pub fn intent_of(key: &str) -> Intent {
    let canonical = field(key).map_or(key, |schema| schema.key);
    for intent in Intent::ALL {
        if intent.curated().contains(&canonical) {
            return intent;
        }
    }
    field(key).map_or(Intent::Advanced, |schema| {
        Intent::category_default(schema.category)
    })
}

/// Case-insensitive search over a schema field: key, legacy key, help text,
/// env var, category, scope, rule, and the registry/experimental words
/// (`allowed`/`not allowed`, `experimental`/`stable`). An empty query matches
/// everything. A bare `allowed` does not match the `not allowed` text.
pub fn field_matches(field: &ConfigField, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let query = query.to_ascii_lowercase();
    let env = field.env_compat.unwrap_or_default();
    let registry = if field.registry_allowed {
        "allowed"
    } else {
        "not allowed"
    };
    let experimental = if field.experimental {
        "experimental"
    } else {
        "stable"
    };
    [
        field.key.to_owned(),
        field.legacy_key.to_owned(),
        field.help.to_owned(),
        env.to_owned(),
        format!("{:?}", field.category),
        format!("{:?}", field.scope),
        format!("{:?}", field.rule),
        registry.to_owned(),
        experimental.to_owned(),
    ]
    .iter()
    .any(|candidate| {
        let candidate = candidate.to_ascii_lowercase();
        if !candidate.contains(&query) {
            return false;
        }
        // "not allowed" must not match a bare "allowed" query.
        if query == "allowed" && candidate == "not allowed" {
            return false;
        }
        true
    })
}

/// Case-insensitive search over a `developer.*` key: the key itself and its
/// `HIPFIRE_*` environment name. An empty query matches everything.
pub fn key_matches(key: &str, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let query = query.to_ascii_lowercase();
    key.to_ascii_lowercase().contains(&query)
        || developer_env_for_key(key).is_some_and(|env| env.to_ascii_lowercase().contains(&query))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields;

    #[test]
    fn all_is_display_order_with_titles_and_headers() {
        assert_eq!(
            Intent::ALL.map(Intent::title),
            [
                "Quality",
                "Speed",
                "Memory & context",
                "Speculation",
                "Hardware & devices",
                "Advanced"
            ]
        );
        assert_eq!(
            Intent::Quality.header(),
            "Changes the answers; better is usually slower."
        );
        assert_eq!(
            Intent::Speed.header(),
            "Faster or slower; doesn't trade away quality."
        );
        assert_eq!(
            Intent::MemoryContext.header(),
            "How much fits: longer chats, bigger models."
        );
        assert_eq!(
            Intent::Speculation.header(),
            "Drafting ahead to type faster; designed not to change answers."
        );
        assert_eq!(
            Intent::HardwareDevices.header(),
            "Which GPUs, and how they work together."
        );
        assert_eq!(
            Intent::Advanced.header(),
            "Everything else, by canonical key."
        );
    }

    #[test]
    fn curated_quality_order_matches_the_contract() {
        assert_eq!(
            Intent::Quality.curated(),
            [
                "kernel.iu4_prefill",
                "kernel.gfx11_a8_prefill",
                "developer.qwen4_moe_sym_iu4",
                "memory.ple_format",
                "reasoning.mode",
                "reasoning.effort",
                "reasoning.budget",
            ]
        );
        assert!(Intent::Advanced.curated().is_empty());
    }

    #[test]
    fn curated_keys_exist_and_belong_to_their_group() {
        for intent in Intent::ALL {
            for key in intent.curated() {
                assert_eq!(intent_of(key), intent, "{key}");
                let in_schema = field(key).is_some();
                let known_gap = *key == "kernel.gfx11_a8_prefill";
                assert!(
                    in_schema || known_gap || crate::is_developer_key(key),
                    "curated key {key} is not in the schema"
                );
            }
        }
    }

    #[test]
    fn curated_present_drops_only_keys_missing_from_the_schema() {
        let present = Intent::Quality.curated_present();
        assert_eq!(
            present.contains(&"kernel.gfx11_a8_prefill"),
            field("kernel.gfx11_a8_prefill").is_some()
        );
        assert!(present.contains(&"developer.qwen4_moe_sym_iu4"));
        for intent in Intent::ALL {
            for key in intent.curated_present() {
                assert!(intent.curated().contains(&key));
            }
        }
    }

    #[test]
    fn no_key_is_curated_twice() {
        let mut seen = std::collections::BTreeSet::new();
        for intent in Intent::ALL {
            for key in intent.curated() {
                assert!(seen.insert(*key), "{key} curated in two groups");
            }
        }
    }

    #[test]
    fn curated_wins_over_the_category_default() {
        // Kernel / Memory defaults would put these elsewhere.
        assert_eq!(intent_of("kernel.iu4_prefill"), Intent::Quality);
        assert_eq!(intent_of("memory.ple_format"), Intent::Quality);
        assert_eq!(intent_of("developer.qwen4_moe_sym_iu4"), Intent::Quality);
        assert_eq!(intent_of("experimental.graph.forward"), Intent::Speed);
        assert_eq!(intent_of("serve.multi_slot"), Intent::Speed);
        assert_eq!(intent_of("vision.mode"), Intent::MemoryContext);
        assert_eq!(intent_of("attention.flash"), Intent::Speed);
    }

    #[test]
    fn legacy_spelling_resolves_to_the_canonical_group() {
        assert_eq!(intent_of("kv_cache"), Intent::MemoryContext);
        assert_eq!(intent_of("iu4_prefill"), Intent::Quality);
    }

    #[test]
    fn category_defaults_and_developer_keys() {
        for schema in fields() {
            let want = match schema.category {
                ConfigCategory::Reasoning => Intent::Quality,
                ConfigCategory::Memory => Intent::MemoryContext,
                ConfigCategory::Speculation => Intent::Speculation,
                ConfigCategory::Hardware => Intent::HardwareDevices,
                _ => Intent::Advanced,
            };
            let curated = Intent::ALL
                .iter()
                .find(|i| i.curated().contains(&schema.key))
                .copied();
            assert_eq!(
                intent_of(schema.key),
                curated.unwrap_or(want),
                "{}",
                schema.key
            );
        }
        assert_eq!(
            intent_of("memory.session_cache_bytes"),
            Intent::MemoryContext
        );
        assert_eq!(intent_of("speculation.mtp_sampled"), Intent::Speculation);
        assert_eq!(intent_of("serve.port"), Intent::Advanced);
        assert_eq!(intent_of("developer.anything_else"), Intent::Advanced);
        assert_eq!(intent_of("not.a.key"), Intent::Advanced);
    }

    fn field_text_contains(field: &ConfigField, needle: &str) -> bool {
        let needle = needle.to_ascii_lowercase();
        [
            field.key.to_owned(),
            field.legacy_key.to_owned(),
            field.help.to_owned(),
            field.env_compat.unwrap_or_default().to_owned(),
            format!("{:?}", field.category),
            format!("{:?}", field.scope),
            format!("{:?}", field.rule),
        ]
        .iter()
        .any(|candidate| candidate.to_ascii_lowercase().contains(&needle))
    }

    #[test]
    fn field_matches_help_legacy_and_environment_metadata() {
        let help_field = fields()
            .iter()
            .find(|field| !field.help.is_empty())
            .unwrap();
        assert!(field_matches(
            help_field,
            help_field.help.split_whitespace().next().unwrap()
        ));
        assert!(field_matches(help_field, help_field.legacy_key));
        let env_field = fields()
            .iter()
            .find(|field| field.env_compat.is_some())
            .unwrap();
        assert!(field_matches(env_field, env_field.env_compat.unwrap()));
        assert!(field_matches(
            env_field,
            &env_field.env_compat.unwrap().to_lowercase()
        ));
        assert!(field_matches(env_field, ""));
        assert!(!field_matches(env_field, "zz-no-such-text-zz"));
    }

    #[test]
    fn field_matches_registry_and_experimental_metadata() {
        let allowed = fields()
            .iter()
            .find(|field| {
                field.registry_allowed
                    && !field_text_contains(field, "not allowed")
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "experimental")
            })
            .expect("registry-allowed field");
        let not_allowed = fields()
            .iter()
            .find(|field| {
                !field.registry_allowed
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "experimental")
            })
            .expect("registry-not-allowed field");
        assert!(field_matches(allowed, "allowed"));
        assert!(!field_matches(allowed, "not allowed"));
        assert!(field_matches(not_allowed, "not allowed"));
        assert!(!field_matches(not_allowed, "allowed"));

        let experimental = fields()
            .iter()
            .find(|field| {
                field.experimental
                    && !field_text_contains(field, "stable")
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "not allowed")
            })
            .expect("experimental field");
        let stable = fields()
            .iter()
            .find(|field| {
                !field.experimental
                    && !field_text_contains(field, "experimental")
                    && !field_text_contains(field, "allowed")
                    && !field_text_contains(field, "not allowed")
            })
            .expect("stable field");
        assert!(field_matches(experimental, "experimental"));
        assert!(!field_matches(experimental, "stable"));
        assert!(field_matches(stable, "stable"));
        assert!(!field_matches(stable, "experimental"));
    }

    #[test]
    fn key_matches_key_and_env_name() {
        let key = "developer.qwen4_moe_sym_iu4";
        assert!(key_matches(key, ""));
        assert!(key_matches(key, "SYM_IU4"));
        assert!(key_matches(key, "developer.qwen4"));
        assert!(key_matches(key, "HIPFIRE_QWEN4_MOE_SYM_IU4"));
        assert!(key_matches(key, "hipfire_qwen4"));
        assert!(!key_matches(key, "kv_cache"));
    }

    #[test]
    fn moe_sym_iu4_is_an_uncurated_advanced_developer_key_with_its_own_env_name() {
        // Default-OFF opt-in route: not a curated Quality row, never aliased to the
        // Flash-Next key.
        let key = "developer.moe_sym_iu4";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_MOE_SYM_IU4"));
        assert!(key_matches(key, "moe_sym_iu4"));
        assert!(!key_matches(
            "developer.qwen4_moe_sym_iu4",
            "HIPFIRE_MOE_SYM_IU4"
        ));
    }

    #[test]
    fn moe_router_fused_prefill_is_an_uncurated_advanced_developer_key() {
        let key = "developer.moe_router_fused_prefill";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_MOE_ROUTER_FUSED_PREFILL"));
        assert!(key_matches(key, "moe_router_fused_prefill"));
    }

    #[test]
    fn moe_router_topk_multirow_is_an_uncurated_advanced_developer_key_with_its_own_env_name() {
        let key = "developer.moe_router_topk_multirow";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_MOE_ROUTER_TOPK_MULTIROW"));
        assert!(key_matches(key, "moe_router_topk_multirow"));
        assert!(!key_matches(key, "HIPFIRE_MOE_ROUTER_FUSED_PREFILL"));
        assert!(!key_matches(
            "developer.moe_router_fused_prefill",
            "HIPFIRE_MOE_ROUTER_TOPK_MULTIROW"
        ));
    }

    #[test]
    fn moe_shared_prepared_iu4_is_an_uncurated_advanced_developer_key_with_its_own_env_name() {
        // Default-on capability route (`0`/`false` kills it): not a curated row,
        // never aliased to the other symmetric IU4 keys.
        let key = "developer.moe_shared_prepared_iu4";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_MOE_SHARED_PREPARED_IU4"));
        assert!(key_matches(key, "moe_shared_prepared_iu4"));
        assert!(!key_matches("developer.moe_sym_iu4", "SHARED_PREPARED"));
    }

    #[test]
    fn prefill_iu4_producer_fused_is_an_uncurated_advanced_developer_key() {
        // Qualified capability route (exact-gfx1201 default resolved by rdna-compute, not a
        // schema default).
        let key = "developer.prefill_iu4_producer_fused";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_PREFILL_IU4_PRODUCER_FUSED"));
        assert!(key_matches(key, "prefill_iu4_producer_fused"));
    }

    #[test]
    fn prefill_iu4_c8_pm_is_an_uncurated_advanced_developer_key() {
        // Raw-inventory-only capability route: opt-in, no schema default.
        let key = "developer.prefill_iu4_c8_pm";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_PREFILL_IU4_C8_PM"));
        assert!(key_matches(key, "prefill_iu4_c8_pm"));
    }

    #[test]
    fn moe_shared_down_sigmoid_epi_is_an_uncurated_advanced_developer_key() {
        // Default-off opt-in capability route (no qualified row, no schema default).
        let key = "developer.moe_shared_down_sigmoid_epi";
        assert_eq!(intent_of(key), Intent::Advanced);
        for intent in Intent::ALL {
            assert!(!intent.curated().contains(&key), "{intent:?}");
        }
        assert!(key_matches(key, "HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI"));
        assert!(key_matches(key, "moe_shared_down_sigmoid_epi"));
    }
}
