// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Effective values: what a resolved setting means on this GPU for this model.
//!
//! [`resolve`](crate::resolve) answers "which layer set this key". This module
//! answers the next question the Settings screen and `hipfire config` ask:
//! "what will it actually do here". Built-in `auto` values and arch-gated
//! defaults become a concrete value plus a one-line reason, and a key that the
//! GPU or model never reads is flagged. Everything here is pure (no env, no
//! GPU, no I/O except [`install_gpu_arch`]) and mirrors the runtime rules; the
//! cross-check tests next to `hipfire-runtime/src/kv_mode.rs` and
//! `rdna-compute/src/feature_flags.rs` keep it from drifting.
//!
//! Arch rules only fire when nothing above the built-in layer set the key
//! (`ConfigSource::BuiltIn`, or an unset `developer.*` key). A value a registry
//! card, config file, model entry or environment variable wrote is passed
//! through unchanged with `by_arch_rule = false`.

use std::path::Path;

use serde::Serialize;

use crate::{
    developer_env_for_key, field, memory::PleFormatRequest, mtp_k_default,
    qwen4_row_exact_verify_rows_table, ConfigLayer, ConfigScope, ConfigSource, ConfigValue,
    ResolvedConfig,
};

const KV_KEY: &str = "memory.kv_cache";
const PLE_KEY: &str = "memory.ple_format";
const STATE_QUANT_KEY: &str = "memory.state_quant";
const MTP_K_KEY: &str = "speculation.mtp_k";
const IU4_KEY: &str = "kernel.iu4_prefill";
const A8_KEY: &str = "kernel.gfx11_a8_prefill";
const SYM_IU4_KEY: &str = "developer.qwen4_moe_sym_iu4";

/// Native Qwen4 (Flash-Next) `model_arch` id.
const QWEN4_ARCH_ID: u32 = 16;

/// Where PR #902 (`kernel.gfx11_a8_prefill`) stands in this build's schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum A8Prefill {
    /// The key is not in the schema.
    NotLanded,
    /// In the schema, off unless asked for.
    OptIn,
    /// In the schema, with a built-in default of `true`.
    DefaultOn,
}

impl A8Prefill {
    /// Detect by schema, never by version: absent → `NotLanded`; present with
    /// a `Bool(true)` default → `DefaultOn`; present otherwise → `OptIn`.
    pub fn detect() -> Self {
        match field(A8_KEY) {
            None => Self::NotLanded,
            Some(schema) => match schema.default.to_value() {
                ConfigValue::Bool(true) => Self::DefaultOn,
                _ => Self::OptIn,
            },
        }
    }
}

/// Everything an arch rule may look at besides the config itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArchCtx {
    /// Exact GPU arch (`gfx1151`, …); `None` when unknown.
    pub gpu_arch: Option<String>,
    /// Visible GPUs; `<= 1` counts as a single GPU.
    pub gpu_count: usize,
    /// Active model's `model_arch` id (16 = Flash-Next), when known.
    pub model_arch_id: Option<u32>,
    /// Registry tag of the active model (`qwen3.8:27b`), when it has one.
    pub registry_tag: Option<String>,
    /// PR #902 state; [`A8Prefill::detect`] in production, explicit in tests.
    pub a8_prefill: A8Prefill,
}

impl Default for ArchCtx {
    /// Unknown GPU and model, one GPU, A8 state detected from the schema.
    fn default() -> Self {
        Self {
            gpu_arch: None,
            gpu_count: 1,
            model_arch_id: None,
            registry_tag: None,
            a8_prefill: A8Prefill::detect(),
        }
    }
}

impl ArchCtx {
    fn arch(&self) -> &str {
        self.gpu_arch.as_deref().unwrap_or("")
    }

    /// `Some(true/false)` once the model family is known, `None` when it has
    /// to be assumed.
    fn is_qwen_family(&self) -> Option<bool> {
        if self.model_arch_id == Some(QWEN4_ARCH_ID) {
            return Some(true);
        }
        self.registry_tag
            .as_deref()
            .map(|tag| tag.trim().to_ascii_lowercase().starts_with("qwen"))
    }
}

/// When a changed value takes effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Applies {
    NextMessage,
    NewChat,
    NextModelLoad,
    ServeRestart,
}

impl Applies {
    /// `Request` → next message, `Session` → new chat, `ModelLoad` → next
    /// model load, `Process`/`Diagnostic` → serve restart.
    pub fn from_scope(scope: ConfigScope) -> Self {
        match scope {
            ConfigScope::Request => Self::NextMessage,
            ConfigScope::Session => Self::NewChat,
            ConfigScope::ModelLoad => Self::NextModelLoad,
            ConfigScope::Process | ConfigScope::Diagnostic => Self::ServeRestart,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::NextMessage => "next message",
            Self::NewChat => "new chat",
            Self::NextModelLoad => "next model load",
            Self::ServeRestart => "serve restart",
        }
    }
}

/// One key's predicted value on this GPU for this model.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Effective {
    /// Concrete value (`auto` replaced where a rule resolves it). `Null` for
    /// an unknown key.
    pub value: ConfigValue,
    /// One line explaining a rule-derived or unused value.
    pub reason: Option<String>,
    /// The built-in value was chosen by an arch/model rule (not the
    /// arch-neutral default), e.g. `fp8` on gfx1201.
    pub by_arch_rule: bool,
    /// The GPU reads this key at all. Unknown GPU counts as used.
    pub used_on_gpu: bool,
    /// The active model reads this key; `None` when not model-specific or the
    /// model is unknown.
    pub used_by_model: Option<bool>,
    pub applies: Applies,
}

/// Predict `key`'s effective value. The key may be canonical or legacy;
/// absent `developer.*` keys count as unset. An unknown key yields
/// `Null` with a reason.
pub fn effective(key: &str, resolved: &ResolvedConfig, ctx: &ArchCtx) -> Effective {
    let schema = field(key);
    let canonical = match crate::canonical_config_key(key) {
        Some(canonical) => canonical,
        None => {
            return Effective {
                value: ConfigValue::Null,
                reason: Some(format!("unknown key {key}")),
                by_arch_rule: false,
                used_on_gpu: true,
                used_by_model: None,
                applies: Applies::ServeRestart,
            }
        }
    };
    let applies = schema.map_or(Applies::ServeRestart, |s| Applies::from_scope(s.scope));
    let slot = resolved.get(&canonical);
    let builtin = slot.is_none_or(|v| matches!(v.source, ConfigSource::BuiltIn));
    let value = slot.map_or(ConfigValue::Null, |v| v.value.clone());
    let mut out = Effective {
        value,
        reason: None,
        by_arch_rule: false,
        used_on_gpu: true,
        used_by_model: None,
        applies,
    };

    match canonical.as_str() {
        KV_KEY if builtin && is_auto(&out.value) => kv_auto(&mut out, ctx),
        PLE_KEY => {
            out.used_by_model =
                known_model(ctx).then_some(ctx.model_arch_id == Some(QWEN4_ARCH_ID));
            if builtin && is_auto(&out.value) {
                ple_auto(&mut out, ctx);
            }
        }
        STATE_QUANT_KEY if builtin && is_auto(&out.value) => state_quant_auto(&mut out, ctx),
        MTP_K_KEY if builtin && is_auto(&out.value) => mtp_k_auto(&mut out, ctx),
        IU4_KEY => {
            if !matches!(ctx.arch(), "" | "gfx1100" | "gfx1151" | "gfx1201") {
                not_used_on_gpu(&mut out);
            }
        }
        A8_KEY => a8_rule(&mut out, builtin, ctx),
        SYM_IU4_KEY => sym_iu4_rule(&mut out, slot.is_none(), ctx),
        _ => {}
    }
    out
}

fn is_auto(value: &ConfigValue) -> bool {
    matches!(value, ConfigValue::String(s) if s == "auto")
}

fn known_model(ctx: &ArchCtx) -> bool {
    ctx.model_arch_id.is_some()
}

fn not_used_on_gpu(out: &mut Effective) {
    out.used_on_gpu = false;
    out.reason = Some("not used on this GPU".into());
}

/// `memory.kv_cache = auto`. Qwen4 (`model_arch` 16) mirrors
/// `kv_mode::resolve_qwen4` (fp8 on exact gfx1201, q8 on gfx1151/gfx1100,
/// bf16 = the exact F32 state elsewhere); every other Qwen-family model
/// mirrors `kv_mode::resolve_kv_pair`'s auto default (fp8 on exact gfx1201
/// with one GPU, else q8).
fn kv_auto(out: &mut Effective, ctx: &ArchCtx) {
    let arch = ctx.arch();
    let set = |out: &mut Effective, value: &str, arch_rule: bool, reason: String| {
        out.value = ConfigValue::String(value.into());
        out.by_arch_rule = arch_rule;
        out.reason = Some(reason);
    };
    if ctx.model_arch_id == Some(QWEN4_ARCH_ID) {
        match arch {
            "gfx1201" => set(out, "fp8", true, "auto: fp8 on exact gfx1201, Qwen".into()),
            "gfx1151" | "gfx1100" => set(
                out,
                "q8",
                true,
                format!("auto: q8 on {arch} (bf16 keeps the exact state), Qwen"),
            ),
            _ => set(
                out,
                "bf16",
                false,
                "auto: bf16 (exact state; fp8 only on exact gfx1201), Qwen".into(),
            ),
        }
        return;
    }
    let family = ctx.is_qwen_family();
    let qwen = family.unwrap_or(true);
    let assumed = if family.is_none() {
        " (model unknown; assumed Qwen)"
    } else {
        ""
    };
    if qwen && arch == "gfx1201" && ctx.gpu_count <= 1 {
        set(
            out,
            "fp8",
            true,
            format!("auto: fp8 on exact gfx1201, one GPU, Qwen{assumed}"),
        );
    } else if !qwen {
        set(out, "q8", false, "auto: q8 (fp8 auto is Qwen-only)".into());
    } else if arch == "gfx1201" {
        set(
            out,
            "q8",
            false,
            format!("auto: q8 (fp8 auto needs one GPU){assumed}"),
        );
    } else if arch.is_empty() {
        set(out, "q8", false, format!("auto: q8 (GPU unknown){assumed}"));
    } else {
        set(out, "q8", false, format!("auto: q8{assumed}"));
    }
}

/// `memory.ple_format = auto` → `PleFormatRequest::Auto.resolve(arch)`.
fn ple_auto(out: &mut Effective, ctx: &ArchCtx) {
    let arch = ctx.arch();
    let format = PleFormatRequest::Auto.resolve(arch);
    out.value = ConfigValue::String(format.name().into());
    out.by_arch_rule = arch == "gfx1151";
    out.reason = Some(if arch == "gfx1151" {
        "auto: q4 on gfx1151 (unified memory)".into()
    } else if arch.is_empty() {
        "auto: q8 (GPU unknown)".into()
    } else {
        "auto: q8".into()
    });
}

/// `memory.state_quant = auto`. Flash-Next (`model_arch` 16) mirrors
/// `hipfire_arch_qwen4::state::resolve_gdn_format_for_arch`: the exact F32
/// GDN state on exact gfx1151, Q8 elsewhere (the AR-only stock-FP8 profile is
/// F32-only, which this view cannot see). Other models keep `auto`; Qwen3.5
/// resolves it to Q8 at load.
fn state_quant_auto(out: &mut Effective, ctx: &ArchCtx) {
    if ctx.model_arch_id != Some(QWEN4_ARCH_ID) {
        return;
    }
    let arch = ctx.arch();
    let (value, reason) = match arch {
        "gfx1151" => (
            "fp32",
            "auto: fp32 on gfx1151 (exact GDN state), Flash-Next".to_string(),
        ),
        "" => (
            "q8",
            "auto: q8 (GPU unknown; fp32 on gfx1151), Flash-Next".to_string(),
        ),
        _ => (
            "q8",
            format!("auto: q8 on {arch} (fp32 only on gfx1151), Flash-Next"),
        ),
    };
    out.value = ConfigValue::String(value.into());
    out.by_arch_rule = arch == "gfx1151";
    out.reason = Some(reason);
}

/// `speculation.mtp_k = auto` → `mtp_k_default` over the GPU's table
/// row-exact verify rows (the runtime additionally honors the developer
/// `HIPFIRE_QWEN4_WIDE_VERIFY_ROWS` oracle override, which the config view
/// does not model).
fn mtp_k_auto(out: &mut Effective, ctx: &ArchCtx) {
    let arch = ctx.arch();
    let rows = qwen4_row_exact_verify_rows_table(arch);
    let k = mtp_k_default(ctx.model_arch_id.unwrap_or(0), rows);
    out.value = ConfigValue::Integer(k as i64);
    out.by_arch_rule = k != crate::MTP_K_DEFAULT;
    out.reason = Some(if out.by_arch_rule {
        format!("auto: {k} for Flash-Next (row-exact verify rows {rows} - 1)")
    } else if ctx.model_arch_id.is_none() {
        format!("auto: {k} (model unknown)")
    } else {
        format!("auto: {k}")
    });
}

/// `kernel.gfx11_a8_prefill` (only reachable once the key is in the schema).
fn a8_rule(out: &mut Effective, builtin: bool, ctx: &ArchCtx) {
    let arch = ctx.arch();
    if !matches!(arch, "" | "gfx1100" | "gfx1151") {
        not_used_on_gpu(out);
        return;
    }
    if builtin && ctx.a8_prefill == A8Prefill::DefaultOn && !arch.is_empty() {
        let on = arch == "gfx1151";
        out.value = ConfigValue::Bool(on);
        out.by_arch_rule = on;
        out.reason = Some(if on {
            "unset: on for gfx1151".into()
        } else {
            "unset: off on this GPU".into()
        });
    }
}

/// `developer.qwen4_moe_sym_iu4`: the route is read on gfx1100, gfx1151 and
/// gfx1201. An unset key requests it on all three (`=0` / `false` is the kill
/// switch).
fn sym_iu4_rule(out: &mut Effective, unset: bool, ctx: &ArchCtx) {
    let arch = ctx.arch();
    out.used_by_model = known_model(ctx).then_some(ctx.model_arch_id == Some(QWEN4_ARCH_ID));
    let on_arch = matches!(arch, "gfx1100" | "gfx1151" | "gfx1201");
    if !arch.is_empty() && !on_arch {
        out.used_on_gpu = false;
        if unset {
            out.value = ConfigValue::Bool(false);
        }
        out.reason = Some("not used on this GPU".into());
        return;
    }
    if unset {
        out.value = ConfigValue::Bool(true);
        out.by_arch_rule = on_arch;
        out.reason = Some("unset: on for gfx1100/gfx1151/gfx1201".into());
    }
}

/// `gpu_arch` recorded by `hipfire install` in `<root>/install.json`.
/// `None` when the file is missing, unreadable, not JSON, or has no usable
/// `gpu_arch` string.
pub fn install_gpu_arch(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("install.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let arch = json.get("gpu_arch")?.as_str()?.trim();
    (!arch.is_empty()).then(|| arch.to_owned())
}

/// Which layer set a key's value, as the Settings "From" column and
/// `hipfire config list` name it (spec §3.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "kebab-case")]
pub enum Origin {
    /// Built-in default.
    Default,
    /// Built-in `auto` or arch-gated default resolved by an arch rule on this GPU.
    DefaultArch(String),
    /// The model's registry card.
    Registry,
    /// A named profile's bundle wrote it into config.toml.
    Profile(String),
    /// The applied tuning preset owns it in config.toml (attributed by
    /// `tuning`; [`origin_of`] never returns it).
    Preset,
    /// You set it for all models (config.toml).
    You,
    /// Per-model value in models.toml.
    ThisModel,
    /// An environment variable of this process; the UI locks the row.
    Env(String),
    /// A one-shot command-line argument.
    OneShot,
}

impl Origin {
    /// Short "From" label: `default`, `default · gfx1201`, `registry`,
    /// `profile dev`, `preset`, `you`, `this model`, `env HIPFIRE_KV_MODE`.
    pub fn label(&self) -> String {
        match self {
            Self::Default => "default".into(),
            Self::DefaultArch(arch) => format!("default · {arch}"),
            Self::Registry => "registry".into(),
            Self::Profile(name) => format!("profile {name}"),
            Self::Preset => "preset".into(),
            Self::You => "you".into(),
            Self::ThisModel => "this model".into(),
            Self::Env(name) => format!("env {name}"),
            Self::OneShot => "one-shot".into(),
        }
    }

    /// Stable machine name (`list --json` "origin").
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Default | Self::DefaultArch(_) => "default",
            Self::Registry => "registry",
            Self::Profile(_) => "profile",
            Self::Preset => "preset",
            Self::You => "you",
            Self::ThisModel => "model",
            Self::Env(_) => "env",
            Self::OneShot => "one-shot",
        }
    }

    /// The value was set by something other than the defaults or the model
    /// card (the `●` marker).
    pub fn marked(&self) -> bool {
        !matches!(self, Self::Default | Self::DefaultArch(_) | Self::Registry)
    }
}

/// Attribute `key`'s value to a layer. `profile` is the detected profile
/// (`detect_config_profile`) and its bundle (`load_config_profile`): a global
/// value equal to that bundle's reads `profile <name>`, any other global
/// value reads `you`. Preset ownership is layered on top by `tuning`.
pub fn origin_of(
    key: &str,
    resolved: &ResolvedConfig,
    effective: &Effective,
    ctx: &ArchCtx,
    profile: Option<(&str, &ConfigLayer)>,
) -> Origin {
    let canonical = crate::canonical_config_key(key).unwrap_or_else(|| key.to_owned());
    let Some(slot) = resolved.get(&canonical) else {
        return builtin_origin(effective, ctx);
    };
    match &slot.source {
        ConfigSource::BuiltIn => builtin_origin(effective, ctx),
        ConfigSource::RegistryModel { .. } | ConfigSource::RegistryTarget { .. } => {
            Origin::Registry
        }
        ConfigSource::GlobalUser { .. } => match profile {
            Some((name, bundle)) if bundle.get(&canonical) == Some(&slot.value) => {
                Origin::Profile(name.to_owned())
            }
            _ => Origin::You,
        },
        ConfigSource::ModelUser { .. } => Origin::ThisModel,
        ConfigSource::LegacyEnv { name } => Origin::Env(
            field(&canonical)
                .and_then(|schema| schema.env_compat.map(str::to_owned))
                .or_else(|| developer_env_for_key(&canonical))
                .unwrap_or_else(|| name.clone()),
        ),
        ConfigSource::OneShot { .. } => Origin::OneShot,
    }
}

fn builtin_origin(effective: &Effective, ctx: &ArchCtx) -> Origin {
    match ctx.gpu_arch.as_deref() {
        Some(arch) if effective.by_arch_rule && !arch.is_empty() => {
            Origin::DefaultArch(arch.to_owned())
        }
        _ => Origin::Default,
    }
}

/// One option of a virtual row: the key writes that select it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Choice {
    pub id: &'static str,
    pub label: &'static str,
    /// `(key, Some(value))` sets the key; `(key, None)` removes it.
    pub writes: Vec<(&'static str, Option<ConfigValue>)>,
    /// This choice is what the built-in defaults give (nothing written).
    pub builtin_default: bool,
}

const FAST_LABEL: &str = "Fast (4-bit math)";
const ACCURATE_LABEL: &str = "Accurate (8-bit math)";
const ACCURATE_FP8_LABEL: &str = "Accurate (fp8)";

/// Prompt processing options for the detected A8 state.
pub fn quality_choices(ctx: &ArchCtx) -> Vec<Choice> {
    quality_choices_for(ctx, ctx.a8_prefill)
}

/// [`quality_choices`] with the A8 state given explicitly (the pure core).
/// Exact gfx1151, gfx1201 and gfx1100 offer Fast/Accurate; any other or
/// unknown GPU has no route and returns an empty list.
pub fn quality_choices_for(ctx: &ArchCtx, a8: A8Prefill) -> Vec<Choice> {
    let choice = |id, label, writes, builtin_default| Choice {
        id,
        label,
        writes,
        builtin_default,
    };
    let bool_write = |key, value| vec![(key, Some(ConfigValue::Bool(value)))];
    match ctx.arch() {
        "gfx1151" => match a8 {
            A8Prefill::NotLanded => vec![
                choice("fast", FAST_LABEL, vec![(IU4_KEY, None)], true),
                choice(
                    "accurate",
                    ACCURATE_LABEL,
                    bool_write(IU4_KEY, false),
                    false,
                ),
            ],
            A8Prefill::OptIn => vec![
                choice("fast", FAST_LABEL, vec![(A8_KEY, None)], true),
                choice("accurate", ACCURATE_LABEL, bool_write(A8_KEY, true), false),
            ],
            A8Prefill::DefaultOn => vec![
                choice("fast", FAST_LABEL, bool_write(A8_KEY, false), false),
                choice("accurate", ACCURATE_LABEL, vec![(A8_KEY, None)], true),
            ],
        },
        "gfx1201" => vec![
            choice("fast", FAST_LABEL, vec![(IU4_KEY, None)], true),
            choice(
                "accurate",
                ACCURATE_FP8_LABEL,
                bool_write(IU4_KEY, false),
                false,
            ),
        ],
        // A8 is never offered on gfx1100: PR #902 has no gfx1100 hardware run.
        "gfx1100" => vec![
            choice("fast", FAST_LABEL, vec![(IU4_KEY, None)], true),
            choice(
                "accurate",
                ACCURATE_LABEL,
                bool_write(IU4_KEY, false),
                false,
            ),
        ],
        _ => Vec::new(),
    }
}

/// Index of the choice the current config selects. `get` reads the key as
/// the user's layers have it (absent → `None`). A removal write matches an
/// absent key or one equal to the schema default; a set write matches an equal
/// value. The first choice whose writes all match wins.
pub fn active_choice(
    choices: &[Choice],
    get: impl Fn(&str) -> Option<ConfigValue>,
) -> Option<usize> {
    choices.iter().position(|choice| {
        choice.writes.iter().all(|(key, want)| {
            let have = get(key);
            match want {
                Some(value) => have.as_ref() == Some(value),
                None => match have {
                    None => true,
                    Some(have) => {
                        field(key).is_some_and(|schema| schema.default.to_value() == have)
                    }
                },
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{resolve, ConfigLayer, NamedLayer};

    fn ctx(arch: &str) -> ArchCtx {
        ArchCtx {
            gpu_arch: Some(arch.into()),
            gpu_count: 1,
            model_arch_id: None,
            registry_tag: Some("qwen3.8:27b".into()),
            a8_prefill: A8Prefill::NotLanded,
        }
    }

    fn defaults() -> ResolvedConfig {
        resolve([]).unwrap()
    }

    fn eff(key: &str, ctx: &ArchCtx) -> Effective {
        effective(key, &defaults(), ctx)
    }

    fn user_layer(pairs: &[(&str, ConfigValue)]) -> NamedLayer {
        let mut layer = ConfigLayer::default();
        for (key, value) in pairs {
            layer.set(key, value.clone()).unwrap();
        }
        NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }
    }

    fn s(v: &str) -> ConfigValue {
        ConfigValue::String(v.into())
    }

    #[test]
    fn kv_auto_is_fp8_only_on_single_gpu_gfx1201_qwen() {
        let e = eff(KV_KEY, &ctx("gfx1201"));
        assert_eq!(e.value, s("fp8"));
        assert!(e.by_arch_rule);
        assert_eq!(
            e.reason.as_deref(),
            Some("auto: fp8 on exact gfx1201, one GPU, Qwen")
        );
        assert_eq!(e.applies, Applies::NextModelLoad);
        for arch in ["gfx1100", "gfx1151", "gfx1030"] {
            let e = eff(KV_KEY, &ctx(arch));
            assert_eq!(e.value, s("q8"), "{arch}");
            assert!(!e.by_arch_rule, "{arch}");
        }
        let mut two = ctx("gfx1201");
        two.gpu_count = 2;
        assert_eq!(eff(KV_KEY, &two).value, s("q8"));
        let mut zero = ctx("gfx1201");
        zero.gpu_count = 0;
        assert_eq!(eff(KV_KEY, &zero).value, s("fp8"));
    }

    #[test]
    fn kv_auto_unknown_model_assumes_qwen_and_says_so() {
        let mut c = ctx("gfx1201");
        c.registry_tag = None;
        let e = eff(KV_KEY, &c);
        assert_eq!(e.value, s("fp8"));
        assert!(e.reason.unwrap().contains("assumed Qwen"));
        // A known non-Qwen model keeps q8.
        c.registry_tag = Some("gemma4:e4b".into());
        assert_eq!(eff(KV_KEY, &c).value, s("q8"));
    }

    #[test]
    fn kv_auto_mirrors_resolve_qwen4_for_model_arch_16() {
        let mut c = ctx("gfx1201");
        c.model_arch_id = Some(16);
        assert_eq!(eff(KV_KEY, &c).value, s("fp8"));
        for arch in ["gfx1100", "gfx1151"] {
            let mut c = ctx(arch);
            c.model_arch_id = Some(16);
            assert_eq!(eff(KV_KEY, &c).value, s("q8"), "{arch}");
        }
        for arch in ["gfx942", "gfx1030"] {
            let mut c = ctx(arch);
            c.model_arch_id = Some(16);
            assert_eq!(eff(KV_KEY, &c).value, s("bf16"), "{arch}");
        }
    }

    #[test]
    fn registry_card_kv_mode_passes_through() {
        let card = NamedLayer {
            source: ConfigSource::RegistryModel {
                tag: "qwen3.8:27b".into(),
                revision: "r1".into(),
            },
            layer: {
                let mut layer = ConfigLayer::default();
                layer.set(KV_KEY, s("fwht3")).unwrap();
                layer
            },
        };
        let resolved = resolve([card]).unwrap();
        let e = effective(KV_KEY, &resolved, &ctx("gfx1201"));
        assert_eq!(e.value, s("fwht3"));
        assert!(!e.by_arch_rule);
        assert_eq!(e.reason, None);
        assert!(matches!(
            resolved.get(KV_KEY).unwrap().source,
            ConfigSource::RegistryModel { .. }
        ));
    }

    #[test]
    fn explicit_user_value_is_not_rewritten_by_an_arch_rule() {
        let resolved = resolve([user_layer(&[(PLE_KEY, s("q8")), (KV_KEY, s("q8"))])]).unwrap();
        let e = effective(PLE_KEY, &resolved, &ctx("gfx1151"));
        assert_eq!(e.value, s("q8"));
        assert!(!e.by_arch_rule);
        assert_eq!(effective(KV_KEY, &resolved, &ctx("gfx1201")).value, s("q8"));
    }

    #[test]
    fn ple_auto_is_q4_on_gfx1151_and_q8_elsewhere() {
        let mut c = ctx("gfx1151");
        let e = eff(PLE_KEY, &c);
        assert_eq!(e.value, s("q4"));
        assert!(e.by_arch_rule);
        assert_eq!(e.used_by_model, None);
        c.model_arch_id = Some(16);
        assert_eq!(eff(PLE_KEY, &c).used_by_model, Some(true));
        c.model_arch_id = Some(5);
        assert_eq!(eff(PLE_KEY, &c).used_by_model, Some(false));
        for arch in ["gfx1100", "gfx1201", "gfx1030"] {
            let e = eff(PLE_KEY, &ctx(arch));
            assert_eq!(e.value, s("q8"), "{arch}");
            assert!(!e.by_arch_rule);
        }
        assert_eq!(eff(PLE_KEY, &c).applies, Applies::ServeRestart);
    }

    #[test]
    fn physical_verify_capacity_does_not_promote_shipping_admission() {
        assert_eq!(crate::QWEN4_VERIFY_MAX_ROWS, 64);
        assert_eq!(crate::QWEN4_FEW_ROW_ROUTE_ROWS, 8);
        for arch in [
            "gfx1151",
            "gfx1201",
            "gfx1100",
            "gfx1151:sramecc+",
            "unknown",
            "",
        ] {
            let rows = qwen4_row_exact_verify_rows_table(arch);
            assert_eq!(rows, 8, "{arch}");
            assert!(rows <= crate::QWEN4_VERIFY_MAX_ROWS, "{arch}");
            let mut c = ctx(arch);
            c.model_arch_id = Some(16);
            assert_eq!(eff(MTP_K_KEY, &c).value, ConfigValue::Integer(7), "{arch}");
        }
    }

    #[test]
    fn mtp_k_auto_is_row_exact_rows_minus_one_for_flash_next_else_3() {
        for arch in ["gfx1151", "gfx1201", "gfx1100", ""] {
            let mut c = ctx(arch);
            c.model_arch_id = Some(16);
            let e = eff(MTP_K_KEY, &c);
            assert_eq!(e.value, ConfigValue::Integer(7), "{arch}");
            assert!(e.by_arch_rule, "{arch}");
        }
        for (arch, model) in [("gfx1151", Some(5)), ("gfx1151", None), ("gfx1201", None)] {
            let mut c = ctx(arch);
            c.model_arch_id = model;
            let e = eff(MTP_K_KEY, &c);
            assert_eq!(e.value, ConfigValue::Integer(3), "{arch} {model:?}");
            assert!(!e.by_arch_rule);
        }
        // An explicit value is kept.
        let resolved = resolve([user_layer(&[(MTP_K_KEY, ConfigValue::Integer(5))])]).unwrap();
        let mut c = ctx("gfx1151");
        c.model_arch_id = Some(16);
        assert_eq!(
            effective(MTP_K_KEY, &resolved, &c).value,
            ConfigValue::Integer(5)
        );
    }

    #[test]
    fn state_quant_auto_is_fp32_for_flash_next_on_gfx1151_else_q8() {
        let mut c = ctx("gfx1151");
        c.model_arch_id = Some(16);
        let e = eff(STATE_QUANT_KEY, &c);
        assert_eq!(e.value, s("fp32"));
        assert!(e.by_arch_rule);
        assert_eq!(e.applies, Applies::NextModelLoad);
        for arch in ["gfx1100", "gfx1201", ""] {
            let mut c = ctx(arch);
            c.model_arch_id = Some(16);
            let e = eff(STATE_QUANT_KEY, &c);
            assert_eq!(e.value, s("q8"), "{arch}");
            assert!(!e.by_arch_rule, "{arch}");
        }
        // Other or unknown models keep `auto`.
        for model in [Some(5), None] {
            let mut c = ctx("gfx1151");
            c.model_arch_id = model;
            assert_eq!(eff(STATE_QUANT_KEY, &c).value, s("auto"), "{model:?}");
        }
        // An explicit value is kept.
        let resolved = resolve([user_layer(&[(STATE_QUANT_KEY, s("q8"))])]).unwrap();
        let mut c = ctx("gfx1151");
        c.model_arch_id = Some(16);
        let e = effective(STATE_QUANT_KEY, &resolved, &c);
        assert_eq!(e.value, s("q8"));
        assert!(!e.by_arch_rule);
    }

    #[test]
    fn iu4_prefill_is_used_only_on_gfx1100_gfx1151_gfx1201() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let e = eff(IU4_KEY, &ctx(arch));
            assert!(e.used_on_gpu, "{arch}");
            assert_eq!(e.value, ConfigValue::Bool(true));
            assert_eq!(e.reason, None);
            assert_eq!(e.applies, Applies::ServeRestart);
        }
        for arch in ["gfx1030", "gfx1101", "gfx942"] {
            let e = eff(IU4_KEY, &ctx(arch));
            assert!(!e.used_on_gpu, "{arch}");
            assert_eq!(e.reason.as_deref(), Some("not used on this GPU"));
        }
        let unknown = ArchCtx {
            gpu_arch: None,
            ..ctx("gfx1030")
        };
        let e = eff(IU4_KEY, &unknown);
        assert!(e.used_on_gpu);
        assert_eq!(e.reason, None);
    }

    #[test]
    fn sym_iu4_unset_is_on_for_gfx1100_gfx1151_and_gfx1201() {
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let e = eff(SYM_IU4_KEY, &ctx(arch));
            assert_eq!(e.value, ConfigValue::Bool(true), "{arch}");
            assert!(e.by_arch_rule && e.used_on_gpu);
            assert_eq!(
                e.reason.as_deref(),
                Some("unset: on for gfx1100/gfx1151/gfx1201")
            );
            assert_eq!(e.applies, Applies::ServeRestart);
        }
        let mut c = ctx("gfx1151");
        c.model_arch_id = Some(16);
        assert_eq!(eff(SYM_IU4_KEY, &c).used_by_model, Some(true));
        c.model_arch_id = Some(3);
        assert_eq!(eff(SYM_IU4_KEY, &c).used_by_model, Some(false));
    }

    #[test]
    fn sym_iu4_is_unused_on_other_arches_and_explicit_gfx1100_is_kept() {
        for arch in ["gfx1030", "gfx1101", "gfx942"] {
            let e = eff(SYM_IU4_KEY, &ctx(arch));
            assert!(!e.used_on_gpu, "{arch}");
            assert_eq!(e.value, ConfigValue::Bool(false), "{arch}");
            assert_eq!(e.reason.as_deref(), Some("not used on this GPU"), "{arch}");
        }
        // An explicit value on gfx1100 (`false` is the kill switch) is kept and the route is
        // read there.
        for value in [false, true] {
            let resolved =
                resolve([user_layer(&[(SYM_IU4_KEY, ConfigValue::Bool(value))])]).unwrap();
            let e = effective(SYM_IU4_KEY, &resolved, &ctx("gfx1100"));
            assert_eq!(e.value, ConfigValue::Bool(value));
            assert!(!e.by_arch_rule && e.used_on_gpu);
            assert_eq!(e.reason, None);
        }
    }

    #[test]
    fn sym_iu4_explicit_false_is_kept() {
        let resolved = resolve([user_layer(&[(SYM_IU4_KEY, ConfigValue::Bool(false))])]).unwrap();
        let e = effective(SYM_IU4_KEY, &resolved, &ctx("gfx1151"));
        assert_eq!(e.value, ConfigValue::Bool(false));
        assert!(!e.by_arch_rule);
        assert!(e.used_on_gpu);
        assert_eq!(e.reason, None);
    }

    #[test]
    fn a8_default_on_rule_is_on_for_gfx1151_only() {
        // The rule is unreachable through `effective()` until the key is in the
        // schema, so drive it directly with a synthetic built-in `true`.
        let run = |arch: &str, a8, builtin| {
            let mut c = ctx(arch);
            c.a8_prefill = a8;
            let mut out = Effective {
                value: ConfigValue::Bool(a8 == A8Prefill::DefaultOn),
                reason: None,
                by_arch_rule: false,
                used_on_gpu: true,
                used_by_model: None,
                applies: Applies::ServeRestart,
            };
            a8_rule(&mut out, builtin, &c);
            out
        };
        let e = run("gfx1151", A8Prefill::DefaultOn, true);
        assert_eq!(e.value, ConfigValue::Bool(true));
        assert!(e.by_arch_rule && e.used_on_gpu);
        let e = run("gfx1100", A8Prefill::DefaultOn, true);
        assert_eq!(e.value, ConfigValue::Bool(false));
        assert!(e.used_on_gpu && !e.by_arch_rule);
        let e = run("gfx1030", A8Prefill::DefaultOn, true);
        assert!(!e.used_on_gpu);
        assert_eq!(e.reason.as_deref(), Some("not used on this GPU"));
        // Opt-in and an explicit value pass through.
        let e = run("gfx1151", A8Prefill::OptIn, true);
        assert_eq!(e.value, ConfigValue::Bool(false));
        assert!(!e.by_arch_rule);
        let e = run("gfx1151", A8Prefill::DefaultOn, false);
        assert_eq!(e.value, ConfigValue::Bool(true));
        assert!(!e.by_arch_rule);
    }

    #[test]
    fn a8_rule_follows_the_schema_state() {
        // Only runs when the key exists in this build's schema.
        let Some(schema) = field(A8_KEY) else {
            let e = eff(A8_KEY, &ctx("gfx1151"));
            assert_eq!(e.value, ConfigValue::Null);
            return;
        };
        let e = eff(A8_KEY, &ctx("gfx1030"));
        assert!(!e.used_on_gpu);
        let e = eff(A8_KEY, &ctx("gfx1100"));
        assert!(e.used_on_gpu);
        let mut c = ctx("gfx1151");
        c.a8_prefill = A8Prefill::detect();
        let e = eff(A8_KEY, &c);
        if schema.default.to_value() == ConfigValue::Bool(true) {
            assert_eq!(e.value, ConfigValue::Bool(true));
            assert!(e.by_arch_rule);
        } else {
            assert_eq!(e.value, ConfigValue::Bool(false));
        }
    }

    #[test]
    fn a8_detect_matches_the_schema() {
        let want = match field(A8_KEY).map(|f| f.default.to_value()) {
            None => A8Prefill::NotLanded,
            Some(ConfigValue::Bool(true)) => A8Prefill::DefaultOn,
            Some(_) => A8Prefill::OptIn,
        };
        assert_eq!(A8Prefill::detect(), want);
        assert_eq!(ArchCtx::default().a8_prefill, want);
    }

    #[test]
    fn other_keys_pass_through_with_scope_applies() {
        let c = ctx("gfx1151");
        for key in ["reasoning.effort", "memory.max_seq", "serve.port"] {
            let schema = field(key).unwrap();
            let e = eff(key, &c);
            assert_eq!(e.value, schema.default.to_value(), "{key}");
            assert_eq!(e.applies, Applies::from_scope(schema.scope), "{key}");
            assert!(
                e.used_on_gpu && !e.by_arch_rule && e.reason.is_none(),
                "{key}"
            );
        }
        // Legacy spelling resolves to the canonical key.
        assert_eq!(eff("kv_cache", &c).value, s("q8"));
        // developer.* keys are always ServeRestart.
        assert_eq!(
            eff("developer.something_else", &c).applies,
            Applies::ServeRestart
        );
        assert_eq!(eff("developer.something_else", &c).value, ConfigValue::Null);
        assert_eq!(eff("nope.nope", &c).value, ConfigValue::Null);
    }

    #[test]
    fn applies_mapping_and_labels() {
        use ConfigScope::*;
        for (scope, applies, label) in [
            (Request, Applies::NextMessage, "next message"),
            (Session, Applies::NewChat, "new chat"),
            (ModelLoad, Applies::NextModelLoad, "next model load"),
            (Process, Applies::ServeRestart, "serve restart"),
            (Diagnostic, Applies::ServeRestart, "serve restart"),
        ] {
            assert_eq!(Applies::from_scope(scope), applies);
            assert_eq!(applies.label(), label);
        }
    }

    fn key_writes(c: &Choice) -> Vec<(&'static str, Option<ConfigValue>)> {
        c.writes.clone()
    }

    fn b(v: bool) -> Option<ConfigValue> {
        Some(ConfigValue::Bool(v))
    }

    #[test]
    fn quality_choices_match_spec_for_every_arch_and_a8_state() {
        type Row = (&'static str, Vec<(&'static str, Option<ConfigValue>)>, bool);
        let row = |label, writes, default| -> Row { (label, writes, default) };
        let summary = |arch: &str, a8| -> Vec<Row> {
            quality_choices_for(&ctx(arch), a8)
                .into_iter()
                .map(|c| (c.label, key_writes(&c), c.builtin_default))
                .collect()
        };
        assert_eq!(
            summary("gfx1151", A8Prefill::NotLanded),
            [
                row(FAST_LABEL, vec![(IU4_KEY, None)], true),
                row(ACCURATE_LABEL, vec![(IU4_KEY, b(false))], false),
            ]
        );
        assert_eq!(
            summary("gfx1151", A8Prefill::OptIn),
            [
                row(FAST_LABEL, vec![(A8_KEY, None)], true),
                row(ACCURATE_LABEL, vec![(A8_KEY, b(true))], false),
            ]
        );
        assert_eq!(
            summary("gfx1151", A8Prefill::DefaultOn),
            [
                row(FAST_LABEL, vec![(A8_KEY, b(false))], false),
                row(ACCURATE_LABEL, vec![(A8_KEY, None)], true),
            ]
        );
        for a8 in [A8Prefill::NotLanded, A8Prefill::OptIn, A8Prefill::DefaultOn] {
            assert_eq!(
                summary("gfx1201", a8),
                [
                    row(FAST_LABEL, vec![(IU4_KEY, None)], true),
                    row(ACCURATE_FP8_LABEL, vec![(IU4_KEY, b(false))], false),
                ]
            );
            // A8 is never offered on gfx1100.
            assert_eq!(
                summary("gfx1100", a8),
                [
                    row(FAST_LABEL, vec![(IU4_KEY, None)], true),
                    row(ACCURATE_LABEL, vec![(IU4_KEY, b(false))], false),
                ]
            );
            for arch in ["gfx1030", "gfx1101", "gfx942"] {
                assert!(summary(arch, a8).is_empty(), "{arch}");
            }
        }
        let unknown = ArchCtx {
            gpu_arch: None,
            ..ctx("gfx1151")
        };
        assert!(quality_choices_for(&unknown, A8Prefill::OptIn).is_empty());
        // quality_choices uses the ctx's own state.
        let mut c = ctx("gfx1151");
        c.a8_prefill = A8Prefill::OptIn;
        assert_eq!(
            quality_choices(&c),
            quality_choices_for(&c, A8Prefill::OptIn)
        );
        let ids: Vec<_> = quality_choices(&c).iter().map(|c| c.id).collect();
        assert_eq!(ids, ["fast", "accurate"]);
    }

    #[test]
    fn active_choice_reads_the_layer_state() {
        let layer = |pairs: &'static [(&'static str, Option<bool>)]| {
            move |key: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == key)
                    .and_then(|(_, v)| v.map(ConfigValue::Bool))
            }
        };
        let not_landed = quality_choices_for(&ctx("gfx1151"), A8Prefill::NotLanded);
        assert_eq!(active_choice(&not_landed, layer(&[])), Some(0));
        // Explicit true equals the schema default: still Fast.
        assert_eq!(
            active_choice(&not_landed, layer(&[(IU4_KEY, Some(true))])),
            Some(0)
        );
        assert_eq!(
            active_choice(&not_landed, layer(&[(IU4_KEY, Some(false))])),
            Some(1)
        );
        let opt_in = quality_choices_for(&ctx("gfx1151"), A8Prefill::OptIn);
        assert_eq!(active_choice(&opt_in, layer(&[])), Some(0));
        assert_eq!(
            active_choice(&opt_in, layer(&[(A8_KEY, Some(true))])),
            Some(1)
        );
        let default_on = quality_choices_for(&ctx("gfx1151"), A8Prefill::DefaultOn);
        assert_eq!(active_choice(&default_on, layer(&[])), Some(1));
        assert_eq!(
            active_choice(&default_on, layer(&[(A8_KEY, Some(false))])),
            Some(0)
        );
        assert_eq!(active_choice(&[], layer(&[])), None);
    }

    #[test]
    fn install_gpu_arch_reads_install_json() {
        let root =
            std::env::temp_dir().join(format!("hipfire-effective-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(install_gpu_arch(&root), None);
        std::fs::write(root.join("install.json"), "not json").unwrap();
        assert_eq!(install_gpu_arch(&root), None);
        std::fs::write(root.join("install.json"), r#"{"gpu_arch": null}"#).unwrap();
        assert_eq!(install_gpu_arch(&root), None);
        std::fs::write(root.join("install.json"), r#"{"gpu_arch": ""}"#).unwrap();
        assert_eq!(install_gpu_arch(&root), None);
        std::fs::write(root.join("install.json"), r#"{"commit":"x"}"#).unwrap();
        assert_eq!(install_gpu_arch(&root), None);
        std::fs::write(
            root.join("install.json"),
            r#"{"commit":"x","gpu_arch":"gfx1151"}"#,
        )
        .unwrap();
        assert_eq!(install_gpu_arch(&root).as_deref(), Some("gfx1151"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn origin_labels_each_layer() {
        let gfx1201 = ctx("gfx1201");
        let resolved = defaults();
        let kv = effective("memory.kv_cache", &resolved, &gfx1201);
        let origin = origin_of("memory.kv_cache", &resolved, &kv, &gfx1201, None);
        assert_eq!(origin, Origin::DefaultArch("gfx1201".into()));
        assert_eq!(origin.label(), "default · gfx1201");
        assert!(!origin.marked());
        let effort = effective("reasoning.effort", &resolved, &gfx1201);
        assert_eq!(
            origin_of("reasoning.effort", &resolved, &effort, &gfx1201, None),
            Origin::Default
        );

        let mut env = ConfigLayer::default();
        env.set("memory.kv_cache", ConfigValue::String("q8".into()))
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: env,
        }])
        .unwrap();
        let kv = effective("memory.kv_cache", &resolved, &gfx1201);
        let origin = origin_of("memory.kv_cache", &resolved, &kv, &gfx1201, None);
        assert_eq!(origin.label(), "env HIPFIRE_KV_MODE");
        assert!(origin.marked());

        let resolved = resolve([user_layer(&[(
            "reasoning.effort",
            ConfigValue::String("low".into()),
        )])])
        .unwrap();
        let effort = effective("reasoning.effort", &resolved, &gfx1201);
        assert_eq!(
            origin_of("reasoning.effort", &resolved, &effort, &gfx1201, None),
            Origin::You
        );
        let mut bundle = ConfigLayer::default();
        bundle
            .set("reasoning.effort", ConfigValue::String("low".into()))
            .unwrap();
        let origin = origin_of(
            "reasoning.effort",
            &resolved,
            &effort,
            &gfx1201,
            Some(("dev", &bundle)),
        );
        assert_eq!(origin.label(), "profile dev");
        assert_eq!(origin.kind(), "profile");
    }
}
