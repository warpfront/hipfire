// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Engine-side RoPE scaling policy: what a checkpoint's text config declares
//! (`rope_parameters`, or the legacy `rope_scaling`), with the operator's
//! `attention.yarn_factor` override applied. Families call [`rope_scaling`]
//! and pass the result through as part of an [`rdna_compute::rope::Rope`].

use rdna_compute::rope::{RopeScaling, Yarn};
use serde_json::Value;

/// The RoPE scaling `text_config` declares, overridden by
/// `attention.yarn_factor` / `HIPFIRE_YARN_FACTOR` when set (`1` disables a
/// checkpoint's YaRN; above 1 enables YaRN with that factor over the
/// checkpoint's original context).
pub fn rope_scaling(text_config: &Value) -> Result<RopeScaling, String> {
    resolve(text_config, crate::config::get().yarn_factor)
}

fn resolve(text: &Value, yarn_factor: Option<f64>) -> Result<RopeScaling, String> {
    let params = ["rope_parameters", "rope_scaling"]
        .iter()
        .find_map(|key| text.get(*key).filter(|v| v.is_object()));
    let float = |key: &str| params.and_then(|p| p.get(key)).and_then(Value::as_f64);
    let kind = params
        .and_then(|p| p.get("rope_type").or_else(|| p.get("type")))
        .and_then(Value::as_str)
        .unwrap_or("default");
    let declared_yarn = match kind {
        "default" => false,
        "yarn" => true,
        other => {
            return Err(format!(
                "unsupported rope_type {other:?} (supported: default, yarn)"
            ))
        }
    };
    if !declared_yarn && yarn_factor.is_none() {
        return Ok(RopeScaling::None);
    }
    let max_position_embeddings = text
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok());
    let original = params
        .and_then(|p| p.get("original_max_position_embeddings"))
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .or(max_position_embeddings)
        .filter(|&n| n > 0)
        .ok_or("YaRN needs original_max_position_embeddings or max_position_embeddings")?;
    let mscale = float("mscale").zip(float("mscale_all_dim"));
    let factor = match yarn_factor {
        Some(factor) if factor <= 1.0 => return Ok(RopeScaling::None),
        Some(factor) => factor,
        // HF: an absent factor is max_position_embeddings / original.
        None => match float("factor") {
            Some(factor) => factor,
            None => max_position_embeddings.ok_or("YaRN needs a factor")? as f64 / original as f64,
        },
    };
    if !(factor.is_finite() && factor >= 1.0) {
        return Err(format!("YaRN factor {factor} must be finite and >= 1"));
    }
    // A declared attention_factor was tuned for the declared factor only.
    let attention_factor = if yarn_factor.is_none() {
        float("attention_factor")
    } else {
        None
    };
    Ok(RopeScaling::Yarn(Yarn::new(
        factor,
        original,
        float("beta_fast"),
        float("beta_slow"),
        attention_factor,
        mscale,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qwen38(rope_parameters: Value) -> Value {
        json!({"max_position_embeddings": 262144, "rope_parameters": rope_parameters})
    }

    fn yarn(scaling: RopeScaling) -> Yarn {
        match scaling {
            RopeScaling::Yarn(y) => y,
            RopeScaling::None => panic!("expected YaRN"),
        }
    }

    #[test]
    fn default_rope_stays_unscaled_without_an_override() {
        let text = qwen38(
            json!({"rope_type": "default", "rope_theta": 1e7, "mrope_section": [11, 11, 10]}),
        );
        assert_eq!(resolve(&text, None).unwrap(), RopeScaling::None);
        assert_eq!(
            resolve(&json!({"rope_theta": 1e6}), None).unwrap(),
            RopeScaling::None
        );
    }

    #[test]
    fn readme_yarn_override_in_config_and_from_the_operator_agree() {
        // Qwen3.8 README: rope_type yarn, factor 4, original 262144.
        let edited = qwen38(json!({
            "rope_type": "yarn", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
            "factor": 4.0, "original_max_position_embeddings": 262144
        }));
        let declared = yarn(resolve(&edited, None).unwrap());
        assert_eq!(
            (declared.factor, declared.original_max_position_embeddings),
            (4.0, 262_144)
        );
        assert_eq!((declared.beta_fast, declared.beta_slow), (32.0, 1.0));
        let operator = resolve(&qwen38(json!({"rope_type": "default"})), Some(4.0)).unwrap();
        assert_eq!(yarn(operator), declared);
    }

    #[test]
    fn operator_factor_replaces_or_disables_a_declared_yarn() {
        let text = json!({
            "max_position_embeddings": 163840,
            "rope_scaling": {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096,
                             "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0,
                             "attention_factor": 0.9}
        });
        let declared = yarn(resolve(&text, None).unwrap());
        assert_eq!((declared.factor, declared.attention_factor), (40.0, 0.9));
        let replaced = yarn(resolve(&text, Some(8.0)).unwrap());
        assert_eq!(
            (replaced.factor, replaced.original_max_position_embeddings),
            (8.0, 4096)
        );
        // mscale == mscale_all_dim: the recomputed factor cancels to 1.
        assert_eq!(replaced.attention_factor, 1.0);
        assert_eq!(resolve(&text, Some(1.0)).unwrap(), RopeScaling::None);
    }

    #[test]
    fn absent_factor_follows_hf_and_unknown_types_are_refused() {
        let text = qwen38(json!({"rope_type": "yarn", "original_max_position_embeddings": 65536}));
        assert_eq!(yarn(resolve(&text, None).unwrap()).factor, 4.0);
        let err = resolve(&qwen38(json!({"rope_type": "longrope"})), None).unwrap_err();
        assert!(err.contains("longrope"), "{err}");
        // A shrinking factor would invert the ramp and the context limit.
        let shrink = qwen38(json!({"rope_type": "yarn", "factor": 0.5}));
        assert!(resolve(&shrink, None).unwrap_err().contains(">= 1"));
    }
}
