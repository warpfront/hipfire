// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DSpine drafter sidecar metadata (arch 24).
//!
//! `dspine_convert` writes the fail-closed contract at conversion time; the
//! loader re-checks it here before any bridge tensor reaches the GPU (the
//! plan doc's double-check rule, mirroring the upstream runtime's own config
//! validation). A sidecar that claims arch 24 but violates the contract is a
//! pointing refusal, never a silent degradation.

use crate::dflash::DflashConfig;
use crate::hfq::HfqFile;

/// Parsed + re-validated DSpine sidecar metadata. `trunk` carries the DFlash
/// decoder knobs (the trunk rides the existing DFlash draft-load path); the
/// remaining fields are the code-bridge contract the arch-24 loader consumes.
#[derive(Debug)]
pub struct DspineConfig {
    pub trunk: DflashConfig,
    pub code_size: usize,
    pub bridge_rank: usize,
    pub second_pass_mode: String,
    pub first_exact: usize,
    pub head_topk: usize,
    pub feature_layers: Vec<usize>,
}

impl DspineConfig {
    /// Open an HFQ sidecar and parse its metadata. `Ok(None)` when the file
    /// is not a DSpine sidecar (no `dspine` metadata block) — callers pick
    /// the right refusal for that case; `Err` carries the contract violation.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Option<Self>, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("metadata JSON malformed: {e}"))?;
        Self::from_metadata(&meta)
    }

    /// Metadata-only parse + contract double-check.
    pub fn from_metadata(meta: &serde_json::Value) -> Result<Option<Self>, String> {
        let dsp = match meta.get("dspine") {
            Some(v) => v,
            None => return Ok(None),
        };
        let trunk = DflashConfig::from_metadata(meta)
            .ok_or("sidecar is missing the `dflash` trunk metadata block")?;

        let code_size = dsp
            .get("code_size")
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing code_size")? as usize;
        let bridge_rank = dsp
            .get("bridge_rank")
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing bridge_rank")? as usize;
        let mode = dsp
            .get("second_pass")
            .and_then(|s| s.get("mode"))
            .and_then(|v| v.as_str())
            .ok_or("dspine block missing second_pass.mode")?;
        if mode != "serial_table" {
            return Err(format!("second_pass.mode {mode:?} != \"serial_table\""));
        }
        let first_exact = dsp
            .get("second_pass")
            .and_then(|s| s.get("first_exact"))
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing second_pass.first_exact")? as usize;
        if first_exact != trunk.n_layers.saturating_sub(1) {
            return Err(format!(
                "second_pass.first_exact {first_exact} != num_hidden_layers {} - 1",
                trunk.n_layers
            ));
        }
        let head_topk = dsp
            .get("second_pass")
            .and_then(|s| s.get("head_topk"))
            .and_then(|v| v.as_u64())
            .ok_or("dspine block missing second_pass.head_topk")? as usize;
        if !(2..=64).contains(&head_topk) {
            return Err(format!("head_topk {head_topk} outside [2, 64]"));
        }
        if !head_topk.is_power_of_two() {
            return Err(format!("head_topk {head_topk} is not a power of two"));
        }
        let feature_layers = dsp
            .get("feature_layers")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect::<Vec<usize>>()
            })
            .ok_or("dspine block missing feature_layers")?;
        if feature_layers.is_empty() {
            return Err("feature_layers is empty".into());
        }
        for &l in &feature_layers {
            if l == 0 || l > trunk.n_layers {
                return Err(format!(
                    "feature_layers entry {l} outside 1..={}",
                    trunk.n_layers
                ));
            }
        }

        // The four version markers the converter validated; the loader
        // re-checks them against the same contract (plan doc §6).
        let versions = dsp
            .get("versions")
            .and_then(|v| v.as_object())
            .ok_or("dspine block missing versions")?;
        const VERSION_KEYS: [&str; 4] = [
            "draft_block_causal_version",
            "last_write_refine_version",
            "selected_prefix_bridge.independent_write_version",
            "selected_prefix_bridge.terminal_disabled_version",
        ];
        for key in VERSION_KEYS {
            match versions.get(key).and_then(|v| v.as_i64()) {
                Some(1) => {}
                Some(other) => return Err(format!("version scalar {key} = {other}, expected 1")),
                None => return Err(format!("versions missing {key}")),
            }
        }

        Ok(Some(Self {
            trunk,
            code_size,
            bridge_rank,
            second_pass_mode: mode.to_string(),
            first_exact,
            head_topk,
            feature_layers,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::DspineConfig;
    use serde_json::json;

    fn valid_metadata() -> serde_json::Value {
        json!({
            "architecture": "dspine",
            "dflash": {
                "num_hidden_layers": 5,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 248320,
                "block_size": 16,
                "mask_token_id": 248070,
                "target_layer_ids": [1, 16, 31, 46, 61],
                "num_target_layers": 64,
                "draft_dtype": "mq4"
            },
            "dspine": {
                "code_size": 1024,
                "bridge_rank": 512,
                "feature_layers": [1, 2, 3, 4, 5],
                "second_pass": {"mode": "serial_table", "first_exact": 4, "head_topk": 16},
                "versions": {
                    "draft_block_causal_version": 1,
                    "last_write_refine_version": 1,
                    "selected_prefix_bridge.independent_write_version": 1,
                    "selected_prefix_bridge.terminal_disabled_version": 1
                }
            }
        })
    }

    fn with_dspine_key(key: &str, value: serde_json::Value) -> serde_json::Value {
        let mut m = valid_metadata();
        m.get_mut("dspine")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), value);
        m
    }

    #[test]
    fn accepts_the_converted_contract_shape() {
        let c = DspineConfig::from_metadata(&valid_metadata())
            .unwrap()
            .expect("valid metadata must parse");
        assert_eq!(c.code_size, 1024);
        assert_eq!(c.bridge_rank, 512);
        assert_eq!(c.trunk.hidden, 5120);
        assert_eq!(c.trunk.block_size, 16);
        assert_eq!(c.trunk.target_layer_ids, vec![1, 16, 31, 46, 61]);
        assert_eq!(c.head_topk, 16);
        assert_eq!(c.feature_layers, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn a_non_dspine_hfq_is_ok_none() {
        let mut m = valid_metadata();
        m.as_object_mut().unwrap().remove("dspine");
        assert!(DspineConfig::from_metadata(&m).unwrap().is_none());
    }

    #[test]
    fn missing_trunk_block_is_a_refusal() {
        let mut m = valid_metadata();
        m.as_object_mut().unwrap().remove("dflash");
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("trunk metadata"), "{err}");
    }

    #[test]
    fn non_serial_table_mode_is_refused() {
        let mut m = valid_metadata();
        m["dspine"]["second_pass"] = json!({"mode": "chain", "first_exact": 4, "head_topk": 16});
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("serial_table"), "{err}");
    }

    #[test]
    fn first_exact_must_match_layers_minus_one() {
        let mut m = valid_metadata();
        m["dspine"]["second_pass"]["first_exact"] = json!(3);
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("first_exact"), "{err}");
    }

    #[test]
    fn head_topk_bounds_and_pow2() {
        for bad in [1u64, 3, 48, 128] {
            let mut m = valid_metadata();
            m["dspine"]["second_pass"]["head_topk"] = json!(bad);
            let err = DspineConfig::from_metadata(&m).unwrap_err();
            assert!(err.contains("head_topk"), "{err}");
        }
    }

    #[test]
    fn feature_layers_must_land_on_draft_layers() {
        let m = with_dspine_key("feature_layers", json!([0, 2, 3, 4, 5]));
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("feature_layers"), "{err}");
        let m = with_dspine_key("feature_layers", json!([1, 2, 3, 4, 6]));
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("feature_layers"), "{err}");
        let m = with_dspine_key("feature_layers", json!([]));
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn every_version_scalar_is_rechecked() {
        for key in [
            "draft_block_causal_version",
            "last_write_refine_version",
            "selected_prefix_bridge.independent_write_version",
            "selected_prefix_bridge.terminal_disabled_version",
        ] {
            let mut m = valid_metadata();
            m["dspine"]["versions"][key] = json!(2);
            let err = DspineConfig::from_metadata(&m).unwrap_err();
            assert!(err.contains(key) && err.contains("expected 1"), "{err}");
        }
        let m = with_dspine_key(
            "versions",
            json!({"draft_block_causal_version": 1, "last_write_refine_version": 1}),
        );
        let err = DspineConfig::from_metadata(&m).unwrap_err();
        assert!(err.contains("missing"), "{err}");
    }
}
