// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validated view of the Inkling-2 weights inside an HFQ file.
//!
//! Phase 1 stores physical weights once (24 layers) with per-expert tensors
//! for the routed MoE (128 experts × gate_up/down per layer). This module
//! indexes and validates that set by name/shape, then hands raw `quant_type`
//! + bytes to the forward/GPU upload layer. No numerics live here.

use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use std::collections::HashMap;

/// A single raw HFQ tensor: metadata plus a copy of its encoded bytes.
#[derive(Clone)]
pub struct RawTensor {
    pub info: HfqTensorInfo,
    pub data: Vec<u8>,
}

/// Index of every Inkling-2 tensor in the trunk HFQ, keyed by verbatim HFQ name.
pub struct Inkling2Weights {
    pub tensors: HashMap<String, RawTensor>,
}

impl Inkling2Weights {
    /// Load and index all tensors, validating the expected physical-layer and
    /// per-expert structure against `cfg`.
    pub fn load(hfq: &HfqFile, cfg: &crate::config::Inkling2Config) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        for info in hfq.tensor_infos() {
            let (i, data) = hfq
                .tensor_data_vec(&info.name)
                .ok_or_else(|| format!("inkling2: cannot read tensor {}", info.name))?;
            tensors.insert(
                info.name.clone(),
                RawTensor {
                    info: i.clone(),
                    data: data.to_vec(),
                },
            );
        }

        // Structural validation: every physical layer must carry its router,
        // norms and the full 128-expert set. Fail closed on a partial artifact.
        for il in 0..cfg.n_phys {
            let p = format!("model.llm.layers.{il}.");
            // Every layer has router, MLP norm and attention output norm;
            // SWA layers use `attn_norm`, full layers use three input norms.
            for suffix in ["mlp.gate.weight", "mlp_norm.weight"] {
                let name = format!("{p}{suffix}");
                if !tensors.contains_key(&name) {
                    return Err(format!("inkling2: missing {name}"));
                }
            }
            let attn_norm = if cfg.swa_layers[il] {
                "attn_norm.weight"
            } else {
                "q_input_norm.weight"
            };
            let name = format!("{p}{attn_norm}");
            if !tensors.contains_key(&name) {
                return Err(format!("inkling2: missing {name}"));
            }
            for e in 0..cfg.n_experts {
                for leaf in ["gate_up_proj.weight", "down_proj.weight"] {
                    let name = format!("{p}mlp.experts.{e}.{leaf}");
                    let t = tensors
                        .get(&name)
                        .ok_or_else(|| format!("inkling2: missing {name}"))?;
                    if t.info.shape.len() != 2 {
                        return Err(format!("inkling2: {name} not 2-D: {:?}", t.info.shape));
                    }
                }
            }
        }
        for name in ["model.llm.embed.weight", "model.llm.norm.weight"] {
            if !tensors.contains_key(name) {
                return Err(format!("inkling2: missing {name}"));
            }
        }

        Ok(Self { tensors })
    }

    pub fn get(&self, name: &str) -> Option<&RawTensor> {
        self.tensors.get(name)
    }

    /// Decode a named weight to true (un-rotated) f32 values. Uses hipfire's
    /// public HFQ decoder, which inverse-FWHTs the MQ4G256/G256V2 formats so
    /// the result is the original weight matrix (row-major, `shape`).
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let t = self
            .get(name)
            .ok_or_else(|| format!("inkling2: tensor not found: {name}"))?;
        let shape: Vec<usize> = t.info.shape.iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let vals = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            t.info.quant_type,
            &t.data,
            n,
        );
        if vals.len() != n {
            return Err(format!(
                "inkling2: decoded {name} to {} values, expected {n}",
                vals.len()
            ));
        }
        Ok((shape, vals))
    }

    /// Expected expert tensor count: n_phys × n_experts × 2 (gate_up + down).
    pub fn expected_expert_tensors(cfg: &crate::config::Inkling2Config) -> usize {
        cfg.n_phys * cfg.n_experts * 2
    }

    pub fn expert_tensor_count(&self) -> usize {
        self.tensors
            .keys()
            .filter(|n| n.contains(".mlp.experts.") && n.ends_with(".weight"))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_expert_count_for_mini() {
        let cfg = crate::config::LoopConfig::mini();
        // Use the typed config shape indirectly: 24 phys × 128 experts × 2.
        assert_eq!(
            24 * 128 * 2,
            Inkling2Weights::expected_expert_tensors(&tiny_cfg())
        );
        let _ = cfg;
    }

    fn tiny_cfg() -> crate::config::Inkling2Config {
        // Reuse the metadata parser test's config by parsing the same JSON.
        crate::config::Inkling2Config::from_metadata_json(TEST_META).unwrap()
    }

    pub(crate) fn test_meta() -> &'static str {
        TEST_META
    }

    const TEST_META: &str = r#"{"config":{"eos_token_id":200006,"text_config":{
        "num_hidden_layers":24,"hidden_size":2304,"vocab_size":201024,
        "unpadded_vocab_size":200058,"num_attention_heads":24,"num_key_value_heads":8,
        "head_dim":128,"sliding_window_size":1024,"d_rel":16,"rel_extent":1024,
        "qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
        "log_scaling_n_floor":8192,"log_scaling_alpha":0.1,"sconv_kernel_size":4,
        "n_layers_per_attn_res_block":6,"final_logit_softcapping":30.0,
        "n_routed_experts":128,"num_experts_per_tok":8,"intermediate_size":1152,
        "route_scale":8.0,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,
        "local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
        "loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64,"is_varloop":false}
    }}}"#;

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact"]
    fn loads_real_artifact() {
        let hfq = HfqFile::open(std::path::Path::new(
            r"D:\hipfire-models\inkling2-mini-mq4.hfq",
        ))
        .expect("open artifact");
        let cfg = tiny_cfg();
        let w = Inkling2Weights::load(&hfq, &cfg).expect("load weights");
        assert_eq!(
            w.expert_tensor_count(),
            Inkling2Weights::expected_expert_tensors(&cfg)
        );
        assert!(w.get("model.llm.embed.weight").is_some());
    }
}
