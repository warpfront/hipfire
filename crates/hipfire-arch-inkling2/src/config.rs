// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Loop geometry and fail-closed validation, ported from llama.cpp's
//! `llama_model_inkling2_loop::load_arch_hparams` and the merged converter's
//! `Inkling2LoopModel.__init__` (merge `29b52d398`).

/// Provisional primary arch id; cheap to change before any artifact ships.
pub const ARCH_ID: u32 = 24;
/// Provisional MTP draft sidecar id.
pub const MTP_SIDECAR_ARCH_ID: u32 = 25;
/// Upstream `LLAMA_MAX_LAYERS` (llama-hparams.h at the merge commit).
pub const LLAMA_MAX_LAYERS: usize = 512;

/// Verified Mini geometry from the live `config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopConfig {
    pub n_phys: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub is_varloop: bool,
    pub n_layers_per_attn_res_block: usize,
}

impl LoopConfig {
    /// `thinkingmachines/Inkling-2-Mini`: 24 physical layers, core [6,18)
    /// visited 3 times, gate rank 64, fixed loop, 6-layer residual blocks.
    pub fn mini() -> Self {
        Self {
            n_phys: 24,
            core_start: 6,
            core_end: 18,
            num_passes: 3,
            gate_rank: 64,
            is_varloop: false,
            n_layers_per_attn_res_block: 6,
        }
    }

    /// Number of cache slots / per-visit stream states.
    pub fn n_slot(&self) -> usize {
        self.n_phys + (self.num_passes - 1) * (self.core_end - self.core_start)
    }

    /// Upstream's structural sharing: visits at or past `n_phys` reuse the
    /// physical layer's weights but keep their own KV/conv state.
    pub fn layer_phys(&self, il: usize) -> usize {
        if il < self.n_phys {
            il
        } else {
            self.core_start + (il - self.n_phys) % (self.core_end - self.core_start)
        }
    }

    /// The visit order of a decode step: all physical layers once, then the
    /// core repeated `num_passes - 1` more times.
    pub fn visit_schedule(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.n_phys).collect();
        for _ in 1..self.num_passes {
            v.extend(self.core_start..self.core_end);
        }
        v
    }

    /// Mirror upstream's validation exactly — no stricter and no looser.
    /// A fixed loop needs at least 2 passes, a varying loop at least 1.
    pub fn validate(&self) -> Result<(), String> {
        let block = self.n_layers_per_attn_res_block;
        let min_passes = if self.is_varloop { 1 } else { 2 };
        if self.num_passes < min_passes {
            return Err(format!(
                "loop needs at least {min_passes} pass(es), got {}",
                self.num_passes
            ));
        }
        if self.gate_rank == 0 {
            return Err("loop gate rank must be >= 1".to_string());
        }
        if block == 0 || self.core_start == 0 || self.core_start >= self.core_end {
            return Err(format!(
                "invalid loop core [{}, {}) with block size {block}",
                self.core_start, self.core_end
            ));
        }
        if self.core_end > self.n_phys {
            return Err(format!(
                "loop end {} exceeds {} physical layers",
                self.core_end, self.n_phys
            ));
        }
        if self.core_start % block != 0 || self.core_end % block != 0 {
            return Err(format!(
                "loop core [{}, {}) is not aligned to block size {block}",
                self.core_start, self.core_end
            ));
        }
        let n_slot = self.n_slot();
        if n_slot > LLAMA_MAX_LAYERS {
            return Err(format!(
                "loop expands to {n_slot} slots, exceeds LLAMA_MAX_LAYERS={LLAMA_MAX_LAYERS}"
            ));
        }
        Ok(())
    }
}

/// Typed model geometry parsed from the HFQ `metadata_json` that the quantizer
/// embeds verbatim from the source `config.json`.
#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub n_phys: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub n_rel: usize,
    pub n_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: usize,
    pub log_alpha: f32,
    pub conv_kernel: usize,
    pub attn_res_block: usize,
    pub softcap: f32,
    pub n_experts: usize,
    pub n_expert_used: usize,
    pub inter: usize,
    pub route_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub rms_norm_eps: f32,
    pub eos_token: u32,
    pub swa_layers: Vec<bool>,
    pub loop_cfg: LoopConfig,
}

impl Inkling2Config {
    /// Parse the `metadata_json` string stored in the HFQ header.
    pub fn from_metadata_json(json: &str) -> Result<Self, String> {
        let meta: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("bad HFQ metadata JSON: {e}"))?;
        let cfg = &meta["config"];
        let t = &cfg["text_config"];
        let loop_v = &t["loop_config"];
        let num = |v: &serde_json::Value, key: &str| -> Result<usize, String> {
            v[key]
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("missing or non-integer {key}"))
        };
        let fnum = |v: &serde_json::Value, key: &str| -> Result<f32, String> {
            v[key]
                .as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| format!("missing or non-numeric {key}"))
        };
        let n_phys = num(t, "num_hidden_layers")?;
        let n_layers_per_attn_res_block = num(t, "n_layers_per_attn_res_block")?;
        let local_ids: Vec<usize> = serde_json::from_value(t["local_layer_ids"].clone())
            .map_err(|e| format!("bad local_layer_ids: {e}"))?;
        let mut swa_layers = vec![false; n_phys];
        for id in local_ids {
            if id >= n_phys {
                return Err(format!("local layer id {id} >= {n_phys}"));
            }
            swa_layers[id] = true;
        }
        let loop_cfg = LoopConfig {
            n_phys,
            core_start: num(loop_v, "core_start")?,
            core_end: num(loop_v, "core_end")?,
            num_passes: num(loop_v, "num_passes")?,
            gate_rank: num(loop_v, "gate_rank")?,
            is_varloop: loop_v["is_varloop"].as_bool().ok_or("missing is_varloop")?,
            n_layers_per_attn_res_block,
        };
        loop_cfg.validate()?;
        Ok(Self {
            n_phys,
            n_embd: num(t, "hidden_size")?,
            vocab_size: num(t, "vocab_size")?,
            unpadded_vocab_size: num(t, "unpadded_vocab_size")?,
            n_head: num(t, "num_attention_heads")?,
            n_head_kv: num(t, "num_key_value_heads")?,
            head_dim: num(t, "head_dim")?,
            sliding_window: num(t, "sliding_window_size")?,
            n_rel: num(t, "d_rel")?,
            n_extent: num(t, "rel_extent")?,
            qk_clip: fnum(t, "qk_l2_clip_threshold")?,
            rel_clip: fnum(t, "rel_l2_clip_threshold")?,
            log_floor: num(t, "log_scaling_n_floor")?,
            log_alpha: fnum(t, "log_scaling_alpha")?,
            conv_kernel: num(t, "sconv_kernel_size")?,
            attn_res_block: n_layers_per_attn_res_block,
            softcap: fnum(t, "final_logit_softcapping")?,
            n_experts: num(t, "n_routed_experts")?,
            n_expert_used: num(t, "num_experts_per_tok")?,
            inter: num(t, "intermediate_size")?,
            route_scale: fnum(t, "route_scale")?,
            situ_beta: fnum(t, "mlp_gate_softcap")?,
            situ_linear_beta: fnum(t, "mlp_up_softcap")?,
            rms_norm_eps: fnum(t, "rms_norm_eps")?,
            eos_token: cfg["eos_token_id"].as_u64().ok_or("missing eos_token_id")? as u32,
            swa_layers,
            loop_cfg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_schedule_is_48_visits_with_correct_counts() {
        let cfg = LoopConfig::mini();
        cfg.validate().unwrap();
        let sched = cfg.visit_schedule();
        assert_eq!(sched.len(), 48);
        assert_eq!(cfg.n_slot(), 48);
        for il in 0..24 {
            let visits = sched.iter().filter(|&&v| v == il).count();
            let want = if (6..18).contains(&il) { 3 } else { 1 };
            assert_eq!(visits, want, "physical layer {il}");
        }
        assert_eq!(&sched[..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(&sched[24..26], &[6, 7]);
        assert_eq!(&sched[46..48], &[16, 17]);
    }

    #[test]
    fn layer_phys_shares_core_weights() {
        let cfg = LoopConfig::mini();
        assert_eq!(cfg.layer_phys(0), 0);
        assert_eq!(cfg.layer_phys(23), 23);
        assert_eq!(cfg.layer_phys(24), 6);
        assert_eq!(cfg.layer_phys(35), 17);
        assert_eq!(cfg.layer_phys(36), 6);
        assert_eq!(cfg.layer_phys(47), 17);
    }

    #[test]
    fn refuses_upstream_invalid_configs() {
        let base = LoopConfig::mini();
        let mut c = base.clone();
        c.num_passes = 1;
        assert!(c.validate().is_err(), "fixed loop needs >= 2 passes");
        c = base.clone();
        c.gate_rank = 0;
        assert!(c.validate().is_err());
        c = base.clone();
        c.core_start = 0;
        assert!(c.validate().is_err());
        c = base.clone();
        c.core_start = 18;
        assert!(c.validate().is_err());
        c = base.clone();
        c.core_end = 25;
        assert!(c.validate().is_err());
        c = base.clone();
        c.core_start = 7;
        assert!(c.validate().is_err(), "core start not block aligned");
        c = base.clone();
        c.num_passes = 100;
        assert!(c.validate().is_err(), "slot count exceeds LLAMA_MAX_LAYERS");
    }

    #[test]
    fn varying_loop_allows_one_pass() {
        let mut c = LoopConfig::mini();
        c.is_varloop = true;
        c.num_passes = 1;
        c.validate().unwrap();
        assert_eq!(c.n_slot(), 24);
    }

    #[test]
    fn arch_ids_are_the_provisional_guesses() {
        assert_eq!(ARCH_ID, 24);
        assert_eq!(MTP_SIDECAR_ARCH_ID, 25);
    }

    #[test]
    fn parses_embedded_metadata_shape() {
        let json = r#"{
          "config": {
            "eos_token_id": 200006,
            "text_config": {
              "num_hidden_layers": 24, "hidden_size": 2304,
              "vocab_size": 201024, "unpadded_vocab_size": 200058,
              "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
              "sliding_window_size": 1024, "d_rel": 16, "rel_extent": 1024,
              "qk_l2_clip_threshold": 22.627416997969522,
              "rel_l2_clip_threshold": 8.0, "log_scaling_n_floor": 8192,
              "log_scaling_alpha": 0.1, "sconv_kernel_size": 4,
              "n_layers_per_attn_res_block": 6, "final_logit_softcapping": 30.0,
              "n_routed_experts": 128, "num_experts_per_tok": 8,
              "intermediate_size": 1152, "route_scale": 8.0,
              "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,"rms_norm_eps": 1e-6,
              "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
              "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3,
                              "gate_rank": 64, "is_varloop": false}
            }
          }
        }"#;
        let cfg = Inkling2Config::from_metadata_json(json).unwrap();
        assert_eq!(cfg.n_phys, 24);
        assert_eq!(cfg.n_embd, 2304);
        assert_eq!(cfg.eos_token, 200006);
        assert_eq!(cfg.swa_layers.iter().filter(|x| **x).count(), 20);
        assert!(!cfg.swa_layers[5]);
        assert!(cfg.swa_layers[6]);
        assert_eq!(cfg.loop_cfg.n_slot(), 48);
        assert_eq!(cfg.n_experts, 128);
    }
}
