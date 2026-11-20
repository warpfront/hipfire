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
}
