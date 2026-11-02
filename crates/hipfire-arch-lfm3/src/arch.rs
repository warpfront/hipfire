// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for LFM3 (arch_id = 24).
//!
//! Thin marker + delegation, mirroring `hipfire-arch-lfm2moe`'s `arch.rs`.
//! The forward pass is NOT on the trait — it lives as free functions in
//! `crate::forward` (hot-path static dispatch), called directly by the
//! daemon's `arch_id == 24` generate branch.

use crate::config::Lfm3Config;
use crate::lfm3::{Lfm3State, Lfm3Weights};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for LFM3. Trait dispatch uses the type.
pub struct Lfm3;

impl Architecture for Lfm3 {
    type Weights = Lfm3Weights;
    type State = Lfm3State;
    type Config = Lfm3Config;

    /// Canonical architecture ID for LFM3.
    fn arch_id() -> u32 {
        24
    }

    fn name() -> &'static str {
        "lfm3"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Lfm3Config::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Lfm3Weights::load(hfq, cfg, gpu)
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        Lfm3State::new(gpu, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm3_arch_id_and_name() {
        assert_eq!(Lfm3::arch_id(), 24);
        assert_eq!(Lfm3::name(), "lfm3");
    }
}
