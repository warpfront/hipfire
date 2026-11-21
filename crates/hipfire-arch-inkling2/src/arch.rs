// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait impl for Inkling-2-Mini (arch_id = 24).
//! The decode forward lives as free functions in `crate::forward`.

use crate::arch_model::{Inkling2Bundle, Inkling2State};
use crate::config::Inkling2Config;
use crate::weights::Inkling2Weights;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Zero-sized type marker for Inkling-2-Mini.
pub struct Inkling2;

impl Architecture for Inkling2 {
    type Weights = Inkling2Weights;
    type State = Inkling2State;
    type Config = Inkling2Config;

    fn arch_id() -> u32 {
        crate::config::ARCH_ID
    }

    fn name() -> &'static str {
        "inkling2"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        Inkling2Config::from_metadata_json(&hfq.metadata_json)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Inkling2Weights::load(hfq, cfg)
    }

    fn new_state(_gpu: &mut Gpu, _cfg: &Self::Config) -> Result<Self::State, String> {
        Ok(Inkling2State { seq_pos: 0 })
    }
}
