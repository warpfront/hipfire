// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_arch_gemma4::gemma4::FullKvTier;
use hipfire_arch_gemma4::{forward, Gemma4Config, Gemma4State, Gemma4Weights};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::Path;

fn main() {
    let path = Path::new("/local/models/google/gemma-4-12B-it.hfq");
    let mut gpu = Gpu::init().expect("failed to init GPU");
    let hfq = HfqFile::open(path).expect("failed to open HFQ");

    let config = Gemma4Config::from_hfq(&hfq).expect("failed config");
    let weights = Gemma4Weights::load(&hfq, &config, &mut gpu).expect("failed weights");
    // Sliding tier is always Q8; the full tier keeps the asym3 cache this probe used.
    let mut state = Gemma4State::new_with_full_tier(&mut gpu, &config, 2048, FullKvTier::LegacyAsym3)
        .expect("failed state");

    println!("Running Layer 0 forward with BOS (2)...");
    forward::decode_step(&config, &weights, &mut state, &mut gpu, 2, 0).expect("BOS forward failed");

    println!("Running Layer 0 forward with 'Hello' (9259)...");
    forward::decode_step(&config, &weights, &mut state, &mut gpu, 9259, 1)
        .expect("Hello forward failed");
}
