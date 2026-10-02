// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
//! GPU gate for per-slot RNG state across a multi-token decode.
//!
//! `sample_per_slot` hands each sampling slot its own seed. If the state it
//! draws with is not carried forward, every step redraws the same uniform, and
//! a temperature slot silently collapses onto one fixed quantile of its own
//! distribution — indistinguishable from working sampling on a single token,
//! which is why this walks several.
//!
//! Run: `cargo run --release -p rdna-compute --features lab --example sample_slot_rng_advance`

use rdna_compute::sampling::SlotSampleParams;
use rdna_compute::{DType, Gpu, GpuTensor};

const VOCAB: usize = 4096;
const SLOTS: usize = 2;
const STEPS: usize = 24;

/// Flat over the top 64 ids, so which one is drawn is the RNG's answer alone;
/// argmax would pin the same id every step.
fn flat_logits() -> Vec<f32> {
    let mut logits = vec![-60.0f32; VOCAB * SLOTS];
    for slot in 0..SLOTS {
        for id in 0..64usize {
            logits[slot * VOCAB + id * 37 % VOCAB] = 4.0;
        }
    }
    logits
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let logits = gpu.upload_f32(&flat_logits(), &[SLOTS * VOCAB])?;
    let out_tokens = gpu.zeros(&[SLOTS], DType::F32)?;

    let mut params: Vec<SlotSampleParams> = (0..SLOTS)
        .map(|slot| SlotSampleParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 64,
            seed: 1 + slot as u32 * 7919,
            // No penalties: this example pins per-slot RNG advance, not the
            // penalty math. A zero window disables every token penalty, but
            // the sampler still requires one repeat buffer per slot.
            repeat_window: 0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
        })
        .collect();
    let repeat_windows: Vec<GpuTensor> = (0..SLOTS)
        .map(|_| gpu.zeros(&[2048usize], DType::F32))
        .collect::<Result<Vec<_>, _>>()?;

    let mut seeds: Vec<Vec<u32>> = vec![Vec::with_capacity(STEPS); SLOTS];
    let mut drawn: Vec<Vec<u32>> = vec![Vec::with_capacity(STEPS); SLOTS];
    for _ in 0..STEPS {
        for (slot, p) in params.iter().enumerate() {
            seeds[slot].push(p.seed);
        }
        gpu.sample_per_slot(
            &logits,
            &mut params,
            &repeat_windows,
            SLOTS,
            VOCAB,
            &out_tokens,
        )?;
        gpu.hip.device_synchronize()?;
        let mut ids = vec![0i32; SLOTS];
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, SLOTS * 4) };
        gpu.hip.memcpy_dtoh(bytes, &out_tokens.buf)?;
        for (slot, id) in ids.iter().enumerate() {
            drawn[slot].push(*id as u32);
        }
    }

    for slot in 0..SLOTS {
        let states = &seeds[slot];
        let unique: std::collections::HashSet<u32> = states.iter().copied().collect();
        assert_eq!(
            unique.len(),
            STEPS,
            "slot {slot}: RNG state repeated across {STEPS} tokens — the sampler's \
             new state is not being carried forward (states: {states:?})"
        );
        let ids: std::collections::HashSet<u32> = drawn[slot].iter().copied().collect();
        assert!(
            ids.len() > 1,
            "slot {slot}: {STEPS} draws from a flat distribution all returned the \
             same id ({:?}) — a frozen RNG state, not sampling",
            drawn[slot]
        );
        println!(
            "slot {slot}: {} distinct states, {} distinct tokens over {STEPS} steps",
            unique.len(),
            ids.len()
        );
    }
    assert_ne!(
        seeds[0], seeds[1],
        "slots seeded differently must not share an RNG trajectory"
    );

    println!("PASS: per-slot RNG state advances per token and stays per-slot");
    Ok(())
}
