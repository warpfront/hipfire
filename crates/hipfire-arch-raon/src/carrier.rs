// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon bundle loader (HFQ). Owner: slice A.
//!
//! Follows `hipfire-arch-lfm2moe::carrier`: the loader-owned `Carrier`
//! (arch 19 registration) calls [`load_raon_bundle`] and wraps the bundle in
//! `LoadedModel`. This crate does not depend on the loader.

use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

use crate::bundle::{RaonBundle, PREFILL_SCRATCH_ROWS};
use crate::config::RaonConfig;
use crate::state::{RaonPrefillScratch, RaonState};
use crate::weights::load_raon_weights;

/// Build the Raon GPU bundle from an HFQ source.
///
/// Refused up front (before any allocation): pipeline/expert parallelism,
/// a draft/speculation sidecar, a safetensors directory (the official MXFP8
/// checkpoint is quantized offline by `hipfire quantize`, never half-loaded),
/// a non-arch-19 container, and a context length outside `1..=max_position`.
/// Tensor parallelism never reaches this carrier (it has its own route).
pub fn load_raon_bundle(src: ModelSource, ctx: &mut LoadCtx<'_>) -> Result<RaonBundle, String> {
    if ctx.pp > 1 {
        return Err(format!(
            "raon1_mini: pipeline parallelism (topology.pp = {}) is unsupported",
            ctx.pp
        ));
    }
    let ep = hipfire_config::topology::ep()?;
    if ep > 1 {
        return Err(format!(
            "raon1_mini: expert parallelism (topology.ep = {ep}) is unsupported"
        ));
    }
    if ctx.draft_path.is_some() {
        return Err("raon1_mini: speculative draft models are unsupported".into());
    }
    let hfq = match src {
        ModelSource::Hfq(hfq) => hfq,
        ModelSource::Dir(_) => {
            return Err(
                "raon1_mini: a safetensors directory cannot be served directly (the official \
                 checkpoint stores routed experts as MXFP8 and needs the symmetric MQ4V2 conversion); \
                 run `hipfire quantize` on it (Raon symmetric qt44 recipe) and load the resulting .hfq"
                    .into(),
            )
        }
    };
    if hfq.arch_id != crate::ARCH_ID {
        return Err(format!(
            "raon1_mini: HFQ arch_id {} is not {}",
            hfq.arch_id,
            crate::ARCH_ID
        ));
    }
    let config = RaonConfig::from_hfq(&hfq)?;
    if ctx.max_seq == 0 || ctx.max_seq > config.max_position {
        return Err(format!(
            "raon1_mini: max_seq {} outside 1..={}",
            ctx.max_seq, config.max_position
        ));
    }
    let weights = load_raon_weights(&hfq, ctx.gpu, &config)?;
    let dims = config.state_dims();
    let state = match RaonState::new(ctx.gpu, &dims, ctx.max_seq) {
        Ok(s) => s,
        Err(e) => {
            weights.free_gpu(ctx.gpu);
            return Err(format!("raon1_mini: RaonState::new failed: {e}"));
        }
    };
    let rows = PREFILL_SCRATCH_ROWS.min(ctx.max_seq);
    let scratch = match RaonPrefillScratch::new(ctx.gpu, &dims, rows) {
        Ok(s) => s,
        Err(e) => {
            if let Err(fe) = state.free_gpu(ctx.gpu) {
                eprintln!("raon1_mini: state free after scratch failure: {fe}");
            }
            weights.free_gpu(ctx.gpu);
            return Err(format!("raon1_mini: RaonPrefillScratch::new failed: {e}"));
        }
    };
    RaonBundle::new(ctx.gpu, config, weights, state, scratch)
}
