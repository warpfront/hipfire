// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_inkling2_bundle` entry point for the loader's `Carrier` impl.
//! HFQ-only: parse config, index/validate weights, upload the GPU bundle
//! (brief 01), resolve tokenizer, take eos from metadata. No forward, no
//! generation route — generate still fails closed downstream.

use crate::arch::Inkling2;
use crate::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Admitted context window. Matches the loader's omitted-`max_seq` default.
pub const INKLING2_MAX_CONTEXT: usize = 32768;

/// Build the Inkling-2-Mini bundle from an HFQ source.
pub fn load_inkling2_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Inkling2Bundle, String> {
    if ctx.pp > 1 {
        return Err(match &src {
            ModelSource::Hfq(_) => "inkling2: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "inkling2: safetensors + pp>1 unsupported",
        }
        .into());
    }
    match src {
        ModelSource::Hfq(mut hfq_file) => {
            if ctx.max_seq == 0 || ctx.max_seq > INKLING2_MAX_CONTEXT {
                return Err(format!(
                    "inkling2: max_seq {} is outside the admitted context range 1..={INKLING2_MAX_CONTEXT}",
                    ctx.max_seq
                ));
            }
            let config = <Inkling2 as Architecture>::config_from_hfq(&hfq_file)?;
            let weights =
                <Inkling2 as Architecture>::load_weights(&mut hfq_file, &config, ctx.gpu)?;
            // Fail-closed: artifact must carry a tokenizer.
            let _tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_file.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            // GPU bundle (brief 01): VRAM weights + per-visit state. The host
            // `weights` index is retained: `weights.rs` is owned by brief
            // 04/06 and has no drain API yet, so the ~13.9 GB host copy stays
            // mapped for this brief (no host-OOM on a desktop host, but a
            // known deviation from "drop after upload" — a `drain_after_upload`
            // on `Inkling2Weights` closes it).
            let free0 = ctx.gpu.device_mem_info().ok().map(|(f, _)| f);
            let (gpu_weights, gpu_state) =
                crate::gpu::upload(&weights, &config, ctx.gpu, ctx.max_seq)?;
            let summary = gpu_weights.summary.text();
            if let Some(f0) = free0 {
                if let Ok((f1, _)) = ctx.gpu.device_mem_info() {
                    eprintln!(
                        "  inkling2 measured VRAM delta: {} MiB (accounted {:.3} GiB)",
                        (f0 as i64 - f1 as i64) / (1 << 20),
                        gpu_weights.summary.total_bytes() as f64 / (1u64 << 30) as f64,
                    );
                }
            }
            eprintln!("  {summary}");
            if let Some(line) = gpu_weights
                .summary
                .prefill_plan
                .and_then(|p| p.fallback_line(&config))
            {
                eprintln!("  {line}");
            }
            // Typed XTS layout only: one capability line (legacy prints nothing new).
            if let Some(routes) = gpu_weights.xts_route_capability() {
                eprintln!("  {}", routes.load_line());
            }
            Ok(Inkling2Bundle {
                eos_tok: config.eos_token,
                config,
                weights,
                state: Inkling2State { seq_pos: 0 },
                gpu_weights: Some(gpu_weights),
                gpu_state: Some(gpu_state),
            })
        }
        ModelSource::Dir(_) => Err(
            "inkling2: safetensors directory sources are not supported; load a single HFQ file"
                .into(),
        ),
    }
}
