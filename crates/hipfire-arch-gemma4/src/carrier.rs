// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 carrier bundle loader — HFQ path.
//!
//! The loader retains `LoadedModel` assembly, `SourceMeta`/`resolve_source_meta`,
//! chat-template and tokenizer handling, the session cache attach and the
//! EAGLE speculator (`crate::mtp::Gemma4Drafter`). This module owns the GPU bundle
//! construction: config admission, weights, state and both KV caches, for
//! every Gemma 4 variant (dense, E-series, MoE).

use crate::bundle::Gemma4Bundle;
use crate::config::Gemma4Config;
use crate::gemma4::{FullKvTier, Gemma4State, Gemma4Weights};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

fn gemma4_validate_drafter_route(is_e_series: bool, has_drafter: bool) -> Result<(), String> {
    if is_e_series && has_drafter {
        return Err(
            "gemma4: E2B/E4B EAGLE spec-decode is not yet supported; load the E-series target without params.drafter"
                .into(),
        );
    }
    Ok(())
}

/// Refuse MoE checkpoints whose expert formats have no indexed device kernel
/// pair (the routed-expert step is capture-safe and never falls back to a
/// host loop).
fn unsupported_expert_formats(weights: &Gemma4Weights) -> Option<String> {
    weights.layers.iter().find_map(|layer| {
        let moe = layer.moe()?;
        let (gate_up, down) = (moe.gate_up_dtype, moe.down_dtype);
        (!hipfire_dispatch::pipeline::sandwich::RoutedExperts::supports(gate_up, down)).then(|| {
            format!(
                "gemma4 MoE: expert formats gate_up={gate_up:?} down={down:?} have no indexed \
                 kernel; requantize experts to MQ4G256(V2)/MQ6G256/HFQ4G256/HFQ6G256/Q8_0 gate_up with \
                 Q8_0/HFQ4G128 down"
            )
        })
    })
}

/// Resolve the full-attention KV tier from the raw `kv_cache` value through
/// [`GEMMA4_FULL_POLICY`](hipfire_runtime::kv_mode::GEMMA4_FULL_POLICY):
/// `auto`/`q8` → Q8, `legacy-asym3` → the previous Givens asym3 tier.
/// Unsupported names fall back to Q8 with the returned warning; fp8/bf16 fail
/// closed (no Gemma4 attend site admits a native tier).
fn resolve_full_kv_tier(mode_raw: &str) -> Result<(FullKvTier, Option<&'static str>), String> {
    use hipfire_runtime::kv_mode::{self, KvMode};
    let rr = kv_mode::resolve(mode_raw, &kv_mode::GEMMA4_FULL_POLICY);
    match rr.mode {
        KvMode::Q8 => Ok((FullKvTier::Q8, rr.warning)),
        KvMode::Asym3 => Ok((FullKvTier::LegacyAsym3, rr.warning)),
        other => Err(format!(
            "gemma4: kv_cache={mode_raw} ({other:?}) has no full-attention KV tier; use auto, q8 or legacy-asym3"
        )),
    }
}

/// Build the Gemma 4 GPU bundle from an HFQ source.
pub fn load_gemma4_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Gemma4Bundle, String> {
    if ctx.kv_backend != hipfire_runtime::kv_backend::KvBackend::Legacy {
        return Err("gemma4: sliding/full KV owners require legacy backend".into());
    }
    let hfq = match src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(_) => {
            return Err("gemma4: safetensors Dir load not yet wired — use HFQ (quantize with --arch-id 13) or add config_from_source to hipfire-arch-gemma4".into());
        }
    };
    let config = Gemma4Config::from_hfq(&hfq)?;
    let is_e_series = config.hidden_size_per_layer_input != 0 || config.num_kv_shared_layers != 0;
    if is_e_series {
        eprintln!(
            "  gemma4 E-series: {:?} (PLE + shared KV)",
            config.e_series_variant()?
        );
    }
    gemma4_validate_drafter_route(is_e_series, ctx.gemma4_drafter_path.is_some())?;

    // Pure CPU: resolve the full-attention tier before any device upload.
    let mode_raw = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());
    let (tier, warning) = resolve_full_kv_tier(&mode_raw)?;
    if let Some(w) = warning {
        eprintln!(
            "  KV cache: {w} (site {})",
            hipfire_runtime::kv_mode::GEMMA4_FULL_POLICY.site
        );
    }

    let weights = Gemma4Weights::load(&hfq, &config, ctx.gpu)?;
    if let Some(e) = unsupported_expert_formats(&weights) {
        weights.free_gpu(ctx.gpu);
        return Err(e);
    }
    let state = match Gemma4State::new_with_full_tier(ctx.gpu, &config, ctx.max_seq, tier) {
        Ok(state) => state,
        Err(e) => {
            weights.free_gpu(ctx.gpu);
            return Err(format!("gemma4: state: {e}"));
        }
    };
    eprintln!(
        "  gemma4: {} (sliding q8 + full {tier:?} KV; kv_cache={mode_raw})",
        if config.enable_moe_block {
            "moe"
        } else {
            "dense"
        }
    );
    Ok(Gemma4Bundle::new(config, weights, state, ctx.gpu))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_kv_tier_follows_kv_cache() {
        for raw in ["", "auto", "q8"] {
            assert_eq!(
                resolve_full_kv_tier(raw),
                Ok((FullKvTier::Q8, None)),
                "{raw:?}"
            );
        }
        assert_eq!(
            resolve_full_kv_tier("legacy-asym3"),
            Ok((FullKvTier::LegacyAsym3, None))
        );
        // Bare asym3 is the Qwen FWHT alias: no hd512 FWHT tier, so q8 + warning.
        let (tier, warning) = resolve_full_kv_tier("asym3").unwrap();
        assert_eq!(tier, FullKvTier::Q8);
        assert!(warning.is_some());
        // Native presets fail closed instead of silently becoming q8.
        assert!(resolve_full_kv_tier("fp8").is_err());
        assert!(resolve_full_kv_tier("bf16").is_err());
    }
}
