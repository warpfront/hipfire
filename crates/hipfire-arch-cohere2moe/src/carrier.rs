// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::cohere2moe::Cohere2MoeState;
use crate::spec_impl::Cohere2MoeBundle;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Build the Cohere2-MoE GPU bundle from an HFQ source.
///
/// Verbatim relocation of the `Cohere2MoeCarrier::load` model work plus the
/// extracted `crate::load_cohere2moe` HFQ helper. Preserves every early-return
/// and error string byte-for-byte: the pp refusal
/// (`"cohere2moe: pp>1 unsupported via registry"`), the HFQ tokenizer error
/// (`"cohere2moe: tokenizer not found: {e}"`), and the HFQ eos fallback
/// `unwrap_or(255001)`. Safetensors directories are refused.
///
/// `dir_diag`, `resolve_source_meta`, `build_speculator`, `resolve_chat_template`,
/// and `LoadedModel::skeleton` stay in the loader (loader-private / cycle edge).
pub fn load_cohere2moe_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx,
) -> Result<Cohere2MoeBundle, String> {
    if ctx.pp > 1 {
        return Err("cohere2moe: pp>1 unsupported via registry".into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => {
            // HFQ path — mirrors `crate::load_cohere2moe` (config/weights/state/eos)
            // without the loader's `LoadedModel`/`chat_template` tail.
            let config = <crate::arch::Cohere2Moe as Architecture>::config_from_hfq(&hfq)?;
            let weights = <crate::arch::Cohere2Moe as Architecture>::load_weights(
                &mut hfq, &config, ctx.gpu,
            )?;
            let state = Cohere2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| format!("cohere2moe: tokenizer not found: {e}"))?;
            // Preserve HFQ-specific eos fallback: 255001.
            let eos_tok: u32 = {
                let try_one = |s: &str| -> Option<u32> {
                    let ids = tokenizer.encode(s);
                    if ids.len() == 1 {
                        Some(ids[0])
                    } else {
                        None
                    }
                };
                try_one("<|END_OF_TURN_TOKEN|>")
                    .or_else(|| try_one("</s>"))
                    .or_else(|| try_one("<|endoftext|>"))
                    .unwrap_or(255001)
            };
            Ok(Cohere2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
        ModelSource::Dir(_) => {
            Err("cohere2moe: safetensors directory unsupported; load an HFQ build".into())
        }
    }
}
