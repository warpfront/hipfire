// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_inkling2_bundle` entry point for the loader's `Carrier` impl.
//! HFQ-only skeleton: parse config, index/validate weights, resolve tokenizer,
//! take eos from metadata. No GPU upload, no state alloc.

use crate::arch::Inkling2;
use crate::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Admitted context window. Matches the loader's omitted-`max_seq` default.
pub const INKLING2_MAX_CONTEXT: usize = 32768;

/// Build the Inkling-2-Mini skeleton bundle from an HFQ source.
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
            Ok(Inkling2Bundle {
                eos_tok: config.eos_token,
                config,
                weights,
                state: Inkling2State { seq_pos: 0 },
            })
        }
        ModelSource::Dir(_) => Err(
            "inkling2: safetensors directory sources are not supported; load a single HFQ file"
                .into(),
        ),
    }
}
