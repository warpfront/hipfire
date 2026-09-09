// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Arch-crate bundle for Spark-X2.5: the loaded (config, weights, state) triple
//! plus its ArchModel view. Mirrors MapleBundle / Gemma4Bundle.

use crate::config::Spark25Config;
use crate::spark25::{Spark25State, Spark25Weights};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use rdna_compute::Gpu;

pub struct Spark25Bundle {
    pub config: Spark25Config,
    pub weights: Spark25Weights,
    pub state: Spark25State,
    pub eos_tok: u32,
}

impl ArchModel for Spark25Bundle {
    fn dim(&self) -> usize {
        self.config.dim
    }

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "spark2_5"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        // Dual KvCaches (sliding/full); not the single llama::KvCache FlashCASK expects.
        None
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let Spark25Bundle {
            config: _,
            weights,
            state,
            eos_tok: _,
        } = *self;
        state.free_gpu(gpu);
        weights.free_gpu(gpu);
    }
}

/// Config-declared eos_token_id default for the published 4B checkpoint.
pub const SPARK25_EOS_FALLBACK: u32 = 1;

/// Resolve EOS from tokenizer single-token encodings, else config fallback.
pub fn resolve_eos(tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> u32 {
    for c in [
        "<eos>",
        "<|im_end|>",
        "<end_of_turn>",
        "</s>",
        "<|endoftext|>",
    ] {
        let enc = tokenizer.encode(c);
        if enc.len() == 1 {
            return enc[0];
        }
    }
    SPARK25_EOS_FALLBACK
}

/// Load a Spark bundle from an already-open HFQ file.
pub fn load_spark25_from_hfq(
    hfq: &mut HfqFile,
    gpu: &mut Gpu,
    max_seq: usize,
) -> Result<Spark25Bundle, String> {
    let config = Spark25Config::from_hfq(hfq)?;
    let weights = Spark25Weights::load(hfq, &config, gpu)?;
    let state = Spark25State::new(&config, gpu, max_seq)?;
    // Prefer HF config eos_token_id; fall back to tokenizer probe / constant.
    let eos_tok = if config.eos_token != 0 {
        config.eos_token
    } else if let Ok(tk) =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
    {
        resolve_eos(&tk)
    } else {
        SPARK25_EOS_FALLBACK
    };
    Ok(Spark25Bundle {
        config,
        weights,
        state,
        eos_tok,
    })
}

/// Build the Spark GPU bundle from a loader `ModelSource`.
pub fn load_spark25_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Spark25Bundle, String> {
    if ctx.pp > 1 {
        return Err("spark25: pp>1 unsupported".into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => load_spark25_from_hfq(&mut hfq, ctx.gpu, ctx.max_seq),
        ModelSource::Dir(_) => Err(
            "spark25: safetensors-directory loading is unsupported — convert first with \
             `hipfire-quantize --format mq4v2 --input <dir> --output <model.hfq>`"
                .into(),
        ),
    }
}
