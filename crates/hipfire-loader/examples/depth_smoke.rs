// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 4a standalone validation: load the LFM3-Omni HFQ, run the text
//! backbone on a prompt, feed the LM's `last_hidden_state` (=
//! `embedding_norm(state.h)` = `state.final_norm_buf`) into the depth decoder,
//! and MaskGIT-sample one audio frame's 12 codebooks.
//!
//! Validates that the depth decoder forward + MaskGIT produces valid,
//! non-degenerate codes from a real LM hidden state (codes in range, finite
//! logits, not all-identical / all-eos). A broken forward (wrong RoPE, wrong
//! attention, wrong residual) yields NaN/garbage logits or degenerate codes.
//!
//! Usage: `depth_smoke [hfq_path] [prompt]`

use hipfire_arch_lfm3::forward::decode_step;
use hipfire_arch_lfm3_omni::{sample_audio_frame, DepthRng};
use hipfire_loader::{load_model, ModelState};
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use rdna_compute::Gpu;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Omni-2B.q8f16.hfq".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Hello, how are you today?".to_string());

    eprintln!("depth_smoke: initializing GPU...");
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("  GPU: {}", gpu.arch);

    eprintln!("depth_smoke: loading {path} (max_seq=2048)...");
    let mut model = load_model(
        &path,
        2048,
        None,
        None,
        None,
        None,
        &CaskConfig::default(),
        1,
        SpecLoadCfg::default(),
        &mut gpu,
    )
    .expect("load_model");
    eprintln!("  arch_id={}  depth_decoder={}", model.arch_id, model.lfm3_depth_weights.is_some());

    // Tokenize the prompt. The omni model was trained on ChatML + BOS
    // (<|startoftext|> = 124894); a bare prompt produces repetition collapse.
    let chatml = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>");
    let mut tokens = model
        .tokenizer
        .as_ref()
        .expect("tokenizer")
        .encode(&chatml);
    tokens.insert(0, 124_894); // BOS <|startoftext|>
    eprintln!("  prompt (ChatML+BOS): {chatml:?}");
    eprintln!("  tokens ({}): {:?}…", tokens.len(), &tokens[..tokens.len().min(12)]);

    // Phase 1: run the text backbone (sequential prefill = decode_step per
    // token). After the last token, state.final_norm_buf holds the LM's
    // last_hidden_state — the depth decoder's input.
    eprintln!("depth_smoke: prefilling text backbone ({} tokens)...", tokens.len());
    let hidden: Vec<f32> = {
        let bundle = match model.state.as_mut() {
            Some(ModelState::Lfm3(b)) => b,
            _ => panic!("loaded model is not LFM3"),
        };
        let mut last_logits: Option<Vec<f32>> = None;
        for (i, &tok_id) in tokens.iter().enumerate() {
            last_logits = Some(
                decode_step(&bundle.config, &bundle.weights, &mut bundle.state, &mut gpu, tok_id, i as u32)
                    .unwrap_or_else(|e| panic!("decode_step @ {i}: {e}")),
            );
        }
        // Greedy text decode (coherence check — rules out a Q8 text-backbone
        // regression as the cause of any downstream audio-path issue).
        let mut pos = tokens.len() as u32 - 1;
        let mut gen: Vec<u32> = Vec::with_capacity(24);
        for _ in 0..24 {
            let next = last_logits
                .as_ref()
                .unwrap()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0 as u32;
            gen.push(next);
            pos += 1;
            last_logits = Some(
                decode_step(&bundle.config, &bundle.weights, &mut bundle.state, &mut gpu, next, pos)
                    .unwrap_or_else(|e| panic!("greedy decode_step: {e}")),
            );
        }
        let gen_text = model.tokenizer.as_ref().unwrap().decode(&gen);
        eprintln!("  greedy text (24 tok): {gen_text:?}");
        gpu.download_f32(&bundle.state.final_norm_buf)
            .expect("download final_norm_buf")
    };
    eprintln!(
        "  hidden[0..4] = {:?}  (len={}, finite={})",
        &hidden[..4.min(hidden.len())],
        hidden.len(),
        hidden.iter().all(|v| v.is_finite())
    );

    // Phase 2: MaskGIT-sample one audio frame from the hidden state.
    let depth_cfg = model
        .lfm3_depth_config
        .as_ref()
        .expect("depth config missing");
    let depth_weights = model
        .lfm3_depth_weights
        .as_ref()
        .expect("depth weights missing");

    eprintln!(
        "depth_smoke: sampling audio frame (maskgit {:?}, temp={}, rep={})...",
        depth_cfg.maskgit_schedule, depth_cfg.audio_temperature, depth_cfg.audio_repetition_penalty
    );
    let mut rng = DepthRng::new(42);
    let codes = sample_audio_frame(
        depth_cfg,
        depth_weights,
        &mut gpu,
        &hidden,
        &Vec::<Vec<u32>>::new(),
        &mut rng,
    )
    .expect("sample_audio_frame");

    // ── Verify ──
    let cb = depth_cfg.codebook_size as u32;
    let eos = depth_cfg.audio_eos_code;
    let all_in_range = codes.iter().all(|&c| c < cb);
    let all_same = codes.windows(2).all(|w| w[0] == w[1]);
    let all_eos = codes.iter().all(|&c| c == eos);
    let n_distinct = {
        let mut s = codes.clone();
        s.sort_unstable();
        s.dedup();
        s.len()
    };

    println!("codes ({} codebooks): {codes:?}", codes.len());
    println!(
        "codebook_size={cb}  eos_code={eos}  all_in_range={all_in_range}  all_same={all_same}  all_eos={all_eos}  n_distinct={n_distinct}"
    );

    // Also exercise get_audio_out_embeds (the feedback path for the omni loop).
    let embed = hipfire_arch_lfm3_omni::get_audio_out_embeds(depth_cfg, depth_weights, &codes);
    println!(
        "audio_out_embed: len={} finite={} [0..4]={:?}",
        embed.len(),
        embed.iter().all(|v| v.is_finite()),
        &embed[..4.min(embed.len())]
    );

    let pass = all_in_range && !all_same && !all_eos && n_distinct >= 2
        && hidden.iter().all(|v| v.is_finite())
        && embed.iter().all(|v| v.is_finite());
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
