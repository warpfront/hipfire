// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batch-vs-streaming equivalence: run `omni_generate` (batch) and
//! `omni_stream_prefill` + `omni_stream_step` (streaming) with the SAME seed
//! on the same prompt, in one process, and compare the generated token IDs.
//! Any divergence is a state-carry bug in the streaming path.

use hipfire_arch_lfm3_omni::{
    omni_generate, omni_stream_prefill, omni_stream_step, DepthRng, MIXED_START_TOKEN,
};
use hipfire_loader::load_model;
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use rdna_compute::Gpu;

fn build_prompt(model: &hipfire_loader::LoadedModel, prompt: &str) -> Vec<u32> {
    let tok = model.tokenizer.as_ref().unwrap();
    let chatml =
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>");
    let mut t = tok.encode(&chatml);
    t.insert(0, 124_894);
    t.push(MIXED_START_TOKEN);
    t
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Omni-2B.q8f16.hfq".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Count from one to three.".to_string());
    let seed = 42u64;

    let mut gpu = Gpu::init().expect("GPU init");

    // ── Batch run 1 ──
    eprintln!("== batch run 1 ==");
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
    let prompt_tokens = build_prompt(&model, &prompt);
    let depth_cfg = model.lfm3_depth_config().unwrap().clone();
    let (text_ids_b, audio_codes_b, text_n_b) = {
        let bundle = match model.state.as_mut() {
            Some(s) => (s.as_mut() as &mut dyn std::any::Any)
                .downcast_mut::<hipfire_loader::Lfm3Bundle>()
                .expect("not lfm3"),
            None => panic!("not lfm3"),
        };
        let dw = bundle.depth_weights.as_ref().unwrap();
        let mut rng = DepthRng::new(seed);
        let out = omni_generate(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            &depth_cfg,
            dw,
            &mut gpu,
            &prompt_tokens,
            40,
            &mut rng,
        )
        .expect("batch omni_generate");
        (
            out.text_tokens.clone(),
            out.audio_codes.clone(),
            out.text_tokens.len(),
        )
    };
    eprintln!(
        "  batch: {text_n_b} text tokens {text_ids_b:?} + {} audio frames ({} codes)",
        audio_codes_b.len() / 12,
        audio_codes_b.len()
    );

    // ── Batch run 2 (determinism check: fresh state, same seed) ──
    eprintln!("== batch run 2 ==");
    drop(model);
    let mut model2 = load_model(
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
    .expect("load_model (batch2)");
    let (text_ids_b2, audio_codes_b2) = {
        let bundle = match model2.state.as_mut() {
            Some(s) => (s.as_mut() as &mut dyn std::any::Any)
                .downcast_mut::<hipfire_loader::Lfm3Bundle>()
                .expect("not lfm3"),
            None => panic!("not lfm3"),
        };
        let dw = bundle.depth_weights.as_ref().unwrap();
        let mut rng = DepthRng::new(seed);
        let out = omni_generate(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            &depth_cfg,
            dw,
            &mut gpu,
            &prompt_tokens,
            40,
            &mut rng,
        )
        .expect("batch omni_generate run2");
        (out.text_tokens.clone(), out.audio_codes.clone())
    };
    let frames_b2 = audio_codes_b2.len() / 12;
    let mut b12_div = None;
    for i in 0..text_ids_b.len().min(text_ids_b2.len()) {
        if text_ids_b[i] != text_ids_b2[i] {
            b12_div = Some(i);
            break;
        }
    }
    let mut b12_code_div = None;
    for f in 0..(audio_codes_b.len() / 12).min(frames_b2) {
        if audio_codes_b[f * 12..(f + 1) * 12] != audio_codes_b2[f * 12..(f + 1) * 12] {
            b12_code_div = Some(f);
            break;
        }
    }
    match (b12_div, b12_code_div) {
        (None, None)
            if text_ids_b.len() == text_ids_b2.len() && audio_codes_b.len() / 12 == frames_b2 =>
        {
            println!("  batch-vs-batch: IDENTICAL (deterministic)");
        }
        _ => {
            println!(
                "  batch-vs-batch: DIVERGES (text_div={b12_div:?}, code_div={b12_code_div:?}) ⇒ the LM decode is NON-DETERMINISTIC (Q8 KV stochastic state) — streaming-vs-batch divergence is expected",
            );
        }
    }
    drop(model2);

    // ── Fresh model for streaming (fresh Lfm3State) ──
    eprintln!("== streaming ==");
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
    .expect("load_model (streaming)");
    let prompt_tokens_s = build_prompt(&model, &prompt);
    let depth_cfg = model.lfm3_depth_config().unwrap().clone();
    assert_eq!(
        prompt_tokens, prompt_tokens_s,
        "prompt construction differs"
    );

    let mut text_ids_s: Vec<u32> = Vec::new();
    let mut audio_codes_s: Vec<u32> = Vec::new();
    {
        let bundle = match model.state.as_mut() {
            Some(s) => (s.as_mut() as &mut dyn std::any::Any)
                .downcast_mut::<hipfire_loader::Lfm3Bundle>()
                .expect("not lfm3"),
            None => panic!("not lfm3"),
        };
        let dw = bundle.depth_weights.as_ref().unwrap();
        let mut rng = DepthRng::new(seed);
        let mut stream = omni_stream_prefill(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            &mut gpu,
            &prompt_tokens_s,
        )
        .expect("stream prefill");
        eprintln!("  prefill done (position={})", stream.position);
        let mut step = 0usize;
        while !stream.finished && step < 40 {
            let out = omni_stream_step(
                &bundle.config,
                &bundle.weights,
                &mut bundle.state,
                &depth_cfg,
                dw,
                &mut gpu,
                &mut stream,
                12,
                &mut rng,
            )
            .expect("stream step");
            text_ids_s.extend_from_slice(&out.text_tokens);
            audio_codes_s.extend_from_slice(&out.audio_codes);
            eprintln!(
                "  step chunk {step}: +{} text, +{} audio, finished={}",
                out.text_tokens.len(),
                out.n_audio_frames,
                out.finished
            );
            step += 1;
        }
    }
    eprintln!(
        "  streaming: {} text tokens {text_ids_s:?} + {} audio frames ({} codes)",
        text_ids_s.len(),
        audio_codes_s.len() / 12,
        audio_codes_s.len()
    );

    // TEST-1: bound the generated frame/text counts by the step budget
    // (STREAM-1's max_steps cap + EOS-1's post-eos TEXT continuation must
    // never exceed it). The batch path is bounded by the same max_steps=40.
    const MAX_STEPS: usize = 40;
    assert!(
        audio_codes_s.len() / 12 <= MAX_STEPS,
        "streaming emitted {} audio frames > max_steps {MAX_STEPS}",
        audio_codes_s.len() / 12
    );
    assert!(
        text_ids_s.len() <= MAX_STEPS,
        "streaming emitted {} text tokens > max_steps {MAX_STEPS}",
        text_ids_s.len()
    );

    // ── Compare text tokens ──
    let n_common = text_ids_b.len().min(text_ids_s.len());
    let mut div = None;
    for i in 0..n_common {
        if text_ids_b[i] != text_ids_s[i] {
            div = Some(i);
            break;
        }
    }

    // ── Compare audio codes (frame-by-frame) ──
    let n_frames_b = audio_codes_b.len() / 12;
    let n_frames_s = audio_codes_s.len() / 12;
    let mut code_div = None;
    for f in 0..n_frames_b.min(n_frames_s) {
        if audio_codes_b[f * 12..(f + 1) * 12] != audio_codes_s[f * 12..(f + 1) * 12] {
            code_div = Some(f);
            break;
        }
    }
    let frames_b = audio_codes_b.len() / 12;
    let frames_s = audio_codes_s.len() / 12;
    match code_div {
        Some(f) => {
            println!(
                "RESULT: AUDIO CODES DIVERGED at frame {f}: batch={:?} stream={:?}",
                &audio_codes_b[f * 12..(f + 1) * 12],
                &audio_codes_s[f * 12..(f + 1) * 12],
            );
            std::process::exit(1);
        }
        None => println!(
            "  audio codes IDENTICAL for {} frames",
            frames_b.min(frames_s)
        ),
    }

    let div_txt = match div {
        Some(i) => format!(
            "DIVERGED at text token {i}: batch={} stream={}",
            text_ids_b[i], text_ids_s[i],
        ),
        None => "text prefix identical".to_string(),
    };
    if div.is_none() && text_ids_b.len() == text_ids_s.len() && frames_b == frames_s {
        println!(
            "RESULT: IDENTICAL ({} text tokens, {frames_b} audio frames) — {div_txt}",
            text_ids_b.len()
        );
    } else {
        println!(
            "RESULT: {} (batch: {} text/{frames_b} audio; stream: {} text/{frames_s} audio)",
            div_txt,
            text_ids_b.len(),
            text_ids_s.len(),
        );
    }
}
