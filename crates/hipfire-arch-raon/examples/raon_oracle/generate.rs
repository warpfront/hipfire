// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `generate` subcommand (PLAN §6.2, slice I join): greedy text
//! generation through the production bundle.
//!
//! Opts: `model` (HFQ), `prompt-file` (UTF-8 text = the user message),
//! optional `system-file`, `n` (new tokens, default 64), `think on|off`
//! (default on; the template's `enable_thinking`), `prefill serial|batched`
//! (default batched), `json` (also print the result JSON to stdout).
//!
//! The prompt is the HFQ-embedded Jinja chat template rendered by the
//! production `JinjaChatFrame` and tokenized by the production tokenizer built
//! from the same HFQ metadata (no BOS). Decoding is greedy (host argmax over the
//! downloaded device logits, first maximum on ties) and stops at an EOS id
//! (`bundle.is_eos`; the EOS token is not part of the output) or after `n`
//! tokens.
//!
//! stdout carries only: `=== prompt tokens: N ===`, `=== generated ===`, the
//! generated text, `=== end (M tokens, stop: eos|length) ===` (M excludes the
//! EOS terminator). Diagnostics go to stderr. Result:
//! `{"pass":true,"tokens":[..],"text":..,"stop":..,"prompt_tokens":N}`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::prompt_frame::JinjaChatFrame;
use hipfire_runtime::tokenizer::{TokenTextStream, Tokenizer};
use rdna_compute::Gpu;
use serde_json::{json, Value};

use super::logits::{check_tokens, load_bundle, parse_prefill};

/// First maximum (lowest index on ties).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn read_text(path: &str, what: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("read {what} {path}: {e}"))
}

pub fn run(_fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let model = opts
        .get("model")
        .ok_or("generate: missing --model")?
        .clone();
    let prompt_file = opts
        .get("prompt-file")
        .ok_or("generate: missing --prompt-file")?;
    let user = read_text(prompt_file, "prompt")?;
    let system = match opts.get("system-file") {
        Some(p) => Some(read_text(p, "system")?),
        None => None,
    };
    let n_new = match opts.get("n") {
        Some(s) => s.parse::<usize>().map_err(|e| format!("--n `{s}`: {e}"))?,
        None => 64,
    };
    if n_new == 0 {
        return Err("--n must be >= 1".into());
    }
    let think = match opts.get("think").map(String::as_str).unwrap_or("on") {
        "on" => true,
        "off" => false,
        other => return Err(format!("--think `{other}`: on|off")),
    };
    let mode = parse_prefill(opts)?;

    // Tokenizer and template come from the HFQ metadata, exactly as the daemon builds them.
    let hfq = HfqFile::open(Path::new(&model)).map_err(|e| format!("open {model}: {e}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let template = hfq
        .chat_template()
        .ok_or("the HFQ carries no chat_template (tokenizer_config.chat_template)")?;
    let frame = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: system.as_deref(),
        user: &user,
        enable_thinking: think,
        // The Raon template never reads bos_token (bos_token is null upstream).
        bos_token: Some(""),
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let prompt = frame.render_and_encode()?;
    if prompt.is_empty() {
        return Err("rendered prompt tokenized to zero tokens".into());
    }

    let mut gpu = Gpu::init().map_err(|e| format!("generate: gpu init: {e:?}"))?;
    let max_seq = (prompt.len() + n_new + 64).max(512);
    let mut bundle = load_bundle(&mut gpu, ModelSource::Hfq(hfq), &model, max_seq)?;
    // `--mirror-budget-rows N`: force the decompressed-route mirror budget (rows) so the
    // beyond-budget routes (capacity refusal / absorbed with HIPFIRE_RAON_MLA_ABSORB=1) run
    // at test-sized contexts.
    if let Some(v) = opts.get("mirror-budget-rows") {
        let rows = v
            .parse::<usize>()
            .map_err(|e| format!("--mirror-budget-rows `{v}`: {e}"))?;
        bundle.state.set_mirror_budget_rows(rows);
    }
    let r = (|| -> Result<Value, String> {
        check_tokens(&prompt, bundle.config.vocab, "prompt")?;
        bundle.set_prefill_mode(mode);
        bundle.reset(&mut gpu)?;

        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        let w = |e: std::io::Error| format!("stdout: {e}");
        writeln!(out, "=== prompt tokens: {} ===", prompt.len()).map_err(w)?;
        writeln!(out, "=== generated ===").map_err(w)?;
        out.flush().map_err(w)?;

        let t0 = Instant::now();
        bundle.prefill(&mut gpu, &prompt)?;
        let prefill_secs = t0.elapsed().as_secs_f64();

        let mut generated: Vec<u32> = Vec::with_capacity(n_new);
        let mut text = String::new();
        let mut stream = TokenTextStream::new();
        let mut stop = "length";
        let t1 = Instant::now();
        loop {
            let logits = bundle.download_logits(&gpu)?;
            if logits.iter().any(|x| !x.is_finite()) {
                return Err(format!(
                    "non-finite logits at generated token {}",
                    generated.len()
                ));
            }
            let tok = argmax(&logits) as u32;
            if bundle.is_eos(tok) {
                stop = "eos";
                break;
            }
            generated.push(tok);
            let piece = stream.push(&tokenizer, tok);
            if !piece.is_empty() {
                write!(out, "{piece}").map_err(w)?;
                out.flush().map_err(w)?;
                text.push_str(&piece);
            }
            if generated.len() >= n_new {
                break;
            }
            bundle.decode_step(&mut gpu, tok)?;
        }
        let tail = stream.flush();
        if !tail.is_empty() {
            write!(out, "{tail}").map_err(w)?;
            text.push_str(&tail);
        }
        writeln!(out).map_err(w)?;
        writeln!(
            out,
            "=== end ({} tokens, stop: {stop}) ===",
            generated.len()
        )
        .map_err(w)?;
        out.flush().map_err(w)?;
        let decode_secs = t1.elapsed().as_secs_f64();
        let (captures, replays) = bundle.decode_graph_stats();
        eprintln!(
            "generate: decode_mode={} (graph captures {captures}, replays {replays}); prefill {} tok in {prefill_secs:.2}s ({:.1} tok/s); decode {} tok in {decode_secs:.2}s ({:.1} tok/s)",
            bundle.decode_mode(),
            prompt.len(),
            prompt.len() as f64 / prefill_secs.max(1e-9),
            generated.len(),
            generated.len() as f64 / decode_secs.max(1e-9),
        );

        let result = json!({
            "pass": true,
            "tokens": generated,
            "text": text,
            "stop": stop,
            "prompt_tokens": prompt.len(),
            "think": think,
            "decode_mode": bundle.decode_mode(),
            "prefill": match mode {
                hipfire_arch_raon::bundle::PrefillMode::Serial => "serial",
                hipfire_arch_raon::bundle::PrefillMode::Batched => "batched",
            },
        });
        if opts.contains_key("json") {
            let s = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
            writeln!(out, "{s}").map_err(w)?;
            out.flush().map_err(w)?;
        }
        Ok(result)
    })();
    bundle.free_all(&mut gpu);
    r
}
