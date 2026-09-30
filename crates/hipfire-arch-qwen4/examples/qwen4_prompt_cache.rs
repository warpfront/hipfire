// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_prompt_cache — numeric oracle for Qwen4 prompt-cache reuse.
//!
//! A prompt-cache hit continues the device state after a committed turn and
//! prefills only the new suffix. This probe checks that continuation against a
//! cold full prefill of the same token stream:
//!
//! - warm: prefill `P[..a]`, greedy-decode `g` tokens `G`, prefill the suffix
//!   `P[a..a+s]`, then greedy-continue;
//! - cold: reset, prefill `P[..a] ++ G ++ P[a..a+s]` in one call, then
//!   greedy-continue.
//!
//! The greedy continuations must agree token for token, except that a
//! mismatch where the cold top-1/top-2 margin is below 0.05 ends the
//! comparison as a numeric tie.
//!
//! Usage: qwen4_prompt_cache MODEL.hfq PROMPT.txt

use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

const N_CTX: usize = 2048;
const CONTINUE: usize = 16;
const TIE_MARGIN: f32 = 0.05;
/// `(a, g, s)`: `a + g` ends mid QSA pool (341 % 4 == 1); the second config
/// makes the cold prefill ≥ 512 rows so it takes the F16 WMMA arms.
const CONFIGS: [(usize, usize, usize); 2] = [(300, 41, 150), (400, 41, 150)];

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    logits: GpuTensor,
    tokenizer: Tokenizer,
}

impl Model {
    /// Same range-loaded path as the serve loader (see `qwen4_kld`).
    fn load(path: &Path) -> Result<Self, String> {
        let mut hfq =
            HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let receipt = admit_hfqm_artifact(&hfq).map_err(|e| format!("qwen4 admission: {e}"))?;
        let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
        if gpu.is_uma() {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let source = HfqModelSource::from_hfq(hfq);
        let transaction = fulfill_manifest_from_payloads(
            &receipt.manifest.weights,
            &mesh,
            receipt.config.num_hidden_layers,
            &mut gpu,
            expected,
            |entry| {
                source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name))
            },
        )
        .map_err(|e| format!("qwen4 manifest fulfillment: {e}"))?;
        let vocab = receipt.config.vocab_size;
        let mut bundle = Qwen4Bundle::assemble_with_metadata(
            receipt.config,
            transaction,
            &receipt.placements,
            &mut gpu,
            N_CTX,
            receipt.ple,
        )
        .map_err(|e| format!("qwen4 bundle assembly: {e}"))?;
        bundle
            .attach_forward(&mut gpu, N_CTX)
            .map_err(|e| format!("qwen4 forward setup: {e}"))?;
        let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        eprintln!(
            "qwen4_prompt_cache: loaded {} on {}",
            path.display(),
            gpu.arch
        );
        Ok(Self {
            gpu,
            bundle,
            logits,
            tokenizer,
        })
    }

    fn reset(&mut self) -> Result<(), String> {
        self.bundle.reset(&mut self.gpu).map_err(|e| e.to_string())
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.bundle
            .forward_chunk_final(&mut self.gpu, tokens, &self.logits, None)
            .map_err(|e| e.to_string())?;
        self.download()
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.bundle
            .forward_token_or_argmax(&mut self.gpu, Some(token), &self.logits)
            .map_err(|e| e.to_string())?;
        self.download()
    }

    fn download(&mut self) -> Result<Vec<f32>, String> {
        let row = self
            .gpu
            .download_f32(&self.logits)
            .map_err(|e| e.to_string())?;
        if row.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok(row)
    }

    /// Greedy continuation from `logits`: `CONTINUE` host-argmax tokens and
    /// each pick's top-1 − top-2 margin.
    fn greedy(&mut self, mut logits: Vec<f32>) -> Result<(Vec<u32>, Vec<f32>), String> {
        let mut tokens = Vec::with_capacity(CONTINUE);
        let mut margins = Vec::with_capacity(CONTINUE);
        for _ in 0..CONTINUE {
            let (token, margin) = top2(&logits);
            tokens.push(token);
            margins.push(margin);
            logits = self.step(token)?;
        }
        Ok((tokens, margins))
    }
}

/// Host argmax (first maximum) and its lead over the runner-up.
fn top2(logits: &[f32]) -> (u32, f32) {
    let (mut best, mut first, mut second) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (index, &value) in logits.iter().enumerate() {
        if value > first {
            second = first;
            first = value;
            best = index;
        } else if value > second {
            second = value;
        }
    }
    (best as u32, first - second)
}

/// Returns whether the config passed.
fn run_config(
    model: &mut Model,
    prompt: &[u32],
    (a, g, s): (usize, usize, usize),
) -> Result<bool, String> {
    // Warm arm: committed turn (prefix + greedy decode), then suffix prefill.
    model.reset()?;
    let mut logits = model.prefill(&prompt[..a])?;
    let mut generated = Vec::with_capacity(g);
    for _ in 0..g {
        let token = top2(&logits).0;
        generated.push(token);
        logits = model.step(token)?;
    }
    if model.bundle.state.position != a + g {
        return Err(format!(
            "warm position {} != {}",
            model.bundle.state.position,
            a + g
        ));
    }
    let warm = model.prefill(&prompt[a..a + s])?;
    if model.bundle.state.position != a + g + s {
        return Err(format!(
            "warm position {} != {}",
            model.bundle.state.position,
            a + g + s
        ));
    }
    let (warm_tokens, _) = model.greedy(warm.clone())?;

    // Cold arm: the same stream in one prefill.
    model.reset()?;
    let stream: Vec<u32> = prompt[..a]
        .iter()
        .chain(&generated)
        .chain(&prompt[a..a + s])
        .copied()
        .collect();
    let cold = model.prefill(&stream)?;
    let (cold_tokens, cold_margins) = model.greedy(cold.clone())?;

    let max_abs = warm
        .iter()
        .zip(&cold)
        .map(|(w, c)| (w - c).abs())
        .fold(0.0f32, f32::max);
    let mut equal = 0usize;
    let mut tie_at = None;
    let mut pass = true;
    for i in 0..CONTINUE {
        if warm_tokens[i] == cold_tokens[i] {
            equal += 1;
            continue;
        }
        if cold_margins[i] < TIE_MARGIN {
            tie_at = Some(i);
        } else {
            pass = false;
        }
        break;
    }
    println!(
        "config={a}/{g}/{s} argmax_w={} argmax_c={} max_abs={max_abs:.6} tokens_equal={equal}/{CONTINUE} tie_at={} verdict={}",
        warm_tokens[0],
        cold_tokens[0],
        tie_at.map_or("-".to_string(), |i| i.to_string()),
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        println!(
            "  warm={:?}\n  cold={:?}\n  warm_text={:?}\n  cold_text={:?}",
            warm_tokens,
            cold_tokens,
            model.tokenizer.decode(&warm_tokens),
            model.tokenizer.decode(&cold_tokens)
        );
    }
    Ok(pass)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: qwen4_prompt_cache MODEL.hfq PROMPT.txt");
        std::process::exit(2);
    }
    let result = (|| -> Result<bool, String> {
        let mut model = Model::load(Path::new(&args[1]))?;
        let text = std::fs::read_to_string(&args[2]).map_err(|e| format!("{}: {e}", args[2]))?;
        let prompt = model.tokenizer.encode(&text);
        if prompt.len() < 600 {
            return Err(format!("prompt has {} tokens, need ≥ 600", prompt.len()));
        }
        let mut all = true;
        for config in CONFIGS {
            all &= run_config(&mut model, &prompt, config)?;
        }
        Ok(all)
    })();
    match result {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(error) => {
            eprintln!("qwen4_prompt_cache: {error}");
            std::process::exit(1);
        }
    }
}
