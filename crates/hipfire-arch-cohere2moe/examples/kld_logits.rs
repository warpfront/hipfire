// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! kld_logits (cohere2moe) — per-position next-token logits for a fixed token
//! list, for the BF16/Q8/MQ6/MQ4 KLD + PPL sweep of North-Mini-Code-1.0.
//!
//! Two modes:
//!   * `--dump <out.bin>`  : run ONE model over the token list (per-token
//!     prefill via `decode_step`) and write its per-position logits to a binary
//!     (u32 n_pos, u32 vocab, then n_pos*vocab f32 LE). Run this once per tier;
//!     KL(oracle‖tier) and wikitext PPL are then computed offline from the
//!     dumps + the token list (each 61 GB model loads only once).
//!   * `--model-a <ref> --model-b <cand>` : load both and print the per-position
//!     KL(softmax(ref) || softmax(cand)) distribution directly.
//!
//! Usage:
//!   kld_logits --model-a <model.hfq> --dump <out.bin> --tokens <tokens.json> [--max N]
//!   kld_logits --model-a <ref.hfq> --model-b <cand.hfq> --tokens <tokens.json> [--max N]
//!   [--prefill-tokens N] [--prefill-chunk N] [--reset-cycles N] [--state-out DIR]
//!
//!   --tokens : JSON array of u32 token ids, e.g. [504, 2849, 8868, ...]

use hipfire_arch_cohere2moe::cohere2moe::{Cohere2MoeState, Cohere2MoeWeights};
use hipfire_arch_cohere2moe::config::Cohere2MoeConfig;
use hipfire_arch_cohere2moe::forward::{decode_step, forward_batch, forward_batch_supported};
use hipfire_runtime::hfq::HfqFile;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
struct Observation {
    cycle: usize,
    ordinal: usize,
    phase: &'static str,
    end_position: usize,
    n_tokens: usize,
    k_bytes: Vec<usize>,
    v_bytes: Vec<usize>,
}

fn prepare_state_out(path: &Path) -> Result<(), String> {
    match fs::metadata(path) {
        Ok(meta) => {
            if !meta.is_dir() {
                return Err(format!(
                    "--state-out is not a directory: {}",
                    path.display()
                ));
            }
            let mut entries = fs::read_dir(path)
                .map_err(|e| format!("read --state-out {}: {e}", path.display()))?;
            if entries
                .next()
                .transpose()
                .map_err(|e| format!("read --state-out {}: {e}", path.display()))?
                .is_some()
            {
                return Err(format!(
                    "--state-out directory is not empty: {}",
                    path.display()
                ));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path)
                .map_err(|e| format!("create --state-out {}: {e}", path.display()))?;
        }
        Err(e) => {
            return Err(format!("stat --state-out {}: {e}", path.display()));
        }
    }
    Ok(())
}

fn record_observation(
    gpu: &mut rdna_compute::Gpu,
    state: &Cohere2MoeState,
    state_out: Option<&Path>,
    cycle: usize,
    ordinal: usize,
    phase: &'static str,
    end_position: usize,
) -> Result<Observation, String> {
    let k_gpu = &state.kv.k_gpu;
    let v_gpu = &state.kv.v_gpu;
    if k_gpu.is_empty() || v_gpu.is_empty() {
        return Err("cohere2moe state has no GPU K/V cache arrays".to_string());
    }
    if k_gpu.len() != v_gpu.len() {
        return Err(format!(
            "cohere2moe state K/V cache layer count mismatch: K={} V={}",
            k_gpu.len(),
            v_gpu.len()
        ));
    }

    let k_bytes: Vec<usize> = k_gpu.iter().map(|t| t.buf.size()).collect();
    let v_bytes: Vec<usize> = v_gpu.iter().map(|t| t.buf.size()).collect();
    if let Some(root) = state_out {
        let observation_dir = root.join(format!("c{cycle}")).join(format!("o{ordinal}"));
        fs::create_dir_all(&observation_dir).map_err(|e| {
            format!(
                "create state observation directory {}: {e}",
                observation_dir.display()
            )
        })?;

        for layer in 0..k_gpu.len() {
            let k_buf = &k_gpu[layer].buf;
            let mut k_data = vec![0u8; k_buf.size()];
            gpu.hip
                .memcpy_dtoh(&mut k_data, k_buf)
                .map_err(|e| format!("download state K layer {layer}: {e:?}"))?;
            fs::write(observation_dir.join(format!("l{layer}.k.bin")), &k_data)
                .map_err(|e| format!("write state K layer {layer}: {e}"))?;

            let v_buf = &v_gpu[layer].buf;
            let mut v_data = vec![0u8; v_buf.size()];
            gpu.hip
                .memcpy_dtoh(&mut v_data, v_buf)
                .map_err(|e| format!("download state V layer {layer}: {e:?}"))?;
            fs::write(observation_dir.join(format!("l{layer}.v.bin")), &v_data)
                .map_err(|e| format!("write state V layer {layer}: {e}"))?;
        }
    }

    Ok(Observation {
        cycle,
        ordinal,
        phase,
        end_position,
        n_tokens: state.n_tokens,
        k_bytes,
        v_bytes,
    })
}

fn write_observations(root: &Path, observations: &[Observation]) -> Result<(), String> {
    let mut json = serde_json::to_vec_pretty(observations)
        .map_err(|e| format!("serialize observations: {e}"))?;
    json.push(b'\n');
    let temporary = root.join("observations.json.tmp");
    fs::write(&temporary, json).map_err(|e| format!("write {}: {e}", temporary.display()))?;
    fs::rename(&temporary, root.join("observations.json"))
        .map_err(|e| format!("publish observations.json: {e}"))?;
    Ok(())
}

struct Args {
    model_a: String,
    model_b: Option<String>,
    tokens: String,
    max: usize,
    dump: Option<String>,
    prefill_tokens: usize,
    prefill_chunk: usize,
    reset_cycles: usize,
    state_out: Option<String>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model_a = None;
    let mut model_b = None;
    let mut tokens = None;
    let mut max = usize::MAX;
    let mut dump = None;
    let mut prefill_tokens = 0;
    let mut prefill_chunk = 1;
    let mut reset_cycles = 1;
    let mut state_out = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model-a" => {
                model_a = Some(argv[i + 1].clone());
                i += 2;
            }
            "--model-b" => {
                model_b = Some(argv[i + 1].clone());
                i += 2;
            }
            "--tokens" => {
                tokens = Some(argv[i + 1].clone());
                i += 2;
            }
            "--max" => {
                max = argv[i + 1].parse().expect("--max");
                i += 2;
            }
            "--dump" => {
                dump = Some(argv[i + 1].clone());
                i += 2;
            }
            "--prefill-tokens" => {
                prefill_tokens = argv[i + 1].parse().expect("--prefill-tokens");
                i += 2;
            }
            "--prefill-chunk" => {
                prefill_chunk = argv[i + 1].parse().expect("--prefill-chunk");
                i += 2;
            }
            "--reset-cycles" => {
                reset_cycles = argv[i + 1].parse().expect("--reset-cycles");
                i += 2;
            }
            "--state-out" => {
                state_out = Some(argv[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(1);
            }
        }
    }
    Args {
        model_a: model_a.expect("--model-a required"),
        model_b,
        tokens: tokens.expect("--tokens required"),
        max,
        dump,
        prefill_tokens,
        prefill_chunk,
        reset_cycles,
        state_out,
    }
}

fn validate_args(args: &Args) -> Result<(), String> {
    if args.prefill_chunk == 0 {
        return Err("--prefill-chunk must be greater than zero".to_string());
    }
    if args.reset_cycles == 0 {
        return Err("--reset-cycles must be greater than zero".to_string());
    }
    if args.prefill_chunk > 512 {
        return Err("--prefill-chunk must not exceed 512".to_string());
    }
    if args.state_out.is_some() && args.model_b.is_some() {
        return Err("--state-out cannot be used with --model-b".to_string());
    }
    Ok(())
}

fn load_tokens(path: &str) -> Vec<u32> {
    let raw = fs::read_to_string(path).expect("read tokens json");
    serde_json::from_str(&raw).expect("parse tokens json (expected JSON array of u32)")
}

/// Run `path` over the token list, returning the per-position next-token logits.
fn run_model(path: &str, args: &Args) -> Result<Vec<Vec<f32>>, String> {
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    assert_eq!(
        hfq.arch_id, 12,
        "kld_logits(cohere2moe): expected arch_id 12, got {}",
        hfq.arch_id
    );
    let cfg = Cohere2MoeConfig::from_hfq(&hfq).expect("config");
    eprintln!(
        "[{}] cohere2moe hidden={} layers={} experts={}/{} dense_prefix={} vocab={}",
        path,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.first_k_dense_replace,
        cfg.vocab_size,
    );
    let weights = Cohere2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    let tokens = load_tokens(&args.tokens);
    let n = tokens.len().min(args.max);
    if args.prefill_tokens > n {
        let error = format!(
            "--prefill-tokens ({}) exceeds available token count ({n})",
            args.prefill_tokens
        );
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        return Err(error);
    }
    let state_root = args.state_out.as_ref().map(|path| Path::new(path.as_str()));
    if let Some(root) = state_root {
        if let Err(e) = prepare_state_out(root) {
            weights.free_gpu(&mut gpu);
            gpu.drain_pool();
            return Err(e);
        }
    }
    if args.prefill_chunk > 1 && !forward_batch_supported(&weights) {
        let error =
            "--prefill-chunk above one requires forward_batch_supported(&weights)".to_string();
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        return Err(error);
    }
    let max_seq = match n.checked_add(16) {
        Some(max_seq) => max_seq,
        None => {
            weights.free_gpu(&mut gpu);
            gpu.drain_pool();
            return Err("token count overflows state sequence capacity".to_string());
        }
    };
    let mut state = match Cohere2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq) {
        Ok(state) => state,
        Err(e) => {
            weights.free_gpu(&mut gpu);
            gpu.drain_pool();
            return Err(format!("state: {e}"));
        }
    };

    let result = (|| -> Result<(Vec<Vec<f32>>, Vec<Observation>), String> {
        let mut all = Vec::new();
        let mut observations = Vec::new();
        for cycle in 0..args.reset_cycles {
            state.reset(&mut gpu)?;
            let mut consumed = 0usize;
            let mut ordinal = 0usize;

            while consumed < args.prefill_tokens {
                let batch = (args.prefill_tokens - consumed).min(args.prefill_chunk);
                let end = consumed + batch;
                let logits = if args.prefill_chunk == 1 {
                    decode_step(
                        &cfg,
                        &weights,
                        &mut state,
                        &mut gpu,
                        tokens[consumed],
                        consumed as u32,
                    )?
                } else {
                    forward_batch(
                        &cfg,
                        &weights,
                        &mut state,
                        &mut gpu,
                        &tokens[consumed..end],
                        consumed,
                    )?
                };
                let observation = record_observation(
                    &mut gpu,
                    &state,
                    state_root,
                    cycle,
                    ordinal,
                    "prefill",
                    end - 1,
                )?;
                all.push(logits);
                observations.push(observation);
                if (end - 1) % 64 == 0 {
                    eprintln!("  prefill pos {}/{}", end - 1, n);
                }
                consumed = end;
                ordinal += 1;
            }

            while consumed < n {
                let position = consumed;
                let logits = decode_step(
                    &cfg,
                    &weights,
                    &mut state,
                    &mut gpu,
                    tokens[position],
                    position as u32,
                )?;
                let observation = record_observation(
                    &mut gpu, &state, state_root, cycle, ordinal, "decode", position,
                )?;
                all.push(logits);
                observations.push(observation);
                if position % 64 == 0 {
                    eprintln!("  pos {position}/{n}");
                }
                consumed += 1;
                ordinal += 1;
            }
        }
        Ok((all, observations))
    })();

    let output = match result {
        Ok((all, observations)) => match state_root {
            Some(root) => write_observations(root, &observations).map(|()| all),
            None => Ok(all),
        },
        Err(e) => Err(e),
    };
    state.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    output
}

/// Per-position logits → binary: u32 n_pos, u32 vocab, then n_pos*vocab f32 LE.
fn dump_logits(path: &str, logits: &[Vec<f32>]) {
    let n = logits.len() as u32;
    let vocab = logits.first().map(|r| r.len()).unwrap_or(0) as u32;
    let mut f = std::io::BufWriter::new(fs::File::create(path).expect("create dump"));
    f.write_all(&n.to_le_bytes()).unwrap();
    f.write_all(&vocab.to_le_bytes()).unwrap();
    for row in logits {
        for &v in row {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    eprintln!("dumped {n} positions × {vocab} vocab → {path}");
}

/// KL(softmax(ref) || softmax(cand)) in nats (max-subtracted for stability).
fn compute_kl(r: &[f32], c: &[f32]) -> f64 {
    assert_eq!(r.len(), c.len());
    let rmax = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let cmax = c.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let (mut rs, mut cs) = (0.0f64, 0.0f64);
    for i in 0..r.len() {
        rs += ((r[i] as f64) - rmax).exp();
        cs += ((c[i] as f64) - cmax).exp();
    }
    let (lrs, lcs) = (rs.ln() + rmax, cs.ln() + cmax);
    let mut kl = 0.0f64;
    for i in 0..r.len() {
        let lp = (r[i] as f64) - lrs;
        let p = lp.exp();
        if p > 0.0 {
            kl += p * (lp - ((c[i] as f64) - lcs));
        }
    }
    if kl < 0.0 && kl > -1e-9 {
        kl = 0.0;
    }
    kl
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_summary(kls: &[f64]) {
    let n = kls.len();
    if n == 0 {
        eprintln!("no positions");
        return;
    }
    let mut s = kls.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = kls.iter().sum::<f64>() / n as f64;
    println!("=== KL(ref || cand), nats ===");
    println!("positions : {n}");
    println!("mean      : {mean:.6}");
    println!("median    : {:.6}", percentile(&s, 0.50));
    println!("p99       : {:.6}", percentile(&s, 0.99));
    println!("max       : {:.6}", s.last().unwrap());
    println!(
        "frac>0.1  : {:.4}",
        kls.iter().filter(|&&k| k > 0.1).count() as f64 / n as f64
    );
}

fn main() {
    let args = parse_args();
    if let Err(e) = validate_args(&args) {
        eprintln!("kld_logits: {e}");
        std::process::exit(1);
    }
    if let Some(dump_path) = &args.dump {
        eprintln!("=== dump model: {} ===", args.model_a);
        let logits = run_model(&args.model_a, &args).unwrap_or_else(|e| {
            eprintln!("kld_logits: {e}");
            std::process::exit(1);
        });
        dump_logits(dump_path, &logits);
        return;
    }
    eprintln!("=== model-a (reference): {} ===", args.model_a);
    let r = run_model(&args.model_a, &args).unwrap_or_else(|e| {
        eprintln!("kld_logits: {e}");
        std::process::exit(1);
    });
    let mb = args
        .model_b
        .clone()
        .expect("--model-b required (or use --dump)");
    eprintln!("=== model-b (candidate): {mb} ===");
    let c = run_model(&mb, &args).unwrap_or_else(|e| {
        eprintln!("kld_logits: {e}");
        std::process::exit(1);
    });
    let n = r.len().min(c.len());
    let kls: Vec<f64> = (0..n).map(|i| compute_kl(&r[i], &c[i])).collect();
    print_summary(&kls);
}
