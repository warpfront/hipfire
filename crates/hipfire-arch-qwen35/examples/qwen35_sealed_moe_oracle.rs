// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validation-only Qwen3.5 Single MoE oracle.
//!
//! This example deliberately drives the existing Qwen loading and forward APIs.
//! The `moe-oracle` feature only adds logical readback and report comparison;
//! it does not select a different model, kernel, or execution topology.

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, HfqSource, Layout, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::{KvCache, KvCacheExt, KvDims, KvLayers, KvTarget};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const PINNED_BASE: &str = "090ec173121e085a1f24f903fb3f0ff1f2b480f3";
const REPEAT_WINDOW: usize = 128;
const LIFECYCLE_CYCLES: usize = 4;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt_file: PathBuf,
    prefill_chunk: usize,
    decode_positions: usize,
    out: PathBuf,
    compare: Option<PathBuf>,
}

fn usage() -> &'static str {
    "qwen35_sealed_moe_oracle --model PATH --prompt-file PATH --prefill-chunk N \
--decode-positions N --out DIR [--compare DIR]"
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut prompt_file = None;
    let mut prefill_chunk = None;
    let mut decode_positions = None;
    let mut out = None;
    let mut compare = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = |name: &str, args: &mut std::iter::Skip<std::env::Args>| {
            args.next()
                .ok_or_else(|| format!("missing value for {name}; usage: {}", usage()))
        };
        match flag.as_str() {
            "--model" => {
                if model.is_some() {
                    return Err("duplicate --model".to_string());
                }
                model = Some(PathBuf::from(value("--model", &mut args)?));
            }
            "--prompt-file" => {
                if prompt_file.is_some() {
                    return Err("duplicate --prompt-file".to_string());
                }
                prompt_file = Some(PathBuf::from(value("--prompt-file", &mut args)?));
            }
            "--prefill-chunk" => {
                if prefill_chunk.is_some() {
                    return Err("duplicate --prefill-chunk".to_string());
                }
                let raw = value("--prefill-chunk", &mut args)?;
                let parsed = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --prefill-chunk {raw:?}"))?;
                if parsed == 0 {
                    return Err("--prefill-chunk must be greater than zero".to_string());
                }
                prefill_chunk = Some(parsed);
            }
            "--decode-positions" => {
                if decode_positions.is_some() {
                    return Err("duplicate --decode-positions".to_string());
                }
                let raw = value("--decode-positions", &mut args)?;
                let parsed = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --decode-positions {raw:?}"))?;
                if parsed == 0 {
                    return Err("--decode-positions must be greater than zero".to_string());
                }
                decode_positions = Some(parsed);
            }
            "--out" => {
                if out.is_some() {
                    return Err("duplicate --out".to_string());
                }
                out = Some(PathBuf::from(value("--out", &mut args)?));
            }
            "--compare" => {
                if compare.is_some() {
                    return Err("duplicate --compare".to_string());
                }
                compare = Some(PathBuf::from(value("--compare", &mut args)?));
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}; usage: {}", usage())),
        }
    }
    let model = model.ok_or_else(|| format!("missing --model; usage: {}", usage()))?;
    let prompt_file =
        prompt_file.ok_or_else(|| format!("missing --prompt-file; usage: {}", usage()))?;
    let prefill_chunk =
        prefill_chunk.ok_or_else(|| format!("missing --prefill-chunk; usage: {}", usage()))?;
    let decode_positions = decode_positions
        .ok_or_else(|| format!("missing --decode-positions; usage: {}", usage()))?;
    let out = out.ok_or_else(|| format!("missing --out; usage: {}", usage()))?;
    if model.as_os_str().is_empty()
        || prompt_file.as_os_str().is_empty()
        || out.as_os_str().is_empty()
    {
        return Err("model, prompt, and output paths must not be empty".to_string());
    }
    Ok(Args {
        model,
        prompt_file,
        prefill_chunk,
        decode_positions,
        out,
        compare,
    })
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_file(path: &Path) -> Result<(String, usize), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let size = bytes.len();
    Ok((sha256(&bytes), size))
}

fn token_digest(tokens: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256(&bytes)
}

fn git_head() -> Result<String, String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| format!("run git rev-parse HEAD: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let head = String::from_utf8(output.stdout)
        .map_err(|e| format!("git returned non-UTF8 commit: {e}"))?
        .trim()
        .to_string();
    if head.len() != 40 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("git returned malformed commit {head:?}"));
    }
    Ok(head)
}

fn config_identity(config: &Qwen35Config, metadata_sha256: &str) -> Value {
    let layer_types: Vec<&str> = config
        .layer_types
        .iter()
        .map(|layer| match layer {
            qwen35::LayerType::LinearAttention => "linear_attention",
            qwen35::LayerType::FullAttention => "full_attention",
        })
        .collect();
    json!({
        "metadata_sha256": metadata_sha256,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "vocab_size": config.vocab_size,
        "norm_eps": config.norm_eps,
        "eos_token": config.eos_token,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "rope_theta": config.rope_theta,
        "partial_rotary_factor": config.partial_rotary_factor,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "linear_key_head_dim": config.linear_key_head_dim,
        "linear_value_head_dim": config.linear_value_head_dim,
        "conv_kernel_dim": config.conv_kernel_dim,
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_intermediate_size": config.moe_intermediate_size,
        "shared_expert_intermediate_size": config.shared_expert_intermediate_size,
        "has_shared_expert": config.has_shared_expert,
        "norm_topk_prob": config.norm_topk_prob,
        "layer_types": layer_types,
    })
}

fn load_weights(
    model: &Path,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> Result<Qwen35Weights, String> {
    let mut hfq =
        HfqFile::open(model).map_err(|e| format!("open model {}: {e}", model.display()))?;
    let mut source = HfqSource::new(&mut hfq, config);
    qwen35::load_weights(
        &mut source,
        std::slice::from_mut(gpu),
        &Layout::single(config.n_layers),
    )
    .map_err(|e| format!("load Qwen35 weights: {e:?}"))
}

fn make_kv(gpu: &mut Gpu, config: &Qwen35Config, max_seq: usize) -> Result<KvCache, String> {
    let is_kv_layer = config
        .layer_types
        .iter()
        .map(|layer| *layer == qwen35::LayerType::FullAttention)
        .collect();
    <KvCache as KvCacheExt>::from_mode(
        KvMode::Q8,
        KvTarget::Single(gpu),
        &KvDims {
            layers: KvLayers::Mask(is_kv_layer),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            max_seq,
            physical_cap: Some(max_seq),
        },
    )
    .map_err(|e| format!("allocate Q8 KV cache: {e:?}"))
}

fn forward_prefill(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start: usize,
    kv: &mut KvCache,
    dn: &mut DeltaNetState,
    scratch: &Qwen35Scratch,
    chunk: usize,
) -> Result<(), String> {
    if tokens.is_empty() {
        return Ok(());
    }
    if chunk == 1 {
        for (offset, &token) in tokens.iter().enumerate() {
            let position = start
                .checked_add(offset)
                .ok_or_else(|| "single-token prefill position overflow".to_string())?;
            hipfire_arch_qwen35::qwen35::oracle::set_decode_position(position)?;
            qwen35::forward_scratch(gpu, weights, config, token, position, kv, dn, scratch)
                .map_err(|e| format!("single-token prefill at position {position}: {e:?}"))?;
        }
        return Ok(());
    }
    qwen35::forward_prefill_batch_capped(
        gpu, weights, config, tokens, start, kv, dn, scratch, None, None, None, None, chunk,
    )
    .map_err(|e| format!("prefill at position {start}: {e:?}"))
}

fn run_sequence(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    prefill_chunk: usize,
    decode_positions: usize,
    label: &str,
    warm_prefix: bool,
) -> Result<(), String> {
    let max_seq = tokens
        .len()
        .checked_add(decode_positions)
        .and_then(|value| value.checked_add(32))
        .ok_or_else(|| "sequence capacity overflow".to_string())?;
    let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, REPEAT_WINDOW, max_seq)
        .map_err(|e| format!("allocate scratch for {label}: {e:?}"))?;
    let mut kv = make_kv(gpu, config, max_seq)?;
    let mut dn = DeltaNetState::new_with_quant(gpu, config, StateQuant::FP32)
        .map_err(|e| format!("allocate DeltaNet state for {label}: {e:?}"))?;
    let result = (|| -> Result<(), String> {
        hipfire_arch_qwen35::qwen35::oracle::set_sequence(label)?;
        if warm_prefix {
            let split = tokens.len() / 2;
            if split == 0 || split == tokens.len() {
                return Err("warm-prefix sequence requires at least two prompt tokens".to_string());
            }
            forward_prefill(
                gpu,
                weights,
                config,
                &tokens[..split],
                0,
                &mut kv,
                &mut dn,
                &scratch,
                prefill_chunk,
            )?;
            forward_prefill(
                gpu,
                weights,
                config,
                &tokens[split..],
                split,
                &mut kv,
                &mut dn,
                &scratch,
                prefill_chunk,
            )?;
        } else {
            forward_prefill(
                gpu,
                weights,
                config,
                tokens,
                0,
                &mut kv,
                &mut dn,
                &scratch,
                prefill_chunk,
            )?;
        }
        let last_prompt_token = *tokens
            .last()
            .ok_or_else(|| "prompt tokenization produced no tokens".to_string())?;
        hipfire_arch_qwen35::qwen35::oracle::record_step(
            gpu,
            &scratch,
            &kv,
            &dn,
            last_prompt_token,
            tokens.len() - 1,
        )?;
        for offset in 0..decode_positions {
            let token = gpu
                .argmax_f32(&scratch.logits, config.vocab_size)
                .map_err(|e| format!("argmax at decode offset {offset}: {e:?}"))?;
            let position = tokens
                .len()
                .checked_add(offset)
                .ok_or_else(|| "decode position overflow".to_string())?;
            qwen35::forward_scratch(
                gpu, weights, config, token, position, &mut kv, &mut dn, &scratch,
            )
            .map_err(|e| format!("decode position {position}: {e:?}"))?;
            hipfire_arch_qwen35::qwen35::oracle::record_step(
                gpu, &scratch, &kv, &dn, token, position,
            )?;
        }
        Ok(())
    })();
    let scratch_result = scratch
        .free_gpu(gpu)
        .map_err(|e| format!("free scratch: {e:?}"));
    dn.free_gpu(gpu);
    let kv_result = kv
        .free_gpu(gpu)
        .map_err(|e| format!("free KV cache: {e:?}"));
    result.and(scratch_result).and(kv_result)
}

fn run_lifecycle_cycles(
    gpu: &mut Gpu,
    model: &Path,
    config: &Qwen35Config,
    tokens: &[u32],
    prefill_chunk: usize,
) -> Result<(), String> {
    let first = *tokens
        .first()
        .ok_or_else(|| "prompt tokenization produced no tokens".to_string())?;
    for cycle in 0..LIFECYCLE_CYCLES {
        let weights = load_weights(model, config, gpu)?;
        let max_seq = prefill_chunk
            .max(2)
            .checked_add(REPEAT_WINDOW)
            .ok_or_else(|| "lifecycle capacity overflow".to_string())?;
        let scratch = Qwen35Scratch::new_with_kv_max(gpu, config, REPEAT_WINDOW, max_seq)
            .map_err(|e| format!("allocate lifecycle scratch {cycle}: {e:?}"))?;
        let mut kv = make_kv(gpu, config, max_seq)?;
        let mut dn = DeltaNetState::new_with_quant(gpu, config, StateQuant::FP32)
            .map_err(|e| format!("allocate lifecycle DeltaNet state {cycle}: {e:?}"))?;
        hipfire_arch_qwen35::qwen35::oracle::set_sequence(&format!("lifecycle-{cycle}"))?;
        qwen35::forward_scratch(gpu, &weights, config, first, 0, &mut kv, &mut dn, &scratch)
            .map_err(|e| format!("lifecycle forward {cycle}: {e:?}"))?;
        dn.reset(gpu)
            .map_err(|e| format!("reset DeltaNet state in lifecycle {cycle}: {e:?}"))?;
        scratch
            .free_gpu(gpu)
            .map_err(|e| format!("free lifecycle scratch {cycle}: {e:?}"))?;
        dn.free_gpu(gpu);
        kv.free_gpu(gpu)
            .map_err(|e| format!("free lifecycle KV {cycle}: {e:?}"))?;
        weights.free_gpu(gpu);
        gpu.drain_pool();
    }
    Ok(())
}

fn run(args: Args) -> Result<PathBuf, String> {
    if !args.model.is_file() {
        return Err(format!(
            "model is not a regular file: {}",
            args.model.display()
        ));
    }
    if !args.prompt_file.is_file() {
        return Err(format!(
            "prompt is not a regular file: {}",
            args.prompt_file.display()
        ));
    }
    if args.out.join("report.json").exists() {
        return Err(format!(
            "refusing to overwrite existing report: {}",
            args.out.join("report.json").display()
        ));
    }
    if let Some(compare) = &args.compare {
        if !compare.join("report.json").is_file() {
            return Err(format!(
                "comparison report is missing: {}",
                compare.display()
            ));
        }
    }
    let model = fs::canonicalize(&args.model)
        .map_err(|e| format!("canonicalize model {}: {e}", args.model.display()))?;
    let prompt_path = fs::canonicalize(&args.prompt_file)
        .map_err(|e| format!("canonicalize prompt {}: {e}", args.prompt_file.display()))?;
    let prompt_bytes = fs::read(&prompt_path)
        .map_err(|e| format!("read prompt {}: {e}", prompt_path.display()))?;
    let prompt = String::from_utf8(prompt_bytes.clone())
        .map_err(|e| format!("prompt is not valid UTF-8: {e}"))?;
    if prompt.is_empty() {
        return Err("prompt file is empty".to_string());
    }
    let (model_sha256, model_bytes) = sha256_file(&model)?;
    let (binary_sha256, binary_bytes) =
        sha256_file(&env::current_exe().map_err(|e| format!("locate oracle binary: {e}"))?)?;
    let hfq = HfqFile::open(&model).map_err(|e| format!("open model: {e}"))?;
    let metadata_sha256 = sha256(hfq.metadata_json.as_bytes());
    let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("read Qwen35 config: {e}"))?;
    if config.num_experts == 0 || config.num_experts_per_tok == 0 {
        return Err("sealed MoE oracle requires a Qwen35 MoE model".to_string());
    }
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("load tokenizer: {e}"))?;
    let tokens = tokenizer.encode(&prompt);
    if tokens.len() < 2 {
        return Err(
            "prompt must tokenize to at least two tokens for warm-prefix validation".to_string(),
        );
    }
    drop(hfq);
    let token_ids_sha256 = token_digest(&tokens);
    let commit = git_head()?;
    let mut gpu = Gpu::init().map_err(|e| format!("GPU init: {e:?}"))?;
    let identity = json!({
        "fixture": {
            "model_sha256": model_sha256,
            "model_bytes": model_bytes,
            "prompt_sha256": sha256(&prompt_bytes),
            "prompt_bytes": prompt_bytes.len(),
            "token_ids_sha256": token_ids_sha256,
            "token_count": tokens.len(),
            "prefill_chunk": args.prefill_chunk,
            "decode_positions": args.decode_positions,
        },
        "source": {
            "model_path": model.display().to_string(),
            "model_sha256": model_sha256,
            "metadata_sha256": metadata_sha256,
            "prompt_path": prompt_path.display().to_string(),
            "prompt_sha256": sha256(&prompt_bytes),
            "token_ids": tokens,
            "token_ids_sha256": token_ids_sha256,
        },
        "binary": {
            "path": env::current_exe().map_err(|e| format!("locate oracle binary: {e}"))?.display().to_string(),
            "sha256": binary_sha256,
            "bytes": binary_bytes,
        },
        "base_commit": PINNED_BASE,
        "commit": commit,
        "config": config_identity(&config, &metadata_sha256),
        "gpu": {
            "arch": &gpu.arch,
            "device_id": gpu.device_id,
        },
    });
    hipfire_arch_qwen35::qwen35::oracle::begin(&args.out, identity)?;
    let weights = load_weights(&model, &config, &mut gpu)?;
    hipfire_arch_qwen35::qwen35::oracle::record_ownership(&gpu, &weights)?;
    run_sequence(
        &mut gpu,
        &weights,
        &config,
        &tokens,
        args.prefill_chunk,
        args.decode_positions,
        "fresh-state",
        false,
    )?;
    run_sequence(
        &mut gpu,
        &weights,
        &config,
        &tokens,
        args.prefill_chunk,
        args.decode_positions,
        "warm-prefix",
        true,
    )?;
    hipfire_arch_qwen35::qwen35::oracle::compare_sequences("fresh-state", "warm-prefix")?;
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    run_lifecycle_cycles(&mut gpu, &model, &config, &tokens, args.prefill_chunk)?;
    let report = hipfire_arch_qwen35::qwen35::oracle::finish(json!({
        "reset_unload_reload_cycles": LIFECYCLE_CYCLES,
        "warm_prefix_compared_with_fresh_state": true,
        "physical_collectives": 0,
    }))?;
    if let Some(compare) = args.compare {
        hipfire_arch_qwen35::qwen35::oracle::compare(&args.out, &compare)?;
    }
    Ok(report)
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(report) => {
            println!("{}", report.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
