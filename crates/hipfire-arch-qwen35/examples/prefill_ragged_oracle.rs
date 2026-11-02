// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Shared byte-identical prefill oracle (test-only; not a performance bench).
//!
//! prefill_ragged_oracle --model M.hfq --n ROWS [--max-seq N]
//!     [--prompt-file ids.u32le] [--logits-out FILE] [--hidden-out FILE]
//!     [--decode-logits-out FILE] [--tokens-out FILE]
//!
//! Loads through the production `load_qwen35_bundle` (PP=1, no MTP / drafter /
//! vision, automatic KV mode, production DeltaNet state quant), so the KV
//! tier, weight route and widened-prefill admission are exactly what serving
//! resolves. Route and chunk policy follow the process environment and
//! `prefill.*` config only: set `HIPFIRE_IU4_PREFILL`, `HIPFIRE_PREFILL_CHUNK_ROWS`,
//! `HIPFIRE_PREFILL_MAX_BATCH`, ... before each fresh process; this oracle
//! adds no chunk or route knob of its own.
//!
//! Prompt ids are the daemon `bench_prefill` synthetic ids, `10 + i % 1000`
//! for `i in 0..ROWS` (`--n 16059` plans 8192 + 7867 on a widened gfx11 arch;
//! `--n 32055` follows the same production planner). `--prompt-file` replaces
//! them with raw u32 little-endian ids (then `--n`, if given, must equal the
//! file's id count). `--max-seq` defaults to `n + 32`, the bench headroom, and
//! must be >= `n + 31`.
//!
//! The prompt runs through one `qwen35::forward_prefill_batch` call from a
//! fresh state (the exact `bench_prefill` carrier call), so the production
//! chunk planner, including a ragged final chunk, is what executes. The
//! final-position logits (`scratch.logits`) and the last prefill row's
//! post-output-norm hidden (`scratch.tmp`, the buffer the head reads) are
//! read back before any decode. Exactly 32 greedy tokens follow: token 0 is
//! the argmax of the prefill logits, tokens 1..32 come from 31 production
//! `qwen35::forward_scratch` decode steps (host argmax over the full f32
//! logits, first-max on ties).
//!
//! Stdout is one JSON object: sha256 of the raw native-f32 little-endian
//! final-position logits (full vocab) and last hidden row, the 32 tokens and
//! the sha256 of their u32 little-endian bytes, plus route receipts (arch,
//! KV tier, chunk ceiling, recorded HIPFIRE_*/ROCR_*/HIP_* env). Files, when
//! requested, hold the raw bytes: `--logits-out` the final-position vocab
//! logits, `--hidden-out` the last hidden row, `--decode-logits-out` the 32
//! per-token logits vectors (first = prefill logits), `--tokens-out` the 32
//! u32le tokens. Existing output files are never overwritten. A non-finite
//! logit is an error. No tolerance and no coherence judgement: compare the
//! hashes (or raw files) between routes.

use hipfire_arch_qwen35::qwen35::{self, Qwen35Config};
use hipfire_arch_qwen35::{free_qwen35_bundle, load_qwen35_bundle, Qwen35Bundle};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const SCHEMA: &str = "hipfire.prefill_ragged_oracle.v1";
/// Greedy tokens emitted: one from the prefill logits plus 31 decode steps.
const GREEDY_TOKENS: usize = 32;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

struct Args(BTreeMap<String, String>);

impl Args {
    fn parse(raw: &[String]) -> Result<Self> {
        let mut map = BTreeMap::new();
        let mut it = raw.iter();
        while let Some(key) = it.next() {
            let key = key
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {key:?}"))?;
            let value = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            if map.insert(key.to_owned(), value.clone()).is_some() {
                return Err(format!("--{key} given twice").into());
            }
        }
        const KNOWN: [&str; 8] = [
            "model",
            "n",
            "max-seq",
            "prompt-file",
            "logits-out",
            "hidden-out",
            "decode-logits-out",
            "tokens-out",
        ];
        for k in map.keys() {
            if !KNOWN.contains(&k.as_str()) {
                return Err(format!("unknown option --{k}").into());
            }
        }
        Ok(Self(map))
    }
    fn req(&self, key: &str) -> Result<&str> {
        self.opt(key)
            .ok_or_else(|| format!("missing --{key}").into())
    }
    fn opt(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
}

/// Create `path` exclusively (never overwrite) and write `bytes`.
fn write_new(path: &str, bytes: &[u8]) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

/// Refuse up front so a long run cannot be lost to an existing output path.
fn require_absent(path: Option<&str>) -> Result<()> {
    if let Some(p) = path {
        if Path::new(p).exists() {
            return Err(format!("output {p} already exists").into());
        }
    }
    Ok(())
}

/// Synchronize, then read `bytes` of `tensor` (f32 elements) back raw.
fn download(
    gpu: &Gpu,
    tensor: &GpuTensor,
    elems: usize,
    what: &str,
) -> Result<(Vec<f32>, Vec<u8>)> {
    gpu.hip.device_synchronize()?;
    let bytes = elems * 4;
    if bytes > tensor.buf.size() {
        return Err(format!(
            "{what}: {bytes} bytes exceeds allocation {}",
            tensor.buf.size()
        )
        .into());
    }
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &tensor.buf)?;
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(format!("non-finite {what} at index {i}").into());
    }
    Ok((values, raw))
}

fn recorded_env() -> Value {
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("HIPFIRE_") || k.starts_with("ROCR_") || k.starts_with("HIP_")
        })
        .collect();
    vars.sort();
    let mut env = Map::new();
    for (k, v) in vars {
        env.insert(k, Value::String(v));
    }
    Value::Object(env)
}

/// Production carrier load at an explicit `max_seq`, PP=1, no MTP/drafter/
/// vision, automatic KV mode: the same `LoadCtx` the daemon builds for a
/// default load (`kv_mode_override` / state-quant overrides unset).
fn load_bundle(model: &str, max_seq: usize, gpu: &mut Gpu) -> Result<Qwen35Bundle> {
    let cask = CaskConfig::default();
    let qwen_default_q8 = !matches!(
        hipfire_config::developer_var("HIPFIRE_QWEN_KV_DEFAULT_Q8")
            .ok()
            .as_deref(),
        Some("0")
    );
    let mut lc = LoadCtx {
        path: model,
        max_seq,
        sequence: None,
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".to_string(),
        kv_mode_override: None,
        kv_k_override: None,
        kv_v_override: None,
        qwen_default_q8,
        kv_backend: KvBackend::default(),
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        pp_layer_counts: None,
        admitted_mesh: None,
        band_placement: None,
        spec: SpecLoadCfg::default(),
        gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
        batch_request: Default::default(),
    };
    Ok(load_qwen35_bundle(ModelSource::from_path(model)?, &mut lc)
        .map_err(|e| format!("load_qwen35_bundle: {e}"))?)
}

fn prompt_ids(args: &Args) -> Result<Vec<u32>> {
    let n_arg: Option<usize> = args.opt("n").map(str::parse).transpose()?;
    match args.opt("prompt-file") {
        Some(path) => {
            let bytes = fs::read(path)?;
            if bytes.is_empty() || bytes.len() % 4 != 0 {
                return Err(format!(
                    "{path}: {} bytes is not a non-empty u32le id array",
                    bytes.len()
                )
                .into());
            }
            let ids: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if let Some(n) = n_arg {
                if n != ids.len() {
                    return Err(format!("--n {n} != {} ids in {path}", ids.len()).into());
                }
            }
            Ok(ids)
        }
        None => {
            let n = n_arg.ok_or("missing --n (or --prompt-file)")?;
            if n == 0 || n > u32::MAX as usize {
                return Err("--n must be in 1..=u32::MAX".into());
            }
            // Daemon `bench_prefill` synthetic ids.
            Ok((0..n as u32).map(|i| 10 + (i % 1000)).collect())
        }
    }
}

fn route_receipt(gpu: &Gpu, config: &Qwen35Config, bundle: &Qwen35Bundle, max_seq: usize) -> Value {
    let kv = &bundle.kv_cache;
    json!({
        "arch": gpu.arch.as_str(),
        "max_seq": max_seq,
        "kv_fp8": kv.quant_fp8,
        "kv_vmm": kv.uses_vmm_backend(),
        "kv_physical_cap": kv.physical_cap,
        "dn_quant": format!("{:?}", bundle.dn_state.quant),
        "dn_ef": bundle.dn_state.s_ef_residual.len() == bundle.dn_state.s_matrices.len()
            && !bundle.dn_state.s_ef_residual.is_empty(),
        "n_layers": config.n_layers,
        "vocab": config.vocab_size,
        "dim": config.dim,
        "graph_ar": gpu.flags.graph_ar,
        "graph_forward": gpu.flags.graph_forward,
        "prefill_chunk_rows_requested": qwen35::prefill::prefill_chunk_rows_requested(gpu),
        "iu4_prefill_enabled": gpu.flags.iu4_prefill_enabled(),
    })
}

/// Post-init device identity gate, the same check the daemon runs in
/// `reserve_and_verify_gpus` (`hipfire-daemon/src/main.rs`): observe every
/// HIP-visible logical device from the live runtime of the initialized `Gpu`
/// and require `hipfire_config::devices::active_devices()` (the cards
/// `HIPFIRE_DEVICES` resolved) to `verify_visible` them (arch + PCI bus ID).
/// Only on success is the daemon's `[devices] verified ...` line printed.
/// Errors when no `HIPFIRE_DEVICES` selection is active: nothing to verify.
fn verify_devices(gpu: &Gpu) -> Result<()> {
    use hipfire_config::devices::{DeviceRoot, IdentitySource, ObservedDevice};
    let active = hipfire_config::devices::active_devices().ok_or(
        "no active HIPFIRE_DEVICES selection: set HIPFIRE_DEVICES so the GPU can be verified",
    )?;
    let with_uuid = DeviceRoot::system().identity_source() == IdentitySource::Hip;
    let count = gpu.hip.device_count()?;
    if count < 1 {
        return Err("no visible GPUs to verify".into());
    }
    let observed = (0..count)
        .map(|id| -> Result<ObservedDevice> {
            Ok(ObservedDevice {
                logical: id as usize,
                arch: gpu.hip.get_arch(id)?,
                pci_bus_id: gpu.hip.device_pci_bus_id(id)?,
                uuid: if with_uuid {
                    Some(gpu.hip.device_uuid(id)?)
                } else {
                    None
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    active.verify_visible(&observed)?;
    for (seen, device) in observed.iter().zip(&active.devices) {
        eprintln!(
            "[devices] verified logical {}: HIP {} {} = index {} {} rocr={}",
            seen.logical,
            seen.arch,
            seen.pci_bus_id,
            device.index,
            device.lock_identity(),
            device.rocr_selector()
        );
    }
    Ok(())
}

fn run(raw: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let args = Args::parse(raw)?;
    let model = args.req("model")?.to_owned();
    let tokens = prompt_ids(&args)?;
    let n = tokens.len();
    let max_seq: usize = match args.opt("max-seq") {
        Some(v) => v.parse()?,
        None => n + 32,
    };
    // Prompt rows plus 31 fed decode tokens address positions < n + 31.
    if n + GREEDY_TOKENS - 1 > max_seq {
        return Err(format!(
            "n {n} + {} decode positions exceeds max_seq {max_seq}",
            GREEDY_TOKENS - 1
        )
        .into());
    }
    let logits_out = args.opt("logits-out");
    let hidden_out = args.opt("hidden-out");
    let decode_logits_out = args.opt("decode-logits-out");
    let tokens_out = args.opt("tokens-out");
    for p in [logits_out, hidden_out, decode_logits_out, tokens_out] {
        require_absent(p)?;
    }
    let prompt_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();

    let mut gpu = Gpu::init()?;
    verify_devices(&gpu)?;
    let bundle = load_bundle(&model, max_seq, &mut gpu)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut bundle = bundle;
    let config = bundle.config.clone();
    let vocab = config.vocab_size;
    let dim = config.dim;
    if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= vocab) {
        return Err(format!("prompt id {bad} >= vocab {vocab}").into());
    }
    let route = route_receipt(&gpu, &config, &bundle, max_seq);
    // A CASK/eviction-bounded cache can have physical_cap < max_seq; this oracle
    // writes n + 31 KV positions with no eviction.
    if bundle.kv_cache.physical_cap < n + GREEDY_TOKENS - 1 {
        return Err(format!(
            "kv physical_cap {} < n {n} + {} decode positions",
            bundle.kv_cache.physical_cap,
            GREEDY_TOKENS - 1
        )
        .into());
    }
    let admitted_ceiling = qwen35::ordinary_prefill_chunk_limit(
        &gpu,
        &bundle.weights,
        &config,
        &bundle.dn_state,
        &bundle.kv_cache,
        None,
    )
    .ok();

    // The exact `bench_prefill` carrier call (production chunk planner).
    gpu.hip.device_synchronize()?;
    let t_prefill = Instant::now();
    qwen35::forward_prefill_batch(
        &mut gpu,
        &bundle.weights,
        &config,
        &tokens,
        0,
        &mut bundle.kv_cache,
        &mut bundle.dn_state,
        &bundle.scratch,
        None,
        None,
        None,
        None,
    )?;
    let (values, logits_raw) =
        download(&gpu, &bundle.scratch.logits, vocab, "final-position logits")?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    // `scratch.tmp` is the last row's post-output-norm hidden (the head's input),
    // left by the legacy last-token path of the final chunk.
    let (_, hidden_raw) = download(&gpu, &bundle.scratch.tmp, dim, "last hidden row")?;

    let mut generated: Vec<u32> = Vec::with_capacity(GREEDY_TOKENS);
    let mut decode_logits: Vec<u8> = Vec::new();
    let keep_decode = decode_logits_out.is_some();
    if keep_decode {
        decode_logits.extend_from_slice(&logits_raw);
    }
    generated.push(llama::argmax(&values));
    let t_decode = Instant::now();
    for step in 1..GREEDY_TOKENS {
        qwen35::forward_scratch(
            &mut gpu,
            &bundle.weights,
            &config,
            generated[step - 1],
            n + step - 1,
            &mut bundle.kv_cache,
            &mut bundle.dn_state,
            &bundle.scratch,
        )?;
        let (v, r) = download(&gpu, &bundle.scratch.logits, vocab, "decode logits")?;
        if keep_decode {
            decode_logits.extend_from_slice(&r);
        }
        generated.push(llama::argmax(&v));
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(generated.len(), GREEDY_TOKENS);
    let token_bytes: Vec<u8> = generated.iter().flat_map(|t| t.to_le_bytes()).collect();

    if let Some(p) = logits_out {
        write_new(p, &logits_raw)?;
    }
    if let Some(p) = hidden_out {
        write_new(p, &hidden_raw)?;
    }
    if let Some(p) = decode_logits_out {
        write_new(p, &decode_logits)?;
    }
    if let Some(p) = tokens_out {
        write_new(p, &token_bytes)?;
    }

    let free_result = free_qwen35_bundle(bundle, &mut gpu);
    let report = json!({
        "schema": SCHEMA,
        "model": model,
        "n": n,
        "prompt_ids_sha256": sha256(&prompt_bytes),
        "prompt_source": args.opt("prompt-file").unwrap_or("synthetic 10 + i % 1000"),
        "route": route,
        "admitted_chunk_ceiling": admitted_ceiling,
        "env": recorded_env(),
        "logits": { "elems": vocab, "bytes": logits_raw.len(), "sha256": sha256(&logits_raw) },
        "hidden": { "elems": dim, "bytes": hidden_raw.len(), "sha256": sha256(&hidden_raw) },
        "decode_logits_sha256": keep_decode.then(|| sha256(&decode_logits)),
        "tokens": generated,
        "tokens_sha256": sha256(&token_bytes),
        "timing_ms": { "load": load_ms, "prefill": prefill_ms, "decode": decode_ms },
        "free_ok": free_result.is_ok(),
        "free_error": free_result.as_ref().err(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run(&args)
}
