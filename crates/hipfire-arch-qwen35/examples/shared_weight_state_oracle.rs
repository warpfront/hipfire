// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Test-only G3 oracle for the shared-weight host (WEIGHT-HOST-DESIGN §8 G3).
//! Not a serving or performance acceptance test.
//!
//! run --model M --prompt FILE --max-seq N --out NEW_DIR [--prompt-tokens N]
//!     [--gen 256] [--arm NAME] [--rep N] [--barrier DIR] [--keep-raw]
//! compare BASE_DIR CANDIDATE_DIR
//! weights --model M --out NEW_DIR [--max-seq N] [--keep-raw] [--hold DIR]   (G1)
//! compare-weights BASE_DIR CANDIDATE_DIR
//!
//! `run` loads through the production `load_qwen35_bundle` (explicit max_seq,
//! PP=1, no MTP/drafter/vision). Private vs imported weights is selected only
//! by the process environment (`HIPFIRE_WEIGHT_HOST`, `HIPFIRE_WEIGHT_HOST_SOCKET`);
//! eager vs hipGraph only by `HIPFIRE_AR_GRAPH`. Set those before each fresh
//! process. It prefills the pinned prompt, then decodes `gen` greedy tokens
//! (host argmax over the full f32 logits). It records the SHA-256 of every
//! logits vector, the token IDs, the decoded text, and the DeltaNet state
//! (codes, scales, EF, conv) plus addressed KV bytes at matched snapshot
//! points. Teardown is `free_qwen35_bundle`, so a lease Release is the real one.
//!
//! `compare` reports every differing item. Zero changed bytes is the only pass;
//! there is no tolerance.

use hipfire_arch_qwen35::qwen35::{self, LayerType, Qwen35Config};
use hipfire_arch_qwen35::{free_qwen35_bundle, load_qwen35_bundle, shared_weights, Qwen35Bundle};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SpecLoadCfg};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const SCHEMA: &str = "hipfire.g3.oracle.v1";
/// Decode steps (count of completed `forward_scratch` calls) that snapshot
/// state; `prefill` is always taken. Steps beyond `gen - 1` are skipped.
const SNAPSHOT_STEPS: [usize; 4] = [1, 64, 128, 255];
const BARRIER_TIMEOUT: Duration = Duration::from_secs(600);

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

struct Args {
    map: BTreeMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self> {
        let mut map = BTreeMap::new();
        let mut flags = Vec::new();
        let mut it = raw.iter();
        while let Some(key) = it.next() {
            let key = key
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {key:?}"))?;
            if key == "keep-raw" {
                flags.push(key.to_owned());
                continue;
            }
            let value = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
            if map.insert(key.to_owned(), value.clone()).is_some() {
                return Err(format!("--{key} given twice").into());
            }
        }
        Ok(Self { map, flags })
    }
    fn req(&self, key: &str) -> Result<&str> {
        self.map
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing --{key}").into())
    }
    fn opt(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(String::as_str)
    }
    fn usize_or(&self, key: &str, default: usize) -> Result<usize> {
        self.opt(key).map_or(Ok(default), |v| Ok(v.parse()?))
    }
    fn flag(&self, key: &str) -> bool {
        self.flags.iter().any(|f| f == key)
    }
}

struct Recorder<'a> {
    dir: &'a Path,
    keep_raw: bool,
    raw_count: usize,
}

impl Recorder<'_> {
    /// Download exactly `bytes` of `tensor`, hash them, optionally keep raw.
    fn item(&mut self, gpu: &Gpu, tensor: &GpuTensor, bytes: usize) -> Result<Value> {
        if bytes > tensor.buf.size() {
            return Err(format!(
                "dump of {bytes} bytes exceeds allocation of {}",
                tensor.buf.size()
            )
            .into());
        }
        let mut raw = vec![0u8; bytes];
        gpu.hip.memcpy_dtoh(&mut raw, &tensor.buf)?;
        let digest = sha256(&raw);
        let file = if self.keep_raw {
            let name = format!("raw/{:05}.bin", self.raw_count);
            self.raw_count += 1;
            fs::write(self.dir.join(&name), &raw)?;
            Value::String(name)
        } else {
            Value::Null
        };
        Ok(json!({ "bytes": bytes, "sha256": digest, "raw": file }))
    }
}

fn snapshot(
    rec: &mut Recorder,
    gpu: &Gpu,
    config: &Qwen35Config,
    bundle: &Qwen35Bundle,
    positions: usize,
    kv_row_bytes: usize,
) -> Result<Value> {
    gpu.hip.device_synchronize()?;
    let dn = &bundle.dn_state;
    let kv = &bundle.kv_cache;
    let mut out = Map::new();
    let state_n = config.linear_num_value_heads * config.linear_value_head_dim.pow(2);
    let addressed = positions
        .checked_mul(kv_row_bytes)
        .ok_or("addressed KV bytes overflow")?;
    let mut la = 0;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        let p = format!("layer_{layer:02}");
        if *ty == LayerType::LinearAttention {
            out.insert(
                format!("{p}/state_codes"),
                rec.item(gpu, &dn.s_matrices[la], state_n)?,
            );
            out.insert(
                format!("{p}/state_scales"),
                rec.item(gpu, &dn.s_scales[la], dn.s_scales[la].numel() * 4)?,
            );
            out.insert(
                format!("{p}/ef"),
                rec.item(gpu, &dn.s_ef_residual[la], state_n * 2)?,
            );
            out.insert(
                format!("{p}/conv"),
                rec.item(gpu, &dn.conv_states[la], dn.conv_states[la].numel() * 4)?,
            );
            la += 1;
        } else {
            out.insert(
                format!("{p}/key"),
                rec.item(gpu, &kv.k_gpu[layer], addressed)?,
            );
            out.insert(
                format!("{p}/value"),
                rec.item(gpu, &kv.v_gpu[layer], addressed)?,
            );
        }
    }
    Ok(Value::Object(out))
}

fn download_logits(gpu: &Gpu, scratch: &GpuTensor, vocab: usize) -> Result<(Vec<f32>, Vec<u8>)> {
    gpu.hip.device_synchronize()?;
    let mut raw = vec![0u8; vocab * 4];
    gpu.hip.memcpy_dtoh(&mut raw, &scratch.buf)?;
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(format!("non-finite logit at index {i}").into());
    }
    Ok((values, raw))
}

fn mem_free(gpu: &Gpu) -> Result<(usize, usize)> {
    Ok(gpu.device_mem_info()?)
}

fn barrier(dir: &Path, t0: Instant) -> Result<Value> {
    let ready = dir.join(format!("ready.{}", std::process::id()));
    fs::write(&ready, b"")?;
    let ready_ms = ms(t0);
    let go = dir.join("go");
    let deadline = Instant::now() + BARRIER_TIMEOUT;
    while !go.exists() {
        if Instant::now() > deadline {
            return Err(format!("barrier {} timed out", go.display()).into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(json!({ "dir": dir.display().to_string(), "ready_ms": ready_ms, "go_ms": ms(t0) }))
}

fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

fn recorded_env() -> Value {
    let mut env = Map::new();
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("HIPFIRE_") || k.starts_with("ROCR_") || k.starts_with("HIP_")
        })
        .collect();
    vars.sort();
    for (k, v) in vars {
        env.insert(k, Value::String(v));
    }
    Value::Object(env)
}

fn run(raw: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let args = Args::parse(raw)?;
    let model = args.req("model")?.to_owned();
    let prompt_path = PathBuf::from(args.req("prompt")?);
    let max_seq: usize = args.req("max-seq")?.parse()?;
    let gen = args.usize_or("gen", 256)?;
    let rep = args.usize_or("rep", 0)?;
    let arm = args.opt("arm").unwrap_or("unnamed").to_owned();
    let out = PathBuf::from(args.req("out")?);
    let keep_raw = args.flag("keep-raw");
    if gen == 0 {
        return Err("--gen must be >= 1".into());
    }
    fs::create_dir(&out)?; // never overwrite an existing record
    if keep_raw {
        fs::create_dir(out.join("raw"))?;
    }

    let prompt_bytes = fs::read(&prompt_path)?;
    let prompt_text = std::str::from_utf8(&prompt_bytes)?;
    let hfq = HfqFile::open(Path::new(&model))?;
    let config = qwen35::config_from_hfq(&hfq)?;
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("{e:?}"))?;
    drop(hfq);
    let hfq_open_ms = ms(t0);
    let seed = tokenizer.encode(prompt_text);
    if seed.is_empty() {
        return Err("prompt tokenized to an empty sequence".into());
    }
    let tokens: Vec<u32> = match args.opt("prompt-tokens") {
        Some(n) => seed.iter().copied().cycle().take(n.parse()?).collect(),
        None => seed,
    };
    let prompt_len = tokens.len();
    // prompt + (gen - 1) fed decode tokens must address within max_seq.
    if prompt_len + gen > max_seq {
        return Err(format!("prompt {prompt_len} + gen {gen} exceeds max_seq {max_seq}").into());
    }
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();

    let mut gpu = Gpu::init()?;
    let (start_free, total) = mem_free(&gpu)?;
    let t_load = Instant::now();
    let bundle = load_bundle(&model, max_seq, &mut gpu)?;
    let bundle_load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let (after_load_free, _) = mem_free(&gpu)?;

    // Refuse any route other than the one G3 pins.
    let kv = &bundle.kv_cache;
    let dn = &bundle.dn_state;
    if !kv.uses_vmm_backend() || !kv.quant_fp8 {
        return Err(format!(
            "G3 pins VMM native-fp8 KV; got vmm={} fp8={}",
            kv.uses_vmm_backend(),
            kv.quant_fp8
        )
        .into());
    }
    if !matches!(dn.quant, qwen35::StateQuant::Q8) || dn.s_ef_residual.len() != dn.s_matrices.len()
    {
        return Err("G3 pins DeltaNet Q8 with EF residual".into());
    }
    if config.num_experts != 0 {
        return Err("G3 v1 is dense only".into());
    }
    let kv_row_bytes = kv.fp8_bytes_per_token()?;
    let route = json!({
        "kv_vmm": true, "kv_fp8": true, "kv_row_bytes": kv_row_bytes,
        "kv_physical_cap": kv.physical_cap, "dn_quant": "Q8", "dn_ef": true,
        "max_seq": max_seq, "graph_ar": gpu.flags.graph_ar,
        "graph_forward": gpu.flags.graph_forward,
    });

    let barrier_rec = match args.opt("barrier") {
        Some(dir) => barrier(Path::new(dir), t0)?,
        None => json!({ "dir": null, "ready_ms": null, "go_ms": null }),
    };

    let mut rec = Recorder {
        dir: &out,
        keep_raw,
        raw_count: 0,
    };
    let mut snapshots = Map::new();
    let mut logits_sha = Vec::with_capacity(gen);
    let mut generated = Vec::with_capacity(gen);

    let t_prefill = Instant::now();
    let prefill_start_ms = ms(t0);
    let mut bundle = bundle;
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
    let (values, raw) = download_logits(&gpu, &bundle.scratch.logits, config.vocab_size)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    let first_token_ms = ms(t0);
    logits_sha.push(sha256(&raw));
    if keep_raw {
        fs::write(out.join("raw/logits_prefill.bin"), &raw)?;
    }
    generated.push(llama::argmax(&values));
    snapshots.insert(
        "prefill".into(),
        snapshot(&mut rec, &gpu, &config, &bundle, prompt_len, kv_row_bytes)?,
    );

    let t_decode = Instant::now();
    for step in 1..gen {
        let pos = prompt_len + step - 1;
        qwen35::forward_scratch(
            &mut gpu,
            &bundle.weights,
            &config,
            generated[step - 1],
            pos,
            &mut bundle.kv_cache,
            &mut bundle.dn_state,
            &bundle.scratch,
        )?;
        let (values, raw) = download_logits(&gpu, &bundle.scratch.logits, config.vocab_size)?;
        logits_sha.push(sha256(&raw));
        if keep_raw && step + 1 == gen {
            fs::write(out.join("raw/logits_last.bin"), &raw)?;
        }
        generated.push(llama::argmax(&values));
        if SNAPSHOT_STEPS.contains(&step) {
            snapshots.insert(
                format!("step_{step}"),
                snapshot(&mut rec, &gpu, &config, &bundle, pos + 1, kv_row_bytes)?,
            );
        }
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    let (after_decode_free, _) = mem_free(&gpu)?;

    let text = tokenizer.decode(&generated);
    let t_free = Instant::now();
    let free_result = free_qwen35_bundle(bundle, &mut gpu);
    let free_ms = t_free.elapsed().as_secs_f64() * 1e3;
    let (end_free, _) = mem_free(&gpu)?;

    let record = json!({
        "schema": SCHEMA,
        "build": {
            "exe": std::env::current_exe()?.display().to_string(),
            "build_id": std::env::var("HIPFIRE_G3_BUILD_ID").ok(),
        },
        "arm": arm,
        "rep": rep,
        "pid": std::process::id(),
        "model": { "path": model, "vocab": config.vocab_size, "n_layers": config.n_layers },
        "card": { "arch": gpu.arch.clone(), "devices_env": std::env::var("HIPFIRE_DEVICES").ok() },
        "env": recorded_env(),
        "route": route,
        "prompt": {
            "file": prompt_path.display().to_string(),
            "sha256": sha256(&prompt_bytes),
            "tokens": prompt_len,
            "token_ids_sha256": sha256(&token_bytes),
        },
        "gen": gen,
        "barrier": barrier_rec,
        "phases_ms": {
            "hfq_open": hfq_open_ms,
            "bundle_load": bundle_load_ms,
            "prefill_start": prefill_start_ms,
            "prefill": prefill_ms,
            "first_token": first_token_ms,
            "decode_total": decode_ms,
            "free": free_ms,
        },
        "mem": {
            "total": total, "start_free": start_free, "after_load_free": after_load_free,
            "after_decode_free": after_decode_free, "end_free": end_free,
        },
        "logits_sha256": logits_sha,
        "tokens": generated,
        "text": text,
        "text_sha256": sha256(text.as_bytes()),
        "snapshots": Value::Object(snapshots),
        "free_ok": free_result.is_ok(),
        "free_error": free_result.as_ref().err(),
    });
    fs::write(out.join("record.json"), serde_json::to_vec_pretty(&record)?)?;
    eprintln!("G3_RECORD {}", out.join("record.json").display());
    free_result.map_err(|e| format!("free_qwen35_bundle: {e}").into())
}

fn load_record(dir: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(dir.join("record.json"))?)?)
}

/// First differing byte offset of two kept raw files, `None` when equal.
fn raw_first_diff(a: &Path, b: &Path) -> Result<Option<usize>> {
    let (x, y) = (fs::read(a)?, fs::read(b)?);
    Ok(x.iter()
        .zip(&y)
        .position(|(p, q)| p != q)
        .or_else(|| (x.len() != y.len()).then(|| x.len().min(y.len()))))
}

fn compare(raw: &[String]) -> Result<()> {
    let [base_dir, cand_dir] = raw else {
        return Err("compare BASE_DIR CANDIDATE_DIR".into());
    };
    let (base_dir, cand_dir) = (Path::new(base_dir), Path::new(cand_dir));
    let (a, b) = (load_record(base_dir)?, load_record(cand_dir)?);
    for r in [&a, &b] {
        if r["schema"] != SCHEMA {
            return Err(format!("unexpected record schema {}", r["schema"]).into());
        }
    }
    let mut diffs = Vec::new();
    for key in ["gen", "text_sha256"] {
        if a[key] != b[key] {
            diffs.push(json!({ "item": key, "base": a[key], "candidate": b[key] }));
        }
    }
    for key in ["token_ids_sha256", "tokens"] {
        if a["prompt"][key] != b["prompt"][key] {
            diffs.push(json!({ "item": format!("prompt/{key}") }));
        }
    }
    let seq_diff = |name: &str, x: &Value, y: &Value| -> Option<Value> {
        let (x, y) = (x.as_array()?, y.as_array()?);
        let differing: Vec<usize> = (0..x.len().max(y.len()))
            .filter(|&i| x.get(i) != y.get(i))
            .collect();
        (!differing.is_empty()).then(|| {
            json!({ "item": name, "first": differing[0], "count": differing.len(),
                    "base_len": x.len(), "candidate_len": y.len() })
        })
    };
    diffs.extend(seq_diff(
        "logits_sha256",
        &a["logits_sha256"],
        &b["logits_sha256"],
    ));
    diffs.extend(seq_diff("tokens", &a["tokens"], &b["tokens"]));
    let empty = Map::new();
    let sa = a["snapshots"].as_object().unwrap_or(&empty);
    let sb = b["snapshots"].as_object().unwrap_or(&empty);
    let stages: std::collections::BTreeSet<&String> = sa.keys().chain(sb.keys()).collect();
    for stage in stages {
        let (Some(xa), Some(xb)) = (
            sa.get(stage).and_then(Value::as_object),
            sb.get(stage).and_then(Value::as_object),
        ) else {
            diffs.push(json!({ "item": format!("snapshot/{stage}"), "missing": true }));
            continue;
        };
        let names: std::collections::BTreeSet<&String> = xa.keys().chain(xb.keys()).collect();
        for name in names {
            let (ea, eb) = (xa.get(name), xb.get(name));
            let same = matches!((ea, eb), (Some(p), Some(q))
                if p["sha256"] == q["sha256"] && p["bytes"] == q["bytes"]);
            if same {
                continue;
            }
            let mut d = json!({ "item": format!("snapshot/{stage}/{name}") });
            if let (Some(p), Some(q)) = (ea, eb) {
                if let (Some(rp), Some(rq)) = (p["raw"].as_str(), q["raw"].as_str()) {
                    d["first_diff_byte"] =
                        json!(raw_first_diff(&base_dir.join(rp), &cand_dir.join(rq))?);
                }
            }
            diffs.push(d);
        }
    }
    for f in ["raw/logits_prefill.bin", "raw/logits_last.bin"] {
        let (p, q) = (base_dir.join(f), cand_dir.join(f));
        if p.exists() && q.exists() {
            if let Some(off) = raw_first_diff(&p, &q)? {
                diffs.push(json!({ "item": f, "first_diff_byte": off }));
            }
        }
    }
    let equal = diffs.is_empty();
    let report = json!({
        "schema": "hipfire.g3.compare.v1",
        "base": base_dir.display().to_string(),
        "candidate": cand_dir.display().to_string(),
        "equal": equal,
        "diffs": diffs,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if equal {
        Ok(())
    } else {
        Err("MISMATCH".into())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("compare") => compare(&args[1..]),
        Some("weights") => weights(&args[1..]),
        Some("compare-weights") => compare_weights(&args[1..]),
        _ => Err(
            "usage: run --model M --prompt FILE --max-seq N --out DIR [...] | compare BASE CAND \
                  | weights --model M --out DIR [--max-seq N] | compare-weights BASE CAND"
                .into(),
        ),
    }
}

/// Production carrier load at an explicit (non-automatic) `max_seq`, PP=1,
/// no MTP/drafter/vision. The weight route follows the environment.
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

/// Start time of this process in clock ticks since boot (`/proc/self/stat`
/// field 22), so a parent can pin pid identity.
fn proc_starttime() -> Option<u64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    stat.rsplit_once(')')?
        .1
        .split_whitespace()
        .nth(19)?
        .parse()
        .ok()
}

/// G1: hash every visited dense tensor's accessible bytes, keyed by stable role.
/// `--keep-raw` also writes each role's bytes for a literal byte compare.
/// `--hold DIR` keeps the weights mapped after readback: writes
/// `DIR/held.<pid>` (with the partial record) and waits for `DIR/release`
/// before freeing, so a parent can sample live fdinfo and start a second
/// importer while this one is still mapped.
fn weights(raw: &[String]) -> Result<()> {
    let args = Args::parse(raw)?;
    let model = args.req("model")?.to_owned();
    let max_seq: usize = args.usize_or("max-seq", 512)?;
    let out = PathBuf::from(args.req("out")?);
    let keep_raw = args.flag("keep-raw");
    fs::create_dir(&out)?;
    if keep_raw {
        fs::create_dir(out.join("raw"))?;
    }
    let mut gpu = Gpu::init()?;
    let (start_free, total) = mem_free(&gpu)?;
    let t = Instant::now();
    let bundle = load_bundle(&model, max_seq, &mut gpu)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let (after_load_free, _) = mem_free(&gpu)?;
    let route = match std::env::var("HIPFIRE_WEIGHT_HOST").as_deref() {
        Ok("1") => "import",
        _ => "private",
    };
    let typed = shared_weights::dense_typed_tensor_lengths(&bundle.weights, &bundle.config, &gpu)
        .map_err(|e| format!("dense_typed_tensor_lengths: {e}"))?;
    let mut tensors = Map::new();
    let (mut unique, mut unique_typed, mut unique_alloc, mut slack_total) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut tail_checked, mut tail_nonzero) = (0usize, 0usize);
    {
        let gpu_ref = &gpu;
        let err = |m: String| hip_bridge::HipError::new(0, &m);
        let mut visit = |role: &str, tensor: &GpuTensor| -> hip_bridge::HipResult<()> {
            let typed_bytes = *typed
                .get(role)
                .ok_or_else(|| err(format!("{role}: no typed length")))?;
            let allocated = tensor.buf.size();
            if typed_bytes > allocated {
                return Err(err(format!(
                    "{role}: typed {typed_bytes} > allocated {allocated}"
                )));
            }
            let mut host = vec![0u8; typed_bytes];
            gpu_ref.hip.memcpy_dtoh(&mut host, &tensor.buf)?;
            let borrowed = tensor.buf.is_borrowed();
            if !borrowed {
                unique += 1;
                unique_typed += typed_bytes;
                unique_alloc += allocated;
                slack_total += allocated - typed_bytes;
            }
            // Imported owners: the registered mapped extent must be the frozen
            // physical size, and [typed, physical) must read back all zero.
            let tail = if route == "import" && !borrowed {
                let mapped = gpu_ref.vmm_mapped_bytes(tensor).ok_or_else(|| {
                    err(format!(
                        "{role}: imported owner is not a registered VMM arena"
                    ))
                })?;
                let granularity = gpu_ref
                    .vmm_granularity(tensor)
                    .ok_or_else(|| err(format!("{role}: no VMM granularity")))?;
                let expected =
                    shared_weights::expected_physical_bytes(typed_bytes as u64, granularity as u64);
                if expected != Some(mapped as u64) {
                    return Err(err(format!(
                        "{role}: mapped {mapped} != expected physical {expected:?} (typed {typed_bytes}, granularity {granularity})"
                    )));
                }
                let tail_len = mapped - typed_bytes;
                let mut bytes = vec![0u8; tail_len];
                // SAFETY: [ptr+typed, ptr+mapped) lies inside this owner's live,
                // registered mapping; the borrowed view is dropped before the owner.
                let view = unsafe {
                    hip_bridge::DeviceBuffer::from_raw(
                        tensor.buf.as_ptr().wrapping_byte_add(typed_bytes),
                        tail_len,
                    )
                };
                gpu_ref.hip.memcpy_dtoh(&mut bytes, &view)?;
                drop(view);
                let first_nonzero = bytes.iter().position(|&b| b != 0);
                tail_checked += 1;
                tail_nonzero += usize::from(first_nonzero.is_some());
                json!({ "checked": true, "physical_bytes": mapped, "granularity": granularity,
                        "zero": first_nonzero.is_none(), "first_nonzero": first_nonzero })
            } else {
                json!({ "checked": false, "physical_bytes": null, "zero": null, "first_nonzero": null })
            };
            let file = if keep_raw {
                let name = format!("raw/{:05}.bin", tensors.len());
                fs::write(out.join(&name), &host).map_err(|e| err(format!("{name}: {e}")))?;
                Value::String(name)
            } else {
                Value::Null
            };
            tensors.insert(
                role.to_owned(),
                json!({
                    "typed_bytes": typed_bytes, "allocated_bytes": allocated,
                    "slack_bytes": allocated - typed_bytes, "sha256": sha256(&host),
                    "borrowed": borrowed, "raw": file, "tail": tail,
                }),
            );
            Ok(())
        };
        bundle.weights.for_each_dense_tensor(&mut visit)?;
    }
    if tensors.len() != typed.len() {
        return Err(format!(
            "visited {} roles but typed map has {}",
            tensors.len(),
            typed.len()
        )
        .into());
    }
    let (after_readback_free, _) = mem_free(&gpu)?;
    let identity = json!({ "pid": std::process::id(), "starttime_ticks": proc_starttime() });
    let hold = match args.opt("hold") {
        Some(dir) => {
            let dir = Path::new(dir);
            let held = dir.join(format!("held.{}", std::process::id()));
            fs::write(
                &held,
                serde_json::to_vec(&json!({
                    "identity": identity, "unique_owners": unique,
                    "unique_typed_bytes": unique_typed, "after_readback_free": after_readback_free,
                    "tail_checked_owners": tail_checked, "tail_nonzero_owners": tail_nonzero,
                }))?,
            )?;
            eprintln!("G1_HELD {}", held.display());
            let release = dir.join("release");
            let deadline = Instant::now() + Duration::from_secs(3600);
            while !release.exists() {
                if Instant::now() > deadline {
                    return Err(format!(
                        "hold {} timed out (weights still mapped)",
                        release.display()
                    )
                    .into());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            json!({ "dir": dir.display().to_string(), "released": true })
        }
        None => Value::Null,
    };
    let free_result = free_qwen35_bundle(bundle, &mut gpu);
    let (end_free, _) = mem_free(&gpu)?;
    let record = json!({
        "schema": "hipfire.g1.weights.v2",
        "model": model,
        "identity": identity,
        "env": recorded_env(),
        "load_ms": load_ms,
        "mem": {
            "total": total, "start_free": start_free, "after_load_free": after_load_free,
            "after_readback_free": after_readback_free, "end_free": end_free,
        },
        "hold": hold,
        "route": route,
        "unique_owners": unique,
        "unique_typed_bytes": unique_typed,
        "unique_allocated_bytes": unique_alloc,
        "total_slack_bytes": slack_total,
        "tail_checked_owners": tail_checked,
        "tail_nonzero_owners": tail_nonzero,
        "tensors": Value::Object(tensors),
        "free_ok": free_result.is_ok(),
        "free_error": free_result.as_ref().err(),
    });
    fs::write(
        out.join("weights.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    eprintln!(
        "G1_WEIGHTS route={route} unique_owners={unique} unique_typed_bytes={unique_typed} \
         tail_checked={tail_checked} tail_nonzero={tail_nonzero}"
    );
    free_result.map_err(|e| format!("free_qwen35_bundle: {e}").into())
}

fn compare_weights(raw: &[String]) -> Result<()> {
    let [a, b] = raw else {
        return Err("compare-weights BASE_DIR CANDIDATE_DIR".into());
    };
    let (da, db) = (Path::new(a), Path::new(b));
    let read = |d: &Path| -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(d.join("weights.json"))?)?)
    };
    let (ra, rb) = (read(da)?, read(db)?);
    let empty = Map::new();
    let ta = ra["tensors"].as_object().unwrap_or(&empty);
    let tb = rb["tensors"].as_object().unwrap_or(&empty);
    let roles: std::collections::BTreeSet<&String> = ta.keys().chain(tb.keys()).collect();
    let mut diffs = Vec::new();
    let (mut raw_compared, mut raw_bytes, mut size_mismatch) = (0usize, 0usize, 0usize);
    for r in roles {
        let (x, y) = (ta.get(r), tb.get(r));
        let (Some(xv), Some(yv)) = (x, y) else {
            diffs.push(json!({ "role": r, "reason": "missing", "base": x, "candidate": y, "first_diff_byte": null }));
            continue;
        };
        let mut reasons = Vec::new();
        if xv["typed_bytes"] != yv["typed_bytes"] {
            size_mismatch += 1;
            reasons.push("typed_bytes");
        }
        if xv["sha256"] != yv["sha256"] {
            reasons.push("sha256");
        }
        // Literal byte compare whenever both sides kept raw bytes.
        let raw_diff = match (xv["raw"].as_str(), yv["raw"].as_str()) {
            (Some(p), Some(q)) => {
                raw_compared += 1;
                raw_bytes += xv["typed_bytes"].as_u64().unwrap_or(0) as usize;
                raw_first_diff(&da.join(p), &db.join(q))?
            }
            _ => None,
        };
        if raw_diff.is_some() {
            reasons.push("raw");
        }
        if yv["tail"]["checked"] == true && yv["tail"]["zero"] != true {
            reasons.push("tail");
        }
        for reason in reasons {
            diffs.push(
                json!({ "role": r, "reason": reason, "base": xv, "candidate": yv,
                               "first_diff_byte": raw_diff }),
            );
        }
    }
    let tail_nonzero = rb["tail_nonzero_owners"].as_u64().unwrap_or(0);
    let equal = diffs.is_empty() && tail_nonzero == 0;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "hipfire.g1.compare.v2", "equal": equal,
            "base_unique_owners": ra["unique_owners"], "candidate_unique_owners": rb["unique_owners"],
            "base_unique_typed_bytes": ra["unique_typed_bytes"],
            "candidate_unique_typed_bytes": rb["unique_typed_bytes"],
            "raw_roles_compared": raw_compared, "raw_bytes_compared": raw_bytes,
            "typed_size_mismatches": size_mismatch,
            "tail_nonzero_owners": tail_nonzero,
            "diffs": diffs,
        }))?
    );
    if equal {
        Ok(())
    } else {
        Err("MISMATCH".into())
    }
}
