// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Same-input D2/all-GPU slot measurements and artifact-backed CPU fixtures.
//!
//! HIPFIRE_OFFLOAD_EXEC=cpu cargo run --release -p hipfire-loader --features lab \
//!   --example qwen4_slab_d2_capture -- MODEL TEXT.txt --json OUT.json \
//!   --fixture-dir FIXTURES --prompt-rows 64 --steps 16 --layers first:4 \
//!   --fixture-experts 2,1,0 --samples-per-expert 4
//!
//! TEXT is a UTF-8 file, not a chat template. Context is capped at 4096 tokens;
//! no speculation/MTP is loaded. Layer numbers are zero-based trunk indices.
//! Fixture sample indices are zero-based records of 3 * 2560 little-endian f32s
//! in samples.bin, ordered x_rot, all-GPU slot, CPU-installed D2 slot.
//! Percentiles use nearest rank; the median averages the middle two values.
//! Measurements are not accuracy thresholds. Any error exits with status 2.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_dispatch::cpu_split::{
    SplitEndpoint, SplitParityConfig, SplitParityMode, SplitParityRecord, SPLIT_PARITY_MAX_RECORDS,
};
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

type R<T> = Result<T, String>;
const DIM: usize = 2560;
const MAX_SEQ: usize = 4096;
const INDEX_PREFIX_LEN: usize = 18_405_482;
const CODECS: [&str; 3] = ["E8P", "MQ4", "W3"];
const USAGE: &str = "usage: qwen4_slab_d2_capture MODEL TEXT.txt --json OUT.json \
    [--prompt-rows N (64)] [--steps S (16)] [--layers first:N|L,L,..|all (first:4)] \
    [--fixture-dir DIR] [--fixture-experts E8P_COUNT,MQ4_COUNT,W3_COUNT (2,1,0)] \
    [--samples-per-expert K (4)]";

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

struct Args {
    model: PathBuf,
    text: PathBuf,
    prompt_rows: usize,
    steps: usize,
    layers: String,
    fixture_dir: Option<PathBuf>,
    fixture_experts: [usize; 3],
    samples_per_expert: usize,
    json: PathBuf,
    identity: Vec<String>,
}

fn number(s: &str, name: &str) -> R<usize> {
    s.parse()
        .map_err(|_| format!("{name}: invalid nonnegative integer {s:?}"))
}

fn parse_args() -> R<Args> {
    let identity: Vec<String> = std::env::args().collect();
    if identity.len() < 3 {
        return Err(USAGE.into());
    }
    let mut args = Args {
        model: identity[1].clone().into(),
        text: identity[2].clone().into(),
        prompt_rows: 64,
        steps: 16,
        layers: "first:4".into(),
        fixture_dir: None,
        fixture_experts: [2, 1, 0],
        samples_per_expert: 4,
        json: PathBuf::new(),
        identity: identity.clone(),
    };
    let mut flags = identity[3..].iter();
    while let Some(flag) = flags.next() {
        let value = flags
            .next()
            .ok_or_else(|| format!("missing value for {flag}\n{USAGE}"))?;
        match flag.as_str() {
            "--prompt-rows" => args.prompt_rows = number(value, flag)?,
            "--steps" => args.steps = number(value, flag)?,
            "--layers" => args.layers = value.clone(),
            "--fixture-dir" => args.fixture_dir = Some(value.into()),
            "--fixture-experts" => {
                let counts = value
                    .split(',')
                    .map(|s| number(s, flag))
                    .collect::<R<Vec<_>>>()?;
                args.fixture_experts = counts.try_into().map_err(|_| {
                    "--fixture-experts needs exactly E8P_COUNT,MQ4_COUNT,W3_COUNT".to_string()
                })?;
            }
            "--samples-per-expert" => args.samples_per_expert = number(value, flag)?,
            "--json" => args.json = value.into(),
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    if args.json.as_os_str().is_empty() {
        return Err(format!("--json is required\n{USAGE}"));
    }
    if args.prompt_rows == 0 || args.steps == 0 || args.samples_per_expert == 0 {
        return Err("--prompt-rows, --steps and --samples-per-expert must be positive".into());
    }
    if args
        .prompt_rows
        .checked_add(args.steps)
        .is_none_or(|n| n > MAX_SEQ)
    {
        return Err(format!(
            "prompt rows + steps must not exceed max_seq={MAX_SEQ}"
        ));
    }
    Ok(args)
}

fn select_layers(selection: &str, armed: &[usize]) -> R<Vec<usize>> {
    let mut layers = if selection == "all" {
        armed.to_vec()
    } else if let Some(n) = selection.strip_prefix("first:") {
        armed
            .iter()
            .copied()
            .take(number(n, "--layers first:N")?)
            .collect()
    } else {
        selection
            .split(',')
            .map(|s| number(s, "--layers"))
            .collect::<R<Vec<_>>>()?
    };
    layers.sort_unstable();
    if layers.is_empty() || layers.windows(2).any(|w| w[0] == w[1]) {
        return Err("--layers must select a nonempty set without duplicates".into());
    }
    for layer in &layers {
        if !armed.contains(layer) {
            return Err(format!(
                "layer {layer} is not D2-armed; armed layers: {armed:?}"
            ));
        }
    }
    Ok(layers)
}

fn tensor_name(layer: usize, expert: usize, projection: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj")
}

fn codec(hfq: &HfqFile, layer: usize, expert: usize) -> R<&'static str> {
    let name = tensor_name(layer, expert, "gate_up");
    let info = hfq
        .find_tensor_info(&name)
        .ok_or_else(|| format!("missing artifact tensor {name}"))?;
    match info.quant_type {
        44 => Ok("MQ4"),
        135 => Ok("E8P"),
        131 => Ok("W3"),
        qt => Err(format!("{name}: unsupported expert quant type {qt}")),
    }
}

#[derive(Default)]
struct Stats {
    rel_l2: Vec<f64>,
    max_abs: f64,
    worst: Option<Value>,
    max_rel_l2: f64,
}

impl Stats {
    fn add(
        &mut self,
        rel_l2: f64,
        max_abs: f64,
        layer: usize,
        expert: usize,
        step: usize,
        rank: usize,
    ) {
        if self.worst.is_none() || rel_l2 > self.max_rel_l2 {
            self.max_rel_l2 = rel_l2;
            self.worst =
                Some(json!({"layer": layer, "expert": expert, "step": step, "rank": rank}));
        }
        self.max_abs = self.max_abs.max(max_abs);
        self.rel_l2.push(rel_l2);
    }

    fn summary(&mut self) -> Value {
        self.rel_l2.sort_unstable_by(f64::total_cmp);
        let n = self.rel_l2.len();
        let median = if n == 0 {
            None
        } else if n % 2 == 0 {
            Some((self.rel_l2[n / 2 - 1] + self.rel_l2[n / 2]) / 2.0)
        } else {
            Some(self.rel_l2[n / 2])
        };
        let p99 = if n == 0 {
            None
        } else {
            Some(self.rel_l2[n - n / 100 - 1])
        };
        json!({
            "slots": n, "max_rel_l2": self.max_rel_l2, "median_rel_l2": median,
            "p99_rel_l2": p99, "max_abs": self.max_abs, "worst": self.worst,
        })
    }
}

struct Sample {
    step: usize,
    rank: usize,
    weight: f32,
    rel_l2: f64,
    max_abs: f64,
    values: Vec<f32>,
}

struct Pair {
    codec: &'static str,
    count: usize,
    samples: Vec<Sample>,
}

#[derive(Default)]
struct Capture {
    stats: BTreeMap<&'static str, Stats>,
    layers: BTreeMap<usize, BTreeMap<&'static str, usize>>,
    pairs: BTreeMap<(usize, usize), Pair>,
}

impl Capture {
    fn record(
        &mut self,
        hfq: &HfqFile,
        args: &Args,
        step: usize,
        record: SplitParityRecord,
    ) -> R<()> {
        if record.rows != 1
            || record.dim != DIM
            || record.mi != 640
            || record.top_k != 10
            || record.endpoint != SplitEndpoint::Natural
        {
            return Err(format!(
                "step {step} layer {}: expected Natural rows=1 dim=2560 mi=640 top_k=10, got {:?}",
                record.layer,
                (
                    record.endpoint,
                    record.rows,
                    record.dim,
                    record.mi,
                    record.top_k
                )
            ));
        }
        if record.x_rot.len() != DIM
            || record.ids.len() != record.top_k
            || record.weights.len() != record.top_k
            || record.d2_expanded.len() != record.top_k * DIM
            || record.gpu_expanded.len() != record.top_k * DIM
        {
            return Err(format!(
                "step {step} layer {}: malformed parity record lengths",
                record.layer
            ));
        }
        if record
            .x_rot
            .iter()
            .chain(&record.weights)
            .chain(&record.d2_expanded)
            .chain(&record.gpu_expanded)
            .any(|v| !v.is_finite())
        {
            return Err(format!(
                "step {step} layer {}: non-finite capture values",
                record.layer
            ));
        }
        for (rank, &id) in record.ids.iter().enumerate() {
            if !(0..512).contains(&id) {
                continue;
            }
            let expert = id as usize;
            let codec = codec(hfq, record.layer, expert)?;
            let gpu = &record.gpu_expanded[rank * DIM..(rank + 1) * DIM];
            let d2 = &record.d2_expanded[rank * DIM..(rank + 1) * DIM];
            let (mut diff2, mut norm2, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
            for (&cpu, &reference) in d2.iter().zip(gpu) {
                let delta = f64::from(cpu) - f64::from(reference);
                diff2 += delta * delta;
                norm2 += f64::from(reference) * f64::from(reference);
                max_abs = max_abs.max(delta.abs());
            }
            let rel_l2 = diff2.sqrt() / norm2.sqrt().max(1.0e-30);
            if !rel_l2.is_finite() || !max_abs.is_finite() {
                return Err(format!(
                    "step {step} layer {} expert {expert}: non-finite metrics",
                    record.layer
                ));
            }
            self.stats.entry(codec).or_default().add(
                rel_l2,
                max_abs,
                record.layer,
                expert,
                step,
                rank,
            );
            *self
                .layers
                .entry(record.layer)
                .or_default()
                .entry(codec)
                .or_default() += 1;
            if args.fixture_dir.is_some() {
                let pair = self
                    .pairs
                    .entry((record.layer, expert))
                    .or_insert_with(|| Pair {
                        codec,
                        count: 0,
                        samples: Vec::new(),
                    });
                pair.count += 1;
                if pair.samples.len() < args.samples_per_expert {
                    let mut values = Vec::with_capacity(3 * DIM);
                    values.extend_from_slice(&record.x_rot);
                    values.extend_from_slice(gpu);
                    values.extend_from_slice(d2);
                    pair.samples.push(Sample {
                        step,
                        rank,
                        weight: record.weights[rank],
                        rel_l2,
                        max_abs,
                        values,
                    });
                }
            }
        }
        Ok(())
    }
}

fn parity(mode: SplitParityMode, layers: Option<Vec<usize>>) -> SplitParityConfig {
    SplitParityConfig {
        mode,
        layers,
        endpoint: SplitEndpoint::Natural,
        serial: false,
        shadow: None,
        max_records: if mode == SplitParityMode::Off {
            0
        } else {
            SPLIT_PARITY_MAX_RECORDS
        },
    }
}

fn argmax(logits: &[f32]) -> R<u32> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite logits".into());
    }
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| "empty logits".into())
}

fn run_capture(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    hfq: &HfqFile,
    args: &Args,
    tokens: &[u32],
    layers: &[usize],
) -> R<Capture> {
    bundle
        .split_parity_configure(parity(SplitParityMode::Full, Some(layers.to_vec())))
        .map_err(err)?;
    let result = (|| {
        bundle.reset(gpu).map_err(err)?;
        let logits = gpu
            .zeros(&[bundle.config.vocab_size], DType::F32)
            .map_err(err)?;
        bundle
            .forward_chunk_final(gpu, &tokens[..args.prompt_rows], &logits, None)
            .map_err(err)?;
        let stray = bundle.split_take_parity_records();
        if !stray.is_empty() {
            return Err(format!(
                "prefill unexpectedly produced {} D2 records; use a wide --prompt-rows",
                stray.len()
            ));
        }
        let mut next = argmax(&gpu.download_f32(&logits).map_err(err)?)?;
        let mut capture = Capture::default();
        for step in 0..args.steps {
            bundle
                .forward_token(gpu, next, &logits, None)
                .map_err(|e| format!("step {step}: {e}"))?;
            next = argmax(&gpu.download_f32(&logits).map_err(err)?)?;
            let records = bundle.split_take_parity_records();
            if records.len() != layers.len()
                || records.iter().map(|r| r.layer).collect::<BTreeSet<_>>()
                    != layers.iter().copied().collect()
            {
                return Err(format!(
                    "step {step}: expected one record per selected layer {layers:?}, got {:?}",
                    records.iter().map(|r| r.layer).collect::<Vec<_>>()
                ));
            }
            for record in records {
                capture.record(hfq, args, step, record)?;
            }
        }
        Ok(capture)
    })();
    let off = bundle
        .split_parity_configure(parity(SplitParityMode::Off, None))
        .map_err(err);
    match (result, off) {
        (Ok(capture), Ok(())) => Ok(capture),
        (Err(e), Ok(())) => Err(e),
        (Ok(_), Err(e)) => Err(format!("clear parity: {e}")),
        (Err(e), Err(off)) => Err(format!("{e}; clear parity: {off}")),
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_json(path: &Path, value: &Value) -> R<()> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value).map_err(err)?;
    writer.write_all(b"\n").map_err(err)?;
    writer.flush().map_err(err)
}

fn write_tensor(
    hfq: &HfqFile,
    dir: &Path,
    layer: usize,
    expert: usize,
    projection: &str,
) -> R<Value> {
    let name = tensor_name(layer, expert, projection);
    let (info, bytes) = hfq
        .tensor_data(&name)
        .ok_or_else(|| format!("missing artifact bytes for {name}"))?;
    let dtype = match info.quant_type {
        44 => "MQ4G256V2",
        53 => "MQ4G128V2",
        135 => "E8PI4G128R256",
        136 => "E8PI4G128R128",
        131 => "MQ3I4G128R256",
        132 => "MQ3I4G128R128",
        qt => return Err(format!("{name}: unsupported fixture quant type {qt}")),
    };
    let file = format!("L{layer}_E{expert}.{projection}.bin");
    std::fs::write(dir.join(&file), bytes).map_err(err)?;
    Ok(
        json!({"qt": info.quant_type, "dtype": dtype, "file_offset": info.data_offset,
        "len": info.data_size, "sha256": sha256(bytes), "file": file}),
    )
}

fn write_fixture(hfq: &HfqFile, args: &Args, capture: &Capture, provenance: &Value) -> R<()> {
    let Some(dir) = &args.fixture_dir else {
        return Ok(());
    };
    let mut selected = Vec::new();
    for (codec, &count) in CODECS.iter().zip(&args.fixture_experts) {
        let mut candidates = capture
            .pairs
            .iter()
            .filter(|(_, p)| p.codec == *codec)
            .collect::<Vec<_>>();
        candidates.sort_by(|(ka, a), (kb, b)| b.count.cmp(&a.count).then_with(|| ka.cmp(kb)));
        if candidates.len() < count {
            return Err(format!("fixture needs {count} distinct {codec} experts, captured only {}; increase --steps/--layers or adjust --fixture-experts", candidates.len()));
        }
        selected.extend(candidates.into_iter().take(count));
    }
    std::fs::create_dir_all(dir).map_err(err)?;
    let mut samples = BufWriter::new(std::fs::File::create(dir.join("samples.bin")).map_err(err)?);
    let mut sample_index = 0usize;
    let mut pairs = Vec::new();
    for (&(layer, expert), pair) in selected {
        let gate_up = write_tensor(hfq, dir, layer, expert, "gate_up")?;
        let down = write_tensor(hfq, dir, layer, expert, "down")?;
        let mut metadata = Vec::new();
        for sample in &pair.samples {
            for value in &sample.values {
                samples.write_all(&value.to_le_bytes()).map_err(err)?;
            }
            metadata.push(
                json!({"index": sample_index, "step": sample.step, "rank": sample.rank,
                "weight": sample.weight, "rel_l2": sample.rel_l2, "max_abs": sample.max_abs}),
            );
            sample_index += 1;
        }
        pairs.push(
            json!({"layer": layer, "expert": expert, "codec": pair.codec,
            "captured_samples": pair.count, "gate_up": gate_up, "down": down, "samples": metadata}),
        );
    }
    samples.flush().map_err(err)?;
    write_json(
        &dir.join("manifest.json"),
        &json!({
            "model": provenance, "capture_args": args.identity,
            "sample_file": "samples.bin", "sample_index_unit": "record",
            "sample_bytes": 3 * DIM * 4, "sample_dtype": "f32_le",
            "sample_layout": ["x_rot[2560]", "gpu_expanded_slot[2560]", "d2_expanded_slot[2560]"],
            "pairs": pairs,
        }),
    )
}

fn run() -> R<()> {
    let args = parse_args()?;
    // A separate base-file mapping survives the product loader consuming its
    // source, and guarantees fixtures describe the artifact, not GPU/overlay bytes.
    let hfq = HfqFile::open_at_offset(&args.model, 0).map_err(err)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(err)?;
    let text = std::fs::read_to_string(&args.text)
        .map_err(|e| format!("read {}: {e}", args.text.display()))?;
    let tokens = tokenizer.encode(&text);
    if tokens.len() < args.prompt_rows + 1 {
        return Err(format!(
            "TEXT has {} tokens; need at least prompt_rows + 1 = {}",
            tokens.len(),
            args.prompt_rows + 1
        ));
    }
    let stat = std::fs::metadata(&args.model).map_err(err)?;
    let mtime = stat
        .modified()
        .map_err(err)?
        .duration_since(UNIX_EPOCH)
        .map_err(err)?;
    let prefix = hfq.data_range(0, INDEX_PREFIX_LEN).ok_or_else(|| {
        format!("artifact does not contain the required {INDEX_PREFIX_LEN}-byte provenance prefix")
    })?;
    let provenance = json!({"path": args.model, "file_size": stat.len(),
        "mtime_unix_secs": mtime.as_secs(), "mtime_nanoseconds": mtime.subsec_nanos(),
        "prefix_len": INDEX_PREFIX_LEN, "prefix_sha256": sha256(prefix)});
    let mut gpu = Gpu::init().map_err(err)?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu.vmm_recommended_granularity().map_err(err)?,
        Qwen4KvBackend::Legacy => 1,
    };
    let path = args
        .model
        .to_str()
        .ok_or("MODEL path must be UTF-8")?
        .to_string();
    let shared = Qwen4RankShared {
        continuous_batch: false,
        path: path.clone(),
        max_seq: MAX_SEQ,
        sequence: None,
        vision_reserve_bytes: 0,
        backend,
        vmm_granularity,
        kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
        state_quant: None,
        spec: SpecLoadCfg {
            ngram_draft: Some(false),
            mtp: Some(false),
            dflash: Some(false),
            dspark: Some(false),
            ..SpecLoadCfg::default()
        },
        native_mtp: false,
        max_k: 0,
    };
    let src = ModelSource::from_path(&path)?;
    // carriers is private; hipfire-loader re-exports this exact product entry point.
    let mut loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, src)?;
    let armed = loaded.bundle.split_layers();
    println!("D2 armed layers: {armed:?}");
    if armed.is_empty() {
        return Err("D2 is not armed: request HIPFIRE_OFFLOAD_EXEC=cpu (or the per-load CPU default) and ensure expert layers are host-mapped".into());
    }
    let layers = select_layers(&args.layers, &armed)?;
    println!("Capturing layers: {layers:?}");
    let mut capture = run_capture(&mut loaded.bundle, &mut gpu, &hfq, &args, &tokens, &layers)?;
    let mut codecs = BTreeMap::new();
    for codec in CODECS {
        codecs.insert(codec, capture.stats.entry(codec).or_default().summary());
    }
    let report = json!({"model": provenance, "capture_args": args.identity,
        "armed_layers": armed, "selected_layers": layers, "prompt_rows": args.prompt_rows,
        "steps": args.steps, "max_seq": MAX_SEQ, "codecs": codecs, "per_layer": capture.layers});
    println!("{}", serde_json::to_string_pretty(&report).map_err(err)?);
    write_json(&args.json, &report)?;
    write_fixture(&hfq, &args, &capture, &provenance)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen4_slab_d2_capture: {error}");
        std::process::exit(2);
    }
}
