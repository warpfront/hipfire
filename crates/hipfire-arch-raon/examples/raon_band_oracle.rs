// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Layer-band identity oracle (one device): the model loaded as layer bands on
//! the SAME `Gpu`, driven band after band with the mHC stream edge copied
//! device-to-device between them, must reproduce the whole model bit for bit.
//!
//! ```text
//! raon_band_oracle --model M.hfq [--prompt-file F] [--prompt-tokens 600] [--decode 64]
//!                  [--chunk-rows 512] [--cuts "1;13;24;1,13,24"] [--band-decode eager|graph|both]
//!                  [--out DIR]
//! ```
//!
//! Run 0 is the whole model (`RaonBand::whole`, the production load/assembly
//! path): reset, prefill the prompt in chunk-row pieces (>= 2 chunks at the
//! default 600 tokens / 512 rows), then `--decode` greedy (argmax) decode steps.
//! It is recorded on the product decode route (graph), then again on the eager
//! route of the same bundle. Every `--cuts` entry is one band run: a comma list
//! of layer indices a band starts at (`13` = layers `0..13` | `13..25`;
//! `1,13,24` = four bands). Each band run loads one bundle per band on the same
//! `Gpu`, drives band 0 then copies `scratch.stream_a` rows into the next
//! band's `stream_a` (`copy_edge_same_device`), and so on to the last band,
//! which reads the logits. A run records: the prompt-chunk last-row logits hash
//! of every chunk, the argmax ids, and an FNV-1a 64 hash of every decode step's
//! logits bytes. The verdict compares each band run against the whole run on
//! the SAME decode route (eager-vs-eager, graph-vs-graph) and against the
//! other route for information.
//!
//! Exit status: 0 every required comparison is bit-identical, 1 otherwise.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use hipfire_arch_raon::bundle::{DecodeMode, PrefillMode};
use hipfire_arch_raon::{
    copy_edge_same_device, edge_bytes, load_raon_band_bundle, BandInput, RaonBand, RaonBundle,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde_json::{json, Value};

const DEFAULT_PROMPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/prompts/raon/state.txt"
);

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn hash_logits(l: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(l.len() * 4);
    for v in l {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a(&bytes)
}

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

/// What one run recorded.
#[derive(Clone, Debug)]
struct Trace {
    label: String,
    /// Last-row logits hash after each prompt chunk.
    prefill_hashes: Vec<u64>,
    /// `ids[0]` = argmax of the last prompt row; `ids[i]` = argmax after decode step `i`.
    ids: Vec<u32>,
    /// Logits hash after decode step `1..=n`.
    decode_hashes: Vec<u64>,
    prefill_secs: f64,
    decode_secs: f64,
    decode_route: String,
}

struct Cmp {
    ids_identical: bool,
    first_divergent_step: Option<usize>,
    prefill_mismatches: usize,
    decode_mismatches: usize,
    prefill_rows: usize,
    decode_rows: usize,
}

fn compare(a: &Trace, b: &Trace, prompt_tokens: usize) -> Cmp {
    let n = a.ids.len().min(b.ids.len());
    let first = (0..n).find(|&i| a.ids[i] != b.ids[i]);
    let pm = a
        .prefill_hashes
        .iter()
        .zip(&b.prefill_hashes)
        .filter(|(x, y)| x != y)
        .count()
        + a.prefill_hashes.len().abs_diff(b.prefill_hashes.len());
    let dm = a
        .decode_hashes
        .iter()
        .zip(&b.decode_hashes)
        .filter(|(x, y)| x != y)
        .count()
        + a.decode_hashes.len().abs_diff(b.decode_hashes.len());

    Cmp {
        ids_identical: a.ids == b.ids,
        first_divergent_step: first,
        prefill_mismatches: pm,
        decode_mismatches: dm,
        prefill_rows: prompt_tokens,
        decode_rows: a.decode_hashes.len().min(b.decode_hashes.len()),
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "y"
    } else {
        "n"
    }
}

fn opt<'a>(opts: &'a BTreeMap<String, String>, key: &str) -> Option<&'a str> {
    opts.get(key).map(String::as_str)
}

fn opt_usize(opts: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match opts.get(key) {
        Some(s) => s.parse().map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

/// Prompt ids: the prompt file tokenized, repeated to `want` tokens, truncated.
fn prompt_ids(hfq: &HfqFile, file: &str, want: usize) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("read prompt {file}: {e}"))?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let base = tok.encode(&text);
    if base.is_empty() {
        return Err(format!("{file} tokenized to zero ids"));
    }
    Ok(base.iter().cycle().take(want).copied().collect())
}

/// Greedy decode `n` steps from the prefill logits; `step(token)` runs one
/// decode step through every band and returns the logits.
fn decode_loop(
    first_logits: &[f32],
    n: usize,
    mut step: impl FnMut(u32) -> Result<Vec<f32>, String>,
) -> Result<(Vec<u32>, Vec<u64>), String> {
    let mut ids = vec![argmax(first_logits)];
    let mut hashes = Vec::with_capacity(n);
    for _ in 0..n {
        let logits = step(*ids.last().expect("seeded"))?;
        hashes.push(hash_logits(&logits));
        ids.push(argmax(&logits));
    }
    Ok((ids, hashes))
}

/// The whole model on one bundle: `prefill` per chunk-row piece, then decode.
fn run_whole(
    gpu: &mut Gpu,
    bundle: &mut RaonBundle,
    prompt: &[u32],
    chunk_rows: usize,
    n_decode: usize,
    mode: DecodeMode,
) -> Result<Trace, String> {
    bundle.set_prefill_mode(PrefillMode::Batched);
    bundle.set_decode_mode(mode);
    bundle.reset(gpu)?;
    let t0 = Instant::now();
    let mut prefill_hashes = Vec::new();
    let mut last = Vec::new();
    for piece in prompt.chunks(chunk_rows) {
        bundle.prefill(gpu, piece)?;
        last = bundle.download_logits(gpu)?;
        prefill_hashes.push(hash_logits(&last));
    }
    let prefill_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (ids, decode_hashes) = decode_loop(&last, n_decode, |tok| {
        bundle.decode_step(gpu, tok)?;
        bundle.download_logits(gpu)
    })?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    Ok(Trace {
        label: format!("whole/{}", mode.as_str()),
        prefill_hashes,
        ids,
        decode_hashes,
        prefill_secs,
        decode_secs: t1.elapsed().as_secs_f64(),
        decode_route: bundle.decode_mode().to_string(),
    })
}

/// Copy the `rows` edge rows of `bands[i]` into `bands[i + 1]` (same device).
fn hop(gpu: &Gpu, bands: &[RaonBundle], i: usize, rows: usize) -> Result<(), String> {
    let src = bands[i].edge_view(rows)?;
    let dst = bands[i + 1].edge_view(rows)?;
    copy_edge_same_device(gpu, &dst, &src, edge_bytes(&bands[i].config, rows))
}

/// The model as `bands` (loaded on `gpu`), driven band after band.
fn run_banded(
    gpu: &mut Gpu,
    bands: &mut [RaonBundle],
    prompt: &[u32],
    chunk_rows: usize,
    n_decode: usize,
    mode: DecodeMode,
    label: String,
) -> Result<Trace, String> {
    let last_i = bands.len() - 1;
    for b in bands.iter_mut() {
        b.set_prefill_mode(PrefillMode::Batched);
        b.set_decode_mode(mode);
        b.reset(gpu)?;
    }
    let t0 = Instant::now();
    let mut prefill_hashes = Vec::new();
    let mut last = Vec::new();
    for piece in prompt.chunks(chunk_rows) {
        let rows = piece.len();
        for i in 0..=last_i {
            let input = if i == 0 {
                BandInput::Tokens(piece)
            } else {
                BandInput::Streams { rows }
            };
            bands[i].prefill_band_chunk(gpu, input, i == last_i)?;
            if i < last_i {
                hop(gpu, bands, i, rows)?;
            }
        }
        last = bands[last_i].download_logits(gpu)?;
        prefill_hashes.push(hash_logits(&last));
    }
    let prefill_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (ids, decode_hashes) = decode_loop(&last, n_decode, |tok| {
        bands[0].decode_step(gpu, tok)?;
        for i in 1..=last_i {
            hop(gpu, bands, i - 1, 1)?;
            bands[i].decode_step_from_edge(gpu)?;
        }
        bands[last_i].download_logits(gpu)
    })?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e:?}"))?;
    let tokens = bands.iter().map(|b| b.n_tokens()).collect::<Vec<_>>();
    if tokens.iter().any(|&t| t != prompt.len() + n_decode) {
        return Err(format!(
            "band slots disagree on consumed tokens: {tokens:?}, expected {}",
            prompt.len() + n_decode
        ));
    }
    Ok(Trace {
        label,
        prefill_hashes,
        ids,
        decode_hashes,
        prefill_secs,
        decode_secs: t1.elapsed().as_secs_f64(),
        decode_route: bands[0].decode_mode().to_string(),
    })
}

fn parse_cuts(spec: &str) -> Result<Vec<Vec<usize>>, String> {
    spec.split(';')
        .map(|run| {
            run.split(',')
                .map(|t| {
                    t.trim()
                        .parse::<usize>()
                        .map_err(|e| format!("--cuts `{run}`: {e}"))
                })
                .collect()
        })
        .collect()
}

fn trace_json(t: &Trace) -> Value {
    json!({
        "label": t.label,
        "decode_route": t.decode_route,
        "prefill_hashes": t.prefill_hashes.iter().map(|h| format!("{h:016x}")).collect::<Vec<_>>(),
        "ids": t.ids,
        "decode_hashes": t.decode_hashes.iter().map(|h| format!("{h:016x}")).collect::<Vec<_>>(),
        "prefill_secs": t.prefill_secs,
        "decode_secs": t.decode_secs,
    })
}

fn run() -> Result<bool, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = BTreeMap::new();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument `{}`", argv[i]))?;
        let val = argv
            .get(i + 1)
            .ok_or_else(|| format!("`--{key}` needs a value"))?;
        opts.insert(key.to_string(), val.clone());
        i += 2;
    }
    let model = opt(&opts, "model").ok_or("missing --model")?.to_string();
    let n_prompt = opt_usize(&opts, "prompt-tokens", 600)?;
    let n_decode = opt_usize(&opts, "decode", 64)?;
    let chunk_rows = opt_usize(&opts, "chunk-rows", 512)?;
    let cuts = parse_cuts(opt(&opts, "cuts").unwrap_or("1;13;24;1,13,24"))?;
    let band_decode = opt(&opts, "band-decode").unwrap_or("both");
    let modes: Vec<DecodeMode> = match band_decode {
        "eager" => vec![DecodeMode::Eager],
        "graph" => vec![DecodeMode::Graph],
        "both" => vec![DecodeMode::Eager, DecodeMode::Graph],
        other => return Err(format!("--band-decode `{other}`: eager|graph|both")),
    };
    if n_prompt <= chunk_rows {
        eprintln!(
            "raon_band_oracle: NOTE {n_prompt} prompt tokens fit one {chunk_rows}-row chunk \
             (the identity run wants >= 2 chunks)"
        );
    }
    let max_seq = (n_prompt + n_decode + 64).max(512);

    let hfq = HfqFile::open(Path::new(&model)).map_err(|e| format!("open {model}: {e}"))?;
    let prompt = prompt_ids(
        &hfq,
        opt(&opts, "prompt-file").unwrap_or(DEFAULT_PROMPT),
        n_prompt,
    )?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let total_layers = hipfire_arch_raon::RaonConfig::from_hfq(&hfq)?.n_layers;
    eprintln!(
        "raon_band_oracle: {} prompt tokens in {}-row chunks ({} chunks), {n_decode} decode steps, \
         max_seq {max_seq}, {total_layers} layers, device {}",
        prompt.len(),
        chunk_rows,
        prompt.len().div_ceil(chunk_rows),
        gpu.arch
    );

    // ── run 0: the whole model, graph (product) then eager (reference) ──
    let whole_band = {
        let cfg = hipfire_arch_raon::RaonConfig::from_hfq(&hfq)?;
        RaonBand::whole(&cfg)
    };
    let mut whole = load_raon_band_bundle(&hfq, &mut gpu, max_seq, &whole_band, Some(chunk_rows))?;
    if whole.prefill_chunk_rows() != chunk_rows {
        return Err(format!(
            "whole bundle admitted {} chunk rows, expected {chunk_rows}",
            whole.prefill_chunk_rows()
        ));
    }
    let whole_graph = run_whole(
        &mut gpu,
        &mut whole,
        &prompt,
        chunk_rows,
        n_decode,
        DecodeMode::Graph,
    )?;
    let whole_eager = run_whole(
        &mut gpu,
        &mut whole,
        &prompt,
        chunk_rows,
        n_decode,
        DecodeMode::Eager,
    )?;
    let (caps, reps) = whole.decode_graph_stats();
    eprintln!(
        "whole: graph route captures {caps} replays {reps} ({} kernel nodes); \
         prefill {:.3}s graph-run / {:.3}s eager-run; decode {:.2} / {:.2} ms/step",
        whole.decode_graph_kernel_nodes(),
        whole_graph.prefill_secs,
        whole_eager.prefill_secs,
        whole_graph.decode_secs * 1e3 / n_decode as f64,
        whole_eager.decode_secs * 1e3 / n_decode as f64
    );
    whole.free_all(&mut gpu);
    // The pool keeps freed buffers; hand them back so the bands can load.
    gpu.drain_pool();

    let mut all_ok = true;
    let mut report_runs = Vec::new();
    let mut lines = Vec::new();
    let c = compare(&whole_graph, &whole_eager, prompt.len());
    lines.push(format!(
        "{:<40} vs {:<14} ids identical {} first divergent step {:?} prefill-hash mismatches {}/{} decode-hash mismatches {}/{} (rows: prefill {} decode {}) [info]",
        "whole/graph", "whole/eager", yn(c.ids_identical), c.first_divergent_step,
        c.prefill_mismatches, whole_graph.prefill_hashes.len(),
        c.decode_mismatches, c.decode_rows, c.prefill_rows, c.decode_rows
    ));

    // ── band runs ──
    for cut_list in &cuts {
        let bands = RaonBand::from_cuts(total_layers, cut_list)?;
        let name = bands
            .iter()
            .map(|b| format!("{}..{}", b.layers.start, b.layers.end))
            .collect::<Vec<_>>()
            .join("|");
        for &mode in &modes {
            let mut loaded: Vec<RaonBundle> = Vec::new();
            let mut load_err = None;
            for b in &bands {
                match load_raon_band_bundle(&hfq, &mut gpu, max_seq, b, Some(chunk_rows)) {
                    Ok(bundle) => loaded.push(bundle),
                    Err(e) => {
                        load_err = Some(format!("band {}: {e}", b.label()));
                        break;
                    }
                }
            }
            let result = match load_err {
                Some(e) => Err(e),
                None => run_banded(
                    &mut gpu,
                    &mut loaded,
                    &prompt,
                    chunk_rows,
                    n_decode,
                    mode,
                    format!("bands {name} /{}", mode.as_str()),
                ),
            };
            for b in loaded {
                b.free_all(&mut gpu);
            }
            gpu.drain_pool();
            let t = match result {
                Ok(t) => t,
                Err(e) => {
                    all_ok = false;
                    lines.push(format!("bands {name} /{}: FAILED: {e}", mode.as_str()));
                    report_runs.push(
                        json!({"label": format!("bands {name} /{}", mode.as_str()), "error": e}),
                    );
                    continue;
                }
            };
            // Required: same decode route as the whole run; the other route is info.
            let (same, other) = match mode {
                DecodeMode::Eager => (&whole_eager, &whole_graph),
                DecodeMode::Graph => (&whole_graph, &whole_eager),
            };
            let req = compare(&t, same, prompt.len());
            let inf = compare(&t, other, prompt.len());
            let ok = req.ids_identical && req.prefill_mismatches == 0 && req.decode_mismatches == 0;
            all_ok &= ok;
            for (cmp, against, tag) in [
                (&req, &same.label, "REQUIRED"),
                (&inf, &other.label, "info"),
            ] {
                lines.push(format!(
                    "{:<40} vs {:<14} ids identical {} first divergent step {:?} prefill-hash mismatches {}/{} decode-hash mismatches {}/{} (rows: prefill {} decode {}) [{tag}{}]",
                    t.label,
                    against,
                    yn(cmp.ids_identical),
                    cmp.first_divergent_step,
                    cmp.prefill_mismatches,
                    t.prefill_hashes.len(),
                    cmp.decode_mismatches,
                    cmp.decode_rows,
                    cmp.prefill_rows,
                    cmp.decode_rows,
                    if tag == "REQUIRED" {
                        if ok { " PASS" } else { " FAIL" }
                    } else {
                        ""
                    }
                ));
            }

            lines.push(format!(
                "{:<40} timing: prefill {:.3}s decode {:.2} ms/step ({} route)",
                t.label,
                t.prefill_secs,
                t.decode_secs * 1e3 / n_decode as f64,
                t.decode_route
            ));
            report_runs.push(json!({
                "trace": trace_json(&t),
                "required_against": same.label,
                "pass": ok,
                "ids_identical": req.ids_identical,
                "first_divergent_step": req.first_divergent_step,
                "prefill_hash_mismatches": req.prefill_mismatches,
                "decode_hash_mismatches": req.decode_mismatches,
                "prefill_rows_compared": req.prefill_rows,
                "decode_rows_compared": req.decode_rows,
            }));
        }
    }

    println!("\n=== raon_band_oracle ===");
    println!(
        "model {model}\nprompt {} tokens / {chunk_rows}-row chunks ({} chunks); decode {n_decode} greedy steps; layers {total_layers}",
        prompt.len(),
        prompt.len().div_ceil(chunk_rows)
    );
    for l in &lines {
        println!("{l}");
    }
    println!("verdict: {}", if all_ok { "PASS" } else { "FAIL" });
    if let Some(dir) = opt(&opts, "out") {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {dir}: {e}"))?;
        let doc = json!({
            "model": model,
            "prompt_tokens": prompt.len(),
            "chunk_rows": chunk_rows,
            "decode_steps": n_decode,
            "total_layers": total_layers,
            "device_arch": gpu.arch,
            "whole_graph": trace_json(&whole_graph),
            "whole_eager": trace_json(&whole_eager),
            "runs": report_runs,
            "summary": lines,
            "pass": all_ok,
        });
        std::fs::write(
            Path::new(dir).join("band_oracle.json"),
            serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("write result: {e}"))?;
    }
    Ok(all_ok)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("raon_band_oracle: {e}");
            std::process::exit(1);
        }
    }
}
