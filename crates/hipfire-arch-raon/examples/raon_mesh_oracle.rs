// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Layer-band identity oracle across devices: the model loaded as pipeline bands
//! on two GPUs and driven through the GENERIC mesh executor
//! (`Gpus::submit_mesh_epoch`) must reproduce the whole model on one device
//! bit for bit.
//!
//! ```text
//! raon_mesh_oracle --model M.hfq [--prompt-file F] [--prompt-tokens 600] [--decode 64]
//!                  [--chunk-rows 512] [--cuts "1;13;24"] [--band-decode eager|graph|both]
//!                  [--edge-transport host|peer|auto] [--rank-order output-last|visible]
//!                  [--out DIR]
//! ```
//!
//! Run 0 is the whole model on one device (the production load path): reset,
//! prefill the prompt in chunk-row pieces (>= 2 chunks at the default 600
//! tokens / 512 rows), then `--decode` greedy (argmax) decode steps, recorded on
//! the graph route and then on the eager route. Every `--cuts` entry is one
//! PP2 run (`13` = layers `0..13` on rank 0, `13..25` on rank 1): it builds a
//! `Gpus` over the two devices of `HIPFIRE_DEVICES` (rank order: the first
//! visible device is the output band, last rank, unless `--rank-order
//! visible`), the admitted mesh and band placement, provisions the edges, loads
//! one band bundle per rank on its own device, then drives prefill and decode
//! through the executor. A run records the last-row logits hash after every
//! prompt piece, the argmax ids and an FNV-1a 64 hash of every decode step's
//! logits bytes, and is compared with the whole run on the SAME decode route.
//! One more prefill of the whole prompt in a single call checks the
//! multi-chunk trunk-only path against the whole run's final-chunk logits.
//!
//! Exit status: 0 every required comparison is bit-identical, 1 otherwise.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use hipfire_arch_raon::bundle::{DecodeMode, PrefillMode};
use hipfire_arch_raon::{load_raon_band_bundle, RaonBand, RaonBundle, RaonConfig, RaonMeshModel};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mesh_placement::{
    pp_rank_order, resolve_edge_transports, AdmittedMesh, AdmittedRank, BandPlacement,
    EdgeTransportPolicy,
};
use hipfire_runtime::multi_gpu::transport::qualify_peer;
use hipfire_runtime::multi_gpu::Gpus;
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
    /// Last-row logits hash after each prompt piece.
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

/// The model as mesh bands, driven through the executor: `prefill` per
/// chunk-row piece (each piece's last row has logits), then decode.
fn run_mesh(
    gpus: &mut Gpus,
    model: &mut RaonMeshModel,
    prompt: &[u32],
    chunk_rows: usize,
    n_decode: usize,
    mode: DecodeMode,
    label: String,
) -> Result<Trace, String> {
    model.set_decode_mode(mode);
    model.reset(gpus)?;
    let t0 = Instant::now();
    let mut prefill_hashes = Vec::new();
    let mut last = Vec::new();
    for piece in prompt.chunks(chunk_rows) {
        model.prefill(gpus, piece)?;
        last = model.download_logits(gpus)?;
        prefill_hashes.push(hash_logits(&last));
    }
    let prefill_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (ids, decode_hashes) = decode_loop(&last, n_decode, |tok| {
        model.decode_step(gpus, tok)?;
        model.download_logits(gpus)
    })?;
    for gpu in &gpus.devices {
        gpu.bind_thread().map_err(|e| e.to_string())?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
    }
    let secs = t1.elapsed().as_secs_f64();
    if model.n_tokens() != prompt.len() + n_decode {
        return Err(format!(
            "mesh slots consumed {} tokens, expected {}",
            model.n_tokens(),
            prompt.len() + n_decode
        ));
    }
    Ok(Trace {
        label,
        prefill_hashes,
        ids,
        decode_hashes,
        prefill_secs,
        decode_secs: secs,
        decode_route: model.decode_mode().to_string(),
    })
}

/// Reset, prefill the whole prompt in ONE call (every chunk but the last is a
/// trunk-only epoch), and hash the last row's logits.
fn trunk_path_hash(
    gpus: &mut Gpus,
    model: &mut RaonMeshModel,
    prompt: &[u32],
) -> Result<u64, String> {
    model.reset(gpus)?;
    model.prefill(gpus, prompt)?;
    Ok(hash_logits(&model.download_logits(gpus)?))
}

fn parse_cuts(spec: &str) -> Result<Vec<usize>, String> {
    spec.split(';')
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .map_err(|e| format!("--cuts `{t}`: {e} (one cut per PP2 run)"))
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

/// One `Gpus` over the first two devices of `HIPFIRE_DEVICES`, permuted to the
/// admitted logical rank order (`output-last`: the first visible device is the
/// last rank, the daemon-style output card), `gpus.mesh` the admitted mesh and
/// a launch stream on each device.
fn init_pp2(
    counts: &[usize],
    rank_order: &str,
    transport: &str,
) -> Result<(Gpus, AdmittedMesh, BandPlacement), String> {
    let stages = counts.len();
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, stages)]).map_err(|e| format!("mesh: {e:?}"))?;
    let mut gpus = Gpus::init_layers(counts).map_err(|e| format!("Gpus::init_layers: {e}"))?;
    let visible: Vec<AdmittedRank> = gpus
        .devices
        .iter()
        .enumerate()
        .map(|(logical, gpu)| {
            Ok(AdmittedRank {
                logical,
                device_ordinal: gpu.device_id,
                pci_bus_id: Some(
                    gpu.hip
                        .device_pci_bus_id(gpu.device_id)
                        .map_err(|e| format!("pci id: {e}"))?,
                ),
                arch: gpu.arch.clone(),
            })
        })
        .collect::<Result<_, String>>()?;
    let ranks = match rank_order {
        "output-last" => pp_rank_order(&visible)?,
        "visible" => visible.clone(),
        other => return Err(format!("--rank-order `{other}`: output-last|visible")),
    };
    let mut pool: Vec<Option<Gpu>> = std::mem::take(&mut gpus.devices)
        .into_iter()
        .map(Some)
        .collect();
    let mut ordered = Vec::with_capacity(stages);
    for rank in &ranks {
        let slot = pool.iter_mut().find(|slot| {
            slot.as_ref()
                .is_some_and(|g| g.device_id == rank.device_ordinal)
        });
        ordered.push(
            slot.and_then(Option::take)
                .ok_or_else(|| format!("rank {} names an unopened device", rank.logical))?,
        );
    }
    gpus.devices = ordered;
    gpus.mesh = mesh.clone();
    for gpu in gpus.devices.iter_mut() {
        gpu.ensure_capture_stream()
            .map_err(|e| format!("launch stream: {e}"))?;
    }
    let policy = match transport {
        "host" => EdgeTransportPolicy::Host,
        "peer" => EdgeTransportPolicy::Peer,
        "auto" => EdgeTransportPolicy::Auto,
        other => return Err(format!("--edge-transport `{other}`: host|peer|auto")),
    };
    let edge_list: Vec<(usize, usize)> = (0..stages - 1).map(|s| (s, s + 1)).collect();
    // Byte-qualify each pair only when the policy may use peer transport.
    let mut qualified = vec![false; stages];
    if policy != EdgeTransportPolicy::Host {
        for &(src, dst) in &edge_list {
            let (a, b) = gpus.devices.split_at_mut(dst);
            qualified[src] = qualify_peer(&mut a[src], &mut b[0])?;
        }
    }
    let edges = resolve_edge_transports(policy, &|src, _| qualified[src], &edge_list)?;
    let admitted = AdmittedMesh {
        mesh: mesh.clone(),
        ranks,
        edges,
        family: "raon1_mini".to_string(),
        numerical_policy: "raon-mesh-v1".to_string(),
    };
    let placement = BandPlacement::from_counts(&mesh, counts.iter().sum(), counts)?;
    Ok((gpus, admitted, placement))
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
    let model_path = opt(&opts, "model").ok_or("missing --model")?.to_string();
    let n_prompt = opt_usize(&opts, "prompt-tokens", 600)?;
    let n_decode = opt_usize(&opts, "decode", 64)?;
    let chunk_rows = opt_usize(&opts, "chunk-rows", 512)?;
    let cuts = parse_cuts(opt(&opts, "cuts").unwrap_or("1;13;24"))?;
    let band_decode = opt(&opts, "band-decode").unwrap_or("both");
    let rank_order = opt(&opts, "rank-order").unwrap_or("output-last");
    let transport = opt(&opts, "edge-transport").unwrap_or("host");
    let modes: Vec<DecodeMode> = match band_decode {
        "eager" => vec![DecodeMode::Eager],
        "graph" => vec![DecodeMode::Graph],
        "both" => vec![DecodeMode::Eager, DecodeMode::Graph],
        other => return Err(format!("--band-decode `{other}`: eager|graph|both")),
    };
    if n_prompt <= chunk_rows {
        eprintln!(
            "raon_mesh_oracle: NOTE {n_prompt} prompt tokens fit one {chunk_rows}-row chunk \
             (the identity run wants >= 2 chunks)"
        );
    }
    let max_seq = (n_prompt + n_decode + 64).max(512);

    let hfq =
        HfqFile::open(Path::new(&model_path)).map_err(|e| format!("open {model_path}: {e}"))?;
    let prompt = prompt_ids(
        &hfq,
        opt(&opts, "prompt-file").unwrap_or(DEFAULT_PROMPT),
        n_prompt,
    )?;
    let cfg = RaonConfig::from_hfq(&hfq)?;
    let total_layers = cfg.n_layers;
    let loadavg_start = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!(
        "raon_mesh_oracle: {} prompt tokens in {}-row chunks ({} chunks), {n_decode} decode \
         steps, max_seq {max_seq}, {total_layers} layers, loadavg {}",
        prompt.len(),
        chunk_rows,
        prompt.len().div_ceil(chunk_rows),
        loadavg_start.trim()
    );

    // ── run 1: the whole model on one device, graph (product) then eager ──
    let (whole_graph, whole_eager) = {
        let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
        eprintln!(
            "raon_mesh_oracle: whole model on device {} ({})",
            gpu.device_id, gpu.arch
        );
        let mut whole = load_raon_band_bundle(
            &hfq,
            &mut gpu,
            max_seq,
            &RaonBand::whole(&cfg),
            Some(chunk_rows),
        )?;
        if whole.prefill_chunk_rows() != chunk_rows {
            return Err(format!(
                "whole bundle admitted {} chunk rows, expected {chunk_rows}",
                whole.prefill_chunk_rows()
            ));
        }
        let g = run_whole(
            &mut gpu,
            &mut whole,
            &prompt,
            chunk_rows,
            n_decode,
            DecodeMode::Graph,
        )?;
        let e = run_whole(
            &mut gpu,
            &mut whole,
            &prompt,
            chunk_rows,
            n_decode,
            DecodeMode::Eager,
        )?;
        whole.free_all(&mut gpu);
        gpu.drain_pool();
        (g, e)
    };
    eprintln!(
        "whole: prefill {:.3}s graph-run / {:.3}s eager-run; decode {:.2} / {:.2} ms/step",
        whole_graph.prefill_secs,
        whole_eager.prefill_secs,
        whole_graph.decode_secs * 1e3 / n_decode as f64,
        whole_eager.decode_secs * 1e3 / n_decode as f64
    );

    let mut all_ok = true;
    let mut report_runs = Vec::new();
    let mut lines = Vec::new();
    let c = compare(&whole_graph, &whole_eager, prompt.len());
    lines.push(format!(
        "{:<44} vs {:<12} ids identical {} first divergent step {:?} prefill-hash mismatches {}/{} decode-hash mismatches {}/{} (rows: prefill {} decode {}) [info]",
        "whole/graph", "whole/eager", yn(c.ids_identical), c.first_divergent_step,
        c.prefill_mismatches, whole_graph.prefill_hashes.len(),
        c.decode_mismatches, c.decode_rows, c.prefill_rows, c.decode_rows
    ));
    let tok_s = |tokens: usize, secs: f64| tokens as f64 / secs;
    lines.push(format!(
        "{:<44} prefill {:.1} tok/s ({} tokens in {:.3}s), decode {:.2} tok/s ({:.2} ms/step) [graph route]; eager route: prefill {:.1} tok/s, decode {:.2} tok/s ({:.2} ms/step)",
        "whole (1 device)",
        tok_s(prompt.len(), whole_graph.prefill_secs), prompt.len(), whole_graph.prefill_secs,
        tok_s(n_decode, whole_graph.decode_secs), whole_graph.decode_secs * 1e3 / n_decode as f64,
        tok_s(prompt.len(), whole_eager.prefill_secs),
        tok_s(n_decode, whole_eager.decode_secs), whole_eager.decode_secs * 1e3 / n_decode as f64,
    ));

    // ── runs 2..: PP2 on both devices, one run per cut ──
    let mut edge_json = Vec::new();
    let (mut gpus, admitted, _) = init_pp2(
        &[total_layers / 2, total_layers - total_layers / 2],
        rank_order,
        transport,
    )?;
    let device_line = admitted
        .ranks
        .iter()
        .map(|r| {
            format!(
                "rank {} = device {} ({})",
                r.logical, r.device_ordinal, r.arch
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!(
        "raon_mesh_oracle: PP2 {device_line}; edges {:?}",
        admitted
            .edges
            .iter()
            .map(|e| format!("{}->{} {}", e.src, e.dst, e.transport))
            .collect::<Vec<_>>()
    );
    let capacity = RaonMeshModel::edge_capacity_bytes(&cfg, chunk_rows)?;
    for &cut in &cuts {
        if cut == 0 || cut >= total_layers {
            return Err(format!(
                "--cuts {cut}: a cut is a layer index in 1..{total_layers}"
            ));
        }
        let counts = [cut, total_layers - cut];
        let mesh = admitted.mesh.clone();
        let placement = BandPlacement::from_counts(&mesh, total_layers, &counts)?;
        let name = format!("{}..{}|{}..{}", 0, cut, cut, total_layers);
        gpus.provision_mesh_edges(&admitted, &placement, capacity)
            .map_err(|e| format!("provision_mesh_edges: {e}"))?;
        let mut loaded: Vec<RaonBundle> = Vec::new();
        let mut load_err = None;
        for rank in 0..counts.len() {
            let band = RaonBand::new(placement.band(rank), total_layers)?;
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread().map_err(|e| e.to_string())?;
            match load_raon_band_bundle(&hfq, gpu, max_seq, &band, Some(chunk_rows)) {
                Ok(bundle) => loaded.push(bundle),
                Err(e) => {
                    load_err = Some(format!("rank {rank} {}: {e}", band.label()));
                    break;
                }
            }
        }
        let mut results: Vec<(DecodeMode, Result<Trace, String>)> = Vec::new();
        let mut trunk_check: Option<Result<u64, String>> = None;
        let mut graph_stats = Vec::new();
        match load_err {
            Some(e) => {
                for rank_bundle in loaded {
                    let rank = 0;
                    let _ = rank;
                    drop(rank_bundle);
                }
                let _ = gpus.drain_mesh_edges();
                return Err(e);
            }
            None => match RaonMeshModel::new(&cfg, &mesh, &placement, loaded) {
                Err((e, bundles)) => {
                    std::mem::forget(bundles);
                    return Err(e);
                }
                Ok(mut model) => {
                    for &mode in &modes {
                        let r = run_mesh(
                            &mut gpus,
                            &mut model,
                            &prompt,
                            chunk_rows,
                            n_decode,
                            mode,
                            format!("PP2 {name} /{}", mode.as_str()),
                        );
                        results.push((mode, r));
                    }
                    trunk_check = Some(trunk_path_hash(&mut gpus, &mut model, &prompt));
                    graph_stats = model.decode_graph_stats();
                    model.free(&mut gpus)?;
                }
            },
        }
        for gpu in gpus.devices.iter_mut() {
            gpu.bind_thread().map_err(|e| e.to_string())?;
            gpu.drain_pool();
        }
        for (mode, result) in results {
            let label = format!("PP2 {name} /{}", mode.as_str());
            let t = match result {
                Ok(t) => t,
                Err(e) => {
                    all_ok = false;
                    lines.push(format!("{label}: FAILED: {e}"));
                    report_runs.push(json!({"label": label, "error": e}));
                    continue;
                }
            };
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
                    "{:<44} vs {:<12} ids identical {} first divergent step {:?} prefill-hash mismatches {}/{} decode-hash mismatches {}/{} (rows: prefill {} decode {}) [{tag}{}]",
                    t.label, against, yn(cmp.ids_identical), cmp.first_divergent_step,
                    cmp.prefill_mismatches, t.prefill_hashes.len(),
                    cmp.decode_mismatches, cmp.decode_rows, cmp.prefill_rows, cmp.decode_rows,
                    if tag == "REQUIRED" { if ok { " PASS" } else { " FAIL" } } else { "" }
                ));
            }
            lines.push(format!(
                "{:<44} timing: prefill {:.1} tok/s ({:.3}s), decode {:.2} tok/s ({:.2} ms/step) on the {} route; whole same route: prefill {:.1} tok/s, decode {:.2} tok/s ({:.2} ms/step)",
                t.label,
                tok_s(prompt.len(), t.prefill_secs), t.prefill_secs,
                tok_s(n_decode, t.decode_secs), t.decode_secs * 1e3 / n_decode as f64,
                t.decode_route,
                tok_s(prompt.len(), same.prefill_secs),
                tok_s(n_decode, same.decode_secs), same.decode_secs * 1e3 / n_decode as f64,
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
        if let Some(check) = trunk_check {
            let want = *whole_eager
                .prefill_hashes
                .last()
                .expect("prompt has chunks");
            match check {
                Ok(h) => {
                    let ok = h == want;
                    all_ok &= ok;
                    lines.push(format!(
                        "PP2 {name} single-call prefill ({} chunks, trunk-only epochs before the last): last-row logits hash {} whole's {} [REQUIRED {}]",
                        prompt.len().div_ceil(chunk_rows),
                        if ok { "==" } else { "!=" },
                        format_args!("{want:016x}"),
                        if ok { "PASS" } else { "FAIL" }
                    ));
                    // The whole run's final chunk is the same rows only when the
                    // single-call chunking equals the piecewise chunking (it does).
                    edge_json.push(json!({"cut": cut, "trunk_path_hash": format!("{h:016x}"), "whole": format!("{want:016x}"), "pass": ok}));
                }
                Err(e) => {
                    all_ok = false;
                    lines.push(format!("PP2 {name} single-call prefill: FAILED: {e}"));
                }
            }
        }
        lines.push(format!(
            "PP2 {name} per-rank decode graph (captures, replays): {graph_stats:?}"
        ));
    }

    println!("\n=== raon_mesh_oracle ===");
    println!(
        "model {model_path}\nprompt {} tokens / {chunk_rows}-row chunks ({} chunks); decode {n_decode} greedy steps; layers {total_layers}; edge transport {transport}; {device_line}\nloadavg at start {}",
        prompt.len(),
        prompt.len().div_ceil(chunk_rows),
        loadavg_start.trim()
    );
    for l in &lines {
        println!("{l}");
    }
    println!("verdict: {}", if all_ok { "PASS" } else { "FAIL" });
    if let Some(dir) = opt(&opts, "out") {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {dir}: {e}"))?;
        let doc = json!({
            "model": model_path,
            "prompt_tokens": prompt.len(),
            "chunk_rows": chunk_rows,
            "decode_steps": n_decode,
            "total_layers": total_layers,
            "edge_transport": transport,
            "ranks": device_line,
            "loadavg_start": loadavg_start.trim(),
            "whole_graph": trace_json(&whole_graph),
            "whole_eager": trace_json(&whole_eager),
            "runs": report_runs,
            "single_call_prefill": edge_json,
            "summary": lines,
            "pass": all_ok,
        });
        std::fs::write(
            Path::new(dir).join("mesh_oracle.json"),
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
            eprintln!("raon_mesh_oracle: {e}");
            std::process::exit(1);
        }
    }
}
