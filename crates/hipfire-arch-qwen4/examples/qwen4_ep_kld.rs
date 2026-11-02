// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_ep_kld — served-route (expert-parallel, N cards) KLD scorer for Qwen4
//! (Qwen3.8 Flash-Next), lab.
//!
//! `qwen4_kld eval --decode` is single-device (`DeviceMesh::single`). This
//! tool scores the SAME HFKLDR v3 references with the SAME estimator, window,
//! per-chunk reset and decode route (prefill the unscored prefix, then one
//! `forward_token` per scored row, teacher-forced), but loads the model
//! through `hipfire_loader::load_model` exactly as the daemon does, so the
//! topology comes from `topology.ep` / `HIPFIRE_TOPOLOGY_EP` and the whole EP
//! route (including `HIPFIRE_QWEN4_EP_TRUNK_SHARD`) is what is scored. The
//! `.kldseq` bytes are the `qwen4_kld` HFKSEQ v2 format and pair with
//! `qwen4_kld`'s decode `.kldseq` chunk for chunk.
//!
//! Usage:
//! ```text
//! qwen4_ep_kld --model M.mq4-xts --out DIR \
//!     --score wt2:/path/wt2.kldref --score code24:/path/code24.kldref \
//!     [--max-chunks N] [--max-seq 4096] [--prefix chunk-final|prefill-final] \
//!     [--no-rank-check] [--kv-mode M] [--state-quant Q]
//! ```
//! One model load scores every `--score NAME:REF`; `DIR/NAME.kldseq` and
//! `DIR/NAME.json` (mean KLD, top-1, PPL, logits sha256, per-rank logits
//! agreement) are written, `DIR/meta.json` records binary / model / env /
//! topology. Under EP the rank-1.. logits mirrors are compared byte for byte
//! with rank 0's row on every scored row (`--no-rank-check` skips it); a
//! disagreement is counted in the receipt and does not stop the run.
//! Exit 0 = every score finished; 2 = refusal or error.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use rdna_compute::{DType, Gpu};
use saddle_quant::eval::estimator::{kld_from_block, nll_of, TopKBlock};
use saddle_quant::format::kldref::KldRef;
use saddle_quant::format::kldseq::{self, ChunkScore};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

const TOOL: &str = "qwen4_ep_kld";
const USAGE: &str =
    "usage: qwen4_ep_kld --model M --out DIR --score NAME:REF.kldref [--score ...] \
[--max-chunks N] [--max-seq N] [--prefix chunk-final|prefill-final] [--no-rank-check] \
[--kv-mode M] [--state-quant Q]";

fn e<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Prefix {
    /// `forward_chunk_final` (what `qwen4_kld eval --decode` runs).
    ChunkFinal,
    /// `prefill_final` (the serve prompt schedule, cold start).
    PrefillFinal,
}

struct Args {
    model: String,
    out: PathBuf,
    scores: Vec<(String, PathBuf)>,
    max_chunks: Option<usize>,
    max_seq: usize,
    prefix: Prefix,
    rank_check: bool,
    kv_mode: Option<String>,
    state_quant: Option<String>,
}

fn parse_args(args: &[String]) -> R<Args> {
    let mut a = Args {
        model: String::new(),
        out: PathBuf::new(),
        scores: Vec::new(),
        max_chunks: None,
        max_seq: 4096,
        prefix: Prefix::ChunkFinal,
        rank_check: true,
        kv_mode: None,
        state_quant: None,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag}: missing value\n{USAGE}"))
        };
        match flag.as_str() {
            "--model" => a.model = value()?,
            "--out" => a.out = PathBuf::from(value()?),
            "--score" => {
                let v = value()?;
                let (name, path) = v
                    .split_once(':')
                    .ok_or_else(|| format!("--score wants NAME:REF.kldref, got {v}"))?;
                if name.is_empty()
                    || !name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    return Err(format!("--score name {name:?} must be [A-Za-z0-9_-]+"));
                }
                if a.scores.iter().any(|(n, _)| n == name) {
                    return Err(format!("duplicate --score name {name}"));
                }
                a.scores.push((name.to_string(), PathBuf::from(path)));
            }
            "--max-chunks" => a.max_chunks = Some(value()?.parse().map_err(e)?),
            "--max-seq" => a.max_seq = value()?.parse().map_err(e)?,
            "--prefix" => {
                a.prefix = match value()?.as_str() {
                    "chunk-final" => Prefix::ChunkFinal,
                    "prefill-final" => Prefix::PrefillFinal,
                    other => return Err(format!("--prefix {other}: chunk-final|prefill-final")),
                }
            }
            "--no-rank-check" => a.rank_check = false,
            "--kv-mode" => a.kv_mode = Some(value()?),
            "--state-quant" => a.state_quant = Some(value()?),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if a.model.is_empty() || a.out.as_os_str().is_empty() || a.scores.is_empty() {
        return Err(USAGE.to_string());
    }
    Ok(a)
}

fn sha256_file(path: &Path) -> R<String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|x| format!("open {}: {x}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = file.read(&mut buf).map_err(e)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Rank `rank`'s logits mirror bytes (the row every EP rank holds after a forward).
fn peer_logits(bundle: &mut Qwen4Bundle, gpu: &mut Gpu, rank: usize) -> R<Vec<u8>> {
    bundle.with_rank_mut(gpu, rank, |_, peer_gpu, mirrors| {
        let mirror = mirrors
            .logits()
            .ok_or_else(|| format!("rank {rank} has no logits mirror"))?;
        peer_gpu.hip.device_synchronize().map_err(e)?;
        peer_gpu.download_raw_bytes(mirror).map_err(e)
    })
}

struct ScoreOut {
    name: String,
    mean_kld: f64,
    top1: f64,
    ppl: f64,
    rows: usize,
    logits_sha256: String,
    rank_rows_checked: usize,
    rank_rows_disagree: usize,
}

#[allow(clippy::too_many_arguments)]
fn score_one(
    args: &Args,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    name: &str,
    ref_path: &Path,
    ranks: usize,
) -> R<ScoreOut> {
    let reference = KldRef::open(ref_path).map_err(e)?;
    let file = std::fs::File::open(ref_path).map_err(e)?;
    // SAFETY: read-only map of a file this process does not modify.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(e)?;
    let h = &reference.header;
    let (n_ctx, window) = (h.n_ctx, h.window);
    let n_chunk = h.n_chunk.min(args.max_chunks.unwrap_or(usize::MAX));
    let tokens = reference.tokens(&mmap).map_err(e)?;
    let vocab = bundle.config.vocab_size;
    if vocab != h.n_vocab {
        return Err(format!(
            "model vocab {vocab} != reference vocab {}",
            h.n_vocab
        ));
    }
    if n_ctx + 1 > args.max_seq {
        return Err(format!(
            "--max-seq {} < reference n_ctx {n_ctx} + 1",
            args.max_seq
        ));
    }
    let (from, to) = (window.score_from, window.score_to);
    gpu.bind_thread().map_err(e)?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(e)?;
    let row_t = logits.sub_offset(0, vocab);
    eprintln!(
        "{TOOL} {name}: {n_chunk} chunks x {n_ctx} (decode route, {ranks} rank(s), prefix {:?}); teacher PPL {:.4} ({})",
        args.prefix, h.oracle.ppl, h.teacher.path
    );

    let started = Instant::now();
    let mut chunks = Vec::with_capacity(n_chunk);
    let (mut kld_sum, mut nll_sum, mut top1_hits, mut scored) = (0.0f64, 0.0f64, 0usize, 0usize);
    let mut digest = Sha256::new();
    let (mut rank_checked, mut rank_bad) = (0usize, 0usize);
    for c in 0..n_chunk {
        let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
        bundle.reset(gpu).map_err(e)?;
        if from > 0 {
            match args.prefix {
                Prefix::ChunkFinal => bundle
                    .forward_chunk_final(gpu, &chunk[..from], &row_t, None)
                    .map_err(e)?,
                Prefix::PrefillFinal => bundle
                    .prefill_final(gpu, &chunk[..from], 0, &row_t)
                    .map_err(e)?,
            }
        }
        let mut klds = Vec::with_capacity(window.scored_per_chunk());
        let mut chunk_nll = 0.0f64;
        for (i, pos) in (from..to).enumerate() {
            bundle
                .forward_token(gpu, chunk[pos], &row_t, None)
                .map_err(e)?;
            gpu.hip.device_synchronize().map_err(e)?;
            let rank0_bytes = gpu.download_raw_bytes(&row_t).map_err(e)?;
            if args.rank_check && ranks > 1 {
                for rank in 1..ranks {
                    let peer = peer_logits(bundle, gpu, rank)?;
                    rank_checked += 1;
                    if peer != rank0_bytes {
                        rank_bad += 1;
                        if rank_bad <= 5 {
                            eprintln!(
                                "{TOOL} {name}: RANK {rank} LOGITS DISAGREE chunk {c} pos {pos} (rank0 {} vs {})",
                                &sha256_hex(&rank0_bytes)[..16],
                                &sha256_hex(&peer)[..16]
                            );
                        }
                    }
                }
                gpu.bind_thread().map_err(e)?;
            }
            digest.update(&rank0_bytes);
            let row: Vec<f32> = rank0_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if row.len() != vocab || row.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite or short logits at chunk {c} pos {pos}"));
            }
            let block = reference.block(&mmap, c, i).map_err(e)?;
            let block = TopKBlock {
                residual_logprob: block.residual_logprob,
                top: block.top.to_vec(),
            };
            klds.push(kld_from_block(&block, &row).map_err(e)?);
            chunk_nll += nll_of(&row, chunk[pos + 1] as usize).map_err(e)?;
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32);
            top1_hits += usize::from(argmax == Some(block.top[0].0));
        }
        let n = klds.len();
        kld_sum += klds.iter().sum::<f64>();
        nll_sum += chunk_nll;
        scored += n;
        let mean_kld = klds.iter().sum::<f64>() / n as f64;
        klds.sort_by(f64::total_cmp);
        let p99_kld = klds[((n as f64 * 0.99).ceil() as usize).clamp(1, n) - 1];
        chunks.push(ChunkScore {
            mean_kld,
            p99_kld,
            mean_nll: chunk_nll / n as f64,
        });
        eprintln!(
            "  {name} chunk {}/{n_chunk}  KLD {mean_kld:.6}  running KLD {:.6}  rank-bad {rank_bad}  {:.0}s",
            c + 1,
            kld_sum / scored as f64,
            started.elapsed().as_secs_f64()
        );
    }
    let out_path = args.out.join(format!("{name}.kldseq"));
    kldseq::write(&out_path, &chunks).map_err(e)?;
    let mean_nll = nll_sum / scored as f64;
    let result = ScoreOut {
        name: name.to_string(),
        mean_kld: kld_sum / scored as f64,
        top1: top1_hits as f64 / scored as f64,
        ppl: mean_nll.exp(),
        rows: scored,
        logits_sha256: format!("{:x}", digest.finalize()),
        rank_rows_checked: rank_checked,
        rank_rows_disagree: rank_bad,
    };
    eprintln!(
        "{TOOL} {name}: mean KLD = {:.6}  mean NLL = {mean_nll:.6}  PPL = {:.4}  top1 = {:.4}  ({scored} tokens)  logits sha256 {} -> {}",
        result.mean_kld,
        result.ppl,
        result.top1,
        result.logits_sha256,
        out_path.display()
    );
    if rank_checked > 0 {
        eprintln!(
            "{TOOL} {name}: rank logits agreement: {} of {rank_checked} rank-row comparisons DISAGREE",
            rank_bad
        );
    }
    let receipt = serde_json::json!({
        "name": result.name,
        "ref": ref_path.display().to_string(),
        "ref_sha256": sha256_file(ref_path)?,
        "kldseq": out_path.display().to_string(),
        "kldseq_sha256": sha256_file(&out_path)?,
        "route": "decode (EP served route, teacher-forced forward_token)",
        "prefix": format!("{:?}", args.prefix),
        "n_chunk": n_chunk, "n_ctx": n_ctx, "score_from": from, "score_to": to,
        "ranks": ranks,
        "rows": result.rows,
        "mean_kld": result.mean_kld,
        "top1": result.top1,
        "ppl": result.ppl,
        "logits_sha256": result.logits_sha256,
        "rank_check": args.rank_check && ranks > 1,
        "rank_rows_checked": result.rank_rows_checked,
        "rank_rows_disagree": result.rank_rows_disagree,
        "wall_s": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        args.out.join(format!("{name}.json")),
        serde_json::to_string_pretty(&receipt).map_err(e)? + "\n",
    )
    .map_err(e)?;
    Ok(result)
}

fn run(args: &Args) -> R<()> {
    std::fs::create_dir_all(&args.out).map_err(e)?;
    let envs: serde_json::Map<String, serde_json::Value> = std::env::vars()
        .filter(|(k, _)| {
            ["HIPFIRE_", "HIP_", "ROCR_", "HSA_", "GPU_", "FABRIC_"]
                .iter()
                .any(|p| k.starts_with(p))
        })
        .map(|(k, v)| (k, serde_json::Value::String(v)))
        .collect();
    eprintln!("{TOOL}: loading {}", args.model);
    let mut gpu = Gpu::init().map_err(|x| format!("Gpu::init: {x:?}"))?;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let mut model = hipfire_loader::load_model(
        &args.model,
        args.max_seq,
        None,
        args.kv_mode.as_deref(),
        None,
        args.state_quant.as_deref(),
        &cask,
        1,
        hipfire_runtime::loader_api::SpecLoadCfg::default(),
        &mut gpu,
    )?;
    let bundle = model
        .qwen4_mut()
        .ok_or("loaded model is not a Qwen4 (Flash-Next) bundle")?;
    let ranks = bundle.ranks();
    let ep = bundle.ep_mesh().is_some();
    if ranks < 2 || !ep {
        eprintln!(
            "{TOOL}: WARNING: ranks={ranks} ep_mesh={ep}: this is NOT an expert-parallel load (set HIPFIRE_TOPOLOGY_EP=2 and [topology] ep = 2)"
        );
    }
    eprintln!(
        "{TOOL}: loaded: ranks={ranks} ep_mesh={ep} arch={} vocab={}",
        gpu.arch, bundle.config.vocab_size
    );
    let mut results = Vec::new();
    for (name, ref_path) in &args.scores {
        results.push(score_one(args, bundle, &mut gpu, name, ref_path, ranks)?);
    }
    let meta = serde_json::json!({
        "tool": TOOL,
        "model": args.model,
        "ranks": ranks, "ep_mesh": ep, "arch": gpu.arch,
        "max_seq": args.max_seq,
        "prefix": format!("{:?}", args.prefix),
        "kv_mode": args.kv_mode, "state_quant": args.state_quant,
        "env": envs,
        "scores": results.iter().map(|r| serde_json::json!({
            "name": r.name, "mean_kld": r.mean_kld, "top1": r.top1, "ppl": r.ppl,
            "rows": r.rows, "logits_sha256": r.logits_sha256,
            "rank_rows_checked": r.rank_rows_checked, "rank_rows_disagree": r.rank_rows_disagree,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        args.out.join("meta.json"),
        serde_json::to_string_pretty(&meta).map_err(e)? + "\n",
    )
    .map_err(e)?;
    Ok(())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match parse_args(&argv).and_then(|a| run(&a)) {
        Ok(()) => 0,
        Err(msg) => {
            eprintln!("{TOOL}: {msg}");
            2
        }
    };
    std::process::exit(code);
}
