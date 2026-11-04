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
//!     [--route decode|lanes:K] [--max-chunks N] [--max-seq 4096] \
//!     [--prefix chunk-final|prefill-final] [--no-rank-check] [--kv-mode M] [--state-quant Q]
//! ```
//!
//! `--route lanes:K` (lab; K in 1..=7) scores the SAME references through the
//! lane / mesh verify path instead of one `forward_token` per row: the scored
//! chunks of each `--score` ref run eight at a time as eight continuous-batching
//! lanes of the MTP lane store (staged exactly as `fn_mesh_lane_oracle --spec mtp
//! --k K --ks 8`, EP mesh root + ranks). Each lane's unscored prefix goes through
//! the lane executor's prefill tiles; the scored tokens are then teacher-forced in
//! lane verify steps of K+1 rows per lane (`LaneRequest::forced`, the mesh's
//! emitter-forced window: rows `[seed, f1..f_K]`, every row consumed, the lane
//! advances K+1 positions per step), so every step is a 8x(K+1)-row lane verify
//! step of the same shape the oracle's K arm times. The step's resident verify
//! logits are read in `AcceptRule::observe_verify` (the oracle's
//! `--verify-logits-dump` hook) and each row goes through the same
//! `score_row` / `chunk_score` code as the decode route, so `NAME.kldseq` pairs
//! chunk for chunk and row for row with the decode route's. Lanes whose chunk ended
//! drop out (fewer rows in the step); `meta.json` records the observed step row-count
//! histogram and the planned one. Rank-mirror comparison does not apply to this route.
//! Run it under `HIPFIRE_QWEN4_EP_WIDE_VERIFY=1 HIPFIRE_QWEN4_EP_WIDE_VERIFY_ROWS=48` etc.
//! (the oracle's flag set F; 8x(K+1) rows must fit the wide cap).
//! One model load scores every `--score NAME:REF`; `DIR/NAME.kldseq` and
//! `DIR/NAME.json` (mean KLD, top-1, PPL, logits sha256, per-rank logits
//! agreement) are written, `DIR/meta.json` records binary / model / env /
//! topology. Under EP the rank-1.. logits mirrors are compared byte for byte
//! with rank 0's row on every scored row (`--no-rank-check` skips it); a
//! disagreement is counted in the receipt and does not stop the run.
//! Exit 0 = every score finished; 2 = refusal or error.
//! `NAME.json` and `meta.json` also carry `chunk_logits_sha256` (one sha256 per chunk's
//! row bytes) so two routes' logits compare chunk for chunk: the decode route's
//! `logits_chunks_sha256` equals the lanes route's `logits_sha256` when every row is
//! byte-identical.
//! HFKSEQ v2 holds per-chunk numbers only (mean / p99 KLD, mean NLL), so each score also writes
//! `NAME.rowkld`, every scored row's KLD (f64 LE, chunk-major, header `ROWKLD01` n_chunk per),
//! the row-level pairing key between two runs.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::ep_mesh_lanes::{EpMeshLanes, MeshLaneAdmission, MeshStepError};
use hipfire_arch_qwen4::gpu_forward_lanes::LANE_MAX_ROWS;
use hipfire_arch_qwen4::lane::{LabRoutes, LaneSpecConfig};
use hipfire_arch_qwen4::mesh_lane_protocol::{
    AcceptRule, LanePlan, LaneRequest, SamplerOutput, SamplingMode, WindowBudgets,
};
use hipfire_arch_qwen4::sampled_lane::LaneRng;
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use hipfire_runtime::spec::GreedyAccept;
use rayon::prelude::*;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use saddle_quant::eval::estimator::{kld_from_block, nll_of, TopKBlock};
use saddle_quant::format::kldref::KldRef;
use saddle_quant::format::kldseq::{self, ChunkScore};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

const TOOL: &str = "qwen4_ep_kld";
const USAGE: &str =
    "usage: qwen4_ep_kld --model M --out DIR --score NAME:REF.kldref [--score ...] \
[--route decode|lanes:K] [--max-chunks N] [--max-seq N] [--prefix chunk-final|prefill-final] \
[--no-rank-check] [--kv-mode M] [--state-quant Q]";

/// Process-wide host counters read before/after a token (lab diagnosis of rank-asymmetric host stalls).
#[derive(Clone, Copy, Default)]
struct HostProbe {
    majflt: u64,
    read_bytes: u64,
    psi_mem_us: u64,
    psi_io_us: u64,
    vol_ctx: u64,
    invol_ctx: u64,
}

impl HostProbe {
    fn read() -> Self {
        let mut p = Self::default();
        if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
            // Field 12 (majflt) counts from after the ")" closing the command name.
            if let Some(rest) = stat.rsplit(')').next() {
                p.majflt = rest
                    .split_whitespace()
                    .nth(9)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
        }
        if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
            for l in io.lines() {
                if let Some(v) = l.strip_prefix("read_bytes: ") {
                    p.read_bytes = v.trim().parse().unwrap_or(0);
                }
            }
        }
        let psi = |path: &str| -> u64 {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|t| {
                    t.lines().find(|l| l.starts_with("some")).and_then(|l| {
                        l.split_whitespace()
                            .find_map(|w| w.strip_prefix("total=")?.parse().ok())
                    })
                })
                .unwrap_or(0)
        };
        p.psi_mem_us = psi("/proc/pressure/memory");
        p.psi_io_us = psi("/proc/pressure/io");
        // All threads of the process (every rank's host thread): sum the per-task context switches.
        if let Ok(dir) = std::fs::read_dir("/proc/self/task") {
            for t in dir.flatten() {
                if let Ok(st) = std::fs::read_to_string(t.path().join("status")) {
                    for l in st.lines() {
                        if let Some(v) = l.strip_prefix("voluntary_ctxt_switches:") {
                            p.vol_ctx += v.trim().parse().unwrap_or(0u64);
                        } else if let Some(v) = l.strip_prefix("nonvoluntary_ctxt_switches:") {
                            p.invol_ctx += v.trim().parse().unwrap_or(0u64);
                        }
                    }
                }
            }
        }
        p
    }

    fn since(&self, before: &Self) -> Self {
        Self {
            majflt: self.majflt.saturating_sub(before.majflt),
            read_bytes: self.read_bytes.saturating_sub(before.read_bytes),
            psi_mem_us: self.psi_mem_us.saturating_sub(before.psi_mem_us),
            psi_io_us: self.psi_io_us.saturating_sub(before.psi_io_us),
            vol_ctx: self.vol_ctx.saturating_sub(before.vol_ctx),
            invol_ctx: self.invol_ctx.saturating_sub(before.invol_ctx),
        }
    }
}

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

/// Which scoring route runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// One teacher-forced `forward_token` per scored row (the default).
    Decode,
    /// Eight lanes, lane verify steps of `k + 1` forced rows per lane.
    Lanes { k: usize },
}

impl Route {
    fn parse(v: &str) -> R<Self> {
        if v == "decode" {
            return Ok(Route::Decode);
        }
        let k = v
            .strip_prefix("lanes:")
            .ok_or_else(|| format!("--route {v}: decode|lanes:K"))?;
        let k: usize = k
            .parse()
            .map_err(|_| format!("--route {v}: K must be an integer"))?;
        if k == 0 || k + 1 > LANE_MAX_ROWS {
            return Err(format!(
                "--route {v}: K must be in 1..={} (K + 1 verify rows per lane)",
                LANE_MAX_ROWS - 1
            ));
        }
        Ok(Route::Lanes { k })
    }

    fn label(self) -> String {
        match self {
            Route::Decode => "decode".to_string(),
            Route::Lanes { k } => format!("lanes:{k}"),
        }
    }
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
    route: Route,
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
        route: Route::Decode,
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
            "--route" => a.route = Route::parse(&value()?)?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if a.model.is_empty() || a.out.as_os_str().is_empty() || a.scores.is_empty() {
        return Err(USAGE.to_string());
    }
    if matches!(a.route, Route::Lanes { .. }) && (a.kv_mode.is_some() || a.state_quant.is_some()) {
        return Err(
            "--route lanes:K loads like the oracle (no --kv-mode / --state-quant)".to_string(),
        );
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
    route: String,
    /// Route-specific receipt fields (merged into `NAME.json`, listed in `meta.json`).
    extra: serde_json::Map<String, serde_json::Value>,
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
    let mut chunk_sha: Vec<String> = Vec::with_capacity(n_chunk);
    let mut row_klds: Vec<f64> = Vec::new();
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
        let mut chunk_digest = Sha256::new();
        for (i, pos) in (from..to).enumerate() {
            let t_tok = Instant::now();
            let probe0 = HostProbe::read();
            bundle
                .forward_token(gpu, chunk[pos], &row_t, None)
                .map_err(e)?;
            let t_enq = t_tok.elapsed();
            gpu.hip.device_synchronize().map_err(e)?;
            let t_tot = t_tok.elapsed();
            if t_tot.as_millis() > 120 {
                let d = HostProbe::read().since(&probe0);
                eprintln!(
                    "{TOOL} {name}: SLOW TOKEN t_end={} chunk {c} pos {pos}: forward_token returned after {} ms, device sync after {} ms; \
                     during it: major faults {}, disk read {} KiB, memory-PSI some {} ms, io-PSI some {} ms, ctx switches (vol/invol) {}/{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |t| t.as_millis()),
                    t_enq.as_millis(),
                    t_tot.as_millis(),
                    d.majflt,
                    d.read_bytes / 1024,
                    d.psi_mem_us / 1000,
                    d.psi_io_us / 1000,
                    d.vol_ctx,
                    d.invol_ctx
                );
            }
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
            chunk_digest.update(&rank0_bytes);
            let row = decode_row(&rank0_bytes, vocab)
                .map_err(|m| format!("{m} at chunk {c} pos {pos}"))?;
            let s = score_row(&reference, &mmap, c, i, &row, chunk[pos + 1])?;
            klds.push(s.kld);
            chunk_nll += s.nll;
            top1_hits += usize::from(s.top1);
        }
        row_klds.extend_from_slice(&klds);
        let n = klds.len();
        let (cs, chunk_kld_sum) = chunk_score(&mut klds, chunk_nll);
        kld_sum += chunk_kld_sum;
        nll_sum += chunk_nll;
        scored += n;
        let mean_kld = cs.mean_kld;
        chunks.push(cs);
        chunk_sha.push(format!("{:x}", chunk_digest.finalize()));
        eprintln!(
            "  {name} chunk {}/{n_chunk}  KLD {mean_kld:.6}  running KLD {:.6}  rank-bad {rank_bad}  {:.0}s",
            c + 1,
            kld_sum / scored as f64,
            started.elapsed().as_secs_f64()
        );
    }
    finish_score(Finish {
        args,
        name,
        ref_path,
        chunks: &chunks,
        kld_sum,
        nll_sum,
        top1_hits,
        scored,
        logits_sha256: format!("{:x}", digest.finalize()),
        logits_sha256_kind: "stream",
        chunk_sha,
        row_klds,
        rank_rows_checked: rank_checked,
        rank_rows_disagree: rank_bad,
        rank_check: args.rank_check && ranks > 1,
        n_chunk,
        n_ctx,
        from,
        to,
        ranks,
        started,
        route: "decode (EP served route, teacher-forced forward_token)".to_string(),
        prefix: format!("{:?}", args.prefix),
        extra: serde_json::Map::new(),
    })
}

// ------------------------------------------------------------ shared scoring

/// One logits row from its raw little-endian f32 bytes; refuses a short or non-finite row.
fn decode_row(bytes: &[u8], vocab: usize) -> R<Vec<f32>> {
    let row: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if row.len() != vocab || row.iter().any(|v| !v.is_finite()) {
        return Err("non-finite or short logits".to_string());
    }
    Ok(row)
}

struct RowScore {
    kld: f64,
    nll: f64,
    top1: bool,
}

/// The estimator every route uses for one scored row: KLD against reference block `(c, i)`,
/// NLL of `next`, and whether the argmax is the teacher's top-1.
fn score_row(
    reference: &KldRef,
    mmap: &memmap2::Mmap,
    c: usize,
    i: usize,
    row: &[f32],
    next: u32,
) -> R<RowScore> {
    let block = reference.block(mmap, c, i).map_err(e)?;
    let block = TopKBlock {
        residual_logprob: block.residual_logprob,
        top: block.top.to_vec(),
    };
    let kld = kld_from_block(&block, row).map_err(e)?;
    let nll = nll_of(row, next as usize).map_err(e)?;
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32);
    Ok(RowScore {
        kld,
        nll,
        top1: argmax == Some(block.top[0].0),
    })
}

/// `NAME.rowkld`: the per-row KLDs behind the HFKSEQ per-chunk means (the only row-level pairing
/// key): 16-byte header `ROWKLD01` + n_chunk:u32 LE + rows_per_chunk:u32 LE, then
/// `n_chunk * rows_per_chunk` f64 LE, chunk-major in row order.
fn write_rowkld(path: &Path, n_chunk: usize, per: usize, rows: &[f64]) -> R<()> {
    if rows.len() != n_chunk * per {
        return Err(format!(
            "rowkld: {} rows for {n_chunk} chunks x {per}",
            rows.len()
        ));
    }
    let mut out = Vec::with_capacity(16 + rows.len() * 8);
    out.extend_from_slice(b"ROWKLD01");
    out.extend_from_slice(&(n_chunk as u32).to_le_bytes());
    out.extend_from_slice(&(per as u32).to_le_bytes());
    for v in rows {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|x| format!("write {}: {x}", path.display()))
}

/// One chunk's HFKSEQ entry from its per-row KLDs (in row order) and summed NLL, plus the
/// chunk's KLD sum (taken before the p99 sort, so the global sum order is unchanged).
fn chunk_score(klds: &mut [f64], chunk_nll: f64) -> (ChunkScore, f64) {
    let n = klds.len();
    let sum = klds.iter().sum::<f64>();
    let mean_kld = klds.iter().sum::<f64>() / n as f64;
    klds.sort_by(f64::total_cmp);
    let p99_kld = klds[((n as f64 * 0.99).ceil() as usize).clamp(1, n) - 1];
    (
        ChunkScore {
            mean_kld,
            p99_kld,
            mean_nll: chunk_nll / n as f64,
        },
        sum,
    )
}

struct Finish<'a> {
    args: &'a Args,
    name: &'a str,
    ref_path: &'a Path,
    chunks: &'a [ChunkScore],
    kld_sum: f64,
    nll_sum: f64,
    top1_hits: usize,
    scored: usize,
    logits_sha256: String,
    /// `stream`: one sha256 over every row's bytes in chunk order; `chunks`: sha256 of the
    /// per-chunk hex digests joined with `\n` (what an interleaved route can produce).
    logits_sha256_kind: &'static str,
    chunk_sha: Vec<String>,
    /// Every scored row's KLD, chunk-major in row order (written as `NAME.rowkld`).
    row_klds: Vec<f64>,
    rank_rows_checked: usize,
    rank_rows_disagree: usize,
    rank_check: bool,
    n_chunk: usize,
    n_ctx: usize,
    from: usize,
    to: usize,
    ranks: usize,
    started: Instant,
    route: String,
    prefix: String,
    extra: serde_json::Map<String, serde_json::Value>,
}

/// Write `NAME.kldseq` and `NAME.json` and print the summary line (shared by every route).
fn finish_score(f: Finish<'_>) -> R<ScoreOut> {
    let (name, scored) = (f.name, f.scored);
    let out_path = f.args.out.join(format!("{name}.kldseq"));
    kldseq::write(&out_path, f.chunks).map_err(e)?;
    let rowkld_path = f.args.out.join(format!("{name}.rowkld"));
    write_rowkld(&rowkld_path, f.n_chunk, f.to - f.from, &f.row_klds)?;
    let mean_nll = f.nll_sum / scored as f64;
    let logits_chunks_sha256 = sha256_hex(f.chunk_sha.join("\n").as_bytes());
    let result = ScoreOut {
        name: name.to_string(),
        mean_kld: f.kld_sum / scored as f64,
        top1: f.top1_hits as f64 / scored as f64,
        ppl: mean_nll.exp(),
        rows: scored,
        logits_sha256: f.logits_sha256,
        rank_rows_checked: f.rank_rows_checked,
        rank_rows_disagree: f.rank_rows_disagree,
        route: f.route.clone(),
        extra: f.extra.clone(),
    };
    eprintln!(
        "{TOOL} {name}: mean KLD = {:.6}  mean NLL = {mean_nll:.6}  PPL = {:.4}  top1 = {:.4}  ({scored} tokens)  logits sha256 {} -> {}",
        result.mean_kld,
        result.ppl,
        result.top1,
        result.logits_sha256,
        out_path.display()
    );
    if f.rank_rows_checked > 0 {
        eprintln!(
            "{TOOL} {name}: rank logits agreement: {} of {} rank-row comparisons DISAGREE",
            f.rank_rows_disagree, f.rank_rows_checked
        );
    }
    let mut receipt = serde_json::json!({
        "name": result.name,
        "ref": f.ref_path.display().to_string(),
        "ref_sha256": sha256_file(f.ref_path)?,
        "kldseq": out_path.display().to_string(),
        "kldseq_sha256": sha256_file(&out_path)?,
        "rowkld": rowkld_path.display().to_string(),
        "rowkld_sha256": sha256_file(&rowkld_path)?,
        "route": f.route,
        "prefix": f.prefix,
        "n_chunk": f.n_chunk, "n_ctx": f.n_ctx, "score_from": f.from, "score_to": f.to,
        "ranks": f.ranks,
        "rows": result.rows,
        "mean_kld": result.mean_kld,
        "top1": result.top1,
        "ppl": result.ppl,
        "logits_sha256": result.logits_sha256,
        "logits_sha256_kind": f.logits_sha256_kind,
        "logits_chunks_sha256": logits_chunks_sha256,
        "chunk_logits_sha256": f.chunk_sha,
        "rank_check": f.rank_check,
        "rank_rows_checked": result.rank_rows_checked,
        "rank_rows_disagree": result.rank_rows_disagree,
        "wall_s": f.started.elapsed().as_secs_f64(),
    });
    if let serde_json::Value::Object(m) = &mut receipt {
        for (k, v) in f.extra {
            m.insert(k, v);
        }
    }
    std::fs::write(
        f.args.out.join(format!("{name}.json")),
        serde_json::to_string_pretty(&receipt).map_err(e)? + "\n",
    )
    .map_err(e)?;
    Ok(result)
}

// ------------------------------------------------ lane-batch scheduling (pure)

/// Lanes per lane group: the staged store's `max_lanes` (the oracle's `--ks 8`).
const LANES: usize = 8;

/// Consecutive chunk ranges of at most `lanes` chunks; one range runs as one lane group.
fn lane_groups(n_chunk: usize, lanes: usize) -> Vec<std::ops::Range<usize>> {
    let lanes = lanes.max(1);
    (0..n_chunk)
        .step_by(lanes)
        .map(|s| s..(s + lanes).min(n_chunk))
        .collect()
}

/// One lane's share of a verify step: scored rows `first..first + rows` of its chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Take {
    lane: usize,
    first: usize,
    rows: usize,
}

/// The next verify step of a group: every lane with scored rows left takes up to `k + 1` of
/// them (the drafts are the reference tokens, so every row is consumed). Lanes that are done
/// do not appear; an empty result ends the group.
fn next_step(scored: &[usize], done: &[usize], k: usize) -> Vec<Take> {
    (0..scored.len())
        .filter_map(|lane| {
            let left = scored[lane] - done[lane];
            (left > 0).then(|| Take {
                lane,
                first: done[lane],
                rows: left.min(k + 1),
            })
        })
        .collect()
}

/// Every step of a group with per-lane scored row counts `scored`.
fn replay_schedule(scored: &[usize], k: usize) -> Vec<Vec<Take>> {
    let mut done = vec![0usize; scored.len()];
    let mut steps = Vec::new();
    loop {
        let step = next_step(scored, &done, k);
        if step.is_empty() {
            return steps;
        }
        for t in &step {
            done[t.lane] += t.rows;
        }
        steps.push(step);
    }
}

/// Planned `total rows in a step -> number of steps` over every group of `n_chunk` chunks of
/// `per` scored rows each.
fn planned_rows_hist(n_chunk: usize, per: usize, k: usize) -> BTreeMap<usize, usize> {
    let mut hist = BTreeMap::new();
    for group in lane_groups(n_chunk, LANES) {
        for step in replay_schedule(&vec![per; group.len()], k) {
            *hist.entry(step.iter().map(|t| t.rows).sum()).or_insert(0) += 1;
        }
    }
    hist
}

fn hist_json(h: &BTreeMap<usize, usize>) -> serde_json::Value {
    serde_json::Value::Object(
        h.iter()
            .map(|(rows, steps)| (rows.to_string(), serde_json::json!(steps)))
            .collect(),
    )
}

#[derive(Default)]
struct LaneStats {
    groups: usize,
    prefill_steps: usize,
    verify_steps: usize,
    /// total rows in a step -> steps
    step_rows: BTreeMap<usize, usize>,
    /// lanes in a step -> steps
    step_lanes: BTreeMap<usize, usize>,
    prefill_wall_s: f64,
    admit_wall_s: f64,
    verify_wall_s: f64,
    score_wall_s: f64,
}

impl LaneStats {
    fn observe(&mut self, takes: &[Take]) {
        self.verify_steps += 1;
        *self
            .step_rows
            .entry(takes.iter().map(|t| t.rows).sum())
            .or_insert(0) += 1;
        *self.step_lanes.entry(takes.len()).or_insert(0) += 1;
    }
}

// ------------------------------------------------------------- lane plumbing

/// One request's rows in a step (the oracle's `RowSpec`).
struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    /// Absolute position of the first row.
    start: usize,
    kind: RequestStepKind,
}

/// One step's plan: per-slot rows for every slot `0..max_lanes` (idle slots empty), requests in
/// slot order, absolute positions (the oracle's `build_plan`).
fn build_plan(max_lanes: usize, mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> =
        (0..max_lanes).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let mut plan = BatchStepPlan {
        batch: SlotBatch::build(&triples),
        ..Default::default()
    };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows {
            epoch: r.epoch,
            rows: RowRange {
                begin,
                len: r.tokens.len(),
            },
            kind: r.kind,
        });
        match r.kind {
            RequestStepKind::Ar => plan.decode_rows += r.tokens.len(),
            RequestStepKind::Prefill => plan.prefill_rows += r.tokens.len(),
            RequestStepKind::Verify { .. } => plan.verify_rows += r.tokens.len(),
            RequestStepKind::Forced => plan.forced_rows += r.tokens.len(),
        }
        begin += r.tokens.len();
    }
    plan
}

/// provision -> forward -> commit of one prefill plan through the bundle's lane executor (the
/// oracle's `LocalLanes::step` without the logits read); a failure after provisioning aborts.
fn run_plan(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    plan: &BatchStepPlan,
) -> R<Vec<hipfire_runtime::slot_batch::RequestAdvance>> {
    let mut ex = bundle.lane_executor().map_err(e)?;
    ex.provision_step(gpu, plan)
        .map_err(|m| format!("provision_step failed: {m}"))?;
    let out = match ex.forward_step(gpu, plan) {
        Ok(out) => out,
        Err(m) => {
            ex.abort_step(plan);
            return Err(format!("forward_step failed: {m}"));
        }
    };
    match ex.commit_step(gpu, plan, out) {
        Ok(advances) => Ok(advances),
        Err(m) => {
            ex.abort_step(plan);
            Err(format!("commit_step failed: {m}"))
        }
    }
}

fn read_view(gpu: &Gpu, t: &GpuTensor, offset: usize, elems: usize) -> R<Vec<u8>> {
    let view = t.sub_offset(offset, elems);
    let bytes = view.byte_size();
    if bytes > view.buf.size() {
        return Err(format!(
            "read of {bytes} bytes exceeds the {}-byte device buffer",
            view.buf.size()
        ));
    }
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &view.buf).map_err(e)?;
    Ok(out)
}

fn mesh_step_error(error: MeshStepError) -> String {
    match error {
        MeshStepError::Preflight(v) => format!("mesh preflight rejected: {v:?}"),
        MeshStepError::Aborted(record) => format!("mesh step aborted: {record}"),
        MeshStepError::Protocol(p) => format!("mesh protocol: {p}"),
    }
}

/// The accept rule of the forced windows: never asked to accept (forced lanes consume every
/// row), it only reads the step's resident verify logits in `observe_verify`, which the root
/// calls on rank 0's thread after every rank's verify forward and before any prepare (the
/// oracle's `--verify-logits-dump` `VerifyDump` hook): rows are in plan order, lanes concatenated.
struct RowCapture {
    /// SAFETY: the rank-0 `Gpu` outlives the step that borrows this rule (`mesh.step` runs it on
    /// this thread); it is read only for the row download.
    gpu: *const Gpu,
    /// A view of the store's step logits.
    logits: GpuTensor,
    vocab: usize,
    /// Request tag -> the lane's rows of this step (raw f32 LE, row after row).
    rows: HashMap<u64, Vec<u8>>,
    error: Option<String>,
}

impl AcceptRule for RowCapture {
    fn accept(
        &mut self,
        _lane: &LanePlan,
        _drafts: &[u32],
        _picks: &[u32],
        _rng: &mut LaneRng,
    ) -> Result<GreedyAccept, String> {
        Err(
            "the lanes KLD route plans forced windows only; a decode window was asked to accept"
                .to_string(),
        )
    }

    fn observe_verify(&mut self, out: &SamplerOutput) {
        self.rows.clear();
        let mut first = 0usize;
        for (key, picks) in &out.picks {
            let n = picks.len();
            // SAFETY: see `gpu`.
            let gpu = unsafe { &*self.gpu };
            match read_view(gpu, &self.logits, first * self.vocab, n * self.vocab) {
                Ok(bytes) => {
                    self.rows.insert(key.request_epoch.request_tag, bytes);
                }
                Err(x) => {
                    self.error.get_or_insert(x);
                    return;
                }
            }
            first += n;
        }
    }
}

/// Everything `score_lanes` shares across a score's groups.
struct ScoreCtx<'a> {
    reference: &'a KldRef,
    mmap: &'a memmap2::Mmap,
    tokens: &'a [u32],
    n_ctx: usize,
    from: usize,
    to: usize,
    vocab: usize,
}

/// One lane's running score of its chunk.
struct LaneAcc {
    chunk: usize,
    klds: Vec<f64>,
    nll: f64,
    top1: usize,
    hasher: Sha256,
}

impl LaneAcc {
    fn new(chunk: usize, per: usize) -> Self {
        Self {
            chunk,
            klds: Vec::with_capacity(per),
            nll: 0.0,
            top1: 0,
            hasher: Sha256::new(),
        }
    }

    /// Score scored rows `first..first + rows` of this chunk from `bytes` (their raw logits rows).
    fn consume(&mut self, ctx: &ScoreCtx<'_>, first: usize, rows: usize, bytes: &[u8]) -> R<()> {
        let stride = ctx.vocab * 4;
        if bytes.len() != rows * stride {
            return Err(format!(
                "chunk {}: {} logits bytes for {rows} rows of {} vocab",
                self.chunk,
                bytes.len(),
                ctx.vocab
            ));
        }
        let chunk = &ctx.tokens[self.chunk * ctx.n_ctx..(self.chunk + 1) * ctx.n_ctx];
        for j in 0..rows {
            let i = first + j;
            let pos = ctx.from + i;
            if self.klds.len() != i {
                return Err(format!(
                    "chunk {}: row {i} arrives after {} scored rows",
                    self.chunk,
                    self.klds.len()
                ));
            }
            let rb = &bytes[j * stride..(j + 1) * stride];
            self.hasher.update(rb);
            let row = decode_row(rb, ctx.vocab)
                .map_err(|m| format!("{m} at chunk {} pos {pos}", self.chunk))?;
            let s = score_row(ctx.reference, ctx.mmap, self.chunk, i, &row, chunk[pos + 1])?;
            self.klds.push(s.kld);
            self.nll += s.nll;
            self.top1 += usize::from(s.top1);
        }
        Ok(())
    }
}

/// One admitted lane of a group.
struct LaneRun {
    epoch: RequestEpoch,
    slot: usize,
    chunk: usize,
    admitted: bool,
    in_mesh: bool,
}

/// Release every lane still held (mesh first, then the bundle), remembering the first error.
fn release_lanes(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    lanes: &mut [LaneRun],
) -> R<()> {
    let mut first_error = None;
    for lane in lanes.iter_mut() {
        if lane.in_mesh {
            lane.in_mesh = false;
            if let Err(x) = mesh.release_lane(bundle, gpu, lane.epoch) {
                first_error.get_or_insert(format!("mesh release {:?}: {x}", lane.epoch));
            }
        }
        if lane.admitted {
            lane.admitted = false;
            if let Err(x) = bundle.release_lane(&lane.epoch) {
                first_error.get_or_insert(format!("release {:?}: {x}", lane.epoch));
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// Admit and prefill the lanes of `group`, then teacher-force every scored row in lane verify
/// steps of `k + 1` rows per lane, scoring each row into `accs` (one per lane, chunk order).
#[allow(clippy::too_many_arguments)]
fn run_group(
    ctx: &ScoreCtx<'_>,
    k: usize,
    max_seq: usize,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    budgets: &WindowBudgets,
    next_tag: &mut u64,
    group: std::ops::Range<usize>,
    stats: &mut LaneStats,
    lanes: &mut Vec<LaneRun>,
    accs: &mut Vec<LaneAcc>,
) -> R<()> {
    let (from, per) = (ctx.from, ctx.to - ctx.from);
    let n = group.len();
    let chunk_tokens = |c: usize| &ctx.tokens[c * ctx.n_ctx..(c + 1) * ctx.n_ctx];
    for c in group {
        *next_tag += 1;
        let epoch = RequestEpoch {
            request_tag: *next_tag,
            owner_generation: 1,
        };
        let slot = bundle
            .admit_lane(gpu, epoch, chunk_tokens(c)[..from].to_vec())
            .map_err(e)?;
        lanes.push(LaneRun {
            epoch,
            slot,
            chunk: c,
            admitted: true,
            in_mesh: false,
        });
        accs.push(LaneAcc::new(c, per));
    }

    // Prefill: one tile per still-prefilling lane per step, the lane executor's singleton tile length.
    let t_prefill = Instant::now();
    let mut prefilled = vec![0usize; n];
    while prefilled.iter().any(|&p| p < from) {
        let mut specs = Vec::with_capacity(n);
        for (i, lane) in lanes.iter().enumerate() {
            if prefilled[i] >= from {
                continue;
            }
            let len = bundle
                .lane_executor()
                .map_err(e)?
                .exact_prefill_chunk_len(&lane.epoch)?;
            if len == 0 || prefilled[i] + len > from {
                return Err(format!(
                    "lane {i}: prefill tile of {len} rows at {}/{from}",
                    prefilled[i]
                ));
            }
            specs.push(RowSpec {
                slot: lane.slot,
                epoch: lane.epoch,
                tokens: chunk_tokens(lane.chunk)[prefilled[i]..prefilled[i] + len].to_vec(),
                start: prefilled[i],
                kind: RequestStepKind::Prefill,
            });
        }
        let plan = build_plan(LANES, specs);
        let advances = run_plan(bundle, gpu, &plan)?;
        stats.prefill_steps += 1;
        for (request, advance) in plan.requests.iter().zip(&advances) {
            let idx = lanes
                .iter()
                .position(|lane| lane.epoch == request.epoch)
                .ok_or("prefill plan epoch is not one of this group's lanes")?;
            prefilled[idx] += request.rows.len;
            if prefilled[idx] == from && advance.committed_ids.len() != 1 {
                return Err(format!(
                    "lane {idx}: prompt-completing tile committed {:?}",
                    advance.committed_ids
                ));
            }
        }
    }
    stats.prefill_wall_s += t_prefill.elapsed().as_secs_f64();
    mesh.note_prefill();

    // Every lane now holds its prefix; its next row is the reference token at `from`, not the
    // prefill's own pick: set the pending seed on every rank's table, then admit it to the mesh.
    let t_admit = Instant::now();
    for lane in lanes.iter_mut() {
        let chunk = chunk_tokens(lane.chunk);
        bundle
            .force_lane_seed(&lane.epoch, chunk[from])
            .map_err(e)?;
        mesh.admit_lane(
            bundle,
            gpu,
            MeshLaneAdmission {
                epoch: lane.epoch,
                position: from,
                seed: chunk[from],
                eos: None,
                rng_seed: lane.epoch.request_tag,
                prompt: chunk[..from].to_vec(),
                spec_k: Some(k),
            },
        )?;
        lane.in_mesh = true;
    }
    stats.admit_wall_s += t_admit.elapsed().as_secs_f64();

    // Teacher-forced verify steps.
    let scored = vec![per; n];
    let mut done = vec![0usize; n];
    loop {
        let takes = next_step(&scored, &done, k);
        if takes.is_empty() {
            break;
        }
        let mut requests = Vec::with_capacity(takes.len());
        for t in &takes {
            let lane = &lanes[t.lane];
            let chunk = chunk_tokens(lane.chunk);
            let at = from + t.first;
            let (position, seed) = mesh
                .root()
                .lane_position(lane.epoch)
                .ok_or_else(|| format!("lane {}: the root lost its position", t.lane))?;
            if position != at || seed != chunk[at] {
                return Err(format!(
                    "lane {}: root at position {position} seed {seed}, schedule expects {at} seed {}",
                    t.lane, chunk[at]
                ));
            }
            requests.push(LaneRequest {
                wide_verify: false,
                epoch: lane.epoch,
                remaining: per - t.first,
                context_left: max_seq.saturating_sub(position),
                forced: Some(chunk[at + 1..at + 1 + t.rows].to_vec()),
                sampling: SamplingMode::Greedy,
            });
        }
        let tensor = bundle
            .lanes()
            .and_then(|store| store.logits())
            .ok_or("the lane store has no step logits (not an output store)")?;
        let mut rule = RowCapture {
            gpu: &*gpu as *const Gpu,
            logits: tensor.sub_offset(0, tensor.numel()),
            vocab: ctx.vocab,
            rows: HashMap::new(),
            error: None,
        };
        let t_step = Instant::now();
        let report = mesh
            .step(bundle, gpu, &requests, budgets, &mut rule)
            .map_err(mesh_step_error)?;
        stats.verify_wall_s += t_step.elapsed().as_secs_f64();
        stats.observe(&takes);
        if let Some(x) = rule.error.take() {
            return Err(x);
        }
        if report.output.lanes.len() != takes.len() {
            return Err(format!(
                "step returned {} lane outputs for {} lanes",
                report.output.lanes.len(),
                takes.len()
            ));
        }
        for out in &report.output.lanes {
            let t = takes
                .iter()
                .find(|t| lanes[t.lane].epoch == out.key.request_epoch)
                .ok_or("step output epoch is not one of this step's lanes")?;
            let chunk = chunk_tokens(lanes[t.lane].chunk);
            let at = from + t.first;
            if !out.forced
                || out.consumed_rows != t.rows
                || out.emitted != chunk[at + 1..at + 1 + t.rows]
                || out.next_seed != chunk[at + t.rows]
            {
                return Err(format!(
                    "lane {}: forced window consumed {} of {} rows (forced {}, next seed {})",
                    t.lane, out.consumed_rows, t.rows, out.forced, out.next_seed
                ));
            }
        }
        let t_score = Instant::now();
        let mut payload: Vec<Option<(Take, Vec<u8>)>> = (0..n).map(|_| None).collect();
        for t in &takes {
            let bytes = rule
                .rows
                .remove(&lanes[t.lane].epoch.request_tag)
                .ok_or_else(|| format!("lane {}: no verify rows were captured", t.lane))?;
            payload[t.lane] = Some((*t, bytes));
        }
        accs.par_iter_mut()
            .zip(payload.par_iter())
            .try_for_each(|(acc, p)| match p {
                Some((t, bytes)) => acc.consume(ctx, t.first, t.rows, bytes),
                None => Ok(()),
            })?;
        stats.score_wall_s += t_score.elapsed().as_secs_f64();
        for t in &takes {
            done[t.lane] += t.rows;
        }
        // Lanes whose chunk ended leave.
        for t in &takes {
            if done[t.lane] == per {
                release_lanes(bundle, gpu, mesh, &mut lanes[t.lane..=t.lane])?;
            }
        }
    }
    Ok(())
}

/// Score one reference through the lane verify route: chunks run `LANES` at a time.
#[allow(clippy::too_many_arguments)]
fn score_lanes(
    args: &Args,
    k: usize,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    budgets: &WindowBudgets,
    next_tag: &mut u64,
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
    if from == 0 || to <= from || to >= n_ctx {
        return Err(format!(
            "the lanes route needs 0 < score_from < score_to < n_ctx, got {from}..{to} of {n_ctx}"
        ));
    }
    let per = to - from;
    if per != window.scored_per_chunk() {
        return Err(format!(
            "scored window {from}..{to} != scored_per_chunk {}",
            window.scored_per_chunk()
        ));
    }
    gpu.bind_thread().map_err(e)?;
    let planned = planned_rows_hist(n_chunk, per, k);
    eprintln!(
        "{TOOL} {name}: {n_chunk} chunks x {n_ctx} (lanes route K={k}, {LANES} lanes, {ranks} rank(s)); \
         planned step rows -> steps {planned:?}; teacher PPL {:.4} ({})",
        h.oracle.ppl, h.teacher.path
    );
    let ctx = ScoreCtx {
        reference: &reference,
        mmap: &mmap,
        tokens: &tokens[..],
        n_ctx,
        from,
        to,
        vocab,
    };

    let started = Instant::now();
    let mut stats = LaneStats::default();
    let mut chunks = Vec::with_capacity(n_chunk);
    let mut chunk_sha: Vec<String> = Vec::with_capacity(n_chunk);
    let mut row_klds: Vec<f64> = Vec::with_capacity(n_chunk * per);
    let (mut kld_sum, mut nll_sum, mut top1_hits, mut scored) = (0.0f64, 0.0f64, 0usize, 0usize);
    for group in lane_groups(n_chunk, LANES) {
        let mut lanes: Vec<LaneRun> = Vec::with_capacity(group.len());
        let mut accs: Vec<LaneAcc> = Vec::with_capacity(group.len());
        let first_chunk = group.start;
        let run = run_group(
            &ctx,
            k,
            args.max_seq,
            bundle,
            gpu,
            mesh,
            budgets,
            next_tag,
            group,
            &mut stats,
            &mut lanes,
            &mut accs,
        );
        let cleanup = release_lanes(bundle, gpu, mesh, &mut lanes);
        run?;
        cleanup?;
        stats.groups += 1;
        for acc in accs {
            if acc.klds.len() != per {
                return Err(format!(
                    "chunk {}: {} of {per} rows scored",
                    acc.chunk,
                    acc.klds.len()
                ));
            }
            row_klds.extend_from_slice(&acc.klds);
            let mut klds = acc.klds;
            let (cs, chunk_kld_sum) = chunk_score(&mut klds, acc.nll);
            kld_sum += chunk_kld_sum;
            nll_sum += acc.nll;
            top1_hits += acc.top1;
            scored += per;
            chunks.push(cs);
            chunk_sha.push(format!("{:x}", acc.hasher.finalize()));
        }
        eprintln!(
            "  {name} chunks {}..{}/{n_chunk}  running KLD {:.6}  {:.0}s (prefill {:.0}s verify {:.0}s score {:.0}s)",
            first_chunk + 1,
            chunks.len(),
            kld_sum / scored as f64,
            started.elapsed().as_secs_f64(),
            stats.prefill_wall_s,
            stats.verify_wall_s,
            stats.score_wall_s
        );
    }
    let observed = stats.step_rows.clone();
    let mut extra = serde_json::Map::new();
    extra.insert("k".into(), serde_json::json!(k));
    extra.insert("lanes".into(), serde_json::json!(LANES));
    extra.insert("lane_groups".into(), serde_json::json!(stats.groups));
    extra.insert("rows_per_lane_step".into(), serde_json::json!(k + 1));
    extra.insert("verify_steps".into(), serde_json::json!(stats.verify_steps));
    extra.insert(
        "prefill_steps".into(),
        serde_json::json!(stats.prefill_steps),
    );
    extra.insert("step_rows_hist".into(), hist_json(&observed));
    extra.insert("step_rows_hist_planned".into(), hist_json(&planned));
    extra.insert(
        "step_rows_hist_matches_plan".into(),
        serde_json::json!(observed == planned),
    );
    extra.insert("step_lanes_hist".into(), hist_json(&stats.step_lanes));
    extra.insert(
        "prefill_wall_s".into(),
        serde_json::json!(stats.prefill_wall_s),
    );
    extra.insert("admit_wall_s".into(), serde_json::json!(stats.admit_wall_s));
    extra.insert(
        "verify_wall_s".into(),
        serde_json::json!(stats.verify_wall_s),
    );
    extra.insert("score_wall_s".into(), serde_json::json!(stats.score_wall_s));
    let chunks_digest = sha256_hex(chunk_sha.join("\n").as_bytes());
    finish_score(Finish {
        args,
        name,
        ref_path,
        chunks: &chunks,
        kld_sum,
        nll_sum,
        top1_hits,
        scored,
        logits_sha256: chunks_digest,
        logits_sha256_kind: "chunks",
        chunk_sha,
        row_klds,
        rank_rows_checked: 0,
        rank_rows_disagree: 0,
        rank_check: false,
        n_chunk,
        n_ctx,
        from,
        to,
        ranks,
        started,
        route: format!("lanes:{k} (EP mesh lane verify, teacher-forced forced windows)"),
        prefix: "lane-exact-tiles".to_string(),
        extra,
    })
}

/// `load_model`'s admitted route with the per-load lane reservation and the MTP head armed, as
/// `fn_mesh_lane_oracle --spec mtp --reserve-lanes 8` loads: the loader reserves `LANES` MTP
/// lanes (depth `k`) in the EP expert placement. Planner capacity only; forward numerics unchanged.
fn load_lanes_model(
    args: &Args,
    k: usize,
    cask: &hipfire_runtime::loader_api::CaskConfig,
    gpu: &mut Gpu,
) -> R<hipfire_loader::LoadedModel> {
    use hipfire_loader::admission::{
        admit_source_with_options, legacy_warning, qwen4_kv_adaptive_requested,
        qwen_default_q8_enabled, BatchLoadRequest, KvBackendHints, KvBackendRequest,
        SourceAdmissionOptions,
    };
    let spec = hipfire_runtime::loader_api::SpecLoadCfg {
        mtp: Some(true),
        ..Default::default()
    };
    let mut admission = admit_source_with_options(
        &args.model,
        1,
        1,
        KvBackendRequest::Automatic,
        None,
        gpu.arch.as_str(),
        None,
        "auto",
        None,
        args.max_seq,
        KvBackendHints {
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: qwen_default_q8_enabled(),
            kv_adaptive: None,
            cask: Some(cask),
            deepseek4_heterogeneous: false,
            vmm_runtime_available: gpu.vmm_recommended_granularity().is_ok(),
            free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
        },
        SourceAdmissionOptions {
            spec,
            kv_adaptive: qwen4_kv_adaptive_requested(None),
            cask: cask.sidecar.is_some(),
            ..Default::default()
        },
    )?;
    admission.batch_request = BatchLoadRequest {
        vmm_batch: true,
        continuous_batch_size: LANES,
        row_budget: LANES,
        spec_k: Some(k),
        spec_copy: false,
    };
    let warning = legacy_warning(admission.kv_backend, admission.kv_backend_reason.as_deref());
    let loaded = hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        &args.model,
        args.max_seq,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        None,
        None,
        hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN,
        None,
        None,
        None,
        None,
        None,
        cask,
        1,
        spec,
        gpu,
    );
    if loaded.is_ok() {
        if let Some(warning) = warning {
            eprintln!("{warning}");
        }
    }
    loaded
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
    // Daemon parity: keep host-mapped expert memory out of reclaim before the first HIP call. Without it
    // KFD userptr invalidations evict every queue of the process for seconds under host memory pressure
    // (ROCm/rocm-systems#12528: GPUs idle, HIP's signal wait spins a core) and the fused AR's 500 ms
    // bounded wait trips (the EP2 route just stalls).
    hipfire_loader::prepare_host_memory_for(Path::new(&args.model));
    let mut gpu = Gpu::init().map_err(|x| format!("Gpu::init: {x:?}"))?;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let mut model = match args.route {
        Route::Decode => hipfire_loader::load_model(
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
        )?,
        Route::Lanes { k } => load_lanes_model(args, k, &cask, &mut gpu)?,
    };
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
    match args.route {
        Route::Decode => {
            for (name, ref_path) in &args.scores {
                results.push(score_one(args, bundle, &mut gpu, name, ref_path, ranks)?);
            }
        }
        Route::Lanes { k } => {
            if !ep {
                return Err("--route lanes:K needs an EP mesh load (HIPFIRE_TOPOLOGY_EP=2 and [topology] ep = 2)".to_string());
            }
            gpu.bind_thread().map_err(e)?;
            let before = gpu.device_mem_info().map(|(free, _)| free).unwrap_or(0);
            bundle
                .stage_lanes_with_copy(
                    &mut gpu,
                    LANES,
                    LANES,
                    Some(LaneSpecConfig {
                        k,
                        end_of_turn: None,
                        spec_lanes: None,
                    }),
                    None,
                )
                .map_err(e)?;
            let after = gpu.device_mem_info().map(|(free, _)| free).unwrap_or(0);
            eprintln!(
                "{TOOL}: lane stage (MTP k={k}) consumed {} MiB on rank 0 ({} MiB free after)",
                before.saturating_sub(after) >> 20,
                after >> 20
            );
            let mut mesh = EpMeshLanes::new(bundle, 1, false)?;
            let budgets = mesh.window_budgets(bundle, LabRoutes::Batched)?;
            if budgets.verify_rows_per_lane < k + 1 {
                return Err(format!(
                    "the staged store verifies {} rows per lane, K={k} needs {}",
                    budgets.verify_rows_per_lane,
                    k + 1
                ));
            }
            let mut next_tag = 0u64;
            for (name, ref_path) in &args.scores {
                results.push(score_lanes(
                    args,
                    k,
                    bundle,
                    &mut gpu,
                    &mut mesh,
                    &budgets,
                    &mut next_tag,
                    name,
                    ref_path,
                    ranks,
                )?);
            }
        }
    }
    let meta = serde_json::json!({
        "tool": TOOL,
        "model": args.model,
        "route": args.route.label(),
        "k": match args.route { Route::Decode => serde_json::Value::Null, Route::Lanes { k } => serde_json::json!(k) },
        "ranks": ranks, "ep_mesh": ep, "arch": gpu.arch,
        "max_seq": args.max_seq,
        "prefix": match args.route { Route::Decode => format!("{:?}", args.prefix), Route::Lanes { .. } => "lane-exact-tiles".to_string() },
        "kv_mode": args.kv_mode, "state_quant": args.state_quant,
        "env": envs,
        "scores": results.iter().map(|r| serde_json::json!({
            "name": r.name, "route": r.route, "mean_kld": r.mean_kld, "top1": r.top1, "ppl": r.ppl,
            "rows": r.rows, "logits_sha256": r.logits_sha256,
            "rank_rows_checked": r.rank_rows_checked, "rank_rows_disagree": r.rank_rows_disagree,
            "route_detail": r.extra,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    const BASE: &str = "--model m --out o --score wt2:/r/wt2.kldref";

    #[test]
    fn route_defaults_to_decode_and_parses_lanes_k() {
        assert_eq!(parse_args(&argv(BASE)).unwrap().route, Route::Decode);
        let a = parse_args(&argv(&format!("{BASE} --route lanes:3"))).unwrap();
        assert_eq!(a.route, Route::Lanes { k: 3 });
        assert_eq!(a.route.label(), "lanes:3");
        let a = parse_args(&argv(&format!("{BASE} --route decode"))).unwrap();
        assert_eq!(a.route, Route::Decode);
        assert_eq!(a.route.label(), "decode");
        let a = parse_args(&argv(&format!("{BASE} --route lanes:7 --max-chunks 8"))).unwrap();
        assert_eq!((a.route, a.max_chunks), (Route::Lanes { k: 7 }, Some(8)));
    }

    #[test]
    fn route_refuses_bad_values() {
        for bad in [
            "lanes", "lanes:", "lanes:0", "lanes:8", "lanes:x", "lanes:-1", "batch",
        ] {
            assert!(
                parse_args(&argv(&format!("{BASE} --route {bad}"))).is_err(),
                "--route {bad} must be refused"
            );
        }
        assert!(parse_args(&argv(&format!("{BASE} --route"))).is_err());
        assert!(parse_args(&argv(&format!("{BASE} --route lanes:3 --kv-mode q8"))).is_err());
        assert!(parse_args(&argv(&format!("{BASE} --route lanes:3 --state-quant q8"))).is_err());
        // the decode route keeps taking them
        assert!(parse_args(&argv(&format!("{BASE} --kv-mode q8"))).is_ok());
    }

    #[test]
    fn lane_groups_are_consecutive_chunks_of_at_most_eight() {
        assert_eq!(lane_groups(32, 8), vec![0..8, 8..16, 16..24, 24..32]);
        assert_eq!(lane_groups(24, 8), vec![0..8, 8..16, 16..24]);
        assert_eq!(lane_groups(10, 8), vec![0..8, 8..10]);
        assert_eq!(lane_groups(1, 8), vec![0..1]);
        assert!(lane_groups(0, 8).is_empty());
    }

    #[test]
    fn full_lanes_advance_k_plus_one_rows_per_step() {
        // 255 scored rows, K = 3: 63 steps of 8 x 4 = 32 rows and a last step of 8 x 3 = 24.
        let steps = replay_schedule(&[255; 8], 3);
        assert_eq!(steps.len(), 64);
        for (s, step) in steps.iter().enumerate() {
            assert_eq!(step.len(), 8, "every lane is in every step");
            for (lane, t) in step.iter().enumerate() {
                let rows = if s < 63 { 4 } else { 3 };
                assert_eq!(
                    *t,
                    Take {
                        lane,
                        first: s * 4,
                        rows
                    }
                );
            }
        }
        let hist = planned_rows_hist(32, 255, 3);
        assert_eq!(hist, BTreeMap::from([(24, 4), (32, 4 * 63)]));
        // code24: 24 chunks = 3 groups.
        assert_eq!(
            planned_rows_hist(24, 255, 3),
            BTreeMap::from([(24, 3), (32, 3 * 63)])
        );
    }

    #[test]
    fn a_short_last_group_runs_fewer_lanes() {
        // 10 chunks: a group of 8 and a group of 2; K = 3, 255 rows.
        assert_eq!(
            planned_rows_hist(10, 255, 3),
            BTreeMap::from([(24, 1), (32, 63), (6, 1), (8, 63)])
        );
    }

    #[test]
    fn lanes_whose_chunk_ended_drop_out_and_every_row_is_taken_once() {
        // Unequal lanes (the scheduler is per-lane generic): 5, 2 and 7 scored rows, K = 1.
        let steps = replay_schedule(&[5, 2, 7], 1);
        let t = |lane, first, rows| Take { lane, first, rows };
        assert_eq!(
            steps,
            vec![
                vec![t(0, 0, 2), t(1, 0, 2), t(2, 0, 2)],
                vec![t(0, 2, 2), t(2, 2, 2)],
                vec![t(0, 4, 1), t(2, 4, 2)],
                vec![t(2, 6, 1)],
            ]
        );
        // coverage: each lane's takes are contiguous from 0 and add up to its scored rows.
        for (lane, want) in [5usize, 2, 7].into_iter().enumerate() {
            let mut next = 0;
            for step in &steps {
                for take in step.iter().filter(|t| t.lane == lane) {
                    assert_eq!(take.first, next);
                    assert!((1..=2).contains(&take.rows));
                    next += take.rows;
                }
            }
            assert_eq!(next, want);
        }
        assert!(next_step(&[5, 2, 7], &[5, 2, 7], 1).is_empty());
    }

    #[test]
    fn one_row_per_step_is_the_decode_shape() {
        // K = 0 is refused by the CLI, but the scheduler's k + 1 = 1 degenerates to one row a step.
        let steps = replay_schedule(&[3, 3], 0);
        assert_eq!(steps.len(), 3);
        assert!(steps.iter().all(|s| s.iter().all(|t| t.rows == 1)));
    }

    #[test]
    fn chunk_score_keeps_the_sum_order_and_p99() {
        let mut klds = vec![0.3, 0.1, 0.2];
        let (cs, sum) = chunk_score(&mut klds, 3.0);
        assert_eq!(sum, 0.3 + 0.1 + 0.2);
        assert_eq!(cs.mean_kld, (0.3 + 0.1 + 0.2) / 3.0);
        assert_eq!(cs.mean_nll, 1.0);
        assert_eq!(cs.p99_kld, 0.3);
    }
}
