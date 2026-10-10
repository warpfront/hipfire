// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! VMM continuous-batch state oracle (PLAN §6 Slice2B, §7 acceptance item 2).
//!
//! cb_vmm_state_oracle <model> --ks 1,2,3,4,5,6,7,8 --contexts 512,8192,32768
//!     --steps 256 --out <absolute.json> [--short 256] [--artifacts <absolute dir>]
//!     [--phase all|singleton|probe] [--stop-repeats N] [--drift-cycles N]
//!     [--spec off|mtp|dflash|mixed] [--mtp-head <path>]
//!     [--dflash-draft <path>] [--dflash-probe-window <N>]
//!
//! DFlash Gate 0 (`--spec dflash|mixed`): a one-window replay probe. For each
//! lane the isolated singleton `DflashSpeculator` (prefill + `N` full-block
//! windows, then one window with output budget `E`) is the reference; the
//! same pre-window state is rebuilt per lane and the window is replayed as
//! 1/2/3(/4) lanes through one `forward_prefill_batch_multi` (ChainVerify
//! fusion + per-lane hidden ring + tape), one shared head, and
//! `dflash_greedy_accept_commit_parts`. Raw named byte arrays (trunk/draft
//! state, verify rows, tape, ring, KV rows) are compared with first-byte
//! difference reporting. `--dflash-probe-window N` = number of full windows
//! run before the probed one (0 = the prefill boundary). `--phase probe`
//! skips the AR singleton/controls/batch phases (probe only).
//!
//! Exact wide verify chunks (PLAN-CHUNK64 §5, `--spec dflash|mixed`): the
//! probe also replays aggregate-boundary windows (`--aggregates`, default
//! 63,64,65,96,128 rows of whole short per-request blocks) as lanes of one
//! shared trunk, once through a single `forward_prefill_batch_multi` (exact
//! iff the rows fit the effective cap, else refused before any write) and
//! once through `dflash_cb_verify` (whole lanes packed into chunks of the
//! effective cap: ONE 128-row chunk at C8 with `HIPFIRE_CB_VERIFY_CHUNK128=1`).
//! Every lane must byte-equal the isolated singleton window (state, KV rows,
//! tape, ring, picks, committed ids, and the KV rows just past the window).
//! Per window the profile collector's launched symbols must show exactly the
//! frozen wide symbols (`..._vt{4,8}w{4,8}_k32` for `HIPFIRE_CB_VERIFY_PM=0`,
//! `mq4_verify_*_pm_gfx1201_bt{4,8}w{4,8}` otherwise) with the per-chunk
//! launch counts of the model, never an IU4/quantized launch, and a refused
//! window launches nothing. Controls: over-cap and lone-64-row refusals,
//! graph-capture / replay-recording refusals before mutation (then the real
//! window still matches), byte-flip and lane-swap negative controls. Both
//! flags and the effective cap are in the receipt env. `--peer <dir>` also
//! compares each lane against another run's frozen arrays (that run needs
//! `--emit-refs 1`, plus `--emit-candidates 1` for candidate-vs-candidate):
//! run A with the flag off / PM=0, run B with the flag on / PM=1, B `--peer A`.
//!
//! `--contexts` are test prefix lengths, never max_seq overrides: the model
//! loads through the production `load_qwen35_bundle` with automatic VMM
//! sequence sizing, and the loaded ack (kv_backend/max_seq/K mode) is
//! recorded. Request r of a k-request case uses prefix length
//! `[contexts.., short][r % n]` and a request-unique prompt, so every k>1 case
//! mixes unequal prompts and contexts.
//!
//! Singleton phase (base route, runs first and is a self-test): every
//! distinct fixture is prefilled and decoded greedily `--steps` times on the
//! existing singleton route; logits/hidden per step and full KV prefix,
//! DeltaNet S/scales/EF and conv state at each stage boundary are written as
//! raw bytes. A second independent run must be byte-identical (any differing
//! byte fails). Negative controls must FAIL: wrong row position (row_slot),
//! another request's reference (epoch), and publishing a rejected verify tail
//! (rejected-tail read). A rejected verify with DN restored and the KV tail
//! left poisoned must still MATCH (mask positive control).
//!
//! Comparisons are actual byte comparisons; sha256 digests are receipts only.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, LayerType, Qwen35Config, StateQuant};
use hipfire_arch_qwen35::forward_slots::vmm::{Qwen35RequestState, Qwen35VmmStore, VmmRequestInit, VmmRoute};
use hipfire_arch_qwen35::forward_slots::vmm::spec::VmmSpecEngine;
use hipfire_arch_qwen35::mtp_head::{load_mtp_head, MtpKvMode, Qwen35MtpHead};
use hipfire_arch_qwen35::mtp_spec::cb::{mtp_cb_cycle, MtpCbLane, MtpCbScratch};
use hipfire_arch_qwen35::mtp_spec::{prefill_trunk_and_mtp_cache, MtpPromptRoute, MtpSamplingConfig, MtpSpecState};
use hipfire_arch_qwen35::mtp_speculator::Qwen35MtpDrafter;
use hipfire_arch_qwen35::dflash_cb::{dflash_cb_draft, dflash_cb_head_argmax, dflash_cb_verify, dflash_lane_draft, DflashCbDraftLane, DflashCbScratch, DflashCbVerifyLane, DflashVmmLaneState};
use hipfire_arch_qwen35::dflash_spec::{build_dflash_speculator, load_dflash_state, DflashSpeculator, DflashState};
use hipfire_arch_qwen35::qwen35::prefill::multi::{
    forward_prefill_batch_multi, multi_chunk_pack_cap, multi_chunk_row_cap, multi_chunk_wide_admitted, pack_whole_lanes, MultiChunkRequest,
    MultiChunkScratch, MULTI_CHUNK_MAX_ROWS, MULTI_CHUNK_PRODUCT_MAX_ROWS,
};
use hipfire_arch_qwen35::speculative::{
    dflash_greedy_accept_commit_parts, DflashCbDraft, DflashTargetParts, DflashVerifyOutput, ModelSlot, VerifyScratch,
};
use hipfire_runtime::dflash::{ring_segments, DraftCtxMode};
use hipfire_runtime::spec::{MtpDrafter, PrefillOutcome, SpecRequestConfig, Speculator};
use hipfire_arch_qwen35::{load_qwen35_bundle, Qwen35Bundle};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch, StepOutput,
};
use hipfire_runtime::sampler::SamplerConfig;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const CORPUS: &str = "The scheduler admits each request with its own key/value mapping, \
recurrent state and convolution ring. A committed step boundary is the only place where \
new work may join; rejected speculative rows are masked by absolute position and never \
published. Write a careful explanation of how a merge sort splits, recurses and merges, \
then implement it in Rust with tests, and finally discuss its cache behaviour on large \
inputs compared with an in-place quicksort.\n";

/// `--spec`: which speculative phases run after the AR phases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpecMode {
    Off,
    Mtp,
    Dflash,
    /// MTP singleton/batched phase plus the DFlash Gate 0 probe (the real
    /// mixed MTP+DFlash executor cases need the VMM DFlash engine).
    Mixed,
}

impl SpecMode {
    fn mtp(self) -> bool {
        matches!(self, SpecMode::Mtp | SpecMode::Mixed)
    }
    fn dflash(self) -> bool {
        matches!(self, SpecMode::Dflash | SpecMode::Mixed)
    }
}

struct Args {
    model: String,
    ks: Vec<usize>,
    contexts: Vec<usize>,
    short: usize,
    steps: usize,
    out: PathBuf,
    artifacts: PathBuf,
    /// Run the executor batch phase after the singleton phase.
    batch: bool,
    /// In-process repetitions of the stop_id case (repeatability check).
    stop_repeats: usize,
    /// Sequential 2-request cycles for the memory-drift check (0 = off).
    drift_cycles: usize,
    /// Speculative phases (`--spec off|mtp|dflash|mixed`).
    spec: SpecMode,
    /// Run the singleton AR reference + controls (false for `--phase probe`).
    ar_phase: bool,
    /// DFlash draft artifact (`--dflash-draft`), required by dflash/mixed.
    dflash_draft: Option<PathBuf>,
    /// Gate 0: full windows run before the probed window (0 = prefill boundary).
    dflash_probe_window: Option<usize>,
    /// MTP head sidecar (default: model path with extension `mtp`).
    mtp_head: Option<PathBuf>,
    /// Aggregate verify-row totals of the wide-route fixtures (`--aggregates`,
    /// default 63,64,65,96,128; empty = off).
    aggregates: Vec<usize>,
    /// Another run's `--artifacts` dir: this run's candidate arrays must equal
    /// that run's frozen singleton arrays (`--peer`).
    peer: Option<PathBuf>,
    /// Freeze every exact case's candidate arrays (`--emit-candidates 1`).
    emit_candidates: bool,
    /// Freeze singleton refs of refused cases too (`--emit-refs 1`).
    emit_refs: bool,
}

fn parse_list(s: &str) -> Result<Vec<usize>> {
    let v: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()?;
    if v.is_empty() || v.contains(&0) {
        return Err(format!("list {s:?} must be non-empty positive integers").into());
    }
    Ok(v)
}

fn parse_01(s: &str) -> Result<bool> {
    match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{s:?} must be 0 or 1").into()),
    }
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: cb_vmm_state_oracle <model> --ks 1,..,8 --contexts 512,8192,32768 \
                 --steps 256 --out <absolute.json> [--short 256] [--artifacts <absolute dir>] \
                 [--phase all|singleton|probe] [--stop-repeats N] [--drift-cycles N] \
                 [--spec off|mtp|dflash|mixed] [--mtp-head <path>] \
                 [--dflash-draft <path>] [--dflash-probe-window <N>] \
                 [--aggregates 63,64,65,96,128|off] [--peer <other run's absolute artifacts dir>] \
                 [--emit-candidates 0|1] [--emit-refs 0|1]";
    let mut batch = true;
    let mut stop_repeats = 1usize;
    let mut drift_cycles = 0usize;
    let (mut spec, mut mtp_head) = (SpecMode::Off, None);
    let mut ar_phase = true;
    let (mut dflash_draft, mut dflash_probe_window) = (None, None);
    let mut aggregates = vec![63usize, 64, 65, 96, 128];
    let mut aggregates_given = false;
    let (mut peer, mut emit_candidates, mut emit_refs) = (None::<PathBuf>, false, false);
    let model = raw.first().ok_or(usage)?.clone();
    let (mut ks, mut contexts, mut steps, mut out, mut short, mut artifacts) =
        (None, None, None, None, 256usize, None);
    let mut i = 1;
    while i < raw.len() {
        let val = raw.get(i + 1).ok_or(usage)?;
        match raw[i].as_str() {
            "--ks" => ks = Some(parse_list(val)?),
            "--contexts" => contexts = Some(parse_list(val)?),
            "--steps" => steps = Some(val.parse::<usize>()?),
            "--out" => out = Some(PathBuf::from(val)),
            "--short" => short = val.parse()?,
            "--artifacts" => artifacts = Some(PathBuf::from(val)),
            "--stop-repeats" => stop_repeats = val.parse::<usize>()?.max(1),
            "--drift-cycles" => drift_cycles = val.parse()?,
            "--spec" => {
                spec = match val.as_str() {
                    "off" => SpecMode::Off,
                    "mtp" => SpecMode::Mtp,
                    "dflash" => SpecMode::Dflash,
                    "mixed" => SpecMode::Mixed,
                    _ => return Err(usage.into()),
                }
            }
            "--mtp-head" => mtp_head = Some(PathBuf::from(val)),
            "--dflash-draft" => dflash_draft = Some(PathBuf::from(val)),
            "--dflash-probe-window" => dflash_probe_window = Some(val.parse::<usize>()?),
            "--aggregates" => {
                aggregates_given = true;
                aggregates = if val == "off" { Vec::new() } else { parse_list(val)? };
            }
            "--peer" => peer = Some(PathBuf::from(val)),
            "--emit-candidates" => emit_candidates = parse_01(val)?,
            "--emit-refs" => emit_refs = parse_01(val)?,
            "--phase" => match val.as_str() {
                "all" => batch = true,
                "singleton" => batch = false,
                "probe" => {
                    batch = false;
                    ar_phase = false;
                }
                _ => return Err(usage.into()),
            },
            other => return Err(format!("unknown flag {other}; {usage}").into()),
        }
        i += 2;
    }
    let out: PathBuf = out.ok_or(usage)?;
    if !out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    let artifacts = artifacts.unwrap_or_else(|| out.with_extension("artifacts"));
    if !artifacts.is_absolute() {
        return Err("--artifacts must be absolute".into());
    }
    let ks = ks.ok_or(usage)?;
    if ks.iter().any(|&k| k > 8) {
        return Err("--ks values must be 1..=8".into());
    }
    let steps = steps.ok_or(usage)?;
    if steps < 8 {
        return Err("--steps must be >= 8 (negative controls need 8 steps)".into());
    }
    if spec.dflash() {
        let draft = dflash_draft.as_ref().ok_or("--spec dflash|mixed requires --dflash-draft <path>")?;
        if !draft.is_file() {
            return Err(format!("--dflash-draft {} is not a file", draft.display()).into());
        }
    } else if dflash_draft.is_some() || dflash_probe_window.is_some() {
        return Err("--dflash-draft / --dflash-probe-window require --spec dflash|mixed".into());
    }
    if !spec.dflash() && (aggregates_given || peer.is_some() || emit_candidates || emit_refs) {
        return Err("--aggregates / --peer / --emit-candidates / --emit-refs require --spec dflash|mixed".into());
    }
    if aggregates.iter().any(|&t| !(2..=256).contains(&t)) {
        return Err("--aggregates totals must be 2..=256".into());
    }
    if let Some(p) = &peer {
        if !p.is_absolute() || !p.join("dflash_gate0").is_dir() {
            return Err(format!("--peer {} must be an absolute artifacts dir containing dflash_gate0/", p.display()).into());
        }
    }
    if !ar_phase && spec != SpecMode::Dflash {
        return Err("--phase probe requires --spec dflash (the MTP phase compares against the AR singleton traces)".into());
    }
    Ok(Args {
        model, ks, contexts: contexts.ok_or(usage)?, short, steps, out, artifacts, batch, stop_repeats, drift_cycles,
        spec, ar_phase, dflash_draft, dflash_probe_window, mtp_head, aggregates, peer, emit_candidates, emit_refs,
    })
}

/// One isolated request fixture: request-unique prompt at an exact length.
#[derive(Clone)]
struct Fixture {
    request: usize,
    prefix: usize,
    tokens: Vec<u32>,
}

impl Fixture {
    fn name(&self) -> String {
        format!("r{}_p{}", self.request, self.prefix)
    }
}

fn fixtures(args: &Args, tok: &Tokenizer) -> Result<Vec<Fixture>> {
    let lens: Vec<usize> = args.contexts.iter().copied().chain([args.short]).collect();
    let max_k = *args.ks.iter().max().unwrap();
    let mut out = Vec::with_capacity(max_k);
    for r in 0..max_k {
        let prefix = lens[r % lens.len()];
        let head = tok.encode(&format!("Request {r} of the continuous-batch oracle.\n"));
        let body = tok.encode(CORPUS);
        if body.is_empty() || head.is_empty() {
            return Err("tokenizer produced empty fixture".into());
        }
        // Rotate the shared corpus per request so streams differ at every row.
        let tokens: Vec<u32> = head
            .iter()
            .copied()
            .chain(body.iter().copied().cycle().skip(r * 7))
            .take(prefix)
            .collect();
        out.push(Fixture { request: r, prefix, tokens });
    }
    Ok(out)
}

fn read_dev(gpu: &Gpu, t: &GpuTensor, offset: usize, bytes: usize) -> Result<Vec<u8>> {
    if offset + bytes > t.buf.size() {
        return Err(format!("read {offset}+{bytes} exceeds allocation {}", t.buf.size()).into());
    }
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh_at(&mut raw, &t.buf, offset)?;
    Ok(raw)
}

fn sha(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn argmax(logits: &[u8]) -> Result<u32> {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, c) in logits.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes(c.try_into().unwrap());
        if !v.is_finite() {
            return Err(format!("nonfinite logit at {i}").into());
        }
        if v > best.1 {
            best = (i, v);
        }
    }
    Ok(best.0 as u32)
}

/// Per-position KV row stride for the loaded cache encoding.
fn kv_row_bytes(kv: &KvCache) -> Result<(usize, usize)> {
    if kv.quant_fp8 {
        let r = KvCache::fp8_row_bytes(kv.n_kv_heads, kv.head_dim)?;
        return Ok((r, r));
    }
    if kv.quant_bf16 {
        let r = KvCache::bf16_row_bytes(kv.n_kv_heads, kv.head_dim)?;
        return Ok((r, r));
    }
    let rotated = kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4 || kv.quant_fwht;
    if kv.quant_q8 && !rotated {
        let r = kv.n_kv_heads * (kv.head_dim / 32) * 34;
        return Ok((r, r));
    }
    Err("oracle supports the resolved default fp8/q8 (and bf16) KV encodings only".into())
}

/// Named raw state bytes at a committed boundary of `rows` positions.
fn state_bytes(
    gpu: &Gpu,
    config: &Qwen35Config,
    kv: &KvCache,
    dn: &DeltaNetState,
    rows: usize,
) -> Result<Vec<(String, Vec<u8>)>> {
    gpu.hip.device_synchronize()?;
    if dn.quant != StateQuant::Q8 {
        return Err(format!("oracle state layout covers Q8 DeltaNet only (got {:?})", dn.quant).into());
    }
    // Logical Q8 layout (same as mq4_prefill_state_oracle): i8 codes and f16
    // EF over heads×Dv², f32 scales/conv by element count. Allocation sizes
    // may be rounded and are never compared.
    let n = config.linear_num_value_heads * config.linear_value_head_dim.pow(2);
    let (kr, vr) = kv_row_bytes(kv)?;
    let mut out = Vec::new();
    let mut la = 0;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        if *ty == LayerType::LinearAttention {
            out.push((format!("L{layer:02}.dn_s"), read_dev(gpu, &dn.s_matrices[la], 0, n)?));
            if let Some(t) = dn.s_scales.get(la) {
                out.push((format!("L{layer:02}.dn_scales"), read_dev(gpu, t, 0, t.numel() * 4)?));
            }
            if let Some(t) = dn.s_ef_residual.get(la) {
                out.push((format!("L{layer:02}.dn_ef"), read_dev(gpu, t, 0, n * 2)?));
            }
            let c = &dn.conv_states[la];
            out.push((format!("L{layer:02}.conv"), read_dev(gpu, c, 0, c.numel() * 4)?));
            la += 1;
        } else {
            out.push((format!("L{layer:02}.k"), read_dev(gpu, &kv.k_gpu[layer], 0, rows * kr)?));
            out.push((format!("L{layer:02}.v"), read_dev(gpu, &kv.v_gpu[layer], 0, rows * vr)?));
        }
    }
    Ok(out)
}

/// Full trace of one request: per-step logits/hidden + boundary states.
struct Trace {
    /// logits after prefill, then after each decode step.
    logits: Vec<PathBuf>,
    hidden: Vec<PathBuf>,
    committed: Vec<u32>,
    position: usize,
    pending_seed: u32,
    states: Vec<(String, PathBuf)>,
    /// Prompt chunk lengths the route executed, in order.
    prefill_chunks: Vec<usize>,
}

struct Ctx {
    gpu: Gpu,
    b: Qwen35Bundle,
}

impl Ctx {
    fn logits(&self) -> Result<Vec<u8>> {
        self.gpu.hip.device_synchronize()?;
        read_dev(&self.gpu, &self.b.scratch.logits, 0, self.b.config.vocab_size * 4)
    }
    fn hidden(&self) -> Result<Vec<u8>> {
        read_dev(&self.gpu, &self.b.scratch.x, 0, self.b.config.dim * 4)
    }
    fn reset(&mut self) -> Result<()> {
        self.b.dn_state.reset(&mut self.gpu)?;
        pin_gdn_frame(&self.b.dn_state);
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32], start: usize) -> Result<()> {
        let b = &mut self.b;
        qwen35::forward_prefill_batch(
            &mut self.gpu, &b.weights, &b.config, tokens, start, &mut b.kv_cache,
            &mut b.dn_state, &b.scratch, None, None, None, None,
        )?;
        Ok(())
    }
    /// Default singleton serve prefill: the same outer chunking as the serve
    /// route (`ordinary_prefill_chunk_limit` + `ordinary_serve_prefill_chunk_len`),
    /// re-evaluated before every chunk. Returns the executed chunk lengths.
    fn prefill_serve(&mut self, tokens: &[u32]) -> Result<Vec<usize>> {
        let mut chunks = Vec::new();
        let mut done = 0;
        while done < tokens.len() {
            let rem = tokens.len() - done;
            let b = &self.b;
            let ceiling = qwen35::ordinary_prefill_chunk_limit(&self.gpu, &b.weights, &b.config, &b.dn_state, &b.kv_cache, None)?;
            let len = qwen35::prefill::ordinary_serve_prefill_chunk_len(rem, ceiling).unwrap_or(rem.min(ceiling).max(1));
            self.prefill(&tokens[done..done + len], done)?;
            chunks.push(len);
            done += len;
        }
        Ok(chunks)
    }
    fn decode(&mut self, token: u32, pos: usize) -> Result<()> {
        let b = &mut self.b;
        qwen35::forward_scratch(
            &mut self.gpu, &b.weights, &b.config, token, pos, &mut b.kv_cache,
            &mut b.dn_state, &b.scratch,
        )?;
        Ok(())
    }
    fn snapshot_dn(&self) -> Result<Vec<Vec<u8>>> {
        self.gpu.hip.device_synchronize()?;
        let dn = &self.b.dn_state;
        dn.s_matrices
            .iter()
            .chain(&dn.s_scales)
            .chain(&dn.s_ef_residual)
            .chain(&dn.conv_states)
            .map(|t| read_dev(&self.gpu, t, 0, t.buf.size()))
            .collect()
    }
    fn restore_dn(&self, snap: &[Vec<u8>]) -> Result<()> {
        let dn = &self.b.dn_state;
        let all: Vec<&GpuTensor> = dn
            .s_matrices
            .iter()
            .chain(&dn.s_scales)
            .chain(&dn.s_ef_residual)
            .chain(&dn.conv_states)
            .collect();
        if all.len() != snap.len() {
            return Err("DN snapshot arity mismatch".into());
        }
        for (t, bytes) in all.iter().zip(snap) {
            self.gpu.hip.memcpy_htod(&t.buf, bytes)?;
        }
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }
    /// Overwrite KV rows [from, to) of every attention layer with 0x7f bytes.
    fn poison_kv(&mut self, from: usize, to: usize) -> Result<()> {
        self.b.kv_cache.ensure_mapped_capacity(&mut self.gpu, to)?;
        let (kr, vr) = kv_row_bytes(&self.b.kv_cache)?;
        for (layer, ty) in self.b.config.layer_types.iter().enumerate() {
            if *ty == LayerType::FullAttention {
                let kv = &self.b.kv_cache;
                self.gpu.hip.memcpy_htod_offset(&kv.k_gpu[layer].buf, from * kr, &vec![0x7f; (to - from) * kr])?;
                self.gpu.hip.memcpy_htod_offset(&kv.v_gpu[layer].buf, from * vr, &vec![0x7f; (to - from) * vr])?;
            }
        }
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }
}

/// With GDN error feedback off (`HIPFIRE_DN_STATE_EF=0`), Q8 DeltaNet requant
/// rounds stochastically off a process-global frame counter, so a request's
/// bytes depend on the counter at its start. Every reference run and every
/// executor request starts at frame 0 (the executor threads its frame per
/// request from admission). With EF on the counter is unused: no-op.
fn pin_gdn_frame(dn: &DeltaNetState) {
    if dn.s_ef_residual.is_empty() {
        rdna_compute::norm::restore_gdn_requant_frame_checkpoint(0);
    }
}

fn write_art(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Reference run: record every step to `dir`.
fn record(ctx: &mut Ctx, f: &Fixture, steps: usize, dir: &Path) -> Result<Trace> {
    fs::create_dir_all(dir.parent().ok_or("reference dir has no parent")?)?;
    fs::create_dir(dir)?; // never overwrite a previous reference
    ctx.reset()?;
    let prefill_chunks = ctx.prefill_serve(&f.tokens)?;
    let mut tr = Trace {
        logits: vec![],
        hidden: vec![],
        committed: vec![],
        position: f.prefix,
        pending_seed: 0,
        states: vec![],
        prefill_chunks,
    };
    let save_states = |ctx: &Ctx, tr: &mut Trace, stage: &str, rows: usize| -> Result<()> {
        for (name, bytes) in state_bytes(&ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, rows)? {
            let p = dir.join(stage).join(format!("{name}.bin"));
            write_art(&p, &bytes)?;
            tr.states.push((format!("{stage}/{name}"), p));
        }
        Ok(())
    };
    let l = ctx.logits()?;
    let p = dir.join("logits_0000.bin");
    write_art(&p, &l)?;
    tr.logits.push(p);
    tr.pending_seed = argmax(&l)?;
    save_states(ctx, &mut tr, "prefill", f.prefix)?;
    for s in 1..=steps {
        let tok = tr.pending_seed;
        ctx.decode(tok, tr.position)?;
        tr.committed.push(tok);
        tr.position += 1;
        let l = ctx.logits()?;
        let h = ctx.hidden()?;
        let (pl, ph) = (dir.join(format!("logits_{s:04}.bin")), dir.join(format!("hidden_{s:04}.bin")));
        write_art(&pl, &l)?;
        write_art(&ph, &h)?;
        tr.logits.push(pl);
        tr.hidden.push(ph);
        tr.pending_seed = argmax(&l)?;
    }
    let end = tr.position;
    save_states(ctx, &mut tr, "final", end)?;
    Ok(tr)
}

/// Byte comparison report: first differing (item, byte) and count of items.
#[derive(Default)]
struct Diff {
    compared_items: usize,
    compared_bytes: usize,
    differing_items: Vec<String>,
}

impl Diff {
    fn check(&mut self, item: &str, reference: &[u8], candidate: &[u8]) {
        self.compared_items += 1;
        self.compared_bytes += reference.len();
        if reference.len() != candidate.len() {
            self.differing_items.push(format!("{item}: len {} vs {}", reference.len(), candidate.len()));
        } else if let Some(i) = reference.iter().zip(candidate).position(|(a, b)| a != b) {
            let n = reference.iter().zip(candidate).filter(|(a, b)| a != b).count();
            self.differing_items.push(format!("{item}: first byte {i}, {n} bytes differ"));
        }
    }
    fn ids(&mut self, item: &str, reference: &[u32], candidate: &[u32]) {
        let to = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        self.check(item, &to(reference), &to(candidate));
    }
    fn exact(&self) -> bool {
        self.differing_items.is_empty() && self.compared_items > 0
    }
    fn json(&self) -> Value {
        json!({
            "exact": self.exact(),
            "compared_items": self.compared_items,
            "compared_bytes": self.compared_bytes,
            "differing_items": self.differing_items.len(),
            "first_differences": self.differing_items.iter().take(16).collect::<Vec<_>>(),
        })
    }
}

/// Replay `f` on the singleton route and byte-compare against `reference`.
/// `perturb` lets negative controls corrupt exactly one invariant.
#[derive(Clone, Copy, PartialEq)]
enum Perturb {
    None,
    /// Decode step 1 is written at position+1 (wrong row slot/position).
    RowSlot,
    /// Verify 4 rows at step 1, reject all drafts, restore DN, keep poisoned
    /// KV tail masked, then decode normally (must still match).
    RejectedTailMasked,
    /// Same verify, but publish the rejected tail: DN not restored and the
    /// committed position advances over the rejected rows (must fail).
    RejectedTailRead,
}

fn replay(ctx: &mut Ctx, f: &Fixture, steps: usize, reference: &Trace, perturb: Perturb) -> Result<Diff> {
    let mut d = Diff::default();
    let rd = |p: &Path| fs::read(p);
    ctx.reset()?;
    ctx.prefill_serve(&f.tokens)?;
    let l = ctx.logits()?;
    d.check("logits_0000", &rd(&reference.logits[0])?, &l);
    let mut seed = argmax(&l)?;
    let mut position = f.prefix;
    let mut committed = Vec::with_capacity(steps);
    let pre = state_bytes(&ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, f.prefix)?;
    for (name, bytes) in &pre {
        let (_, p) = reference.states.iter().find(|(n, _)| n == &format!("prefill/{name}")).ok_or("missing ref state")?;
        d.check(&format!("prefill/{name}"), &rd(p)?, bytes);
    }
    for s in 1..=steps {
        if s == 1 && matches!(perturb, Perturb::RejectedTailMasked | Perturb::RejectedTailRead) {
            let snap = ctx.snapshot_dn()?;
            // Spec rollback also restores the stochastic-requant frame (EF off).
            let frame = rdna_compute::norm::gdn_requant_frame_checkpoint();
            ctx.poison_kv(position, position + 4)?;
            // Drafts that the target will reject: off-by-one token ids.
            let block = [seed, seed.wrapping_add(1) % 1000, 17, 23];
            ctx.prefill(&block, position)?;
            if perturb == Perturb::RejectedTailRead {
                committed.extend_from_slice(&block);
                position += block.len();
                seed = argmax(&ctx.logits()?)?;
            } else {
                ctx.restore_dn(&snap)?;
                rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame);
            }
        }
        let pos = if perturb == Perturb::RowSlot && s == 1 { position + 1 } else { position };
        ctx.decode(seed, pos)?;
        committed.push(seed);
        position += 1;
        let l = ctx.logits()?;
        let h = ctx.hidden()?;
        d.check(&format!("logits_{s:04}"), &rd(&reference.logits[s])?, &l);
        d.check(&format!("hidden_{s:04}"), &rd(&reference.hidden[s - 1])?, &h);
        seed = argmax(&l)?;
    }
    d.ids("committed_ids", &reference.committed, &committed);
    d.ids("position", &[reference.position as u32], &[position as u32]);
    d.ids("pending_seed", &[reference.pending_seed], &[seed]);
    let fin = state_bytes(&ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, reference.position)?;
    for (name, bytes) in &fin {
        let (_, p) = reference.states.iter().find(|(n, _)| n == &format!("final/{name}")).ok_or("missing ref state")?;
        d.check(&format!("final/{name}"), &rd(p)?, bytes);
    }
    Ok(d)
}

fn load(model: &str) -> Result<(Ctx, Tokenizer, Value)> {
    let hfq = HfqFile::open(Path::new(model))?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("{e:?}"))?;
    let meta: Value = serde_json::from_str(&hfq.metadata_json)?;
    let cfg = meta.get("config").unwrap_or(&meta);
    let model_ctx = cfg
        .get("text_config")
        .unwrap_or(cfg)
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .ok_or("checkpoint has no max_position_embeddings")? as usize;
    drop(hfq);
    let mut gpu = Gpu::init()?;
    let src = ModelSource::from_path(model)?;
    let cask = CaskConfig::default();
    let qwen_default_q8 = !matches!(
        hipfire_config::developer_var("HIPFIRE_QWEN_KV_DEFAULT_Q8").ok().as_deref(),
        Some("0")
    );
    // Automatic sequence sizing exactly as admission's "pending" bound: the
    // carrier re-measures the card after weights. No max_seq/kv_backend override.
    let mut lc = LoadCtx {
        path: model,
        max_seq: model_ctx,
        sequence: Some(SequenceHint { model_ctx, automatic: true, card_cap: 0 }),
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
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
    };
    let b = load_qwen35_bundle(src, &mut lc).map_err(|e| format!("load_qwen35_bundle: {e}"))?;
    let max_seq = lc.max_seq;
    let kv = &b.kv_cache;
    let ack = json!({
        "kv_backend": if kv.uses_vmm_backend() { "vmm" } else { "legacy" },
        "kv_backend_legacy": !kv.uses_vmm_backend(),
        "max_seq_bound": max_seq,
        "model_ctx": model_ctx,
        "kv_k_mode": format!("{:?}", kv.current_kv_mode().ok()),
        "kv_fp8": kv.quant_fp8,
        "kv_q8": kv.quant_q8,
        "dn_quant": format!("{:?}", b.dn_state.quant),
        "dn_state_ef": !b.dn_state.s_ef_residual.is_empty(),
        "arch": gpu.arch.clone(),
        "qwen_default_q8": qwen_default_q8,
    });
    eprintln!("loaded ack: {ack}");
    if !kv.uses_vmm_backend() {
        return Err("loaded KV is not VMM; oracle refuses a legacy owner".into());
    }
    Ok((Ctx { gpu, b }, tok, ack))
}

// ── Batch phase: Qwen35VmmStore / executor (§4.2) ────────────────────

/// Requests in flight per store. Prefill chunk lengths come from the store's
/// `exact_prefill_chunk_len` (the singleton serve chunking), one prefill
/// request per step so the row budget is width + one chunk.
const WIDTH: usize = 8;

enum Sink {
    /// Isolated executor reference (k=1 on the batch route).
    Record { dir: PathBuf, trace: Trace },
    /// Batch request: byte-compare to the isolated executor reference (state
    /// bleed; must be exact) and to the singleton route (route exactness).
    Compare { b: usize, a: usize, vs_b: Diff, vs_a: Diff },
}

struct Req {
    fx: usize,
    epoch: RequestEpoch,
    slot: usize,
    /// Picks to commit (pick 0 comes from the final prompt chunk).
    max_tokens: usize,
    stop: Option<u32>,
    cancel_at: Option<usize>,
    fed: usize,
    committed: Vec<u32>,
    finish: Option<String>,
    store_watermark_errors: Vec<String>,
    /// Prompt chunk lengths the executor ran for this request.
    chunks: Vec<usize>,
    sink: Sink,
}

impl Req {
    fn new(fx: usize, tag: u64, generation: u64, slot: usize, max_tokens: usize, sink: Sink) -> Self {
        Self {
            fx,
            epoch: RequestEpoch { request_tag: tag, owner_generation: generation },
            slot,
            max_tokens,
            stop: None,
            cancel_at: None,
            fed: 0,
            committed: Vec::new(),
            finish: None,
            store_watermark_errors: Vec::new(),
            chunks: Vec::new(),
            sink,
        }
    }
}

struct Refs<'a> {
    b: &'a [Option<Trace>],
    a: &'a [Trace],
}

fn state_cmp(d: &mut Diff, stage: &str, now: &[(String, Vec<u8>)], reference: &Trace) -> Result<()> {
    for (name, bytes) in now {
        let key = format!("{stage}/{name}");
        match reference.states.iter().find(|(n, _)| *n == key) {
            Some((_, p)) => d.check(&key, &fs::read(p)?, bytes),
            None => d.differing_items.push(format!("{key}: missing in reference")),
        }
    }
    Ok(())
}

/// Record or compare pick `i` (logits after the head row, hidden of that row).
fn sink_pick(sink: &mut Sink, refs: &Refs, i: usize, logits: &[u8], hidden: &[u8]) -> Result<()> {
    match sink {
        Sink::Record { dir, trace } => {
            let pl = dir.join(format!("logits_{i:04}.bin"));
            write_art(&pl, logits)?;
            trace.logits.push(pl);
            if i > 0 {
                let ph = dir.join(format!("hidden_{i:04}.bin"));
                write_art(&ph, hidden)?;
                trace.hidden.push(ph);
            }
        }
        Sink::Compare { b, a, vs_b, vs_a } => {
            for (d, t) in [(vs_b, refs.b[*b].as_ref().ok_or("missing isolated reference")?), (vs_a, &refs.a[*a])] {
                match t.logits.get(i) {
                    Some(p) => d.check(&format!("logits_{i:04}"), &fs::read(p)?, logits),
                    None => d.differing_items.push(format!("logits_{i:04}: beyond reference")),
                }
                if i > 0 {
                    d.check(&format!("hidden_{i:04}"), &fs::read(&t.hidden[i - 1])?, hidden);
                }
            }
        }
    }
    Ok(())
}

fn sink_state(sink: &mut Sink, refs: &Refs, stage: &str, now: Vec<(String, Vec<u8>)>) -> Result<()> {
    match sink {
        Sink::Record { dir, trace } => {
            for (name, bytes) in now {
                let p = dir.join(stage).join(format!("{name}.bin"));
                write_art(&p, &bytes)?;
                trace.states.push((format!("{stage}/{name}"), p));
            }
        }
        Sink::Compare { b, a, vs_b, vs_a } => {
            state_cmp(vs_b, stage, &now, refs.b[*b].as_ref().ok_or("missing isolated reference")?)?;
            state_cmp(vs_a, stage, &now, &refs.a[*a])?;
        }
    }
    Ok(())
}

/// Frontier/terminal comparisons once a request stops committing.
fn sink_finish(sink: &mut Sink, refs: &Refs, r_committed: &[u32], position: usize, pending: Option<u32>) -> Result<()> {
    match sink {
        Sink::Record { trace, .. } => {
            // Reference convention: committed = fed picks, pending = last pick.
            let (fed, last) = r_committed.split_at(r_committed.len() - 1);
            trace.committed = fed.to_vec();
            trace.pending_seed = last[0];
            trace.position = position;
            if pending != Some(last[0]) {
                return Err("store pending_seed disagrees with last commit".into());
            }
        }
        Sink::Compare { b, a, vs_b, vs_a } => {
            for (d, t) in [(vs_b, refs.b[*b].as_ref().ok_or("missing isolated reference")?), (vs_a, &refs.a[*a])] {
                let picks: Vec<u32> = t.committed.iter().copied().chain([t.pending_seed]).collect();
                let n = r_committed.len().min(picks.len());
                d.ids("committed_ids", &picks[..n], r_committed);
                let prefix = t.position - t.committed.len();
                d.ids("position", &[(prefix + r_committed.len() - 1) as u32], &[position as u32]);
                d.ids("pending_seed", &[picks[r_committed.len() - 1]], &[pending.unwrap_or(u32::MAX)]);
            }
        }
    }
    Ok(())
}

/// Prompt chunk sequence the executor ran. Informational, not a parity
/// item: the admitted ceiling follows free VRAM (other live requests), and
/// the inner planner keeps 512-row state/KV commits, so chunking may differ
/// while every state byte matches. Recorded in the reference and case JSON.
fn sink_chunks(sink: &mut Sink, chunks: &[usize]) {
    if let Sink::Record { trace, .. } = sink {
        trace.prefill_chunks = chunks.to_vec();
    }
}

fn free_vram(gpu: &Gpu) -> Result<usize> {
    Ok(gpu.hip.get_vram_info()?.0)
}

fn new_store(ctx: &mut Ctx) -> Result<Qwen35VmmStore> {
    // The singleton phase leaves its optional widened prefill scratch cached
    // (32K prefix); the batch route never uses it, so release it first.
    if let Some(pbs) = ctx.b.scratch.widened_prefill_batch.borrow_mut().take() {
        pbs.free_gpu(&mut ctx.gpu)?;
    }
    // Exact route: prefill rows run on the singleton prefill and do not count
    // against the trunk row budget, so the budget is the decode width.
    // Shared physical KV budget: free VRAM minus a 6 GiB margin for
    // per-request DeltaNet state and transients. This is an admission
    // budget, not a max_seq override.
    let free = free_vram(&ctx.gpu)?;
    let budget = free.saturating_sub(6 << 30);
    eprintln!("batch store: route=Exact width={WIDTH} row_budget={WIDTH} free_vram={free} kv_budget={budget}");
    Ok(Qwen35VmmStore::new(&mut ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, WIDTH, WIDTH, budget, VmmRoute::Exact)?)
}

fn admit(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], r: &Req) -> Result<()> {
    let stop = r.stop.into_iter().collect();
    // Greedy sampler: plain argmax, the singleton reference semantics.
    let init = VmmRequestInit {
        prompt_len: fx[r.fx].prefix,
        stop_ids: stop,
        sampler: SamplerConfig::greedy(),
        rng_state: 0,
        history: vec![],
    };
    // Same starting requant frame as the singleton reference (EF-off only).
    pin_gdn_frame(&ctx.b.dn_state);
    let st = Qwen35RequestState::new_like(
        &mut ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, r.epoch, r.slot, init,
    )?;
    if let Err((st, e)) = store.admit(st) {
        st.free_gpu(&mut ctx.gpu)?;
        return Err(e.into());
    }
    Ok(())
}

fn retire(ctx: &mut Ctx, store: &mut Qwen35VmmStore, epoch: &RequestEpoch) -> Result<()> {
    store.retire(epoch)?.free_gpu(&mut ctx.gpu)?;
    Ok(())
}

/// Build the next step: one AR row per decoding request plus one prefill
/// chunk (first prefilling request in slot order) of the exact route's
/// singleton chunk length. Other prefilling requests idle (masked) this step.
fn build_plan(ctx: &Ctx, store: &Qwen35VmmStore, fx: &[Fixture], reqs: &[Req]) -> Result<BatchStepPlan> {
    let mut per: Vec<(usize, Vec<u32>, usize, Option<(RequestEpoch, RequestStepKind)>)> =
        (0..WIDTH).map(|s| (s, vec![], 0, None)).collect();
    let mut live: Vec<&Req> = reqs.iter().filter(|r| r.finish.is_none()).collect();
    live.sort_by_key(|r| r.slot);
    let mut prefill_taken = false;
    for r in live {
        store.request_state(&r.epoch).ok_or("live request missing from store")?;
        let f = &fx[r.fx];
        let (toks, start, kind) = if r.fed < f.prefix {
            if prefill_taken {
                continue;
            }
            prefill_taken = true;
            let len = store.exact_prefill_chunk_len(&ctx.gpu, &ctx.b.weights, &ctx.b.config, &r.epoch, f.prefix - r.fed)?;
            (f.tokens[r.fed..r.fed + len].to_vec(), r.fed, RequestStepKind::Prefill)
        } else {
            let seed = *r.committed.last().ok_or("decode without a pick")?;
            (vec![seed], f.prefix + r.committed.len() - 1, RequestStepKind::Ar)
        };
        per[r.slot] = (r.slot, toks, start, Some((r.epoch, kind)));
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p, _)| (SlotId(*s), t.as_slice(), *p)).collect();
    let batch = SlotBatch::build(&triples);
    let mut requests = Vec::new();
    let (mut begin, mut decode_rows, mut prefill_rows) = (0, 0, 0);
    for (_, t, _, who) in &per {
        if let Some((epoch, kind)) = who {
            requests.push(RequestRows { epoch: *epoch, rows: RowRange { begin, len: t.len() }, kind: *kind });
            match kind {
                RequestStepKind::Ar => decode_rows += t.len(),
                _ => prefill_rows += t.len(),
            }
        }
        begin += t.len();
    }
    Ok(BatchStepPlan { batch, requests, decode_rows, prefill_rows, verify_rows: 0, forced_rows: 0 })
}

/// Run one planned step and publish every head into its request's sink.
fn run_step(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], refs: &Refs, plan: &BatchStepPlan) -> Result<()> {
    let Ctx { gpu, b } = ctx;
    let out = {
        let mut ex = store.executor(&b.weights, &b.config, &b.scratch);
        ex.provision_step(gpu, plan)?;
        ex.forward_step(gpu, plan)?
    };
    let (vocab, dim) = (b.config.vocab_size, b.config.dim);
    // Device heads are valid until the next forward: read before commit.
    let mut heads = Vec::new();
    // Exact route: hidden() holds decode rows in plan order of the AR
    // requests; prefill rows have no hidden (pick 0 records none).
    let mut ar_ordinal = 0;
    for rr in &plan.requests {
        let last = rr.rows.end() - 1;
        let ar_row = (rr.kind == RequestStepKind::Ar).then(|| {
            ar_ordinal += 1;
            ar_ordinal - 1
        });
        if out.target_picks[last] != u32::MAX {
            let slot = plan.batch.row_slot[last] as usize;
            let l = read_dev(gpu, store.logits(), slot * vocab * 4, vocab * 4)?;
            let h = match ar_row {
                Some(row) => read_dev(gpu, store.hidden(), row * dim * 4, dim * 4)?,
                None => Vec::new(),
            };
            if argmax(&l)? != out.target_picks[last] {
                return Err(format!("device pick {} != host argmax of slot logits", out.target_picks[last]).into());
            }
            heads.push((rr.epoch, l, h));
        }
    }
    let advances = store.executor(&b.weights, &b.config, &b.scratch).commit_step(gpu, plan, out)?;
    for rr in &plan.requests {
        let r = reqs.iter_mut().find(|r| r.epoch == rr.epoch).ok_or("plan epoch not in case")?;
        if rr.kind == RequestStepKind::Prefill {
            r.fed += rr.rows.len;
            r.chunks.push(rr.rows.len);
            if r.fed == fx[r.fx].prefix {
                let st = store.request_state(&r.epoch).ok_or("request vanished")?;
                let now = state_bytes(gpu, &b.config, &st.kv, &st.dn, r.fed)?;
                sink_state(&mut r.sink, refs, "prefill", now)?;
                sink_chunks(&mut r.sink, &r.chunks);
            }
        }
    }
    if advances.len() != heads.len() {
        return Err(format!("{} advances for {} heads", advances.len(), heads.len()).into());
    }
    for (adv, (epoch, l, h)) in advances.iter().zip(heads) {
        if adv.epoch != epoch || adv.committed_ids.len() != 1 {
            return Err("advance order/arity differs from plan heads".into());
        }
        let r = reqs.iter_mut().find(|r| r.epoch == epoch).unwrap();
        let i = r.committed.len();
        r.committed.push(adv.committed_ids[0]);
        sink_pick(&mut r.sink, refs, i, &l, &h)?;
        let st = store.request_state(&epoch).ok_or("request vanished")?;
        let expect_pos = fx[r.fx].prefix + r.committed.len() - 1;
        if adv.committed_position != expect_pos || st.position != expect_pos || st.pending_seed != Some(adv.committed_ids[0]) {
            r.store_watermark_errors.push(format!(
                "pick {i}: advance pos {} store pos {} expected {expect_pos}, pending {:?}",
                adv.committed_position, st.position, st.pending_seed
            ));
        }
        if let Some(f) = &adv.finish {
            r.finish = Some(f.clone());
        } else if r.committed.len() >= r.max_tokens {
            r.finish = Some("max_tokens".into());
        }
    }
    Ok(())
}

/// Retire finished/cancelled requests, publishing final comparisons.
fn reap(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], refs: &Refs, steps: usize) -> Result<()> {
    for r in reqs.iter_mut() {
        if r.finish.is_none() && r.cancel_at.is_some_and(|c| r.committed.len() >= c) {
            r.finish = Some("cancelled".into());
        }
        let Some(fin) = r.finish.clone() else { continue };
        let Some(st) = store.request_state(&r.epoch) else { continue };
        if fin != "cancelled" && !r.committed.is_empty() {
            let pos = st.position;
            let pending = st.pending_seed;
            if r.committed.len() == steps + 1 {
                let now = state_bytes(&ctx.gpu, &ctx.b.config, &st.kv, &st.dn, pos)?;
                sink_state(&mut r.sink, refs, "final", now)?;
            }
            sink_finish(&mut r.sink, refs, &r.committed, pos, pending)?;
        }
        let _ = fx;
        retire(ctx, store, &r.epoch)?;
    }
    Ok(())
}

/// Admit every request, step until all finish. On any error every admitted
/// owner is aborted/retired so the store is reusable.
fn drive(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], refs: &Refs, steps: usize, controls: Option<&mut serde_json::Map<String, Value>>) -> Result<()> {
    let mut controls = controls;
    let res = (|| -> Result<()> {
        for r in reqs.iter() {
            admit(ctx, store, fx, r)?;
        }
        loop {
            reap(ctx, store, fx, reqs, refs, steps)?;
            if reqs.iter().all(|r| r.finish.is_some()) {
                return Ok(());
            }
            if let Some(map) = controls.as_deref_mut() {
                let live: Vec<usize> = (0..reqs.len()).filter(|&i| reqs[i].finish.is_none()).collect();
                if live.len() >= 2 && live.iter().all(|&i| reqs[i].fed == fx[reqs[i].fx].prefix && reqs[i].committed.len() >= 4) {
                    executor_controls(ctx, store, fx, reqs, &live, map)?;
                    controls = None;
                    continue;
                }
            }
            let plan = build_plan(ctx, store, fx, reqs)?;
            run_step(ctx, store, fx, reqs, refs, &plan).inspect_err(|_| store.abort_step(&plan))?;
        }
    })();
    if res.is_err() {
        for r in reqs.iter() {
            if store.request_state(&r.epoch).is_some() {
                let _ = retire(ctx, store, &r.epoch);
            }
        }
    }
    res
}

fn expect_err(map: &mut serde_json::Map<String, Value>, name: &str, r: std::result::Result<(), String>) {
    let ok = r.is_err();
    eprintln!("executor control {name}: refused={ok} -> {}", if ok { "OK" } else { "FAIL" });
    map.insert(name.into(), json!({"expected": "refused", "ok": ok, "error": r.err()}));
}

/// Executor-level negative/positive controls with two live decoding
/// requests. Refusals must leave state untouched: the case continues and
/// its byte comparisons cover that.
fn executor_controls(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], live: &[usize], map: &mut serde_json::Map<String, Value>) -> Result<()> {
    let (i0, i1) = (live[0], live[1]);
    let base = build_plan(ctx, store, fx, reqs)?;
    let try_plan = |ctx: &mut Ctx, store: &mut Qwen35VmmStore, plan: &BatchStepPlan| -> std::result::Result<(), String> {
        let Ctx { gpu, b } = ctx;
        let r = store.executor(&b.weights, &b.config, &b.scratch).provision_step(gpu, plan);
        if r.is_ok() {
            store.abort_step(plan); // provisioned only: no device writes, no poison
        }
        r
    };
    // row_slot: request0's AR row addressed at request1's slot.
    let mut p = base.clone();
    let row0 = p.requests.iter().find(|q| q.epoch == reqs[i0].epoch).unwrap().rows.begin;
    p.batch.row_slot[row0] = reqs[i1].slot as i32;
    expect_err(map, "neg_row_slot", try_plan(ctx, store, &p));
    // epoch: stale owner generation.
    let mut p = base.clone();
    p.requests[0].epoch.owner_generation += 1;
    expect_err(map, "neg_epoch_generation", try_plan(ctx, store, &p));
    // rejected-tail read: poison rows past request1's frontier (masked
    // positive control, verified by the case's exact comparison), then an
    // AR row whose position skips onto the poisoned tail must be refused.
    let pos1 = store.request_state(&reqs[i1].epoch).unwrap().position;
    {
        let Ctx { gpu, b } = ctx;
        let st = store.request_state_mut(&reqs[i1].epoch).unwrap();
        st.kv.ensure_mapped_capacity(gpu, pos1 + 4)?;
        let (kr, vr) = kv_row_bytes(&st.kv)?;
        for (layer, ty) in b.config.layer_types.iter().enumerate() {
            if *ty == LayerType::FullAttention {
                gpu.hip.memcpy_htod_offset(&st.kv.k_gpu[layer].buf, (pos1 + 1) * kr, &vec![0x7f; 3 * kr])?;
                gpu.hip.memcpy_htod_offset(&st.kv.v_gpu[layer].buf, (pos1 + 1) * vr, &vec![0x7f; 3 * vr])?;
            }
        }
        gpu.hip.device_synchronize()?;
    }
    map.insert("pos_rejected_tail_masked".into(), json!({"poisoned_rows": [pos1 + 1, pos1 + 4], "request": reqs[i1].fx, "verified_by": "case byte comparison"}));
    let mut p = base.clone();
    let row1 = p.requests.iter().find(|q| q.epoch == reqs[i1].epoch).unwrap().rows.begin;
    p.batch.positions[row1] += 2;
    expect_err(map, "neg_rejected_tail_read", try_plan(ctx, store, &p));
    // Stale commit on a step planned for request0 only: a foreign step id
    // must be refused. The step is then aborted after its forward, which
    // must poison request0 (device state written, never committed) while
    // request1 continues and must stay byte-exact over its poisoned tail.
    let solo = build_plan(ctx, store, fx, std::slice::from_ref(&reqs[i0]))?;
    {
        let Ctx { gpu, b } = ctx;
        let mut ex = store.executor(&b.weights, &b.config, &b.scratch);
        ex.provision_step(gpu, &solo)?;
        let out = ex.forward_step(gpu, &solo)?;
        let forged = StepOutput { step_id: out.step_id + 1, target_picks: out.target_picks.clone() };
        let r = ex.commit_step(gpu, &solo, forged).map(|_| ());
        store.abort_step(&solo);
        let ok = r.is_err();
        eprintln!("executor control neg_stale_commit: refused={ok} -> {}", if ok { "OK" } else { "FAIL" });
        map.insert("neg_stale_commit".into(), json!({"expected": "refused", "ok": ok, "error": r.err()}));
    }
    let poisoned = store.request_state(&reqs[i0].epoch).is_some_and(|s| s.poisoned)
        && store.request_state(&reqs[i1].epoch).is_some_and(|s| !s.poisoned);
    eprintln!("executor control poison_after_aborted_forward: {} -> {}", poisoned, if poisoned { "OK" } else { "FAIL" });
    map.insert("poison_after_aborted_forward".into(), json!({"ok": poisoned}));
    expect_err(map, "neg_poisoned_provision", try_plan(ctx, store, &base));
    reqs[i0].finish = Some("cancelled".into());
    Ok(())
}

fn diff_ok(sink: &Sink) -> (bool, bool, Value, Value) {
    match sink {
        Sink::Compare { vs_b, vs_a, .. } => (vs_b.exact(), vs_a.exact(), vs_b.json(), vs_a.json()),
        Sink::Record { .. } => (true, true, Value::Null, Value::Null),
    }
}

/// Per-case receipt; returns whether the batch-vs-isolated comparison is exact.
fn case_json(name: &str, fx: &[Fixture], reqs: &[Req]) -> (bool, bool, Value) {
    let (mut bleed_exact, mut route_exact) = (true, true);
    let rows: Vec<Value> = reqs
        .iter()
        .map(|r| {
            let (b, a, jb, ja) = diff_ok(&r.sink);
            let wm = r.store_watermark_errors.is_empty();
            bleed_exact &= b && wm;
            route_exact &= a;
            json!({
                "fixture": fx[r.fx].name(), "slot": r.slot, "epoch": [r.epoch.request_tag, r.epoch.owner_generation],
                "max_tokens": r.max_tokens, "stop": r.stop, "cancel_at": r.cancel_at, "prefill_chunks": r.chunks,
                "finish": r.finish, "committed": r.committed.len(),
                "watermarks_ok": wm, "watermark_errors": r.store_watermark_errors,
                "vs_isolated_executor": jb, "vs_singleton_route": ja,
            })
        })
        .collect();
    eprintln!("batch {name}: isolated-exact={bleed_exact} singleton-route-exact={route_exact}");
    (bleed_exact, route_exact, json!({"case": name, "bleed_exact": bleed_exact, "route_exact": route_exact, "requests": rows}))
}

fn compare_sink(i: usize) -> Sink {
    Sink::Compare { b: i, a: i, vs_b: Diff::default(), vs_a: Diff::default() }
}

/// Free VRAM and pool counters `(free, pool_new, pool_reused, pool_bytes_new)`.
fn mem_sample(gpu: &Gpu) -> Result<(usize, usize, usize, usize)> {
    gpu.hip.device_synchronize()?;
    let (n, r, b) = gpu.pool_stats();
    Ok((free_vram(gpu)?, n, r, b))
}

/// Memory drift over `cycles` sequential 2-request admit/run/retire cycles
/// (the two shortest fixtures, 8 picks each, byte-compared like every
/// case), plus isolation probes that attribute any drift: DeltaNet state
/// alone and a whole request owner (VMM KV + DN) allocated and freed with
/// no forward. Bounded = the last 10 cycles together lose < 16 MiB.
fn drift_check(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], refs: &Refs, supported: &[usize], cycles: usize) -> Result<(Value, bool)> {
    let mut by_len = supported.to_vec();
    by_len.sort_by_key(|&i| fx[i].prefix);
    let (c0, c1) = (by_len[0], by_len[1]);
    let mut rows = Vec::with_capacity(cycles);
    let mut exact_all = true;
    let start = mem_sample(&ctx.gpu)?;
    let mut refused: Option<(usize, String)> = None;
    for c in 0..cycles {
        let before = mem_sample(&ctx.gpu)?;
        let tag = 20_000 + 10 * c as u64;
        let mut reqs = vec![
            Req::new(c0, tag, 1, c % WIDTH, 8, compare_sink(c0)),
            Req::new(c1, tag + 1, 1, (c + 4) % WIDTH, 8, compare_sink(c1)),
        ];
        // `steps` far above 8 picks: no "final" (256-step) state comparison.
        if let Err(e) = drive(ctx, store, fx, &mut reqs, refs, usize::MAX - 1, None) {
            eprintln!("drift cycle {c}: REFUSED/FAILED: {e}");
            refused = Some((c, e.to_string()));
            break;
        }
        let after = mem_sample(&ctx.gpu)?;
        let exact = reqs.iter().all(|r| matches!(&r.sink, Sink::Compare { vs_b, vs_a, .. } if vs_b.exact() && vs_a.exact()) && r.store_watermark_errors.is_empty());
        exact_all &= exact;
        let d_free = after.0 as i64 - before.0 as i64;
        // Singleton's cached widened prefill scratch (exact prefill runs on it).
        let widened = ctx.b.scratch.widened_prefill_batch.borrow().as_ref().map(|p| p.max_batch);
        let store_mapped = store.mapped_kv_bytes()?;
        rows.push(json!({
            "cycle": c, "free_before": before.0, "free_after": after.0, "free_delta": d_free,
            "pool_new_delta": after.1 - before.1, "pool_reused_delta": after.2 - before.2,
            "pool_bytes_new_delta": after.3 - before.3, "widened_pbs_rows": widened,
            "store_mapped_after_retire": store_mapped, "exact": exact,
        }));
        eprintln!(
            "drift cycle {c}: free_before={} free_delta={d_free} pool_new+={} pool_reused+={} pool_bytes_new+={} widened_pbs={widened:?} store_mapped={store_mapped} exact={exact}",
            before.0, after.1 - before.1, after.2 - before.2, after.3 - before.3
        );
    }
    let end = mem_sample(&ctx.gpu)?;
    let deltas: Vec<i64> = rows.iter().map(|r| r["free_delta"].as_i64().unwrap()).collect();
    let last10: i64 = deltas.iter().rev().take(10).sum();
    let first10: i64 = deltas.iter().take(10).sum();
    // Bounded only if every cycle ran and the last 10 together lose < 16 MiB.
    let bounded = refused.is_none() && rows.len() >= 10 && last10 > -(16 << 20);
    // Attribution probes (no forward, no admit).
    let probe = |ctx: &mut Ctx, kind: &str, n: usize| -> Result<Value> {
        let s0 = mem_sample(&ctx.gpu)?;
        for i in 0..n {
            match kind {
                "dn_state" => {
                    let dn = DeltaNetState::new_with_quant(&mut ctx.gpu, &ctx.b.config, ctx.b.dn_state.quant)?;
                    dn.free_gpu(&mut ctx.gpu);
                }
                _ => {
                    let init = VmmRequestInit { prompt_len: 1, stop_ids: vec![], sampler: SamplerConfig::greedy(), rng_state: 0, history: vec![] };
                    let epoch = RequestEpoch { request_tag: 30_000 + i as u64, owner_generation: 1 };
                    let st = Qwen35RequestState::new_like(&mut ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, epoch, 0, init)?;
                    st.free_gpu(&mut ctx.gpu)?;
                }
            }
        }
        let s1 = mem_sample(&ctx.gpu)?;
        let v = json!({"iterations": n, "free_delta": s1.0 as i64 - s0.0 as i64, "pool_new_delta": s1.1 - s0.1, "pool_bytes_new_delta": s1.3 - s0.3});
        eprintln!("drift probe {kind}: {v}");
        Ok(v)
    };
    let p_dn = probe(ctx, "dn_state", 10).unwrap_or_else(|e| json!({"error": e.to_string()}));
    let p_owner = probe(ctx, "request_owner", 10).unwrap_or_else(|e| json!({"error": e.to_string()}));
    eprintln!(
        "drift: cycles_run={}/{cycles} total_free_delta={} first10={first10} last10={last10} bounded={bounded} exact={exact_all} refused={refused:?}",
        rows.len(),
        end.0 as i64 - start.0 as i64
    );
    let v = json!({
        "cycles_requested": cycles, "cycles_run": rows.len(),
        "refused": refused.as_ref().map(|(c, e)| json!({"cycle": c, "error": e})),
        "fixtures": [fx[c0].name(), fx[c1].name()], "picks_per_request": 8,
        "start": {"free": start.0, "pool_new": start.1, "pool_reused": start.2, "pool_bytes_new": start.3},
        "end": {"free": end.0, "pool_new": end.1, "pool_reused": end.2, "pool_bytes_new": end.3},
        "total_free_delta": end.0 as i64 - start.0 as i64, "first10_free_delta": first10, "last10_free_delta": last10,
        "bounded": bounded, "all_exact": exact_all,
        "probe_dn_state_x10": p_dn, "probe_request_owner_x10": p_owner, "per_cycle": rows,
    });
    Ok((v, bounded && exact_all))
}

/// Whole batch phase. Returns (report, pass).
fn batch_phase(ctx: &mut Ctx, args: &Args, fx: &[Fixture], refs_a: &[Trace]) -> Result<(Value, bool)> {
    let mut store = new_store(ctx)?;
    let steps = args.steps;
    let mut pass = true;
    let mut out = serde_json::Map::new();
    // Isolated executor references (k=1 on the batch route).
    let mut refs_b: Vec<Option<Trace>> = Vec::with_capacity(fx.len());
    let mut isolated = Vec::new();
    for (i, f) in fx.iter().enumerate() {
        let dir = args.artifacts.join("executor_k1").join(f.name());
        fs::create_dir_all(&dir)?;
        let trace = Trace { logits: vec![], hidden: vec![], committed: vec![], position: 0, pending_seed: 0, states: vec![], prefill_chunks: vec![] };
        let mut reqs = [Req::new(i, 1000 + i as u64, 1, i % WIDTH, steps + 1, Sink::Record { dir, trace })];
        let none: [Option<Trace>; 0] = [];
        let r = drive(ctx, &mut store, fx, &mut reqs, &Refs { b: &none, a: refs_a }, steps, None);
        let [req] = reqs;
        match (r, req.sink) {
            (Ok(()), Sink::Record { trace, .. }) => {
                // Isolated executor vs singleton route (route exactness at k=1).
                let mut d = Diff::default();
                let t = &refs_a[i];
                for (j, p) in trace.logits.iter().enumerate() {
                    d.check(&format!("logits_{j:04}"), &fs::read(&t.logits[j])?, &fs::read(p)?);
                }
                for (j, p) in trace.hidden.iter().enumerate() {
                    d.check(&format!("hidden_{:04}", j + 1), &fs::read(&t.hidden[j])?, &fs::read(p)?);
                }
                for (n, p) in &trace.states {
                    let (_, q) = t.states.iter().find(|(m, _)| m == n).ok_or("missing state")?;
                    d.check(n, &fs::read(q)?, &fs::read(p)?);
                }
                d.ids("committed_ids", &t.committed, &trace.committed);
                d.ids("position", &[t.position as u32], &[trace.position as u32]);
                d.ids("pending_seed", &[t.pending_seed], &[trace.pending_seed]);
                eprintln!("executor k1 {}: vs singleton route exact={}", f.name(), d.exact());
                isolated.push(json!({
                    "fixture": f.name(), "supported": true, "vs_singleton_route": d.json(),
                    "prefill_chunks_singleton": t.prefill_chunks, "prefill_chunks_executor": trace.prefill_chunks,
                }));
                refs_b.push(Some(trace));
            }
            (Err(e), _) => {
                eprintln!("executor k1 {}: REFUSED/FAILED: {e}", f.name());
                isolated.push(json!({"fixture": f.name(), "supported": false, "error": e.to_string()}));
                refs_b.push(None);
            }
            _ => unreachable!(),
        }
    }
    out.insert("executor_k1".into(), Value::Array(isolated));
    let refs = Refs { b: &refs_b, a: refs_a };
    let supported: Vec<usize> = (0..fx.len()).filter(|&i| refs_b[i].is_some()).collect();
    let mut cases = Vec::new();
    let mut route_exact_all = true;
    let mut run_case = |ctx: &mut Ctx, store: &mut Qwen35VmmStore, name: String, mut reqs: Vec<Req>, controls: bool, cases: &mut Vec<Value>, pass: &mut bool| -> Result<()> {
        let mut map = serde_json::Map::new();
        let r = drive(ctx, store, fx, &mut reqs, &refs, steps, controls.then_some(&mut map));
        let (bleed, route, mut j) = case_json(&name, fx, &reqs);
        route_exact_all &= route;
        let ctl_ok = map.values().all(|v| v["ok"].as_bool().unwrap_or(true));
        if controls {
            j["executor_controls"] = Value::Object(map);
        }
        if let Err(e) = &r {
            j["error"] = json!(e.to_string());
        }
        *pass &= r.is_ok() && bleed && ctl_ok && (!controls || j["executor_controls"].as_object().is_some_and(|m| m.len() >= 6));
        cases.push(j);
        Ok(())
    };
    // k = 1..8 with unequal prompts/contexts; request r uses fixture r.
    for &k in &args.ks {
        let chosen: Vec<usize> = (0..k).filter(|i| refs_b[*i].is_some()).collect();
        let refused: Vec<String> = (0..k).filter(|i| refs_b[*i].is_none()).map(|i| fx[i].name()).collect();
        if !refused.is_empty() {
            cases.push(json!({"case": format!("k{k}"), "refused_fixtures": refused, "note": "executor refused these fixtures in isolation; case runs the remaining requests"}));
            pass = false;
        }
        if chosen.is_empty() {
            continue;
        }
        // max_tokens clip: unequal exits (request r stops 3r picks early).
        let reqs = chosen.iter().map(|&i| Req::new(i, (k * 100 + i) as u64, 1, i, (steps + 1).saturating_sub(3 * i).max(2), compare_sink(i))).collect();
        run_case(ctx, &mut store, format!("k{k}"), reqs, false, &mut cases, &mut pass)?;
    }
    if supported.len() >= 2 {
        let (s0, s1) = (supported[0], supported[1]);
        // Stop id inside the stream: first pick equal to the reference's 6th.
        let t = refs_b[s0].as_ref().unwrap();
        let picks: Vec<u32> = t.committed.iter().copied().chain([t.pending_seed]).collect();
        let stop = picks[5];
        let stop_at = picks.iter().position(|&p| p == stop).unwrap() + 1;
        // Repeated in this process (`--stop-repeats`): retire of a stopped
        // request while its peer is mid-prefill must reproduce every time.
        for rep in 0..args.stop_repeats {
            let tag = 9001 + 10 * rep as u64;
            let mut a = Req::new(s0, tag, 1, 2, steps + 1, compare_sink(s0));
            a.stop = Some(stop);
            let b = Req::new(s1, tag + 1, 1, 5, steps + 1, compare_sink(s1));
            let name = if rep == 0 { "stop_id".to_string() } else { format!("stop_id_rep{rep}") };
            eprintln!("batch {name}: free_vram before={}", free_vram(&ctx.gpu)?);
            run_case(ctx, &mut store, name.clone(), vec![a, b], false, &mut cases, &mut pass)?;
            let got = cases.last().unwrap()["requests"][0].clone();
            let ok = got["finish"] == json!("stop") && got["committed"] == json!(stop_at);
            eprintln!("batch {name}: finish={} committed={} expected {stop_at} -> {}", got["finish"], got["committed"], if ok { "OK" } else { "FAIL" });
            pass &= ok;
        }
        // Regression: stop_id under memory pressure (the ks56 failure at
        // 34c1ee400b). Ballast lowers the singleton prefill ceiling to <=1024
        // so r1's 8192 prompt runs in several chunks and r0's stop+retire
        // lands between two of them (r0 picks 0..5 on steps 1..6, retired
        // before step 7 = r1's sixth chunk). Exercised only if r1 actually
        // ran more than five chunks of <=1024 rows.
        for rep in 0..args.stop_repeats {
            if fx[s1].prefix <= 5 * 1024 {
                // Needs a peer prompt of >5 chunks at <=1024 rows (the default
                // contexts give 8192); recorded, not silently passed.
                cases.push(json!({"case": "stop_id_pressure", "skipped": format!("peer prompt {} rows <= 5120", fx[s1].prefix)}));
                eprintln!("batch stop_id_pressure: SKIPPED (peer prompt {} rows)", fx[s1].prefix);
                break;
            }
            let name = if rep == 0 { "stop_id_pressure".to_string() } else { format!("stop_id_pressure_rep{rep}") };
            let tag = 9051 + 10 * rep as u64;
            let mut ballast: Vec<GpuTensor> = Vec::new();
            let mut ceilings = Vec::new();
            let limit = |ctx: &Ctx| -> Result<usize> {
                let b = &ctx.b;
                Ok(qwen35::ordinary_prefill_chunk_limit(&ctx.gpu, &b.weights, &b.config, &b.dn_state, &b.kv_cache, None)?)
            };
            let mut ceiling = limit(ctx)?;
            ceilings.push(ceiling);
            while ceiling > 1024 && ballast.len() < 128 && free_vram(&ctx.gpu)? > (1usize << 30) {
                ballast.push(ctx.gpu.zeros(&[256 << 20], DType::Raw)?);
                ceiling = limit(ctx)?;
                ceilings.push(ceiling);
            }
            let ballast_bytes = ballast.len() * (256 << 20);
            eprintln!("batch {name}: ballast={ballast_bytes} ceiling={ceiling} free_vram={}", free_vram(&ctx.gpu)?);
            let mut a = Req::new(s0, tag, 1, 2, steps + 1, compare_sink(s0));
            a.stop = Some(stop);
            let b = Req::new(s1, tag + 1, 1, 5, steps + 1, compare_sink(s1));
            let res = run_case(ctx, &mut store, name.clone(), vec![a, b], false, &mut cases, &mut pass);
            for t in ballast {
                ctx.gpu.free_tensor(t)?;
            }
            res?;
            let j = cases.last_mut().unwrap();
            let chunks: Vec<u64> = j["requests"][1]["prefill_chunks"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            let exercised = chunks.len() > 5 && chunks.iter().all(|&c| c <= 1024);
            let ok = j["requests"][0]["finish"] == json!("stop") && j["requests"][0]["committed"] == json!(stop_at);
            j["ballast_bytes"] = json!(ballast_bytes);
            j["ceilings"] = json!(ceilings);
            j["pressure_exercised"] = json!(exercised);
            eprintln!("batch {name}: r1 chunks={chunks:?} exercised={exercised} stop_ok={ok}");
            pass &= exercised && ok;
        }
        // Cancel mid-stream: the survivor must stay byte-exact.
        let mut a = Req::new(s0, 9101, 1, 0, steps + 1, compare_sink(s0));
        a.cancel_at = Some(9);
        let b = Req::new(s1, 9102, 1, 1, steps + 1, compare_sink(s1));
        run_case(ctx, &mut store, "cancel".into(), vec![a, b], false, &mut cases, &mut pass)?;
        // Executor controls (row_slot / epoch / rejected tail / stale commit /
        // poison) need both requests decoding together: the two shortest.
        let mut by_len = supported.clone();
        by_len.sort_by_key(|&i| fx[i].prefix);
        let (c0, c1) = (by_len[0], by_len[1]);
        let a = Req::new(c0, 9201, 1, 3, steps + 1, compare_sink(c0));
        let b = Req::new(c1, 9202, 1, 6, steps + 1, compare_sink(c1));
        run_case(ctx, &mut store, "controls".into(), vec![a, b], true, &mut cases, &mut pass)?;
        // Slot wrap/reuse: the same slots now hold new epochs (generation 2)
        // with swapped fixtures; must still match their isolated references.
        let a = Req::new(s1, 9201, 2, 3, steps + 1, compare_sink(s1));
        let b = Req::new(s0, 9202, 2, 6, steps + 1, compare_sink(s0));
        run_case(ctx, &mut store, "slot_reuse".into(), vec![a, b], false, &mut cases, &mut pass)?;
    } else {
        pass = false;
    }
    // Verify rows belong to slice 2: record the executor's current answer.
    {
        let probe = supported.first().copied().unwrap_or(0);
        let r = Req::new(probe, 9301, 1, 0, steps + 1, compare_sink(probe));
        admit(ctx, &mut store, fx, &r)?;
        let mut p = build_plan(ctx, &store, fx, std::slice::from_ref(&r))?;
        p.requests[0].kind = RequestStepKind::Verify { draft_len: 0 };
        let Ctx { gpu, b } = ctx;
        let res = store.executor(&b.weights, &b.config, &b.scratch).provision_step(gpu, &p);
        if res.is_ok() {
            store.abort_step(&p);
        }
        retire(ctx, &mut store, &r.epoch)?;
        out.insert("spec_verify_probe".into(), json!({
            "accepted_by_executor": res.is_ok(), "error": res.err(),
            "note": "full/partial accept, EOS-in-draft and spec max_tokens clipping need Verify rows (Slice2A); not certified until accepted",
        }));
    }
    if args.drift_cycles > 0 && supported.len() >= 2 {
        let (drift, ok) = drift_check(ctx, &mut store, fx, &refs, &supported, args.drift_cycles)?;
        out.insert("memory_drift".into(), drift);
        pass &= ok;
    }
    out.insert("cases".into(), Value::Array(cases));
    let k1_route = out["executor_k1"]
        .as_array()
        .is_some_and(|v| v.iter().all(|e| e["vs_singleton_route"]["exact"] == json!(true)));
    out.insert("route_exact_vs_singleton".into(), json!(route_exact_all && k1_route));
    let receipt = store.receipt()?;
    out.insert("store_receipt".into(), json!({
        "kv_backend": receipt.kv_backend, "kv_mode": receipt.kv_mode, "max_seq_bound": receipt.max_seq_bound,
        "mapped_bytes": receipt.mapped_bytes, "mapped_high_water": receipt.mapped_high_water,
    }));
    store.free_gpu(&mut ctx.gpu)?;
    Ok((Value::Object(out), pass))
}

// ── Spec phase: isolated singleton MTP references (§4.3, §7 item 2) ─────
//
// The reference is the default singleton MTP route: `Qwen35MtpDrafter` on a
// `ModelSlot` driven exactly like `MtpSpeculator` (k = min(max_emit-1,
// proposal_capacity), eos from the target, greedy). Per cycle it records the
// window (position, seed, k, committed, accepted, drafted) and a sha256
// digest of the state the cycle wrote: trunk KV rows [start, end), DeltaNet
// S/scales/EF/conv, MTP-head KV rows [start, end) and prev_hidden. Full state
// bytes are kept at prefill, final, and the first full/partial/zero-accept
// cycle. Rejected rows past the committed watermark are never compared.

#[derive(Clone)]
struct SpecCycle {
    position: usize,
    seed: u32,
    k: usize,
    committed: Vec<u32>,
    accepted: usize,
    drafted: usize,
    class: &'static str,
    delta_sha256: String,
}

struct SpecTrace {
    /// Picks in order: prefill token, then every committed id.
    emitted: Vec<u32>,
    cycles: Vec<SpecCycle>,
    finish: String,
    position: usize,
    pending_seed: u32,
    states: Vec<(String, PathBuf)>,
}

fn spec_class(w: &hipfire_runtime::spec::MtpWindow) -> &'static str {
    match (w.drafts_generated, w.accepted) {
        (0, _) => "ar",
        (d, a) if a == d => "full",
        (_, 0) => "zero",
        _ => "partial",
    }
}

fn mtp_row_bytes(st: &hipfire_arch_qwen35::mtp_spec::MtpSpecState) -> usize {
    st.mtp_kv.n_head_kv * (st.mtp_kv.head_dim / 32) * 34
}

/// MTP head state: KV rows [from, to) (K then V), prev_hidden, its position.
fn mtp_state_bytes(gpu: &Gpu, st: &MtpSpecState, from: usize, to: usize) -> Result<Vec<(String, Vec<u8>)>> {
    gpu.hip.device_synchronize()?;
    let row = mtp_row_bytes(st);
    let mut out = Vec::new();
    for (side, bufs) in [("k", &st.mtp_kv.inner.k_gpu), ("v", &st.mtp_kv.inner.v_gpu)] {
        for (i, t) in bufs.iter().enumerate().filter(|(_, t)| t.buf.size() >= row) {
            out.push((format!("mtp.{side}{i}"), read_dev(gpu, t, from * row, (to - from) * row)?));
        }
    }
    let ph = st.prev_hidden.byte_size().min(st.prev_hidden.buf.size());
    out.push(("mtp.prev_hidden".into(), read_dev(gpu, &st.prev_hidden, 0, ph)?));
    out.push(("mtp.prev_hidden_pos".into(), (st.prev_hidden_pos.map_or(u64::MAX, |p| p as u64)).to_le_bytes().to_vec()));
    Ok(out)
}

/// Trunk KV rows [from, to) plus whole DeltaNet state.
fn trunk_rows_bytes(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, dn: &DeltaNetState, from: usize, to: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    let (kr, vr) = kv_row_bytes(kv)?;
    for (name, bytes) in state_bytes(gpu, config, kv, dn, 0)? {
        if !name.ends_with(".k") && !name.ends_with(".v") {
            out.push((name, bytes));
        }
    }
    for (layer, ty) in config.layer_types.iter().enumerate() {
        if *ty == LayerType::FullAttention {
            out.push((format!("L{layer:02}.k[{from}..{to})"), read_dev(gpu, &kv.k_gpu[layer], from * kr, (to - from) * kr)?));
            out.push((format!("L{layer:02}.v[{from}..{to})"), read_dev(gpu, &kv.v_gpu[layer], from * vr, (to - from) * vr)?));
        }
    }
    Ok(out)
}

/// Delta digest of one cycle: trunk rows [from, to), DeltaNet, MTP rows [from, to).
fn cycle_digest(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, dn: &DeltaNetState, st: &MtpSpecState, from: usize, to: usize) -> Result<String> {
    let mut delta = trunk_rows_bytes(gpu, config, kv, dn, from, to)?;
    delta.extend(mtp_state_bytes(gpu, st, from, to)?);
    Ok(digest(&delta))
}

fn digest(items: &[(String, Vec<u8>)]) -> String {
    let mut h = Sha256::new();
    for (n, b) in items {
        h.update(n.as_bytes());
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Full committed state at `pos`: trunk KV prefix, DeltaNet, MTP head prefix.
fn spec_full_state(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, dn: &DeltaNetState, st: &MtpSpecState, pos: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let mut v = state_bytes(gpu, config, kv, dn, pos)?;
    v.extend(mtp_state_bytes(gpu, st, 0, pos)?);
    Ok(v)
}

fn slot_full_state(gpu: &Gpu, slot: &ModelSlot, d: &Qwen35MtpDrafter, pos: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let st = d.mtp_live_state().ok_or("MTP state not allocated")?;
    spec_full_state(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, st, pos)
}

/// One isolated singleton MTP greedy run (`budget` picks, `eos`).
fn spec_run(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Qwen35MtpDrafter, f: &Fixture, budget: usize, eos: u32, dir: Option<&Path>) -> Result<SpecTrace> {
    d.configure_request(SpecRequestConfig::default());
    let seed0 = d.mtp_prefill(gpu, slot, &f.tokens, &f.tokens, 0, false, &|| false)?;
    let mut tr = SpecTrace { emitted: vec![seed0], cycles: vec![], finish: String::new(), position: f.prefix, pending_seed: seed0, states: vec![] };
    let mut history = f.tokens.clone();
    history.push(seed0);
    let save = |gpu: &Gpu, slot: &ModelSlot, d: &Qwen35MtpDrafter, tr: &mut SpecTrace, stage: &str| -> Result<()> {
        if let Some(dir) = dir {
            for (name, bytes) in slot_full_state(gpu, slot, d, tr.position)? {
                let p = dir.join(stage).join(format!("{name}.bin"));
                write_art(&p, &bytes)?;
                tr.states.push((format!("{stage}/{name}"), p));
            }
        }
        Ok(())
    };
    save(gpu, slot, d, &mut tr, "prefill")?;
    let mut first_of: Vec<&'static str> = Vec::new();
    tr.finish = if seed0 == eos { "stop".into() } else { String::new() };
    while tr.finish.is_empty() {
        if tr.emitted.len() >= budget {
            tr.finish = "length".into();
            break;
        }
        let max_emit = budget - tr.emitted.len();
        let k = max_emit.saturating_sub(1).min(d.proposal_capacity());
        let start = tr.position;
        let w = d.mtp_step(gpu, slot, start, tr.pending_seed, &history, k, eos, None)?;
        if w.committed.is_empty() || w.committed.len() > max_emit {
            return Err(format!("MTP window committed {} with max_emit {max_emit}", w.committed.len()).into());
        }
        let class = spec_class(&w);
        let end = start + w.committed.len();
        let st = d.mtp_live_state().ok_or("MTP state not allocated")?;
        let delta_sha256 = cycle_digest(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, st, start, end)?;
        tr.cycles.push(SpecCycle {
            position: start, seed: tr.pending_seed, k, committed: w.committed.clone(), accepted: w.accepted,
            drafted: w.drafts_generated, class, delta_sha256,
        });
        tr.emitted.extend_from_slice(&w.committed);
        history.extend_from_slice(&w.committed);
        tr.position = end;
        tr.pending_seed = *w.committed.last().unwrap();
        if w.committed.contains(&eos) {
            tr.finish = "stop".into();
        }
        if !first_of.contains(&class) && class != "ar" {
            first_of.push(class);
            let stage = format!("first_{class}_cycle{}", tr.cycles.len() - 1);
            save(gpu, slot, d, &mut tr, &stage)?;
        }
    }
    save(gpu, slot, d, &mut tr, "final")?;
    Ok(tr)
}

/// Byte/ID comparison of a replayed trace against the reference trace.
fn spec_compare(reference: &SpecTrace, got: &SpecTrace, got_states: &[(String, Vec<u8>)]) -> Result<Diff> {
    let mut d = Diff::default();
    d.ids("emitted", &reference.emitted, &got.emitted);
    d.ids("position", &[reference.position as u32], &[got.position as u32]);
    d.ids("pending_seed", &[reference.pending_seed], &[got.pending_seed]);
    d.check("finish", reference.finish.as_bytes(), got.finish.as_bytes());
    let n = reference.cycles.len().max(got.cycles.len());
    for c in 0..n {
        match (reference.cycles.get(c), got.cycles.get(c)) {
            (Some(a), Some(b)) => {
                let wa = [a.position as u32, a.seed, a.k as u32, a.accepted as u32, a.drafted as u32];
                let wb = [b.position as u32, b.seed, b.k as u32, b.accepted as u32, b.drafted as u32];
                d.ids(&format!("cycle{c}.window"), &wa, &wb);
                d.ids(&format!("cycle{c}.committed"), &a.committed, &b.committed);
                d.check(&format!("cycle{c}.delta_sha256"), a.delta_sha256.as_bytes(), b.delta_sha256.as_bytes());
            }
            _ => d.differing_items.push(format!("cycle{c}: present in only one trace")),
        }
    }
    for (name, bytes) in got_states {
        match reference.states.iter().find(|(n, _)| n == name) {
            Some((_, p)) => d.check(name, &fs::read(p)?, bytes),
            None => d.differing_items.push(format!("{name}: missing in reference")),
        }
    }
    Ok(d)
}

/// Run `f` again in memory and collect the same stage states for comparison.
fn spec_replay(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Qwen35MtpDrafter, f: &Fixture, budget: usize, eos: u32, reference: &SpecTrace) -> Result<Diff> {
    // Intermediate saved stages are re-derived by a replay stopped there.
    let stages: std::collections::BTreeSet<String> =
        reference.states.iter().map(|(n, _)| n.split('/').next().unwrap().to_string()).collect();
    let mut states = Vec::new();
    for stage in stages.iter().filter(|s| s.as_str() != "final") {
        let stop_after = if stage == "prefill" { 0 } else { stage.rsplit("cycle").next().unwrap().parse::<usize>()? + 1 };
        let pos = spec_partial(gpu, slot, d, f, budget, eos, stop_after)?;
        for (name, bytes) in slot_full_state(gpu, slot, d, pos)? {
            states.push((format!("{stage}/{name}"), bytes));
        }
    }
    let got = spec_run(gpu, slot, d, f, budget, eos, None)?;
    for (name, bytes) in slot_full_state(gpu, slot, d, got.position)? {
        states.push((format!("final/{name}"), bytes));
    }
    spec_compare(reference, &got, &states)
}

/// Run `f` for `cycles` windows (0 = prefill only); returns the committed position.
fn spec_partial(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Qwen35MtpDrafter, f: &Fixture, budget: usize, eos: u32, cycles: usize) -> Result<usize> {
    d.configure_request(SpecRequestConfig::default());
    let mut seed = d.mtp_prefill(gpu, slot, &f.tokens, &f.tokens, 0, false, &|| false)?;
    let (mut pos, mut emitted) = (f.prefix, 1usize);
    let mut history = f.tokens.clone();
    history.push(seed);
    let mut c = 0;
    while c < cycles && emitted < budget && seed != eos {
        let max_emit = budget - emitted;
        let k = max_emit.saturating_sub(1).min(d.proposal_capacity());
        let w = d.mtp_step(gpu, slot, pos, seed, &history, k, eos, None)?;
        emitted += w.committed.len();
        history.extend_from_slice(&w.committed);
        pos += w.committed.len();
        seed = *w.committed.last().unwrap();
        c += 1;
        if w.committed.contains(&eos) {
            break;
        }
    }
    Ok(pos)
}

fn spec_trace_json(t: &SpecTrace) -> Value {
    let count = |c: &str| t.cycles.iter().filter(|x| x.class == c).count();
    json!({
        "picks": t.emitted.len(), "cycles": t.cycles.len(), "finish": t.finish, "position": t.position,
        "pending_seed": t.pending_seed,
        "classes": {"full": count("full"), "partial": count("partial"), "zero": count("zero"), "ar": count("ar")},
        "accepted_total": t.cycles.iter().map(|c| c.accepted).sum::<usize>(),
        "emitted": t.emitted,
        "windows": t.cycles.iter().map(|c| json!([c.position, c.k, c.accepted, c.drafted, c.class, c.committed])).collect::<Vec<_>>(),
    })
}

/// Spec phase: per fixture an isolated singleton MTP reference (artifacts),
/// its determinism self-test, and the derived EOS-in-draft and
/// max_tokens-mid-draft variants. Returns (report, pass).
fn spec_phase(ctx: Ctx, args: &Args, fx: &[Fixture], ar_refs: &[Trace]) -> Result<(Value, bool)> {
    let Ctx { mut gpu, b } = ctx;
    let mut slot = ModelSlot::from_bundle(b, Path::new(&args.model)).map_err(|(_, e)| format!("ModelSlot::from_bundle: {e}"))?;
    let head_path = args.mtp_head.clone().unwrap_or_else(|| Path::new(&args.model).with_extension("mtp"));
    let max_prefix = fx.iter().map(|f| f.prefix).max().unwrap_or(0);
    let cap = max_prefix + args.steps + 64;
    let head = load_mtp_head(&head_path, &mut gpu, cap)?;
    let k = hipfire_runtime::config::get().mtp_k.clamp(1, 10);
    let mut d = Qwen35MtpDrafter::new(head, k, cap);
    let budget = args.steps + 1;
    let eos = slot.config.eos_token;
    let mut pass = true;
    let mut rows = Vec::new();
    // Every reference trace with its run parameters, for the batched cases.
    let mut refs: Vec<(usize, usize, u32, SpecTrace)> = Vec::new(); // (fixture, budget, eos, trace)
    let (mut eos_ref, mut clip_ref) = (vec![None; fx.len()], vec![None; fx.len()]);
    for f in fx {
        let dir = args.artifacts.join("spec_mtp").join(f.name());
        let t0 = std::time::Instant::now();
        let tr = spec_run(&mut gpu, &mut slot, &mut d, f, budget, eos, Some(&dir))?;
        let fi = rows.len();
        let secs = t0.elapsed().as_secs_f64();
        let det = spec_replay(&mut gpu, &mut slot, &mut d, f, budget, eos, &tr)?;
        pass &= det.exact();
        let mut row = json!({"fixture": f.name(), "k": k, "record_seconds": secs, "reference": spec_trace_json(&tr), "determinism": det.json()});
        eprintln!("spec {}: picks={} cycles={} classes={} determinism exact={} ({secs:.1}s)", f.name(), tr.emitted.len(), tr.cycles.len(), row["reference"]["classes"], det.exact());
        // EOS inside a draft: the 2nd committed id of the first window that
        // accepted >=2 drafts becomes the stop id.
        if let Some((ci, c)) = tr.cycles.iter().enumerate().find(|(_, c)| c.accepted >= 2) {
            let stop = c.committed[1];
            let vdir = args.artifacts.join("spec_mtp").join(format!("{}_eos", f.name()));
            let v = spec_run(&mut gpu, &mut slot, &mut d, f, budget, stop, Some(&vdir))?;
            let first = tr.emitted.iter().position(|&t| t == stop).unwrap() + 1;
            let ok = v.finish == "stop" && v.emitted.len() == first && v.emitted[..] == tr.emitted[..first];
            pass &= ok;
            eprintln!("spec {} eos_in_draft: stop={stop} cycle={ci} picks={} expected {first} -> {}", f.name(), v.emitted.len(), if ok { "OK" } else { "FAIL" });
            row["eos_in_draft"] = json!({"stop": stop, "cycle": ci, "ok": ok, "trace": spec_trace_json(&v)});
            eos_ref[fi] = Some(refs.len());
            refs.push((fi, budget, stop, v));
        } else {
            row["eos_in_draft"] = json!({"skipped": "no window accepted >=2 drafts"});
        }
        // max_tokens clipping mid-draft: budget ends one pick into a window
        // that committed >=3, so k shrinks below the drafter's capacity.
        if let Some((ci, _)) = tr.cycles.iter().enumerate().find(|(_, c)| c.committed.len() >= 3) {
            let before: usize = 1 + tr.cycles[..ci].iter().map(|c| c.committed.len()).sum::<usize>();
            let clip = before + 2;
            let vdir = args.artifacts.join("spec_mtp").join(format!("{}_clip", f.name()));
            let v = spec_run(&mut gpu, &mut slot, &mut d, f, clip, eos, Some(&vdir))?;
            let last = v.cycles.last();
            let ok = v.finish == "length" && v.emitted.len() == clip && v.emitted[..] == tr.emitted[..clip]
                && last.is_some_and(|c| c.k < k);
            pass &= ok;
            eprintln!("spec {} max_tokens_mid_draft: budget={clip} picks={} last_k={:?} -> {}", f.name(), v.emitted.len(), last.map(|c| c.k), if ok { "OK" } else { "FAIL" });
            row["max_tokens_mid_draft"] = json!({"budget": clip, "cycle": ci, "ok": ok, "trace": spec_trace_json(&v)});
            clip_ref[fi] = Some(refs.len());
            refs.push((fi, clip, eos, v));
        } else {
            row["max_tokens_mid_draft"] = json!({"skipped": "no window committed >=3"});
        }
        rows.push(row);
        refs.push((fi, budget, eos, tr));
    }
    let main_ref: Vec<usize> = (0..fx.len())
        .map(|i| refs.iter().position(|(f, b, e, _)| *f == i && *b == budget && *e == eos).unwrap())
        .collect();
    let classes_seen: Vec<&str> = ["full", "partial", "zero"]
        .into_iter()
        .filter(|c| rows.iter().any(|r| r["reference"]["classes"][*c].as_u64().unwrap_or(0) > 0))
        .collect();
    Box::new(d).mtp_free(&mut gpu);

    // ── Batched: mtp_cb_cycle lanes vs the isolated singleton references ──
    let head_b = load_mtp_head(&head_path, &mut gpu, cap)?;
    let mut cases = Vec::new();
    let run = |gpu: &mut Gpu, slot: &mut ModelSlot, name: String, lanes: Vec<LaneSpec>, cases: &mut Vec<Value>, pass: &mut bool| -> Result<()> {
        let (j, ok) = spec_batch_case(gpu, slot, &head_b, k, fx, &refs, &lanes, &name)?;
        eprintln!("spec batch {name}: exact={ok}");
        *pass &= ok;
        cases.push(j);
        Ok(())
    };
    for &kk in &args.ks {
        let lanes = (0..kk.min(fx.len())).map(|i| LaneSpec { reference: main_ref[i], drop_after: None }).collect();
        run(&mut gpu, &mut slot, format!("k{kk}"), lanes, &mut cases, &mut pass)?;
    }
    // Mixed: EOS-inside-draft lanes, max_tokens-mid-draft lanes (k below
    // capacity, k=0 seed-only rows at the budget edge) and plain lanes.
    let mixed: Vec<LaneSpec> = (0..fx.len().min(8))
        .map(|i| {
            let r = match i % 3 {
                0 => eos_ref[i],
                1 => clip_ref[i],
                _ => None,
            };
            LaneSpec { reference: r.unwrap_or(main_ref[i]), drop_after: None }
        })
        .collect();
    run(&mut gpu, &mut slot, "eos_clip_mix".into(), mixed, &mut cases, &mut pass)?;
    // Cancel: lane 0 is retired after 3 cycles; survivors must stay exact.
    let cancel: Vec<LaneSpec> = (0..fx.len().min(4))
        .map(|i| LaneSpec { reference: main_ref[i], drop_after: (i == 0).then_some(3) })
        .collect();
    run(&mut gpu, &mut slot, "cancel".into(), cancel, &mut cases, &mut pass)?;
    head_b.free_gpu(&mut gpu);

    // ── Executor: Verify rows through provision/forward/commit (Slice2A) ──
    let mut exec_cases = Vec::new();
    let exec = spec_exec_cases(&mut gpu, &mut slot, args, fx, &refs, &main_ref, &eos_ref, &clip_ref, ar_refs, &head_path, cap, k, &mut exec_cases);
    match exec {
        Ok(ok) => pass &= ok,
        Err(e) => {
            pass = false;
            exec_cases.push(json!({"case": "executor", "error": e.to_string()}));
        }
    }
    let report = json!({
        "route": "reference: singleton Qwen35MtpDrafter on ModelSlot (MtpSpeculator k/eos contract), greedy; \
                  batch: mtp_spec::cb::mtp_cb_cycle over per-request KV/DN/MTP state; executor: Qwen35VmmStore \
                  (VmmRoute::Exact) + VmmSpecEngine, Verify rows through provision/forward/commit",
        "head": head_path, "k": k, "budget_picks": budget, "accept_classes_seen": classes_seen,
        "fixtures": rows, "batch_cases": cases, "executor_cases": exec_cases,
    });
    Ok((report, pass && classes_seen.len() == 3))
}

/// One executor lane: a spec lane reproducing `refs[reference]` (optionally
/// finishing on a stop id inside a window), or the AR lane reproducing a
/// singleton AR trace.
struct ExecLane {
    epoch: RequestEpoch,
    slot: usize,
    /// Spec lane: index into refs. AR lane: None.
    reference: Option<usize>,
    ar: Option<usize>,
    stop: Option<u32>,
    budget: usize,
    tr: SpecTrace,
    ar_picks: Vec<u32>,
    fed: usize,
    diff: Diff,
    done: bool,
    cancelled: bool,
}

fn exec_stage(st: &Qwen35RequestState, slot: &ModelSlot, gpu: &Gpu, pos: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let m = st.mtp.as_ref().ok_or("spec lane has no MTP state")?;
    spec_full_state(gpu, &slot.config, &st.kv, &st.dn, m, pos)
}

fn exec_plan(store: &Qwen35VmmStore, gpu: &Gpu, slot: &ModelSlot, fx: &[Fixture], lanes: &[ExecLane], k: usize, only: Option<usize>) -> Result<BatchStepPlan> {
    let mut plan = BatchStepPlan::default();
    plan.batch.m_per_slot = vec![0; EXEC_WIDTH];
    let push = |plan: &mut BatchStepPlan, epoch: RequestEpoch, s: usize, toks: &[u32], pos0: usize, kind: RequestStepKind| {
        let begin = plan.batch.tokens.len();
        for (j, &t) in toks.iter().enumerate() {
            plan.batch.tokens.push(t);
            plan.batch.positions.push((pos0 + j) as i32);
            plan.batch.row_slot.push(s as i32);
        }
        plan.batch.m_per_slot[s] = toks.len();
        plan.requests.push(RequestRows { epoch, rows: RowRange { begin, len: toks.len() }, kind });
        match kind {
            RequestStepKind::Verify { .. } => plan.verify_rows += toks.len(),
            RequestStepKind::Prefill => plan.prefill_rows += toks.len(),
            _ => plan.decode_rows += toks.len(),
        }
    };
    let mut order: Vec<usize> = (0..lanes.len()).filter(|&i| !lanes[i].done && only.is_none_or(|o| o == i)).collect();
    order.sort_by_key(|&i| lanes[i].slot);
    for i in order {
        let l = &lanes[i];
        let st = store.request_state(&l.epoch).ok_or("lane missing from store")?;
        if let Some(ai) = l.ar {
            let f = &fx[ai];
            if l.fed < f.prefix {
                let n = store.exact_prefill_chunk_len(gpu, &slot.weights, &slot.config, &l.epoch, f.prefix - l.fed)?;
                push(&mut plan, l.epoch, l.slot, &f.tokens[l.fed..l.fed + n], l.fed, RequestStepKind::Prefill);
            } else {
                push(&mut plan, l.epoch, l.slot, &[st.pending_seed.ok_or("AR seed")?], st.position, RequestStepKind::Ar);
            }
        } else {
            let max_emit = l.budget - l.tr.emitted.len();
            let kk = max_emit.saturating_sub(1).min(k);
            let mut toks = vec![st.pending_seed.ok_or("spec seed")?];
            toks.extend(std::iter::repeat_n(0u32, kk));
            push(&mut plan, l.epoch, l.slot, &toks, st.position, RequestStepKind::Verify { draft_len: kk });
        }
    }
    Ok(plan)
}

const EXEC_WIDTH: usize = 9;

/// Publish one committed step into the lanes (window, digest, stage bytes).
#[allow(clippy::too_many_arguments)]
fn exec_absorb(gpu: &Gpu, slot: &ModelSlot, store: &Qwen35VmmStore, fx: &[Fixture], refs: &[(usize, usize, u32, SpecTrace)], ar_refs: &[Trace], lanes: &mut [ExecLane], plan: &BatchStepPlan, advances: &[hipfire_runtime::slot_batch::RequestAdvance], k: usize, prior: &[(usize, u32)]) -> Result<()> {
    let eos = slot.config.eos_token;
    for rr in &plan.requests {
        let i = lanes.iter().position(|l| l.epoch == rr.epoch).ok_or("plan epoch not a lane")?;
        let lane = &mut lanes[i];
        if let Some(ai) = lane.ar {
            if rr.kind == RequestStepKind::Prefill {
                lane.fed += rr.rows.len;
                if lane.fed == fx[ai].prefix {
                    let st = store.request_state(&lane.epoch).ok_or("lane vanished")?;
                    let now = state_bytes(gpu, &slot.config, &st.kv, &st.dn, lane.fed)?;
                    state_cmp(&mut lane.diff, "prefill", &now, &ar_refs[ai])?;
                }
            }
        }
    }
    for a in advances {
        let i = lanes.iter().position(|l| l.epoch == a.epoch).ok_or("advance epoch not a lane")?;
        let lane = &mut lanes[i];
        let st = store.request_state(&a.epoch).ok_or("lane vanished")?;
        if let Some(ai) = lane.ar {
            lane.ar_picks.extend_from_slice(&a.committed_ids);
            if lane.ar_picks.len() >= lane.budget {
                lane.done = true;
                let t = &ar_refs[ai];
                let picks: Vec<u32> = t.committed.iter().copied().chain([t.pending_seed]).collect();
                lane.diff.ids("ar_picks", &picks, &lane.ar_picks);
                lane.diff.ids("position", &[t.position as u32], &[st.position as u32]);
                let now = state_bytes(gpu, &slot.config, &st.kv, &st.dn, st.position)?;
                state_cmp(&mut lane.diff, "final", &now, t)?;
            }
            continue;
        }
        let reference = &refs[lane.reference.unwrap()].3;
        let (start, seed) = prior[i];
        let max_emit = lane.budget - lane.tr.emitted.len();
        let kk = max_emit.saturating_sub(1).min(k);
        let end = start + a.committed_ids.len();
        if st.position != end || a.committed_position != end {
            lane.diff.differing_items.push(format!("cycle{}: watermark store {} advance {} expected {end}", lane.tr.cycles.len(), st.position, a.committed_position));
        }
        let m = st.mtp.as_ref().ok_or("spec lane has no MTP state")?;
        let delta_sha256 = cycle_digest(gpu, &slot.config, &st.kv, &st.dn, m, start, end)?;
        let drafted = a.verified_rows.saturating_sub(1);
        let class = spec_class(&hipfire_runtime::spec::MtpWindow { committed: a.committed_ids.clone(), accepted: a.accepted_drafts, drafts_generated: drafted });
        lane.tr.cycles.push(SpecCycle { position: start, seed, k: kk, committed: a.committed_ids.clone(), accepted: a.accepted_drafts, drafted, class, delta_sha256 });
        lane.tr.emitted.extend_from_slice(&a.committed_ids);
        lane.tr.position = end;
        lane.tr.pending_seed = *a.committed_ids.last().ok_or("empty advance")?;
        if st.pending_seed != Some(lane.tr.pending_seed) {
            lane.diff.differing_items.push(format!("cycle{}: store pending_seed {:?}", lane.tr.cycles.len() - 1, st.pending_seed));
        }
        let stage = format!("first_{class}_cycle{}", lane.tr.cycles.len() - 1);
        if lane.stop.is_none() && has_stage(reference, &stage) {
            let now = exec_stage(st, slot, gpu, end)?;
            stage_check(&mut lane.diff, reference, &stage, now)?;
        }
        if a.finish.as_deref() == Some("stop") || a.committed_ids.contains(&eos) {
            lane.tr.finish = "stop".into();
        } else if lane.tr.emitted.len() >= lane.budget {
            lane.tr.finish = "length".into();
        }
        lane.done = !lane.tr.finish.is_empty();
    }
    Ok(())
}

/// Final comparison of an executor lane against its reference.
fn exec_finish(gpu: &Gpu, slot: &ModelSlot, store: &Qwen35VmmStore, refs: &[(usize, usize, u32, SpecTrace)], lane: &mut ExecLane) -> Result<()> {
    let Some(ri) = lane.reference else { return Ok(()) };
    let reference = &refs[ri].3;
    if lane.cancelled || lane.stop.is_some() {
        // Committed prefix must match the plain reference cycle by cycle.
        for (c, b) in lane.tr.cycles.iter().enumerate() {
            match reference.cycles.get(c) {
                Some(a) => {
                    lane.diff.ids(&format!("cycle{c}.committed"), &a.committed, &b.committed);
                    lane.diff.check(&format!("cycle{c}.delta_sha256"), a.delta_sha256.as_bytes(), b.delta_sha256.as_bytes());
                }
                None => lane.diff.differing_items.push(format!("cycle{c}: beyond reference")),
            }
        }
        if let Some(stop) = lane.stop {
            // Finishes on the first window committing `stop`, never earlier.
            let first = lane.tr.cycles.iter().position(|c| c.committed.contains(&stop));
            let ok = lane.tr.finish == "stop" && first == Some(lane.tr.cycles.len() - 1);
            if !ok {
                lane.diff.differing_items.push(format!("stop {stop}: finish {:?} at cycle {:?} of {}", lane.tr.finish, first, lane.tr.cycles.len()));
            }
        }
        return Ok(());
    }
    let st = store.request_state(&lane.epoch).ok_or("lane vanished")?;
    let now = exec_stage(st, slot, gpu, lane.tr.position)?;
    stage_check(&mut lane.diff, reference, "final", now)?;
    let d2 = spec_compare(reference, &lane.tr, &[])?;
    lane.diff.compared_items += d2.compared_items;
    lane.diff.compared_bytes += d2.compared_bytes;
    lane.diff.differing_items.extend(d2.differing_items);
    Ok(())
}

/// Executor Verify-row cases. Returns pass.
#[allow(clippy::too_many_arguments)]
fn spec_exec_cases(gpu: &mut Gpu, slot: &mut ModelSlot, args: &Args, fx: &[Fixture], refs: &[(usize, usize, u32, SpecTrace)], main_ref: &[usize], eos_ref: &[Option<usize>], clip_ref: &[Option<usize>], ar_refs: &[Trace], head_path: &Path, cap: usize, k: usize, out: &mut Vec<Value>) -> Result<bool> {
    if let Some(pbs) = slot.scratch.widened_prefill_batch.borrow_mut().take() {
        pbs.free_gpu(gpu)?;
    }
    // The reference/mtp_cb phases freed their owners into the GpuPool free
    // list; VMM mapping draws on device free memory, so return them first.
    gpu.drain_pool();
    let free = free_vram(gpu)?;
    let budget_kv = free.saturating_sub(6 << 30);
    eprintln!("spec exec store: width={EXEC_WIDTH} free_vram={free} kv_budget={budget_kv}");
    let mut store = Qwen35VmmStore::new(gpu, &slot.config, &slot.kv_cache, EXEC_WIDTH, EXEC_WIDTH * (k + 1), budget_kv, VmmRoute::Exact)?;
    let head = load_mtp_head(head_path, gpu, cap)?;
    let route = MtpPromptRoute::from_own_prefill(hipfire_config::mtp_own_prefill());
    let engine = VmmSpecEngine::new(gpu, &slot.config, head, k, EXEC_WIDTH, route)?;
    if store.install_spec(engine).is_err() {
        return Err("install_spec refused".into());
    }
    let ar_budget = args.steps + 1;
    let mut pass = true;
    let mut tag = 70_000u64;
    let mut case = |gpu: &mut Gpu, slot: &mut ModelSlot, store: &mut Qwen35VmmStore, name: &str, spec: Vec<(usize, Option<u32>)>, ar: Option<usize>, controls: bool| -> Result<bool> {
        let mut lanes: Vec<ExecLane> = Vec::new();
        let mut ctl = serde_json::Map::new();
        let res = (|| -> Result<()> {
            for (s, &(ri, stop)) in spec.iter().enumerate() {
                tag += 1;
                let (fi, budget, _, ref reference) = refs[ri];
                let epoch = RequestEpoch { request_tag: tag, owner_generation: 1 };
                let init = VmmRequestInit { prompt_len: fx[fi].prefix, stop_ids: stop.into_iter().collect(), sampler: SamplerConfig::greedy(), rng_state: 0, history: vec![] };
                let o = Qwen35RequestState::new_like(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, epoch, s, init)?;
                store.admit(o).map_err(|(o, e)| { let _ = o.free_gpu(gpu); e })?;
                let seed = match store.spec_prefill(gpu, &slot.weights, &slot.config, &slot.scratch, &epoch, &fx[fi].tokens, SpecRequestConfig::default()) {
                    Ok(s) => s,
                    Err(e) => {
                        store.retire(&epoch)?.free_gpu(gpu)?;
                        return Err(e.into());
                    }
                };
                let mut diff = Diff::default();
                diff.ids("prefill_seed", &[reference.emitted[0]], &[seed]);
                let st = store.request_state(&epoch).ok_or("lane")?;
                stage_check(&mut diff, reference, "prefill", exec_stage(st, slot, gpu, fx[fi].prefix)?)?;
                let tr = SpecTrace { emitted: vec![seed], cycles: vec![], finish: String::new(), position: fx[fi].prefix, pending_seed: seed, states: vec![] };
                let done = seed == slot.config.eos_token || budget <= 1;
                lanes.push(ExecLane { epoch, slot: s, reference: Some(ri), ar: None, stop, budget, tr, ar_picks: vec![], fed: 0, diff, done, cancelled: false });
            }
            if let Some(ai) = ar {
                tag += 1;
                let epoch = RequestEpoch { request_tag: tag, owner_generation: 1 };
                let init = VmmRequestInit { prompt_len: fx[ai].prefix, stop_ids: vec![], sampler: SamplerConfig::greedy(), rng_state: 0, history: vec![] };
                pin_gdn_frame(&slot.dn_state);
                let o = Qwen35RequestState::new_like(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, epoch, EXEC_WIDTH - 1, init)?;
                store.admit(o).map_err(|(o, e)| { let _ = o.free_gpu(gpu); e })?;
                let tr = SpecTrace { emitted: vec![], cycles: vec![], finish: String::new(), position: 0, pending_seed: 0, states: vec![] };
                lanes.push(ExecLane { epoch, slot: EXEC_WIDTH - 1, reference: None, ar: Some(ai), stop: None, budget: ar_budget, tr, ar_picks: vec![], fed: 0, diff: Diff::default(), done: false, cancelled: false });
            }
            let mut step = 0usize;
            loop {
                if controls && step == 3 {
                    exec_controls(gpu, slot, store, fx, &mut lanes, k, &mut ctl)?;
                }
                let plan = exec_plan(store, gpu, slot, fx, &lanes, k, None)?;
                if plan.requests.is_empty() {
                    return Ok(());
                }
                let prior: Vec<(usize, u32)> = lanes
                    .iter()
                    .map(|l| store.request_state(&l.epoch).map_or((0, 0), |s| (s.position, s.pending_seed.unwrap_or(0))))
                    .collect();
                let advances = {
                    let mut ex = store.executor(&slot.weights, &slot.config, &slot.scratch);
                    ex.provision_step(gpu, &plan)?;
                    let o = ex.forward_step(gpu, &plan)?;
                    ex.commit_step(gpu, &plan, o)?
                };
                exec_absorb(gpu, slot, store, fx, refs, ar_refs, &mut lanes, &plan, &advances, k, &prior)?;
                step += 1;
            }
        })();
        let mut rows = Vec::new();
        let mut ok = res.is_ok();
        for lane in lanes.iter_mut() {
            if res.is_ok() && store.request_state(&lane.epoch).is_some() {
                exec_finish(gpu, slot, store, refs, lane)?;
            }
            ok &= lane.diff.exact();
            rows.push(json!({
                "slot": lane.slot,
                "lane": if lane.ar.is_some() { "ar".to_string() } else { fx[refs[lane.reference.unwrap()].0].name() },
                "stop": lane.stop, "cancelled": lane.cancelled,
                "finish": if lane.ar.is_some() { json!(if lane.done { "length" } else { "" }) } else { json!(lane.tr.finish) },
                "picks": if lane.ar.is_some() { lane.ar_picks.len() } else { lane.tr.emitted.len() },
                "cycles": lane.tr.cycles.len(),
                "ks_used": lane.tr.cycles.iter().map(|c| c.k).collect::<std::collections::BTreeSet<_>>(),
                "exact": lane.diff.exact(), "diff": lane.diff.json(),
            }));
        }
        for lane in &lanes {
            if store.request_state(&lane.epoch).is_some() {
                store.retire(&lane.epoch)?.free_gpu(gpu)?;
            }
        }
        let ctl_ok = ctl.values().all(|v| v["ok"].as_bool().unwrap_or(true));
        if controls {
            ok &= ctl_ok && ctl.len() >= 5;
        }
        eprintln!("spec exec {name}: exact={ok}{}", if controls { format!(" controls_ok={ctl_ok}") } else { String::new() });
        let mut j = json!({"case": name, "exact": ok, "lanes": rows});
        if controls {
            j["controls"] = Value::Object(ctl);
        }
        if let Err(e) = res {
            j["error"] = json!(e.to_string());
        }
        out.push(j);
        Ok(ok)
    };
    for &kk in &args.ks {
        let n = kk.min(fx.len()).min(EXEC_WIDTH - 1);
        let lanes: Vec<(usize, Option<u32>)> = (0..n).map(|i| (main_ref[i], None)).collect();
        // Mixed AR+MTP: an AR request shares every step (prefill chunks, then AR rows).
        pass &= case(gpu, slot, &mut store, &format!("exec_k{kk}"), lanes, Some(kk % fx.len()), false)?;
    }
    let mixed: Vec<(usize, Option<u32>)> = (0..fx.len().min(EXEC_WIDTH - 1))
        .map(|i| match i % 3 {
            0 => eos_ref[i].map_or((main_ref[i], None), |r| (main_ref[i], Some(refs[r].2))),
            1 => (clip_ref[i].unwrap_or(main_ref[i]), None),
            _ => (main_ref[i], None),
        })
        .collect();
    pass &= case(gpu, slot, &mut store, "exec_stop_clip_mix", mixed, None, false)?;
    let ctl_lanes: Vec<(usize, Option<u32>)> = (0..fx.len().min(4)).map(|i| (main_ref[i], None)).collect();
    pass &= case(gpu, slot, &mut store, "exec_controls", ctl_lanes, Some(0), true)?;
    store.free_gpu(gpu)?;
    Ok(pass)
}

fn ctl_record(map: &mut serde_json::Map<String, Value>, name: &str, ok: bool, detail: Value) {
    eprintln!("spec exec control {name}: {}", if ok { "OK" } else { "FAIL" });
    map.insert(name.into(), json!({"ok": ok, "detail": detail}));
}

/// Executor spec controls on live lanes 0 and 1 (both spec lanes):
/// stale-epoch provision refused; abort after provision drops the draft and
/// leaves the lane live (it must stay byte-exact); a forwarded Verify step
/// committed with a stale epoch or a forged step id is refused; aborting it
/// (cancel during verify) poisons that lane only, which is then retired.
fn exec_controls(gpu: &mut Gpu, slot: &mut ModelSlot, store: &mut Qwen35VmmStore, fx: &[Fixture], lanes: &mut [ExecLane], k: usize, map: &mut serde_json::Map<String, Value>) -> Result<()> {
    let live: Vec<usize> = (0..lanes.len()).filter(|&i| !lanes[i].done && lanes[i].ar.is_none()).collect();
    if live.len() < 2 {
        ctl_record(map, "setup", false, json!("fewer than two live spec lanes at step 3"));
        return Ok(());
    }
    let (a, b) = (live[0], live[1]);
    let ex = |store: &mut Qwen35VmmStore, slot: &ModelSlot, gpu: &mut Gpu, plan: &BatchStepPlan| -> std::result::Result<(), String> {
        store.executor(&slot.weights, &slot.config, &slot.scratch).provision_step(gpu, plan)
    };
    // Stale epoch at provision.
    let mut p = exec_plan(store, gpu, slot, fx, lanes, k, Some(b))?;
    p.requests[0].epoch.owner_generation += 1;
    let r = ex(store, slot, gpu, &p);
    if r.is_ok() {
        store.abort_step(&p);
    }
    ctl_record(map, "neg_stale_epoch_provision", r.is_err(), json!(r.err()));
    // Abort after provision: drafts dropped, lane b stays live (not poisoned).
    let p = exec_plan(store, gpu, slot, fx, lanes, k, Some(b))?;
    let r = ex(store, slot, gpu, &p);
    store.abort_step(&p);
    let live_b = store.request_state(&lanes[b].epoch).is_some_and(|s| !s.poisoned);
    ctl_record(map, "abort_after_provision_keeps_lane", r.is_ok() && live_b, json!({"provision": r.err(), "live": live_b}));
    // Forwarded Verify step for lane a, then stale commits, then cancel.
    let p = exec_plan(store, gpu, slot, fx, lanes, k, Some(a))?;
    let (stale, forged) = {
        let mut e = store.executor(&slot.weights, &slot.config, &slot.scratch);
        e.provision_step(gpu, &p)?;
        let o = e.forward_step(gpu, &p)?;
        let mut ps = p.clone();
        ps.requests[0].epoch.owner_generation += 1;
        let stale = e.commit_step(gpu, &ps, StepOutput { step_id: o.step_id, target_picks: o.target_picks.clone() }).map(|_| ());
        let forged = e.commit_step(gpu, &p, StepOutput { step_id: o.step_id + 1, target_picks: o.target_picks.clone() }).map(|_| ());
        (stale, forged)
    };
    store.abort_step(&p);
    ctl_record(map, "neg_stale_epoch_commit", stale.is_err(), json!(stale.err()));
    ctl_record(map, "neg_forged_step_commit", forged.is_err(), json!(forged.err()));
    let poisoned_a = store.request_state(&lanes[a].epoch).is_some_and(|s| s.poisoned);
    let others_ok = lanes.iter().enumerate().filter(|(i, l)| *i != a && !l.done).all(|(_, l)| store.request_state(&l.epoch).is_some_and(|s| !s.poisoned));
    ctl_record(map, "cancel_during_verify_poisons_only_that_lane", poisoned_a && others_ok, json!({"poisoned": poisoned_a, "others_live": others_ok}));
    let p = exec_plan(store, gpu, slot, fx, lanes, k, Some(a))?;
    let r = ex(store, slot, gpu, &p);
    if r.is_ok() {
        store.abort_step(&p);
    }
    ctl_record(map, "neg_poisoned_provision", r.is_err(), json!(r.err()));
    store.retire(&lanes[a].epoch)?.free_gpu(gpu)?;
    lanes[a].cancelled = true;
    lanes[a].done = true;
    lanes[a].tr.finish = "cancelled".into();
    Ok(())
}

/// One batched lane: which reference it must reproduce, optional retire.
struct LaneSpec {
    reference: usize,
    drop_after: Option<usize>,
}

struct Lane {
    rs: Qwen35RequestState,
    st: MtpSpecState,
    tr: SpecTrace,
    history: Vec<u32>,
    budget: usize,
    eos: u32,
    diff: Diff,
    done: bool,
    cancelled: bool,
}

fn stage_check(d: &mut Diff, reference: &SpecTrace, stage: &str, states: Vec<(String, Vec<u8>)>) -> Result<()> {
    for (name, bytes) in states {
        let key = format!("{stage}/{name}");
        match reference.states.iter().find(|(n, _)| *n == key) {
            Some((_, p)) => d.check(&key, &fs::read(p)?, &bytes),
            None => d.differing_items.push(format!("{key}: missing in reference")),
        }
    }
    Ok(())
}

fn has_stage(reference: &SpecTrace, stage: &str) -> bool {
    reference.states.iter().any(|(n, _)| n.starts_with(&format!("{stage}/")))
}

/// Open a request lane: private VMM KV + DeltaNet + MTP state, prefilled by
/// the singleton MTP prompt fill exactly as `Qwen35MtpDrafter::mtp_prefill`
/// (cold: DN and MTP state reset, greedy request installed).
fn lane_open(gpu: &mut Gpu, slot: &mut ModelSlot, head: &Qwen35MtpHead, k: usize, f: &Fixture, budget: usize, eos: u32, tag: u64) -> Result<Lane> {
    let init = VmmRequestInit { prompt_len: f.prefix, stop_ids: vec![], sampler: SamplerConfig::greedy(), rng_state: 0, history: vec![] };
    let epoch = RequestEpoch { request_tag: tag, owner_generation: 1 };
    let mut rs = Qwen35RequestState::new_like(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, epoch, 0, init)?;
    std::mem::swap(&mut slot.kv_cache, &mut rs.kv);
    std::mem::swap(&mut slot.dn_state, &mut rs.dn);
    let result = (|| -> Result<(MtpSpecState, u32)> {
        let mut st = MtpSpecState::new_for_slot_with_kv_mode_and_verify_capacity(gpu, slot, head, k, k, MtpKvMode::Q8)?;
        if let Some(cvs) = head.weights.compressed_vocab_size {
            st.mtp_scratch.ensure_compressed_logits(gpu, cvs)?;
            st.ensure_compressed_lm_logits(gpu, cvs)?;
        }
        // `apply_request(SpecRequestConfig::default())`: greedy, penalties off.
        let cfg = SpecRequestConfig::default();
        st.set_sampling(
            MtpSamplingConfig {
                temp: cfg.temp, top_k: cfg.ar_candidate_cap(), top_p: cfg.top_p.min(1.0), min_p: cfg.min_p,
                repeat_penalty: cfg.repeat_penalty, repeat_window: 0,
                presence_penalty: cfg.presence_penalty, frequency_penalty: cfg.frequency_penalty,
            },
            cfg.rng_seed,
        );
        st.penalty = hipfire_runtime::spec_sampling::PenaltyHistory::new(0);
        slot.dn_state.reset(gpu)?;
        slot.kv_cache.compact_offset = 0;
        st.reset(gpu)?;
        let route = MtpPromptRoute::from_own_prefill(hipfire_config::mtp_own_prefill());
        prefill_trunk_and_mtp_cache(gpu, slot, head, &mut st, &f.tokens, 0, route)?;
        gpu.hip.device_synchronize()?;
        let seed = argmax(&read_dev(gpu, &slot.scratch.logits, 0, slot.config.vocab_size * 4)?)?;
        Ok((st, seed))
    })();
    std::mem::swap(&mut slot.kv_cache, &mut rs.kv);
    std::mem::swap(&mut slot.dn_state, &mut rs.dn);
    let (st, seed) = match result {
        Ok(v) => v,
        Err(e) => {
            rs.free_gpu(gpu)?;
            return Err(e);
        }
    };
    let mut history = f.tokens.clone();
    history.push(seed);
    let finish = if seed == eos { "stop".to_string() } else { String::new() };
    let tr = SpecTrace { emitted: vec![seed], cycles: vec![], finish, position: f.prefix, pending_seed: seed, states: vec![] };
    let done = !tr.finish.is_empty();
    Ok(Lane { rs, st, tr, history, budget, eos, diff: Diff::default(), done, cancelled: false })
}

/// One batched case: open every lane, run shared `mtp_cb_cycle`s until all
/// finish, compare each lane's windows, ids, per-cycle state digests and
/// stage state bytes with its isolated singleton reference.
fn spec_batch_case(gpu: &mut Gpu, slot: &mut ModelSlot, head: &Qwen35MtpHead, k: usize, fx: &[Fixture], refs: &[(usize, usize, u32, SpecTrace)], specs: &[LaneSpec], name: &str) -> Result<(Value, bool)> {
    let mut lanes: Vec<Lane> = Vec::with_capacity(specs.len());
    let mut result = (|| -> Result<()> {
        for (i, s) in specs.iter().enumerate() {
            let (fi, budget, eos, ref reference) = refs[s.reference];
            let mut lane = lane_open(gpu, slot, head, k, &fx[fi], budget, eos, 50_000 + i as u64)?;
            let st = spec_full_state(gpu, &slot.config, &lane.rs.kv, &lane.rs.dn, &lane.st, lane.tr.position)?;
            stage_check(&mut lane.diff, reference, "prefill", st)?;
            lanes.push(lane);
        }
        let cb = MtpCbScratch::new(gpu, &slot.config, specs.len() * (k + 1))?;
        let run = (|| -> Result<()> {
            loop {
                let active: Vec<usize> = (0..lanes.len()).filter(|&i| !lanes[i].done).collect();
                if active.is_empty() {
                    return Ok(());
                }
                let results = {
                    let mut cbl: Vec<MtpCbLane> = Vec::with_capacity(active.len());
                    for lane in lanes.iter_mut().filter(|l| !l.done) {
                        let max_emit = lane.budget - lane.tr.emitted.len();
                        let kk = max_emit.saturating_sub(1).min(k);
                        cbl.push(MtpCbLane {
                            kv_cache: &mut lane.rs.kv, dn_state: &mut lane.rs.dn, state: &mut lane.st,
                            cur_pos: lane.tr.position, last_committed: lane.tr.pending_seed,
                            emitted: &lane.history, eos_token_id: lane.eos, k: kk,
                        });
                    }
                    mtp_cb_cycle(gpu, &slot.weights, &slot.config, &mut slot.scratch, head, &cb, &mut cbl)?.0
                };
                for (&i, w) in active.iter().zip(results) {
                    let lane = &mut lanes[i];
                    let reference = &refs[specs[i].reference].3;
                    let max_emit = lane.budget - lane.tr.emitted.len();
                    let kk = max_emit.saturating_sub(1).min(k);
                    if w.committed.is_empty() || w.committed.len() > max_emit {
                        return Err(format!("lane {i}: committed {} with max_emit {max_emit}", w.committed.len()).into());
                    }
                    let start = lane.tr.position;
                    let end = start + w.committed.len();
                    let class = spec_class(&hipfire_runtime::spec::MtpWindow {
                        committed: w.committed.clone(), accepted: w.accept_count, drafts_generated: w.drafts_generated,
                    });
                    let delta_sha256 = cycle_digest(gpu, &slot.config, &lane.rs.kv, &lane.rs.dn, &lane.st, start, end)?;
                    lane.tr.cycles.push(SpecCycle {
                        position: start, seed: lane.tr.pending_seed, k: kk, committed: w.committed.clone(),
                        accepted: w.accept_count, drafted: w.drafts_generated, class, delta_sha256,
                    });
                    lane.tr.emitted.extend_from_slice(&w.committed);
                    lane.history.extend_from_slice(&w.committed);
                    lane.tr.position = end;
                    lane.tr.pending_seed = *w.committed.last().unwrap();
                    let c = lane.tr.cycles.len() - 1;
                    let stage = format!("first_{class}_cycle{c}");
                    if has_stage(reference, &stage) {
                        let st = spec_full_state(gpu, &slot.config, &lane.rs.kv, &lane.rs.dn, &lane.st, end)?;
                        stage_check(&mut lane.diff, reference, &stage, st)?;
                    }
                    if w.committed.contains(&lane.eos) {
                        lane.tr.finish = "stop".into();
                    } else if lane.tr.emitted.len() >= lane.budget {
                        lane.tr.finish = "length".into();
                    }
                    if specs[i].drop_after == Some(lane.tr.cycles.len()) && lane.tr.finish.is_empty() {
                        lane.cancelled = true;
                        lane.tr.finish = "cancelled".into();
                    }
                    lane.done = !lane.tr.finish.is_empty();
                }
            }
        })();
        cb.free_gpu(gpu)?;
        run
    })();
    let mut rows = Vec::new();
    let mut all = true;
    let mut mixed_k = false;
    for (i, lane) in lanes.iter_mut().enumerate() {
        let (fi, budget, eos, ref reference) = refs[specs[i].reference];
        if result.is_ok() && !lane.cancelled {
            let st = spec_full_state(gpu, &slot.config, &lane.rs.kv, &lane.rs.dn, &lane.st, lane.tr.position)?;
            stage_check(&mut lane.diff, reference, "final", st)?;
            let d2 = spec_compare(reference, &lane.tr, &[])?;
            lane.diff.compared_items += d2.compared_items;
            lane.diff.compared_bytes += d2.compared_bytes;
            lane.diff.differing_items.extend(d2.differing_items);
        } else if lane.cancelled {
            // Cancelled lane: its committed prefix must still match.
            let n = lane.tr.cycles.len();
            for c in 0..n {
                let (a, b) = (&reference.cycles[c], &lane.tr.cycles[c]);
                lane.diff.ids(&format!("cycle{c}.committed"), &a.committed, &b.committed);
                lane.diff.check(&format!("cycle{c}.delta_sha256"), a.delta_sha256.as_bytes(), b.delta_sha256.as_bytes());
            }
        }
        mixed_k |= lane.tr.cycles.iter().any(|c| c.k < k);
        let exact = lane.diff.exact();
        all &= exact;
        rows.push(json!({
            "lane": i, "fixture": fx[fi].name(), "budget": budget, "eos": eos, "cancelled": lane.cancelled,
            "finish": lane.tr.finish, "picks": lane.tr.emitted.len(), "cycles": lane.tr.cycles.len(),
            "ks_used": lane.tr.cycles.iter().map(|c| c.k).collect::<std::collections::BTreeSet<_>>(),
            "classes": spec_trace_json(&lane.tr)["classes"], "exact": exact, "diff": lane.diff.json(),
        }));
    }
    for lane in lanes {
        lane.st.free_gpu(gpu);
        if let Err(e) = lane.rs.free_gpu(gpu) {
            result = result.and(Err(e.into()));
        }
    }
    let ok = result.is_ok() && all;
    let mut j = json!({"case": name, "lanes": specs.len(), "exact": ok, "mixed_k": mixed_k, "requests": rows});
    if let Err(e) = result {
        j["error"] = json!(e.to_string());
    }
    Ok((j, ok))
}

// ── DFlash Gate 0: one-window replay probe ───────────────────────────
//
// Question answered: is a DFlash chain-verify window, replayed as 1..4 lanes of
// ONE shared multi-request trunk forward (ChainVerify fusion, per-lane hidden
// ring + tape) + ONE shared head + the extracted greedy accept consumer,
// byte-identical (verify rows, KV rows, DeltaNet, tape, ring, draft state,
// committed ids) to the isolated singleton `DflashSpeculator` window?
//
// Every lane's pre-window state is produced by the SAME deterministic
// singleton trajectory (prefill + `--dflash-probe-window` full windows) on
// that lane's own VMM KV/DeltaNet, then its draft owners are moved out with
// `take_vmm_lane`/`from_snapshot`. The pre-window bytes of every lane are
// compared with a separately produced reference trajectory (determinism +
// lossless-transfer check) before the window is replayed.

type Named = Vec<(String, Vec<u8>)>;
type Frozen = Vec<(String, PathBuf)>;

/// Output budget of the full windows that precede the probed one (a full
/// block, no budget clamp).
const PRE_WINDOW_EMIT: usize = 1 << 20;

fn le_u64s(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Device read that tolerates zero-length spans.
fn rd(gpu: &Gpu, t: &GpuTensor, offset: usize, bytes: usize) -> Result<Vec<u8>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    read_dev(gpu, t, offset, bytes)
}

fn art_name(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' }).collect()
}

fn freeze(dir: &Path, set: &Named) -> Result<Frozen> {
    let mut out = Vec::with_capacity(set.len());
    for (name, bytes) in set {
        let p = dir.join(format!("{}.bin", art_name(name)));
        write_art(&p, bytes)?;
        out.push((name.clone(), p));
    }
    Ok(out)
}

/// Byte-compare `cand` against the frozen reference arrays (both directions:
/// an array present on one side only is a difference).
fn cmp_frozen(d: &mut Diff, stage: &str, frozen: &Frozen, cand: &Named) -> Result<()> {
    for (name, bytes) in cand {
        let key = format!("{stage}/{name}");
        match frozen.iter().find(|(n, _)| n == name) {
            Some((_, p)) => d.check(&key, &fs::read(p)?, bytes),
            None => d.differing_items.push(format!("{key}: missing in reference")),
        }
    }
    for (name, _) in frozen {
        if !cand.iter().any(|(n, _)| n == name) {
            d.differing_items.push(format!("{stage}/{name}: missing in candidate"));
        }
    }
    Ok(())
}

/// Live span of a draft context ring: the last `min(rows, modulus)` rows
/// ending at `rows`, in logical order (slot = row % modulus; identity when
/// `modulus == usize::MAX`). Empty when the tensor cannot hold the span
/// (an unused allocation).
fn ring_span(gpu: &Gpu, t: &GpuTensor, row_bytes: usize, rows: usize, modulus: usize) -> Result<Vec<u8>> {
    let start = rows.saturating_sub(modulus);
    if t.buf.size() < rows.min(modulus) * row_bytes {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity((rows - start) * row_bytes);
    for (_row0, slot0, len) in ring_segments(start, rows, modulus) {
        out.extend(rd(gpu, t, slot0 * row_bytes, len * row_bytes)?);
    }
    Ok(out)
}

/// Lane/singleton draft + ring state that survives between windows: hidden
/// ring cursors and live rows in logical order, `TargetHiddenLog` cursors and
/// absolute positions, valid spans of the draft context rings
/// (`target_hidden`, projection, per-layer K/V, windowed full-layer K/V),
/// host-shadow length, and checkpoint positions. Never the unwritten parts of
/// an allocation.
fn df_arrays(gpu: &Gpu, df: &DflashState, ckpt: &[usize]) -> Result<Named> {
    gpu.hip.device_synchronize()?;
    let mut out: Named = Vec::new();
    let rb = &df.hidden_rb;
    let live = rb.written.min(rb.max_positions);
    out.push(("ring.cursors".into(), le_u64s(&[rb.head as u64, rb.written as u64, live as u64, rb.max_positions as u64])));
    let row = rb.hidden_dim * 4;
    let s = (rb.head + rb.max_positions - live) % rb.max_positions;
    let first = live.min(rb.max_positions - s);
    for (i, t) in rb.layer_bufs.iter().enumerate() {
        let mut v = rd(gpu, t, s * row, first * row)?;
        v.extend(rd(gpu, t, 0, (live - first) * row)?);
        out.push((format!("ring.L{:02}", rb.extract_layers[i]), v));
    }
    let sc = &df.draft_scratch;
    let th = &sc.thlog;
    let (uploaded, proj, full) = (th.uploaded_rows(), th.proj_cached_rows(), th.full_cached_rows());
    out.push(("thlog.cursors".into(), le_u64s(&[uploaded as u64, proj as u64, full as u64])));
    out.push(("thlog.abs_positions".into(), th.abs_positions().iter().flat_map(|p| p.to_le_bytes()).collect()));
    let (w, wf) = match sc.ctx_mode {
        DraftCtxMode::Legacy => (usize::MAX, usize::MAX),
        DraftCtxMode::Windowed { w, w_full } => (w, w_full),
    };
    let (h, ne, kvd) = (df.draft_config.hidden, df.draft_config.num_extract(), df.draft_config.kv_dim());
    out.push(("draft.target_hidden".into(), ring_span(gpu, &sc.target_hidden, ne * h * 4, uploaded, w)?));
    out.push(("draft.target_hidden_proj".into(), ring_span(gpu, &sc.target_hidden_proj, h * 4, proj, w)?));
    let n_l = sc.k_ctx_cached.len();
    for l in 0..n_l {
        // Windowed mode: the last layer's K/V live in the full-layer rings.
        if sc.k_full_cached.is_some() && l + 1 == n_l {
            continue;
        }
        out.push((format!("draft.k_ctx.L{l:02}"), ring_span(gpu, &sc.k_ctx_cached[l], kvd * 4, proj, w)?));
        out.push((format!("draft.v_ctx.L{l:02}"), ring_span(gpu, &sc.v_ctx_cached[l], kvd * 4, proj, w)?));
    }
    if let (Some(k), Some(v)) = (&sc.k_full_cached, &sc.v_full_cached) {
        out.push(("draft.k_full".into(), ring_span(gpu, k, kvd * 4, full, wf)?));
        out.push(("draft.v_full".into(), ring_span(gpu, v, kvd * 4, full, wf)?));
    }
    out.push(("draft.sizes".into(), le_u64s(&[df.target_hidden_host.len() as u64, df.ctx_capacity as u64, df.block_size as u64])));
    out.push(("checkpoints.positions".into(), le_u64s(&ckpt.iter().map(|&p| p as u64).collect::<Vec<_>>())));
    Ok(out)
}

/// Per-window arrays that survive the accept: the verify post-norm hidden and
/// logits rows, packed argmax, ring staging rows of the actual `b`, every
/// tape layer's qkv/alpha/beta rows, and every KV row the verify wrote
/// (`[pos, pos+b)`, including the rejected tail).
fn window_arrays(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, df: &DflashState, pos: usize, b: usize) -> Result<Named> {
    gpu.hip.device_synchronize()?;
    let (dim, vocab) = (config.dim, config.vocab_size);
    let vs = &df.verify_scratch;
    let mut out: Named = vec![
        ("verify.final_hidden".into(), rd(gpu, &vs.final_hidden, 0, b * dim * 4)?),
        ("verify.logits".into(), rd(gpu, &vs.logits, 0, b * vocab * 4)?),
        ("verify.argmax".into(), rd(gpu, &vs.argmax, 0, b * 4)?),
    ];
    let rb = &df.hidden_rb;
    for (i, t) in rb.staging_bufs.iter().enumerate() {
        out.push((format!("ring.staging.L{:02}", rb.extract_layers[i]), rd(gpu, t, 0, b * rb.hidden_dim * 4)?));
    }
    let tp = &df.gdn_tape;
    for i in 0..tp.qkv_bufs.len() {
        out.push((format!("tape.{i:02}.qkv"), rd(gpu, &tp.qkv_bufs[i], 0, b * tp.qkv_dim * 4)?));
        out.push((format!("tape.{i:02}.alpha"), rd(gpu, &tp.alpha_bufs[i], 0, b * tp.n_v_heads * 4)?));
        out.push((format!("tape.{i:02}.beta"), rd(gpu, &tp.beta_bufs[i], 0, b * tp.n_v_heads * 4)?));
    }
    let (kr, vr) = kv_row_bytes(kv)?;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        if *ty == LayerType::FullAttention {
            out.push((format!("kv_verify.L{layer:02}.k"), rd(gpu, &kv.k_gpu[layer], pos * kr, b * kr)?));
            out.push((format!("kv_verify.L{layer:02}.v"), rd(gpu, &kv.v_gpu[layer], pos * vr, b * vr)?));
        }
    }
    Ok(out)
}

/// Committed target state at `rows` plus the lane's draft/ring state.
fn full_state(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, dn: &DeltaNetState, lane: &DflashVmmLaneState, rows: usize) -> Result<Named> {
    let mut v = state_bytes(gpu, config, kv, dn, rows)?;
    let ck: Vec<usize> = lane.checkpoints.iter().map(|(p, _)| *p).collect();
    v.extend(df_arrays(gpu, &lane.df, &ck)?);
    Ok(v)
}

/// State a refused shared trunk must leave untouched: the committed target
/// prefix + DeltaNet and the lane's hidden ring (the draft owners already ran
/// their own draft before the refusal, so they are excluded).
fn refusal_state(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, dn: &DeltaNetState, lane: &DflashVmmLaneState, rows: usize) -> Result<Named> {
    let mut v = state_bytes(gpu, config, kv, dn, rows)?;
    let ck: Vec<usize> = Vec::new();
    v.extend(df_arrays(gpu, &lane.df, &ck)?.into_iter().filter(|(n, _)| n.starts_with("ring.")));
    Ok(v)
}

fn dflash_of(d: &mut Box<dyn Speculator>) -> Result<&mut DflashSpeculator> {
    let any = d.drafter_any_mut().ok_or("speculator exposes no concrete drafter (not DFlash?)")?;
    any.downcast_mut::<DflashSpeculator>().ok_or_else(|| Box::<dyn Error>::from("speculator is not a DflashSpeculator"))
}

struct G0Cfg {
    /// Full windows run before the probed window.
    windows: usize,
    /// Configured (full) block size of the loaded draft.
    block: usize,
    dir: PathBuf,
}

struct Walked {
    position: usize,
    seed: u32,
    history: Vec<u32>,
}

/// The singleton trajectory: cold prefill through the production
/// `Speculator::prefill`, then `windows` full-block `Speculator::step`s.
fn dflash_walk(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Box<dyn Speculator>, f: &Fixture, windows: usize) -> Result<Walked> {
    // `DeltaNetSnapshot` copies by tensor allocation size (`save_from`: src
    // size, `restore_to`: snapshot size), so its buffers must be exactly the
    // sizes of the DeltaNet that runs. A fresh request DN's pooled tensors can
    // be larger than the resident one's (`Qwen35RequestState::copy_dn_from`
    // documents it): re-size the speculator's live state from the DN about
    // to run by moving it out and back (`take_vmm_lane` allocates its
    // replacement from the `dn` passed) and dropping the moved-out copy.
    let resized = dflash_of(d)?.take_vmm_lane(gpu, &slot.config, &slot.dn_state, 0)?;
    resized.free_gpu(gpu);
    d.configure_request(SpecRequestConfig::default());
    slot.dn_state.reset(gpu)?;
    slot.kv_cache.compact_offset = 0;
    pin_gdn_frame(&slot.dn_state);
    let first = match d.prefill(gpu, &mut *slot, &f.tokens, &f.tokens, 0, false, None, &|| false)? {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => return Err("DFlash prefill aborted".into()),
    };
    let mut history = f.tokens.clone();
    history.push(first);
    let (mut position, mut seed) = (f.prefix, first);
    for _ in 0..windows {
        let s = d.step(gpu, &mut *slot, position, seed, &history, None, 0.0, PRE_WINDOW_EMIT)?;
        history.extend_from_slice(&s.emit);
        position += s.emit.len();
        seed = s.next_seed;
    }
    Ok(Walked { position, seed, history })
}

/// Move the speculator's live draft owners out as a lane (`take_vmm_lane` +
/// `from_snapshot`); returns the lane and the snapshot's reported rows.
fn take_lane(gpu: &mut Gpu, slot: &ModelSlot, d: &mut Box<dyn Speculator>, position: usize) -> Result<(DflashVmmLaneState, usize)> {
    let snap = dflash_of(d)?.take_vmm_lane(gpu, &slot.config, &slot.dn_state, position)?;
    let rows = snap.rows();
    Ok((DflashVmmLaneState::from_snapshot(snap), rows))
}

fn ctx_mode_str(df: &DflashState) -> String {
    match df.draft_scratch.ctx_mode {
        DraftCtxMode::Legacy => "Legacy".into(),
        DraftCtxMode::Windowed { w, w_full } => format!("Windowed w={w} w_full={w_full}"),
    }
}

/// Frozen pre-window bytes of one fixture's singleton trajectory.
struct PreRef {
    frozen: Frozen,
    position: usize,
    seed: u32,
    rows: usize,
    ctx_mode: String,
}

/// Frozen outputs of the isolated singleton probed window of `(fixture, E)`.
struct PostRef {
    frozen: Frozen,
    position: usize,
    seed: u32,
    b: usize,
    emit: Vec<u32>,
    next_seed: u32,
    accepted: usize,
    proposed: usize,
    new_pos: usize,
    step_ms: f64,
    /// Verify route the singleton took (`graph_replay`/`graph_capture`/`eager_warmup`; graph path only when HIPFIRE_VERIFY_GRAPH admits it).
    route_hint: &'static str,
}

#[derive(Default)]
struct G0Refs {
    pre: std::collections::HashMap<usize, PreRef>,
    post: std::collections::HashMap<(usize, usize), PostRef>,
    draft: std::collections::HashMap<usize, DraftRef>,
}

fn ensure_pre(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Box<dyn Speculator>, cfg: &G0Cfg, fx: &[Fixture], refs: &mut G0Refs, fi: usize) -> Result<()> {
    if refs.pre.contains_key(&fi) {
        return Ok(());
    }
    eprintln!("gate0 trace: ref_pre walk {}", fx[fi].name());
    let w = dflash_walk(gpu, slot, d, &fx[fi], cfg.windows)?;
    let (lane, rows) = take_lane(gpu, slot, d, w.position)?;
    let set = full_state(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, &lane, w.position);
    let ctx_mode = ctx_mode_str(&lane.df);
    lane.free_gpu(gpu);
    let frozen = freeze(&cfg.dir.join(format!("ref_pre_{}", fx[fi].name())), &set?)?;
    refs.pre.insert(fi, PreRef { frozen, position: w.position, seed: w.seed, rows, ctx_mode });
    Ok(())
}

fn ensure_post(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Box<dyn Speculator>, cfg: &G0Cfg, fx: &[Fixture], refs: &mut G0Refs, fi: usize, e: usize) -> Result<()> {
    if refs.post.contains_key(&(fi, e)) {
        return Ok(());
    }
    eprintln!("gate0 trace: ref_post walk {} e={e}", fx[fi].name());
    let w = dflash_walk(gpu, slot, d, &fx[fi], cfg.windows)?;
    let b = cfg.block.min(e.max(2));
    // Which verify body the singleton takes for this window (graph cache is
    // empty after every `take_vmm_lane`, so this is deterministic in N).
    let route_hint = if gpu.graphs.verify_has_graph(b) {
        "graph_replay"
    } else if gpu.graphs.verify_needs_warmup(b) {
        "eager_warmup"
    } else {
        "graph_capture"
    };
    let t0 = std::time::Instant::now();
    let s = d.step(gpu, &mut *slot, w.position, w.seed, &w.history, None, 0.0, e)?;
    gpu.hip.device_synchronize()?;
    let step_ms = t0.elapsed().as_secs_f64() * 1e3;
    let new_pos = w.position + s.emit.len();
    // Target side (singleton owners) before the draft owners move out.
    let mut set = state_bytes(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, new_pos)?;
    let (lane, _rows) = take_lane(gpu, slot, d, new_pos)?;
    let df_set = (|| -> Result<Named> {
        let ck: Vec<usize> = lane.checkpoints.iter().map(|(p, _)| *p).collect();
        let mut v = df_arrays(gpu, &lane.df, &ck)?;
        v.extend(window_arrays(gpu, &slot.config, &slot.kv_cache, &lane.df, w.position, b)?);
        Ok(v)
    })();
    lane.free_gpu(gpu);
    set.extend(df_set?);
    let frozen = freeze(&cfg.dir.join(format!("ref_post_{}_e{e}", fx[fi].name())), &set)?;
    refs.post.insert(
        (fi, e),
        PostRef {
            frozen,
            position: w.position,
            seed: w.seed,
            b,
            emit: s.emit.to_vec(),
            next_seed: s.next_seed,
            accepted: s.accepted,
            proposed: s.proposed,
            new_pos,
            step_ms,
            route_hint,
        },
    );
    Ok(())
}

/// One batched lane between open and compare.
struct LaneRun {
    fi: usize,
    e: usize,
    b: usize,
    rs: Qwen35RequestState,
    lane: DflashVmmLaneState,
    position: usize,
    seed: u32,
    draft: Option<DflashCbDraft>,
    snapshot_rows: usize,
    pre_diff: Diff,
    post_diff: Diff,
}

/// Open a lane: its own VMM KV/DeltaNet runs the singleton trajectory to the
/// pre-window state; the draft owners are then moved out into the lane. The
/// pre-window bytes are compared with the fixture's frozen reference.
#[allow(clippy::too_many_arguments)]
fn g0_lane_open(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Box<dyn Speculator>, cfg: &G0Cfg, fx: &[Fixture], refs: &G0Refs, fi: usize, e: usize, tag: u64) -> Result<LaneRun> {
    let f = &fx[fi];
    let init = VmmRequestInit { prompt_len: f.prefix, stop_ids: vec![], sampler: SamplerConfig::greedy(), rng_state: 0, history: vec![] };
    let epoch = RequestEpoch { request_tag: tag, owner_generation: 1 };
    let mut rs = Qwen35RequestState::new_like(gpu, &slot.config, &slot.kv_cache, &slot.dn_state, epoch, 0, init)?;
    std::mem::swap(&mut slot.kv_cache, &mut rs.kv);
    std::mem::swap(&mut slot.dn_state, &mut rs.dn);
    {
        let all = |dn: &DeltaNetState| -> Vec<usize> {
            dn.s_matrices.iter().chain(&dn.s_scales).chain(&dn.conv_states).chain(&dn.s_ef_residual).map(|t| t.buf.size()).collect()
        };
        // After the swap above: slot.dn_state is the request owner, rs.dn the resident native one.
        let (nat, req) = (all(&rs.dn), all(&slot.dn_state));
        let bad: Vec<(usize, usize, usize)> = nat.iter().zip(&req).enumerate().filter(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, *a, *b)).collect();
        eprintln!("gate0 trace: DN tensors native={} request={} mismatching(idx,native,request)={:?}", nat.len(), req.len(), &bad[..bad.len().min(8)]);
    }
    eprintln!("gate0 trace: lane walk {} (request-owned KV/DN swapped in)", f.name());
    let walked = dflash_walk(gpu, slot, d, f, cfg.windows);
    std::mem::swap(&mut slot.kv_cache, &mut rs.kv);
    std::mem::swap(&mut slot.dn_state, &mut rs.dn);
    let w = match walked {
        Ok(w) => w,
        Err(err) => {
            rs.free_gpu(gpu)?;
            return Err(err);
        }
    };
    // The lane moves out with the singleton's NATIVE DeltaNet sizing: every
    // replacement state `take_vmm_lane` allocates is sized from the `dn`
    // passed here, so the resident `slot.dn_state` (not the swapped request
    // owner) keeps all trajectories' snapshots interchangeable.
    let (lane, snapshot_rows) = match take_lane(gpu, slot, d, w.position) {
        Ok(v) => v,
        Err(err) => {
            rs.free_gpu(gpu)?;
            return Err(err);
        }
    };
    let pre_set = match full_state(gpu, &slot.config, &rs.kv, &rs.dn, &lane, w.position) {
        Ok(v) => v,
        Err(err) => {
            lane.free_gpu(gpu);
            rs.free_gpu(gpu)?;
            return Err(err);
        }
    };
    let pr = &refs.pre[&fi];
    let mut pre_diff = Diff::default();
    pre_diff.ids("pre.position_seed_rows", &[pr.position as u32, pr.seed, pr.rows as u32], &[w.position as u32, w.seed, snapshot_rows as u32]);
    if let Err(err) = cmp_frozen(&mut pre_diff, "pre", &pr.frozen, &pre_set) {
        lane.free_gpu(gpu);
        rs.free_gpu(gpu)?;
        return Err(err);
    }
    Ok(LaneRun {
        fi,
        e,
        b: cfg.block.min(e.max(2)),
        rs,
        lane,
        position: w.position,
        seed: w.seed,
        draft: None,
        snapshot_rows,
        pre_diff,
        post_diff: Diff::default(),
    })
}

fn free_lanes(gpu: &mut Gpu, lanes: Vec<LaneRun>) -> Result<()> {
    let mut res = Ok(());
    for l in lanes {
        l.lane.free_gpu(gpu);
        if let Err(e) = l.rs.free_gpu(gpu) {
            res = Err(e.into());
        }
    }
    res
}

// ── Exact wide verify chunks (PLAN-CHUNK64 §5, slice S) ──────────────
//
// The aggregate-boundary fixtures replay 63/64/65/96/128-row windows of whole
// short per-request blocks as lanes of the shared trunk, through two drivers:
// `Direct` (one `forward_prefill_batch_multi` + one head: exact iff the rows
// fit the effective cap, otherwise refused before any state write) and
// `Product` (`dflash_cb_verify`: whole lanes packed into chunks of the
// effective packing cap, i.e. 48+48+32 at 63 rows and ONE 128-row chunk with
// the wide route on). Every lane must equal the isolated singleton window.

/// Whole short per-request blocks summing to `total` rows, each lane `2..=block`
/// rows (e.g. 63 = 16+16+16+15, 65 = 16+16+16+9+8).
fn lane_blocks(total: usize, block: usize) -> Result<Vec<usize>> {
    if block < 3 || total < 2 {
        return Err(format!("cannot split {total} rows into whole blocks of 2..={block}").into());
    }
    let (full, rem) = (total / block, total % block);
    let mut v = vec![block; full];
    match rem {
        0 => {}
        1 => {
            if full == 0 {
                return Err(format!("cannot split {total} rows into whole blocks of 2..={block}").into());
            }
            v.pop();
            let a = (block + 2) / 2;
            v.push(a);
            v.push(block + 1 - a);
        }
        r => v.push(r),
    }
    Ok(v)
}

/// `(fixture, E)` per lane of `blocks`: a full block uses a budget past the
/// block (as the pre-existing 16-row cases), a short block `E = B`.
fn lane_specs(blocks: &[usize], block: usize, n_fixtures: usize) -> Vec<(usize, usize)> {
    blocks.iter().enumerate().map(|(i, &b)| (i % n_fixtures, if b == block { 64 } else { b })).collect()
}

/// Verify-window driver.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Direct,
    Product,
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Route::Direct => "direct",
            Route::Product => "product",
        }
    }
}

/// Graph-refusal arm engaged around one verify entry (the whole-CB graph
/// route stays refused in this ticket; this proves the refusal precedes any
/// mutation, launch or wide-route fallback).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GraphArm {
    /// `gpu.graphs.capture_mode` set (no real stream capture is opened, so a
    /// missing refusal would run the window eagerly and be caught by bytes).
    CaptureMode,
    /// `gpu.replay.begin_capture()` recording window.
    ReplayRecording,
}

impl GraphArm {
    fn name(self) -> &'static str {
        match self {
            GraphArm::CaptureMode => "graphs.capture_mode",
            GraphArm::ReplayRecording => "replay.is_recording",
        }
    }
}

/// Shared verify buffers of the probe.
struct G0Scratch<'a> {
    vs: &'a VerifyScratch,
    mc: &'a MultiChunkScratch,
    cb: &'a mut DflashCbScratch,
}

/// Per-launch kernel names of a span, from the profile collector (it records
/// every launch routed through `begin_timer`, with the launched symbol).
struct LaunchProbe;

impl LaunchProbe {
    fn start() -> Self {
        rdna_compute::profile::start();
        LaunchProbe
    }
    fn finish(self) -> BTreeMap<&'static str, usize> {
        let entries = rdna_compute::profile::stop().unwrap_or_default();
        std::mem::forget(self);
        let mut m = BTreeMap::new();
        for e in entries {
            *m.entry(e.kernel).or_insert(0) += 1;
        }
        m
    }
}

impl Drop for LaunchProbe {
    fn drop(&mut self) {
        let _ = rdna_compute::profile::stop();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum WideOp {
    Qkvza,
    Qkv,
    GateUp,
    Residual,
}

impl WideOp {
    const ALL: [WideOp; 4] = [WideOp::Qkvza, WideOp::Qkv, WideOp::GateUp, WideOp::Residual];
    fn label(self) -> &'static str {
        match self {
            WideOp::Qkvza => "qkvza",
            WideOp::Qkv => "qkv",
            WideOp::GateUp => "gate_up",
            WideOp::Residual => "residual",
        }
    }
}

struct WideSym {
    op: WideOp,
    pm: bool,
    bt: usize,
}

/// The frozen wide symbols (PLAN-CHUNK64 §3): HIP twins
/// `<existing prefix>_vt{4,8}w{4,8}_k32` and PM twins
/// `mq4_verify_{qkvza,qkv,gate_up,residual}_pm_gfx1201_bt{4,8}w{4,8}`.
fn parse_wide_symbol(name: &str) -> Option<WideSym> {
    const HIP: [(&str, WideOp); 4] = [
        ("gemm_qkvza_mq4g256v2_wmma_gfx12", WideOp::Qkvza),
        ("gemm_qkv_mq4g256v2_wmma_gfx12", WideOp::Qkv),
        ("gemm_gate_up_mq4g256v2_wmma_gfx12", WideOp::GateUp),
        ("gemm_mq4g256v2_residual_wmma_gfx12", WideOp::Residual),
    ];
    const PM: [(&str, WideOp); 4] = [
        ("mq4_verify_qkvza_pm_gfx1201_", WideOp::Qkvza),
        ("mq4_verify_qkv_pm_gfx1201_", WideOp::Qkv),
        ("mq4_verify_gate_up_pm_gfx1201_", WideOp::GateUp),
        ("mq4_verify_residual_pm_gfx1201_", WideOp::Residual),
    ];
    // "<pre>8w4" -> BT 8
    fn tile(s: &str, pre: &str) -> Option<usize> {
        let rest = s.strip_prefix(pre)?;
        let (bt, w) = rest.split_once('w')?;
        let (bt, w) = (bt.parse::<usize>().ok()?, w.parse::<usize>().ok()?);
        ((bt == 4 || bt == 8) && (w == 4 || w == 8)).then_some(bt)
    }
    for (p, op) in HIP {
        if let Some(rest) = name.strip_prefix(p) {
            let rest = rest.strip_suffix("_k32")?;
            return Some(WideSym { op, pm: false, bt: tile(rest, "_vt")? });
        }
    }
    for (p, op) in PM {
        if let Some(rest) = name.strip_prefix(p) {
            return Some(WideSym { op, pm: true, bt: tile(rest, "bt")? });
        }
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WideFamily {
    /// Wide route off or not admitted: 63-row product route, no wide symbol.
    Off,
    /// HIP K32 twins (`HIPFIRE_CB_VERIFY_PM=0`).
    Hip,
    /// PeaceMaker twins (the default inside the enabled wide route).
    Pm,
}

impl WideFamily {
    fn name(self) -> &'static str {
        match self {
            WideFamily::Off => "off",
            WideFamily::Hip => "hip",
            WideFamily::Pm => "pm",
        }
    }
}

/// What this process must observe for the wide route.
#[derive(Clone)]
struct WideExpect {
    family: WideFamily,
    /// `multi_chunk_row_cap(gpu)`.
    cap: usize,
    admitted: bool,
    pm_requested: bool,
    n_dn: usize,
    n_fa: usize,
    n_layers: usize,
}

impl WideExpect {
    fn new(gpu: &Gpu, slot: &ModelSlot) -> Self {
        let cap = multi_chunk_row_cap(gpu);
        let admitted = multi_chunk_wide_admitted(gpu, &slot.weights, &slot.config);
        let pm_requested = gpu.mq4_verify_pm_selected();
        let family = if admitted && cap > MULTI_CHUNK_PRODUCT_MAX_ROWS {
            if pm_requested {
                WideFamily::Pm
            } else {
                WideFamily::Hip
            }
        } else {
            WideFamily::Off
        };
        let lt = &slot.config.layer_types;
        Self {
            family,
            cap,
            admitted,
            pm_requested,
            n_dn: lt.iter().filter(|t| **t == LayerType::LinearAttention).count(),
            n_fa: lt.iter().filter(|t| **t == LayerType::FullAttention).count(),
            n_layers: lt.len(),
        }
    }

    /// Rows one shared chunk can hold on this process.
    fn route_cap(&self) -> usize {
        if self.family == WideFamily::Off {
            MULTI_CHUNK_PRODUCT_MAX_ROWS
        } else {
            self.cap
        }
    }

    /// Launches of `op` in ONE wide chunk: qkvza on every DeltaNet layer, qkv
    /// on every full-attention layer, gate_up and down on every layer; the
    /// residual family also runs wo/fa_wo on every layer and the lm_head once.
    fn per_chunk(&self, op: WideOp) -> usize {
        match op {
            WideOp::Qkvza => self.n_dn,
            WideOp::Qkv => self.n_fa,
            WideOp::GateUp => self.n_layers,
            WideOp::Residual => 2 * self.n_layers + 1,
        }
    }

    fn json(&self) -> Value {
        json!({
            "family": self.family.name(), "multi_chunk_row_cap": self.cap, "route_cap": self.route_cap(),
            "wide_admitted": self.admitted, "pm_requested": self.pm_requested,
            "n_deltanet_layers": self.n_dn, "n_full_attn_layers": self.n_fa, "n_layers": self.n_layers,
            "per_wide_chunk": {
                "qkvza": self.per_chunk(WideOp::Qkvza), "qkv": self.per_chunk(WideOp::Qkv),
                "gate_up": self.per_chunk(WideOp::GateUp), "residual_incl_wo_down_head": self.per_chunk(WideOp::Residual),
            },
        })
    }
}

/// Wide-route identity and launch-count assertions for one verify span.
/// `chunks` are the rows of every chunk the span ran (empty when refused).
/// Every chunk above 63 rows runs `per_chunk` launches of each family on the
/// BT tile for its rows (64 -> BT4, 65..=128 -> BT8) in the expected symbol
/// family (HIP `_k32` or PM, never mixed), with no quantized IU4 launch; a
/// refused span launches nothing at all.
fn assert_wide(exp: &WideExpect, chunks: &[usize], launches: &BTreeMap<&'static str, usize>, refused: bool) -> (bool, Value) {
    let mut want: BTreeMap<(WideOp, usize), usize> = BTreeMap::new();
    let mut wide_chunks = 0usize;
    if !refused {
        for &r in chunks {
            if r > MULTI_CHUNK_PRODUCT_MAX_ROWS {
                wide_chunks += 1;
                let bt = if r == 64 { 4 } else { 8 };
                for op in WideOp::ALL {
                    *want.entry((op, bt)).or_insert(0) += exp.per_chunk(op);
                }
            }
        }
    }
    let mut got: BTreeMap<(WideOp, usize), usize> = BTreeMap::new();
    let (mut hip_n, mut pm_n, mut qkvza_all) = (0usize, 0usize, 0usize);
    let mut wide_syms: BTreeMap<String, usize> = BTreeMap::new();
    let mut forbidden: Vec<String> = Vec::new();
    for (name, &n) in launches {
        if let Some(s) = parse_wide_symbol(name) {
            *got.entry((s.op, s.bt)).or_insert(0) += n;
            if s.pm {
                pm_n += n;
            } else {
                hip_n += n;
            }
            wide_syms.insert(name.to_string(), n);
        }
        if name.contains("mmq_iu4") || name.contains("quantize_int4") || name.contains("block_i4") {
            forbidden.push(format!("{name} x{n}"));
        }
        if name.contains("qkvza") {
            qkvza_all += n;
        }
    }
    let counts_ok = want == got;
    let family_ok = match exp.family {
        WideFamily::Pm => hip_n == 0,
        WideFamily::Hip => pm_n == 0,
        WideFamily::Off => hip_n + pm_n == 0,
    };
    let silent_when_refused = !refused || launches.is_empty();
    // Every shared chunk runs one qkvza launch per DeltaNet layer on any route.
    let chunks_by_qkvza = (exp.n_dn > 0 && qkvza_all % exp.n_dn == 0).then(|| qkvza_all / exp.n_dn);
    let chunk_count_ok = refused || chunks_by_qkvza == Some(chunks.len());
    let ok = counts_ok && family_ok && forbidden.is_empty() && silent_when_refused && (wide_chunks == 0 || chunk_count_ok);
    let table: Vec<Value> = WideOp::ALL
        .iter()
        .flat_map(|&op| [4usize, 8].map(move |bt| (op, bt)))
        .filter_map(|(op, bt)| {
            let (w, g) = (want.get(&(op, bt)).copied().unwrap_or(0), got.get(&(op, bt)).copied().unwrap_or(0));
            (w != 0 || g != 0).then(|| json!({"op": op.label(), "bt": bt, "want": w, "got": g}))
        })
        .collect();
    (
        ok,
        json!({
            "ok": ok, "family_expected": exp.family.name(), "wide_chunks": wide_chunks, "chunk_rows": chunks,
            "counts_ok": counts_ok, "family_ok": family_ok, "hip_launches": hip_n, "pm_launches": pm_n,
            "forbidden_quantized_launches": forbidden, "silent_when_refused": silent_when_refused,
            "qkvza_launches": qkvza_all, "chunks_by_qkvza": chunks_by_qkvza, "chunk_count_ok": chunk_count_ok,
            "expected_vs_observed": table, "wide_symbols": wide_syms, "launch_kinds": launches.len(),
            "launches_total": launches.values().sum::<usize>(),
        }),
    )
}

/// Byte-compare `cand` against a peer run's frozen directory (both
/// directions, by array file name).
fn cmp_dir(d: &mut Diff, stage: &str, dir: &Path, cand: &Named) -> Result<()> {
    if !dir.is_dir() {
        d.differing_items.push(format!("{stage}: peer dir {} missing", dir.display()));
        return Ok(());
    }
    let mut want: BTreeSet<String> = BTreeSet::new();
    for (name, bytes) in cand {
        let file = format!("{}.bin", art_name(name));
        match fs::read(dir.join(&file)) {
            Ok(r) => d.check(&format!("{stage}/{name}"), &r, bytes),
            Err(_) => d.differing_items.push(format!("{stage}/{name}: missing in peer")),
        }
        want.insert(file);
    }
    for ent in fs::read_dir(dir)? {
        let n = ent?.file_name().to_string_lossy().into_owned();
        if n.ends_with(".bin") && !want.contains(&n) {
            d.differing_items.push(format!("{stage}/{n}: missing in candidate"));
        }
    }
    Ok(())
}

/// Rows just past a window that the verify must never write.
const TAIL_GUARD_ROWS: usize = 8;

/// KV rows `[from, from + TAIL_GUARD_ROWS)` of every full-attention layer
/// (clamped to the mapped capacity).
fn tail_guard_rows(gpu: &Gpu, config: &Qwen35Config, kv: &KvCache, from: usize) -> Result<Named> {
    gpu.hip.device_synchronize()?;
    let (kr, vr) = kv_row_bytes(kv)?;
    let mapped = kv.mapped_token_capacity()?.unwrap_or(usize::MAX);
    let rows = TAIL_GUARD_ROWS.min(mapped.saturating_sub(from));
    let mut out: Named = Vec::new();
    if rows == 0 {
        return Ok(out);
    }
    for (layer, ty) in config.layer_types.iter().enumerate() {
        if *ty == LayerType::FullAttention {
            out.push((format!("kv_tail.L{layer:02}.k"), rd(gpu, &kv.k_gpu[layer], from * kr, rows * kr)?));
            out.push((format!("kv_tail.L{layer:02}.v"), rd(gpu, &kv.v_gpu[layer], from * vr, rows * vr)?));
        }
    }
    Ok(out)
}

fn multi_requests(lanes: &mut [LaneRun], fusion: qwen35::DflashFusionCtx) -> Result<Vec<MultiChunkRequest<'_>>> {
    let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(lanes.len());
    for l in lanes.iter_mut() {
        let LaneRun { rs, lane, draft, position, .. } = l;
        let tokens: &[u32] = &draft.as_ref().ok_or("lane not drafted")?.verify_tokens;
        reqs.push(MultiChunkRequest {
            tokens,
            start_pos: *position,
            kv_cache: &mut rs.kv,
            dn_state: &mut rs.dn,
            gdn_tape: Some(&lane.df.gdn_tape),
            fusion,
            hidden_rb: Some(&mut lane.df.hidden_rb),
        });
    }
    Ok(reqs)
}

fn cb_lanes(lanes: &mut [LaneRun]) -> Result<Vec<DflashCbVerifyLane<'_>>> {
    let mut v: Vec<DflashCbVerifyLane<'_>> = Vec::with_capacity(lanes.len());
    for l in lanes.iter_mut() {
        let LaneRun { rs, lane, draft, .. } = l;
        let draft = draft.as_ref().ok_or("lane not drafted")?;
        v.push(DflashCbVerifyLane { kv_cache: &mut rs.kv, dn_state: &mut rs.dn, state: lane, draft });
    }
    Ok(v)
}

/// Per-lane singleton draft and pre-verify DeltaNet snapshot (as the singleton
/// saves it right after the draft).
fn dflash_draft_lanes(gpu: &mut Gpu, slot: &ModelSlot, lanes: &mut [LaneRun]) -> Result<()> {
    for l in lanes.iter_mut() {
        let mark = l.lane.df.draft_scratch.thlog.mark();
        let compact = l.rs.kv.compact_offset as i32;
        let drafted = dflash_lane_draft(gpu, &slot.weights, &slot.config, compact, &mut l.lane.df, l.position, l.seed, l.b)?;
        l.lane.df.target_snap.save_from(&l.rs.dn, gpu)?;
        l.draft = Some(DflashCbDraft { position: l.position, seed: l.seed, verify_tokens: drafted, max_accept: l.e - 1, thlog_mark: mark });
    }
    gpu.hip.device_synchronize()?;
    Ok(())
}

struct VerifyRun {
    /// `Some(reason)` when the shared trunk refused before launching.
    refused: Option<String>,
    /// Per-lane verify picks (`b` ids each).
    picks: Vec<Vec<u32>>,
    launches: BTreeMap<&'static str, usize>,
    /// Rows of every chunk the route plans (whole lanes, in order).
    chunks: Vec<usize>,
    /// The plan covers every lane exactly once, contiguously and in order
    /// (no lane split across chunks, so no partial-row publication).
    whole_lanes: bool,
    /// Per lane: KV rows just past the window before vs after the verify.
    tail_guards: Vec<Diff>,
}

/// ONE verify span over drafted lanes on `route` (draft already done).
fn dflash_verify_lanes(
    gpu: &mut Gpu,
    slot: &ModelSlot,
    lanes: &mut [LaneRun],
    sc: &mut G0Scratch<'_>,
    route: Route,
    fusion: qwen35::DflashFusionCtx,
) -> Result<VerifyRun> {
    let (dim, vocab) = (slot.config.dim, slot.config.vocab_size);
    let rows: Vec<usize> = lanes.iter().map(|l| l.b).collect();
    let total: usize = rows.iter().sum();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    match route {
        Route::Direct => ranges.push(0..lanes.len()),
        Route::Product => {
            let cap = multi_chunk_pack_cap(gpu, &slot.weights, &slot.config, sc.cb.max_rows());
            if pack_whole_lanes(&rows, cap, &mut ranges).is_err() {
                ranges.clear();
            }
        }
    }
    let chunks: Vec<usize> = ranges.iter().map(|r| rows[r.clone()].iter().sum()).collect();
    let whole_lanes = !ranges.is_empty()
        && ranges[0].start == 0
        && ranges.windows(2).all(|w| w[0].end == w[1].start)
        && ranges.last().is_some_and(|r| r.end == rows.len())
        && ranges.iter().all(|r| !r.is_empty());
    let mut before: Vec<Named> = Vec::with_capacity(lanes.len());
    for l in lanes.iter() {
        before.push(tail_guard_rows(gpu, &slot.config, &l.rs.kv, l.position + l.b)?);
    }
    gpu.hip.device_synchronize()?;
    let probe = LaunchProbe::start();
    let run = (|| -> Result<std::result::Result<Vec<Vec<u32>>, String>> {
        match route {
            Route::Direct => {
                let fw = {
                    let mut reqs = multi_requests(lanes, fusion)?;
                    forward_prefill_batch_multi(gpu, &slot.weights, &slot.config, &slot.scratch, sc.mc, &mut reqs, Some(&sc.vs.final_hidden))
                };
                if let Err(e) = fw {
                    return Ok(Err(e.to_string()));
                }
                let picks = dflash_cb_head_argmax(gpu, &slot.weights, &slot.config, sc.vs, total)?;
                let mut out = Vec::with_capacity(lanes.len());
                let mut row = 0usize;
                for l in lanes.iter() {
                    out.push(picks[row..row + l.b].to_vec());
                    row += l.b;
                }
                Ok(Ok(out))
            }
            Route::Product => {
                sc.cb.keep_lane_rows = true;
                let r = {
                    let mut vl = cb_lanes(lanes)?;
                    dflash_cb_verify(gpu, &slot.weights, &slot.config, &slot.scratch, &mut *sc.cb, &mut vl)
                };
                if let Err(e) = r {
                    return Ok(Err(e.to_string()));
                }
                let mut out = Vec::with_capacity(lanes.len());
                for l in lanes.iter_mut() {
                    if !std::mem::replace(&mut l.lane.verified, false) {
                        return Err("product verify left a lane without an outcome".into());
                    }
                    out.push(std::mem::take(&mut l.lane.picks));
                }
                Ok(Ok(out))
            }
        }
    })();
    let launches = probe.finish();
    match run? {
        Err(msg) => Ok(VerifyRun { refused: Some(msg), picks: Vec::new(), launches, chunks, whole_lanes, tail_guards: Vec::new() }),
        Ok(picks) => {
            gpu.hip.device_synchronize()?;
            if route == Route::Direct {
                // The product route's `keep_lane_rows` already copied these.
                let mut row = 0usize;
                for l in lanes.iter() {
                    let v = &l.lane.df.verify_scratch;
                    gpu.memcpy_dtod_at_auto(&v.final_hidden.buf, 0, &sc.vs.final_hidden.buf, row * dim * 4, l.b * dim * 4)?;
                    gpu.memcpy_dtod_at_auto(&v.logits.buf, 0, &sc.vs.logits.buf, row * vocab * 4, l.b * vocab * 4)?;
                    gpu.memcpy_dtod_at_auto(&v.argmax.buf, 0, &sc.vs.argmax.buf, row * 4, l.b * 4)?;
                    row += l.b;
                }
            }
            let mut tail_guards = Vec::with_capacity(lanes.len());
            for (i, l) in lanes.iter().enumerate() {
                let after = tail_guard_rows(gpu, &slot.config, &l.rs.kv, l.position + l.b)?;
                let mut d = Diff::default();
                for ((n, b), (_, a)) in before[i].iter().zip(&after) {
                    d.check(&format!("lane{i}/{n}"), b, a);
                }
                tail_guards.push(d);
            }
            Ok(VerifyRun { refused: None, picks, launches, chunks, whole_lanes, tail_guards })
        }
    }
}

/// Per-lane greedy accept consumer over the verified picks.
fn dflash_accept_lanes(gpu: &mut Gpu, slot: &ModelSlot, lanes: &mut [LaneRun], picks: &[Vec<u32>]) -> Result<Vec<hipfire_arch_qwen35::speculative::SpecStepResult>> {
    let mut results = Vec::with_capacity(lanes.len());
    for (l, p) in lanes.iter_mut().zip(picks) {
        let verified = DflashVerifyOutput { argmax_per_pos: p.clone(), logits_per_pos: Vec::new() };
        let mut tp = DflashTargetParts { weights: &slot.weights, config: &slot.config, kv_cache: &mut l.rs.kv, dn_state: &mut l.rs.dn, scratch: &slot.scratch };
        let draft = l.draft.as_ref().ok_or("lane not drafted")?;
        results.push(dflash_greedy_accept_commit_parts(gpu, &mut tp, &mut l.lane.df, draft, &verified)?);
    }
    gpu.hip.device_synchronize()?;
    Ok(results)
}

/// Engage `arm`, run ONE explicit verify entry of `route`, disengage.
/// Returns `(not_exercised_reason, refusal_message, launches)`.
fn graph_arm_attempt(
    gpu: &mut Gpu,
    slot: &ModelSlot,
    lanes: &mut [LaneRun],
    sc: &mut G0Scratch<'_>,
    route: Route,
    arm: GraphArm,
) -> Result<(Option<String>, Option<String>, BTreeMap<&'static str, usize>)> {
    let mut replay_open = false;
    match arm {
        GraphArm::CaptureMode => gpu.graphs.capture_mode = true,
        GraphArm::ReplayRecording => {
            if let Err(why) = gpu.replay.begin_capture() {
                return Ok((Some(format!("replay.begin_capture refused: {why}")), None, BTreeMap::new()));
            }
            replay_open = true;
            if !gpu.replay.is_recording() {
                let _ = gpu.replay.finish_capture();
                return Ok((Some("begin_capture did not make replay.is_recording() true".into()), None, BTreeMap::new()));
            }
        }
    }
    let probe = LaunchProbe::start();
    let res = (|| -> Result<std::result::Result<(), String>> {
        match route {
            Route::Direct => {
                let mut reqs = multi_requests(lanes, qwen35::DflashFusionCtx::ChainVerify)?;
                Ok(forward_prefill_batch_multi(gpu, &slot.weights, &slot.config, &slot.scratch, sc.mc, &mut reqs, Some(&sc.vs.final_hidden)).map_err(|e| e.to_string()))
            }
            Route::Product => {
                let mut vl = cb_lanes(lanes)?;
                Ok(dflash_cb_verify(gpu, &slot.weights, &slot.config, &slot.scratch, &mut *sc.cb, &mut vl).map_err(|e| e.to_string()))
            }
        }
    })();
    let launches = probe.finish();
    match arm {
        GraphArm::CaptureMode => gpu.graphs.capture_mode = false,
        GraphArm::ReplayRecording => {
            if replay_open {
                let _ = gpu.replay.finish_capture();
            }
        }
    }
    Ok((None, res?.err(), launches))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum G0Expect {
    /// Must reproduce every singleton byte.
    Exact,
    /// The shared trunk must refuse before mutating any lane.
    Refused,
    /// Diagnostic only (e.g. the Off-fusion transplant the plan abandons).
    Info,
}

impl G0Expect {
    fn name(self) -> &'static str {
        match self {
            G0Expect::Exact => "exact",
            G0Expect::Refused => "refused",
            G0Expect::Info => "info",
        }
    }
}

fn class_of(accepted: usize, b: usize) -> &'static str {
    if accepted == 0 {
        "zero"
    } else if accepted + 1 == b {
        "full"
    } else {
        "partial"
    }
}

/// Run-level options of every Gate 0 case.
struct G0Opts<'a> {
    wide: &'a WideExpect,
    /// Another run's `--artifacts` dir (`--peer`).
    peer: Option<&'a Path>,
    emit_candidates: bool,
    /// Freeze singleton refs even for refused cases (a later run's `--peer`).
    ensure_refs: bool,
    /// Byte-flip and lane-swap negative controls on wide (>= 64 row) windows.
    neg_controls: bool,
}

/// One Gate 0 case: lanes `(fixture, E)` replayed together on `route`. Returns
/// the case receipt and whether it met its expectation. `graph_arms` engage
/// the graph-refusal controls between the drafts and the real window.
#[allow(clippy::too_many_arguments)]
fn g0_case(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    d: &mut Box<dyn Speculator>,
    cfg: &G0Cfg,
    fx: &[Fixture],
    refs: &mut G0Refs,
    sc: &mut G0Scratch<'_>,
    opts: &G0Opts<'_>,
    name: &str,
    specs: &[(usize, usize)],
    route: Route,
    fusion: qwen35::DflashFusionCtx,
    expect: G0Expect,
    graph_arms: &[GraphArm],
) -> Result<(Value, bool)> {
    let t0 = std::time::Instant::now();
    for &(fi, e) in specs {
        ensure_pre(gpu, slot, d, cfg, fx, refs, fi)?;
        if expect != G0Expect::Refused || opts.ensure_refs {
            ensure_post(gpu, slot, d, cfg, fx, refs, fi, e)?;
        }
    }
    let mut lanes: Vec<LaneRun> = Vec::with_capacity(specs.len());
    let mut open_err = None;
    for (i, &(fi, e)) in specs.iter().enumerate() {
        match g0_lane_open(gpu, slot, d, cfg, fx, refs, fi, e, 70_000 + i as u64) {
            Ok(l) => lanes.push(l),
            Err(err) => {
                open_err = Some(err);
                break;
            }
        }
    }
    if let Some(err) = open_err {
        free_lanes(gpu, lanes)?;
        return Err(err);
    }
    let rows_total: usize = lanes.iter().map(|l| l.b).sum();
    let mut refusal_unchanged: Option<Diff> = None;
    struct Body {
        verify: VerifyRun,
        results: Vec<hipfire_arch_qwen35::speculative::SpecStepResult>,
        ms: f64,
        graph: Vec<Value>,
        graph_ok: bool,
    }
    let body = (|| -> Result<Body> {
        // State a refused trunk must leave untouched (target + ring/draft).
        let before: Vec<Named> = if expect == G0Expect::Refused {
            let mut v = Vec::new();
            for l in &lanes {
                v.push(refusal_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, l.position)?);
            }
            v
        } else {
            Vec::new()
        };
        dflash_draft_lanes(gpu, slot, &mut lanes)?;
        // Graph-refusal controls: each explicit verify entry must refuse before
        // any launch or state write while capture/recording is engaged.
        let mut graph: Vec<Value> = Vec::new();
        let mut graph_ok = true;
        for &arm in graph_arms {
            for r in [Route::Direct, Route::Product] {
                if r == Route::Direct && rows_total > opts.wide.route_cap() {
                    continue;
                }
                let mut pre: Vec<Named> = Vec::new();
                for l in &lanes {
                    pre.push(refusal_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, l.position)?);
                }
                let (not_exercised, refusal, launches) = graph_arm_attempt(gpu, slot, &mut lanes, sc, r, arm)?;
                let mut dd = Diff::default();
                for (i, l) in lanes.iter().enumerate() {
                    let after = refusal_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, l.position)?;
                    for ((n, b), (_, a)) in pre[i].iter().zip(&after) {
                        dd.check(&format!("lane{i}/{n}"), b, a);
                    }
                }
                let exercised = not_exercised.is_none();
                let refused_for_graph = refusal.as_deref().is_some_and(|m| m.contains("graph capture"));
                let ok = !exercised || (refused_for_graph && launches.is_empty() && dd.differing_items.is_empty() && dd.compared_items > 0);
                // The capture-mode arm is always exercisable; only the replay arm may be skipped.
                let ok = ok && (exercised || arm == GraphArm::ReplayRecording);
                graph_ok &= ok;
                eprintln!(
                    "gate0 {name} graph-refusal arm={} route={} exercised={exercised} refused={} launches={} state_unchanged={} -> {}",
                    arm.name(), r.name(), refusal.is_some(), launches.values().sum::<usize>(), dd.differing_items.is_empty(),
                    if ok { "OK" } else { "FAIL" }
                );
                graph.push(json!({
                    "arm": arm.name(), "route": r.name(), "exercised": exercised, "not_exercised_reason": not_exercised,
                    "refusal": refusal, "refused_as_graph": refused_for_graph, "launches": launches.values().sum::<usize>(),
                    "state_unchanged": dd.json(), "ok": ok,
                }));
            }
        }
        let t1 = std::time::Instant::now();
        let verify = dflash_verify_lanes(gpu, slot, &mut lanes, sc, route, fusion)?;
        let mut results = Vec::new();
        if verify.refused.is_none() {
            results = dflash_accept_lanes(gpu, slot, &mut lanes, &verify.picks)?;
        }
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        if expect == G0Expect::Refused && verify.refused.is_some() {
            let mut dd = Diff::default();
            for (i, l) in lanes.iter().enumerate() {
                let after = refusal_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, l.position)?;
                for ((na, ba), (_, bb)) in before[i].iter().zip(&after) {
                    dd.check(&format!("lane{i}/{na}"), ba, bb);
                }
            }
            refusal_unchanged = Some(dd);
        }
        Ok(Body { verify, results, ms, graph, graph_ok })
    })();
    let mut lane_json = Vec::new();
    let mut exact = true;
    let mut note = None;
    let mut refused = None;
    let mut ms = 0.0;
    let mut err_text = None;
    let mut side_ok = true;
    let mut wide_json = Value::Null;
    let mut wide_ok = true;
    let mut graph_json: Vec<Value> = Vec::new();
    let mut route_extra = serde_json::Map::new();
    match body {
        Ok(out) => {
            refused = out.verify.refused.clone();
            ms = out.ms;
            graph_json = out.graph.clone();
            side_ok &= out.graph_ok;
            let (w_ok, w_json) = assert_wide(opts.wide, &out.verify.chunks, &out.verify.launches, refused.is_some());
            wide_ok = w_ok;
            wide_json = w_json;
            route_extra.insert("whole_lanes_plan".into(), json!(out.verify.whole_lanes));
            side_ok &= out.verify.whole_lanes || refused.is_some();
            if refused.is_none() {
                let keys: Vec<(usize, usize)> = lanes.iter().map(|l| (l.fi, l.e)).collect();
                for (i, l) in lanes.iter_mut().enumerate() {
                    let r = &out.results[i];
                    let pr = &refs.post[&(l.fi, l.e)];
                    let new_pos = l.position + r.accepted + 1;
                    let mut ref_committed = vec![pr.seed];
                    ref_committed.extend_from_slice(&pr.emit);
                    l.post_diff.ids("committed", &ref_committed, &r.committed);
                    l.post_diff.ids(
                        "window",
                        &[pr.accepted as u32, pr.next_seed, pr.b as u32, pr.new_pos as u32, pr.position as u32],
                        &[r.accepted as u32, r.bonus_token, l.b as u32, new_pos as u32, l.position as u32],
                    );
                    let guard = &out.verify.tail_guards[i];
                    let guard_ok = guard.differing_items.is_empty();
                    side_ok &= guard_ok;
                    let post = (|| -> Result<Named> {
                        let mut v = full_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, new_pos)?;
                        v.extend(window_arrays(gpu, &slot.config, &l.rs.kv, &l.lane.df, l.position, l.b)?);
                        Ok(v)
                    })();
                    let mut peer_json = Value::Null;
                    let mut neg_json = Value::Null;
                    match post {
                        Ok(post) => {
                            if let Err(e) = cmp_frozen(&mut l.post_diff, "post", &pr.frozen, &post) {
                                err_text = Some(e.to_string());
                            }
                            if opts.emit_candidates {
                                if let Err(e) = freeze(&cfg.dir.join(format!("cand_{name}_lane{i}")), &post) {
                                    err_text = Some(e.to_string());
                                }
                            }
                            if let Some(peer) = opts.peer {
                                // The peer's isolated singleton arrays of this lane, and (when it ran this
                                // case as exact) its candidate arrays: HIP/PM and wide-off vs wide-on.
                                let pdir = peer.join("dflash_gate0");
                                let mut dref = Diff::default();
                                let refdir = pdir.join(format!("ref_post_{}_e{}", fx[l.fi].name(), l.e));
                                let mut dcand: Option<Diff> = None;
                                let cdir = pdir.join(format!("cand_{name}_lane{i}"));
                                let res = (|| -> Result<()> {
                                    cmp_dir(&mut dref, "peer_ref", &refdir, &post)?;
                                    if cdir.is_dir() {
                                        let mut dc = Diff::default();
                                        cmp_dir(&mut dc, "peer_cand", &cdir, &post)?;
                                        dcand = Some(dc);
                                    }
                                    Ok(())
                                })();
                                if let Err(e) = res {
                                    err_text = Some(e.to_string());
                                }
                                let ok = dref.differing_items.is_empty() && dref.compared_items > 0 && dcand.as_ref().map_or(true, |c| c.differing_items.is_empty());
                                side_ok &= ok;
                                peer_json = json!({
                                    "ok": ok, "peer_ref": dref.json(), "peer_cand": dcand.as_ref().map(|c| c.json()),
                                    "peer_cand_present": dcand.is_some(),
                                });
                                for it in dref.differing_items.iter().chain(dcand.iter().flat_map(|c| c.differing_items.iter())).take(20) {
                                    println!("gate0 {name} lane{i} PEER-DIFF {it}");
                                }
                            }
                            if opts.neg_controls && rows_total > MULTI_CHUNK_PRODUCT_MAX_ROWS && i == 0 {
                                let mut neg = serde_json::Map::new();
                                let mut neg_ok = true;
                                let ref_logits = |key: &(usize, usize)| -> Option<PathBuf> {
                                    refs.post.get(key).and_then(|p| p.frozen.iter().find(|(n, _)| n == "verify.logits").map(|(_, p)| p.clone()))
                                };
                                match (post.iter().find(|(n, _)| n == "verify.logits"), ref_logits(&keys[i])) {
                                    (Some((_, cand)), Some(rp)) => {
                                        let reference = fs::read(&rp).unwrap_or_else(|e| {
                                            err_text = Some(e.to_string());
                                            Vec::new()
                                        });
                                        let mut flipped = cand.clone();
                                        if let Some(b) = flipped.first_mut() {
                                            *b ^= 0x01;
                                        }
                                        let mut dd = Diff::default();
                                        dd.check("neg_byte_flip/verify.logits", &reference, &flipped);
                                        let detected = !dd.differing_items.is_empty();
                                        neg.insert("byte_flip_detected".into(), json!(detected));
                                        neg_ok &= detected;
                                        if let Some(other) = keys.iter().find(|k| **k != keys[i]) {
                                            if let Some(op) = ref_logits(other) {
                                                let mut ds = Diff::default();
                                                let other_bytes = fs::read(&op).unwrap_or_else(|e| {
                                                    err_text = Some(e.to_string());
                                                    Vec::new()
                                                });
                                                ds.check("neg_lane_swap/verify.logits", &other_bytes, cand);
                                                let detected = !ds.differing_items.is_empty();
                                                neg.insert("lane_swap_detected".into(), json!(detected));
                                                neg_ok &= detected;
                                            }
                                        }
                                    }
                                    _ => {
                                        neg.insert("skipped".into(), json!("verify.logits reference/candidate missing"));
                                        neg_ok = false;
                                    }
                                }
                                neg.insert("ok".into(), json!(neg_ok));
                                side_ok &= neg_ok;
                                neg_json = Value::Object(neg);
                            }
                        }
                        Err(e) => err_text = Some(e.to_string()),
                    }
                    lane_json.push(json!({
                        "lane": i, "fixture": fx[l.fi].name(), "prefix": fx[l.fi].prefix, "e": l.e, "b": l.b,
                        "position": l.position, "accepted": r.accepted, "class": class_of(r.accepted, l.b),
                        "snapshot_rows": l.snapshot_rows, "ref_proposed": pr.proposed, "ref_step_ms": pr.step_ms, "singleton_verify_route_hint": pr.route_hint,
                        "pre": l.pre_diff.json(), "post": l.post_diff.json(),
                        "tail_guard": {"ok": guard_ok, "compared_items": guard.compared_items, "differences": guard.differing_items.iter().take(8).collect::<Vec<_>>()},
                        "peer": peer_json, "neg_controls": neg_json,
                    }));
                    println!(
                        "gate0 {name} lane{i} {} E={} B={} accepted={} class={} pre_diffs={} post_diffs={} tail_guard_diffs={} (route {})",
                        fx[l.fi].name(), l.e, l.b, r.accepted, class_of(r.accepted, l.b),
                        l.pre_diff.differing_items.len(), l.post_diff.differing_items.len(), guard.differing_items.len(), pr.route_hint
                    );
                    for it in l.pre_diff.differing_items.iter().chain(l.post_diff.differing_items.iter()).take(60) {
                        println!("gate0 {name} lane{i} DIFF {it}");
                    }
                    exact &= l.pre_diff.exact() && l.post_diff.exact();
                }
            } else {
                for (i, l) in lanes.iter().enumerate() {
                    lane_json.push(json!({
                        "lane": i, "fixture": fx[l.fi].name(), "e": l.e, "b": l.b, "pre": l.pre_diff.json(),
                    }));
                    exact &= l.pre_diff.exact();
                }
            }
        }
        Err(e) => {
            err_text = Some(e.to_string());
            exact = false;
        }
    }
    let unchanged = refusal_unchanged.as_ref().map(|d| d.exact());
    let ok = err_text.is_none()
        && match expect {
            G0Expect::Exact => refused.is_none() && exact && wide_ok && side_ok,
            G0Expect::Refused => refused.is_some() && unchanged == Some(true) && lanes.iter().all(|l| l.pre_diff.exact()) && wide_ok && side_ok,
            G0Expect::Info => true,
        };
    if expect == G0Expect::Info {
        note = Some("diagnostic only: Off-fusion transplant is expected to differ on gfx1201 (plan Gate 0); result does not gate");
    }
    free_lanes(gpu, lanes)?;
    eprintln!(
        "gate0 case {name}: route={} rows={rows_total} fusion={fusion:?} expect={} refused={} exact={exact} wide_ok={wide_ok} side_ok={side_ok} -> {}",
        route.name(), expect.name(), refused.is_some(), if ok { "OK" } else { "FAIL" }
    );
    let j = json!({
        "case": name,
        "route": route.name(),
        "rows_total": rows_total,
        "fusion": format!("{fusion:?}"),
        "expect": expect.name(),
        "refused": refused,
        "refusal_state_unchanged": refusal_unchanged.as_ref().map(|d| d.json()),
        "exact": exact,
        "ok": ok,
        "wide": wide_json,
        "side_checks_ok": side_ok,
        "graph_refusal": graph_json,
        "route_detail": route_extra,
        "note": note,
        "error": err_text,
        "shared_window_ms": ms,
        "case_wall_s": t0.elapsed().as_secs_f64(),
        "lanes": lane_json,
    });
    Ok((j, ok))
}

/// One Gate 0 case of the probe.
struct G0CaseSpec {
    name: String,
    specs: Vec<(usize, usize)>,
    route: Route,
    fusion: qwen35::DflashFusionCtx,
    expect: G0Expect,
    arms: Vec<GraphArm>,
    /// Runs after the draft cases (the replay arm leaves the controller in
    /// its captured state).
    late: bool,
}

/// Run `list`; returns the error text of the first case that errored.
#[allow(clippy::too_many_arguments)]
fn run_case_list(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    d: &mut Box<dyn Speculator>,
    cfg: &G0Cfg,
    fx: &[Fixture],
    refs: &mut G0Refs,
    sc: &mut G0Scratch<'_>,
    opts: &G0Opts<'_>,
    list: &[&G0CaseSpec],
    rows: &mut Vec<Value>,
    pass: &mut bool,
) -> Option<String> {
    for c in list {
        match g0_case(gpu, slot, d, cfg, fx, refs, sc, opts, &c.name, &c.specs, c.route, c.fusion, c.expect, &c.arms) {
            Ok((j, ok)) => {
                *pass &= ok;
                rows.push(j);
            }
            Err(e) => {
                eprintln!("gate0 case {}: ERROR {e}", c.name);
                rows.push(json!({"case": c.name, "ok": false, "error": e.to_string()}));
                *pass = false;
                return Some(e.to_string());
            }
        }
    }
    None
}

/// A lone 64-row request is never admitted (its singleton would take the
/// quantized >= 64-row route): `forward_prefill_batch_multi` must refuse it
/// before any launch or state write, on every flag setting.
#[allow(clippy::too_many_arguments)]
fn lone_row_control(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    d: &mut Box<dyn Speculator>,
    cfg: &G0Cfg,
    fx: &[Fixture],
    refs: &mut G0Refs,
    sc: &mut G0Scratch<'_>,
) -> Result<(Value, bool)> {
    ensure_pre(gpu, slot, d, cfg, fx, refs, 0)?;
    let mut l = g0_lane_open(gpu, slot, d, cfg, fx, refs, 0, 64, 90_000)?;
    let res = (|| -> Result<(bool, Value)> {
        let before = refusal_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, l.position)?;
        let tokens = vec![l.seed; 64];
        let probe = LaunchProbe::start();
        let fw = {
            let LaneRun { rs, lane, position, .. } = &mut l;
            let mut reqs = vec![MultiChunkRequest {
                tokens: &tokens,
                start_pos: *position,
                kv_cache: &mut rs.kv,
                dn_state: &mut rs.dn,
                gdn_tape: Some(&lane.df.gdn_tape),
                fusion: qwen35::DflashFusionCtx::ChainVerify,
                hidden_rb: Some(&mut lane.df.hidden_rb),
            }];
            forward_prefill_batch_multi(gpu, &slot.weights, &slot.config, &slot.scratch, sc.mc, &mut reqs, Some(&sc.vs.final_hidden))
        };
        let launches = probe.finish();
        let after = refusal_state(gpu, &slot.config, &l.rs.kv, &l.rs.dn, &l.lane, l.position)?;
        let mut dd = Diff::default();
        for ((n, b), (_, a)) in before.iter().zip(&after) {
            dd.check(n, b, a);
        }
        let msg = fw.as_ref().err().map(|e| e.to_string());
        let ok = msg.is_some() && launches.is_empty() && dd.differing_items.is_empty() && dd.compared_items > 0 && l.pre_diff.exact();
        eprintln!("gate0 ctl_lone_64_row_request: refused={} launches={} state_unchanged={} -> {}", msg.is_some(), launches.len(), dd.differing_items.is_empty(), if ok { "OK" } else { "FAIL" });
        Ok((ok, json!({
            "case": "ctl_lone_64_row_request", "expect": "refused", "refusal": msg, "launch_kinds": launches.len(),
            "state_unchanged": dd.json(), "pre": l.pre_diff.json(), "ok": ok,
        })))
    })();
    free_lanes(gpu, vec![l])?;
    let (ok, j) = res?;
    Ok((j, ok))
}

/// `HIPFIRE_*` variables recorded in every receipt (both wide-route flags
/// included: `HIPFIRE_CB_VERIFY_CHUNK128` default off, `HIPFIRE_CB_VERIFY_PM`
/// default on inside the enabled route only; the singleton-arm switches that
/// refuse wide admission; the `HIPFIRE_PREFILL_CHUNK_ROWS` and
/// `HIPFIRE_VERIFY_GRAPH` pins).
fn receipt_env() -> serde_json::Map<String, Value> {
    const KEYS: [&str; 19] = [
        "HIPFIRE_CB_VERIFY_CHUNK128", "HIPFIRE_CB_VERIFY_PM", "HIPFIRE_WMMA_BATCH_TILES",
        "HIPFIRE_FP16", "HIPFIRE_LM_HEAD_WMMA", "HIPFIRE_HFQ4G256_LDSSTAGE", "HIPFIRE_PREFILL_CHUNK_ROWS",
        "HIPFIRE_VERIFY_GRAPH", "HIPFIRE_GRAPH", "HIPFIRE_CB_SEG_TWINS", "HIPFIRE_DFLASH_WINDOW", "HIPFIRE_DFLASH_CTX_CAP",
        "HIPFIRE_DFLASH_ADAPTIVE_B", "HIPFIRE_DFLASH_CKPT_RESUME", "HIPFIRE_DFLASH_Q8_LMHEAD_WMMA", "HIPFIRE_SPEC_PHASES",
        "HIPFIRE_DN_STATE_EF", "HIPFIRE_DN_SNAPSHOT_FLIP", "HIPFIRE_CB_DFLASH_DRAFT_BATCH",
    ];
    KEYS.iter().map(|k| (k.to_string(), json!(std::env::var(k).ok()))).collect()
}

/// Frozen singleton draft of one fixture's pre-window state: the drafted
/// `[seed, candidates..]` tokens and the lane's draft/ring state right after
/// the isolated `dflash_lane_draft` (K/V rings, projection cache, thlog).
struct DraftRef {
    frozen: Frozen,
    tokens: Vec<u32>,
}

fn ensure_draft_ref(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Box<dyn Speculator>, cfg: &G0Cfg, fx: &[Fixture], refs: &mut G0Refs, fi: usize) -> Result<()> {
    if refs.draft.contains_key(&fi) {
        return Ok(());
    }
    eprintln!("draftbatch trace: ref_draft walk {}", fx[fi].name());
    let w = dflash_walk(gpu, slot, d, &fx[fi], cfg.windows)?;
    let (mut lane, _rows) = take_lane(gpu, slot, d, w.position)?;
    let compact = slot.kv_cache.compact_offset as i32;
    let r = (|| -> Result<(Vec<u32>, Named)> {
        let tokens = dflash_lane_draft(gpu, &slot.weights, &slot.config, compact, &mut lane.df, w.position, w.seed, cfg.block)?;
        gpu.hip.device_synchronize()?;
        let ck: Vec<usize> = lane.checkpoints.iter().map(|(p, _)| *p).collect();
        Ok((tokens, df_arrays(gpu, &lane.df, &ck)?))
    })();
    lane.free_gpu(gpu);
    let (tokens, set) = r?;
    let frozen = freeze(&cfg.dir.join(format!("ref_draft_{}", fx[fi].name())), &set)?;
    refs.draft.insert(fi, DraftRef { frozen, tokens });
    Ok(())
}

/// One batched-draft case: lanes `specs` (fixtures) are opened at their
/// pre-window state (each compared with the frozen singleton pre-state), then
/// drafted together by `dflash_cb_draft` (`min_lanes`: fewest lanes of a chunk
/// that take the batched forward; `batched=false` is the per-lane singleton
/// call). Every lane's tokens and post-draft draft state must equal the
/// isolated singleton draft byte for byte. Returns `(receipt, ok, draft_ms)`.
/// The cases share the probe's one `DflashCbScratch` `cb`, as the VMM DFlash
/// engine reuses its single scratch every step; only `draft_min_lanes` is set
/// per case and restored afterwards. (A fresh max-rows scratch per case ran the
/// device out of memory on the wide route at probe windows >= 1.)
#[allow(clippy::too_many_arguments)]
fn draft_case(
    gpu: &mut Gpu,
    slot: &mut ModelSlot,
    d: &mut Box<dyn Speculator>,
    cfg: &G0Cfg,
    fx: &[Fixture],
    refs: &mut G0Refs,
    cb: &mut DflashCbScratch,
    name: &str,
    specs: &[usize],
    min_lanes: usize,
    batched: bool,
    expect_batched_lanes: usize,
) -> Result<(Value, bool, f64)> {
    let t_case = std::time::Instant::now();
    for &fi in specs {
        ensure_pre(gpu, slot, d, cfg, fx, refs, fi)?;
        ensure_draft_ref(gpu, slot, d, cfg, fx, refs, fi)?;
    }
    let mut lanes: Vec<LaneRun> = Vec::with_capacity(specs.len());
    let mut open_err = None;
    for (i, &fi) in specs.iter().enumerate() {
        match g0_lane_open(gpu, slot, d, cfg, fx, refs, fi, 64, 80_000 + i as u64) {
            Ok(l) => lanes.push(l),
            Err(err) => {
                open_err = Some(err);
                break;
            }
        }
    }
    if let Some(err) = open_err {
        free_lanes(gpu, lanes)?;
        return Err(err);
    }
    let saved_min_lanes = cb.draft_min_lanes;
    cb.draft_min_lanes = min_lanes;
    cb.draft_stats = (0, 0);
    let body = (|| -> Result<(Vec<Vec<u32>>, f64)> {
        gpu.hip.device_synchronize()?;
        let t0 = std::time::Instant::now();
        let toks = {
            let mut dl: Vec<DflashCbDraftLane<'_>> = lanes
                .iter_mut()
                .map(|l| DflashCbDraftLane { state: &mut l.lane, position: l.position, seed: l.seed, b: l.b, compact_offset: l.rs.kv.compact_offset as i32 })
                .collect();
            dflash_cb_draft(gpu, &slot.weights, &slot.config, &mut *cb, &mut dl, batched)?
        };
        gpu.hip.device_synchronize()?;
        Ok((toks, t0.elapsed().as_secs_f64() * 1e3))
    })();
    let mut lane_json = Vec::new();
    let mut exact = true;
    let mut err_text = None;
    let mut ms = 0.0;
    let stats = cb.draft_stats;
    match body {
        Ok((toks, t)) => {
            ms = t;
            for (i, l) in lanes.iter_mut().enumerate() {
                let dr = &refs.draft[&l.fi];
                l.post_diff.ids("draft.tokens", &dr.tokens, &toks[i]);
                let post = (|| -> Result<Named> {
                    let ck: Vec<usize> = l.lane.checkpoints.iter().map(|(p, _)| *p).collect();
                    df_arrays(gpu, &l.lane.df, &ck)
                })();
                match post {
                    Ok(post) => {
                        if let Err(e) = cmp_frozen(&mut l.post_diff, "post_draft", &dr.frozen, &post) {
                            err_text = Some(e.to_string());
                        }
                    }
                    Err(e) => err_text = Some(e.to_string()),
                }
                lane_json.push(json!({
                    "lane": i, "fixture": fx[l.fi].name(), "prefix": fx[l.fi].prefix, "b": l.b, "position": l.position,
                    "tokens": toks[i], "pre": l.pre_diff.json(), "post_draft": l.post_diff.json(),
                }));
                println!(
                    "draftbatch {name} lane{i} {} B={} pre_diffs={} draft_diffs={}",
                    fx[l.fi].name(), l.b, l.pre_diff.differing_items.len(), l.post_diff.differing_items.len()
                );
                for it in l.pre_diff.differing_items.iter().chain(l.post_diff.differing_items.iter()).take(40) {
                    println!("draftbatch {name} lane{i} DIFF {it}");
                }
                exact &= l.pre_diff.exact() && l.post_diff.exact();
            }
        }
        Err(e) => {
            err_text = Some(e.to_string());
            exact = false;
        }
    }
    let went_batched = stats.0 == expect_batched_lanes;
    let ok = err_text.is_none() && exact && went_batched;
    cb.draft_min_lanes = saved_min_lanes;
    free_lanes(gpu, lanes)?;
    eprintln!(
        "draftbatch case {name}: lanes={} batched_lanes={} chunks={} (expected batched lanes {expect_batched_lanes}) draft_ms={ms:.2} exact={exact} -> {}",
        specs.len(), stats.0, stats.1, if ok { "OK" } else { "FAIL" }
    );
    let j = json!({
        "case": name, "lanes": specs.len(), "min_lanes": min_lanes, "batched": batched,
        "batched_lanes": stats.0, "batched_chunks": stats.1, "expected_batched_lanes": expect_batched_lanes,
        "exact": exact, "ok": ok, "error": err_text, "draft_ms": ms, "case_wall_s": t_case.elapsed().as_secs_f64(),
        "per_lane": lane_json,
    });
    Ok((j, ok, ms))
}

/// Gate-0-style cases of the batched DFlash draft: 1..4 lanes (batched
/// forward proven down to one lane with `min_lanes = 1`), the production
/// chunking (`min_lanes = 2`: four lanes are a three-lane chunk plus a
/// singleton lane), and the timing of eight lanes per step, serial vs
/// batched (draft ms only; separate fresh lane sets, best of two).
fn draft_batch_cases(gpu: &mut Gpu, slot: &mut ModelSlot, d: &mut Box<dyn Speculator>, cfg: &G0Cfg, fx: &[Fixture], refs: &mut G0Refs, cb: &mut DflashCbScratch) -> Result<(Vec<Value>, bool)> {
    let n = fx.len();
    let fxl = |k: usize| -> Vec<usize> { (0..k).map(|i| i % n).collect() };
    // (name, lanes, min_lanes, expected batched lanes)
    let cases: Vec<(&str, usize, usize, usize)> = vec![
        ("draft_n1_batched16", 1, 1, 1),
        ("draft_n2", 2, 1, 2),
        ("draft_n3", 3, 1, 3),
        ("draft_n4_min1", 4, 1, 4),
        ("draft_n4_prod", 4, 2, 3),
    ];
    let mut rows = Vec::new();
    let mut pass = true;
    for (name, k, min_lanes, expect) in cases {
        match draft_case(gpu, slot, d, cfg, fx, refs, cb, name, &fxl(k), min_lanes, true, expect) {
            Ok((j, ok, _)) => {
                pass &= ok;
                rows.push(j);
            }
            Err(e) => {
                eprintln!("draftbatch case {name}: ERROR {e}");
                rows.push(json!({"case": name, "ok": false, "error": e.to_string()}));
                return Ok((rows, false));
            }
        }
    }
    let mut serial = Vec::new();
    let mut batched = Vec::new();
    for rep in 0..2 {
        for (is_batched, sink) in [(false, &mut serial), (true, &mut batched)] {
            let name = format!("draft_n8_{}_rep{rep}", if is_batched { "batched" } else { "serial" });
            match draft_case(gpu, slot, d, cfg, fx, refs, cb, &name, &fxl(8), 2, is_batched, if is_batched { 8 } else { 0 }) {
                Ok((j, ok, ms)) => {
                    pass &= ok;
                    sink.push(ms);
                    rows.push(j);
                }
                Err(e) => {
                    eprintln!("draftbatch case {name}: ERROR {e}");
                    rows.push(json!({"case": name, "ok": false, "error": e.to_string()}));
                    return Ok((rows, false));
                }
            }
        }
    }
    let best = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "draftbatch timing 8 lanes/step: serial {:.2} ms (runs {:?}) batched {:.2} ms (runs {:?})",
        best(&serial), serial, best(&batched), batched
    );
    rows.push(json!({"case": "draft_n8_timing", "serial_ms": serial, "batched_ms": batched, "serial_best_ms": best(&serial), "batched_best_ms": best(&batched)}));
    Ok((rows, pass))
}

fn file_sha(p: &Path) -> Result<(u64, String)> {
    use std::io::Read;
    let mut f = fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let r = f.read(&mut buf)?;
        if r == 0 {
            break;
        }
        h.update(&buf[..r]);
        n += r as u64;
    }
    Ok((n, h.finalize().iter().map(|b| format!("{b:02x}")).collect()))
}

/// Gate 0 phase. Consumes the bundle (as a `ModelSlot`) and returns it.
fn dflash_probe_phase(ctx: Ctx, args: &Args, fx: &[Fixture], ack: &Value) -> Result<(Ctx, Value, bool)> {
    let draft_path = args.dflash_draft.as_ref().ok_or("--dflash-draft required")?;
    let windows = args.dflash_probe_window.unwrap_or(0);
    let Ctx { mut gpu, b } = ctx;
    let mut slot = ModelSlot::from_bundle(b, Path::new(&args.model)).map_err(|(_, e)| format!("ModelSlot::from_bundle: {e}"))?;
    if slot.config.num_experts > 0 {
        return Err("DFlash Gate 0 covers dense targets only (MoE uses the draft FFN graph route)".into());
    }
    if let Some(pbs) = slot.scratch.widened_prefill_batch.borrow_mut().take() {
        pbs.free_gpu(&mut gpu)?;
    }
    let ctx_cap = ack["max_seq_bound"].as_u64().ok_or("loaded ack has no max_seq_bound")? as usize;
    let kv_q8 = {
        let kv = &slot.kv_cache;
        kv.quant_q8 && !kv.quant_fwht && !kv.quant_asym2 && !kv.quant_asym3 && !kv.quant_asym4 && matches!(kv.v_mode, hipfire_runtime::llama::VMode::Q8)
    };
    let draft_str = draft_path.to_str().ok_or("--dflash-draft is not UTF-8")?;
    let df = load_dflash_state(
        draft_str,
        ctx_cap,
        &slot.config,
        &slot.dn_state,
        &mut gpu,
        None,
        None,
        false,
        &slot.weights,
        kv_q8,
        slot.scratch.flash_partials.numel(),
        true,
        false,
    )?;
    eprintln!("gate0: DFlash draft loaded: layers={} hidden={} block={}", df.draft_config.n_layers, df.draft_config.hidden, df.draft_config.block_size);
    let mut d: Box<dyn Speculator> = build_dflash_speculator(df, true, false);
    let block = d.block_size();
    let (draft_bytes, draft_sha) = file_sha(draft_path)?;
    let cfg = G0Cfg { windows, block, dir: args.artifacts.join("dflash_gate0") };
    fs::create_dir_all(&cfg.dir)?;
    let (dim, vocab) = (slot.config.dim, slot.config.vocab_size);
    let vs = VerifyScratch::new(&mut gpu, MULTI_CHUNK_MAX_ROWS, dim, vocab, dim.next_power_of_two())?;
    let mc = MultiChunkScratch::new(&mut gpu, &slot.config, MULTI_CHUNK_MAX_ROWS)?;
    let mut cb = DflashCbScratch::new(&mut gpu, &slot.weights, &slot.config, MULTI_CHUNK_MAX_ROWS)?;
    cb.keep_lane_rows = true;
    let wide = WideExpect::new(&gpu, &slot);
    eprintln!(
        "gate0: wide route family={} cap={} admitted={} (mc scratch rows {}, cb scratch rows {})",
        wide.family.name(), wide.cap, wide.admitted, mc.max_rows(), cb.max_rows()
    );

    use qwen35::DflashFusionCtx::{ChainVerify, Off};
    use G0Expect::{Exact, Info, Refused};
    let n = fx.len();
    let fxi = |v: &[(usize, usize)]| -> Vec<(usize, usize)> { v.iter().map(|&(f, e)| (f % n, e)).collect() };
    let plain = |name: &str, specs: Vec<(usize, usize)>, fusion: qwen35::DflashFusionCtx, expect: G0Expect| G0CaseSpec {
        name: name.into(), specs, route: Route::Direct, fusion, expect, arms: Vec::new(), late: false,
    };
    // E >= 16 is a full block; E < 16 is a budget tail with B = max(E, 2).
    let mut cases: Vec<G0CaseSpec> = vec![
        plain("c1_16", fxi(&[(0, 64)]), ChainVerify, Exact),
        plain("c2_16x2", fxi(&[(0, 64), (1, 64)]), ChainVerify, Exact),
        plain("c3_16x3", fxi(&[(0, 64), (1, 64), (2, 64)]), ChainVerify, Exact),
        plain("c4_63_16x3_15", fxi(&[(0, 64), (1, 64), (2, 64), (3, 15)]), ChainVerify, Exact),
        plain("tail_47_16x2_15", fxi(&[(0, 64), (1, 64), (2, 15)]), ChainVerify, Exact),
        plain("tail_budget_E1_E2_E3", fxi(&[(0, 1), (1, 2), (2, 3)]), ChainVerify, Exact),
        plain("tail_E17_E15_E9", fxi(&[(0, 17), (1, 15), (2, 9)]), ChainVerify, Exact),
    ];
    // Over the effective cap (64 rows on the 63-row route; 129 on the wide
    // route): refused before any state write.
    let over = wide.route_cap() + 1;
    cases.push(plain(&format!("ctl_illegal_{over}"), lane_specs(&lane_blocks(over, block)?, block, n), ChainVerify, Refused));
    cases.push(plain("ctl_off_transplant_16x2", fxi(&[(0, 64), (1, 64)]), Off, Info));
    // Aggregate-boundary fixtures: whole short per-request blocks, one shared
    // trunk call (`direct`: exact iff within the effective cap) and the
    // production packing (`product`: always exact, chunked by the cap).
    for &total in &args.aggregates {
        let specs = lane_specs(&lane_blocks(total, block)?, block, n);
        let direct = if total <= wide.route_cap() { Exact } else { Refused };
        cases.push(G0CaseSpec { name: format!("agg{total:03}_direct"), specs: specs.clone(), route: Route::Direct, fusion: ChainVerify, expect: direct, arms: Vec::new(), late: false });
        cases.push(G0CaseSpec { name: format!("agg{total:03}_product"), specs, route: Route::Product, fusion: ChainVerify, expect: Exact, arms: Vec::new(), late: false });
    }
    // Graph-refusal-before-mutation: the largest admitted aggregate with the
    // capture / recording arms engaged on both explicit entries, then the
    // real window on the same lanes (a refusal must not have poisoned them).
    let gtotal = if wide.route_cap() > MULTI_CHUNK_PRODUCT_MAX_ROWS { wide.route_cap() } else { MULTI_CHUNK_PRODUCT_MAX_ROWS };
    cases.push(G0CaseSpec {
        name: format!("ctl_graph_refusal_{gtotal}"),
        specs: lane_specs(&lane_blocks(gtotal, block)?, block, n),
        route: Route::Product,
        fusion: ChainVerify,
        expect: Exact,
        arms: vec![GraphArm::CaptureMode, GraphArm::ReplayRecording],
        late: true,
    });
    let mut refs = G0Refs::default();
    let mut rows = Vec::new();
    let mut pass = true;
    // CB_ORACLE_DRAFT_ONLY=1 skips the shared-trunk Gate 0 cases and runs the batched-draft cases only.
    let draft_only = std::env::var_os("CB_ORACLE_DRAFT_ONLY").is_some();
    let opts = G0Opts {
        wide: &wide,
        peer: args.peer.as_deref(),
        emit_candidates: args.emit_candidates,
        ensure_refs: args.emit_refs || args.peer.is_some(),
        neg_controls: true,
    };
    let mut sc = G0Scratch { vs: &vs, mc: &mc, cb: &mut cb };
    let mut run_err: Option<String> = None;
    let mut lone = Value::Null;
    if !draft_only {
        let early: Vec<&G0CaseSpec> = cases.iter().filter(|c| !c.late).collect();
        run_err = run_case_list(&mut gpu, &mut slot, &mut d, &cfg, fx, &mut refs, &mut sc, &opts, &early, &mut rows, &mut pass);
        if run_err.is_none() {
            match lone_row_control(&mut gpu, &mut slot, &mut d, &cfg, fx, &mut refs, &mut sc) {
                Ok((j, ok)) => {
                    pass &= ok;
                    lone = j;
                }
                Err(e) => {
                    eprintln!("gate0 ctl_lone_64_row_request: ERROR {e}");
                    lone = json!({"case": "ctl_lone_64_row_request", "ok": false, "error": e.to_string()});
                    pass = false;
                    run_err = Some(e.to_string());
                }
            }
        }
    }
    let mut draft_rows = Vec::new();
    if run_err.is_none() {
        match draft_batch_cases(&mut gpu, &mut slot, &mut d, &cfg, fx, &mut refs, &mut *sc.cb) {
            Ok((r, ok)) => {
                pass &= ok;
                draft_rows = r;
            }
            Err(e) => {
                eprintln!("draftbatch: ERROR {e}");
                run_err = Some(e.to_string());
                pass = false;
            }
        }
    }
    if run_err.is_none() && !draft_only {
        let late: Vec<&G0CaseSpec> = cases.iter().filter(|c| c.late).collect();
        run_err = run_case_list(&mut gpu, &mut slot, &mut d, &cfg, fx, &mut refs, &mut sc, &opts, &late, &mut rows, &mut pass);
    }
    // Actual chunking at C8 (eight whole 16-row blocks = 128 rows): ONE
    // 128-row chunk on the wide route, else the product packing.
    let c8 = rows.iter().find(|c| c["case"] == "agg128_product").map(|c| c["wide"]["chunk_rows"].clone());
    let c8_json = match &c8 {
        Some(got) => {
            let want: Vec<usize> = if wide.family != WideFamily::Off {
                vec![128]
            } else {
                let mut r = Vec::new();
                match pack_whole_lanes(&vec![block; 128 / block], wide.route_cap(), &mut r) {
                    Ok(()) => r.iter().map(|x| (x.end - x.start) * block).collect(),
                    Err(_) => Vec::new(),
                }
            };
            let ok = *got == json!(want);
            pass &= ok;
            json!({"rows_per_chunk": got, "expected": want, "ok": ok})
        }
        None => Value::Null,
    };
    // Identity by loaded symbols: flag off loads neither wide family; HIP
    // loads no PM twin; PM loads no HIP twin.
    let loaded: Vec<String> = gpu.loaded_kernel_names().into_iter().filter(|k| parse_wide_symbol(k).is_some()).collect();
    let loaded_pm = loaded.iter().filter(|k| parse_wide_symbol(k).is_some_and(|s| s.pm)).count();
    let loaded_hip = loaded.len() - loaded_pm;
    let wide_ran = rows.iter().any(|c| c["wide"]["wide_chunks"].as_u64().unwrap_or(0) > 0);
    let loaded_ok = match wide.family {
        WideFamily::Off => loaded.is_empty(),
        WideFamily::Hip => !wide_ran || (loaded_hip > 0 && loaded_pm == 0),
        WideFamily::Pm => !wide_ran || loaded_hip == 0,
    };
    pass &= loaded_ok;
    let env = receipt_env();
    let mut pre_keys: Vec<usize> = refs.pre.keys().copied().collect();
    pre_keys.sort_unstable();
    let pre_json: Vec<Value> = pre_keys
        .iter()
        .map(|fi| {
            let p = &refs.pre[fi];
            json!({"fixture": fx[*fi].name(), "position": p.position, "pending_seed": p.seed, "snapshot_rows": p.rows, "draft_ctx_mode": p.ctx_mode})
        })
        .collect();
    let report = json!({
        "probe_window": windows,
        "block": block,
        "multi_chunk_max_rows": MULTI_CHUNK_MAX_ROWS,
        "draft": {"path": draft_path, "bytes": draft_bytes, "sha256": draft_sha},
        "draft_ctx_capacity": ctx_cap,
        "env": env,
        "wide_route": {
            "expect": wide.json(),
            "aggregates": args.aggregates,
            "peer": args.peer,
            "emit_candidates": args.emit_candidates,
            "c8_chunk128": c8_json,
            "loaded_wide_symbols": loaded,
            "loaded_identity_ok": loaded_ok,
            "wide_chunks_run": wide_ran,
            "lone_64_row_request": lone,
        },
        "arch": gpu.arch.clone(),
        "fixtures": pre_json,
        "cases": rows,
        "draft_batch": draft_rows,
        "error": run_err,
        "unexercised": {
            "dflash_executor_k1_8": "not exercised by this probe: it drives the shared trunk/head/accept entries (direct and dflash_cb_verify product packing) over replayed pre-window states, not the VMM DFlash engine's planner/executor; engine-level and serve cases are the HTTP gates",
            "mixed_mtp_dflash_trunk": "MTP lanes (fusion=Off) in a shared DFlash chunk need the tagged driver (slice C)",
            "terminal_prefix_repair_and_promotion": "slice C/D",
            "whole_cb_graph": "refused by design (dflash_cb_verify / forward_prefill_batch_multi); only the refusal-before-mutation controls ran, no graph parity is claimed",
            "phase_costs": "HIPFIRE_SPEC_PHASES / matched timing is reviewer-owned; shared_window_ms/ref_step_ms here are uncontrolled single runs, not a speed claim",
        },
        "pass": pass,
    });
    // Cleanup (free before returning the bundle).
    drop(sc);
    vs.free_gpu(&mut gpu);
    if let Err(e) = mc.free_gpu(&mut gpu) {
        eprintln!("gate0: free multi scratch: {e}");
    }
    if let Err(e) = cb.free_gpu(&mut gpu) {
        eprintln!("gate0: free cb scratch: {e}");
    }
    d.free(&mut gpu);
    let b = slot.into_bundle();
    Ok((Ctx { gpu, b }, report, pass))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.artifacts.exists() {
        return Err(format!("artifacts dir {} exists; refusing to overwrite", args.artifacts.display()).into());
    }
    fs::create_dir_all(&args.artifacts)?;
    let (mut ctx, tok, ack) = load(&args.model)?;
    let max_prefix = args.contexts.iter().copied().chain([args.short]).max().unwrap();
    if max_prefix + args.steps + 8 > ack["max_seq_bound"].as_u64().unwrap() as usize {
        return Err(format!("context {max_prefix}+{} exceeds loaded max_seq bound", args.steps).into());
    }
    let fx = fixtures(&args, &tok)?;
    let mut report = json!({
        "model": args.model,
        "ks": args.ks,
        "contexts": args.contexts,
        "short": args.short,
        "steps": args.steps,
        "loaded_ack": ack,
        "artifacts": args.artifacts,
    });
    report["env"] = Value::Object(receipt_env());
    report["wide_flags"] = json!({
        "HIPFIRE_CB_VERIFY_CHUNK128": std::env::var("HIPFIRE_CB_VERIFY_CHUNK128").ok(),
        "HIPFIRE_CB_VERIFY_PM": std::env::var("HIPFIRE_CB_VERIFY_PM").ok(),
        "mq4_verify_chunk_rows": ctx.gpu.mq4_verify_chunk_rows(),
        "multi_chunk_row_cap": multi_chunk_row_cap(&ctx.gpu),
        "multi_chunk_max_rows": MULTI_CHUNK_MAX_ROWS,
        "multi_chunk_product_max_rows": MULTI_CHUNK_PRODUCT_MAX_ROWS,
    });

    // ── Singleton phase: references + determinism self-test ──────────
    let mut refs = Vec::with_capacity(fx.len());
    let mut pass = true;
    if args.ar_phase {
    let mut singleton = Vec::new();
    for f in &fx {
        let t0 = std::time::Instant::now();
        let tr = record(&mut ctx, f, args.steps, &args.artifacts.join("singleton").join(f.name()))?;
        let rec_s = t0.elapsed().as_secs_f64();
        let again = replay(&mut ctx, f, args.steps, &tr, Perturb::None)?;
        pass &= again.exact();
        eprintln!("singleton {}: determinism exact={} ({:.1}s)", f.name(), again.exact(), rec_s);
        singleton.push(json!({
            "fixture": f.name(),
            "prefix": f.prefix,
            "prompt_sha256": sha(&f.tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>()),
            "position": tr.position,
            "pending_seed": tr.pending_seed,
            "committed_ids": tr.committed,
            "final_logits_sha256": sha(&fs::read(tr.logits.last().unwrap())?),
            "record_seconds": rec_s,
            "determinism": again.json(),
        }));
        refs.push(tr);
    }
    report["singleton"] = Value::Array(singleton);

    // ── Negative/positive controls on the shortest fixture ───────────
    let ci = (0..fx.len()).min_by_key(|&i| fx[i].prefix).unwrap();
    let (f, tr) = (&fx[ci], &refs[ci]);
    let mut controls = serde_json::Map::new();
    let mut control = |name: &str, d: Diff, must_match: bool| {
        let ok = d.exact() == must_match;
        pass &= ok;
        eprintln!("control {name}: exact={} expected_exact={must_match} -> {}", d.exact(), if ok { "OK" } else { "FAIL" });
        controls.insert(name.into(), json!({"expected_exact": must_match, "ok": ok, "diff": d.json()}));
    };
    let n = args.steps;
    control("neg_row_slot", replay(&mut ctx, f, n, tr, Perturb::RowSlot)?, false);
    control("neg_rejected_tail_read", replay(&mut ctx, f, n, tr, Perturb::RejectedTailRead)?, false);
    control("pos_rejected_tail_masked", replay(&mut ctx, f, n, tr, Perturb::RejectedTailMasked)?, true);
    let same = (0..fx.len()).find(|&i| i != ci && fx[i].prefix == f.prefix);
    if let Some(other) = same.or_else(|| (0..fx.len()).find(|&i| i != ci)) {
        // Epoch control: request ci replayed against another request's reference.
        let mut d = replay(&mut ctx, f, n, &refs[other], Perturb::None)?;
        if fx[other].prefix != f.prefix {
            d.differing_items.push("prefix length differs (epoch mismatch)".into());
        }
        control("neg_epoch", d, false);
    }
    report["controls"] = Value::Object(controls);
    }
    if args.batch {
        let (batch, ok) = batch_phase(&mut ctx, &args, &fx, &refs)?;
        // Isolation (batch == isolated executor, controls) and route
        // exactness (executor == singleton route, §4.3 exact default) are
        // separate verdicts; both are required for the exact-route gate.
        let route = batch["route_exact_vs_singleton"].as_bool() == Some(true);
        report["pass_batch_isolation"] = json!(ok);
        report["route_exact_vs_singleton"] = json!(route);
        report["batch"] = batch;
        pass &= ok && route;
    }
    if args.spec.dflash() {
        // The DFlash probe turns the bundle into a ModelSlot and back.
        match dflash_probe_phase(ctx, &args, &fx, &ack) {
            Ok((c, probe, ok)) => {
                ctx = c;
                report["dflash_gate0"] = probe;
                report["dflash_executor"] = json!({
                    "status": "unexercised",
                    "reason": "the k=1..8 DFlash executor cases (engine planner/executor) are not driven by this oracle; only the Gate 0 shared-trunk probe (direct + dflash_cb_verify product packing, aggregate boundaries, controls) ran",
                });
                pass &= ok;
            }
            Err(e) => {
                report["dflash_gate0"] = json!({"error": e.to_string()});
                report["pass"] = json!(false);
                fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
                eprintln!("wrote {} pass=false (DFlash Gate 0 error: {e})", args.out.display());
                return Err(e);
            }
        }
    }
    if args.spec.mtp() {
        // Last: the spec phase turns the bundle into a ModelSlot.
        let (spec, ok) = spec_phase(ctx, &args, &fx, &refs)?;
        report["spec_mtp"] = spec;
        pass &= ok;
        report["pass"] = json!(pass);
        fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("wrote {} pass={pass}", args.out.display());
        return if pass { Ok(()) } else { Err("oracle FAILED (see report)".into()) };
    }
    report["pass"] = json!(pass);
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("wrote {} pass={pass}", args.out.display());
    if !pass {
        return Err("oracle FAILED (see report)".into());
    }
    Ok(())
}
