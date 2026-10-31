// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `bench` subcommand (PLAN §6.4): prefill / decode throughput and
//! kernel-bucket attribution for Raon-1-mini through the production bundle.
//!
//! ```text
//! raon_oracle bench --model M --pp 512,2048,8192,16384 --tg 128
//!                   [--reps 3] [--warmup 1] [--prompt-file F] [--tg-prompt 32]
//!                   [--prefill batched|serial] [--max-seq N] [--model-md5 skip]
//!                   [--profile] --out FILE
//! raon_oracle bench --bucket-json PROFILE.json --out FILE     (CPU only)
//! ```
//!
//! Conventions (llama-bench style):
//!
//! * **pp N** — a cold slot (`reset`, outside the timer) prefills N fixed token
//!   ids through `RaonBundle::prefill` (the route `serve` runs: bounded batched
//!   chunks, last-row head only); the timer brackets the prefill call and a
//!   device synchronize. tok/s = N / seconds. `--warmup` untimed runs precede
//!   `--reps` timed runs of every N. Reported: per-rep tok/s, mean, sample
//!   stdev, median, min, max.
//! * **token ids** — `benchmarks/prompts/raon/state.txt` tokenized by the
//!   checkpoint's own tokenizer (plain encode, no template, no BOS) and repeated
//!   cyclically to N ids. The md5 of the little-endian u32 id stream of every
//!   N is recorded, so two results are comparable only when it matches.
//! * **tg T** — a cold slot prefills a short prompt (`--tg-prompt` ids, default
//!   32, the head of the same stream), then T greedy decode steps run. Two
//!   measurements per rep: `greedy` times every token on the host (logits
//!   download + first-max argmax + `decode_step`; EOS is fed on like any other
//!   token), and `device_loop` replays the exact ids of that greedy run with no
//!   readback and one synchronize at the end (the `bench_decode` shape).
//! * The decode route is whatever `RaonBundle::decode_step` runs; the result
//!   records the controls that select it.
//!
//! `--profile` attributes one pass per `--pp` N and one `--tg` pass to kernel
//! buckets. It uses launch-level timing (`rdna_compute::profile::start_launches`):
//! every kernel through the launch funnels is bracketed by an event pair and
//! synchronized, so the table is complete even for kernels that never carried a
//! hand-placed `begin_timer`. The legacy `begin_timer` sum is reported next to
//! it as a coverage cross-check. Decode is pinned eager (`HIPFIRE_GRAPH=0` when
//! unset): a replayed graph has no per-launch events. Every span carries a fixed
//! event+dispatch floor (`launch_span_floor_us`, measured with a minimal launch),
//! so bucket times are reported floor-corrected beside the raw spans; the true
//! kernel time of a bucket lies between the two. Reconciliation per section:
//! `wall_unprofiled = corrected_kernel_sum + residual` (the residual — launch
//! gaps, idle GPU, host work, copies, launches outside the funnels — is its own
//! bucket, so bucket fractions of the unprofiled wall add to 1), and
//! `residual_ms_range` brackets it using the raw sum.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hipfire_arch_raon::bundle::{PrefillMode, RaonBundle};
use hipfire_arch_raon::forward::readout_logits;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};

use super::logits::{check_tokens, load_bundle, parse_prefill};

const PROTOCOL: &str = "raon-bench-v1";
const DEFAULT_PROMPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../benchmarks/prompts/raon/state.txt"
);

/// Bucket order of every report (the residual is appended last).
const BUCKETS: [&str; 16] = [
    "moe_grouped",
    "moe_indexed",
    "moe_route",
    "trunk_gemm",
    "act_rotate",
    "rda_recurrence",
    "conv_prepare",
    "mla_attention",
    "mla_decompress",
    "mhc",
    "norms",
    "head",
    "embed",
    "elementwise",
    "other",
    "launch_gaps_untimed",
];

// ───────────────────────── small helpers ─────────────────────────

fn hip<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e:?}"))
}

fn sync(gpu: &Gpu) -> Result<(), String> {
    hip(gpu.hip.device_synchronize(), "device synchronize")
}

fn opt_usize(opts: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match opts.get(key) {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("--{key} `{s}`: {e}")),
        None => Ok(default),
    }
}

fn md5_hex(bytes: &[u8]) -> String {
    format!("{:x}", md5::compute(bytes))
}

fn md5_file(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut ctx = md5::Context::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        ctx.consume(&buf[..n]);
    }
    Ok(format!("{:x}", ctx.finalize()))
}

fn ids_md5(ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for &t in ids {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    md5_hex(&bytes)
}

/// First maximum (lowest index on ties).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample standard deviation (0 for one sample).
fn stdev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stats(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    json!({
        "n": v.len(),
        "mean": mean(v),
        "stdev": stdev(v),
        "median": percentile(&s, 0.5),
        "min": s[0],
        "max": s[s.len() - 1],
    })
}

// ───────────────────────── identity / environment ─────────────────────────

fn read_trim(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// The line of a `pp_dpm_*` table marked active (`*`).
fn active_dpm(path: &str) -> Value {
    match std::fs::read_to_string(path) {
        Ok(s) => s
            .lines()
            .find(|l| l.contains('*'))
            .map(|l| json!(l.trim()))
            .unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

/// Instantaneous clocks/power/temperature of the device at `pci` from sysfs;
/// absent files are `null`.
fn clocks_json(pci: &str) -> Value {
    let base = format!("/sys/bus/pci/devices/{}", pci.to_ascii_lowercase());
    let hwmon = std::fs::read_dir(format!("{base}/hwmon"))
        .ok()
        .and_then(|mut d| d.next())
        .and_then(|e| e.ok())
        .map(|e| e.path().display().to_string());
    let hw = |name: &str| -> Value {
        hwmon
            .as_ref()
            .and_then(|h| read_trim(&format!("{h}/{name}")))
            .and_then(|s| s.parse::<u64>().ok())
            .map(|v| json!(v))
            .unwrap_or(Value::Null)
    };
    json!({
        "pp_dpm_sclk": active_dpm(&format!("{base}/pp_dpm_sclk")),
        "pp_dpm_mclk": active_dpm(&format!("{base}/pp_dpm_mclk")),
        "gpu_busy_percent": read_trim(&format!("{base}/gpu_busy_percent")),
        "hwmon_sclk_hz": hw("freq1_input"),
        "hwmon_power_uw": hw("power1_average"),
        "hwmon_temp_mc": hw("temp1_input"),
    })
}

/// Model identity: size, mtime and md5. The md5 comes from `<model>.md5` when
/// present; otherwise it is computed once and written there (a read-only
/// directory keeps the computed value in the result only). `--model-md5 skip`
/// records size+mtime only.
fn model_identity(model: &str, skip_md5: bool) -> Result<Value, String> {
    let meta = std::fs::metadata(model).map_err(|e| format!("stat {model}: {e}"))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    let sidecar = format!("{model}.md5");
    let (md5, source) = if skip_md5 {
        (Value::Null, "skipped")
    } else if let Some(text) = read_trim(&sidecar) {
        match text.split_whitespace().next() {
            Some(h) if h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()) => {
                (json!(h.to_ascii_lowercase()), "sidecar")
            }
            _ => return Err(format!("{sidecar}: first field is not a 32-hex md5")),
        }
    } else {
        let h = md5_file(Path::new(model))?;
        let name = Path::new(model)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        match std::fs::write(&sidecar, format!("{h}  {name}\n")) {
            Ok(()) => (json!(h), "computed+sidecar-written"),
            Err(e) => {
                eprintln!("bench: cannot write {sidecar}: {e}");
                (json!(h), "computed")
            }
        }
    };
    Ok(json!({
        "path": model,
        "size_bytes": meta.len(),
        "mtime_unix": mtime,
        "md5": md5,
        "md5_source": source,
    }))
}

fn binary_identity() -> Value {
    match std::env::current_exe() {
        Ok(p) => json!({
            "path": p.display().to_string(),
            "size_bytes": std::fs::metadata(&p).ok().map(|m| m.len()),
            "md5": md5_file(&p).ok(),
        }),
        Err(e) => json!({"error": e.to_string()}),
    }
}

fn env_controls() -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in std::env::vars() {
        if k.starts_with("HIPFIRE_")
            || matches!(
                k.as_str(),
                "HIP_VISIBLE_DEVICES" | "ROCR_VISIBLE_DEVICES" | "HSA_OVERRIDE_GFX_VERSION"
            )
        {
            m.insert(k, json!(v));
        }
    }
    Value::Object(m)
}

// ───────────────────────── token streams ─────────────────────────

struct Stream {
    base: Vec<u32>,
}

impl Stream {
    fn ids(&self, n: usize) -> Vec<u32> {
        (0..n).map(|i| self.base[i % self.base.len()]).collect()
    }
}

// ───────────────────────── measurement primitives ─────────────────────────

fn cold(bundle: &mut RaonBundle, gpu: &mut Gpu) -> Result<(), String> {
    bundle.reset(gpu)?;
    sync(gpu)
}

/// Download the logits and return `(argmax, finite)`.
fn read_top1(bundle: &RaonBundle, gpu: &Gpu) -> Result<(usize, bool), String> {
    let l = bundle.download_logits(gpu)?;
    Ok((argmax(&l), l.iter().all(|x| x.is_finite())))
}

/// One timed prefill of `ids` from a cold slot; seconds.
fn timed_prefill(bundle: &mut RaonBundle, gpu: &mut Gpu, ids: &[u32]) -> Result<f64, String> {
    cold(bundle, gpu)?;
    let t0 = Instant::now();
    bundle.prefill(gpu, ids)?;
    sync(gpu)?;
    Ok(t0.elapsed().as_secs_f64())
}

struct GreedyRun {
    ids: Vec<u32>,
    token_secs: Vec<f64>,
    finite: bool,
    eos_seen: bool,
}

/// Cold slot, prefill `prompt`, then `tg` greedy decode steps timed per token.
fn greedy_run(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    tg: usize,
) -> Result<GreedyRun, String> {
    cold(bundle, gpu)?;
    bundle.prefill(gpu, prompt)?;
    sync(gpu)?;
    let mut ids = Vec::with_capacity(tg);
    let mut token_secs = Vec::with_capacity(tg);
    let (mut tok, mut finite) = read_top1(bundle, gpu)?;
    let mut eos_seen = false;
    let mut t_prev = Instant::now();
    for _ in 0..tg {
        let t = tok as u32;
        eos_seen |= bundle.is_eos(t);
        ids.push(t);
        bundle.decode_step(gpu, t)?;
        let (next, fin) = read_top1(bundle, gpu)?;
        let now = Instant::now();
        token_secs.push(now.duration_since(t_prev).as_secs_f64());
        t_prev = now;
        finite &= fin;
        tok = next;
    }
    Ok(GreedyRun {
        ids,
        token_secs,
        finite,
        eos_seen,
    })
}

/// One device-loop rep: wall seconds plus how the decode steps were served.
struct DeviceLoopRep {
    secs: f64,
    /// `RaonBundle::decode_mode()` at the time of the rep.
    mode: &'static str,
    /// Graph captures / replays during the rep's decode steps.
    captures: u64,
    replays: u64,
    /// Host ms spent inside `hipGraphLaunch` per replayed step (0 when eager).
    launch_ms_per_step: f64,
    /// Host-side duration of every `decode_step` call (ms).
    step_ms: Vec<f64>,
    /// Unix seconds at the first decode step and after the closing synchronize.
    t_start_unix: f64,
    t_end_unix: f64,
}

/// Cold slot, prefill `prompt`, then replay `ids` with no readback; the timer
/// brackets the decode steps and one synchronize.
fn device_loop_rep(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    ids: &[u32],
) -> Result<DeviceLoopRep, String> {
    cold(bundle, gpu)?;
    bundle.prefill(gpu, prompt)?;
    sync(gpu)?;
    let (c0, r0) = bundle.decode_graph_stats();
    let l0 = bundle.decode_graph_launch_secs();
    let mut step_ms = Vec::with_capacity(ids.len());
    let unix_now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    };
    let t_start_unix = unix_now();
    let t0 = Instant::now();
    for &t in ids {
        let s0 = Instant::now();
        bundle.decode_step(gpu, t)?;
        step_ms.push(s0.elapsed().as_secs_f64() * 1e3);
    }
    sync(gpu)?;
    let secs = t0.elapsed().as_secs_f64();
    let t_end_unix = unix_now();
    let (c1, r1) = bundle.decode_graph_stats();
    let launched = (c1 - c0 + r1 - r0).max(1) as f64;
    let launch_ms_per_step = (bundle.decode_graph_launch_secs() - l0) * 1e3 / launched;
    Ok(DeviceLoopRep {
        launch_ms_per_step,
        t_start_unix,
        t_end_unix,
        secs,
        mode: bundle.decode_mode(),
        captures: c1 - c0,
        replays: r1 - r0,
        step_ms,
    })
}

/// Cold slot, prefill `prompt`, then replay `ids` with no readback; seconds.
fn device_loop(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    prompt: &[u32],
    ids: &[u32],
) -> Result<f64, String> {
    device_loop_rep(bundle, gpu, prompt, ids).map(|r| r.secs)
}

// ───────────────────────── throughput ─────────────────────────

fn bench_pp(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    stream: &Stream,
    n: usize,
    reps: usize,
    warmup: usize,
    pci: &str,
) -> Result<Value, String> {
    let ids = stream.ids(n);
    check_tokens(&ids, bundle.config.vocab, "pp ids")?;
    for _ in 0..warmup {
        timed_prefill(bundle, gpu, &ids)?;
    }
    let clocks_before = clocks_json(pci);
    let mut secs = Vec::with_capacity(reps);
    let mut tok_s = Vec::with_capacity(reps);
    let mut top1 = Vec::with_capacity(reps);
    let mut finite = true;
    for _ in 0..reps {
        let s = timed_prefill(bundle, gpu, &ids)?;
        secs.push(s);
        tok_s.push(n as f64 / s.max(1e-12));
        let (t, f) = read_top1(bundle, gpu)?;
        top1.push(t);
        finite &= f;
    }
    let clocks_after = clocks_json(pci);
    Ok(json!({
        "n": n,
        "ids_md5": ids_md5(&ids),
        "warmup": warmup,
        "tok_s": tok_s,
        "secs": secs,
        "stats": stats(&tok_s),
        "top1": top1,
        "top1_stable": top1.windows(2).all(|w| w[0] == w[1]),
        "logits_finite": finite,
        "clocks_before": clocks_before,
        "clocks_after": clocks_after,
    }))
}

fn bench_tg(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    stream: &Stream,
    tg: usize,
    prompt_tokens: usize,
    reps: usize,
    warmup: usize,
    pci: &str,
) -> Result<Value, String> {
    let prompt = stream.ids(prompt_tokens);
    check_tokens(&prompt, bundle.config.vocab, "tg prompt")?;
    let mut first_ids: Option<Vec<u32>> = None;
    let mut deterministic = true;
    let mut finite = true;
    let mut eos_seen = false;
    for _ in 0..warmup {
        let g = greedy_run(bundle, gpu, &prompt, tg)?;
        device_loop(bundle, gpu, &prompt, &g.ids)?;
    }
    let clocks_before = clocks_json(pci);
    let mut greedy_tok_s = Vec::new();
    let mut device_tok_s = Vec::new();
    let mut device_reps: Vec<Value> = Vec::new();
    let mut last_token_ms: Vec<f64> = Vec::new();
    let mut greedy_reps: Vec<Value> = Vec::new();
    for _ in 0..reps {
        let (gc0, gr0) = bundle.decode_graph_stats();
        let g = greedy_run(bundle, gpu, &prompt, tg)?;
        let (gc1, gr1) = bundle.decode_graph_stats();
        let total: f64 = g.token_secs.iter().sum();
        greedy_tok_s.push(tg as f64 / total.max(1e-12));
        greedy_reps.push(json!({
            "decode_mode": bundle.decode_mode(),
            "graph_captures": gc1 - gc0,
            "graph_replays": gr1 - gr0,
            "tok_s": tg as f64 / total.max(1e-12),
        }));
        finite &= g.finite;
        eos_seen |= g.eos_seen;
        match &first_ids {
            None => first_ids = Some(g.ids.clone()),
            Some(f) => deterministic &= *f == g.ids,
        }
        last_token_ms = g.token_secs.iter().map(|s| s * 1e3).collect();
        let d = device_loop_rep(bundle, gpu, &prompt, &g.ids)?;
        device_tok_s.push(tg as f64 / d.secs.max(1e-12));
        let mut sm = d.step_ms.clone();
        sm.sort_by(|a, b| a.total_cmp(b));
        device_reps.push(json!({
            "decode_mode": d.mode,
            "graph_captures": d.captures,
            "graph_replays": d.replays,
            "tok_s": tg as f64 / d.secs.max(1e-12),
            "graph_launch_ms_per_step": d.launch_ms_per_step,
            "t_start_unix": d.t_start_unix,
            "t_end_unix": d.t_end_unix,
            "step_ms": {"mean": mean(&d.step_ms), "p50": percentile(&sm, 0.5), "max": sm.last()},
            "slow_steps_over_1p5x_p50": sm.get(sm.len() / 2).map_or(0, |p| d.step_ms.iter().filter(|m| **m > 1.5 * p).count()),
        }));
    }
    let clocks_after = clocks_json(pci);
    let mut sorted = last_token_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let ids = first_ids.unwrap_or_default();
    Ok(json!({
        "tokens": tg,
        "prompt_tokens": prompt_tokens,
        "prompt_ids_md5": ids_md5(&prompt),
        "generated_ids_md5": ids_md5(&ids),
        "generated_ids": ids,
        "greedy_deterministic_across_reps": deterministic,
        "logits_finite": finite,
        "eos_fed_on": eos_seen,
        "warmup": warmup,
        "greedy": {
            "what": "per-token host timing: logits download + first-max argmax + decode_step",
            "tok_s": greedy_tok_s,
            "stats": stats(&greedy_tok_s),
            "reps": greedy_reps,
            "last_rep_token_ms": {
                "mean": mean(&last_token_ms),
                "p50": percentile(&sorted, 0.5),
                "p95": percentile(&sorted, 0.95),
                "min": sorted.first(),
                "max": sorted.last(),
            },
        },
        "device_loop": {
            "what": "greedy ids replayed, no readback, one synchronize",
            "tok_s": device_tok_s,
            "stats": stats(&device_tok_s),
            "reps": device_reps,
        },
        "clocks_before": clocks_before,
        "clocks_after": clocks_after,
    }))
}

// ───────────────────────── profile ─────────────────────────

/// Kernel-name bucket of a (non-head) launch. Head launches are tagged by
/// position, never by name (the head shares generic GEMM kernels).
pub fn bucket_of(kernel: &str) -> &'static str {
    let k = kernel.to_ascii_lowercase();
    let has = |s: &str| k.contains(s);
    if k.starts_with("stream_mix") || k.starts_with("hc_row") || has("mhc") {
        return "mhc";
    }
    if k.starts_with("latent_attention") {
        return if has("serial") || has("gate") || has("flash") {
            "mla_attention"
        } else {
            "mla_decompress"
        };
    }
    if has("moe")
        || has("expert")
        || has("topk")
        || (has("ffn_situ") && has("grouped"))
        || (has("grouped") && has("ffn"))
    {
        if has("router")
            || has("topk")
            || (has("scatter") && !has("unscatter"))
            || has("histogram")
            || has("offsets")
            || has("permute")
            || has("combine_order")
            || has("sorted")
        {
            return "moe_route";
        }
        if has("indexed") {
            return "moe_indexed";
        }
        return "moe_grouped";
    }
    if has("conv") || has("rda_prepare") {
        return "conv_prepare";
    }
    if has("gated_delta") || has("rda") || has("delta_net") {
        return "rda_recurrence";
    }
    if has("norm") {
        return "norms";
    }
    if has("rotate") {
        return "act_rotate";
    }
    if has("ffn_") {
        return "elementwise";
    }
    if has("embedding") {
        return "embed";
    }
    if has("gemm") || has("gemv") || has("mmq") || has("wmma") || has("rocblas") || has("matmul") {
        return "trunk_gemm";
    }
    if has("round")
        || has("convert")
        || has("copy")
        || has("add")
        || has("mul")
        || has("silu")
        || has("swiglu")
        || has("cast")
    {
        return "elementwise";
    }
    "other"
}

/// Accumulated launches: `(kernel, is_head) -> (calls, total_us)`.
type KernelTable = HashMap<(String, bool), (u64, f64)>;

fn absorb(
    table: &mut KernelTable,
    entries: Vec<rdna_compute::profile::LaunchEntry>,
    head: &[String],
) {
    // The head is the tail of a forward that returns logits: tag the trailing
    // launches whose names equal the standalone readout's, in order.
    let n = entries.len();
    let tag_from = if !head.is_empty()
        && n >= head.len()
        && entries[n - head.len()..]
            .iter()
            .zip(head)
            .all(|(e, h)| &e.kernel == h)
    {
        n - head.len()
    } else {
        n
    };
    for (i, e) in entries.into_iter().enumerate() {
        let slot = table.entry((e.kernel, i >= tag_from)).or_insert((0, 0.0));
        slot.0 += 1;
        slot.1 += e.time_us;
    }
}

/// Bucket report of one profiled section. `wall_ms` is the unprofiled wall of
/// the same work; `kernels` come from the launch-level pass.
fn section_report(
    name: &str,
    wall_ms: f64,
    profiled_wall_ms: f64,
    table: &KernelTable,
    legacy_sum_ms: f64,
    units: f64,
    unit_name: &str,
    head_launches: usize,
    floor_us: f64,
) -> Value {
    // Every launch-level span carries a fixed event/dispatch floor (a launch of
    // an empty-ish kernel measures `floor_us`); it is not kernel work and, over
    // thousands of ~10 us decode launches, it dominates. Buckets report
    // floor-corrected time; raw spans are kept beside it.
    let corrected = |calls: u64, us: f64| (us - calls as f64 * floor_us).max(0.0);
    let raw_sum_us: f64 = table.values().map(|v| v.1).sum();
    let kernel_sum_ms: f64 = table.values().map(|v| corrected(v.0, v.1)).sum::<f64>() / 1e3;
    let launches: u64 = table.values().map(|v| v.0).sum();
    let mut bucket_us: BTreeMap<&'static str, (u64, f64, f64)> = BTreeMap::new();
    let mut rows: Vec<Value> = Vec::new();
    let mut entries: Vec<(&(String, bool), &(u64, f64))> = table.iter().collect();
    entries.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
    for ((kernel, is_head), (calls, us)) in entries {
        let bucket = if *is_head { "head" } else { bucket_of(kernel) };
        let c = corrected(*calls, *us);
        let b = bucket_us.entry(bucket).or_insert((0, 0.0, 0.0));
        b.0 += calls;
        b.1 += c;
        b.2 += us;
        rows.push(json!({
            "name": kernel,
            "head": is_head,
            "bucket": bucket,
            "calls": calls,
            "total_us": us,
            "corrected_us": c,
        }));
    }
    let gap_ms = wall_ms - kernel_sum_ms;
    let mut buckets = Vec::new();
    for b in BUCKETS {
        let (calls, ms, raw_ms) = if b == "launch_gaps_untimed" {
            (0, gap_ms, Value::Null)
        } else {
            let (c, us, raw) = bucket_us.get(b).copied().unwrap_or((0, 0.0, 0.0));
            (c, us / 1e3, json!(raw / 1e3))
        };
        buckets.push(json!({
            "bucket": b,
            "ms": ms,
            "raw_span_ms": raw_ms,
            "frac_of_raw_span_sum": if b == "launch_gaps_untimed" { Value::Null } else { json!(raw_ms.as_f64().unwrap_or(0.0) * 1e3 / raw_sum_us) },
            "frac_of_wall": ms / wall_ms,
            "frac_of_kernel_sum": if b == "launch_gaps_untimed" { Value::Null } else { json!(ms / kernel_sum_ms) },
            "launches": calls,
            "per_unit_ms": ms / units,
        }));
    }
    json!({
        "section": name,
        "unit": unit_name,
        "units": units,
        "wall_unprofiled_ms": wall_ms,
        "wall_profiled_ms": profiled_wall_ms,
        "launch_span_floor_us": floor_us,
        "kernel_sum_raw_ms": raw_sum_us / 1e3,
        "kernel_sum_ms": kernel_sum_ms,
        "launches": launches,
        "head_launches_per_forward": head_launches,
        "residual_ms": gap_ms,
        "residual_ms_range": [wall_ms - raw_sum_us / 1e3, gap_ms],
        "residual_frac_of_wall": gap_ms / wall_ms,
        "kernel_sum_exceeds_wall": kernel_sum_ms > wall_ms,
        "profile_inflation_ms": profiled_wall_ms - wall_ms,
        "legacy_begin_timer_sum_ms": legacy_sum_ms,
        "legacy_timer_coverage_of_kernel_sum": legacy_sum_ms / kernel_sum_ms,
        "buckets": buckets,
        "kernels": rows,
        "note": "raw spans = launch-level hipEvent pairs (each launch synchronized); every span carries a fixed \
                 launch_span_floor_us (event + dispatch), so a bucket's true kernel time lies in [corrected, raw]. \
                 buckets.ms is floor-corrected (clamped at 0 per kernel name); residual = unprofiled wall - corrected \
                 kernel_sum (launch gaps, idle GPU, host work, copies, launches outside the funnels, plus any kernel \
                 time the floor over-subtracted); residual_ms_range = [wall - raw_sum, residual]. A negative lower \
                 bound means the raw spans over-count (launch-bound sections). Profiled wall is inflated by the per-launch syncs.",
    })
}

/// Median span of a minimal launch through the funnel: the fixed cost every
/// launch-level span carries regardless of kernel work.
fn launch_floor_us(gpu: &mut Gpu) -> Result<f64, String> {
    let t = hip(gpu.alloc_tensor(&[64], DType::F32), "alloc floor probe")?;
    for _ in 0..16 {
        hip(
            gpu.bf16_round_trip_f32_strided(&t, 1, 0, 64, 64),
            "floor warmup",
        )?;
    }
    sync(gpu)?;
    rdna_compute::profile::start_launches();
    let mut r = Ok(());
    for _ in 0..512 {
        r = r.and_then(|_| {
            hip(
                gpu.bf16_round_trip_f32_strided(&t, 1, 0, 64, 64),
                "floor probe",
            )
        });
    }
    let entries = rdna_compute::profile::stop_launches().unwrap_or_default();
    let _ = gpu.free_tensor(t);
    r?;
    let mut us: Vec<f64> = entries.iter().map(|e| e.time_us).collect();
    if us.is_empty() {
        return Err("floor probe recorded no launches".into());
    }
    us.sort_by(|a, b| a.total_cmp(b));
    Ok(percentile(&us, 0.5))
}

/// Kernel names of one standalone `readout_logits` (final norm + lm_head of a
/// single row) on the live stream state, in launch order.
fn head_signature(bundle: &mut RaonBundle, gpu: &mut Gpu) -> Result<Vec<String>, String> {
    let out = hip(
        gpu.alloc_tensor(&[bundle.config.vocab], DType::F32),
        "alloc head probe",
    )?;
    sync(gpu)?;
    rdna_compute::profile::start_launches();
    let r = readout_logits(
        &bundle.config,
        &bundle.weights,
        &mut bundle.scratch,
        &mut bundle.work,
        gpu,
        0,
        1,
        &out,
    );
    let s = sync(gpu);
    let entries = rdna_compute::profile::stop_launches().unwrap_or_default();
    let _ = gpu.free_tensor(out);
    r?;
    s?;
    Ok(entries.into_iter().map(|e| e.kernel).collect())
}

fn profile_pp(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    stream: &Stream,
    n: usize,
    floor_us: f64,
) -> Result<Value, String> {
    let ids = stream.ids(n);
    check_tokens(&ids, bundle.config.vocab, "profile pp ids")?;
    // Warm every kernel/route once, then take the unprofiled wall.
    timed_prefill(bundle, gpu, &ids)?;
    let wall = timed_prefill(bundle, gpu, &ids)?;
    // The warmed state still holds the final streams: learn the head launches.
    let head = head_signature(bundle, gpu)?;
    // Legacy hand-placed timers (coverage cross-check).
    cold(bundle, gpu)?;
    rdna_compute::profile::start();
    let r = bundle.prefill(gpu, &ids).and_then(|_| sync(gpu));
    let legacy = rdna_compute::profile::stop().unwrap_or_default();
    r?;
    let legacy_ms: f64 = legacy.iter().map(|e| e.time_us).sum::<f64>() / 1e3;
    // Launch-level pass.
    cold(bundle, gpu)?;
    rdna_compute::profile::start_launches();
    let t0 = Instant::now();
    let r = bundle.prefill(gpu, &ids).and_then(|_| sync(gpu));
    let profiled = t0.elapsed().as_secs_f64();
    let entries = rdna_compute::profile::stop_launches().unwrap_or_default();
    r?;
    let mut table = KernelTable::new();
    absorb(&mut table, entries, &head);
    let mut rep = section_report(
        &format!("pp{n}"),
        wall * 1e3,
        profiled * 1e3,
        &table,
        legacy_ms,
        n as f64,
        "token",
        head.len(),
        floor_us,
    );
    rep["n"] = json!(n);
    rep["ids_md5"] = json!(ids_md5(&ids));
    rep["tok_s_unprofiled"] = json!(n as f64 / wall.max(1e-12));
    Ok(rep)
}

fn profile_tg(
    bundle: &mut RaonBundle,
    gpu: &mut Gpu,
    stream: &Stream,
    tg: usize,
    prompt_tokens: usize,
    floor_us: f64,
) -> Result<Value, String> {
    let prompt = stream.ids(prompt_tokens);
    check_tokens(&prompt, bundle.config.vocab, "profile tg prompt")?;
    // Warm + record the greedy ids, then the unprofiled replay wall.
    let g = greedy_run(bundle, gpu, &prompt, tg)?;
    device_loop(bundle, gpu, &prompt, &g.ids)?;
    let wall = device_loop(bundle, gpu, &prompt, &g.ids)?;
    // Head launches of a single-row forward (state is at prompt + tg).
    let head = head_signature(bundle, gpu)?;
    // Legacy timers over the replay.
    cold(bundle, gpu)?;
    bundle.prefill(gpu, &prompt)?;
    sync(gpu)?;
    rdna_compute::profile::start();
    let mut r = Ok(());
    for &t in &g.ids {
        r = r.and_then(|_| bundle.decode_step(gpu, t));
    }
    let r = r.and_then(|_| sync(gpu));
    let legacy = rdna_compute::profile::stop().unwrap_or_default();
    r?;
    let legacy_ms: f64 = legacy.iter().map(|e| e.time_us).sum::<f64>() / 1e3;
    // Launch-level replay, one collector window per step so the head tags per forward.
    cold(bundle, gpu)?;
    bundle.prefill(gpu, &prompt)?;
    sync(gpu)?;
    let mut table = KernelTable::new();
    let mut profiled = 0.0f64;
    for &t in &g.ids {
        rdna_compute::profile::start_launches();
        let t0 = Instant::now();
        let r = bundle.decode_step(gpu, t).and_then(|_| sync(gpu));
        profiled += t0.elapsed().as_secs_f64();
        let entries = rdna_compute::profile::stop_launches().unwrap_or_default();
        r?;
        absorb(&mut table, entries, &head);
    }
    let mut rep = section_report(
        &format!("tg{tg}"),
        wall * 1e3,
        profiled * 1e3,
        &table,
        legacy_ms,
        tg as f64,
        "token",
        head.len(),
        floor_us,
    );
    rep["tokens"] = json!(tg);
    rep["prompt_tokens"] = json!(prompt_tokens);
    rep["generated_ids_md5"] = json!(ids_md5(&g.ids));
    rep["tok_s_unprofiled_device_loop"] = json!(tg as f64 / wall.max(1e-12));
    Ok(rep)
}

/// CPU-only: re-bucket the kernel tables of a saved `--profile` result with the
/// current `bucket_of` rules (head rows stay `head`).
fn rebucket(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))?;
    let result = doc.get("result").unwrap_or(&doc);
    let secs = result
        .get("profile")
        .and_then(|p| p.get("sections"))
        .and_then(Value::as_array)
        .ok_or("no result.profile.sections in the JSON")?;
    let mut out = Vec::new();
    let mut unmatched = std::collections::BTreeSet::new();
    for s in secs {
        let mut table = KernelTable::new();
        for k in s
            .get("kernels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let name = k
                .get("name")
                .and_then(Value::as_str)
                .ok_or("kernel row lacks name")?;
            let head = k.get("head").and_then(Value::as_bool).unwrap_or(false);
            let calls = k.get("calls").and_then(Value::as_u64).unwrap_or(0);
            let us = k.get("total_us").and_then(Value::as_f64).unwrap_or(0.0);
            table.insert((name.to_string(), head), (calls, us));
            if !head && bucket_of(name) == "other" {
                unmatched.insert(name.to_string());
            }
        }
        let f = |key: &str| s.get(key).and_then(Value::as_f64).unwrap_or(0.0);
        out.push(section_report(
            s.get("section").and_then(Value::as_str).unwrap_or("?"),
            f("wall_unprofiled_ms"),
            f("wall_profiled_ms"),
            &table,
            f("legacy_begin_timer_sum_ms"),
            f("units").max(1.0),
            s.get("unit").and_then(Value::as_str).unwrap_or("token"),
            f("head_launches_per_forward") as usize,
            f("launch_span_floor_us"),
        ));
    }
    Ok(json!({
        "pass": true,
        "source": path,
        "sections": out,
        "unclassified_kernels": unmatched,
    }))
}

// ───────────────────────── entry point ─────────────────────────

fn mode_name(m: PrefillMode) -> &'static str {
    match m {
        PrefillMode::Serial => "serial",
        PrefillMode::Batched => "batched",
    }
}

pub fn run(_fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    if let Some(p) = opts.get("bucket-json") {
        return rebucket(p);
    }
    let model = opts.get("model").ok_or("bench: missing --model")?.clone();
    let pps: Vec<usize> = opts
        .get("pp")
        .ok_or("bench: missing --pp")?
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .map_err(|e| format!("--pp `{t}`: {e}"))
        })
        .collect::<Result<_, _>>()?;
    let tg = opt_usize(opts, "tg", 128)?;
    let reps = opt_usize(opts, "reps", 3)?;
    let warmup = opt_usize(opts, "warmup", 1)?;
    let tg_prompt = opt_usize(opts, "tg-prompt", 32)?;
    if reps == 0 {
        return Err("--reps must be >= 1".into());
    }
    if tg > 0 && tg_prompt == 0 {
        return Err("--tg-prompt must be >= 1".into());
    }
    let profile = opts.contains_key("profile");
    let mode = parse_prefill(opts)?;
    let prompt_file: PathBuf = opts
        .get("prompt-file")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PROMPT));
    let text = std::fs::read_to_string(&prompt_file)
        .map_err(|e| format!("read prompt {}: {e}", prompt_file.display()))?;
    let prompt_md5 = md5_hex(text.as_bytes());

    let model_id = model_identity(
        &model,
        opts.get("model-md5").map(String::as_str) == Some("skip"),
    )?;
    let hfq = HfqFile::open(Path::new(&model)).map_err(|e| format!("open {model}: {e}"))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer from HFQ metadata: {e:?}"))?;
    let base = tokenizer.encode(&text);
    if base.is_empty() {
        return Err(format!("{} tokenized to zero ids", prompt_file.display()));
    }
    let stream = Stream { base };

    // Launch-level timing observes funnel launches; a replayed hipGraph has none.
    // Profiling therefore pins the eager decode route (recorded in `controls.env`).
    if profile && std::env::var_os("HIPFIRE_GRAPH").is_none() {
        std::env::set_var("HIPFIRE_GRAPH", "0");
    }
    let max_pp = pps.iter().copied().max().unwrap_or(0);
    let need = max_pp.max(if tg > 0 { tg_prompt + tg } else { 0 });
    let max_seq = match opts.get("max-seq") {
        Some(s) => s
            .parse::<usize>()
            .map_err(|e| format!("--max-seq `{s}`: {e}"))?,
        None => (need + 64).max(512),
    };
    if max_seq < need {
        return Err(format!(
            "--max-seq {max_seq} < the {need} rows the run needs"
        ));
    }

    // Raw request; the resolved per-site effective tiers are reported from the loaded bundle.
    let precision = hipfire_config::developer_var(hipfire_arch_raon::precision::PRECISION_ENV)
        .unwrap_or_else(|_| "default".to_string());
    let mut gpu = Gpu::init().map_err(|e| format!("bench: gpu init: {e:?}"))?;
    let pci = gpu.hip.device_pci_bus_id(gpu.device_id).unwrap_or_default();
    let gpu_json = json!({
        "arch": gpu.arch,
        "device_id": gpu.device_id,
        "pci_bus_id": pci,
        "clocks_at_start": clocks_json(&pci),
    });
    let mut bundle = load_bundle(&mut gpu, ModelSource::Hfq(hfq), &model, max_seq)?;
    // `--mirror-budget-rows N`: force the decompressed-route mirror budget so the absorbed route
    // serves the rows beyond it (emulates a smaller card; recorded in the result JSON).
    let forced_budget = match opts.get("mirror-budget-rows") {
        Some(v) => {
            let rows = v
                .parse::<usize>()
                .map_err(|e| format!("--mirror-budget-rows `{v}`: {e}"))?;
            bundle.state.set_mirror_budget_rows(rows);
            Some(rows)
        }
        None => None,
    };
    eprintln!(
        "bench: mirror budget {} rows",
        bundle.state.mirror_budget_rows()
    );
    let result = (|| -> Result<Value, String> {
        bundle.set_prefill_mode(mode);
        let mut pass = true;
        let mut pp_rows = Vec::new();
        if !profile {
            for &n in &pps {
                match bench_pp(&mut bundle, &mut gpu, &stream, n, reps, warmup, &pci) {
                    Ok(v) => {
                        eprintln!(
                            "bench: pp{n}: {:.1} tok/s mean ({} reps)",
                            v["stats"]["mean"].as_f64().unwrap_or(0.0),
                            reps
                        );
                        pass &= v["logits_finite"].as_bool().unwrap_or(false);
                        pp_rows.push(v);
                    }
                    Err(e) => {
                        eprintln!("bench: pp{n}: ERROR {e}");
                        pass = false;
                        pp_rows.push(json!({"n": n, "error": e}));
                    }
                }
            }
        }
        let mut tg_row = Value::Null;
        if tg > 0 && !profile {
            match bench_tg(
                &mut bundle,
                &mut gpu,
                &stream,
                tg,
                tg_prompt,
                reps,
                warmup,
                &pci,
            ) {
                Ok(v) => {
                    eprintln!(
                        "bench: tg{tg}: greedy {:.2} tok/s mean, device_loop {:.2} tok/s mean",
                        v["greedy"]["stats"]["mean"].as_f64().unwrap_or(0.0),
                        v["device_loop"]["stats"]["mean"].as_f64().unwrap_or(0.0)
                    );
                    pass &= v["logits_finite"].as_bool().unwrap_or(false);
                    tg_row = v;
                }
                Err(e) => {
                    eprintln!("bench: tg{tg}: ERROR {e}");
                    pass = false;
                    tg_row = json!({"tokens": tg, "error": e});
                }
            }
        }
        let mut profile_json = Value::Null;
        if profile {
            let floor_us = launch_floor_us(&mut gpu)?;
            let mut sections = Vec::new();
            for &n in &pps {
                match profile_pp(&mut bundle, &mut gpu, &stream, n, floor_us) {
                    Ok(v) => sections.push(v),
                    Err(e) => {
                        eprintln!("bench: profile pp{n}: ERROR {e}");
                        pass = false;
                        sections.push(json!({"section": format!("pp{n}"), "error": e}));
                    }
                }
            }
            if tg > 0 {
                match profile_tg(&mut bundle, &mut gpu, &stream, tg, tg_prompt, floor_us) {
                    Ok(v) => sections.push(v),
                    Err(e) => {
                        eprintln!("bench: profile tg{tg}: ERROR {e}");
                        pass = false;
                        sections.push(json!({"section": format!("tg{tg}"), "error": e}));
                    }
                }
            }
            for s in &sections {
                print_section(s);
            }
            profile_json = json!({ "sections": sections });
        }
        Ok(json!({
            "pass": pass,
            "protocol": PROTOCOL,
            "timing_class": if profile { "profile (inflated wall; not a throughput claim)" } else { "throughput" },
            "reps": reps,
            "warmup": warmup,
            "binary": binary_identity(),
            "model": model_id,
            "gpu": gpu_json,
            "controls": {
                "prefill_mode": mode_name(bundle.prefill_mode()),
                "prefill_precision": precision,
                "prefill_precision_effective": bundle
                    .precision_plan()
                    .sites
                    .iter()
                    .map(|(site, r)| (site.name().to_string(), r.effective.name()))
                    .collect::<BTreeMap<_, _>>(),
                "decode_route": bundle.decode_mode(),
                "max_seq": max_seq,
                "mirror_budget_rows": bundle.state.mirror_budget_rows(),
                "mirror_budget_forced": forced_budget,
                "env": env_controls(),
            },
            "prompt": {
                "file": prompt_file.display().to_string(),
                "md5": prompt_md5,
                "base_tokens": stream.base.len(),
                "repeat": "cyclic to N ids",
                "tokenizer": "HFQ metadata encode (no template, no BOS)",
            },
            "pp": pp_rows,
            "tg": tg_row,
            "profile": profile_json,
        }))
    })();
    bundle.free_all(&mut gpu);
    result
}

/// Human summary of one profiled section on stderr.
fn print_section(s: &Value) {
    if let Some(e) = s.get("error") {
        eprintln!("profile {}: ERROR {e}", s["section"]);
        return;
    }
    eprintln!(
        "profile {}: wall {:.2} ms (profiled {:.2} ms) kernel_sum {:.2} ms residual {:.2} ms ({:.1}%) launches {}",
        s["section"].as_str().unwrap_or("?"),
        s["wall_unprofiled_ms"].as_f64().unwrap_or(0.0),
        s["wall_profiled_ms"].as_f64().unwrap_or(0.0),
        s["kernel_sum_ms"].as_f64().unwrap_or(0.0),
        s["residual_ms"].as_f64().unwrap_or(0.0),
        s["residual_frac_of_wall"].as_f64().unwrap_or(0.0) * 100.0,
        s["launches"],
    );
    for b in s["buckets"].as_array().into_iter().flatten() {
        eprintln!(
            "  {:<20} {:>9.3} ms corr {:>6.1}% of wall | raw {:>9.3} ms {:>5.1}% of spans | {:>6} launches",
            b["bucket"].as_str().unwrap_or("?"),
            b["ms"].as_f64().unwrap_or(0.0),
            b["frac_of_wall"].as_f64().unwrap_or(0.0) * 100.0,
            b["raw_span_ms"].as_f64().unwrap_or(0.0),
            b["frac_of_raw_span_sum"].as_f64().unwrap_or(0.0) * 100.0,
            b["launches"],
        );
    }
}
