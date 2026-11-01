// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! fn_ep_wide_gather_oracle — lever-A step A2 (AB-DESIGN 1.3): the EP row-gather
//! capacity at 32 rows, on a real EP mesh and real artifact experts.
//!
//! Needs `HIPFIRE_QWEN4_EP_WIDE_VERIFY=1` (the 32-row gather buffers, 320-slot
//! transport and 320-slot owner array) unless `--load-only` (flag-off baseline:
//! prints the free VRAM after load and exits).
//!
//! One probe = one trunk MoE layer called on every rank over the same F32 input
//! rows (`Qwen4Bundle::ep_moe_layer_probe`: the lane step's gathered MoE
//! invocation, sealed per group of rows). Arms per layer and row count `N`:
//!
//! * `ref`: `ceil(N / 8)` groups of at most 8 rows, verify scope closed (what the
//!   4-segment 8-row verify runs today);
//! * `wide`: ONE group of `N` rows, verify scope `N` open (the single-pass shape).
//!
//! Every output byte of `wide` must equal `ref` (rank agreement is enforced by
//! the fan-out itself: a divergence between ranks fails the probe). Balanced
//! (host-mapped, per-call owner assignment) and resident layers are both
//! covered and reported separately.
//!
//! `--seeds 0` skips the byte oracle (soak-only job).
//!
//! `--soak-secs S`: peer-skew soak. Until the deadline, random layer, random row
//! count 9..=32, one rank (random, alternating) sleeps a random 0..=`--skew-ms`
//! before its call; the wide arm is compared with the ref arm each iteration; a
//! watchdog aborts the process (exit 3) when one call exceeds `--hang-secs`.
//!
//! ```text
//! HIPFIRE_QWEN4_EP_WIDE_VERIFY=1 fn_ep_wide_gather_oracle --model M --max-seq 4096 \
//!     [--layers 0,1,..|all] [--rows 9,12,16,17,24,31,32] [--seeds 3] \
//!     [--soak-secs 600] [--skew-ms 40] [--hang-secs 120] [--reserve-lanes 8] --out receipt.json
//! ```
//!
//! Exit: 0 all arms byte-equal (and soak finished), 1 mismatch, 2 usage/runtime
//! error, 3 hang.

use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

struct Args {
    model: String,
    max_seq: usize,
    layers: Option<Vec<usize>>,
    rows: Vec<usize>,
    seeds: usize,
    soak_secs: u64,
    skew_ms: u64,
    hang_secs: u64,
    reserve_lanes: usize,
    load_only: bool,
    out: Option<String>,
}

fn parse_list(value: &str) -> R<Vec<usize>> {
    value
        .split(',')
        .map(|part| part.trim().parse::<usize>().map_err(e))
        .collect()
}

fn parse_args() -> R<Args> {
    let mut args = Args {
        model: String::new(),
        max_seq: 4096,
        layers: None,
        rows: vec![9, 12, 16, 17, 24, 31, 32],
        seeds: 3,
        soak_secs: 0,
        skew_ms: 40,
        hang_secs: 120,
        reserve_lanes: 8,
        load_only: false,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--model" => args.model = value("--model")?,
            "--max-seq" => args.max_seq = value("--max-seq")?.parse().map_err(e)?,
            "--layers" => {
                let v = value("--layers")?;
                args.layers = if v == "all" {
                    None
                } else {
                    Some(parse_list(&v)?)
                };
            }
            "--rows" => args.rows = parse_list(&value("--rows")?)?,
            "--seeds" => args.seeds = value("--seeds")?.parse().map_err(e)?,
            "--soak-secs" => args.soak_secs = value("--soak-secs")?.parse().map_err(e)?,
            "--skew-ms" => args.skew_ms = value("--skew-ms")?.parse().map_err(e)?,
            "--hang-secs" => args.hang_secs = value("--hang-secs")?.parse().map_err(e)?,
            "--reserve-lanes" => {
                args.reserve_lanes = value("--reserve-lanes")?.parse().map_err(e)?
            }
            "--load-only" => args.load_only = true,
            "--out" => args.out = Some(value("--out")?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if args.model.is_empty() {
        return Err("--model is required".into());
    }
    if args.rows.iter().any(|&n| n == 0 || n > 32) {
        return Err("--rows values must be 1..=32".into());
    }
    Ok(args)
}

/// xorshift64*; deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Roughly standard normal (Irwin-Hall, 4 terms, rescaled).
    fn normal(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.uniform()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

fn input_rows(seed: u64, rows: usize, hidden: usize) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    (0..rows * hidden).map(|_| rng.normal()).collect()
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn groups_of_8(n: usize) -> Vec<usize> {
    let mut v = vec![8; n / 8];
    if n % 8 != 0 {
        v.push(n % 8);
    }
    v
}

fn load_with_lane_reservation(
    model: &str,
    max_seq: usize,
    lanes: usize,
    cask: &hipfire_runtime::loader_api::CaskConfig,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
    gpu: &mut Gpu,
) -> R<hipfire_loader::LoadedModel> {
    use hipfire_loader::admission::{
        admit_source_with_options, qwen4_kv_adaptive_requested, qwen_default_q8_enabled,
        BatchLoadRequest, KvBackendHints, KvBackendRequest, SourceAdmissionOptions,
    };
    let mut admission = admit_source_with_options(
        model,
        1,
        1,
        KvBackendRequest::Automatic,
        None,
        gpu.arch.as_str(),
        None,
        "auto",
        None,
        max_seq,
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
        continuous_batch_size: lanes,
        row_budget: lanes,
        spec_k: None,
        spec_copy: false,
    };
    hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        model,
        max_seq,
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
    )
}

fn main() {
    let code = match parse_args().and_then(run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("fn_ep_wide_gather_oracle: {error}");
            2
        }
    };
    std::process::exit(code);
}

fn run(args: Args) -> R<i32> {
    let started = Instant::now();
    let wide_flag = hipfire_arch_qwen4::gpu_forward::qwen4_ep_gather_rows()
        == hipfire_arch_qwen4::gpu_forward::QWEN4_EP_GATHER_VERIFY_ROWS;
    eprintln!(
        "[fn_ep_wide_gather_oracle] loading {} (wide verify gather {})",
        args.model,
        if wide_flag { "ON" } else { "off" }
    );
    let mut gpu = Gpu::init().map_err(|x| format!("Gpu::init: {x:?}"))?;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let mut model = load_with_lane_reservation(
        &args.model,
        args.max_seq,
        args.reserve_lanes,
        &cask,
        Default::default(),
        &mut gpu,
    )?;
    let bundle = model
        .qwen4_mut()
        .ok_or("loaded model is not a Qwen4 (Flash-Next) bundle")?;
    let ranks = bundle.ranks();
    if bundle.ep_mesh().is_none() {
        return Err("no EP mesh attached (set topology.ep / HIPFIRE_TOPOLOGY_EP=2)".into());
    }
    let hidden = bundle.config.hidden_size;
    let top_k = bundle.config.num_experts_per_tok;
    let layers_total = bundle.config.num_hidden_layers;
    // Free VRAM on every rank after the load (the measured footprint basis).
    let mut free = Vec::new();
    for rank in 0..ranks {
        let (f, t) =
            bundle.with_rank_mut(&mut gpu, rank, |_, g, _| g.device_mem_info().map_err(e))?;
        free.push(json!({"rank": rank, "free_bytes": f, "total_bytes": t}));
    }
    let gather_rows = hipfire_arch_qwen4::gpu_forward::qwen4_ep_gather_rows();
    let capacity_bytes =
        hipfire_arch_qwen4::gpu_forward::ep_gather_capacity_bytes(gather_rows, top_k, hidden);
    let off_bytes = hipfire_arch_qwen4::gpu_forward::ep_gather_capacity_bytes(
        hipfire_arch_qwen4::gpu_forward::QWEN4_EP_GATHER_MAX_ROWS,
        top_k,
        hidden,
    );
    let mut report = json!({
        "tool": "fn_ep_wide_gather_oracle",
        "model": args.model,
        "ranks": ranks,
        "hidden": hidden,
        "top_k": top_k,
        "gather_rows": gather_rows,
        "gather_slots": gather_rows * top_k,
        "gather_capacity_bytes_per_rank": capacity_bytes,
        "gather_capacity_bytes_flag_off": off_bytes,
        "gather_capacity_extra_bytes": capacity_bytes - off_bytes,
        "free_after_load": free,
        "wide_verify_flag": wide_flag,
    });
    if args.load_only {
        write_report(&args, &report)?;
        println!("{}", serde_json::to_string_pretty(&report).map_err(e)?);
        return Ok(0);
    }
    if !wide_flag {
        return Err("HIPFIRE_QWEN4_EP_WIDE_VERIFY=1 is required (or --load-only)".into());
    }
    let _ = DType::F32;

    // ---- byte oracle -----------------------------------------------------
    let layers: Vec<usize> = args
        .layers
        .clone()
        .unwrap_or_else(|| (0..layers_total).collect());
    let mut ordinal = 0usize;
    let mut cases = 0usize;
    let mut mismatches: Vec<Value> = Vec::new();
    let mut balanced_cases = 0usize;
    let mut resident_cases = 0usize;
    let mut balanced_layers = Vec::new();
    for &layer in layers.iter().filter(|_| args.seeds > 0) {
        for &n in &args.rows {
            for seed_ix in 0..args.seeds {
                let seed = 0x9E37_79B9_7F4A_7C15u64
                    ^ ((layer as u64) << 32)
                    ^ ((n as u64) << 16)
                    ^ seed_ix as u64;
                let input = input_rows(seed, n, hidden);
                let ref_groups = groups_of_8(n);
                let (reference, balanced) = bundle
                    .ep_moe_layer_probe(&mut gpu, layer, &input, &ref_groups, ordinal, 0, 0, 0)
                    .map_err(|x| format!("layer {layer} rows {n} ref: {x}"))?;
                ordinal += ref_groups.len();
                let (wide, _) = bundle
                    .ep_moe_layer_probe(&mut gpu, layer, &input, &[n], ordinal, n, 0, 0)
                    .map_err(|x| format!("layer {layer} rows {n} wide: {x}"))?;
                ordinal += 1;
                cases += 1;
                if balanced {
                    balanced_cases += 1;
                    if !balanced_layers.contains(&layer) {
                        balanced_layers.push(layer);
                    }
                } else {
                    resident_cases += 1;
                }
                if let Some(at) = first_diff(&reference, &wide) {
                    eprintln!(
                        "MISMATCH layer {layer} rows {n} seed {seed_ix} balanced {balanced}: first byte {at}"
                    );
                    mismatches.push(json!({
                        "layer": layer, "rows": n, "seed": seed_ix, "balanced": balanced,
                        "first_diff_byte": at,
                        "ref_fnv": format!("{:016x}", fnv(&reference)),
                        "wide_fnv": format!("{:016x}", fnv(&wide)),
                    }));
                }
            }
        }
        eprintln!(
            "[fn_ep_wide_gather_oracle] layer {layer} done: cases {cases} mismatches {}",
            mismatches.len()
        );
    }
    let oracle_ran = args.seeds > 0;
    report["oracle"] = json!({
        "ran": oracle_ran,
        "layers": layers.len(),
        "rows": args.rows,
        "seeds": args.seeds,
        "cases": cases,
        "balanced_cases": balanced_cases,
        "resident_cases": resident_cases,
        "balanced_layers": balanced_layers,
        "mismatches": mismatches,
        "pass": !oracle_ran || (mismatches.is_empty() && cases > 0 && balanced_cases > 0),
    });
    let mut pass = !oracle_ran || (mismatches.is_empty() && cases > 0 && balanced_cases > 0);

    // ---- peer-skew soak --------------------------------------------------
    if args.soak_secs > 0 {
        let heartbeat = Arc::new(AtomicU64::new(0));
        {
            let heartbeat = Arc::clone(&heartbeat);
            let hang = args.hang_secs;
            let t0 = Instant::now();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_secs(5));
                let last = heartbeat.load(Ordering::Relaxed);
                let now = t0.elapsed().as_secs();
                if last != u64::MAX && now.saturating_sub(last) > hang {
                    eprintln!("HANG: no probe completed for {hang}s (last at {last}s)");
                    std::process::exit(3);
                }
            });
        }
        let soak_started = Instant::now();
        let t0 = Instant::now();
        let mut rng = Rng(0xC0FF_EE00_1234_5678);
        let mut iters = 0u64;
        let mut soak_mismatch = 0u64;
        let mut skew_rank0 = 0u64;
        let mut skew_peer = 0u64;
        let mut max_call_ms = 0u128;
        while soak_started.elapsed() < Duration::from_secs(args.soak_secs) {
            let layer = layers[(rng.next() % layers.len() as u64) as usize];
            let n = 9 + (rng.next() % 24) as usize;
            let input = input_rows(rng.next(), n, hidden);
            let delay = rng.next() % (args.skew_ms * 1000 + 1);
            let rank0_lags = iters % 2 == 0;
            let (d0, dp) = if rank0_lags { (delay, 0) } else { (0, delay) };
            if rank0_lags {
                skew_rank0 += 1;
            } else {
                skew_peer += 1;
            }
            let call = Instant::now();
            let ref_groups = groups_of_8(n);
            let (reference, _) = bundle
                .ep_moe_layer_probe(&mut gpu, layer, &input, &ref_groups, ordinal, 0, d0, dp)
                .map_err(|x| format!("soak layer {layer} rows {n} ref: {x}"))?;
            ordinal += ref_groups.len();
            let (wide, _) = bundle
                .ep_moe_layer_probe(&mut gpu, layer, &input, &[n], ordinal, n, dp, d0)
                .map_err(|x| format!("soak layer {layer} rows {n} wide: {x}"))?;
            ordinal += 1;
            max_call_ms = max_call_ms.max(call.elapsed().as_millis());
            if first_diff(&reference, &wide).is_some() {
                soak_mismatch += 1;
                eprintln!("SOAK MISMATCH iter {iters} layer {layer} rows {n}");
            }
            iters += 1;
            heartbeat.store(t0.elapsed().as_secs(), Ordering::Relaxed);
        }
        heartbeat.store(u64::MAX, Ordering::Relaxed);
        report["soak"] = json!({
            "seconds": soak_started.elapsed().as_secs_f64(),
            "iterations": iters,
            "rank0_lag_iterations": skew_rank0,
            "peer_lag_iterations": skew_peer,
            "max_skew_ms": args.skew_ms,
            "max_pair_call_ms": max_call_ms,
            "mismatches": soak_mismatch,
            "hang": false,
            "pass": soak_mismatch == 0 && iters > 0,
        });
        pass &= soak_mismatch == 0 && iters > 0;
    }
    report["pass"] = json!(pass);
    report["wall_secs"] = json!(started.elapsed().as_secs_f64());
    write_report(&args, &report)?;
    println!("{}", serde_json::to_string_pretty(&report).map_err(e)?);
    Ok(if pass { 0 } else { 1 })
}

fn write_report(args: &Args, report: &Value) -> R<()> {
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_string_pretty(report).map_err(e)?).map_err(e)?;
    }
    Ok(())
}
