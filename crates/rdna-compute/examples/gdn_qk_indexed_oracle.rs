// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness + timing oracle for the indexed-Q/K sequential Q8 GatedDeltaNet
//! prefill (`HIPFIRE_GDN_QK_INDEXED`, default ON; `0`/`false` kills it) against today's
//! `repeat_interleave_qk_f32_batched` + `gated_delta_net_q8_batch_seq` pair.
//!
//! Shape: 16 key heads, 32 value heads (qk_head_ratio 2), head_dim 128,
//! T in {1, 17, 512, 4096}, Q8 state {zero, nonzero}. Inputs (raw Q/K
//! `[T, 16*128]`, V `[T, 32*128]`, log-decay gate / beta `[T, 32]`) and the
//! nonzero Q8 state (i8 codes + f32 scales) are deterministic splitmix64 data.
//! The zero state is all-zero codes AND scales (a fresh recurrent state).
//!
//! Arms, per (T, state):
//! - A (baseline): `repeat_interleave_qk_f32_batched` raw -> `[T, 32, 128]`,
//!   then `gated_delta_net_q8_batch_seq` (canonical, ratio 1) on the repeated
//!   buffers;
//! - B (indexed): `gated_delta_net_q8_batch_seq_indexed` on the RAW Q/K with
//!   `qk_head_ratio = 2` (no repeat launch);
//! - C (ratio-1 check): `gated_delta_net_q8_batch_seq_indexed` with
//!   `qk_head_ratio = 1` on A's repeated buffers.
//!
//! Every arm starts from byte-identical state / scales (re-uploaded outside
//! any timed region) and an output buffer filled with a NaN sentinel, and every
//! launch is preceded by `restore_gdn_requant_frame_checkpoint(FRAME0)` so the
//! stochastic-rounding frames match; the frame counter after each arm must be
//! `FRAME0 + T`. B and C are compared against A as raw bytes of the downloaded
//! output, Q8 state and scales (EF is not used: `ef_residual = None`, which is
//! the plain stochastic-rounding path this flag targets). B's raw Q/K inputs
//! are re-downloaded and must be unchanged. Timing: hipEvent median of 10
//! iterations of repeat+kernel (A) vs the indexed kernel (B); each iteration
//! resets state and frame OUTSIDE the event pair, arms interleaved.
//!
//! Requant cadence (`HIPFIRE_DN_REQUANT_PER_TOKEN`, the `fast` vs `per-token`
//! kernel the sequential entry picks) is process-wide: run once per mode,
//! e.g. `HIPFIRE_DN_REQUANT_PER_TOKEN=1`. The reported `mode` is read through
//! the same `hipfire_config::developer_var` predicate the dispatcher uses.
//! `--mode fast|per-token` asserts the process is in that mode (exit 2 else).
//!
//! Exactly one JSON object is printed on stdout (progress goes to stderr).
//! Exit status is 1 if any case mismatches, 2 on usage error.
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab,deltanet \
//!     --example gdn_qk_indexed_oracle -- [--plan] [--mode fast|per-token] [--tokens 1,17]
//! ```
//! `--plan` prints the matrix JSON without initializing the GPU.

const NK: usize = 16;
const NV: usize = 32;
const RATIO: usize = NV / NK;
const HD: usize = 128;
const TOKENS: [usize; 4] = [1, 17, 512, 4096];
const STATES: [&str; 2] = ["zero", "nonzero"];
const TIMING_ITERS: usize = 10;
/// Fixed stochastic-rounding frame base every launch starts from.
const FRAME0: u32 = 0x1234_5000;
const OUT_SENTINEL: u32 = 0x7fc0_dead;

fn current_mode() -> &'static str {
    // Same truthy predicate as `norm::dn_requant_per_token`.
    let per_token = hipfire_config::developer_var("HIPFIRE_DN_REQUANT_PER_TOKEN")
        .map(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0"
        })
        .unwrap_or(false);
    if per_token {
        "per-token"
    } else {
        "fast"
    }
}

struct Args {
    plan: bool,
    mode: Option<String>,
    tokens: Vec<usize>,
}

fn parse_args() -> Args {
    let mut args = Args {
        plan: false,
        mode: None,
        tokens: TOKENS.to_vec(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => args.plan = true,
            "--mode" => {
                let m = it.next().unwrap_or_default();
                if m != "fast" && m != "per-token" {
                    eprintln!("--mode expects fast|per-token, got {m:?}");
                    std::process::exit(2);
                }
                args.mode = Some(m);
            }
            "--tokens" => {
                let list = it.next().unwrap_or_default();
                let parsed: Option<Vec<usize>> = list
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .ok()
                            .filter(|t| TOKENS.contains(t))
                    })
                    .collect();
                match parsed {
                    Some(v) if !v.is_empty() => args.tokens = v,
                    _ => {
                        eprintln!(
                            "--tokens expects a comma list drawn from {TOKENS:?}, got {list:?}"
                        );
                        std::process::exit(2);
                    }
                }
            }
            other => {
                eprintln!("unknown argument {other:?}; usage: [--plan] [--mode fast|per-token] [--tokens 1,17]");
                std::process::exit(2);
            }
        }
    }
    args
}

fn plan_json(args: &Args) -> serde_json::Value {
    let cases: Vec<_> = args
        .tokens
        .iter()
        .flat_map(|&t| {
            STATES.iter().map(move |&s| {
                serde_json::json!({
                    "tokens": t,
                    "state": s,
                    "n_key_heads": NK,
                    "n_value_heads": NV,
                    "qk_head_ratio": RATIO,
                    "head_dim": HD,
                })
            })
        })
        .collect();
    serde_json::json!({
        "oracle": "gdn_qk_indexed_oracle",
        "plan_only": true,
        "mode": current_mode(),
        "mode_env": "HIPFIRE_DN_REQUANT_PER_TOKEN",
        "frame0": FRAME0,
        "timing": { "iters": TIMING_ITERS, "stat": "median", "clock": "hipEvent" },
        "arms": {
            "baseline": "repeat_interleave_qk_f32_batched + gated_delta_net_q8_batch_seq (ratio 1)",
            "indexed": "gated_delta_net_q8_batch_seq_indexed (raw Q/K, ratio 2)",
            "ratio1_check": "gated_delta_net_q8_batch_seq_indexed (repeated Q/K, ratio 1) vs baseline",
        },
        "compare": ["output f32 bits", "s_q8 bytes", "s_scales bytes", "frame counter advance", "raw q/k unchanged"],
        "ef_residual": false,
        "cases": cases,
    })
}

#[cfg(not(feature = "deltanet"))]
fn main() {
    let args = parse_args();
    if args.plan {
        println!("{}", plan_json(&args));
        return;
    }
    eprintln!("gdn_qk_indexed_oracle requires --features lab,deltanet (use --plan without it)");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
struct Rng(u64);

#[cfg(feature = "deltanet")]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[cfg(feature = "deltanet")]
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

#[cfg(feature = "deltanet")]
struct Host {
    q_raw: Vec<f32>,
    k_raw: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    s_q8: Vec<u8>,
    scales: Vec<f32>,
    out_sentinel: Vec<u8>,
}

#[cfg(feature = "deltanet")]
fn make_host(t: usize, nonzero: bool) -> Host {
    let mut rng = Rng(0xA3B0_0000_0000_0000 ^ t as u64);
    let q_raw = (0..t * NK * HD).map(|_| rng.sym() * 0.2).collect();
    let k_raw = (0..t * NK * HD).map(|_| rng.sym() * 0.2).collect();
    let v = (0..t * NV * HD).map(|_| rng.sym() * 0.5).collect();
    // log-decay < 0, beta in (0, 1)
    let gate = (0..t * NV).map(|_| -(0.01 + rng.unit() * 0.2)).collect();
    let beta = (0..t * NV).map(|_| 0.05 + rng.unit() * 0.9).collect();
    let mut srng = Rng(0x57A7_E000_0000_0000);
    let (s_q8, scales) = if nonzero {
        (
            (0..NV * HD * HD)
                .map(|_| ((srng.next() % 255) as i32 - 127) as i8 as u8)
                .collect(),
            (0..NV * HD).map(|_| 0.0005 + srng.unit() * 0.01).collect(),
        )
    } else {
        (vec![0u8; NV * HD * HD], vec![0f32; NV * HD])
    };
    let out_sentinel = std::iter::repeat(OUT_SENTINEL.to_ne_bytes())
        .take(t * NV * HD)
        .flatten()
        .collect();
    Host {
        q_raw,
        k_raw,
        v,
        gate,
        beta,
        s_q8,
        scales,
        out_sentinel,
    }
}

#[cfg(feature = "deltanet")]
struct Dev {
    q_raw: rdna_compute::GpuTensor,
    k_raw: rdna_compute::GpuTensor,
    q_rep: rdna_compute::GpuTensor,
    k_rep: rdna_compute::GpuTensor,
    v: rdna_compute::GpuTensor,
    gate: rdna_compute::GpuTensor,
    beta: rdna_compute::GpuTensor,
    s_q8: rdna_compute::GpuTensor,
    scales: rdna_compute::GpuTensor,
    out: rdna_compute::GpuTensor,
}

#[cfg(feature = "deltanet")]
fn make_dev(gpu: &mut rdna_compute::Gpu, h: &Host, t: usize) -> Dev {
    Dev {
        q_raw: gpu.upload_f32(&h.q_raw, &[h.q_raw.len()]).expect("q_raw"),
        k_raw: gpu.upload_f32(&h.k_raw, &[h.k_raw.len()]).expect("k_raw"),
        q_rep: gpu
            .upload_f32(
                &vec![f32::from_bits(OUT_SENTINEL); t * NV * HD],
                &[t * NV * HD],
            )
            .expect("q_rep"),
        k_rep: gpu
            .upload_f32(
                &vec![f32::from_bits(OUT_SENTINEL); t * NV * HD],
                &[t * NV * HD],
            )
            .expect("k_rep"),
        v: gpu.upload_f32(&h.v, &[h.v.len()]).expect("v"),
        gate: gpu.upload_f32(&h.gate, &[h.gate.len()]).expect("gate"),
        beta: gpu.upload_f32(&h.beta, &[h.beta.len()]).expect("beta"),
        s_q8: gpu.upload_raw(&h.s_q8, &[h.s_q8.len()]).expect("s_q8"),
        scales: gpu
            .upload_f32(&h.scales, &[h.scales.len()])
            .expect("scales"),
        out: gpu
            .upload_raw(&h.out_sentinel, &[h.out_sentinel.len()])
            .expect("out"),
    }
}

#[cfg(feature = "deltanet")]
fn free_dev(gpu: &mut rdna_compute::Gpu, d: Dev) {
    for t in [
        d.q_raw, d.k_raw, d.q_rep, d.k_rep, d.v, d.gate, d.beta, d.s_q8, d.scales, d.out,
    ] {
        gpu.free_tensor(t).expect("free");
    }
}

/// Re-initialise state, scales and the output sentinel to the case's initial
/// bytes and restore the frame counter. Always called outside timed regions.
#[cfg(feature = "deltanet")]
fn reset(gpu: &rdna_compute::Gpu, d: &Dev, h: &Host) {
    gpu.bind_thread().expect("bind");
    gpu.hip.device_synchronize().expect("sync before reset");
    gpu.hip
        .memcpy_htod(&d.s_q8.buf, &h.s_q8)
        .expect("reset s_q8");
    gpu.hip
        .memcpy_htod(&d.scales.buf, &f32_bytes(&h.scales))
        .expect("reset scales");
    gpu.hip
        .memcpy_htod(&d.out.buf, &h.out_sentinel)
        .expect("reset out");
    gpu.hip.device_synchronize().expect("sync after reset");
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(FRAME0);
}

#[cfg(feature = "deltanet")]
struct Snap {
    out: Vec<u8>,
    s_q8: Vec<u8>,
    scales: Vec<u8>,
    frame_after: u32,
}

#[cfg(feature = "deltanet")]
fn snapshot(gpu: &rdna_compute::Gpu, d: &Dev) -> Snap {
    gpu.hip.device_synchronize().expect("sync before download");
    let frame_after = rdna_compute::norm::gdn_requant_frame_checkpoint();
    Snap {
        out: gpu.download_raw_bytes(&d.out).expect("download out"),
        s_q8: gpu.download_raw_bytes(&d.s_q8).expect("download s_q8"),
        scales: gpu.download_raw_bytes(&d.scales).expect("download scales"),
        frame_after,
    }
}

/// Number of differing 32-bit words (floats compared by bits) and the first
/// differing word index.
#[cfg(feature = "deltanet")]
fn word_mismatches(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = a.len().abs_diff(b.len()) / 4;
    let mut first = None;
    for (i, (x, y)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

#[cfg(feature = "deltanet")]
fn byte_mismatches(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

#[cfg(feature = "deltanet")]
fn compare(base: &Snap, other: &Snap) -> serde_json::Value {
    let (out_mis, out_first) = word_mismatches(&base.out, &other.out);
    let state_mis = byte_mismatches(&base.s_q8, &other.s_q8);
    let scale_mis = byte_mismatches(&base.scales, &other.scales);
    serde_json::json!({
        "out_bit_mismatches": out_mis,
        "out_first_mismatch_index": out_first,
        "state_byte_mismatches": state_mis,
        "scale_byte_mismatches": scale_mis,
        "equal": out_mis == 0 && state_mis == 0 && scale_mis == 0,
    })
}

#[cfg(feature = "deltanet")]
fn run_baseline(gpu: &mut rdna_compute::Gpu, d: &Dev, t: usize) {
    gpu.repeat_interleave_qk_f32_batched(&d.q_raw, &d.k_raw, &d.q_rep, &d.k_rep, NK, RATIO, HD, t)
        .expect("repeat_interleave_qk_f32_batched");
    gpu.gated_delta_net_q8_batch_seq(
        &d.q_rep, &d.k_rep, &d.v, &d.gate, &d.beta, &d.s_q8, &d.scales, &d.out, t, NV, HD, None,
    )
    .expect("gated_delta_net_q8_batch_seq");
}

#[cfg(feature = "deltanet")]
fn run_indexed(gpu: &mut rdna_compute::Gpu, d: &Dev, t: usize) {
    gpu.gated_delta_net_q8_batch_seq_indexed(
        &d.q_raw, &d.k_raw, &d.v, &d.gate, &d.beta, &d.s_q8, &d.scales, &d.out, t, NV, HD, None,
        RATIO,
    )
    .expect("gated_delta_net_q8_batch_seq_indexed ratio 2");
}

/// Ratio-1 through the indexed entry, on the already-repeated buffers.
#[cfg(feature = "deltanet")]
fn run_indexed_ratio1(gpu: &mut rdna_compute::Gpu, d: &Dev, t: usize) {
    gpu.gated_delta_net_q8_batch_seq_indexed(
        &d.q_rep, &d.k_rep, &d.v, &d.gate, &d.beta, &d.s_q8, &d.scales, &d.out, t, NV, HD, None, 1,
    )
    .expect("gated_delta_net_q8_batch_seq_indexed ratio 1");
}

#[cfg(feature = "deltanet")]
fn event_us(gpu: &mut rdna_compute::Gpu, launch: impl FnOnce(&mut rdna_compute::Gpu)) -> f64 {
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .expect("record start");
    launch(gpu);
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("synchronize stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    gpu.hip.event_destroy(start).expect("destroy start");
    gpu.hip.event_destroy(stop).expect("destroy stop");
    ms as f64 * 1_000.0
}

#[cfg(feature = "deltanet")]
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) * 0.5
}

#[cfg(feature = "deltanet")]
fn run_case(
    gpu: &mut rdna_compute::Gpu,
    t: usize,
    state: &str,
    mode: &str,
) -> (serde_json::Value, bool) {
    let nonzero = state == "nonzero";
    let h = make_host(t, nonzero);
    let d = make_dev(gpu, &h, t);
    let want_frame = FRAME0.wrapping_add(t as u32);

    // A: baseline (also fills q_rep / k_rep for arm C).
    reset(gpu, &d, &h);
    run_baseline(gpu, &d, t);
    let a = snapshot(gpu, &d);
    // B: indexed ratio 2 on raw Q/K.
    reset(gpu, &d, &h);
    run_indexed(gpu, &d, t);
    let b = snapshot(gpu, &d);
    let raw_q_unchanged = gpu.download_raw_bytes(&d.q_raw).expect("q_raw") == f32_bytes(&h.q_raw);
    let raw_k_unchanged = gpu.download_raw_bytes(&d.k_raw).expect("k_raw") == f32_bytes(&h.k_raw);
    // C: indexed ratio 1 on the repeated buffers.
    reset(gpu, &d, &h);
    run_indexed_ratio1(gpu, &d, t);
    let c = snapshot(gpu, &d);

    let nonfinite = a
        .out
        .chunks_exact(4)
        .filter(|w| !f32::from_ne_bytes([w[0], w[1], w[2], w[3]]).is_finite())
        .count();
    let out_nonzero = a
        .out
        .chunks_exact(4)
        .filter(|w| w.iter().any(|&x| x != 0))
        .count();
    let state_changed = a.s_q8 != h.s_q8 || a.scales != f32_bytes(&h.scales);

    let cmp_b = compare(&a, &b);
    let cmp_c = compare(&a, &c);
    let frames_ok =
        a.frame_after == want_frame && b.frame_after == want_frame && c.frame_after == want_frame;
    let pass = cmp_b["equal"] == true
        && cmp_c["equal"] == true
        && frames_ok
        && raw_q_unchanged
        && raw_k_unchanged
        && nonfinite == 0
        && out_nonzero > 0
        && state_changed;

    // Timing: arms interleaved, state + frame reset outside the event pair.
    let mut base_us = Vec::with_capacity(TIMING_ITERS);
    let mut idx_us = Vec::with_capacity(TIMING_ITERS);
    for _ in 0..TIMING_ITERS {
        reset(gpu, &d, &h);
        base_us.push(event_us(gpu, |g| run_baseline(g, &d, t)));
        reset(gpu, &d, &h);
        idx_us.push(event_us(gpu, |g| run_indexed(g, &d, t)));
    }
    let base_med = median(&base_us);
    let idx_med = median(&idx_us);

    free_dev(gpu, d);
    let json = serde_json::json!({
        "tokens": t,
        "state": state,
        "mode": mode,
        "pass": pass,
        "indexed_vs_baseline": cmp_b,
        "ratio1_indexed_vs_canonical": cmp_c,
        "frame_after": { "expected": want_frame, "baseline": a.frame_after, "indexed": b.frame_after, "ratio1": c.frame_after, "ok": frames_ok },
        "raw_qk_unchanged": raw_q_unchanged && raw_k_unchanged,
        "sanity": { "out_nonfinite": nonfinite, "out_nonzero_words": out_nonzero, "state_changed": state_changed },
        "timing_us": {
            "baseline_repeat_plus_kernel_median": base_med,
            "indexed_kernel_median": idx_med,
            "speedup_x": base_med / idx_med,
            "saved_us": base_med - idx_med,
            "baseline_samples": base_us,
            "indexed_samples": idx_us,
        },
    });
    (json, pass)
}

#[cfg(feature = "deltanet")]
fn main() {
    let args = parse_args();
    if args.plan {
        println!("{}", plan_json(&args));
        return;
    }
    let mode = current_mode();
    if let Some(want) = &args.mode {
        if want != mode {
            eprintln!(
                "--mode {want} requested but this process is in mode {mode}; \
                 re-run with HIPFIRE_DN_REQUANT_PER_TOKEN={}",
                if want == "per-token" { "1" } else { "0" }
            );
            std::process::exit(2);
        }
    }
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
    gpu.bind_thread().expect("bind");
    let frame_entry = rdna_compute::norm::gdn_requant_frame_checkpoint();

    let mut cases = Vec::new();
    let mut all_pass = true;
    for &t in &args.tokens {
        for state in STATES {
            eprintln!("case T={t} state={state} mode={mode}");
            let (json, pass) = run_case(&mut gpu, t, state, mode);
            all_pass &= pass;
            cases.push(json);
        }
    }
    rdna_compute::norm::restore_gdn_requant_frame_checkpoint(frame_entry);

    let result = serde_json::json!({
        "oracle": "gdn_qk_indexed_oracle",
        "plan_only": false,
        "arch": gpu.arch,
        "mode": mode,
        "frame0": FRAME0,
        "timing": { "iters": TIMING_ITERS, "stat": "median", "clock": "hipEvent" },
        "n_key_heads": NK,
        "n_value_heads": NV,
        "qk_head_ratio": RATIO,
        "head_dim": HD,
        "pass": all_pass,
        "cases": cases,
    });
    println!("{result}");
    if !all_pass {
        std::process::exit(1);
    }
}
