// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing for the shared-expert prepared-IU4 route (gfx1201,
//! lab): `Gpu::try_gemm_mq4g256v2_shared_router_iu4` plus
//! `Gpu::try_gemm_mq4g256v2_shared_iu4`. Four MQ4G256V2 projections (router,
//! selector, gate, up) share ONE `quantize_int4_mmq_ds128` of the activation.
//!
//! ## Arms
//!
//! * incumbent (the actual A3B generic stage, THREE quantizers):
//!   1. router `Gpu::gemm_mq4g256v2(w_router, x, y_router, 256, k, rows)`
//!      (zero + IU4 ADD, quantize #1),
//!   2. selector `Gpu::gemm_mq4g256v2(w_sel, x, y_sel, 1, k, rows)`
//!      (zero + IU4 ADD, quantize #2), then
//!   3. `Gpu::gemm_gate_up_hfq4g256_mq4v2(w_gate, w_up, x, y_gate, y_up, 512,
//!      512, k, rows)` (gfx12 IU4: quantize #3, then SET consumers).
//! * shared (prepared route, ONE quantizer): the router method
//!   `try_gemm_mq4g256v2_shared_router_iu4(w_router, x, y_router, 256, k,
//!   rows)` quantizes once, runs router zero + ADD and returns an
//!   `Int4MmqPrepared` token; then `try_gemm_mq4g256v2_shared_iu4([(w_sel,
//!   y_sel, 1), (w_gate, y_gate, 512), (w_up, y_up, 512)], x, k, rows,
//!   Some(&token))` consumes it without re-quantizing (3 -> 1 quantizers
//!   in the full stage; the shared-only part is 2 -> 1).
//! * no-router group: `try_gemm_mq4g256v2_shared_iu4(..., None)` quantizes
//!   once itself (2 -> 1 for the shared part). Its three outputs are compared
//!   bytewise against the incumbent snapshot once per case; not timed.
//!
//! The oracle REJECTS a configuration where
//! `Gpu::mq4g256v2_shared_iu4_supported(k, rows)` is false (wrong arch, IU4
//! prefill flag off, A8 prefill active, `rows < 64`, `k % 256 != 0`, ...) with
//! exit code 2 before any JSON is printed, and panics when the router method
//! returns `None` or the group returns `Ok(false)`: it never compares a
//! fallback against the incumbent. The grouped API reads no flag itself; the
//! typed `iu4_prefill_enabled()` flag decides support (default on for gfx1201).
//!
//! ## Cases (6 = 2 K x 3 rows)
//!
//! Each case runs the A3B router and shared-expert projections at hidden `k`
//! against `rows` activation rows:
//!
//! * `shared_gate` m = 512, `shared_up` m = 512, `shared_selector` m = 1,
//!   `router` m = 256. The selector's `m = 1` is OUTPUT CHANNELS, not decode
//!   batch; the incumbent runs it through `gemm_mq4g256v2`, gate/up through
//!   the fused gate/up GEMM, the router through `gemm_mq4g256v2`.
//! * `k = 2048` is the real A3B fixture; `k = 4096` (m still 512) is the
//!   requested stress shape. `rows` = 64, 512, 4096 (batch_size).
//!
//! Source anchors for the confirmed geometry (hipfire-arch-qwen35):
//!
//! * `qwen35/config.rs:165-170`: `num_experts` 256 (router output channels),
//!   `num_experts_per_tok` 8, `moe_intermediate_size` 512,
//!   `shared_expert_intermediate_size` 512.
//! * `qwen35/config.rs:1212-1228` (`moe_fixture`): hidden_size 2048, shared
//!   intermediate 512, 256 experts / top-8.
//! * `qwen35/load.rs:7575-7625`: `shared_expert.gate_proj` / `up_proj` load as
//!   `[smi, config.dim]`; `shared_expert_gate` (selector) loads `[1, config.dim]`.
//!
//! Weights are random but VALID MQ4G256V2: 136 B/group = fp16 `{s0, z0}` and
//! `{s1, z1}` words (two 128-wide halves, finite, mean-centred) + 128 B of
//! packed nibbles (layout read by `kernels/src/gemv_mq4g256v2_xbatch.hip`; the
//! same 136 B stride as `bench_e8_soa_correctness::synth_hfq4g256`). They are
//! uploaded resident via `Gpu::upload_raw`. `x` is finite F32 `[rows, k]`.
//!
//! ## Verification
//!
//! Before every arm launch each `y` is filled with the nonzero sentinel byte
//! `0xA5` (so the router/selector/shared zeroing memsets are exercised and the
//! SET consumers must overwrite every element). Outputs are downloaded as
//! bytes (`Gpu::download_raw_bytes`) and compared bit-for-bit: all four
//! buffers (router + 3 shared) of the prepared route, both right after the
//! first run and again after the whole timing loop (60 runs later), plus the
//! three shared buffers of the no-router call after the first run.
//!
//! ## Timing
//!
//! 10 warmups + 50 reps per arm (incumbent vs prepared route), arms
//! interleaved. Each rep: sentinel memset + `device_synchronize` (untimed),
//! then `Instant::now()`, arm launches (INCLUDING the quantizer(s): 3
//! incumbent, 1 shared), `device_synchronize`, elapsed. Reported as median /
//! p10 / p90 in microseconds (percentiles are nearest-rank on the sorted 50
//! samples: index `round(0.1 * 49)` = 5, `round(0.9 * 49)` = 44; median =
//! mean of the two middle samples).
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab \
//!     --example moe_shared_prepared_iu4_oracle
//! ```
//!
//! stdout = exactly 7 JSON lines: one per case, then the summary
//! `{"summary":true,...,"failures":N}`. Diagnostics go to stderr. Exit code 0
//! when `failures == 0`; 1 when any case failed (after the summary is
//! printed); 2 when the configuration is unsupported.

use half::f16;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::time::Instant;

const WARMUPS: usize = 10;
const REPS: usize = 50;
/// MQ4G256V2 bytes per 256-wide group: 8 B fp16 header + 128 B nibbles.
const GROUP_BYTES: usize = 136;
/// Nonzero fill for every `y` before each launch.
const SENTINEL: u8 = 0xA5;

/// `(k, rows)`: A3B hidden 2048 and the k4096 stress shape, x rows 64/512/4096.
const CASES: [(usize, usize); 6] = [
    (2048, 64),
    (2048, 512),
    (2048, 4096),
    (4096, 64),
    (4096, 512),
    (4096, 4096),
];
/// `(name, m)`: shared gate / up (`smi` = 512), selector (`1` channel) and the
/// A3B router (`num_experts` = 256). Index constants below name the slots.
const PROJECTIONS: [(&str, usize); 4] = [
    ("shared_gate", 512),
    ("shared_up", 512),
    ("shared_selector", 1),
    ("router", 256),
];
const GATE: usize = 0;
const UP: usize = 1;
const SELECTOR: usize = 2;
const ROUTER: usize = 3;
const NPROJ: usize = PROJECTIONS.len();
/// gate, up, selector: the outputs the group helper writes.
const SHARED_OUTPUTS: usize = 3;
type Ys = [GpuTensor; NPROJ];
type Ws = [GpuTensor; NPROJ];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
}

/// Random valid MQ4G256V2 rows: `m` rows of `k / 256` groups of 136 B.
fn synth_mq4g256v2(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            let scale = 0.004 + (rng.next() & 0x3f) as f32 * 2.0e-4;
            let zero = -7.5 * scale + ((rng.next() & 0xf) as f32 - 8.0) * 1.0e-4;
            out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            out.extend_from_slice(&f16::from_f32(zero).to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        }
    }
    assert_eq!(out.len(), m * (k / 256) * GROUP_BYTES);
    out
}

/// Finite F32 activations in `[-1, 1)`.
fn synth_x(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows * k)
        .map(|_| (rng.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0)
        .collect()
}

fn alloc_y(gpu: &mut Gpu, rows: usize, m: usize) -> GpuTensor {
    gpu.alloc_tensor(&[rows, m], DType::F32).expect("alloc y")
}

fn fill_sentinel(gpu: &Gpu, ys: &Ys) {
    for y in ys {
        gpu.hip
            .memset(&y.buf, SENTINEL as i32, y.byte_size())
            .expect("sentinel memset");
    }
}

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().expect("device synchronize");
}

struct Fixture<'a> {
    w: &'a Ws,
    x: &'a GpuTensor,
    k: usize,
    rows: usize,
}

/// Incumbent generic A3B shared stage, THREE quantizers: router GEMM, then
/// selector GEMM, then fused gate/up.
fn run_incumbent(gpu: &mut Gpu, f: &Fixture, ys: &Ys) {
    gpu.gemm_mq4g256v2(
        &f.w[ROUTER],
        f.x,
        &ys[ROUTER],
        PROJECTIONS[ROUTER].1,
        f.k,
        f.rows,
    )
    .expect("incumbent router gemm_mq4g256v2");
    gpu.gemm_mq4g256v2(
        &f.w[SELECTOR],
        f.x,
        &ys[SELECTOR],
        PROJECTIONS[SELECTOR].1,
        f.k,
        f.rows,
    )
    .expect("incumbent selector gemm_mq4g256v2");
    gpu.gemm_gate_up_hfq4g256_mq4v2(
        &f.w[GATE],
        &f.w[UP],
        f.x,
        &ys[GATE],
        &ys[UP],
        PROJECTIONS[GATE].1,
        PROJECTIONS[UP].1,
        f.k,
        f.rows,
    )
    .expect("incumbent gemm_gate_up_hfq4g256_mq4v2");
}

/// Helper contract order is [selector, gate, up]; buffers stay indexed by
/// PROJECTIONS (gate, up, selector, router).
fn shared_projections<'a>(
    f: &'a Fixture,
    ys: &'a Ys,
) -> [(&'a GpuTensor, &'a GpuTensor, usize); 3] {
    [
        (&f.w[SELECTOR], &ys[SELECTOR], PROJECTIONS[SELECTOR].1),
        (&f.w[GATE], &ys[GATE], PROJECTIONS[GATE].1),
        (&f.w[UP], &ys[UP], PROJECTIONS[UP].1),
    ]
}

/// Prepared route, ONE quantizer: the router method quantizes `x` once and
/// returns the token; the group consumes it (`Some(&token)`, no re-quantize).
fn run_shared(gpu: &mut Gpu, f: &Fixture, ys: &Ys) {
    let token = gpu
        .try_gemm_mq4g256v2_shared_router_iu4(
            &f.w[ROUTER],
            f.x,
            &ys[ROUTER],
            PROJECTIONS[ROUTER].1,
            f.k,
            f.rows,
        )
        .expect("try_gemm_mq4g256v2_shared_router_iu4")
        .expect("router arm returned None after the support check passed");
    let ran = gpu
        .try_gemm_mq4g256v2_shared_iu4(shared_projections(f, ys), f.x, f.k, f.rows, Some(&token))
        .expect("try_gemm_mq4g256v2_shared_iu4 (prepared)");
    assert!(
        ran,
        "shared IU4 group returned Ok(false) after the support check passed"
    );
}

/// No-router group call (`None`): quantizes once itself; leaves `ys[ROUTER]`
/// untouched. Compared on the three shared outputs only; not timed.
fn run_shared_no_router(gpu: &mut Gpu, f: &Fixture, ys: &Ys) {
    let ran = gpu
        .try_gemm_mq4g256v2_shared_iu4(shared_projections(f, ys), f.x, f.k, f.rows, None)
        .expect("try_gemm_mq4g256v2_shared_iu4 (no router)");
    assert!(
        ran,
        "shared IU4 group (no router) returned Ok(false) after the support check passed"
    );
}

fn download(gpu: &Gpu, ys: &Ys) -> Vec<Vec<u8>> {
    ys.iter()
        .map(|y| gpu.download_raw_bytes(y).expect("download y"))
        .collect()
}

/// Bytewise compare of the first `n` PROJECTIONS outputs; returns per-output
/// JSON + failure count.
fn compare(inc: &[Vec<u8>], shr: &[Vec<u8>], n: usize) -> (Vec<Value>, usize) {
    let mut failures = 0;
    let mut outs = Vec::new();
    for (i, (name, m)) in PROJECTIONS.iter().enumerate().take(n) {
        let (a, b) = (&inc[i], &shr[i]);
        let first_mismatch = a.iter().zip(b.iter()).position(|(x, y)| x != y);
        let equal = a.len() == b.len() && first_mismatch.is_none();
        if !equal {
            failures += 1;
        }
        outs.push(json!({
            "name": name, "m": m, "bytes": a.len(),
            "equal": equal, "first_mismatch_byte": first_mismatch,
        }));
    }
    (outs, failures)
}

fn stats(mut us: Vec<f64>) -> Value {
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = us.len();
    let pct = |p: f64| us[((n - 1) as f64 * p).round() as usize];
    let median = (us[(n - 1) / 2] + us[n / 2]) / 2.0;
    json!({ "median": median, "p10": pct(0.1), "p90": pct(0.9) })
}

fn run_case(gpu: &mut Gpu, k: usize, rows: usize) -> (Value, usize) {
    let weights: Vec<GpuTensor> = PROJECTIONS
        .iter()
        .enumerate()
        .map(|(i, (_, m))| {
            let bytes = synth_mq4g256v2(*m, k, 0x5eed_0000 + (k as u64) * 16 + i as u64);
            gpu.upload_raw(&bytes, &[bytes.len()])
                .expect("upload weight")
        })
        .collect();
    let w: Ws = weights.try_into().ok().expect("four weights");
    let xh = synth_x(rows, k, 0xa11c_e000 + k as u64 + rows as u64);
    let x = gpu.upload_f32(&xh, &[rows, k]).expect("upload x");
    let alloc_ys = |gpu: &mut Gpu| -> Ys {
        [
            alloc_y(gpu, rows, PROJECTIONS[0].1),
            alloc_y(gpu, rows, PROJECTIONS[1].1),
            alloc_y(gpu, rows, PROJECTIONS[2].1),
            alloc_y(gpu, rows, PROJECTIONS[3].1),
        ]
    };
    let ys_inc = alloc_ys(gpu);
    let ys_shr = alloc_ys(gpu);
    let ys_nr = alloc_ys(gpu);
    let f = Fixture {
        w: &w,
        x: &x,
        k,
        rows,
    };

    // First-run byte verification: router + 3 shared outputs (prepared route),
    // then the no-router group call against the same incumbent snapshot.
    fill_sentinel(gpu, &ys_inc);
    fill_sentinel(gpu, &ys_shr);
    fill_sentinel(gpu, &ys_nr);
    sync(gpu);
    run_incumbent(gpu, &f, &ys_inc);
    run_shared(gpu, &f, &ys_shr);
    run_shared_no_router(gpu, &f, &ys_nr);
    sync(gpu);
    let inc_first = download(gpu, &ys_inc);
    let (_, first_failures) = compare(&inc_first, &download(gpu, &ys_shr), NPROJ);
    let (no_router_outputs, no_router_failures) =
        compare(&inc_first, &download(gpu, &ys_nr), SHARED_OUTPUTS);

    // Timing: interleaved arms, sentinel + sync outside the timed region.
    let mut inc_us = Vec::with_capacity(REPS);
    let mut shr_us = Vec::with_capacity(REPS);
    for rep in 0..WARMUPS + REPS {
        fill_sentinel(gpu, &ys_inc);
        sync(gpu);
        let t = Instant::now();
        run_incumbent(gpu, &f, &ys_inc);
        sync(gpu);
        let inc = t.elapsed().as_secs_f64() * 1e6;

        fill_sentinel(gpu, &ys_shr);
        sync(gpu);
        let t = Instant::now();
        run_shared(gpu, &f, &ys_shr);
        sync(gpu);
        let shr = t.elapsed().as_secs_f64() * 1e6;

        if rep >= WARMUPS {
            inc_us.push(inc);
            shr_us.push(shr);
        }
    }

    // Final-state byte verification (outputs of the last timed reps).
    let (outputs, final_failures) =
        compare(&download(gpu, &ys_inc), &download(gpu, &ys_shr), NPROJ);
    let failures = first_failures + no_router_failures + final_failures;
    let (inc, shr) = (stats(inc_us), stats(shr_us));
    let speedup = inc["median"].as_f64().unwrap() / shr["median"].as_f64().unwrap();
    let line = json!({
        "case": format!("k{k}_rows{rows}"),
        "k": k, "rows": rows,
        "projections": PROJECTIONS.iter().map(|(n, m)| json!({"name": n, "m": m})).collect::<Vec<_>>(),
        "warmups": WARMUPS, "reps": REPS,
        "outputs": outputs,
        "no_router_outputs": no_router_outputs,
        "first_run_failures": first_failures,
        "no_router_failures": no_router_failures,
        "final_run_failures": final_failures,
        "incumbent_us": inc, "shared_us": shr,
        "speedup_median": speedup,
        "pass": failures == 0,
    });
    (line, failures)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");

    // Reject unsupported configurations up front: never compare a fallback.
    let unsupported: Vec<String> = CASES
        .iter()
        .filter(|(k, rows)| !gpu.mq4g256v2_shared_iu4_supported(*k, *rows))
        .map(|(k, rows)| format!("k{k}_rows{rows}"))
        .collect();
    if !unsupported.is_empty() {
        eprintln!(
            "moe_shared_prepared_iu4_oracle: mq4g256v2_shared_iu4_supported is false on arch {} for {} \
             (needs exact gfx1201 WMMA, iu4 prefill enabled, no A8 prefill/FP8 verify scope, \
             eager stream, k % 256 == 0, rows >= 64); refusing to compare a fallback",
            gpu.arch,
            unsupported.join(", ")
        );
        std::process::exit(2);
    }

    let mut failures = 0;
    let mut passed = 0;
    for (k, rows) in CASES {
        let (line, case_failures) = run_case(&mut gpu, k, rows);
        println!("{line}");
        if case_failures == 0 {
            passed += 1;
        }
        failures += case_failures;
    }
    println!(
        "{}",
        json!({
            "summary": true,
            "oracle": "moe_shared_prepared_iu4",
            "arch": gpu.arch,
            "cases": CASES.len(),
            "passed": passed,
            "warmups": WARMUPS,
            "reps": REPS,
            "failures": failures,
        })
    );
    if failures != 0 {
        eprintln!("moe_shared_prepared_iu4_oracle: {failures} failing comparison(s)");
        std::process::exit(1);
    }
}
