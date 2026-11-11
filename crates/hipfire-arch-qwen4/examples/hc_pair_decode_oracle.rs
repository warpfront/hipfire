// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of the decode / wide-verify HC read/write pair fusion
//! (`HIPFIRE_QWEN4_HC_PAIR_DECODE`, `HIPFIRE_QWEN4_HC_PAIR_WIDE`,
//! `hyper_norm_gate_k4_outputs_f32`), rows `1..=64`.
//!
//! ```text
//! hc_pair_decode_oracle [SEEDS]        (default 32 seeds)
//! hc_pair_decode_oracle --bench        (timing only: median us of the fused kernel vs the pair)
//! ```
//!
//! On one GPU, for random HC streams (F32 words and BF16 bits, `state_bf16`), random BF16 norm
//! weights and random BF16 `block_inject` gate weights at hidden 2560 x 4 branches, rows `1..=64`,
//! several activation scale classes per seed, the fused launch is compared bit for bit with the
//! unfused sequence the paired write would run:
//!
//! * `normalized`: `hyper_norm` (`hyper_norm_f32`);
//! * `gates`: the write's k4 gate route over that `normalized` -- `Gpu::gemv_bf16_xf32_k4_rows`
//!   (the launcher the dispatch calls: `gemv_bf16_xf32_k4` at one row, `_k4_rows_r<R>` /
//!   `_otile_r<R>` per the flag at 2..=8, `gemv_bf16_xf32_k4_rows_allrows_m<TM>u<U>` at 9..=64
//!   under `HIPFIRE_QWEN4_HC_K4_ALLROWS`: exactly today's wide route, asserted per row count
//!   via `Gpu::hc_k4_allrows_applies`), the otile entry directly at 2..=8 rows, and each
//!   row alone through `gemv_bf16_xf32_k4`.
//!
//! Outputs sit in sentinel-filled allocations (`0xFF` words, a 16-word tail past the gates);
//! a launch may write only its own rows. One negative control flips an exponent bit of one
//! stream element and must be detected in both outputs of both formats.
//!
//! Last result line: `hc_pair_decode_oracle: <C> cells, <W> words compared, <M> mismatches,
//! control detected`. Exit 0 pass, 1 mismatch / control blind, 2 environment.

use rdna_compute::tensor_ops::{
    hyper_norm, hyper_norm_gate_k4_outputs, HyperNorm, HyperNormGateK4Outputs,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = HIDDEN * BRANCHES;
const MAX_ROWS: usize = 64;
const GATE_TAIL: usize = 16;
/// Activation scale classes (each seed x class is one stream set).
const AMPS: [f32; 4] = [1.0, 1.0 / 1024.0, 48.0, 0.0625];

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Roughly normal in [-6, 6): the sum of four uniforms, centred.
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 1.7320508
    }
}

/// F32 -> BF16 bits, round to nearest even (finite inputs only).
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Gaussian values with heavy-tailed outliers and exact / signed zeros, scaled by `amp`.
fn gen_values(n: usize, amp: f32, rng: &mut Rng) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v = match rng.next() % 64 {
                0 => 0.0,
                1 => -0.0,
                2 => rng.gauss() * 40.0,
                3 => rng.gauss() * 0.001,
                _ => rng.gauss(),
            };
            v * amp
        })
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Stream bytes of `rows` rows: F32 words, or BF16 bits padded to the F32-typed extent
/// (`state_bf16` streams are F32-typed allocations holding BF16 bits).
fn stream_bytes(values: &[f32], bf16: bool) -> Vec<u8> {
    if bf16 {
        let mut b = bf16_bytes(values);
        b.resize(values.len() * 4, 0);
        b
    } else {
        f32_bytes(values)
    }
}

fn upload_typed(
    gpu: &mut Gpu,
    bytes: &[u8],
    numel: usize,
    dtype: DType,
) -> Result<GpuTensor, String> {
    let mut t = gpu.upload_raw(bytes, &[bytes.len()]).map_err(err)?;
    t.shape = vec![numel];
    t.dtype = dtype;
    Ok(t)
}

fn alloc_sentinel(gpu: &mut Gpu, elems: usize) -> Result<GpuTensor, String> {
    let t = gpu.alloc_tensor(&[elems], DType::F32).map_err(err)?;
    gpu.hip.memset(&t.buf, 0xff, t.buf.size()).map_err(err)?;
    Ok(t)
}

#[derive(Default)]
struct Stats {
    cells: u64,
    words: u64,
    mismatches: u64,
}

impl Stats {
    /// Compare `got[..want.len()]` with `want` bitwise; `got[want.len()..]` must be sentinel.
    fn compare(&mut self, tag: &str, got: &[f32], want: &[f32]) {
        self.cells += 1;
        let mut bad = 0u64;
        let mut first = None;
        for (i, g) in got.iter().enumerate() {
            let expect = if i < want.len() {
                want[i].to_bits()
            } else {
                0xffff_ffffu32
            };
            self.words += 1;
            if g.to_bits() != expect {
                bad += 1;
                first.get_or_insert((i, g.to_bits(), expect));
            }
        }
        if bad != 0 {
            self.mismatches += bad;
            let (i, g, w) = first.unwrap();
            eprintln!("MISMATCH {tag}: {bad} words; first at {i}: got {g:#010x}, want {w:#010x}");
        }
    }
}

struct Case<'a> {
    values: &'a [f32],
    bf16: bool,
    rows: usize,
    norm_weight: &'a GpuTensor,
    gate_weight: &'a GpuTensor,
}

/// Reference: (normalized rows, gates rows) of the unfused sequence, default route.
fn reference(gpu: &mut Gpu, c: &Case<'_>) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let n = c.rows * WIDE;
    let input = upload_typed(gpu, &stream_bytes(c.values, c.bf16), n, DType::F32)?;
    let normalized = gpu.alloc_tensor(&[n], DType::F32).map_err(err)?;
    hyper_norm(
        gpu,
        &HyperNorm {
            input: &input,
            norm_weight: c.norm_weight,
            normalized: &normalized,
            branches: BRANCHES,
            hidden: HIDDEN,
            state_bf16: c.bf16,
        },
    )
    .map_err(err)?;
    if c.rows > 8 && !gpu.hc_k4_allrows_applies(WIDE, c.rows) {
        return Err(format!(
            "rows {}: the allrows k4 entry does not apply (HIPFIRE_QWEN4_HC_K4_ALLROWS off?)",
            c.rows
        ));
    }
    let gates = gpu
        .alloc_tensor(&[c.rows * BRANCHES], DType::F32)
        .map_err(err)?;
    gpu.gemv_bf16_xf32_k4_rows(
        c.gate_weight,
        &normalized,
        &gates,
        BRANCHES,
        WIDE,
        None,
        c.rows,
    )
    .map_err(err)?;
    let mut otile = Vec::new();
    if (2..=8).contains(&c.rows) {
        let y = gpu
            .alloc_tensor(&[c.rows * BRANCHES], DType::F32)
            .map_err(err)?;
        gpu.gemv_bf16_xf32_k4_rows_otile(
            c.gate_weight,
            &normalized,
            &y,
            BRANCHES,
            WIDE,
            None,
            c.rows,
        )
        .map_err(err)?;
        otile = gpu.download_f32(&y).map_err(err)?;
        gpu.free_tensor(y).map_err(err)?;
    }
    let out = (
        gpu.download_f32(&normalized).map_err(err)?,
        gpu.download_f32(&gates).map_err(err)?,
        otile,
    );
    for t in [input, normalized, gates] {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(out)
}

/// Each row alone through `hyper_norm` + `gemv_bf16_xf32_k4`.
fn single_rows(gpu: &mut Gpu, c: &Case<'_>) -> Result<Vec<f32>, String> {
    let mut gates_all = Vec::new();
    for r in 0..c.rows {
        let row = &c.values[r * WIDE..(r + 1) * WIDE];
        let input = upload_typed(gpu, &stream_bytes(row, c.bf16), WIDE, DType::F32)?;
        let normalized = gpu.alloc_tensor(&[WIDE], DType::F32).map_err(err)?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &input,
                norm_weight: c.norm_weight,
                normalized: &normalized,
                branches: BRANCHES,
                hidden: HIDDEN,
                state_bf16: c.bf16,
            },
        )
        .map_err(err)?;
        let gates = gpu.alloc_tensor(&[BRANCHES], DType::F32).map_err(err)?;
        gpu.gemv_bf16_xf32_k4(c.gate_weight, &normalized, &gates, BRANCHES, WIDE, None)
            .map_err(err)?;
        gates_all.extend(gpu.download_f32(&gates).map_err(err)?);
        for t in [input, normalized, gates] {
            gpu.free_tensor(t).map_err(err)?;
        }
    }
    Ok(gates_all)
}

/// The fused launch over `stream_values` (which may differ from the reference's input in the
/// negative control): (normalized incl. sentinel tail, gates incl. sentinel tail).
fn fused(
    gpu: &mut Gpu,
    c: &Case<'_>,
    stream_values: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let n = c.rows * WIDE;
    let input = upload_typed(gpu, &stream_bytes(stream_values, c.bf16), n, DType::F32)?;
    let normalized = alloc_sentinel(gpu, n + GATE_TAIL)?;
    let gates = alloc_sentinel(gpu, c.rows * BRANCHES + GATE_TAIL)?;
    hyper_norm_gate_k4_outputs(
        gpu,
        &HyperNormGateK4Outputs {
            input: &input,
            norm_weight: c.norm_weight,
            gate_weight: c.gate_weight,
            normalized: &normalized,
            gates: &gates,
            rows: c.rows,
            branches: BRANCHES,
            hidden: HIDDEN,
            state_bf16: c.bf16,
        },
    )
    .map_err(err)?;
    let out = (
        gpu.download_f32(&normalized).map_err(err)?,
        gpu.download_f32(&gates).map_err(err)?,
    );
    for t in [input, normalized, gates] {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(out)
}

/// `--bench`: median microseconds per launch of the fused kernel and of the `hyper_norm` +
/// k4 gate pair it replaces, at 1, 8, 24, 32, 48 and 64 rows, F32 streams. 200 samples per cell, each the
/// hipEvent time of 10 back-to-back launches divided by 10 (so event overhead is amortised);
/// 20 untimed warm-up launches first. No result is compared here.
fn bench(gpu: &mut Gpu) -> Result<i32, String> {
    const SAMPLES: usize = 200;
    const BATCH: usize = 10;
    let mut rng = Rng(0x5eed_0b5e_55ed_1234);
    let nw = gen_values(WIDE, 0.5, &mut rng);
    let gw = gen_values(BRANCHES * WIDE, 0.05, &mut rng);
    let norm_weight = upload_typed(gpu, &bf16_bytes(&nw), WIDE, DType::BF16)?;
    let gate_weight = upload_typed(gpu, &bf16_bytes(&gw), BRANCHES * WIDE, DType::BF16)?;
    for rows in [1usize, 8, 24, 32, 48, 64] {
        let values = gen_values(rows * WIDE, 1.0, &mut rng);
        let n = rows * WIDE;
        let input = upload_typed(gpu, &stream_bytes(&values, false), n, DType::F32)?;
        let normalized = gpu.alloc_tensor(&[n], DType::F32).map_err(err)?;
        let gates = gpu
            .alloc_tensor(&[rows * BRANCHES], DType::F32)
            .map_err(err)?;
        let mut fused_launch = |gpu: &mut Gpu| -> Result<(), String> {
            hyper_norm_gate_k4_outputs(
                gpu,
                &HyperNormGateK4Outputs {
                    input: &input,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    normalized: &normalized,
                    gates: &gates,
                    rows,
                    branches: BRANCHES,
                    hidden: HIDDEN,
                    state_bf16: false,
                },
            )
            .map_err(err)
        };
        let mut pair_launch = |gpu: &mut Gpu| -> Result<(), String> {
            hyper_norm(
                gpu,
                &HyperNorm {
                    input: &input,
                    norm_weight: &norm_weight,
                    normalized: &normalized,
                    branches: BRANCHES,
                    hidden: HIDDEN,
                    state_bf16: false,
                },
            )
            .map_err(err)?;
            gpu.gemv_bf16_xf32_k4_rows(
                &gate_weight,
                &normalized,
                &gates,
                BRANCHES,
                WIDE,
                None,
                rows,
            )
            .map_err(err)
        };
        let start = gpu.hip.event_create().map_err(err)?;
        let stop = gpu.hip.event_create().map_err(err)?;
        let median_us = |gpu: &mut Gpu,
                         body: &mut dyn FnMut(&mut Gpu) -> Result<(), String>|
         -> Result<f64, String> {
            for _ in 0..20 {
                body(gpu)?;
            }
            let mut samples = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                gpu.hip.event_record(&start, None).map_err(err)?;
                for _ in 0..BATCH {
                    body(gpu)?;
                }
                gpu.hip.event_record(&stop, None).map_err(err)?;
                gpu.hip.event_synchronize(&stop).map_err(err)?;
                let ms = gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?;
                samples.push(ms as f64 * 1_000.0 / BATCH as f64);
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[SAMPLES / 2])
        };
        let fused_us = median_us(gpu, &mut fused_launch)?;
        let pair_us = median_us(gpu, &mut pair_launch)?;
        println!(
            "hc_pair_decode_oracle bench rows {rows}: fused {fused_us:.2} us, hyper_norm + k4 pair {pair_us:.2} us"
        );
        for t in [input, normalized, gates] {
            gpu.free_tensor(t).map_err(err)?;
        }
    }
    Ok(0)
}

fn run() -> Result<i32, String> {
    let arg = std::env::args().nth(1);
    let bench_mode = arg.as_deref() == Some("--bench");
    let seeds: u64 = match arg {
        Some(s) if !bench_mode => s.parse().map_err(|_| format!("bad seed count {s:?}"))?,
        _ => 32,
    };
    let mut gpu = Gpu::init().map_err(err)?;
    println!("device arch: {}", gpu.arch);
    println!("k4 entry at rows 9..=64: gemv_bf16_xf32_k4_rows -> gemv_bf16_xf32_k4_rows_allrows_m<TM>u<U> (HIPFIRE_QWEN4_HC_K4_ALLROWS)");
    if !gpu.arch_caps.has_gfx11_plus_simt() {
        eprintln!(
            "refusing: the fused kernel is a gfx11+ route ({})",
            gpu.arch
        );
        return Ok(2);
    }
    if bench_mode {
        return bench(&mut gpu);
    }
    let mut st = Stats::default();
    let mut control_ok = true;
    for seed in 0..seeds {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0xd1b5_4a32_d192_ed03) | 1);
        let nw = gen_values(WIDE, 0.5, &mut rng);
        let gw = gen_values(BRANCHES * WIDE, 0.05, &mut rng);
        let norm_weight = upload_typed(&mut gpu, &bf16_bytes(&nw), WIDE, DType::BF16)?;
        let gate_weight = upload_typed(&mut gpu, &bf16_bytes(&gw), BRANCHES * WIDE, DType::BF16)?;
        for &amp in &AMPS {
            let all = gen_values(MAX_ROWS * WIDE, amp, &mut rng);
            for bf16 in [false, true] {
                // BF16 streams carry BF16-exact values.
                let values: Vec<f32> = if bf16 {
                    all.iter()
                        .map(|v| f32::from_bits((bf16_bits(*v) as u32) << 16))
                        .collect()
                } else {
                    all.clone()
                };
                for rows in 1..=MAX_ROWS {
                    let c = Case {
                        values: &values[..rows * WIDE],
                        bf16,
                        rows,
                        norm_weight: &norm_weight,
                        gate_weight: &gate_weight,
                    };
                    let tag = format!("seed {seed} amp {amp} bf16 {bf16} rows {rows}");
                    let (want_norm, want_gates, want_otile) = reference(&mut gpu, &c)?;
                    let single = single_rows(&mut gpu, &c)?;
                    let (got_norm, got_gates) = fused(&mut gpu, &c, c.values)?;
                    st.compare(&format!("{tag} normalized"), &got_norm, &want_norm);
                    st.compare(&format!("{tag} gates/route"), &got_gates, &want_gates);
                    st.compare(&format!("{tag} gates/single"), &got_gates, &single);
                    if (2..=8).contains(&rows) {
                        st.compare(&format!("{tag} gates/otile"), &got_gates, &want_otile);
                    }
                    // Negative control (once per format per seed, 3 rows): flip an exponent
                    // bit of one stream element; the fused outputs must differ from the
                    // unperturbed reference.
                    if (rows == 3 || rows == 40) && amp == AMPS[0] {
                        let mut bad = values[..rows * WIDE].to_vec();
                        // 1000.0 is BF16-exact and far from any gaussian draw.
                        bad[WIDE + 17] = 1000.0;
                        let (n2, g2) = fused(&mut gpu, &c, &bad)?;
                        let norm_differs = n2[..want_norm.len()]
                            .iter()
                            .zip(&want_norm)
                            .any(|(a, b)| a.to_bits() != b.to_bits());
                        let gates_differ = g2[..want_gates.len()]
                            .iter()
                            .zip(&want_gates)
                            .any(|(a, b)| a.to_bits() != b.to_bits());
                        if !norm_differs || !gates_differ {
                            eprintln!(
                                "CONTROL BLIND {tag}: normalized differs {norm_differs}, gates differ {gates_differ}"
                            );
                            control_ok = false;
                        }
                    }
                }
            }
        }
        for t in [norm_weight, gate_weight] {
            gpu.free_tensor(t).map_err(err)?;
        }
    }
    println!(
        "hc_pair_decode_oracle: {} cells, {} words compared, {} mismatches, control {}",
        st.cells,
        st.words,
        st.mismatches,
        if control_ok { "detected" } else { "BLIND" }
    );
    Ok(if st.mismatches == 0 && control_ok {
        0
    } else {
        1
    })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("hc_pair_decode_oracle: {e}");
            std::process::exit(2);
        }
    }
}
