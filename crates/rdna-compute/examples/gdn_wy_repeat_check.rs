// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Synthetic GPU oracle for the chunked (WY) Qwen GDN prefill kernel
//! `gated_delta_chunk_gate_wmma` (`gated_delta_step_gate_wmma`, the product call) on gfx1151 / gfx1100.
//!
//! With repeated tokens `k_i . k_j ~ 1` inside a 16-row chunk; the F16 WMMA product form of the WY
//! inverse `T = (I - L)^-1` then rounded F16 partial products of magnitude ~1e3 into an O(1) error in T,
//! the state stopped contracting, and a value head went NaN.  `T` is now an F32 forward substitution.
//!
//! 48 value heads, 16 key heads, dims 128, BF16 projection, F32 state (zero), F32 gated output.  Cases:
//!   (a) one unit key/value per head repeated, beta 0.99, g 0
//!   (b) the same, beta 0.9, g -0.01
//!   (c) random unit keys, beta ~ U(0.5, 0.99), g ~ U(-0.1, 0)
//! each compared with an f64 CPU reference of the same recurrence plus the gated RMSNorm, fed the BF16
//! normalized q/k the chunk kernel reads (so the error is the chunk kernel's own).
//!
//! Usage: cargo run --release -p rdna-compute --example gdn_wy_repeat_check [-- --rows N] [--dump-dir DIR]
//! `--dump-dir DIR` also writes each case's raw GPU output and final state as little-endian f32 files
//! `DIR/case_<a|b|c>_rows<N>_{y,s}.f32le` (`y` = `[rows][VALUE_WIDTH]`, `s` = `[value head][key][value]`).
//! Exit 0: all outputs finite and case (c) rms rel error <= 1e-2.  Exit 1: gate failed.  Exit 2: could
//! not run (no supported GPU, route not taken, GPU error).

use half::bf16;
use rayon::prelude::*;
use rdna_compute::{tensor_ops::*, DType, Gpu};

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const DIM: usize = 128;
const QKV_WIDTH: usize = 2 * KEY_HEADS * DIM + VALUE_HEADS * DIM; // 10240
const VALUE_WIDTH: usize = VALUE_HEADS * DIM; // 6144
const EPS: f64 = 1.0e-6;
const RMS_LIMIT_C: f64 = 1.0e-2;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / 16_777_216.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }
    fn gauss(&mut self) -> f32 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos()
    }
}

fn bf16r(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

/// `gated_delta_qk_norm_bf16_batched` on one 128-wide BF16-exact head (BF16 rounding points of the kernel,
/// plain f32 summation order).
fn qk_norm_bf16(x: &[f32]) -> Vec<f32> {
    let part: f32 = x.iter().map(|&v| bf16r(v * v)).sum();
    let inv = bf16r(1.0 / (bf16r(part) + bf16r(EPS as f32)).sqrt());
    x.iter().map(|&v| bf16r(v * inv)).collect()
}

struct Case {
    name: &'static str,
    /// BF16 projection, `[rows][QKV_WIDTH]` (q | k | v), values BF16-exact.
    projection: Vec<f32>,
    /// `[rows][VALUE_HEADS]`
    gate: Vec<f32>,
    beta: Vec<f32>,
    /// `[rows][VALUE_WIDTH]`
    z: Vec<f32>,
    /// gate: max |S| must stay finite; only case (c) has an accuracy limit
    accuracy_gated: bool,
}

fn weights(rng: &mut Rng) -> Vec<f32> {
    (0..DIM).map(|_| bf16r(1.0 + 0.1 * rng.gauss())).collect()
}

fn build_case(
    name: &'static str,
    rows: usize,
    seed: u64,
    repeated: bool,
    beta: f32,
    g: f32,
) -> Case {
    let mut rng = Rng(seed);
    let mut projection = vec![0f32; rows * QKV_WIDTH];
    if repeated {
        // One unit key (q = k) per key head and one value per value head, repeated over the rows.
        let keys: Vec<Vec<f32>> = (0..KEY_HEADS)
            .map(|_| (0..DIM).map(|_| bf16r(rng.gauss())).collect())
            .collect();
        let vals: Vec<Vec<f32>> = (0..VALUE_HEADS)
            .map(|_| (0..DIM).map(|_| bf16r(rng.gauss())).collect())
            .collect();
        for r in 0..rows {
            let row = &mut projection[r * QKV_WIDTH..(r + 1) * QKV_WIDTH];
            for h in 0..KEY_HEADS {
                row[h * DIM..(h + 1) * DIM].copy_from_slice(&keys[h]);
                row[(KEY_HEADS + h) * DIM..(KEY_HEADS + h + 1) * DIM].copy_from_slice(&keys[h]);
            }
            for h in 0..VALUE_HEADS {
                row[2 * KEY_HEADS * DIM + h * DIM..2 * KEY_HEADS * DIM + (h + 1) * DIM]
                    .copy_from_slice(&vals[h]);
            }
        }
    } else {
        for x in projection.iter_mut() {
            *x = bf16r(rng.gauss());
        }
    }
    let z = (0..rows * VALUE_WIDTH).map(|_| rng.gauss()).collect();
    let (gate, beta) = if repeated {
        (vec![g; rows * VALUE_HEADS], vec![beta; rows * VALUE_HEADS])
    } else {
        (
            (0..rows * VALUE_HEADS)
                .map(|_| rng.range(-0.1, 0.0))
                .collect(),
            (0..rows * VALUE_HEADS)
                .map(|_| rng.range(0.5, 0.99))
                .collect(),
        )
    };
    Case {
        name,
        projection,
        gate,
        beta,
        z,
        accuracy_gated: !repeated,
    }
}

/// f64 recurrence + gated RMSNorm for one value head.  Returns `y [rows][128]` and the final state
/// `[key][value]`.  q/k are the BF16-normalized rows; q is scaled by 1/sqrt(128).
fn reference_head(
    case: &Case,
    vh: usize,
    qn: &[Vec<f32>],
    kn: &[Vec<f32>],
    norm: &[f32],
    rows: usize,
) -> (Vec<f64>, Vec<f64>) {
    let kh = vh / (VALUE_HEADS / KEY_HEADS);
    let scale = 1.0 / (DIM as f64).sqrt();
    let mut s = vec![0f64; DIM * DIM];
    let mut y = vec![0f64; rows * DIM];
    let (mut kv, mut d, mut o) = (vec![0f64; DIM], vec![0f64; DIM], vec![0f64; DIM]);
    for t in 0..rows {
        let g = case.gate[t * VALUE_HEADS + vh] as f64;
        let b = case.beta[t * VALUE_HEADS + vh] as f64;
        let k: Vec<f64> = kn[t * KEY_HEADS + kh].iter().map(|&x| x as f64).collect();
        let q: Vec<f64> = qn[t * KEY_HEADS + kh]
            .iter()
            .map(|&x| x as f64 * scale)
            .collect();
        let v = &case.projection[t * QKV_WIDTH + 2 * KEY_HEADS * DIM + vh * DIM..][..DIM];
        let eg = g.exp();
        kv.iter_mut().for_each(|x| *x = 0.0);
        for key in 0..DIM {
            let row = &mut s[key * DIM..(key + 1) * DIM];
            for (c, x) in row.iter_mut().enumerate() {
                *x *= eg;
                kv[c] += *x * k[key];
            }
        }
        for c in 0..DIM {
            d[c] = (v[c] as f64 - kv[c]) * b;
        }
        o.iter_mut().for_each(|x| *x = 0.0);
        for key in 0..DIM {
            let row = &mut s[key * DIM..(key + 1) * DIM];
            for (c, x) in row.iter_mut().enumerate() {
                *x += k[key] * d[c];
                o[c] += *x * q[key];
            }
        }
        let inv = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / DIM as f64 + EPS).sqrt();
        let z = &case.z[t * VALUE_WIDTH + vh * DIM..][..DIM];
        for c in 0..DIM {
            let zz = bf16r(z[c]) as f64;
            y[t * DIM + c] = o[c] * inv * norm[c] as f64 / (1.0 + (-zz).exp());
        }
    }
    (y, s)
}

fn to_bytes_bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

struct Outcome {
    nonfinite: usize,
    max_rel: f64,
    rms_rel: f64,
    s_gpu: f64,
    s_ref: f64,
    s_rel: f64,
}

/// Write `v` to `path` as little-endian f32, converting through a fixed 64 KiB buffer.
fn dump_f32le(path: &std::path::Path, v: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let mut buf = [0u8; 1 << 16];
    for chunk in v.chunks(buf.len() / 4) {
        for (dst, &x) in buf.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf[..chunk.len() * 4])?;
    }
    f.flush()
}

fn run_case(
    gpu: &mut Gpu,
    case: &Case,
    tag: &str,
    rows: usize,
    norm_w: &[f32],
    dump_dir: Option<&std::path::Path>,
) -> Result<Outcome, String> {
    let err = |what: &str, e: &dyn std::fmt::Display| format!("{}: {what}: {e}", case.name);
    let mut projection = gpu
        .upload_raw(&to_bytes_bf16(&case.projection), &[rows * QKV_WIDTH])
        .map_err(|e| err("upload projection", &e))?;
    projection.dtype = DType::BF16;
    let mut norm = gpu
        .upload_raw(&to_bytes_bf16(norm_w), &[DIM])
        .map_err(|e| err("upload norm", &e))?;
    norm.dtype = DType::BF16;
    let gate = gpu
        .upload_f32(&case.gate, &[rows * VALUE_HEADS])
        .map_err(|e| err("upload gate", &e))?;
    let beta = gpu
        .upload_f32(&case.beta, &[rows * VALUE_HEADS])
        .map_err(|e| err("upload beta", &e))?;
    let z = gpu
        .upload_f32(&case.z, &[rows * VALUE_WIDTH])
        .map_err(|e| err("upload z", &e))?;
    let state = gpu
        .zeros(&[VALUE_HEADS * DIM * DIM], DType::F32)
        .map_err(|e| err("alloc state", &e))?;
    let output = gpu
        .zeros(&[rows * VALUE_WIDTH], DType::F32)
        .map_err(|e| err("alloc output", &e))?;
    let step = GatedDeltaStepBatched {
        projection: &projection,
        gate: &gate,
        beta: &beta,
        state: &state,
        format: GdnStateFormat::F32,
        residual: None,
        output: &output,
        row_states: None,
        rows,
        qkv_width: QKV_WIDTH,
        key_heads: KEY_HEADS,
        value_heads: VALUE_HEADS,
        key_dim: DIM,
        value_dim: DIM,
        position: 0,
    };
    if !gated_delta_chunk_route(gpu, &step) {
        return Err(format!("{}: the chunked WY route is not taken (rows >= 512, gfx11 WMMA, HIPFIRE_QWEN4_GDN_PREFILL=wy, HIPFIRE_QWEN4_F16_WMMA != 0)", case.name));
    }
    let gate_args = GatedDeltaGateBatched {
        recurrent_output: &output,
        z: &z,
        norm: &norm,
        output: &output,
        rows,
        value_heads: VALUE_HEADS,
        value_dim: DIM,
    };
    gated_delta_step_gate_wmma(gpu, &step, &gate_args)
        .map_err(|e| err("gated_delta_step_gate_wmma", &e))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| err("synchronize", &e))?;
    let y_gpu = gpu
        .download_f32(&output)
        .map_err(|e| err("download output", &e))?;
    let s_gpu = gpu
        .download_f32(&state)
        .map_err(|e| err("download state", &e))?;
    for t in [projection, norm, gate, beta, z, state, output] {
        gpu.free_tensor(t).map_err(|e| err("free tensor", &e))?;
    }
    if let Some(dir) = dump_dir {
        for (suffix, data) in [("y", &y_gpu), ("s", &s_gpu)] {
            let path = dir.join(format!("case_{tag}_rows{rows}_{suffix}.f32le"));
            dump_f32le(&path, data).map_err(|e| err(&format!("dump {}", path.display()), &e))?;
        }
    }

    // f64 reference, one rayon task per value head.
    let mut qn = Vec::with_capacity(rows * KEY_HEADS);
    let mut kn = Vec::with_capacity(rows * KEY_HEADS);
    for t in 0..rows {
        let row = &case.projection[t * QKV_WIDTH..(t + 1) * QKV_WIDTH];
        for h in 0..KEY_HEADS {
            qn.push(qk_norm_bf16(&row[h * DIM..(h + 1) * DIM]));
            kn.push(qk_norm_bf16(
                &row[(KEY_HEADS + h) * DIM..(KEY_HEADS + h + 1) * DIM],
            ));
        }
    }
    let refs: Vec<(Vec<f64>, Vec<f64>)> = (0..VALUE_HEADS)
        .into_par_iter()
        .map(|vh| reference_head(case, vh, &qn, &kn, norm_w, rows))
        .collect();

    let (mut nonfinite, mut max_d, mut max_y, mut sum_d2, mut sum_y2) =
        (0usize, 0f64, 0f64, 0f64, 0f64);
    let mut count = 0usize;
    for (vh, (y_ref, _)) in refs.iter().enumerate() {
        for t in 0..rows {
            for c in 0..DIM {
                let got = y_gpu[t * VALUE_WIDTH + vh * DIM + c];
                let want = y_ref[t * DIM + c];
                max_y = max_y.max(want.abs());
                sum_y2 += want * want;
                count += 1;
                if !got.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                let d = (got as f64 - want).abs();
                max_d = max_d.max(d);
                sum_d2 += d * d;
            }
        }
    }
    let s_gpu_max = s_gpu.iter().fold(0f64, |m, &x| {
        if x.is_finite() {
            m.max(x.abs() as f64)
        } else {
            f64::INFINITY
        }
    });
    let s_ref_max = refs
        .iter()
        .flat_map(|(_, s)| s.iter())
        .fold(0f64, |m, &x| m.max(x.abs()));
    let mut s_diff = 0f64;
    for (vh, (_, s_ref)) in refs.iter().enumerate() {
        for i in 0..DIM * DIM {
            let d = (s_gpu[vh * DIM * DIM + i] as f64 - s_ref[i]).abs();
            s_diff = if d.is_finite() {
                s_diff.max(d)
            } else {
                f64::INFINITY
            };
        }
    }
    Ok(Outcome {
        nonfinite,
        max_rel: max_d / max_y,
        rms_rel: (sum_d2 / count as f64).sqrt() / (sum_y2 / count as f64).sqrt(),
        s_gpu: s_gpu_max,
        s_ref: s_ref_max,
        s_rel: s_diff / s_ref_max,
    })
}

fn main() {
    let mut rows = 2048usize;
    let mut dump_dir: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rows" => {
                rows = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--rows needs an integer");
                    std::process::exit(2)
                })
            }
            "--dump-dir" => {
                dump_dir = Some(
                    args.next()
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|| {
                            eprintln!("--dump-dir needs a directory");
                            std::process::exit(2)
                        }),
                )
            }
            other => {
                eprintln!("usage: gdn_wy_repeat_check [--rows N] [--dump-dir DIR]  (unknown argument {other})");
                std::process::exit(2)
            }
        }
    }
    if let Some(dir) = &dump_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("--dump-dir {}: {e}", dir.display());
            std::process::exit(2);
        }
    }
    if rows < 512 {
        eprintln!("--rows must be at least 512 (the chunked WY route's minimum)");
        std::process::exit(2);
    }
    let mut gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("GPU init failed: {e}");
        std::process::exit(2)
    });
    if gpu.arch != "gfx1151" && gpu.arch != "gfx1100" {
        eprintln!(
            "gdn_wy_repeat_check: needs gfx1151 or gfx1100, found {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    let norm_w = weights(&mut Rng(0x6e6f726d));
    let cases = [
        build_case(
            "(a) repeated key/value, beta 0.99, g 0",
            rows,
            0xA1,
            true,
            0.99,
            0.0,
        ),
        build_case(
            "(b) repeated key/value, beta 0.9, g -0.01",
            rows,
            0xB2,
            true,
            0.9,
            -0.01,
        ),
        build_case(
            "(c) random unit keys, beta U(.5,.99), g U(-.1,0)",
            rows,
            0xC3,
            false,
            0.0,
            0.0,
        ),
    ];
    println!("gdn_wy_repeat_check arch={} rows={rows} value_heads={VALUE_HEADS} key_heads={KEY_HEADS} dim={DIM} (BF16 projection, F32 state/output)", gpu.arch);
    println!(
        "{:<52} {:>10} {:>11} {:>11} {:>10} {:>10} {:>10}",
        "case", "non-finite", "max rel err", "rms rel err", "max|S|", "max|S|ref", "max|dS|/|S|"
    );
    let mut failed = false;
    for (case, tag) in cases.iter().zip(["a", "b", "c"]) {
        match run_case(&mut gpu, case, tag, rows, &norm_w, dump_dir.as_deref()) {
            Ok(o) => {
                println!(
                    "{:<52} {:>10} {:>11.3e} {:>11.3e} {:>10.4e} {:>10.4e} {:>10.3e}",
                    case.name, o.nonfinite, o.max_rel, o.rms_rel, o.s_gpu, o.s_ref, o.s_rel
                );
                if o.nonfinite != 0 || !o.s_gpu.is_finite() {
                    println!("FAIL {}: non-finite output/state", case.name);
                    failed = true;
                }
                if case.accuracy_gated && (o.rms_rel.is_nan() || o.rms_rel > RMS_LIMIT_C) {
                    println!(
                        "FAIL {}: rms rel error {:.3e} > {RMS_LIMIT_C:e}",
                        case.name, o.rms_rel
                    );
                    failed = true;
                }
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        }
    }
    println!("{}", if failed { "RESULT FAIL" } else { "RESULT PASS" });
    std::process::exit(i32::from(failed));
}
