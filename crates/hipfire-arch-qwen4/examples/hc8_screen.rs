// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HC8 screen: 8-bit (E4M3 / int8) HC mixer weights through the PeaceMaker module `hc8_wmma_pm_gfx1201`
//! (`rdna_compute::hc8`) against the BF16 incumbents exactly as the product routes them, on ONE real layer
//! (attn and mlp HC down `[320, 10240]` and up `[10240, 320]`), exact gfx1201.
//!
//! ```text
//! hc8_screen --model PATH [--layer 4] [--rows 1,2,4,6,8,16,32,48,64] [--calls 40] [--warm 3] [--out FILE.json]
//! ```
//!
//! Timing is COLD: a 256 MiB memset flushes the MALL/L2 before every call; each call is bracketed by its own
//! hipEvent pair (so it includes launch latency); `--calls` samples, median and p10/p90 reported.
//! Useful GB/s = weight bytes (q + s for hc8, 2*M*K for BF16) / median; `%598` against 598 GB/s.
//!
//! Incumbents (layer_ops.rs `execute_hyper_read_inner`): down rows <= 8 `gemv_bf16_xf32_k4_rows` (act), rows
//! >= 16 `lane_wmma_gemm_fixed`; up 1 row = BF16 GEMV (`project_weight`) + `hyper_read_projected`, 2..=8
//! `hyper_read_up_fused`, >= 16 `lane_wmma_read_up`. hc8 arms: down `act`, up `set`; up is also timed as
//! hc8 GEMM + `hyper_read_projected` (the full replacement).
//!
//! Correctness per hc8 arm: max |y - ref| / max |ref| against an f64 reference from the dequantized weights
//! (down is checked with the `set` epilogue tightly and the `act` epilogue loosely: bf16 round trips and
//! exp); row invariance for rows 16/48/64: every row bitwise equals the rows = 1 run on that row's x.

use std::path::{Path, PathBuf};

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::project_weight;
use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use rdna_compute::hc8::{
    self, dequant_rowmajor, entry_name, hc8_gemm, quantize_e4m3_rows_rowmajor,
    quantize_i8_rows_rowmajor, swizzle_hc8, Hc8Fmt, Hc8Geom, Hc8Weight, GEOMS,
};
use rdna_compute::lane_wmma::{lane_wmma_read_up, LaneWmmaWeight};
use rdna_compute::tensor_ops::{
    hyper_norm, hyper_read_projected, hyper_read_up_fused, HyperNorm, HyperReadProjected,
    HyperReadUpFused,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

type R<T> = Result<T, String>;

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = BRANCHES * HIDDEN;
const RANK: usize = 320;
const ACT: f32 = 0.25;
const PEAK_GBPS: f64 = 598.0;
const FLUSH_BYTES: usize = 256 << 20;
const MAX_ROWS: usize = 64;
const INV_ROWS: [usize; 3] = [16, 48, 64];
/// Max-rel thresholds: `set` is f32 accumulation order only; `act` adds bf16 round trips + exp.
const TOL_SET: f64 = 2e-5;
const TOL_ACT: f64 = 1.2e-2;

fn es<E: std::fmt::Debug>(e: E) -> String {
    format!("{e:?}")
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.unit(), self.unit());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

fn bf16r(v: f32) -> f32 {
    bf16_to_f32(bf16_bits(v))
}

fn bf16_bytes_to_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn normals(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut r = Rng(seed);
    (0..n).map(|_| bf16r(r.normal() * scale)).collect()
}

/// `hc_act` epilogue: all f32 (gemv_bf16_xf32_k4).
fn hc_act_ref(v: f32, scale: f32) -> f32 {
    let mut v = bf16r(v);
    v *= scale;
    v = bf16r(v);
    v /= 1.0 + (-v).exp();
    bf16r(v)
}

/// f64 `x . w^T`: `w [m, k]`, `x [nr, k]` -> `[nr, m]`.
fn dot_ref(w: &[f32], m: usize, k: usize, x: &[f32], nr: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nr * m];
    out.par_chunks_mut(m).enumerate().for_each(|(t, o)| {
        let xr = &x[t * k..(t + 1) * k];
        for (c, slot) in o.iter_mut().enumerate() {
            *slot = w[c * k..(c + 1) * k]
                .iter()
                .zip(xr)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
        }
    });
    out
}

/// `max|a - r| / max|r|`.
fn max_rel(a: &[f32], r: &[f32]) -> f64 {
    let den = r.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let num = a.iter().zip(r).fold(0.0f64, |m, (&x, &y)| {
        m.max((f64::from(x) - f64::from(y)).abs())
    });
    if den == 0.0 {
        num
    } else {
        num / den
    }
}

// ---- device helpers -------------------------------------------------------------------------

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> R<GpuTensor> {
    gpu.upload_f32(v, &[v.len()]).map_err(es)
}

fn alloc_f32(gpu: &mut Gpu, n: usize) -> R<GpuTensor> {
    gpu.alloc_tensor(&[n], DType::F32).map_err(es)
}

fn upload_bf16_bytes(gpu: &mut Gpu, bytes: &[u8]) -> R<GpuTensor> {
    let mut t = gpu.upload_raw(bytes, &[bytes.len() / 2]).map_err(es)?;
    t.dtype = DType::BF16;
    t.shape = vec![bytes.len() / 2];
    Ok(t)
}

fn fetch(gpu: &Gpu, t: &GpuTensor, n: usize) -> R<Vec<f32>> {
    gpu.hip.device_synchronize().map_err(es)?;
    Ok(gpu.download_f32(t).map_err(es)?[..n].to_vec())
}

fn wref(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype: DType::BF16,
        m,
        k,
        row_stride: k * 2,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

struct Hfq(HfqFile);

impl Hfq {
    fn bf16(&self, layer: usize, suffix: &str, shape: &[usize]) -> R<Vec<u8>> {
        for name in [
            format!("model.language_model.layers.{layer}.{suffix}"),
            format!("model.layers.{layer}.{suffix}"),
        ] {
            if self.0.find_tensor_info(&name).is_none() {
                continue;
            }
            let (info, data) = self
                .0
                .tensor_data_vec(&name)
                .ok_or_else(|| format!("{name}: no data"))?;
            let dims: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
            let numel: usize = shape.iter().product();
            if dims != shape || data.len() != numel * 2 {
                return Err(format!("{name}: shape {dims:?}, expected {shape:?} BF16"));
            }
            return Ok(data);
        }
        Err(format!("layer {layer} {suffix} not in the artifact"))
    }
}

// ---- timing ---------------------------------------------------------------------------------

struct Timer {
    flush: hip_bridge::DeviceBuffer,
    calls: usize,
    warm: usize,
}

#[derive(Clone, Copy)]
struct Stat {
    med: f64,
    p10: f64,
    p90: f64,
}

fn pct(v: &[f64], p: f64) -> f64 {
    let i = ((v.len() - 1) as f64 * p).round() as usize;
    v[i]
}

impl Timer {
    /// Cold per-call timing of `f` (us): flush, event, call, event.
    fn time<F: FnMut(&mut Gpu) -> R<()>>(&self, gpu: &mut Gpu, mut f: F) -> R<Stat> {
        let mut v = Vec::with_capacity(self.calls);
        for i in 0..self.warm + self.calls {
            gpu.hip
                .memset(&self.flush, i as i32 & 0x7f, FLUSH_BYTES)
                .map_err(es)?;
            let start = gpu.hip.event_create().map_err(es)?;
            let stop = gpu.hip.event_create().map_err(es)?;
            gpu.hip.event_record(&start, None).map_err(es)?;
            f(gpu)?;
            gpu.hip.event_record(&stop, None).map_err(es)?;
            gpu.hip.event_synchronize(&stop).map_err(es)?;
            let ms = gpu.hip.event_elapsed_ms(&start, &stop).map_err(es)? as f64;
            gpu.hip.event_destroy(start).map_err(es)?;
            gpu.hip.event_destroy(stop).map_err(es)?;
            if i >= self.warm {
                v.push(ms * 1000.0);
            }
        }
        v.sort_by(f64::total_cmp);
        Ok(Stat {
            med: pct(&v, 0.5),
            p10: pct(&v, 0.1),
            p90: pct(&v, 0.9),
        })
    }
}

// ---- records --------------------------------------------------------------------------------

#[derive(Default)]
struct Out {
    recs: Vec<Value>,
    fails: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
impl Out {
    fn timed(
        &mut self,
        tensor: &str,
        family: &str,
        rows: usize,
        arm: &str,
        geom: &str,
        bytes: usize,
        st: Stat,
        extra: Value,
    ) {
        let g = |us: f64| bytes as f64 / us / 1e3;
        println!(
            "{tensor:<9} {family:<5} rows={rows:<3} {arm:<12} {geom:<8} med {:>8.2} us [{:>8.2},{:>8.2}]  {:>6.1} GB/s  {:>5.1}%598  {extra}",
            st.med, st.p10, st.p90, g(st.med), g(st.med) / PEAK_GBPS * 100.0
        );
        self.recs.push(json!({
            "tensor": tensor, "family": family, "rows": rows, "arm": arm, "geom": geom,
            "status": "ok", "weight_bytes": bytes,
            "med_us": st.med, "p10_us": st.p10, "p90_us": st.p90,
            "gbps_med": g(st.med), "pct598": g(st.med) / PEAK_GBPS * 100.0,
            "check": extra,
        }));
    }

    fn unavailable(
        &mut self,
        tensor: &str,
        family: &str,
        rows: usize,
        arm: &str,
        geom: &str,
        why: &str,
    ) {
        println!("{tensor:<9} {family:<5} rows={rows:<3} {arm:<12} {geom:<8} UNAVAILABLE {why}");
        self.recs.push(json!({
            "tensor": tensor, "family": family, "rows": rows, "arm": arm, "geom": geom,
            "status": "UNAVAILABLE", "reason": why,
        }));
    }
}

// ---- one HC tensor pair ---------------------------------------------------------------------

struct Hc {
    name: &'static str, // attn | mlp
    down_bf: GpuTensor,
    up_bf: GpuTensor,
    norm: GpuTensor,
    down_h: Vec<u16>,
    up_h: Vec<u16>,
}

struct Q {
    fmt: Hc8Fmt,
    down: Hc8Weight,
    up: Hc8Weight,
    /// Dequantized host weights (q * s).
    down_dq: Vec<f32>,
    up_dq: Vec<f32>,
    /// Row-major 8-bit bytes and row scales (fixture dump).
    raw: [(Vec<u8>, Vec<f32>); 2],
}

fn build_q(gpu: &mut Gpu, hc: &Hc, fmt: Hc8Fmt) -> R<Q> {
    let quant = |bf: &[u16], m, k| match fmt {
        Hc8Fmt::E4m3 => quantize_e4m3_rows_rowmajor(bf, m, k),
        Hc8Fmt::I8 => quantize_i8_rows_rowmajor(bf, m, k),
    };
    let (dq_q, dq_s) = quant(&hc.down_h, RANK, WIDE);
    let (up_q, up_s) = quant(&hc.up_h, WIDE, RANK);
    let down = Hc8Weight::upload(gpu, fmt, &swizzle_hc8(&dq_q, RANK, WIDE), &dq_s, RANK, WIDE)
        .map_err(es)?;
    let up = Hc8Weight::upload(gpu, fmt, &swizzle_hc8(&up_q, WIDE, RANK), &up_s, WIDE, RANK)
        .map_err(es)?;
    Ok(Q {
        fmt,
        raw: [(dq_q.clone(), dq_s.clone()), (up_q.clone(), up_s.clone())],
        down_dq: dequant_rowmajor(fmt, &dq_q, &dq_s, RANK, WIDE),
        up_dq: dequant_rowmajor(fmt, &up_q, &up_s, WIDE, RANK),
        down,
        up,
    })
}

struct Cli {
    model: PathBuf,
    layer: usize,
    rows: Vec<usize>,
    calls: usize,
    warm: usize,
    out: Option<PathBuf>,
    fixtures: Option<PathBuf>,
    /// `--fmts e4m3,i8` (default e4m3 only: the decided format).
    fmts: Vec<Hc8Fmt>,
}

fn parse_cli() -> R<Cli> {
    let mut cli = Cli {
        model: PathBuf::new(),
        layer: 4,
        rows: vec![1, 2, 4, 6, 8, 16, 32, 48, 64],
        calls: 40,
        warm: 3,
        out: None,
        fixtures: None,
        fmts: vec![Hc8Fmt::E4m3],
    };
    let mut have = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut v = |n: &str| it.next().ok_or_else(|| format!("{n} needs a value"));
        match a.as_str() {
            "--model" => {
                cli.model = PathBuf::from(v("--model")?);
                have = true;
            }
            "--layer" => cli.layer = v("--layer")?.parse().map_err(es)?,
            "--rows" => {
                cli.rows = v("--rows")?
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| t.parse::<usize>().map_err(es))
                    .collect::<R<_>>()?
            }
            "--calls" => cli.calls = v("--calls")?.parse().map_err(es)?,
            "--warm" => cli.warm = v("--warm")?.parse().map_err(es)?,
            "--dump-fixtures" => cli.fixtures = Some(PathBuf::from(v("--dump-fixtures")?)),
            "--fmts" => {
                cli.fmts = v("--fmts")?
                    .split(',')
                    .map(|t| match t {
                        "e4m3" => Ok(Hc8Fmt::E4m3),
                        "i8" => Ok(Hc8Fmt::I8),
                        o => Err(format!("unknown fmt {o}")),
                    })
                    .collect::<R<_>>()?
            }
            "--out" => cli.out = Some(PathBuf::from(v("--out")?)),
            o => return Err(format!("unknown argument {o}")),
        }
    }
    if !have {
        return Err("usage: hc8_screen --model PATH [--layer 4] [--rows ...] [--calls 40] [--out FILE.json]".into());
    }
    if cli.rows.iter().any(|&r| r == 0 || r > MAX_ROWS) || cli.calls == 0 {
        return Err("rows must be in 1..=64 and calls positive".into());
    }
    Ok(cli)
}

/// Row-invariance: rows-run output `y` (`[rows, m]` host) vs per-row singleton runs on `x1[r]`.
#[allow(clippy::too_many_arguments)]
fn invariance(
    gpu: &mut Gpu,
    w: &Hc8Weight,
    x1: &[GpuTensor],
    y_rows: &[f32],
    rows: usize,
    act: Option<f32>,
    g: Hc8Geom,
) -> R<usize> {
    let m = w.m;
    let y1 = alloc_f32(gpu, m)?;
    let mut bad = 0;
    for r in 0..rows {
        hc8_gemm(gpu, w, &x1[r], &y1, 1, act, g).map_err(es)?;
        let one = fetch(gpu, &y1, m)?;
        if one
            .iter()
            .zip(&y_rows[r * m..(r + 1) * m])
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            bad += 1;
        }
    }
    let _ = gpu.free_tensor(y1);
    Ok(bad)
}

#[allow(clippy::too_many_arguments)]
fn run_hc(
    gpu: &mut Gpu,
    tm: &Timer,
    out: &mut Out,
    cli: &Cli,
    hc: &Hc,
    qs: &[Q],
    streams_h: &[f32],
    low_h: &[f32],
) -> R<()> {
    let nmax = MAX_ROWS;
    let w_down = hc.down_bf.numel() * 2;
    let w_up = hc.up_bf.numel() * 2;
    let tname = |f: &str| format!("{}_{f}", hc.name);
    // Normalized x (BF16-exact) from the real hc_norm weight.
    let streams_all = upload_f32(gpu, &streams_h[..nmax * WIDE])?;
    let norm_all = alloc_f32(gpu, nmax * WIDE)?;
    hyper_norm(
        gpu,
        &HyperNorm {
            input: &streams_all,
            norm_weight: &hc.norm,
            normalized: &norm_all,
            branches: BRANCHES,
            hidden: HIDDEN,
            state_bf16: false,
        },
    )
    .map_err(es)?;
    let xn_h = fetch(gpu, &norm_all, nmax * WIDE)?;
    // Per-row singleton inputs (row invariance) and f64 references (prefix property across rows).
    let xn1: Vec<GpuTensor> = (0..nmax)
        .map(|r| upload_f32(gpu, &xn_h[r * WIDE..(r + 1) * WIDE]))
        .collect::<R<_>>()?;
    let low1: Vec<GpuTensor> = (0..nmax)
        .map(|r| upload_f32(gpu, &low_h[r * RANK..(r + 1) * RANK]))
        .collect::<R<_>>()?;
    if let Some(dir) = &cli.fixtures {
        std::fs::create_dir_all(dir).map_err(es)?;
        let f32le = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for q in qs {
            for (i, (fam, m, k, x)) in [
                ("down", RANK, WIDE, &xn_h),
                ("up", WIDE, RANK, &low_h.to_vec()),
            ]
            .into_iter()
            .enumerate()
            {
                let base = dir.join(format!("{}_{}_{}", hc.name, fam, q.fmt.name()));
                let w = |ext: &str, b: &[u8]| {
                    std::fs::write(format!("{}.{ext}", base.display()), b).map_err(es)
                };
                w("q.u8", &q.raw[i].0)?;
                w("s.f32", &f32le(&q.raw[i].1))?;
                w("x.f32", &f32le(&x[..MAX_ROWS * k]))?;
                let js = json!({"m": m, "k": k, "fmt": q.fmt.name(), "rows": MAX_ROWS, "tensor": format!("{}_{fam}", hc.name),
                    "q": format!("{}.q.u8", base.display()), "scales": format!("{}.s.f32", base.display()),
                    "x": format!("{}.x.f32", base.display())});
                w("json", js.to_string().as_bytes())?;
            }
        }
        println!("fixtures written to {}", dir.display());
    }
    struct Refs {
        down_set: Vec<f32>,
        down_act: Vec<f32>,
        up_set: Vec<f32>,
    }
    let refs: Vec<Refs> = qs
        .iter()
        .map(|q| {
            let d = dot_ref(&q.down_dq, RANK, WIDE, &xn_h, nmax);
            Refs {
                down_set: d.iter().map(|&v| v as f32).collect(),
                down_act: d.iter().map(|&v| hc_act_ref(v as f32, ACT)).collect(),
                up_set: dot_ref(&q.up_dq, WIDE, RANK, low_h, nmax)
                    .iter()
                    .map(|&v| v as f32)
                    .collect(),
            }
        })
        .collect();

    for &rows in &cli.rows {
        let xn = upload_f32(gpu, &xn_h[..rows * WIDE])?;
        let low = upload_f32(gpu, &low_h[..rows * RANK])?;
        let streams = upload_f32(gpu, &streams_h[..rows * WIDE])?;
        let y_down = alloc_f32(gpu, rows * RANK)?;
        let up = alloc_f32(gpu, rows * WIDE)?;
        let mixed = alloc_f32(gpu, rows * HIDDEN)?;
        let mixed2 = alloc_f32(gpu, rows * HIDDEN)?;

        // ---- down incumbent ----
        {
            let name = tname("down");
            let r = if rows <= 8 {
                tm.time(gpu, |g| {
                    g.gemv_bf16_xf32_k4_rows(&hc.down_bf, &xn, &y_down, RANK, WIDE, Some(ACT), rows)
                        .map_err(es)
                })
                .map(|s| (s, "gemv_bf16_xf32_k4_rows"))
            } else {
                tm.time(gpu, |g| {
                    g.lane_wmma_gemm_fixed(
                        &LaneWmmaWeight::Bf16(&hc.down_bf),
                        &xn,
                        &y_down,
                        RANK,
                        WIDE,
                        rows,
                        1,
                        Some(ACT),
                    )
                    .map(|_| ())
                    .map_err(es)
                })
                .map(|s| (s, "lane_wmma_gemm_fixed"))
            };
            match r {
                Ok((st, l)) => out.timed(&name, "down", rows, "bf16", l, w_down, st, json!(null)),
                Err(e) => out.unavailable(&name, "down", rows, "bf16", "-", &e),
            }
        }
        // ---- up incumbent ----
        {
            let name = tname("up");
            let (r, l) = if rows == 1 {
                let wr = wref(&hc.up_bf, WIDE, RANK);
                (
                    tm.time(gpu, |g| {
                        project_weight(g, &wr, &low, &up, 1, None).map_err(es)?;
                        hyper_read_projected(
                            g,
                            &HyperReadProjected {
                                input: &streams,
                                norm_weight: &hc.norm,
                                up: &up,
                                normalized: &xn,
                                mixed: &mixed,
                                branches: BRANCHES,
                                hidden: HIDDEN,
                            },
                        )
                        .map_err(es)
                    }),
                    "project_weight+hyper_read_projected",
                )
            } else if rows <= 8 {
                (
                    tm.time(gpu, |g| {
                        hyper_read_up_fused(
                            g,
                            &HyperReadUpFused {
                                up_weight: &hc.up_bf,
                                low: &low,
                                normalized: &xn,
                                mixed: &mixed,
                                rows,
                                hidden: HIDDEN,
                                low_rank: RANK,
                                normalized_bf16: false,
                            },
                        )
                        .map_err(es)
                    }),
                    "hyper_read_up_fused",
                )
            } else {
                (
                    tm.time(gpu, |g| {
                        lane_wmma_read_up(
                            g,
                            &LaneWmmaWeight::Bf16(&hc.up_bf),
                            &low,
                            &xn,
                            &mixed,
                            rows,
                            HIDDEN,
                            RANK,
                            1,
                        )
                        .map(|_| ())
                        .map_err(es)
                    }),
                    "lane_wmma_read_up",
                )
            };
            match r {
                Ok(st) => out.timed(&name, "up", rows, "bf16", l, w_up, st, json!(null)),
                Err(e) => out.unavailable(&name, "up", rows, "bf16", "-", &e),
            }
        }
        let mixed_bf = fetch(gpu, &mixed, rows * HIDDEN)?;

        // ---- hc8 arms ----
        for (qi, q) in qs.iter().enumerate() {
            let arm = q.fmt.name();
            for g in GEOMS {
                let gs = format!("mw{}kw{}", g.mw, g.kw);
                // down, act epilogue
                let name = tname("down");
                match tm.time(gpu, |gp| {
                    hc8_gemm(gp, &q.down, &xn, &y_down, rows, Some(ACT), g)
                        .map(|_| ())
                        .map_err(es)
                }) {
                    Err(e) => out.unavailable(&name, "down", rows, arm, &gs, &e),
                    Ok(st) => {
                        let ya = fetch(gpu, &y_down, rows * RANK)?;
                        let e_act = max_rel(&ya, &refs[qi].down_act[..rows * RANK]);
                        hc8_gemm(gpu, &q.down, &xn, &y_down, rows, None, g).map_err(es)?;
                        let ys = fetch(gpu, &y_down, rows * RANK)?;
                        let e_set = max_rel(&ys, &refs[qi].down_set[..rows * RANK]);
                        let mut ok = e_act < TOL_ACT && e_set < TOL_SET;
                        let mut inv = Value::Null;
                        if INV_ROWS.contains(&rows) {
                            let bad = invariance(gpu, &q.down, &xn1, &ya, rows, Some(ACT), g)?;
                            ok &= bad == 0;
                            inv = json!(bad);
                        }
                        let chk = json!({"max_rel_act": e_act, "max_rel_set": e_set,
                            "row_inv_bad_rows": inv, "pass": ok});
                        if !ok {
                            out.fails
                                .push(format!("{name} {arm} {gs} rows={rows}: {chk}"));
                        }
                        out.timed(&name, "down", rows, arm, &gs, q.down.bytes(), st, chk);
                    }
                }
                // up, set epilogue (GEMM only), then GEMM + hyper_read_projected
                let name = tname("up");
                match tm.time(gpu, |gp| {
                    hc8_gemm(gp, &q.up, &low, &up, rows, None, g)
                        .map(|_| ())
                        .map_err(es)
                }) {
                    Err(e) => out.unavailable(&name, "up", rows, arm, &gs, &e),
                    Ok(st) => {
                        let yu = fetch(gpu, &up, rows * WIDE)?;
                        let e_set = max_rel(&yu, &refs[qi].up_set[..rows * WIDE]);
                        let mut ok = e_set < TOL_SET;
                        let mut inv = Value::Null;
                        if INV_ROWS.contains(&rows) {
                            let bad = invariance(gpu, &q.up, &low1, &yu, rows, None, g)?;
                            ok &= bad == 0;
                            inv = json!(bad);
                        }
                        let chk =
                            json!({"max_rel_set": e_set, "row_inv_bad_rows": inv, "pass": ok});
                        if !ok {
                            out.fails
                                .push(format!("{name} {arm} {gs} rows={rows}: {chk}"));
                        }
                        out.timed(&name, "up", rows, arm, &gs, q.up.bytes(), st, chk);
                        let full = tm.time(gpu, |gp| {
                            hc8_gemm(gp, &q.up, &low, &up, rows, None, g).map_err(es)?;
                            hyper_read_projected(
                                gp,
                                &HyperReadProjected {
                                    input: &streams,
                                    norm_weight: &hc.norm,
                                    up: &up,
                                    normalized: &xn,
                                    mixed: &mixed2,
                                    branches: BRANCHES,
                                    hidden: HIDDEN,
                                },
                            )
                            .map_err(es)
                        });
                        match full {
                            Ok(st) => {
                                let m2 = fetch(gpu, &mixed2, rows * HIDDEN)?;
                                let d = max_rel(&m2, &mixed_bf);
                                out.timed(
                                    &name,
                                    "up_full",
                                    rows,
                                    arm,
                                    &gs,
                                    q.up.bytes(),
                                    st,
                                    json!({"mixed_max_rel_vs_bf16_incumbent": d}),
                                );
                            }
                            Err(e) => out.unavailable(&name, "up_full", rows, arm, &gs, &e),
                        }
                    }
                }
            }
        }
        for t in [xn, low, streams, y_down, up, mixed, mixed2] {
            let _ = gpu.free_tensor(t);
        }
    }
    for t in xn1.into_iter().chain(low1).chain([streams_all, norm_all]) {
        let _ = gpu.free_tensor(t);
    }
    let _ = entry_name;
    Ok(())
}

/// Best (lowest median) arm per rows class for each (tensor, family), over hc8 arms and incumbents.
fn summarize(recs: &[Value]) -> Vec<Value> {
    let classes: [(&str, fn(u64) -> bool); 3] = [
        ("1", |r| r == 1),
        ("2-8", |r| (2..=8).contains(&r)),
        ("16-64", |r| r >= 16),
    ];
    let mut out = vec![];
    for tensor in ["attn_down", "attn_up", "mlp_down", "mlp_up"] {
        let fams: &[&str] = if tensor.ends_with("up") {
            &["up", "up_full"]
        } else {
            &["down"]
        };
        for fam in fams {
            for (cname, pred) in classes {
                // Mean median over the class's rows per (arm, geom); incumbents are always complete.
                let mut agg: std::collections::BTreeMap<(String, String), (f64, usize)> =
                    Default::default();
                let mut nrows = std::collections::BTreeSet::new();
                for r in recs {
                    if r["tensor"] != tensor || r["status"] != "ok" {
                        continue;
                    }
                    let f = r["family"].as_str().unwrap_or("");
                    // up_full competes against the bf16 `up` incumbent (the whole route).
                    if !(f == *fam || (*fam == "up_full" && f == "up" && r["arm"] == "bf16")) {
                        continue;
                    }
                    if *fam == "up" && f == "up_full" {
                        continue;
                    }
                    let rows = r["rows"].as_u64().unwrap_or(0);
                    if !pred(rows) {
                        continue;
                    }
                    nrows.insert(rows);
                    let e = agg
                        .entry((
                            r["arm"].as_str().unwrap().into(),
                            r["geom"].as_str().unwrap().into(),
                        ))
                        .or_insert((0.0, 0));
                    e.0 += r["med_us"].as_f64().unwrap();
                    e.1 += 1;
                }
                let best = agg
                    .iter()
                    .filter(|(_, v)| v.1 == nrows.len())
                    .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
                if let Some(((arm, geom), (sum, n))) = best {
                    println!(
                        "BEST {tensor:<9} {fam:<8} rows {cname:<5} -> {arm} {geom}  mean med {:.2} us over {n} row counts",
                        sum / *n as f64
                    );
                    out.push(json!({"tensor": tensor, "family": fam, "rows_class": cname,
                        "arm": arm, "geom": geom, "mean_med_us": sum / *n as f64}));
                }
            }
        }
    }
    out
}

fn run() -> R<()> {
    let cli = parse_cli()?;
    let mut gpu = Gpu::init().map_err(es)?;
    println!(
        "hc8_screen: arch={} layer={} rows={:?} calls={} (cold, {} MiB flush)",
        gpu.arch,
        cli.layer,
        cli.rows,
        cli.calls,
        FLUSH_BYTES >> 20
    );
    if gpu.arch != "gfx1201" {
        return Err(format!("needs exact gfx1201, arch is {}", gpu.arch));
    }
    let hfq = Hfq(HfqFile::open(Path::new(&cli.model)).map_err(|e| format!("open: {e}"))?);
    let flush = gpu.hip.malloc(FLUSH_BYTES).map_err(es)?;
    let tm = Timer {
        flush,
        calls: cli.calls,
        warm: cli.warm,
    };
    let streams_h = normals(0x57AE_0001, MAX_ROWS * WIDE, 1.0);
    let low_h = normals(0x57AE_0004, MAX_ROWS * RANK, 0.5);
    let mut out = Out::default();
    for name in ["attn", "mlp"] {
        let p = format!("{name}_hyper_connection");
        let down_b = hfq.bf16(
            cli.layer,
            &format!("{p}.input_mix_weight_down.weight"),
            &[RANK, WIDE],
        )?;
        let up_b = hfq.bf16(
            cli.layer,
            &format!("{p}.input_mix_weight_up.weight"),
            &[WIDE, RANK],
        )?;
        let norm_b = hfq.bf16(cli.layer, &format!("{p}.hc_norm.weight"), &[WIDE])?;
        let hc = Hc {
            name: if name == "attn" { "attn" } else { "mlp" },
            down_bf: upload_bf16_bytes(&mut gpu, &down_b)?,
            up_bf: upload_bf16_bytes(&mut gpu, &up_b)?,
            norm: upload_bf16_bytes(&mut gpu, &norm_b)?,
            down_h: bf16_bytes_to_u16(&down_b),
            up_h: bf16_bytes_to_u16(&up_b),
        };
        let qs: Vec<Q> = cli
            .fmts
            .iter()
            .map(|&f| build_q(&mut gpu, &hc, f))
            .collect::<R<_>>()?;
        for q in &qs {
            // Weight-only quantization error (vs BF16 weights), for the record.
            let bf: Vec<f32> = hc.down_h.iter().map(|&h| bf16_to_f32(h)).collect();
            println!(
                "{name} {} down weight max_rel vs bf16: {:.3e}",
                q.fmt.name(),
                max_rel(&q.down_dq, &bf)
            );
        }
        run_hc(&mut gpu, &tm, &mut out, &cli, &hc, &qs, &streams_h, &low_h)?;
    }
    let best = summarize(&out.recs);
    let pass = out.fails.is_empty();
    println!("CORRECTNESS {}", if pass { "PASS" } else { "FAIL" });
    for f in &out.fails {
        println!("FAIL {f}");
    }
    if let Some(o) = &cli.out {
        if let Some(d) = o.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).map_err(es)?;
        }
        let doc = json!({
            "tool": "hc8_screen", "arch": gpu.arch, "model": cli.model.display().to_string(),
            "layer": cli.layer, "rows": cli.rows, "calls": cli.calls, "cold_flush_bytes": FLUSH_BYTES,
            "peak_gbps": PEAK_GBPS, "tol_set": TOL_SET, "tol_act": TOL_ACT,
            "image_sha256": hc8::IMAGE_SHA256.trim(),
            "correctness_pass": pass, "failures": out.fails, "best": best, "arms": out.recs,
        });
        std::fs::write(o, serde_json::to_string_pretty(&doc).map_err(es)?).map_err(es)?;
        println!("wrote {}", o.display());
    }
    if pass {
        Ok(())
    } else {
        Err("correctness FAIL".into())
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hc8_screen: {e}");
        std::process::exit(1);
    }
}
