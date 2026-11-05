// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane WMMA screen: the purpose-built WMMA kernels (`rdna_compute::lane_wmma` and `lane_wmma_lab`, BF16 and planar W8 weights, 1-3 BF16
//! activation split terms) against today's incumbents, on ONE real layer of an HFQM artifact, exact gfx1201.
//!
//! ```text
//! hcwmma_screen --model PATH [--layer 4] [--rows 1,8,16,32,48,4096] [--batches 9] [--calls 40] [--cold]
//!               [--no-prefill] [--hcw-nn 1|2] [--hcw-split K] [--terms 1,2,3] [--out FILE.json]
//! ```
//!
//! Families / arms (per row count; rows 9..=64 run under `Gpu::set_wide_verify_rows(rows)`):
//! * `hc_down` `[320, 10240]`, act 0.25: `incumbent` (rows <= 64 `gemv_bf16_xf32_k4_rows`; rows > 64
//!   `gemm_bf16_xf16_hc_down_fused` on the F16 input), `hcw_bf16_t1`, `hcw_w8_t1` (`lane_wmma_gemm`), rows > 64
//!   also `hcw_w8_prefill` (`lane_wmma_hc_down_fused_w8`).
//! * `hc_up` `[10240, 320]` + branch mix: `incumbent` (rows <= 64 `hyper_read_up_fused`; rows > 64
//!   `hyper_read_up_wmma`), `hcw_bf16_t1`, `hcw_w8_t1` (`lane_wmma_read_up`), rows > 64 also `hcw_w8_prefill`
//!   (`lane_wmma_hc_up_w8`).
//! * `hc_gate` `[4, 10240]`, no act: `incumbent` vs `hcw_{bf16,w8}_t1`.
//! * `shared_gate` / `shared_up` / `shared_down`: `incumbent` (`project_weight` BF16) vs
//!   `hcw_{bf16,w8}_t{terms}` (`--terms`, default 1,2,3; the shared X is NOT BF16-exact).
//!
//! The HC arms always use one split term (their X is BF16-exact); `--terms` selects the shared-expert arms.
//!
//! Inputs are fixed-seed. Errors (first `min(rows, 64)` rows, f64 CPU reference, the arm's epilogue applied in
//! f32 on the rounded dot): `max_rel = max|y - ref| / max|ref|`, `mean_rel = mean|y - ref| / mean|ref|`
//! against the reference on the weights the arm uses (BF16 values, or `q * d` for W8) and, for W8 arms, also
//! against the BF16-weight reference (quantization error). Errors are on the final output (post epilogue).
//! `flips` = outputs whose f32 bits differ from the incumbent's. Timing: `--batches` batches of `--calls`
//! back-to-back calls with one hipEvent pair per batch. A launcher that refuses prints `UNAVAILABLE <reason>`.
//!
//! The W8 planar weights are built from the product quantizer (`Gpu::quantize_bf16_q8_0`) output through
//! `lane_wmma_lab::w8_planar_from_q8_0`.
//!
//! Run one card, one fresh process:
//! `HSA_USERPTR_FOR_PAGED_MEM=0 GPU_PINNED_MIN_XFER_SIZE=100000 hcwmma_screen --model ... --out ...`

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use half::f16;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::project_weight;
use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use rdna_compute::lane_wmma::{lane_wmma_gemm, lane_wmma_read_up, LaneWmmaCfg, LaneWmmaLaunch, LaneWmmaScratch, LaneWmmaWeight};
use rdna_compute::lane_wmma_lab::{lane_wmma_hc_down_fused_w8, lane_wmma_hc_up_w8, w8_planar_from_q8_0};
use rdna_compute::tensor_ops::{hyper_norm, hyper_read_up_fused, hyper_read_up_wmma, HyperNorm, HyperReadUpFused};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult, LabLaunchWitness};
use serde_json::{json, Value};

type R<T> = Result<T, String>;

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = BRANCHES * HIDDEN;
const RANK: usize = 320;
const HC_SCALE: f32 = 0.25;
/// Rank-0 shard of the EP2 shared expert: intermediate rows `[0, SH_I)` of 640.
const SH_FULL_I: usize = 640;
const SH_I: usize = 384;
/// Rows up to this are the decode / verify family; above it the prefill family.
const SMALL_ROWS: usize = 64;
/// Rows of the f64 CPU reference.
const REF_ROWS: usize = 64;

fn es<E: Debug>(e: E) -> String {
    format!("{e:?}")
}

// ---- host helpers ---------------------------------------------------------------------------

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

fn bf16_bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn normals(rng: &mut Rng, n: usize, bf16_exact: bool) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v = rng.normal();
            if bf16_exact {
                bf16r(v)
            } else {
                v
            }
        })
        .collect()
}

/// Dequantized `[m, k]` Q8_0 bytes (`{f16 scale, i8 q[32]}` blocks, row-major).
fn dequant_q8_0(raw: &[u8], m: usize, k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * k);
    for block in raw[..m * k / 32 * 34].chunks_exact(34) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        out.extend(block[2..].iter().map(|&q| scale * f32::from(q as i8)));
    }
    out
}

/// `(max_abs, rel_rms)` of `a` against `r` over the common prefix.
fn err_stats(a: &[f32], r: &[f32]) -> (f64, f64) {
    let n = a.len().min(r.len());
    let (mut max_abs, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let d = f64::from(a[i]) - f64::from(r[i]);
        if !d.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        max_abs = max_abs.max(d.abs());
        num += d * d;
        den += f64::from(r[i]) * f64::from(r[i]);
    }
    (max_abs, if den > 0.0 { (num / den).sqrt() } else { f64::NAN })
}

/// `(max_rel, mean_rel)` = `(max|a - r| / max|r|, mean|a - r| / mean|r|)` over the common prefix.
fn rel_err(a: &[f32], r: &[f32]) -> (f64, f64) {
    let n = a.len().min(r.len());
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let (mut max_d, mut sum_d, mut max_r, mut sum_r) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let d = (f64::from(a[i]) - f64::from(r[i])).abs();
        if !d.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let rv = f64::from(r[i]).abs();
        max_d = max_d.max(d);
        sum_d += d;
        max_r = max_r.max(rv);
        sum_r += rv;
    }
    (
        if max_r > 0.0 { max_d / max_r } else { f64::NAN },
        if sum_r > 0.0 { sum_d / sum_r } else { f64::NAN },
    )
}

fn flips(a: &[f32], inc: &[f32]) -> usize {
    a.iter()
        .zip(inc)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// f64-accumulated `x . w^T`: `w [m, k]`, `x [nr, k]` -> `[nr, m]`.
fn dot_ref(w: &[f32], m: usize, k: usize, x: &[f32], nr: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nr * m];
    out.par_chunks_mut(m).enumerate().for_each(|(t, o)| {
        let xr = &x[t * k..(t + 1) * k];
        for (c, slot) in o.iter_mut().enumerate() {
            let wr = &w[c * k..(c + 1) * k];
            *slot = wr
                .iter()
                .zip(xr)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
        }
    });
    out
}

/// `hc_act` epilogue (gemv_bf16_xf32_k4): all f32.
fn hc_act_ref(v: f32, scale: f32) -> f32 {
    let mut v = bf16r(v);
    v *= scale;
    v = bf16r(v);
    v /= 1.0 + (-v).exp();
    bf16r(v)
}

/// HC up + mix epilogue on the raw dots `h [nr, WIDE]` (row `br * HIDDEN + j`); `normalized [nr, WIDE]`.
fn up_mix_ref(h: &[f64], normalized: &[f32], nr: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nr * HIDDEN];
    out.par_chunks_mut(HIDDEN).enumerate().for_each(|(t, o)| {
        for (j, slot) in o.iter_mut().enumerate() {
            let mut value = 0.0f32;
            for br in 0..BRANCHES {
                let gate_input = bf16r(h[t * WIDE + br * HIDDEN + j] as f32);
                let gate = bf16r(1.0 / (1.0 + (-gate_input).exp()));
                let normed = bf16r(normalized[t * WIDE + br * HIDDEN + j]);
                value += bf16r(gate * normed);
            }
            *slot = bf16r(value / 4.0);
        }
    });
    out
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

// ---- device helpers -------------------------------------------------------------------------

fn upload_u16(gpu: &mut Gpu, v: &[u16], dtype: DType) -> R<GpuTensor> {
    let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[v.len()]).map_err(es)?;
    t.dtype = dtype;
    t.shape = vec![v.len()];
    Ok(t)
}

fn upload_bf16_bytes(gpu: &mut Gpu, bytes: &[u8]) -> R<GpuTensor> {
    let mut t = gpu.upload_raw(bytes, &[bytes.len() / 2]).map_err(es)?;
    t.dtype = DType::BF16;
    t.shape = vec![bytes.len() / 2];
    Ok(t)
}

fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> R<GpuTensor> {
    gpu.upload_f32(v, &[v.len()]).map_err(es)
}

fn alloc(gpu: &mut Gpu, n: usize, dtype: DType) -> R<GpuTensor> {
    gpu.alloc_tensor(&[n], dtype).map_err(es)
}

fn free_all(gpu: &mut Gpu, tensors: Vec<GpuTensor>) {
    for t in tensors {
        let _ = gpu.free_tensor(t);
    }
}

/// First `n` elements of an F32 or BF16 tensor as f32.
fn fetch(gpu: &Gpu, t: &GpuTensor, n: usize) -> R<Vec<f32>> {
    gpu.hip.device_synchronize().map_err(es)?;
    match t.dtype {
        DType::F32 => Ok(gpu.download_f32(t).map_err(es)?[..n].to_vec()),
        DType::BF16 => {
            let bytes = gpu.download_raw_bytes(t).map_err(es)?;
            Ok(bf16_bytes_to_f32(&bytes[..n * 2]))
        }
        other => Err(format!("fetch: unsupported dtype {other:?}")),
    }
}

fn wref(buf: &GpuTensor, dtype: DType, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype,
        m,
        k,
        row_stride: dtype.row_bytes(k).unwrap_or(k * dtype.size()),
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn bf16_weight_bytes(m: usize, k: usize) -> usize {
    m * k * 2
}

/// Planar W8: `q` (m*k bytes) + `d` (m*k/32 f16).
fn w8_weight_bytes(m: usize, k: usize) -> usize {
    m * k + m * (k / 32) * 2
}

fn q8_weight_bytes(m: usize, k: usize) -> usize {
    m * (k / 32) * 34
}

fn launch_note(l: &LaneWmmaLaunch) -> String {
    format!("{} grid{:?} block{:?}", l.kernel, l.grid, l.block)
}

// ---- launch capture -------------------------------------------------------------------------

#[derive(Default)]
struct Capture {
    on: AtomicBool,
    launches: Mutex<Vec<String>>,
}

impl Capture {
    fn start(&self) {
        self.launches.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.on.store(true, Ordering::Relaxed);
    }
    fn stop(&self) -> Vec<String> {
        self.on.store(false, Ordering::Relaxed);
        std::mem::take(&mut *self.launches.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// Consecutive duplicates collapsed: `sym[g]/b x6`.
fn compact_launches(v: &[String]) -> Vec<String> {
    let mut out: Vec<(String, usize)> = Vec::new();
    for s in v {
        match out.last_mut() {
            Some((last, n)) if last == s => *n += 1,
            _ => out.push((s.clone(), 1)),
        }
    }
    out.into_iter()
        .map(|(s, n)| if n > 1 { format!("{s} x{n}") } else { s })
        .collect()
}

// ---- results --------------------------------------------------------------------------------

/// Error references for one arm (first `min(rows, 64)` rows of the output).
#[derive(Clone, Copy)]
struct Check<'a> {
    /// Reference on the weights the arm uses.
    exact: &'a [f32],
    /// BF16-weight reference (W8 arms only): the quantization error.
    bf16: Option<&'a [f32]>,
}

struct Spec<'a> {
    family: &'a str,
    rows: usize,
    arm: &'a str,
    launcher: &'a str,
    weight_bytes: usize,
    scope: Option<usize>,
    chk: Option<Check<'a>>,
    /// This arm's output is the flips baseline of its family/rows.
    inc: bool,
}

struct Ctx {
    batches: usize,
    calls: usize,
    cap: Arc<Capture>,
    /// `family/rows` -> incumbent output (f32 bits compared for `flips`).
    inc: HashMap<String, Vec<f32>>,
    recs: Vec<Value>,
    /// `--cold`: decode-family arms cycle through enough weight copies to exceed the MALL.
    cold: bool,
    /// Weight copies the next `arm` call cycles through (set just before it; reset to 1 by it).
    cold_n: usize,
    /// Set by an arm closure to the launcher's `LaneWmmaLaunch` description.
    launch_note: Rc<RefCell<Option<String>>>,
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic".into()
    }
}

fn time_arm<F: FnMut(&mut Gpu) -> R<()>>(
    gpu: &mut Gpu,
    f: &mut F,
    calls: usize,
    batches: usize,
) -> R<(f64, f64)> {
    for _ in 0..calls {
        f(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(es)?;
    let mut v = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = gpu.hip.event_create().map_err(es)?;
        let stop = gpu.hip.event_create().map_err(es)?;
        gpu.hip.event_record(&start, None).map_err(es)?;
        for _ in 0..calls {
            f(gpu)?;
        }
        gpu.hip.event_record(&stop, None).map_err(es)?;
        gpu.hip.event_synchronize(&stop).map_err(es)?;
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).map_err(es)? as f64;
        gpu.hip.event_destroy(start).map_err(es)?;
        gpu.hip.event_destroy(stop).map_err(es)?;
        v.push(ms * 1000.0 / calls as f64);
    }
    v.sort_by(f64::total_cmp);
    let min = v[0];
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    Ok((min, med))
}

fn fmt_opt(e: Option<(f64, f64)>) -> String {
    match e {
        Some((a, b)) => format!("{a:.3e}/{b:.3e}"),
        None => "-".into(),
    }
}

impl Ctx {
    fn unavailable(&mut self, s: &Spec<'_>, reason: &str) {
        println!(
            "{:<13} rows={:<5} {:<20} UNAVAILABLE {reason}",
            s.family, s.rows, s.arm
        );
        self.recs.push(json!({
            "family": s.family, "rows": s.rows, "arm": s.arm, "launcher": s.launcher,
            "status": "UNAVAILABLE", "reason": reason,
            "weight_bytes": s.weight_bytes, "scope": s.scope,
        }));
    }

    /// Probe (first call, launch capture, panics caught), output + errors + flips, then the batched timing.
    /// `out = (tensor, n)`: the first `n` elements are fetched (all of them are compared for flips; the
    /// reference prefix for the errors).
    fn arm<F: FnMut(&mut Gpu) -> R<()>>(
        &mut self,
        gpu: &mut Gpu,
        s: Spec<'_>,
        out: Option<(&GpuTensor, usize)>,
        mut f: F,
    ) -> Option<(f64, f64)> {
        let cold_n = std::mem::replace(&mut self.cold_n, 1);
        *self.launch_note.borrow_mut() = None;
        let prior = s.scope.map(|r| gpu.set_wide_verify_rows(r));
        self.cap.start();
        let probe = catch_unwind(AssertUnwindSafe(|| -> R<()> {
            f(gpu)?;
            gpu.hip.device_synchronize().map_err(es)
        }));
        let launches = self.cap.stop();
        let hcw_launch = self.launch_note.borrow_mut().take();
        let probe = match probe {
            Ok(r) => r,
            Err(p) => Err(format!("panic: {}", panic_msg(&p))),
        };
        let restore = |gpu: &mut Gpu| {
            if let Some(p) = prior {
                gpu.set_wide_verify_rows(p);
            }
        };
        if let Err(reason) = probe {
            restore(gpu);
            self.unavailable(&s, &reason);
            return None;
        }
        let key = format!("{}/{}", s.family, s.rows);
        let mut err_exact = None;
        let mut err_bf16 = None;
        let mut nflips = None;
        if let Some((t, n)) = out {
            match fetch(gpu, t, n) {
                Ok(vals) => {
                    if let Some(c) = s.chk {
                        err_exact = Some(rel_err(&vals, c.exact));
                        err_bf16 = c.bf16.map(|b| rel_err(&vals, b));
                    }
                    if s.inc {
                        nflips = Some(0);
                        self.inc.insert(key.clone(), vals);
                    } else if let Some(inc) = self.inc.get(&key) {
                        nflips = Some(flips(&vals, inc));
                    }
                }
                Err(e) => {
                    restore(gpu);
                    self.unavailable(&s, &format!("output fetch: {e}"));
                    return None;
                }
            }
        }
        let timed = time_arm(gpu, &mut f, self.calls, self.batches);
        restore(gpu);
        *self.launch_note.borrow_mut() = None;
        let (min_us, med_us) = match timed {
            Ok(t) => t,
            Err(e) => {
                self.unavailable(&s, &format!("timing: {e}"));
                return None;
            }
        };
        let gbps = s.weight_bytes as f64 / min_us / 1e3;
        let kernels = compact_launches(&launches);
        println!(
            "{:<13} rows={:<5} {:<20} min {min_us:>9.2} us  med {med_us:>9.2} us  W {:>9} B {gbps:>7.1} GB/s  err(max/mean rel) {} vsBF16w {}  flips {}  cold N={cold_n}",
            s.family,
            s.rows,
            s.arm,
            s.weight_bytes,
            fmt_opt(err_exact),
            fmt_opt(err_bf16),
            nflips.map_or("-".to_string(), |n| n.to_string()),
        );
        println!(
            "{:<13}   launcher: {}  | hcw launch: {}  | {} observed launch(es): {}",
            "",
            s.launcher,
            hcw_launch.as_deref().unwrap_or("-"),
            launches.len(),
            if kernels.is_empty() {
                "(none)".to_string()
            } else {
                kernels.join(", ")
            }
        );
        self.recs.push(json!({
            "family": s.family, "rows": s.rows, "arm": s.arm, "launcher": s.launcher,
            "status": "ok", "min_us": min_us, "med_us": med_us,
            "weight_bytes": s.weight_bytes, "eff_gbps_on_weight": gbps,
            "max_rel": err_exact.map(|e| e.0), "mean_rel": err_exact.map(|e| e.1),
            "max_rel_vs_bf16_weights": err_bf16.map(|e| e.0),
            "mean_rel_vs_bf16_weights": err_bf16.map(|e| e.1),
            "flips_vs_incumbent": nflips, "ref_rows": s.rows.min(REF_ROWS),
            "hcw_launch": hcw_launch,
            "launches": launches.len(), "kernels": kernels, "scope": s.scope, "cold_copies": cold_n,
        }));
        Some((min_us, med_us))
    }
}

// ---- model ----------------------------------------------------------------------------------

struct Hfq(HfqFile);

impl Hfq {
    /// Raw BF16 bytes of `suffix` of layer `layer`, `[m, k]` checked.
    fn bf16(&self, layer: usize, suffix: &str, shape: &[usize]) -> R<Vec<u8>> {
        let candidates = [
            format!("model.language_model.layers.{layer}.{suffix}"),
            format!("model.layers.{layer}.{suffix}"),
        ];
        for name in &candidates {
            if self.0.find_tensor_info(name).is_none() {
                continue;
            }
            let (info, data) = self
                .0
                .tensor_data_vec(name)
                .ok_or_else(|| format!("{name}: no data"))?;
            let dims: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
            let numel: usize = shape.iter().product();
            if dims != shape || data.len() != numel * 2 {
                return Err(format!(
                    "{name}: shape {dims:?} / {} B, expected {shape:?} BF16 ({} B)",
                    data.len(),
                    numel * 2
                ));
            }
            return Ok(data);
        }
        Err(format!("none of {candidates:?} in the artifact"))
    }
}

/// One device copy of a weight in both formats: BF16 `[m, k]`, planar W8 `q` (i8 bytes, `Raw`) and `d` (F16).
struct WCopy {
    bf: GpuTensor,
    q: GpuTensor,
    d: GpuTensor,
}

struct Wt {
    m: usize,
    k: usize,
    main: WCopy,
    /// Host f32 of the BF16 weight and of the W8 weight (`q * d`).
    h_bf: Vec<f32>,
    h_w8: Vec<f32>,
}

impl Wt {
    fn bf_bytes(&self) -> usize {
        bf16_weight_bytes(self.m, self.k)
    }
    fn w8_bytes(&self) -> usize {
        w8_weight_bytes(self.m, self.k)
    }
}

struct Weights {
    down: Wt,
    up: Wt,
    inj: Wt,
    gate: Wt,
    sup: Wt,
    sdown: Wt,
    norm: GpuTensor,
}

/// Upload BF16 `bytes` `[m, k]`, quantize on device with the product quantizer, repack to planar W8.
fn load_wt(gpu: &mut Gpu, name: &str, bytes: &[u8], m: usize, k: usize) -> R<(Wt, Value)> {
    let bf = upload_bf16_bytes(gpu, bytes)?;
    let q8 = gpu.quantize_bf16_q8_0(&bf, m, k).map_err(es)?;
    gpu.hip.device_synchronize().map_err(es)?;
    let mut raw = vec![0u8; q8_weight_bytes(m, k)];
    gpu.hip.memcpy_dtoh(&mut raw, &q8.buf).map_err(es)?;
    let _ = gpu.free_tensor(q8);
    let (qb, db) = w8_planar_from_q8_0(&raw, m, k);
    if qb.len() != m * k || db.len() != m * (k / 32) {
        return Err(format!(
            "{name}: w8_planar_from_q8_0 returned {} q bytes / {} d (expected {} / {})",
            qb.len(),
            db.len(),
            m * k,
            m * (k / 32)
        ));
    }
    let nb = k / 32;
    let h_w8: Vec<f32> = (0..m * k)
        .map(|i| {
            let d = f16::from_bits(db[(i / k) * nb + (i % k) / 32]).to_f32();
            f32::from(qb[i] as i8) * d
        })
        .collect();
    let dq = dequant_q8_0(&raw, m, k);
    let planar_mismatch = h_w8
        .iter()
        .zip(&dq)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let h_bf = bf16_bytes_to_f32(bytes);
    let (max_abs, rel_rms) = err_stats(&h_w8, &h_bf);
    println!(
        "W8 planar {name:<28} [{m}, {k}]  quant max_abs {max_abs:.4e} rel_rms {rel_rms:.4e}  planar-vs-Q8_0 mismatches {planar_mismatch}"
    );
    let q = gpu.upload_raw(&qb, &[qb.len()]).map_err(es)?;
    let d = upload_u16(gpu, &db, DType::F16)?;
    Ok((
        Wt {
            m,
            k,
            main: WCopy { bf, q, d },
            h_bf,
            h_w8,
        },
        json!({
            "tensor": name, "m": m, "k": k, "max_abs": max_abs, "rel_rms": rel_rms,
            "planar_vs_q8_0_mismatches": planar_mismatch,
        }),
    ))
}

fn load_weights(gpu: &mut Gpu, hfq: &Hfq, layer: usize) -> R<(Weights, Vec<Value>)> {
    let hc = "attn_hyper_connection";
    let down_b = hfq.bf16(layer, &format!("{hc}.input_mix_weight_down.weight"), &[RANK, WIDE])?;
    let up_b = hfq.bf16(layer, &format!("{hc}.input_mix_weight_up.weight"), &[WIDE, RANK])?;
    let norm_b = hfq.bf16(layer, &format!("{hc}.hc_norm.weight"), &[WIDE])?;
    let inject_b = hfq.bf16(layer, &format!("{hc}.block_inject_weight.weight"), &[4, WIDE])?;
    let se = "mlp.shared_expert";
    let gate_full = hfq.bf16(layer, &format!("{se}.gate_proj.weight"), &[SH_FULL_I, HIDDEN])?;
    let up_full = hfq.bf16(layer, &format!("{se}.up_proj.weight"), &[SH_FULL_I, HIDDEN])?;
    let down_full = hfq.bf16(layer, &format!("{se}.down_proj.weight"), &[HIDDEN, SH_FULL_I])?;

    // Rank 0 of EP2: gate/up rows [0, SH_I) are a contiguous prefix; down columns [0, SH_I) repack row by row.
    let gate_s = gate_full[..SH_I * HIDDEN * 2].to_vec();
    let up_s = up_full[..SH_I * HIDDEN * 2].to_vec();
    let mut down_s = Vec::with_capacity(HIDDEN * SH_I * 2);
    for r in 0..HIDDEN {
        let o = r * SH_FULL_I * 2;
        down_s.extend_from_slice(&down_full[o..o + SH_I * 2]);
    }

    println!("--- W8 planar weights (product Q8_0 quantizer -> w8_planar_from_q8_0) ---");
    let mut errs = Vec::new();
    let mut one = |gpu: &mut Gpu, name: &str, b: &[u8], m: usize, k: usize| -> R<Wt> {
        let (w, e) = load_wt(gpu, name, b, m, k)?;
        errs.push(e);
        Ok(w)
    };
    let down = one(gpu, "input_mix_weight_down", &down_b, RANK, WIDE)?;
    let up = one(gpu, "input_mix_weight_up", &up_b, WIDE, RANK)?;
    let inj = one(gpu, "block_inject_weight", &inject_b, 4, WIDE)?;
    let gate = one(gpu, "shared_expert.gate_proj[0:384]", &gate_s, SH_I, HIDDEN)?;
    let sup = one(gpu, "shared_expert.up_proj[0:384]", &up_s, SH_I, HIDDEN)?;
    let sdown = one(gpu, "shared_expert.down_proj[:,0:384]", &down_s, HIDDEN, SH_I)?;
    let norm = upload_bf16_bytes(gpu, &norm_b)?;
    Ok((
        Weights {
            down,
            up,
            inj,
            gate,
            sup,
            sdown,
            norm,
        },
        errs,
    ))
}

// ---- inputs ---------------------------------------------------------------------------------

struct Host {
    streams: Vec<f32>,
    xg: Vec<f32>,
    xd: Vec<f32>,
}

fn make_host(max_rows: usize) -> Host {
    Host {
        streams: normals(&mut Rng(0x57AE_0001), max_rows * WIDE, true),
        xg: normals(&mut Rng(0x57AE_0002), max_rows * HIDDEN, false),
        xd: normals(&mut Rng(0x57AE_0003), max_rows * SH_I, false),
    }
}

/// Bytes of distinct weights a cold decode arm cycles through (the MALL is 64 MB).
const COLD_BYTES: usize = 192 << 20;
/// Bound on extra copies of one tensor (tiny tensors would otherwise need thousands).
const COLD_MAX_COPIES: usize = 512;

/// Extra device copies (beyond the originals in [`Weights`]) of every decode-family weight (BF16 and planar W8
/// together), so that an arm reading `bytes` per call cycles through `ceil(COLD_BYTES / bytes)` distinct
/// buffers.
#[derive(Default)]
struct Cold {
    sets: HashMap<&'static str, Vec<WCopy>>,
}

impl Cold {
    /// The original followed by its extra copies: as many as `bytes_per_call` needs to cover `COLD_BYTES`
    /// (just the original when `on` is false).
    fn pick<'a>(&'a self, on: bool, key: &str, wt: &'a Wt, bytes_per_call: usize) -> Vec<&'a WCopy> {
        if !on {
            return vec![&wt.main];
        }
        let need = COLD_BYTES.div_ceil(bytes_per_call.max(1)).max(1);
        let extras = self.sets.get(key).map(|v| v.as_slice()).unwrap_or(&[]);
        std::iter::once(&wt.main).chain(extras.iter()).take(need).collect()
    }
}

fn download_bytes(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> R<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(es)?;
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &t.buf).map_err(es)?;
    Ok(raw)
}

/// Byte-identical device copies of every decode-family weight (BF16 + `q` + `d`).
fn build_cold(gpu: &mut Gpu, w: &Weights) -> R<Cold> {
    let mut sets = HashMap::new();
    let tensors: [(&'static str, &Wt); 6] = [
        ("down", &w.down),
        ("up", &w.up),
        ("inj", &w.inj),
        ("gate", &w.gate),
        ("sup", &w.sup),
        ("sdown", &w.sdown),
    ];
    for (key, wt) in tensors {
        let bf_raw = download_bytes(gpu, &wt.main.bf, wt.bf_bytes())?;
        let q_raw = download_bytes(gpu, &wt.main.q, wt.m * wt.k)?;
        let d_raw = download_bytes(gpu, &wt.main.d, wt.m * (wt.k / 32) * 2)?;
        let total = COLD_BYTES.div_ceil(wt.w8_bytes()).min(COLD_MAX_COPIES);
        let mut copies = Vec::with_capacity(total.saturating_sub(1));
        for _ in 1..total {
            let bf = upload_bf16_bytes(gpu, &bf_raw)?;
            let q = gpu.upload_raw(&q_raw, &[q_raw.len()]).map_err(es)?;
            let mut d = gpu.upload_raw(&d_raw, &[d_raw.len() / 2]).map_err(es)?;
            d.dtype = DType::F16;
            d.shape = vec![d_raw.len() / 2];
            copies.push(WCopy { bf, q, d });
        }
        println!(
            "cold: {key:<6} bf16 {:>9} B / w8 {:>9} B x {total} copies",
            wt.bf_bytes(),
            wt.w8_bytes()
        );
        sets.insert(key, copies);
    }
    Ok(Cold { sets })
}

// ---- scratch sizing -------------------------------------------------------------------------

fn auto_nn(m: usize) -> usize {
    if m >= 256 && m % 32 == 0 {
        2
    } else {
        1
    }
}

/// The launcher's auto split: the largest divisor `s` of `k / 128` with `tiles * s * ceil(rows / 64) <= 320`.
fn auto_split(m: usize, k: usize, rows: usize, nn: usize) -> usize {
    let tiles = m.div_ceil(16 * nn);
    let rb = rows.div_ceil(64);
    let kb = (k / 128).max(1);
    (1..=kb)
        .rev()
        .find(|s| kb % s == 0 && tiles * s * rb <= 320)
        .unwrap_or(1)
}

/// `(partial floats, counters)` large enough for every `lane_wmma_gemm` call this run makes.
fn scratch_need(cli: &Cli, rows_run: &[usize]) -> (usize, usize) {
    let shapes = [(RANK, WIDE), (4, WIDE), (SH_I, HIDDEN), (SH_I, HIDDEN), (HIDDEN, SH_I)];
    let (mut part, mut cnt) = (1usize, 1usize);
    for &(m, k) in &shapes {
        for &rows in rows_run {
            let nn = cli.nn.unwrap_or_else(|| auto_nn(m));
            let s = if cli.split > 0 { cli.split } else { auto_split(m, k, rows, nn) };
            part = part.max(s * rows * m);
            // Counters for nn = 1 bound the nn = 2 grid too.
            cnt = cnt.max(m.div_ceil(16) * rows.div_ceil(64));
        }
    }
    (part, cnt)
}

// ---- one row count --------------------------------------------------------------------------

macro_rules! launcher {
    ($($t:tt)*) => { &format!($($t)*) };
}

/// Inputs of one `lane_wmma_gemm` arm family.
struct Gm<'a> {
    family: &'a str,
    rows: usize,
    scope: Option<usize>,
    cold_on: bool,
    cold: &'a Cold,
    key: &'a str,
    wt: &'a Wt,
    x: &'a GpuTensor,
    y: &'a GpuTensor,
    act: Option<f32>,
    terms: &'a [usize],
    /// References over the first `min(rows, 64)` rows: BF16 weights and W8 weights.
    ref_bf: &'a [f32],
    ref_w8: &'a [f32],
}

fn gemm_arms(gpu: &mut Gpu, ctx: &mut Ctx, sc: &LaneWmmaScratch, cli: &Cli, g: &Gm<'_>) {
    let (m, k, rows) = (g.wt.m, g.wt.k, g.rows);
    let cfg = LaneWmmaCfg {
        nn: cli.nn.unwrap_or_else(|| auto_nn(m)),
        split_k: cli.split,
    };
    for fmt in ["bf16", "w8"] {
        let bf = fmt == "bf16";
        for &t in g.terms {
            let arm = format!("hcw_{fmt}_t{t}");
            let wbytes = if bf { g.wt.bf_bytes() } else { g.wt.w8_bytes() };
            let list = g.cold.pick(g.cold_on, g.key, g.wt, wbytes);
            let mut i = 0usize;
            let ln = ctx.launch_note.clone();
            ctx.cold_n = list.len();
            let (x, y, act) = (g.x, g.y, g.act);
            ctx.arm(
                gpu,
                Spec {
                    family: g.family,
                    rows,
                    arm: &arm,
                    launcher: launcher!(
                        "lane_wmma_lab::lane_wmma_gemm({fmt}, terms {t}, [{m},{k}], act {act:?}, nn {} split {})",
                        cfg.nn,
                        cfg.split_k
                    ),
                    weight_bytes: wbytes,
                    scope: g.scope,
                    chk: Some(Check {
                        exact: if bf { g.ref_bf } else { g.ref_w8 },
                        bf16: if bf { None } else { Some(g.ref_bf) },
                    }),
                    inc: false,
                },
                Some((y, rows * m)),
                |gp| {
                    let c = list[i % list.len()];
                    i += 1;
                    let w = if bf {
                        LaneWmmaWeight::Bf16(&c.bf)
                    } else {
                        LaneWmmaWeight::W8 { q: &c.q, d: &c.d }
                    };
                    let l = lane_wmma_gemm(gp, &w, x, y, m, k, rows, t, act, cfg, sc).map_err(es)?;
                    *ln.borrow_mut() = Some(launch_note(&l));
                    Ok(())
                },
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_rows(
    gpu: &mut Gpu,
    ctx: &mut Ctx,
    sc: &LaneWmmaScratch,
    cli: &Cli,
    w: &Weights,
    cold: &Cold,
    host: &Host,
    rows: usize,
) -> R<()> {
    let small = rows <= SMALL_ROWS;
    let nr = rows.min(REF_ROWS);
    let scope = (rows > 8 && rows <= SMALL_ROWS).then_some(rows);
    let cold_on = ctx.cold && small;
    ctx.inc.clear();
    println!("=== rows {rows} ({} family, scope {scope:?}, ref rows {nr}) ===", if small { "decode" } else { "prefill" });
    if let Some(r) = scope {
        let prior = gpu.set_wide_verify_rows(r);
        println!(
            "wide scope {r}: few_row_limit {} lane_hc_allrows_applies {} (flag qwen4_lane_hc_allrows_enabled {})",
            gpu.few_row_limit(),
            gpu.lane_hc_allrows_applies(r),
            gpu.flags.qwen4_lane_hc_allrows_enabled()
        );
        gpu.set_wide_verify_rows(prior);
    }

    let streams = upload_f32(gpu, &host.streams[..rows * WIDE])?;
    let norm32 = alloc(gpu, rows * WIDE, DType::F32)?;
    hyper_norm(
        gpu,
        &HyperNorm {
            input: &streams,
            norm_weight: &w.norm,
            normalized: &norm32,
            branches: BRANCHES,
            hidden: HIDDEN,
            state_bf16: false,
        },
    )
    .map_err(es)?;
    let rot = alloc(gpu, rows * WIDE, DType::F32)?;
    let mixed = alloc(gpu, rows * HIDDEN, DType::F32)?;
    let mut owned = vec![];
    let normh = fetch(gpu, &norm32, rows * WIDE)?;
    let norm_ref = &normh[..nr * WIDE];

    // ---- hc_down ------------------------------------------------------------------------------
    let raw_b = dot_ref(&w.down.h_bf, RANK, WIDE, norm_ref, nr);
    let raw_w = dot_ref(&w.down.h_w8, RANK, WIDE, norm_ref, nr);
    let down_rb: Vec<f32> = raw_b.iter().map(|&v| hc_act_ref(v as f32, HC_SCALE)).collect();
    let down_rw: Vec<f32> = raw_w.iter().map(|&v| hc_act_ref(v as f32, HC_SCALE)).collect();
    let mut low_host: Option<Vec<f32>> = None;
    let y_down = alloc(gpu, rows * RANK, DType::F32)?;
    {
        let (wb_b, wb_q) = (w.down.bf_bytes(), w.down.w8_bytes());
        if small {
            let y_b = alloc(gpu, rows * RANK, DType::F32)?;
            let dl = cold.pick(cold_on, "down", &w.down, wb_b);
            let mut ib = 0usize;
            ctx.cold_n = dl.len();
            let b = ctx.arm(
                gpu,
                Spec {
                    family: "hc_down",
                    rows,
                    arm: "incumbent",
                    launcher: launcher!("Gpu::gemv_bf16_xf32_k4_rows(w,x,y,{RANK},{WIDE},Some({HC_SCALE}),{rows})"),
                    weight_bytes: wb_b,
                    scope,
                    chk: Some(Check { exact: &down_rb, bf16: None }),
                    inc: true,
                },
                Some((&y_b, rows * RANK)),
                |g| {
                    let c = dl[ib % dl.len()];
                    ib += 1;
                    g.gemv_bf16_xf32_k4_rows(&c.bf, &norm32, &y_b, RANK, WIDE, Some(HC_SCALE), rows)
                        .map_err(es)
                },
            );
            if b.is_some() {
                low_host = fetch(gpu, &y_b, rows * RANK).ok();
            }
            owned.push(y_b);
        } else {
            // F16 copy of the normalized activations (what the F16 WMMA route's norm writes).
            let x16 = upload_u16(
                gpu,
                &normh.iter().map(|&v| f16::from_f32(v).to_bits()).collect::<Vec<_>>(),
                DType::F16,
            )?;
            let low16 = alloc(gpu, rows * RANK, DType::BF16)?;
            let low16_w8 = alloc(gpu, rows * RANK, DType::BF16)?;
            if gpu.qwen4_hc_down_fused_applies(RANK, WIDE, rows) {
                let ok = ctx.arm(
                    gpu,
                    Spec {
                        family: "hc_down",
                        rows,
                        arm: "incumbent",
                        launcher: launcher!("Gpu::gemm_bf16_xf16_hc_down_fused(w_bf16,x_f16,low_bf16,{RANK},{WIDE},{rows},{HC_SCALE})"),
                        weight_bytes: wb_b,
                        scope,
                        chk: Some(Check { exact: &down_rb, bf16: None }),
                        inc: true,
                    },
                    Some((&low16, rows * RANK)),
                    |g| {
                        g.gemm_bf16_xf16_hc_down_fused(&w.down.main.bf, &x16, &low16, RANK, WIDE, rows, HC_SCALE)
                            .map_err(es)
                    },
                );
                if ok.is_some() {
                    low_host = fetch(gpu, &low16, rows * RANK).ok();
                }
            } else {
                ctx.unavailable(
                    &Spec {
                        family: "hc_down",
                        rows,
                        arm: "incumbent",
                        launcher: "Gpu::gemm_bf16_xf16_hc_down_fused",
                        weight_bytes: wb_b,
                        scope,
                        chk: None,
                        inc: false,
                    },
                    "qwen4_hc_down_fused_applies(320, 10240, rows) is false",
                );
            }
            let ln = ctx.launch_note.clone();
            ctx.arm(
                gpu,
                Spec {
                    family: "hc_down",
                    rows,
                    arm: "hcw_w8_prefill",
                    launcher: launcher!("lane_wmma_lab::lane_wmma_hc_down_fused_w8(q,d,x_f16,low_bf16,{RANK},{WIDE},{rows},{HC_SCALE})"),
                    weight_bytes: wb_q,
                    scope,
                    chk: Some(Check { exact: &down_rw, bf16: Some(&down_rb) }),
                    inc: false,
                },
                Some((&low16_w8, rows * RANK)),
                |g| {
                    let l = lane_wmma_hc_down_fused_w8(
                        g, &w.down.main.q, &w.down.main.d, &x16, &low16_w8, RANK, WIDE, rows, HC_SCALE,
                    )
                    .map_err(es)?;
                    *ln.borrow_mut() = Some(launch_note(&l));
                    Ok(())
                },
            );
            owned.extend([x16, low16, low16_w8]);
        }
        gemm_arms(
            gpu,
            ctx,
            sc,
            cli,
            &Gm {
                family: "hc_down",
                rows,
                scope,
                cold_on,
                cold,
                key: "down",
                wt: &w.down,
                x: &norm32,
                y: &y_down,
                act: Some(HC_SCALE),
                terms: &[1],
                ref_bf: &down_rb,
                ref_w8: &down_rw,
            },
        );
    }

    // `low` for the up arms: the real down output when the incumbent produced it, else fixed-seed random
    // (BF16-rounded so the one-term arms stay exact).
    let low_host = low_host.unwrap_or_else(|| {
        println!("note: the down incumbent did not produce `low`; the up arms use random BF16-exact N(0, 0.5) low");
        normals(&mut Rng(0x57AE_0004 ^ rows as u64), rows * RANK, false)
            .iter()
            .map(|v| bf16r(v * 0.5))
            .collect()
    });
    let low32 = upload_f32(gpu, &low_host)?;

    // ---- hc_up --------------------------------------------------------------------------------
    {
        let (wb_b, wb_q) = (w.up.bf_bytes(), w.up.w8_bytes());
        let low_ref = &low_host[..nr * RANK];
        let up_rb = up_mix_ref(&dot_ref(&w.up.h_bf, WIDE, RANK, low_ref, nr), norm_ref, nr);
        let up_rw = up_mix_ref(&dot_ref(&w.up.h_w8, WIDE, RANK, low_ref, nr), norm_ref, nr);
        let n_out = rows * HIDDEN;
        if small {
            let ul = cold.pick(cold_on, "up", &w.up, wb_b);
            let mut i1 = 0usize;
            ctx.cold_n = ul.len();
            ctx.arm(
                gpu,
                Spec {
                    family: "hc_up",
                    rows,
                    arm: "incumbent",
                    launcher: "tensor_ops::hyper_read_up_fused (BF16 up, low F32)",
                    weight_bytes: wb_b,
                    scope,
                    chk: Some(Check { exact: &up_rb, bf16: None }),
                    inc: true,
                },
                Some((&mixed, n_out)),
                |g| {
                    let c = ul[i1 % ul.len()];
                    i1 += 1;
                    hyper_read_up_fused(
                        g,
                        &HyperReadUpFused {
                            up_weight: &c.bf,
                            low: &low32,
                            normalized: &norm32,
                            mixed: &mixed,
                            rows,
                            hidden: HIDDEN,
                            low_rank: RANK,
                            normalized_bf16: false,
                        },
                    )
                    .map_err(es)
                },
            );
        } else {
            // F16 normalized + packed BF16 low through `hyper_read_up_wmma` (the dispatch's route).
            let x16 = upload_u16(
                gpu,
                &normh.iter().map(|&v| f16::from_f32(v).to_bits()).collect::<Vec<_>>(),
                DType::F16,
            )?;
            let low16 = upload_u16(
                gpu,
                &low_host.iter().map(|&v| bf16_bits(v)).collect::<Vec<_>>(),
                DType::BF16,
            )?;
            let read = HyperReadUpFused {
                up_weight: &w.up.main.bf,
                low: &low16,
                normalized: &x16,
                mixed: &mixed,
                rows,
                hidden: HIDDEN,
                low_rank: RANK,
                normalized_bf16: false,
            };
            ctx.arm(
                gpu,
                Spec {
                    family: "hc_up",
                    rows,
                    arm: "incumbent",
                    launcher: "tensor_ops::hyper_read_up_wmma (BF16 up, packed BF16 low, F16 normalized)",
                    weight_bytes: wb_b,
                    scope,
                    chk: Some(Check { exact: &up_rb, bf16: None }),
                    inc: true,
                },
                Some((&mixed, n_out)),
                |g| hyper_read_up_wmma(g, &read).map_err(es),
            );
            let ln = ctx.launch_note.clone();
            ctx.arm(
                gpu,
                Spec {
                    family: "hc_up",
                    rows,
                    arm: "hcw_w8_prefill",
                    launcher: "lane_wmma_lab::lane_wmma_hc_up_w8 (planar W8 up, packed BF16 low, F16 normalized)",
                    weight_bytes: wb_q,
                    scope,
                    chk: Some(Check { exact: &up_rw, bf16: Some(&up_rb) }),
                    inc: false,
                },
                Some((&mixed, n_out)),
                |g| {
                    let l = lane_wmma_hc_up_w8(g, &w.up.main.q, &w.up.main.d, &read).map_err(es)?;
                    *ln.borrow_mut() = Some(launch_note(&l));
                    Ok(())
                },
            );
            owned.extend([x16, low16]);
        }
        for fmt in ["bf16", "w8"] {
            let bf = fmt == "bf16";
            let arm = format!("hcw_{fmt}_t1");
            let wbytes = if bf { wb_b } else { wb_q };
            let list = cold.pick(cold_on, "up", &w.up, wbytes);
            let mut i = 0usize;
            let ln = ctx.launch_note.clone();
            ctx.cold_n = list.len();
            ctx.arm(
                gpu,
                Spec {
                    family: "hc_up",
                    rows,
                    arm: &arm,
                    launcher: launcher!("lane_wmma_lab::lane_wmma_read_up({fmt}, terms 1, hidden {HIDDEN}, low_rank {RANK})"),
                    weight_bytes: wbytes,
                    scope,
                    chk: Some(Check {
                        exact: if bf { &up_rb } else { &up_rw },
                        bf16: if bf { None } else { Some(&up_rb) },
                    }),
                    inc: false,
                },
                Some((&mixed, n_out)),
                |g| {
                    let c = list[i % list.len()];
                    i += 1;
                    let wt = if bf {
                        LaneWmmaWeight::Bf16(&c.bf)
                    } else {
                        LaneWmmaWeight::W8 { q: &c.q, d: &c.d }
                    };
                    let l = lane_wmma_read_up(g, &wt, &low32, &norm32, &mixed, rows, HIDDEN, RANK, 1).map_err(es)?;
                    *ln.borrow_mut() = Some(launch_note(&l));
                    Ok(())
                },
            );
        }
    }

    // ---- hc_gate (block_inject, no act) -------------------------------------------------------
    {
        let gate_rb = to_f32(&dot_ref(&w.inj.h_bf, 4, WIDE, norm_ref, nr));
        let gate_rw = to_f32(&dot_ref(&w.inj.h_w8, 4, WIDE, norm_ref, nr));
        let y_inc = alloc(gpu, rows * 4, DType::F32)?;
        let y_hcw = alloc(gpu, rows * 4, DType::F32)?;
        let wb_b = w.inj.bf_bytes();
        let gl = cold.pick(cold_on, "inj", &w.inj, wb_b);
        let mut ig = 0usize;
        ctx.cold_n = gl.len();
        ctx.arm(
            gpu,
            Spec {
                family: "hc_gate",
                rows,
                arm: "incumbent",
                launcher: if small {
                    "Gpu::gemv_bf16_xf32_k4_rows(w,x,y,4,10240,None,rows)"
                } else {
                    "pipeline::project_weight(BF16 [4,10240])"
                },
                weight_bytes: wb_b,
                scope,
                chk: Some(Check { exact: &gate_rb, bf16: None }),
                inc: true,
            },
            Some((&y_inc, rows * 4)),
            |g| {
                let c = gl[ig % gl.len()];
                ig += 1;
                if small {
                    g.gemv_bf16_xf32_k4_rows(&c.bf, &norm32, &y_inc, 4, WIDE, None, rows)
                        .map_err(es)
                } else {
                    project_weight(g, &wref(&c.bf, DType::BF16, 4, WIDE), &norm32, &y_inc, rows, Some(&rot))
                        .map_err(es)
                }
            },
        );
        gemm_arms(
            gpu,
            ctx,
            sc,
            cli,
            &Gm {
                family: "hc_gate",
                rows,
                scope,
                cold_on,
                cold,
                key: "inj",
                wt: &w.inj,
                x: &norm32,
                y: &y_hcw,
                act: None,
                terms: &[1],
                ref_bf: &gate_rb,
                ref_w8: &gate_rw,
            },
        );
        owned.extend([y_inc, y_hcw]);
    }

    // ---- shared expert ------------------------------------------------------------------------
    {
        let xg = upload_f32(gpu, &host.xg[..rows * HIDDEN])?;
        let xd = upload_f32(gpu, &host.xd[..rows * SH_I])?;
        let projections: [(&str, &str, &Wt, &GpuTensor, &[f32]); 3] = [
            ("shared_gate", "gate", &w.gate, &xg, &host.xg[..nr * HIDDEN]),
            ("shared_up", "sup", &w.sup, &xg, &host.xg[..nr * HIDDEN]),
            ("shared_down", "sdown", &w.sdown, &xd, &host.xd[..nr * SH_I]),
        ];
        for (family, key, wt, x, xref) in projections {
            let (m, k) = (wt.m, wt.k);
            let rb = to_f32(&dot_ref(&wt.h_bf, m, k, xref, nr));
            let rw = to_f32(&dot_ref(&wt.h_w8, m, k, xref, nr));
            let y_inc = alloc(gpu, rows * m, DType::F32)?;
            let y_hcw = alloc(gpu, rows * m, DType::F32)?;
            let wb_b = wt.bf_bytes();
            let lb = cold.pick(cold_on, key, wt, wb_b);
            let mut ib = 0usize;
            ctx.cold_n = lb.len();
            ctx.arm(
                gpu,
                Spec {
                    family,
                    rows,
                    arm: "incumbent",
                    launcher: launcher!("pipeline::project_weight(BF16 [{m},{k}])"),
                    weight_bytes: wb_b,
                    scope,
                    chk: Some(Check { exact: &rb, bf16: None }),
                    inc: true,
                },
                Some((&y_inc, rows * m)),
                |g| {
                    let c = lb[ib % lb.len()];
                    ib += 1;
                    project_weight(g, &wref(&c.bf, DType::BF16, m, k), x, &y_inc, rows, Some(&rot)).map_err(es)
                },
            );
            gemm_arms(
                gpu,
                ctx,
                sc,
                cli,
                &Gm {
                    family,
                    rows,
                    scope,
                    cold_on,
                    cold,
                    key,
                    wt,
                    x,
                    y: &y_hcw,
                    act: None,
                    terms: &cli.terms,
                    ref_bf: &rb,
                    ref_w8: &rw,
                },
            );
            owned.extend([y_inc, y_hcw]);
        }
        owned.extend([xg, xd]);
    }

    owned.extend([streams, norm32, rot, mixed, low32, y_down]);
    free_all(gpu, owned);
    Ok(())
}

// ---- CLI ------------------------------------------------------------------------------------

struct Cli {
    model: PathBuf,
    layer: usize,
    rows: Vec<usize>,
    batches: usize,
    calls: usize,
    out: Option<PathBuf>,
    cold: bool,
    no_prefill: bool,
    /// `--hcw-nn`: forced n-tiles per block (None: the launcher's auto rule).
    nn: Option<usize>,
    /// `--hcw-split`: forced split-K (0: auto).
    split: usize,
    /// `--terms`: split terms of the shared-expert hcw arms.
    terms: Vec<usize>,
}

fn parse_cli() -> R<Cli> {
    let mut cli = Cli {
        model: PathBuf::new(),
        layer: 4,
        rows: vec![1, 8, 16, 32, 48, 4096],
        batches: 9,
        calls: 40,
        out: None,
        cold: false,
        no_prefill: false,
        nn: None,
        split: 0,
        terms: vec![1, 2, 3],
    };
    let mut have_model = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--model" => {
                cli.model = PathBuf::from(value("--model")?);
                have_model = true;
            }
            "--layer" => cli.layer = value("--layer")?.parse().map_err(es)?,
            "--rows" => {
                cli.rows = value("--rows")?
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| t.parse::<usize>().map_err(es))
                    .collect::<R<_>>()?;
            }
            "--batches" => cli.batches = value("--batches")?.parse().map_err(es)?,
            "--calls" => cli.calls = value("--calls")?.parse().map_err(es)?,
            "--cold" => cli.cold = true,
            "--no-prefill" => cli.no_prefill = true,
            "--hcw-nn" => cli.nn = Some(value("--hcw-nn")?.parse().map_err(es)?),
            "--hcw-split" => cli.split = value("--hcw-split")?.parse().map_err(es)?,
            "--terms" => {
                cli.terms = value("--terms")?
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .map(|t| t.parse::<usize>().map_err(es))
                    .collect::<R<_>>()?;
            }
            "--out" => cli.out = Some(PathBuf::from(value("--out")?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if !have_model {
        return Err("usage: hcwmma_screen --model PATH [--layer 4] [--rows 1,8,16,32,48,4096] [--batches 9] [--calls 40] [--cold] [--no-prefill] [--hcw-nn 1|2] [--hcw-split K] [--terms 1,2,3] [--out FILE.json]".into());
    }
    if cli.rows.is_empty() || cli.rows.contains(&0) || cli.batches == 0 || cli.calls == 0 {
        return Err("rows, batches and calls must be positive".into());
    }
    if cli.nn.is_some_and(|n| n != 1 && n != 2) {
        return Err("--hcw-nn must be 1 or 2".into());
    }
    if cli.terms.is_empty() || cli.terms.iter().any(|t| !(1..=3).contains(t)) {
        return Err("--terms must be a list of 1..=3".into());
    }
    Ok(cli)
}

fn run() -> R<()> {
    let cli = parse_cli()?;
    let mut gpu = Gpu::init().map_err(es)?;
    println!(
        "hcwmma_screen: arch={} model={} layer={} rows={:?} {} batches x {} calls nn={:?} split={} terms={:?}",
        gpu.arch,
        cli.model.display(),
        cli.layer,
        cli.rows,
        cli.batches,
        cli.calls,
        cli.nn,
        cli.split,
        cli.terms
    );
    if gpu.arch != "gfx1201" {
        println!("WARNING: this screen targets exact gfx1201; arch is {} and most arms will be UNAVAILABLE", gpu.arch);
    }
    let cap = Arc::new(Capture::default());
    let sink = cap.clone();
    let observer = Arc::new(
        move |_gpu: &Gpu, w: &LabLaunchWitness<'_>| -> HipResult<()> {
            if sink.on.load(Ordering::Relaxed) {
                sink.launches
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!(
                        "{}[{},{},{}]/{}",
                        w.symbol, w.grid[0], w.grid[1], w.grid[2], w.block[0]
                    ));
            }
            Ok(())
        },
    );
    gpu.set_lab_launch_observer(Some(observer));

    let hfq = Hfq(HfqFile::open(Path::new(&cli.model)).map_err(|e| format!("open {}: {e}", cli.model.display()))?);
    let (w, weight_errors) = load_weights(&mut gpu, &hfq, cli.layer)?;
    let rows_run: Vec<usize> = cli
        .rows
        .iter()
        .copied()
        .filter(|&r| !(cli.no_prefill && r > SMALL_ROWS))
        .collect();
    if rows_run.len() != cli.rows.len() {
        println!("--no-prefill: skipping prefill-family rows (> {SMALL_ROWS}); running {rows_run:?}");
    }
    if rows_run.is_empty() {
        return Err("--no-prefill left no rows to run".into());
    }
    let max_rows = *rows_run.iter().max().expect("rows");
    let host = make_host(max_rows);
    let (part, cnt) = scratch_need(&cli, &rows_run);
    println!("hcw scratch: {part} partial floats ({} MB), {cnt} counters", part * 4 >> 20);
    let scratch = LaneWmmaScratch::new(&mut gpu, part, cnt).map_err(|e| format!("LaneWmmaScratch::new: {e:?}"))?;
    let cold = if cli.cold && rows_run.iter().any(|&r| r <= SMALL_ROWS) {
        println!(
            "--cold: decode-family arms (rows <= {SMALL_ROWS}) cycle weight copies so each arm's working set is >= {} MB (MALL 64 MB); N printed per arm",
            COLD_BYTES >> 20
        );
        build_cold(&mut gpu, &w)?
    } else {
        Cold::default()
    };
    let mut ctx = Ctx {
        batches: cli.batches,
        calls: cli.calls,
        cap,
        inc: HashMap::new(),
        recs: Vec::new(),
        cold: cli.cold,
        cold_n: 1,
        launch_note: Rc::new(RefCell::new(None)),
    };
    for &rows in &rows_run {
        run_rows(&mut gpu, &mut ctx, &scratch, &cli, &w, &cold, &host, rows)?;
    }
    if let Some(out) = &cli.out {
        let doc = json!({
            "tool": "hcwmma_screen",
            "arch": gpu.arch,
            "model": cli.model.display().to_string(),
            "layer": cli.layer,
            "rows": rows_run,
            "cold": cli.cold,
            "no_prefill": cli.no_prefill,
            "batches": cli.batches,
            "calls": cli.calls,
            "hcw_nn": cli.nn,
            "hcw_split": cli.split,
            "terms": cli.terms,
            "scratch": { "partial_floats": part, "counters": cnt },
            "lane_hc_allrows_enabled": gpu.flags.qwen4_lane_hc_allrows_enabled(),
            "weight_only_error": weight_errors,
            "arms": ctx.recs,
        });
        if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        std::fs::write(out, serde_json::to_string_pretty(&doc).map_err(es)?)
            .map_err(|e| format!("write {}: {e}", out.display()))?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hcwmma_screen: {e}");
        std::process::exit(1);
    }
}
