// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `mhc`, `norm` and `head` modules. Owner: slice H.
//!
//! Every case drives the production path: `hipfire_arch_raon::mhc` (site
//! binding, any stream count/width) -> `hipfire_dispatch::pipeline::stream_mix`
//! -> `rdna_compute::stream_mix` kernels on the GPU, then compares each output
//! tensor against the reference fixture (PLAN 6.2 bands). Fixtures are
//! `<fixtures>/<module>/<case>.safetensors` + `manifest.json`, F32 tensors,
//! BF16-boundary values stored as BF16-rounded F32.
//!
//! Tensors (scalars are shape `[1]`), `dims` = `{N, n, C}`:
//!
//! * module `mhc` (one site, full chain):
//!   in:  `x[N,n,C]`, `phi[zdim,n*C]` (rows pre|post|z, `zdim = 2n+(n-1)^2`),
//!        `alpha_pre`, `alpha_post`, `alpha_z`, `eta`, `b_pre[n]`, `b_post[n]`,
//!        `b_z[(n-1)^2]`, `gamma[C]`, `eps_norm`, `selector_eps`, `post_factor`,
//!        `branch_out[N,C]`
//!   out: `z[N,zdim]` (scaled selector), `pre[N,n]`, `post[N,n]`, `res[N,n,n]`,
//!        `agg[N,C]` (aggregate before the norm), `branch_in[N,C]`,
//!        `x_out[N,n,C]`
//! * module `norm`: `x[N,C]`, `gamma[C]`, `eps` -> `y[N,C]`
//! * module `head`: `stream[N,n,C]`, `gamma[C]`, `eps` -> `mean[N,C]`, `h[N,C]`;
//!   cases named `bcast_*`: `emb[N,C]` -> `bcast[N,n,C]`
//!
//! A case may name an extra `weights` safetensors (path relative to the module
//! dir or the fixtures root) whose tensors are merged in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hipfire_arch_raon::mhc::{
    broadcast_embedding, final_mean_norm, RaonMhcHost, RaonMhcShape, RaonMhcSite, RaonMhcWork,
};
use hipfire_dispatch::pipeline::stream_mix::{
    execute_stream_mix, StreamMixOp, StreamMixStage, StreamNorm, StreamRounding,
};
use rdna_compute::stream_mix::reference::round_bf16;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

type Opts = BTreeMap<String, String>;

// ---------------------------------------------------------------- fixtures

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

type Tensors = BTreeMap<String, Tensor>;

fn read_safetensors(path: &Path) -> Result<Tensors, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err(format!("{}: truncated safetensors", path.display()));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let body = 8usize
        .checked_add(hlen)
        .filter(|&b| b <= bytes.len())
        .ok_or_else(|| format!("{}: header overruns file", path.display()))?;
    let header: Value = serde_json::from_slice(&bytes[8..body])
        .map_err(|e| format!("{}: header json: {e}", path.display()))?;
    let mut out = Tensors::new();
    for (name, meta) in header
        .as_object()
        .ok_or("safetensors header is not an object")?
    {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().unwrap_or("");
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or_else(|| format!("{name}: no shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = meta["data_offsets"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        let (s, e) = (
            off[0].as_u64().unwrap_or(0) as usize,
            off[1].as_u64().unwrap_or(0) as usize,
        );
        let raw = bytes
            .get(body + s..body + e)
            .ok_or_else(|| format!("{name}: data range outside file"))?;
        let data: Vec<f32> = match dtype {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
            other => return Err(format!("{name}: unsupported fixture dtype {other}")),
        };
        if data.len() != shape.iter().product::<usize>() {
            return Err(format!("{name}: {} values for shape {shape:?}", data.len()));
        }
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}

struct Case {
    name: String,
    file: String,
    weights: Option<String>,
    mode: StreamRounding,
    mode_name: String,
    dims: BTreeMap<String, usize>,
}

fn load_manifest(dir: &Path, opts: &Opts) -> Result<(Vec<Case>, usize), String> {
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let cases = m["cases"]
        .as_array()
        .ok_or_else(|| format!("{}: no `cases` array", path.display()))?;
    let declared = cases.len();
    let want = opts.get("case");
    let mut out = Vec::new();
    for c in cases {
        let name = c["name"]
            .as_str()
            .ok_or("manifest case without name")?
            .to_string();
        if want.is_some_and(|w| w != &name) {
            continue;
        }
        let mode_name = c["mode"].as_str().unwrap_or("bf16").to_string();
        let mode = match mode_name.as_str() {
            "bf16" => StreamRounding::Bf16,
            "f32" => StreamRounding::F32,
            other => return Err(format!("case {name}: unknown mode `{other}`")),
        };
        let mut dims = BTreeMap::new();
        if let Some(d) = c["dims"].as_object() {
            for (k, v) in d {
                if let Some(u) = v.as_u64() {
                    dims.insert(k.clone(), u as usize);
                }
            }
        }
        out.push(Case {
            file: c["file"]
                .as_str()
                .ok_or_else(|| format!("case {name}: no file"))?
                .to_string(),
            weights: c["weights"].as_str().map(str::to_string),
            name,
            mode,
            mode_name,
            dims,
        });
    }
    let declared = if want.is_some() { out.len() } else { declared };
    Ok((out, declared))
}

fn load_case(dir: &Path, root: &Path, case: &Case) -> Result<Tensors, String> {
    let mut t = read_safetensors(&dir.join(&case.file))?;
    if let Some(w) = &case.weights {
        let p: PathBuf = if dir.join(w).exists() {
            dir.join(w)
        } else {
            root.join(w)
        };
        t.extend(read_safetensors(&p)?);
    }
    Ok(t)
}

fn get<'a>(t: &'a Tensors, name: &str, len: usize) -> Result<&'a [f32], String> {
    let x = t
        .get(name)
        .ok_or_else(|| format!("fixture lacks tensor `{name}`"))?;
    if x.data.len() != len {
        return Err(format!(
            "tensor `{name}` has {} values (shape {:?}), expected {len}",
            x.data.len(),
            x.shape
        ));
    }
    Ok(&x.data)
}

fn scalar(t: &Tensors, name: &str) -> Result<f32, String> {
    Ok(get(t, name, 1)?[0])
}

fn dim(case: &Case, key: &str) -> Result<usize, String> {
    case.dims
        .get(key)
        .copied()
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("case {}: manifest dims lacks `{key}`", case.name))
}

// ---------------------------------------------------------------- compare

struct Cmp {
    pass: bool,
    json: Value,
}

/// PLAN 6.2 bands: F32 path rel-L2 <= 1e-4 and max_abs <= 2e-5 + 2e-4*max|ref|;
/// BF16-boundary path rel-L2 <= 1e-2 and cosine >= 0.9999.
fn compare(got: &[f32], want: &[f32], mode: StreamRounding) -> Cmp {
    let n = want.len();
    if got.len() != n {
        return Cmp {
            pass: false,
            json: json!({"pass": false, "error": format!("length {} != {}", got.len(), n)}),
        };
    }
    let (mut d2, mut r2, mut g2, mut dot) = (0f64, 0f64, 0f64, 0f64);
    let (mut max_abs, mut max_ref, mut worst) = (0f32, 0f32, 0usize);
    let mut nonfinite = None;
    for i in 0..n {
        if !got[i].is_finite() && nonfinite.is_none() {
            nonfinite = Some(i);
        }
        let d = (got[i] - want[i]).abs();
        if d > max_abs || d.is_nan() {
            max_abs = if d.is_nan() { f32::INFINITY } else { d };
            worst = i;
        }
        max_ref = max_ref.max(want[i].abs());
        let (g, r) = (got[i] as f64, want[i] as f64);
        d2 += (g - r) * (g - r);
        r2 += r * r;
        g2 += g * g;
        dot += g * r;
    }
    let rel_l2 = if r2 == 0.0 {
        d2.sqrt()
    } else {
        (d2 / r2).sqrt()
    };
    let cosine = if r2 == 0.0 || g2 == 0.0 {
        f64::from((d2 == 0.0) as u8)
    } else {
        dot / (r2.sqrt() * g2.sqrt())
    };
    let (band, pass, atol) = match mode {
        StreamRounding::F32 => {
            let atol = 2e-5 + 2e-4 * max_ref;
            (
                format!("rel_l2<=1e-4, max_abs<={atol:.3e}"),
                rel_l2 <= 1e-4 && max_abs <= atol,
                atol,
            )
        }
        StreamRounding::Bf16 => (
            "rel_l2<=1e-2, cosine>=0.9999".to_string(),
            rel_l2 <= 1e-2 && cosine >= 0.9999,
            2e-2 * max_ref,
        ),
    };
    let pass = pass && nonfinite.is_none();
    let first = (0..n).find(|&i| !((got[i] - want[i]).abs() <= atol));
    let mismatch = |i: usize| json!({"index": i, "got": got[i], "want": want[i]});
    let mut j = json!({
        "pass": pass, "rel_l2": rel_l2, "max_abs": max_abs, "cosine": cosine,
        "max_ref": max_ref, "band": band, "n": n,
        "worst": mismatch(worst),
    });
    if let Some(i) = first {
        j["first_mismatch"] = mismatch(i);
    }
    if let Some(i) = nonfinite {
        j["nonfinite_at"] = json!(i);
    }
    Cmp { pass, json: j }
}

/// Accumulates per-tensor verdicts of one case.
struct CaseReport {
    pass: bool,
    tensors: serde_json::Map<String, Value>,
    first_fail: Option<String>,
}

impl CaseReport {
    fn new() -> Self {
        Self {
            pass: true,
            tensors: Default::default(),
            first_fail: None,
        }
    }
    fn check(&mut self, name: &str, got: &[f32], want: &[f32], mode: StreamRounding) {
        let c = compare(got, want, mode);
        if !c.pass {
            self.pass = false;
            self.first_fail.get_or_insert_with(|| {
                format!(
                    "{name}: {}",
                    c.json
                        .get("first_mismatch")
                        .or(c.json.get("worst"))
                        .unwrap_or(&Value::Null)
                )
            });
        }
        self.tensors.insert(name.to_string(), c.json);
    }
    fn invariant(&mut self, name: &str, ok: bool, detail: Value) {
        if !ok {
            self.pass = false;
            self.first_fail
                .get_or_insert_with(|| format!("{name}: {detail}"));
        }
        self.tensors
            .insert(name.to_string(), json!({"pass": ok, "detail": detail}));
    }
    fn fail(name: &str, mode: &str, dims: &BTreeMap<String, usize>, err: String) -> Value {
        json!({"name": name, "pass": false, "mode": mode, "dims": dims, "detail": err})
    }
    fn into_json(self, case: &Case) -> Value {
        let mut v = json!({
            "name": case.name, "pass": self.pass, "mode": case.mode_name,
            "dims": case.dims, "tensors": self.tensors,
        });
        if let Some(f) = self.first_fail {
            v["first_mismatch"] = json!(f);
        }
        v
    }
}

// ---------------------------------------------------------------- GPU helpers

#[derive(Default)]
struct Pool(Vec<GpuTensor>);

impl Pool {
    fn up(&mut self, gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
        let t = gpu
            .upload_f32(data, shape)
            .map_err(|e| format!("upload: {e:?}"))?;
        Ok(self.keep(t))
    }
    /// BF16-bits upload (values must already be BF16-representable).
    fn up_bf16(&mut self, gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor, String> {
        let t = gpu
            .alloc_tensor(&[data.len()], DType::BF16)
            .map_err(|e| format!("alloc: {e:?}"))?;
        let bits: Vec<u8> = data
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        gpu.memcpy_htod_auto(&t.buf, &bits)
            .map_err(|e| format!("upload bf16: {e:?}"))?;
        Ok(self.keep(t))
    }
    fn out(&mut self, gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
        let t = gpu
            .alloc_tensor(shape, DType::F32)
            .map_err(|e| format!("alloc: {e:?}"))?;
        Ok(self.keep(t))
    }
    /// Keeps a non-owning duplicate handle so the original can be returned by
    /// value to the caller while the pool still frees the device buffer once.
    fn keep(&mut self, t: GpuTensor) -> GpuTensor {
        let view = t.sub_offset(0, t.buf.size() / t.dtype.size());
        self.0.push(t);
        view
    }
    fn free(self, gpu: &mut Gpu) {
        for t in self.0 {
            let _ = gpu.free_tensor(t);
        }
    }
}

fn download(gpu: &mut Gpu, t: &GpuTensor, len: usize) -> Result<Vec<f32>, String> {
    let mut v = gpu
        .download_f32(t)
        .map_err(|e| format!("download: {e:?}"))?;
    v.truncate(len);
    Ok(v)
}

fn finish(module: &str, results: Vec<Value>, declared: usize) -> Value {
    let pass = !results.is_empty()
        && results.len() == declared
        && results.iter().all(|c| c["pass"].as_bool() == Some(true));
    json!({"module": module, "pass": pass, "declared_cases": declared, "cases": results})
}

fn run_cases(
    module: &str,
    fixtures: &Path,
    opts: &Opts,
    one: fn(&mut Gpu, &Case, &Tensors) -> Result<CaseReport, String>,
) -> Result<Value, String> {
    let dir = fixtures.join(module);
    let (cases, declared) = load_manifest(&dir, opts)?;
    if cases.is_empty() {
        return Err(format!("{}: no cases", dir.display()));
    }
    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    let mut results = Vec::new();
    for case in &cases {
        let res = load_case(&dir, fixtures, case).and_then(|t| one(&mut gpu, case, &t));
        results.push(match res {
            Ok(rep) => rep.into_json(case),
            Err(e) => CaseReport::fail(&case.name, &case.mode_name, &case.dims, e),
        });
    }
    Ok(finish(module, results, declared))
}

// ---------------------------------------------------------------- module mhc

pub fn run(fixtures: &Path, opts: &Opts) -> Result<Value, String> {
    run_cases("mhc", fixtures, opts, site_case)
}

fn site_case(gpu: &mut Gpu, case: &Case, t: &Tensors) -> Result<CaseReport, String> {
    let (rows, n, c) = (dim(case, "N")?, dim(case, "n")?, dim(case, "C")?);
    let nm2 = (n - 1) * (n - 1);
    let zdim = 2 * n + nm2;
    let cols = n * c;
    let shape = RaonMhcShape {
        streams: n,
        width: c,
        post_factor: scalar(t, "post_factor")?,
        selector_eps: scalar(t, "selector_eps")?,
        rounding: case.mode,
    };
    let x = get(t, "x", rows * cols)?;
    let phi = get(t, "phi", zdim * cols)?;
    let gamma = get(t, "gamma", c)?;
    let eps_norm = scalar(t, "eps_norm")?;
    let branch_out = get(t, "branch_out", rows * c)?;
    let host = RaonMhcHost {
        phi_pre: &phi[..n * cols],
        phi_post: &phi[n * cols..2 * n * cols],
        phi_z: &phi[2 * n * cols..],
        alpha_pre: scalar(t, "alpha_pre")?,
        alpha_post: scalar(t, "alpha_post")?,
        alpha_z: scalar(t, "alpha_z")?,
        b_pre: get(t, "b_pre", n)?,
        b_post: get(t, "b_post", n)?,
        b_z: get(t, "b_z", nm2)?,
        eta: scalar(t, "eta")?,
    };
    // Expected outputs fail the case (not the run) when absent.
    let want = |name: &str, len: usize| get(t, name, len);
    let (w_z, w_pre, w_post, w_res) = (
        want("z", rows * zdim)?,
        want("pre", rows * n)?,
        want("post", rows * n)?,
        want("res", rows * n * n)?,
    );
    let (w_agg, w_bin, w_xout) = (
        want("agg", rows * c)?,
        want("branch_in", rows * c)?,
        want("x_out", rows * cols)?,
    );

    let mut pool = Pool::default();
    let site = RaonMhcSite::upload(gpu, &shape, &host)?;
    let work = RaonMhcWork::new(gpu, &shape, rows)?;
    let result = (|| -> Result<CaseReport, String> {
        let streams = pool.up(gpu, x, &[rows, n, c])?;
        let gamma_t = pool.up(gpu, gamma, &[c])?;
        let branch_t = pool.up(gpu, branch_out, &[rows, c])?;
        let agg_t = pool.out(gpu, &[rows, c])?;
        let bin_t = pool.out(gpu, &[rows, c])?;
        let xout_t = pool.out(gpu, &[rows, n, c])?;

        site.route(gpu, rows, &streams, &work)?;
        // aggregate before the norm, through the generic op
        execute_stream_mix(
            gpu,
            &StreamMixOp {
                rows,
                streams: n,
                width: c,
                rounding: case.mode,
                stage: StreamMixStage::Aggregate {
                    streams: &streams,
                    controls: work.controls(),
                    norm: None,
                    out: &agg_t,
                },
            },
        )
        .map_err(|e| e.to_string())?;
        site.aggregate(
            gpu,
            rows,
            &streams,
            &work,
            StreamNorm {
                gamma: &gamma_t,
                eps: eps_norm,
            },
            &bin_t,
        )?;
        site.postmix(gpu, rows, &streams, &branch_t, &work, &xout_t)?;

        let got_pre = download(gpu, work.controls().pre, rows * n)?;
        let got_post = download(gpu, work.controls().post, rows * n)?;
        let got_res = download(gpu, work.controls().res, rows * n * n)?;
        let sel = download(gpu, work.selector_out(), rows * zdim)?;
        // z = round(raw selector) * inv-rms: checks the production selector GEMM.
        let mut got_z = vec![0f32; rows * zdim];
        for r in 0..rows {
            let ss: f32 = x[r * cols..(r + 1) * cols].iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / cols as f32 + shape.selector_eps).sqrt();
            for k in 0..zdim {
                let raw = sel[r * zdim + k];
                got_z[r * zdim + k] = match case.mode {
                    StreamRounding::Bf16 => round_bf16(raw),
                    StreamRounding::F32 => raw,
                } * inv;
            }
        }
        let mut rep = CaseReport::new();
        rep.check("z", &got_z, w_z, case.mode);
        rep.check("pre", &got_pre, w_pre, case.mode);
        rep.check("post", &got_post, w_post, case.mode);
        rep.check("res", &got_res, w_res, case.mode);
        rep.check("agg", &download(gpu, &agg_t, rows * c)?, w_agg, case.mode);
        rep.check(
            "branch_in",
            &download(gpu, &bin_t, rows * c)?,
            w_bin,
            case.mode,
        );
        rep.check(
            "x_out",
            &download(gpu, &xout_t, rows * cols)?,
            w_xout,
            case.mode,
        );
        // The chart is doubly stochastic and nonnegative by construction; these
        // hold for the engine output independently of the reference.
        let (mut dev, mut min) = (0f32, f32::INFINITY);
        for r in 0..rows {
            let m = &got_res[r * n * n..(r + 1) * n * n];
            for i in 0..n {
                let rs: f32 = (0..n).map(|j| m[i * n + j]).sum();
                let cs: f32 = (0..n).map(|j| m[j * n + i]).sum();
                dev = dev.max((rs - 1.0).abs()).max((cs - 1.0).abs());
            }
            min = m.iter().fold(min, |a, &v| a.min(v));
        }
        rep.invariant(
            "res_doubly_stochastic",
            dev <= 1e-4 && min >= -1e-6,
            json!({"max_row_col_sum_dev": dev, "min_entry": min}),
        );
        Ok(rep)
    })();
    pool.free(gpu);
    let _ = work.free(gpu);
    let _ = site.free(gpu);
    result
}

// ---------------------------------------------------------------- module norm

/// Plain-gamma RMSNorm (F32 and, when BF16-representable, BF16 gamma storage)
/// through the stream-mean epilogue with one stream.
pub fn run_norm(fixtures: &Path, opts: &Opts) -> Result<Value, String> {
    run_cases("norm", fixtures, opts, norm_case)
}

fn bf16_exact(v: &[f32]) -> bool {
    v.iter().all(|&g| round_bf16(g) == g)
}

fn norm_case(gpu: &mut Gpu, case: &Case, t: &Tensors) -> Result<CaseReport, String> {
    let (rows, c) = (dim(case, "N")?, dim(case, "C")?);
    let x = get(t, "x", rows * c)?;
    let gamma = get(t, "gamma", c)?;
    let eps = scalar(t, "eps")?;
    let want = get(t, "y", rows * c)?;
    let mut pool = Pool::default();
    let result = (|| -> Result<CaseReport, String> {
        let xt = pool.up(gpu, x, &[rows, 1, c])?;
        let mut rep = CaseReport::new();
        let mut gammas = vec![("y", pool.up(gpu, gamma, &[c])?)];
        if bf16_exact(gamma) {
            gammas.push(("y_bf16_gamma", pool.up_bf16(gpu, gamma)?));
        }
        for (label, g) in gammas {
            let out = pool.out(gpu, &[rows, c])?;
            execute_stream_mix(
                gpu,
                &StreamMixOp {
                    rows,
                    streams: 1,
                    width: c,
                    rounding: case.mode,
                    stage: StreamMixStage::Mean {
                        streams: &xt,
                        norm: Some(StreamNorm { gamma: &g, eps }),
                        out: &out,
                    },
                },
            )
            .map_err(|e| e.to_string())?;
            rep.check(label, &download(gpu, &out, rows * c)?, want, case.mode);
        }
        Ok(rep)
    })();
    pool.free(gpu);
    result
}

// ---------------------------------------------------------------- module head

/// Embedding broadcast, final stream mean and final norm.
pub fn run_head(fixtures: &Path, opts: &Opts) -> Result<Value, String> {
    run_cases("head", fixtures, opts, head_case)
}

fn head_case(gpu: &mut Gpu, case: &Case, t: &Tensors) -> Result<CaseReport, String> {
    let (rows, n, c) = (dim(case, "N")?, dim(case, "n")?, dim(case, "C")?);
    let shape = RaonMhcShape {
        streams: n,
        width: c,
        post_factor: 2.0,
        selector_eps: 1e-5,
        rounding: case.mode,
    };
    let mut pool = Pool::default();
    let result = (|| -> Result<CaseReport, String> {
        let mut rep = CaseReport::new();
        if case.name.starts_with("bcast") {
            let emb = get(t, "emb", rows * c)?;
            let want = get(t, "bcast", rows * n * c)?;
            let et = pool.up(gpu, emb, &[rows, c])?;
            let out = pool.out(gpu, &[rows, n, c])?;
            broadcast_embedding(gpu, &shape, rows, &et, c, &out)?;
            rep.check(
                "bcast",
                &download(gpu, &out, rows * n * c)?,
                want,
                case.mode,
            );
            return Ok(rep);
        }
        let stream = get(t, "stream", rows * n * c)?;
        let gamma = get(t, "gamma", c)?;
        let eps = scalar(t, "eps")?;
        let (w_mean, w_h) = (get(t, "mean", rows * c)?, get(t, "h", rows * c)?);
        let st = pool.up(gpu, stream, &[rows, n, c])?;
        let gt = pool.up(gpu, gamma, &[c])?;
        let mean_t = pool.out(gpu, &[rows, c])?;
        let h_t = pool.out(gpu, &[rows, c])?;
        execute_stream_mix(
            gpu,
            &StreamMixOp {
                rows,
                streams: n,
                width: c,
                rounding: case.mode,
                stage: StreamMixStage::Mean {
                    streams: &st,
                    norm: None,
                    out: &mean_t,
                },
            },
        )
        .map_err(|e| e.to_string())?;
        final_mean_norm(gpu, &shape, rows, &st, StreamNorm { gamma: &gt, eps }, &h_t)?;
        rep.check(
            "mean",
            &download(gpu, &mean_t, rows * c)?,
            w_mean,
            case.mode,
        );
        rep.check("h", &download(gpu, &h_t, rows * c)?, w_h, case.mode);
        Ok(rep)
    })();
    pool.free(gpu);
    result
}
