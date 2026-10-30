// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `rda` subcommand. Owner: slice R.
//!
//! Runs every case in `<fixtures>/rda/manifest.json` through the PRODUCTION
//! launchers (`rda_forward_rows` -> conv, `raon_rda_prepare`, `GatedDeltaOp`,
//! gated RMSNorm) and compares to the reference tensors per PLAN §6.2:
//!
//! * stage `module` (end-to-end from raw projections): BF16-boundary band
//!   (rel-L2 <= 1e-2, cosine >= .9999) for rounded tensors; the FP32 state is
//!   held to rel-L2 <= 1e-2 here because upstream BF16 rounding flips enter it.
//! * stage `recurrence` (reference q/k/vprime/log_alpha/beta in, so rounding
//!   flips cannot enter): final state in the same-format F32 band
//!   (rel-L2 <= 1e-4 for N<=64, <= 5e-4 beyond; max_abs <= 2e-5+2e-4*max|ref|),
//!   y in the BF16-boundary band.
//! * stage `split`: the case re-run as two sequential calls must be
//!   byte-identical (state, histories, gated) to the single call.
//! * stage `scalar_equiv`: PerHead and PerKeyChannel(broadcast) vector kernels vs
//!   the existing scalar GDN kernel and the F32 reference (`gdn_*` tensors).
//!
//! Fixture tensor names (F32 safetensors; BF16-boundary values BF16-rounded):
//! inputs `q_raw,k_raw,v_raw,z,f [N,P]`, `b,delta [N,H]`, `conv_w_{q,k,v} [P,4]`,
//! `hist_{q,k,v} [3,P]` (oldest->newest), `A_log [H]`, `dt_bias [H,K]`,
//! `norm_gamma [K]`, `S0 [H,V,K]`; outputs `q,k,vprime,log_alpha [N,P]`,
//! `beta [N,H]`, `y,gated [N,P]`, `S_out`, `hist_out_{q,k,v}`. Scalar
//! equivalence: `gdn_{q,k,v} [N,H,K]`, `gdn_log_decay [N,H]`, `gdn_beta [N,H]`,
//! `gdn_S0`, `gdn_y`, `gdn_S_out`.
//! Options: `case=<substring>` filters cases.

use hipfire_arch_raon::rda::{
    rda_forward_rows, RaonRdaLayer, RdaGeometry, RdaLayerState, RdaProjected, RdaScratch,
};
use hipfire_dispatch::ops::delta_net::{
    execute_gated_delta, DecayAxis, GatedDeltaOp, GatedDeltaStrides,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

// ── minimal safetensors reader (F32 / BF16 / F16 → f32) ──────────────────

struct Tensors {
    map: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

fn load_safetensors(path: &Path) -> Result<Tensors, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 8 {
        return Err(format!("{}: truncated", path.display()));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(
        bytes
            .get(8..8 + hlen)
            .ok_or_else(|| format!("{}: bad header length", path.display()))?,
    )
    .map_err(|e| format!("{}: header json: {e}", path.display()))?;
    let data = &bytes[8 + hlen..];
    let mut map = BTreeMap::new();
    for (name, meta) in header.as_object().ok_or("header not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().ok_or("missing dtype")?;
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = meta["data_offsets"].as_array().ok_or("missing offsets")?;
        let (a, b) = (
            off[0].as_u64().unwrap_or(0) as usize,
            off[1].as_u64().unwrap_or(0) as usize,
        );
        let raw = data
            .get(a..b)
            .ok_or_else(|| format!("{name}: offsets out of range"))?;
        let vals: Vec<f32> = match dtype {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
            other => return Err(format!("{name}: unsupported dtype {other}")),
        };
        if vals.len() != shape.iter().product::<usize>() {
            return Err(format!("{name}: element count != shape"));
        }
        map.insert(name.clone(), (shape, vals));
    }
    Ok(Tensors { map })
}

impl Tensors {
    fn get(&self, name: &str) -> Option<&[f32]> {
        self.map.get(name).map(|(_, v)| v.as_slice())
    }
    fn need(&self, name: &str) -> Result<&[f32], String> {
        self.get(name)
            .ok_or_else(|| format!("fixture missing tensor `{name}`"))
    }
}

// ── comparison ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Band {
    /// Same-format F32 module band; `rel` is the rel-L2 limit.
    F32 { rel: f64 },
    /// BF16-boundary band: rel-L2 <= 1e-2, cosine >= .9999.
    Bf16,
    /// Relative L2 only (FP32 state accumulated from BF16-boundary inputs).
    Rel { rel: f64 },
    /// Byte-identical.
    Exact,
}

fn compare(name: &str, got: &[f32], want: &[f32], band: Band) -> Value {
    if got.len() != want.len() {
        return json!({"tensor": name, "pass": false,
            "error": format!("length mismatch got {} want {}", got.len(), want.len())});
    }
    let (mut d2, mut r2, mut g2, mut dot) = (0f64, 0f64, 0f64, 0f64);
    let (mut max_abs, mut max_ref) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        let d = g - w;
        d2 += d * d;
        r2 += w * w;
        g2 += g * g;
        dot += g * w;
        max_abs = max_abs.max(d.abs());
        max_ref = max_ref.max(w.abs());
    }
    let rel = if r2 > 0.0 {
        (d2 / r2).sqrt()
    } else {
        d2.sqrt()
    };
    let cos = if r2 > 0.0 && g2 > 0.0 {
        dot / (r2.sqrt() * g2.sqrt())
    } else if d2 == 0.0 {
        1.0
    } else {
        0.0
    };
    let abs_lim = 2e-5 + 2e-4 * max_ref;
    let (pass, band_json, first_bad) = match band {
        Band::F32 { rel: lim } => {
            let first = got
                .iter()
                .zip(want)
                .position(|(g, w)| ((*g as f64) - (*w as f64)).abs() > abs_lim || !g.is_finite());
            (
                rel <= lim && max_abs <= abs_lim && got.iter().all(|v| v.is_finite()),
                json!({"kind": "f32", "rel_l2_max": lim, "max_abs_max": abs_lim}),
                first,
            )
        }
        Band::Bf16 => {
            let first = got.iter().zip(want).position(|(g, w)| {
                ((*g as f64) - (*w as f64)).abs() > 0.02 * (w.abs() as f64) + 1e-3 * max_ref
                    || !g.is_finite()
            });
            (
                rel <= 1e-2 && cos >= 0.9999 && got.iter().all(|v| v.is_finite()),
                json!({"kind": "bf16", "rel_l2_max": 1e-2, "cos_min": 0.9999}),
                first,
            )
        }
        Band::Rel { rel: lim } => {
            let first = got.iter().zip(want).position(|(g, w)| {
                ((*g as f64) - (*w as f64)).abs() > lim * max_ref || !g.is_finite()
            });
            (
                rel <= lim && got.iter().all(|v| v.is_finite()),
                json!({"kind": "rel_l2", "rel_l2_max": lim}),
                first,
            )
        }
        Band::Exact => {
            let first = got
                .iter()
                .zip(want)
                .position(|(g, w)| g.to_bits() != w.to_bits());
            (first.is_none(), json!({"kind": "exact"}), first)
        }
    };
    let mut v = json!({"tensor": name, "pass": pass, "rel_l2": rel, "max_abs": max_abs,
        "max_ref": max_ref, "cosine": cos, "band": band_json});
    if !pass {
        let idx = first_bad.unwrap_or(0);
        v["first_mismatch"] = json!({"index": idx, "got": got[idx], "want": want[idx]});
    }
    v
}

// ── GPU helpers ───────────────────────────────────────────────────────────

struct Pool(Vec<GpuTensor>);

impl Pool {
    fn up(&mut self, gpu: &mut Gpu, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
        let t = gpu
            .upload_f32(data, shape)
            .map_err(|e| format!("upload: {e:?}"))?;
        let view = t.sub_offset(0, data.len());
        self.0.push(t);
        Ok(view)
    }
    fn zeros(&mut self, gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
        let t = gpu
            .zeros(shape, DType::F32)
            .map_err(|e| format!("zeros: {e:?}"))?;
        let view = t.sub_offset(0, shape.iter().product());
        self.0.push(t);
        Ok(view)
    }
}

fn dl(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<f32>, String> {
    gpu.download_f32(t).map_err(|e| format!("download: {e:?}"))
}

fn dims(case: &Value, key: &str) -> Result<usize, String> {
    case["dims"][key]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("case dims missing `{key}`"))
}

// ── per-case stages ───────────────────────────────────────────────────────

struct ModuleOut {
    state: Vec<f32>,
    hist: [Vec<f32>; 3],
    gated: Vec<f32>,
}

/// Run the production module over `[row0, row0+n)` of the fixture inputs.
#[allow(clippy::too_many_arguments)]
fn run_module_rows(
    gpu: &mut Gpu,
    pool: &mut Pool,
    geom: &RdaGeometry,
    t: &Tensors,
    s0: &[f32],
    hist0: [&[f32]; 3],
    splits: &[usize],
    scratch_rows: usize,
    keep_scratch: &mut Option<RdaScratch>,
) -> Result<ModuleOut, String> {
    let (p, h) = (geom.proj(), geom.n_heads);
    let n: usize = splits.iter().sum();
    let k = geom.conv_kernel;
    let q_raw = pool.up(gpu, t.need("q_raw")?, &[n, p])?;
    let k_raw = pool.up(gpu, t.need("k_raw")?, &[n, p])?;
    let v_raw = pool.up(gpu, t.need("v_raw")?, &[n, p])?;
    let zeros_z;
    let z_data: &[f32] = match t.get("z") {
        Some(z) => z,
        None => {
            zeros_z = vec![0f32; n * p];
            &zeros_z
        }
    };
    let z = pool.up(gpu, z_data, &[n, p])?;
    let b = pool.up(gpu, t.need("b")?, &[n, h])?;
    let delta = pool.up(gpu, t.need("delta")?, &[n, h])?;
    let f = pool.up(gpu, t.need("f")?, &[n, p])?;
    let cw = [
        pool.up(gpu, t.need("conv_w_q")?, &[p, k])?,
        pool.up(gpu, t.need("conv_w_k")?, &[p, k])?,
        pool.up(gpu, t.need("conv_w_v")?, &[p, k])?,
    ];
    let a_log = pool.up(gpu, t.need("A_log")?, &[h])?;
    let dt_bias = pool.up(gpu, t.need("dt_bias")?, &[h * geom.head_dim])?;
    let gamma = pool.up(gpu, t.need("norm_gamma")?, &[geom.head_dim])?;
    let state = pool.up(gpu, s0, &[h, geom.head_dim, geom.head_dim])?;
    let hq = pool.up(gpu, hist0[0], &[k - 1, p])?;
    let hk = pool.up(gpu, hist0[1], &[k - 1, p])?;
    let hv = pool.up(gpu, hist0[2], &[k - 1, p])?;
    let gated = pool.zeros(gpu, &[n, p])?;
    if keep_scratch
        .as_ref()
        .map_or(true, |s| s.max_rows < scratch_rows)
    {
        *keep_scratch = Some(RdaScratch::new(gpu, geom, scratch_rows)?);
    }
    let scratch = keep_scratch.as_ref().unwrap();

    let layer = RaonRdaLayer {
        conv_w_q: &cw[0],
        conv_w_k: &cw[1],
        conv_w_v: &cw[2],
        a_log: &a_log,
        dt_bias: &dt_bias,
        o_norm_gamma: &gamma,
    };
    let st = RdaLayerState {
        state: &state,
        hist_q: &hq,
        hist_k: &hk,
        hist_v: &hv,
        slot: 0,
        state_slot_stride: geom.state_elems(),
    };
    let mut row = 0usize;
    for &len in splits {
        let rows = |t: &GpuTensor, cols: usize| t.sub_offset(row * cols, len * cols);
        let (qv, kv, vv, zv, bv, dv, fv, gv) = (
            rows(&q_raw, p),
            rows(&k_raw, p),
            rows(&v_raw, p),
            rows(&z, p),
            rows(&b, h),
            rows(&delta, h),
            rows(&f, p),
            rows(&gated, p),
        );
        let proj = RdaProjected {
            q_raw: &qv,
            k_raw: &kv,
            v_raw: &vv,
            z: &zv,
            b: &bv,
            delta: &dv,
            f: &fv,
        };
        rda_forward_rows(gpu, geom, &layer, &st, &proj, scratch, &gv, len)?;
        row += len;
    }
    Ok(ModuleOut {
        state: dl(gpu, &state)?,
        hist: [dl(gpu, &hq)?, dl(gpu, &hk)?, dl(gpu, &hv)?],
        gated: dl(gpu, &gated)?,
    })
}

fn module_case(
    gpu: &mut Gpu,
    case: &Value,
    t: &Tensors,
    out: &mut Vec<Value>,
) -> Result<(), String> {
    let n = dims(case, "N")?;
    let mut geom = RdaGeometry::raon();
    geom.n_heads = dims(case, "H")?;
    geom.head_dim = dims(case, "K")?;
    // Model dtype follows the case mode: f32 cases have no BF16 boundaries.
    let bf16 = case["mode"].as_str() != Some("f32");
    geom.round_bf16 = bf16;
    let bb = if bf16 {
        Band::Bf16
    } else {
        Band::F32 { rel: 1e-4 }
    };
    let p = geom.proj();
    let h = geom.n_heads;
    let s0 = t.need("S0")?;
    let hist0 = [t.need("hist_q")?, t.need("hist_k")?, t.need("hist_v")?];
    let state_rel = if n <= 64 { 1e-4 } else { 5e-4 };
    let mut keep = None;

    // Stage module: single production call; also read intermediates.
    let mut pool = Pool(Vec::new());
    let single = run_module_rows(gpu, &mut pool, &geom, t, s0, hist0, &[n], n, &mut keep)?;
    {
        let sc = keep.as_ref().unwrap();
        for (name, tensor, want_len) in [
            ("q", &sc.q, n * p),
            ("k", &sc.k, n * p),
            ("vprime", &sc.vprime, n * p),
            ("log_alpha", &sc.log_alpha, n * p),
            ("beta", &sc.beta, n * h),
            ("y", &sc.y, n * p),
        ] {
            if let Some(want) = t.get(name) {
                let got = dl(gpu, tensor)?;
                let band = if name == "log_alpha" {
                    Band::F32 { rel: 1e-4 }
                } else {
                    bb
                };
                out.push(json!({"stage": "module", "cmp":
                    compare(name, &got[..want_len], want, band)}));
            }
        }
    }
    if t.get("z").is_some() {
        if let Some(want) = t.get("gated") {
            out.push(json!({"stage": "module", "cmp":
                compare("gated", &single.gated, want, bb)}));
        }
    }
    if let Some(want) = t.get("S_out") {
        out.push(json!({"stage": "module", "cmp":
            compare("S_out", &single.state, want,
                if bf16 { Band::Rel { rel: 1e-2 } } else { Band::F32 { rel: state_rel } })}));
    }
    for (i, nm) in ["hist_out_q", "hist_out_k", "hist_out_v"]
        .iter()
        .enumerate()
    {
        let want = t.need(nm)?;
        // History is raw pre-conv input values: exact copies.
        out.push(json!({"stage": "module", "cmp":
            compare(nm, &single.hist[i], want, Band::Exact)}));
    }
    for v in pool.0.drain(..) {
        let _ = gpu.free_tensor(v);
    }

    // Stage recurrence: reference-prepared operands straight into GatedDeltaOp.
    if ["q", "k", "vprime", "log_alpha", "beta", "S_out"]
        .iter()
        .all(|k| t.get(k).is_some())
    {
        let mut pool = Pool(Vec::new());
        let q = pool.up(gpu, t.need("q")?, &[n, p])?;
        let k = pool.up(gpu, t.need("k")?, &[n, p])?;
        let v = pool.up(gpu, t.need("vprime")?, &[n, p])?;
        let la = pool.up(gpu, t.need("log_alpha")?, &[n, p])?;
        let be = pool.up(gpu, t.need("beta")?, &[n, h])?;
        let st = pool.up(gpu, s0, &[h, geom.head_dim, geom.head_dim])?;
        let y = pool.zeros(gpu, &[n, p])?;
        let r = execute_gated_delta(
            gpu,
            &GatedDeltaOp {
                query: &q,
                key: &k,
                value: &v,
                log_decay: &la,
                beta: &be,
                state: &st,
                output: &y,
                slot_idx: None,
                n_dev: None,
                n_tokens: n,
                n_heads: h,
                qk_group: 1,
                key_dim: geom.head_dim,
                value_dim: geom.head_dim,
                decay_axis: DecayAxis::PerKeyChannel,
                round_output_bf16: bf16,
                slot: 0,
                strides: GatedDeltaStrides::contiguous(
                    h,
                    1,
                    geom.head_dim,
                    geom.head_dim,
                    DecayAxis::PerKeyChannel,
                ),
            },
        )
        .map_err(|e| e.to_string());
        r?;
        let got_s = dl(gpu, &st)?;
        let got_y = dl(gpu, &y)?;
        out.push(json!({"stage": "recurrence", "cmp":
            compare("S_out", &got_s, t.need("S_out")?, Band::F32 { rel: state_rel })}));
        if let Some(want) = t.get("y") {
            out.push(json!({"stage": "recurrence", "cmp": compare("y", &got_y, want, bb)}));
        }
        for v in pool.0.drain(..) {
            let _ = gpu.free_tensor(v);
        }
    }

    // Stage split: two calls must be byte-identical to the single call.
    if n >= 2 && n <= 1024 {
        let first = n / 2;
        let mut pool = Pool(Vec::new());
        let split = run_module_rows(
            gpu,
            &mut pool,
            &geom,
            t,
            s0,
            hist0,
            &[first, n - first],
            n,
            &mut keep,
        )?;
        out.push(json!({"stage": "split", "cmp": compare("S_out", &split.state, &single.state, Band::Exact)}));
        for (i, nm) in ["hist_out_q", "hist_out_k", "hist_out_v"]
            .iter()
            .enumerate()
        {
            out.push(json!({"stage": "split", "cmp": compare(nm, &split.hist[i], &single.hist[i], Band::Exact)}));
        }
        out.push(json!({"stage": "split", "cmp": compare("gated", &split.gated, &single.gated, Band::Exact)}));
        for v in pool.0.drain(..) {
            let _ = gpu.free_tensor(v);
        }
    }
    if let Some(s) = keep.take() {
        for v in [
            s.qconv,
            s.kconv,
            s.vconv,
            s.q,
            s.k,
            s.vprime,
            s.log_alpha,
            s.beta,
            s.y,
        ] {
            let _ = gpu.free_tensor(v);
        }
    }
    Ok(())
}

fn scalar_equiv_case(
    gpu: &mut Gpu,
    case: &Value,
    t: &Tensors,
    out: &mut Vec<Value>,
) -> Result<(), String> {
    let n = dims(case, "N")?;
    let h = dims(case, "H")?;
    let kd = dims(case, "K")?;
    let hd = h * kd;
    let mut pool = Pool(Vec::new());
    let q = pool.up(gpu, t.need("gdn_q")?, &[n, h, kd])?;
    let k = pool.up(gpu, t.need("gdn_k")?, &[n, h, kd])?;
    let v = pool.up(gpu, t.need("gdn_v")?, &[n, h, kd])?;
    let ld_data = t.need("gdn_log_decay")?;
    let ld = pool.up(gpu, ld_data, &[n, h])?;
    let mut bcast = vec![0f32; n * hd];
    for (row, hh) in (0..n).flat_map(|r| (0..h).map(move |hh| (r, hh))) {
        for c in 0..kd {
            bcast[(row * h + hh) * kd + c] = ld_data[row * h + hh];
        }
    }
    let ld_vec = pool.up(gpu, &bcast, &[n, h, kd])?;
    let beta = pool.up(gpu, t.need("gdn_beta")?, &[n, h])?;
    let s0 = t.need("gdn_S0")?;

    // (1) PerHead vector kernel vs F32 reference.
    let mut results: Vec<(&str, Vec<f32>, Vec<f32>)> = Vec::new();
    for (label, axis, decay) in [
        ("per_head", DecayAxis::PerHead, &ld),
        ("per_key_broadcast", DecayAxis::PerKeyChannel, &ld_vec),
    ] {
        let st = pool.up(gpu, s0, &[h, kd, kd])?;
        let y = pool.zeros(gpu, &[n, h, kd])?;
        execute_gated_delta(
            gpu,
            &GatedDeltaOp {
                query: &q,
                key: &k,
                value: &v,
                log_decay: decay,
                beta: &beta,
                state: &st,
                output: &y,
                slot_idx: None,
                n_dev: None,
                n_tokens: n,
                n_heads: h,
                qk_group: 1,
                key_dim: kd,
                value_dim: kd,
                decay_axis: axis,
                round_output_bf16: false,
                slot: 0,
                strides: GatedDeltaStrides::contiguous(h, 1, kd, kd, axis),
            },
        )
        .map_err(|e| e.to_string())?;
        results.push((label, dl(gpu, &y)?, dl(gpu, &st)?));
    }
    // (2) existing scalar GDN kernel (production).
    let st = pool.up(gpu, s0, &[h, kd, kd])?;
    let y = pool.zeros(gpu, &[n, h, kd])?;
    gpu.gated_delta_net_f32(&q, &k, &v, &ld, &beta, &st, &y, n, h, kd)
        .map_err(|e| format!("scalar gdn: {e:?}"))?;
    let (scalar_y, scalar_s) = (dl(gpu, &y)?, dl(gpu, &st)?);

    let rel = if n <= 64 { 1e-4 } else { 5e-4 };
    for (label, gy, gs) in &results {
        out.push(json!({"stage": "scalar_equiv", "route": label, "cmp":
            compare("y_vs_reference", gy, t.need("gdn_y")?, Band::F32 { rel: 1e-4 })}));
        out.push(json!({"stage": "scalar_equiv", "route": label, "cmp":
            compare("S_out_vs_reference", gs, t.need("gdn_S_out")?, Band::F32 { rel })}));
        out.push(json!({"stage": "scalar_equiv", "route": label, "cmp":
            compare("y_vs_scalar_gdn", gy, &scalar_y, Band::F32 { rel: 1e-4 })}));
        out.push(json!({"stage": "scalar_equiv", "route": label, "cmp":
            compare("S_out_vs_scalar_gdn", gs, &scalar_s, Band::F32 { rel })}));
    }
    // PerHead and broadcast per-key must agree bit-for-bit (same arithmetic).
    out.push(
        json!({"stage": "scalar_equiv", "route": "per_head_vs_per_key", "cmp":
        compare("y", &results[0].1, &results[1].1, Band::Exact)}),
    );
    out.push(
        json!({"stage": "scalar_equiv", "route": "per_head_vs_per_key", "cmp":
        compare("S_out", &results[0].2, &results[1].2, Band::Exact)}),
    );
    for v in pool.0.drain(..) {
        let _ = gpu.free_tensor(v);
    }
    Ok(())
}

fn cmp_pass(v: &Value) -> bool {
    v["cmp"]["pass"].as_bool().unwrap_or(false)
}

pub fn run(fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let dir = fixtures.join("rda");
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json"))
            .map_err(|e| format!("{}: {e}", dir.join("manifest.json").display()))?,
    )
    .map_err(|e| format!("manifest.json: {e}"))?;
    let cases = manifest["cases"]
        .as_array()
        .ok_or("manifest has no `cases`")?;
    if cases.is_empty() {
        return Err("rda manifest has zero cases".into());
    }
    let filter = opts.get("case").cloned();
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let mut report = Vec::new();
    let mut all_pass = true;
    let mut ran = 0usize;
    for case in cases {
        let name = case["name"].as_str().ok_or("case without name")?;
        if filter
            .as_ref()
            .map_or(false, |f| !name.contains(f.as_str()))
        {
            continue;
        }
        ran += 1;
        let file = case["file"].as_str().ok_or("case without file")?;
        let mut checks = Vec::new();
        let result = (|| -> Result<(), String> {
            let t = load_safetensors(&dir.join(file))?;
            let mut did = false;
            if t.get("q_raw").is_some() {
                module_case(&mut gpu, case, &t, &mut checks)?;
                did = true;
            }
            if t.get("gdn_q").is_some() {
                scalar_equiv_case(&mut gpu, case, &t, &mut checks)?;
                did = true;
            }
            if !did {
                return Err("case has neither `q_raw` nor `gdn_q` inputs".into());
            }
            Ok(())
        })();
        let pass = result.is_ok() && !checks.is_empty() && checks.iter().all(cmp_pass);
        all_pass &= pass;
        let first_fail = checks.iter().find(|c| !cmp_pass(c)).cloned();
        let mut entry = json!({"name": name, "pass": pass, "checks": checks});
        if let Err(e) = result {
            entry["error"] = json!(e);
        }
        if let Some(f) = first_fail {
            entry["first_failure"] = f;
        }
        report.push(entry);
    }
    if ran == 0 {
        return Err("no rda cases matched".into());
    }
    Ok(json!({"module": "rda", "pass": all_pass, "cases": report}))
}
