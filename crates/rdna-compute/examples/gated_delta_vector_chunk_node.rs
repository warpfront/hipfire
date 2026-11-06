// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node gate + bench of the chunked RDA kernel (`Gpu::gated_delta_net_vector_chunked_f32`)
//! against the serial PM kernel (`Gpu::gated_delta_net_vector_serial_f32`), gfx1201.
//!
//! ```text
//! gated_delta_vector_chunk_node suite [--big] [--cap ROWS] [--only SUBSTR]
//! gated_delta_vector_chunk_node captured DIR [--cap ROWS]      (DIR/callNNN/{q,k,v,a,beta,state_in,out,state_out}.f32)
//! gated_delta_vector_chunk_node prof [--rows 4096]       (prep-kernel phase timestamps, occupancy)
//! gated_delta_vector_chunk_node bench [--rows 4096] [--caps 512,1024,2048,4096] [--reps 20]
//! ```
//!
//! Gate (METHODOLOGY-AUDIT sec. 7): finite outputs/states; output `|err| <= .005 + .01|ref|`, state
//! `|err| <= .0002 + .002|ref|` against the serial kernel; per-case max error, worst err/tol ratio,
//! violation count and relative L2 norm; deterministic bytes over three chunked runs. One JSON line
//! per case, then a summary; exit 0 only if every case passes.
//!
//! Build: cargo build --release -j 8 -p rdna-compute --features deltanet --example gated_delta_vector_chunk_node
//! Run (GPU, through fabric): ROCR_VISIBLE_DEVICES=<gfx1201> gated_delta_vector_chunk_node suite

use rdna_compute::gated_delta_vector::{GatedDeltaVectorLaunch, GATED_DELTA_VECTOR_HEAD_DIM};
use rdna_compute::gated_delta_vector_chunk::GatedDeltaVectorChunkScratch;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const H: usize = 40;
const D: usize = GATED_DELTA_VECTOR_HEAD_DIM;
const P: usize = H * D;
const OUT_ATOL: f64 = 0.005;
const OUT_RTOL: f64 = 0.01;
const ST_ATOL: f64 = 0.0002;
const ST_RTOL: f64 = 0.002;

// ---------------------------------------------------------------- host data ----------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (a, b) = (self.unif(), self.unif());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

fn bf16(x: f32) -> f32 {
    let u = x.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return x;
    }
    f32::from_bits((u.wrapping_add(0x7fff + ((u >> 16) & 1))) & 0xffff_0000)
}

#[derive(Clone, Copy, Debug)]
enum Decay {
    /// a = -5 sigmoid(u), u ~ per-channel bias + noise (the checkpoint's gate shape)
    Typical,
    None,
    Max,
    /// -20 everywhere (kernel clamps at -10)
    Extreme,
    /// per-channel mix of 0, -5, -20, -1e-3 and typical
    Mixed,
}

struct Inputs {
    t: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

/// `corr`: temporal key/query correlation rho (stresses the delta-rule coupling);
/// `beta_hi`: beta near 1; `state_scale`: initial state magnitude (0 = zero state).
fn gen_inputs(
    t: usize,
    seed: u64,
    decay: Decay,
    corr: f64,
    beta_hi: bool,
    state_scale: f64,
) -> Inputs {
    use rayon::prelude::*;
    let qs = (D as f64).powf(-0.5);
    // per head, sequential in t (correlation), parallel over heads
    let per_head: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0..H)
        .into_par_iter()
        .map(|h| {
            let mut r = Rng(seed ^ ((h as u64 + 1) * 0xA24BAED4963EE407));
            let bias: Vec<f64> = (0..D).map(|_| -1.0 + 2.0 * r.gauss()).collect();
            let mut q = vec![0f32; t * D];
            let mut k = vec![0f32; t * D];
            let mut v = vec![0f32; t * D];
            let mut a = vec![0f32; t * D];
            let mut be = vec![0f32; t];
            let mut kp = vec![0f64; D];
            let mut qp = vec![0f64; D];
            for i in 0..t {
                for prev in [&mut kp, &mut qp] {
                    let mut raw = [0f64; D];
                    let mut n2 = 0.0;
                    for c in 0..D {
                        let nz = r.gauss();
                        raw[c] = if i == 0 {
                            nz
                        } else {
                            corr * prev[c] + (1.0 - corr * corr).sqrt() * nz
                        };
                        n2 += raw[c] * raw[c];
                    }
                    let inv = 1.0 / n2.sqrt().max(1e-6);
                    for c in 0..D {
                        prev[c] = raw[c] * inv * (D as f64).sqrt();
                    }
                }
                for c in 0..D {
                    k[i * D + c] = bf16((kp[c] / (D as f64).sqrt()) as f32);
                    q[i * D + c] = bf16(bf16((qp[c] / (D as f64).sqrt()) as f32) * qs as f32);
                    v[i * D + c] = bf16(r.gauss() as f32);
                    let u = bias[c] + 0.5 * r.gauss();
                    let sig = 1.0 / (1.0 + (-u).exp());
                    a[i * D + c] = match decay {
                        Decay::Typical => (-5.0 * sig) as f32,
                        Decay::None => 0.0,
                        Decay::Max => -5.0,
                        Decay::Extreme => -20.0,
                        Decay::Mixed => match c % 5 {
                            0 => 0.0,
                            1 => -5.0,
                            2 => -20.0,
                            3 => -1e-3,
                            _ => (-5.0 * sig) as f32,
                        },
                    };
                }
                let b = if beta_hi {
                    0.999
                } else {
                    1.0 / (1.0 + (-r.gauss()).exp())
                };
                be[i] = bf16(b as f32);
            }
            (q, k, v, a, be)
        })
        .collect();
    let mut inp = Inputs {
        t,
        q: vec![0.0; t * P],
        k: vec![0.0; t * P],
        v: vec![0.0; t * P],
        a: vec![0.0; t * P],
        beta: vec![0.0; t * H],
        state: vec![0.0; H * D * D],
    };
    for (h, (q, k, v, a, be)) in per_head.iter().enumerate() {
        for i in 0..t {
            let o = i * P + h * D;
            inp.q[o..o + D].copy_from_slice(&q[i * D..(i + 1) * D]);
            inp.k[o..o + D].copy_from_slice(&k[i * D..(i + 1) * D]);
            inp.v[o..o + D].copy_from_slice(&v[i * D..(i + 1) * D]);
            inp.a[o..o + D].copy_from_slice(&a[i * D..(i + 1) * D]);
            inp.beta[i * H + h] = be[i];
        }
    }
    if state_scale > 0.0 {
        let mut r = Rng(seed ^ 0x5151);
        for x in inp.state.iter_mut() {
            *x = (state_scale * r.gauss()) as f32;
        }
    }
    inp
}

fn read_f32(path: &std::path::Path) -> Res<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ---------------------------------------------------------------- comparison ---------------

struct Cmp {
    finite: bool,
    max_abs: f64,
    max_ratio: f64,
    viol: usize,
    rel_l2: f64,
    ref_abs_max: f64,
}

fn compare(r: &[f32], g: &[f32], atol: f64, rtol: f64) -> Cmp {
    let mut c = Cmp {
        finite: true,
        max_abs: 0.0,
        max_ratio: 0.0,
        viol: 0,
        rel_l2: 0.0,
        ref_abs_max: 0.0,
    };
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in r.iter().zip(g) {
        if !y.is_finite() || !x.is_finite() {
            c.finite = false;
            c.viol += 1;
            continue;
        }
        let (x, y) = (x as f64, y as f64);
        let e = (x - y).abs();
        let tol = atol + rtol * x.abs();
        c.max_abs = c.max_abs.max(e);
        c.max_ratio = c.max_ratio.max(e / tol);
        c.ref_abs_max = c.ref_abs_max.max(x.abs());
        if e > tol {
            c.viol += 1;
        }
        num += e * e;
        den += x * x;
    }
    c.rel_l2 = (num / den.max(1e-300)).sqrt();
    c
}

fn cmp_json(c: &Cmp) -> Value {
    json!({"finite": c.finite, "max_abs": c.max_abs, "max_err_over_tol": c.max_ratio,
           "violations": c.viol, "rel_l2": c.rel_l2, "ref_abs_max": c.ref_abs_max})
}

fn cmp_ok(c: &Cmp) -> bool {
    c.finite && c.viol == 0
}

// ---------------------------------------------------------------- gpu helpers --------------

struct Dev {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    a: GpuTensor,
    beta: GpuTensor,
}

fn upload_rows(gpu: &mut Gpu, i: &Inputs, r0: usize, r1: usize) -> Res<Dev> {
    let n = r1 - r0;
    Ok(Dev {
        q: gpu.upload_f32(&i.q[r0 * P..r1 * P], &[n, P])?,
        k: gpu.upload_f32(&i.k[r0 * P..r1 * P], &[n, P])?,
        v: gpu.upload_f32(&i.v[r0 * P..r1 * P], &[n, P])?,
        a: gpu.upload_f32(&i.a[r0 * P..r1 * P], &[n, P])?,
        beta: gpu.upload_f32(&i.beta[r0 * H..r1 * H], &[n, H])?,
    })
}

fn launch<'a>(
    d: &'a Dev,
    state: &'a GpuTensor,
    out: &'a GpuTensor,
    n: usize,
    round: bool,
) -> GatedDeltaVectorLaunch<'a> {
    GatedDeltaVectorLaunch {
        query: &d.q,
        key: &d.k,
        value: &d.v,
        log_decay: &d.a,
        beta: &d.beta,
        state,
        output: out,
        slot_idx: None,
        n_dev: None,
        n_tokens: n,
        n_heads: H,
        qk_group: 1,
        head_dim: D,
        decay_per_key: true,
        round_out_bf16: round,
        slot: 0,
        q_row_stride: P,
        k_row_stride: P,
        v_row_stride: P,
        decay_row_stride: P,
        beta_row_stride: H,
        out_row_stride: P,
        state_head_stride: D * D,
        state_row_stride: D,
        state_slot_stride: H * D * D,
    }
}

fn free_dev(gpu: &mut Gpu, d: Dev) -> Res<()> {
    for t in [d.q, d.k, d.v, d.a, d.beta] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Route {
    Serial,
    Chunked,
}

/// Run `splits` consecutive launches (rows `[b0,b1)`, `[b1,b2)`, ...) over the inputs, each on `route`
/// (state carried through the state tensor); returns (output rows, final state).
fn run_route(
    gpu: &mut Gpu,
    i: &Inputs,
    bounds: &[usize],
    routes: &[Route],
    scratch: &GatedDeltaVectorChunkScratch,
    round: bool,
) -> Res<(Vec<f32>, Vec<f32>)> {
    let state = gpu.upload_f32(&i.state, &[H, D, D])?;
    let mut out_all = vec![0f32; i.t * P];
    for (s, w) in bounds.windows(2).enumerate() {
        let (r0, r1) = (w[0], w[1]);
        let n = r1 - r0;
        let dev = upload_rows(gpu, i, r0, r1)?;
        let out = gpu.alloc_tensor(&[n, P], DType::F32)?;
        gpu.hip.memset(&out.buf, 0xff, n * P * 4)?; // NaN canary: every written row must be overwritten
        {
            let l = launch(&dev, &state, &out, n, round);
            match routes[s] {
                Route::Serial => gpu.gated_delta_net_vector_serial_f32(&l)?,
                Route::Chunked => gpu.gated_delta_net_vector_chunked_f32(&l, scratch)?,
            }
        }
        gpu.hip.device_synchronize()?;
        out_all[r0 * P..r1 * P].copy_from_slice(&gpu.download_f32(&out)?);
        gpu.free_tensor(out)?;
        free_dev(gpu, dev)?;
    }
    let st = gpu.download_f32(&state)?;
    gpu.free_tensor(state)?;
    Ok((out_all, st))
}

struct CaseResult {
    name: String,
    pass: bool,
    json: Value,
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    name: &str,
    i: &Inputs,
    bounds: &[usize],
    chunk_routes: &[Route],
    scratch: &GatedDeltaVectorChunkScratch,
    round: bool,
    reference: Option<(&[f32], &[f32])>,
) -> Res<CaseResult> {
    let serial_routes = vec![Route::Serial; bounds.len() - 1];
    let t0 = Instant::now();
    let (ro, rs) = match reference {
        Some((o, s)) => (o.to_vec(), s.to_vec()),
        None => run_route(gpu, i, &[0, i.t], &serial_routes[..1], scratch, round)?,
    };
    let (co, cs) = run_route(gpu, i, bounds, chunk_routes, scratch, round)?;
    let (co2, cs2) = run_route(gpu, i, bounds, chunk_routes, scratch, round)?;
    let (co3, cs3) = run_route(gpu, i, bounds, chunk_routes, scratch, round)?;
    let det = co == co2 && co2 == co3 && cs == cs2 && cs2 == cs3;
    let o = compare(&ro, &co, OUT_ATOL, OUT_RTOL);
    let s = compare(&rs, &cs, ST_ATOL, ST_RTOL);
    // per-head relative norm (worst head), output and state
    let mut worst_head_out = 0f64;
    let mut worst_head_state = 0f64;
    for h in 0..H {
        let (mut n1, mut d1) = (0f64, 0f64);
        for r in 0..i.t {
            for c in 0..D {
                let x = ro[r * P + h * D + c] as f64;
                let y = co[r * P + h * D + c] as f64;
                n1 += (x - y) * (x - y);
                d1 += x * x;
            }
        }
        worst_head_out = worst_head_out.max((n1 / d1.max(1e-300)).sqrt());
        let (mut n2, mut d2) = (0f64, 0f64);
        for x in 0..D * D {
            let a = rs[h * D * D + x] as f64;
            let b = cs[h * D * D + x] as f64;
            n2 += (a - b) * (a - b);
            d2 += a * a;
        }
        worst_head_state = worst_head_state.max((n2 / d2.max(1e-300)).sqrt());
    }
    let pass = cmp_ok(&o) && cmp_ok(&s) && det;
    let js = json!({
        "record": "case", "name": name, "rows": i.t, "bounds": bounds, "pass": pass,
        "deterministic": det, "output": cmp_json(&o), "state": cmp_json(&s),
        "worst_head_rel_l2": {"output": worst_head_out, "state": worst_head_state},
        "secs": t0.elapsed().as_secs_f64(),
    });
    println!("{js}");
    Ok(CaseResult {
        name: name.to_string(),
        pass,
        json: js,
    })
}

// ---------------------------------------------------------------- suites -------------------

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|p| args.get(p + 1).cloned())
}

fn suite(gpu: &mut Gpu, args: &[String]) -> Res<bool> {
    let cap: usize = arg(args, "--cap")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1024);
    let only = arg(args, "--only");
    let big = args.iter().any(|a| a == "--big");
    let scratch = gpu.alloc_gated_delta_vector_chunk_scratch(cap, H)?;
    let mut results: Vec<CaseResult> = Vec::new();
    let mut want = |n: &str| only.as_deref().map_or(true, |o| n.contains(o));

    let mut cases: Vec<(String, usize, Decay, f64, bool, f64, Vec<usize>)> = Vec::new();
    for &t in &[64usize, 128, 512, 4096] {
        cases.push((
            format!("typical_t{t}"),
            t,
            Decay::Typical,
            0.0,
            false,
            0.0,
            vec![0, t],
        ));
    }
    if big {
        cases.push((
            "typical_t16384".into(),
            16384,
            Decay::Typical,
            0.0,
            false,
            0.0,
            vec![0, 16384],
        ));
        cases.push((
            "corr_nzstate_t16384".into(),
            16384,
            Decay::Typical,
            0.9,
            false,
            0.1,
            vec![0, 16384],
        ));
    }
    for &t in &[65usize, 100, 127, 191, 4096 + 37] {
        cases.push((
            format!("tail_t{t}"),
            t,
            Decay::Typical,
            0.0,
            false,
            0.0,
            vec![0, t],
        ));
    }
    for (nm, dc) in [
        ("nodecay", Decay::None),
        ("maxdecay", Decay::Max),
        ("extreme", Decay::Extreme),
        ("mixed", Decay::Mixed),
    ] {
        cases.push((format!("{nm}_t512"), 512, dc, 0.0, false, 0.0, vec![0, 512]));
        cases.push((
            format!("{nm}_nzstate_t512"),
            512,
            dc,
            0.0,
            false,
            0.1,
            vec![0, 512],
        ));
    }
    cases.push((
        "corr09_t4096".into(),
        4096,
        Decay::Typical,
        0.9,
        false,
        0.0,
        vec![0, 4096],
    ));
    cases.push((
        "corr099_beta1_t512".into(),
        512,
        Decay::Typical,
        0.99,
        true,
        0.0,
        vec![0, 512],
    ));
    cases.push((
        "corr09_beta1_nodecay_t512".into(),
        512,
        Decay::None,
        0.9,
        true,
        0.1,
        vec![0, 512],
    ));
    cases.push((
        "nzstate_t4096".into(),
        4096,
        Decay::Typical,
        0.0,
        false,
        0.1,
        vec![0, 4096],
    ));
    cases.push((
        "nzstate_t64".into(),
        64,
        Decay::Typical,
        0.0,
        false,
        0.5,
        vec![0, 64],
    ));
    for (nm, dc, st) in [
        ("typical", Decay::Typical, 0.0),
        ("mixed", Decay::Mixed, 0.1),
    ] {
        // continuation after prefill: chunk+chunk, chunk+serial, serial+chunk vs one serial pass
        cases.push((
            format!("cont_{nm}_chunk_chunk"),
            1000,
            dc,
            0.5,
            false,
            st,
            vec![0, 576, 1000],
        ));
        cases.push((
            format!("cont_{nm}_chunk_serial"),
            1000,
            dc,
            0.5,
            false,
            st,
            vec![0, 576, 1000],
        ));
        cases.push((
            format!("cont_{nm}_serial_chunk"),
            1000,
            dc,
            0.5,
            false,
            st,
            vec![0, 130, 1000],
        ));
        cases.push((
            format!("cont_{nm}_seams_192_448"),
            1000,
            dc,
            0.5,
            false,
            st,
            vec![0, 192, 448, 1000],
        ));
    }

    for (name, t, dc, corr, bhi, ss, bounds) in cases {
        if !want(&name) {
            continue;
        }
        let inp = gen_inputs(t, 0x5eed ^ (t as u64 * 31), dc, corr, bhi, ss);
        let routes: Vec<Route> = if name.ends_with("chunk_serial") {
            vec![Route::Chunked, Route::Serial]
        } else if name.ends_with("serial_chunk") {
            vec![Route::Serial, Route::Chunked]
        } else {
            vec![Route::Chunked; bounds.len() - 1]
        };
        // chunked launches below 64 rows are refused by design: such splits use the serial route
        let routes: Vec<Route> = routes
            .iter()
            .zip(bounds.windows(2))
            .map(|(&r, w)| if w[1] - w[0] < 64 { Route::Serial } else { r })
            .collect();
        results.push(run_case(
            gpu, &name, &inp, &bounds, &routes, &scratch, true, None,
        )?);
    }
    let all = results.iter().all(|r| r.pass);
    let failed: Vec<&str> = results
        .iter()
        .filter(|r| !r.pass)
        .map(|r| r.name.as_str())
        .collect();
    println!(
        "{}",
        json!({"record": "summary", "cases": results.len(), "all_pass": all, "failed": failed, "segment_rows": scratch.segment_rows()})
    );
    let _ = results.iter().map(|r| &r.json).count();
    Ok(all)
}

fn captured(gpu: &mut Gpu, args: &[String]) -> Res<bool> {
    let dir = std::path::PathBuf::from(args.get(2).ok_or("captured DIR")?);
    let cap: usize = arg(args, "--cap")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1024);
    let scratch = gpu.alloc_gated_delta_vector_chunk_scratch(cap, H)?;
    let mut calls: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("meta.json").exists())
        .collect();
    calls.sort();
    let mut all = true;
    let mut n_cases = 0;
    for c in calls {
        let meta: Value = serde_json::from_slice(&std::fs::read(c.join("meta.json"))?)?;
        let n = meta["n"].as_u64().unwrap() as usize;
        if meta["heads"].as_u64() != Some(H as u64)
            || meta["qk_group"].as_u64() != Some(1)
            || meta["per_key"] != json!(true)
        {
            return Err(format!("{}: unexpected geometry {meta}", c.display()).into());
        }
        let inp = Inputs {
            t: n,
            q: read_f32(&c.join("q.f32"))?,
            k: read_f32(&c.join("k.f32"))?,
            v: read_f32(&c.join("v.f32"))?,
            a: read_f32(&c.join("a.f32"))?,
            beta: read_f32(&c.join("beta.f32"))?,
            state: read_f32(&c.join("state_in.f32"))?,
        };
        let round = meta["round"] == json!(true);
        let name = format!("captured_{}", c.file_name().unwrap().to_string_lossy());
        let bounds = vec![0, n];
        let r = run_case(
            gpu,
            &name,
            &inp,
            &bounds,
            &[Route::Chunked],
            &scratch,
            round,
            None,
        )?;
        // serial recomputation must equal the production-captured output bytes (harness sanity)
        if c.join("out.f32").exists() {
            let (so, ss) = run_route(gpu, &inp, &[0, n], &[Route::Serial], &scratch, round)?;
            let po = read_f32(&c.join("out.f32"))?;
            let ps = read_f32(&c.join("state_out.f32"))?;
            println!(
                "{}",
                json!({"record": "serial_vs_captured", "name": name, "out_bytes_equal": so == po, "state_bytes_equal": ss == ps})
            );
        }
        all &= r.pass;
        n_cases += 1;
    }
    println!(
        "{}",
        json!({"record": "summary", "cases": n_cases, "all_pass": all})
    );
    Ok(all)
}

fn bench(gpu: &mut Gpu, args: &[String]) -> Res<bool> {
    let rows: usize = arg(args, "--rows")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(4096);
    let reps: usize = arg(args, "--reps")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(20);
    let caps: Vec<usize> = arg(args, "--caps")
        .unwrap_or_else(|| "512,1024,2048,4096".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let inp = gen_inputs(rows, 0xbe7c, Decay::Typical, 0.3, false, 0.0);
    let dev = upload_rows(gpu, &inp, 0, rows)?;
    let state = gpu.upload_f32(&inp.state, &[H, D, D])?;
    let out = gpu.alloc_tensor(&[rows, P], DType::F32)?;
    let l = launch(&dev, &state, &out, rows, true);
    let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu) -> Res<()>| -> Res<f64> {
        for _ in 0..3 {
            f(gpu)?;
        }
        gpu.hip.device_synchronize()?;
        let t0 = Instant::now();
        for _ in 0..reps {
            f(gpu)?;
        }
        gpu.hip.device_synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    let serial = time(gpu, &mut |g| Ok(g.gated_delta_net_vector_serial_f32(&l)?))?;
    println!(
        "{}",
        json!({"record": "bench", "route": "serial", "rows": rows, "ms_per_layer": serial, "ms_x18": serial * 18.0})
    );
    for cap in caps {
        let scratch = gpu.alloc_gated_delta_vector_chunk_scratch(cap, H)?;
        let both = time(gpu, &mut |g| {
            Ok(g.gated_delta_net_vector_chunked_f32(&l, &scratch)?)
        })?;
        let prep = time(gpu, &mut |g| {
            Ok(g.gated_delta_net_vector_chunked_stages(&l, &scratch, true, false)?)
        })?;
        let scan = time(gpu, &mut |g| {
            Ok(g.gated_delta_net_vector_chunked_stages(&l, &scratch, false, true)?)
        })?;
        println!(
            "{}",
            json!({"record": "bench", "route": "chunked", "rows": rows, "cap": cap, "ms_per_layer": both, "ms_x18": both * 18.0,
                   "prep_ms": prep, "scan_ms": scan, "speedup_vs_serial": serial / both, "scratch_mib": scratch.bytes() as f64 / 1048576.0})
        );
    }
    Ok(true)
}

fn prof(gpu: &mut Gpu, args: &[String]) -> Res<bool> {
    let rows: usize = arg(args, "--rows")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(4096);
    let inp = gen_inputs(rows, 0xbe7c, Decay::Typical, 0.3, false, 0.0);
    let dev = upload_rows(gpu, &inp, 0, rows)?;
    let state = gpu.upload_f32(&inp.state, &[H, D, D])?;
    let out = gpu.alloc_tensor(&[rows, P], DType::F32)?;
    let scratch = gpu.alloc_gated_delta_vector_chunk_scratch(rows, H)?;
    let nch = rows.div_ceil(64);
    let pt = gpu.alloc_tensor(&[nch * H * 16], DType::F32)?;
    let l = launch(&dev, &state, &out, rows, true);
    for _ in 0..5 {
        gpu.gated_delta_net_vector_chunked_stages_prof(&l, &scratch, true, true, Some(&pt))?;
    }
    gpu.hip.device_synchronize()?;
    let raw = gpu.download_f32(&pt)?;
    let ts: Vec<[u64; 5]> = (0..nch * H)
        .map(|i| {
            let mut a = [0u64; 5];
            for (k, x) in a.iter_mut().enumerate() {
                let (lo, hi) = (
                    raw[i * 16 + 2 * k].to_bits() as u64,
                    raw[i * 16 + 2 * k + 1].to_bits() as u64,
                );
                *x = lo | (hi << 32);
            }
            a
        })
        .collect();
    let t0 = ts.iter().map(|a| a[0]).min().unwrap();
    let t1 = ts.iter().map(|a| a[4]).max().unwrap();
    let n = ts.len() as f64;
    let mean =
        |f: &dyn Fn(&[u64; 5]) -> u64| ts.iter().map(|a| f(a) as f64).sum::<f64>() / n / 100.0; // us
    let (p0, p1, p2, p3) = (
        mean(&|a| a[1] - a[0]),
        mean(&|a| a[2] - a[1]),
        mean(&|a| a[3] - a[2]),
        mean(&|a| a[4] - a[3]),
    );
    let span_us = (t1 - t0) as f64 / 100.0;
    let busy: f64 = ts.iter().map(|a| (a[4] - a[0]) as f64 / 100.0).sum();
    let occ_p = gpu
        .occupancy_max_active_blocks("gated_delta_vector_chunk_prep_gfx12", [256, 1, 1], 0)
        .ok();
    let occ_s = gpu
        .occupancy_max_active_blocks("gated_delta_vector_chunk_scan_gfx12", [32, 1, 1], 0)
        .ok();
    println!(
        "{}",
        json!({"record": "prof", "rows": rows, "chunk_heads": ts.len(), "span_us": span_us,
               "mean_us": {"p0_ell": p0, "p1_pairs": p1, "p2_solve": p2, "p3_gemms": p3, "total": p0 + p1 + p2 + p3},
               "avg_concurrent_wgs": busy / span_us, "occupancy_blocks_per_mp": {"prep": occ_p, "scan": occ_s}})
    );
    Ok(true)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("suite");
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 required, got {}", gpu.arch).into());
    }
    let ok = match mode {
        "suite" => suite(&mut gpu, &args)?,
        "captured" => captured(&mut gpu, &args)?,
        "bench" => bench(&mut gpu, &args)?,
        "prof" => prof(&mut gpu, &args)?,
        m => return Err(format!("unknown mode {m}").into()),
    };
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
