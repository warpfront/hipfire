// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node gate + bench of one Raon RDA layer (`rda_forward_rows`: conv q/k/v, `raon_rda_prepare`, the
//! recurrence, the gated RMSNorm) on deterministic synthetic raw projections.
//!
//! ```text
//! rda_layer_node digest [--rows 64,512,4096,16384] [--calls 2]
//! rda_layer_node bench  [--rows 4096] [--layers 18] [--reps 5] [--sets 3]
//! ```
//!
//! `digest` runs `--calls` consecutive calls (conv histories and recurrent state chain) per row count and
//! prints one JSON line per case with an FNV-1a digest of `gated`, the recurrent state and the three conv
//! histories; two processes with different `HIPFIRE_RAON_RDA_*` flags are bit-identical iff the digests
//! agree. `bench` times `--layers` back-to-back layer calls over `--sets` rotating input sets (so the raw
//! planes are not cache-warm) and prints ms per layer / per `--layers` layers.
//!
//! Build: cargo build --release -j 8 -p hipfire-arch-raon --features lab --example rda_layer_node
//! Run (GPU, through fabric): rda_layer_node digest | bench, flags via HIPFIRE_RAON_RDA_FUSED_PREP /
//! HIPFIRE_RAON_RDA_SEGMENT / HIPFIRE_GATED_DELTA_VECTOR_CHUNKED.

use hipfire_arch_raon::rda::{
    rda_forward_rows, RaonRdaLayer, RdaGeometry, RdaLayerState, RdaProjected, RdaScratch,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Approximately N(0, 1): sum of four uniforms, rescaled (cheap; the distribution shape is irrelevant here).
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let u = |s: u32| ((r >> s) & 0xffff) as f32 / 65536.0;
        (u(0) + u(16) + u(32) + u(48) - 2.0) * 1.732_050_8
    }
}

fn bf16(x: f32) -> f32 {
    let u = x.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return x;
    }
    f32::from_bits((u.wrapping_add(0x7fff + ((u >> 16) & 1))) & 0xffff_0000)
}

fn plane(rng: &mut Rng, n: usize, mean: f32, std: f32, round: bool) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v = mean + std * rng.gauss();
            if round {
                bf16(v)
            } else {
                v
            }
        })
        .collect()
}

fn fnv_f32(v: &[f32], mut h: u64) -> u64 {
    for x in v {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

const FNV0: u64 = 0xcbf29ce484222325;

struct Weights {
    cq: GpuTensor,
    ck: GpuTensor,
    cv: GpuTensor,
    a_log: GpuTensor,
    dt_bias: GpuTensor,
    gamma: GpuTensor,
}

struct Raw {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    z: GpuTensor,
    f: GpuTensor,
    b: GpuTensor,
    delta: GpuTensor,
}

fn make_weights(gpu: &mut Gpu, g: &RdaGeometry) -> Res<Weights> {
    let p = g.proj();
    let mut r = Rng(0x5eed_0001);
    Ok(Weights {
        cq: gpu.upload_f32(&plane(&mut r, p * 4, 0.0, 0.5, true), &[p, 4])?,
        ck: gpu.upload_f32(&plane(&mut r, p * 4, 0.0, 0.5, true), &[p, 4])?,
        cv: gpu.upload_f32(&plane(&mut r, p * 4, 0.0, 0.5, true), &[p, 4])?,
        a_log: gpu.upload_f32(&plane(&mut r, g.n_heads, -0.5, 0.5, false), &[g.n_heads])?,
        dt_bias: gpu.upload_f32(&plane(&mut r, p, 0.0, 1.0, false), &[p])?,
        gamma: gpu.upload_f32(&plane(&mut r, g.head_dim, 1.0, 0.1, false), &[g.head_dim])?,
    })
}

fn make_raw(gpu: &mut Gpu, g: &RdaGeometry, rows: usize, seed: u64) -> Res<Raw> {
    let p = g.proj();
    let h = g.n_heads;
    let mut r = Rng(seed);
    Ok(Raw {
        q: gpu.upload_f32(&plane(&mut r, rows * p, 0.0, 2.0, true), &[rows, p])?,
        k: gpu.upload_f32(&plane(&mut r, rows * p, 0.0, 2.0, true), &[rows, p])?,
        v: gpu.upload_f32(&plane(&mut r, rows * p, 0.0, 2.0, true), &[rows, p])?,
        z: gpu.upload_f32(&plane(&mut r, rows * p, 0.0, 2.0, true), &[rows, p])?,
        f: gpu.upload_f32(&plane(&mut r, rows * p, 0.0, 1.5, true), &[rows, p])?,
        b: gpu.upload_f32(&plane(&mut r, rows * h, 0.0, 1.5, true), &[rows, h])?,
        delta: gpu.upload_f32(&plane(&mut r, rows * h, 0.0, 0.5, true), &[rows, h])?,
    })
}

struct Layer {
    state: GpuTensor,
    hq: GpuTensor,
    hk: GpuTensor,
    hv: GpuTensor,
}

fn make_layer(gpu: &mut Gpu, g: &RdaGeometry, seed: u64) -> Res<Layer> {
    let p = g.proj();
    let mut r = Rng(seed);
    Ok(Layer {
        state: gpu.upload_f32(
            &plane(&mut r, g.state_elems(), 0.0, 0.01, false),
            &[g.n_heads, g.head_dim, g.head_dim],
        )?,
        hq: gpu.upload_f32(&plane(&mut r, 3 * p, 0.0, 1.0, true), &[3, p])?,
        hk: gpu.upload_f32(&plane(&mut r, 3 * p, 0.0, 1.0, true), &[3, p])?,
        hv: gpu.upload_f32(&plane(&mut r, 3 * p, 0.0, 1.0, true), &[3, p])?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_layer(
    gpu: &mut Gpu,
    g: &RdaGeometry,
    w: &Weights,
    l: &Layer,
    raw: &Raw,
    scratch: &RdaScratch,
    gated: &GpuTensor,
    rows: usize,
) -> Res<()> {
    let lw = RaonRdaLayer {
        conv_w_q: &w.cq,
        conv_w_k: &w.ck,
        conv_w_v: &w.cv,
        a_log: &w.a_log,
        dt_bias: &w.dt_bias,
        o_norm_gamma: &w.gamma,
    };
    let st = RdaLayerState {
        state: &l.state,
        hist_q: &l.hq,
        hist_k: &l.hk,
        hist_v: &l.hv,
        slot: 0,
        state_slot_stride: g.state_elems(),
    };
    let x = RdaProjected {
        q_raw: &raw.q,
        k_raw: &raw.k,
        v_raw: &raw.v,
        z: &raw.z,
        b: &raw.b,
        delta: &raw.delta,
        f: &raw.f,
    };
    rda_forward_rows(gpu, g, &lw, &st, &x, scratch, gated, rows)?;
    Ok(())
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|p| args.get(p + 1).cloned())
}

fn flags() -> serde_json::Value {
    let e = |k: &str| std::env::var(k).unwrap_or_default();
    json!({
        "HIPFIRE_RAON_RDA_FUSED_PREP": e("HIPFIRE_RAON_RDA_FUSED_PREP"),
        "HIPFIRE_RAON_RDA_SEGMENT": e("HIPFIRE_RAON_RDA_SEGMENT"),
        "HIPFIRE_GATED_DELTA_VECTOR_CHUNKED": e("HIPFIRE_GATED_DELTA_VECTOR_CHUNKED"),
    })
}

fn digest(gpu: &mut Gpu, args: &[String]) -> Res<()> {
    let rows_list: Vec<usize> = arg(args, "--rows")
        .unwrap_or_else(|| "64,512,4096,16384".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let calls: usize = arg(args, "--calls")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2);
    let g = RdaGeometry::raon();
    let w = make_weights(gpu, &g)?;
    for rows in rows_list {
        // Production sizing: scratch for the prefill chunk (<= 4096 rows), longer prompts run as chunks.
        let chunk = rows.min(4096);
        let scratch = RdaScratch::new(gpu, &g, chunk)?;
        let gated = gpu.alloc_tensor(&[chunk, g.proj()], DType::F32)?;
        let l = make_layer(gpu, &g, 0x1a1a)?;
        let mut h = FNV0;
        let mut gated_h = Vec::new();
        for c in 0..calls {
            let mut done = 0;
            while done < rows {
                let n = (rows - done).min(chunk);
                let raw = make_raw(gpu, &g, n, 0xfeed_0000 + (c * 131 + done) as u64)?;
                run_layer(gpu, &g, &w, &l, &raw, &scratch, &gated, n)?;
                gpu.hip.device_synchronize()?;
                let gh = gpu.download_f32(&gated)?;
                let part = fnv_f32(&gh[..n * g.proj()], FNV0);
                gated_h.push(format!("{part:016x}"));
                h = fnv_f32(&gh[..n * g.proj()], h);
                for t in [raw.q, raw.k, raw.v, raw.z, raw.f, raw.b, raw.delta] {
                    gpu.free_tensor(t)?;
                }
                done += n;
            }
        }
        let st = gpu.download_f32(&l.state)?;
        let hq = gpu.download_f32(&l.hq)?;
        let hk = gpu.download_f32(&l.hk)?;
        let hv = gpu.download_f32(&l.hv)?;
        let sh = fnv_f32(&st, FNV0);
        h = fnv_f32(&st, h);
        let mut hh = Vec::new();
        for v in [&hq, &hk, &hv] {
            hh.push(format!("{:016x}", fnv_f32(v, FNV0)));
            h = fnv_f32(v, h);
        }
        println!(
            "{}",
            json!({"record": "digest", "rows": rows, "calls": calls, "chunk": chunk, "digest": format!("{h:016x}"),
                   "gated_parts": gated_h, "state": format!("{sh:016x}"), "hist": hh,
                   "finite": st.iter().all(|x| x.is_finite()), "flags": flags()})
        );
        for t in [l.state, l.hq, l.hk, l.hv, gated] {
            gpu.free_tensor(t)?;
        }
    }
    Ok(())
}

fn bench(gpu: &mut Gpu, args: &[String]) -> Res<()> {
    let rows: usize = arg(args, "--rows")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(4096);
    let layers: usize = arg(args, "--layers")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(18);
    let reps: usize = arg(args, "--reps")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(5);
    let sets: usize = arg(args, "--sets")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(3);
    let g = RdaGeometry::raon();
    let w = make_weights(gpu, &g)?;
    let chunk = rows.min(4096);
    let scratch = RdaScratch::new(gpu, &g, chunk)?;
    let gated = gpu.alloc_tensor(&[chunk, g.proj()], DType::F32)?;
    let raws: Vec<Raw> = (0..sets)
        .map(|s| make_raw(gpu, &g, chunk, 0xbeef_0000 + s as u64))
        .collect::<Res<_>>()?;
    let ls: Vec<Layer> = (0..layers.min(4))
        .map(|s| make_layer(gpu, &g, 0x2b2b + s as u64))
        .collect::<Res<_>>()?;
    let one_pass = |gpu: &mut Gpu| -> Res<()> {
        for i in 0..layers {
            // rows beyond one chunk repeat the chunk (timing only)
            let mut done = 0;
            while done < rows {
                let n = (rows - done).min(chunk);
                run_layer(
                    gpu,
                    &g,
                    &w,
                    &ls[i % ls.len()],
                    &raws[i % sets],
                    &scratch,
                    &gated,
                    n,
                )?;
                done += n;
            }
        }
        Ok(())
    };
    for _ in 0..2 {
        one_pass(gpu)?;
    }
    gpu.hip.device_synchronize()?;
    let mut ms = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        one_pass(gpu)?;
        gpu.hip.device_synchronize()?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    println!(
        "{}",
        json!({"record": "bench", "rows": rows, "layers": layers, "sets": sets, "reps_ms": ms,
               "median_ms": median, "ms_per_layer": median / layers as f64, "flags": flags()})
    );
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("digest");
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 required, got {}", gpu.arch).into());
    }
    match mode {
        "digest" => digest(&mut gpu, &args),
        "bench" => bench(&mut gpu, &args),
        m => Err(format!("unknown mode {m}").into()),
    }
}
