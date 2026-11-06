// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — node bench + bit-identity gate of the Raon mHC site chain
// (selector GEMM, route finalize, aggregate, postmix) at production geometry.
//
//   bench_stream_mix_chain_gfx1201 [rows=4096] [sites=50] [arms=u,f,b] [rounds=3]
//
// Arms (every arm must produce the same pre/post/res/out/new-stream bits as
// `u`; the program aborts otherwise):
//   u  unfused, F32 streams: gemm, route_helmert, aggregate, postmix  (incumbent)
//   f  fused,   F32 streams: gemm, route_aggregate, postmix
//   b  fused,   BF16 streams: gemm(xbf16), route_aggregate(bf16), postmix(bf16)
//
// Timing is event-bracketed per kernel inside the real chain order with
// ping-pong stream buffers (working set >> last-level cache, as in the model).
// Logical IO bytes are source-level (each kernel's reads + writes), not HBM
// counters.

use rdna_compute::stream_mix::{
    StreamAggregate, StreamGamma, StreamPostmix, StreamRounding, StreamRouteAggregate,
    StreamRouteHelmert,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const N: usize = 4;
const W: usize = 2560;
const XD: usize = N * W;
const ZD: usize = 2 * N + (N - 1) * (N - 1);

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return x;
    }
    f32::from_bits((u + 0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bits(x: f32) -> u16 {
    (bf16_round(x).to_bits() >> 16) as u16
}

fn upload_bf16(gpu: &Gpu, bits: &[u16], shape: &[usize]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), bits.len() * 2) };
    let mut t = gpu.upload_raw(bytes, &[bytes.len()]).expect("upload bf16");
    t.shape = shape.to_vec();
    t.dtype = DType::BF16;
    t
}

struct Bufs {
    z: GpuTensor,
    pre: GpuTensor,
    post: GpuTensor,
    res: GpuTensor,
    out: GpuTensor,
    sa: GpuTensor,
    sb: GpuTensor,
}

struct Data {
    rows: usize,
    selector: GpuTensor,
    params: GpuTensor,
    gamma: GpuTensor,
    branch: GpuTensor,
    streams0: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    U,
    F,
    B,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::U => "u",
            Arm::F => "f",
            Arm::B => "b",
        }
    }
    fn bf16_streams(self) -> bool {
        self == Arm::B
    }
}

fn fresh(gpu: &mut Gpu, arm: Arm, rows: usize, streams0: &[f32]) -> Bufs {
    let mut a = |d: &[usize]| gpu.alloc_tensor(d, DType::F32).expect("alloc");
    let (z, pre, post, res, out) = (
        a(&[rows, ZD]),
        a(&[rows, N]),
        a(&[rows, N]),
        a(&[rows, N, N]),
        a(&[rows, W]),
    );
    let sdt = if arm.bf16_streams() {
        DType::BF16
    } else {
        DType::F32
    };
    let sa = gpu.alloc_tensor(&[rows, N, W], sdt).expect("alloc");
    let sb = gpu.alloc_tensor(&[rows, N, W], sdt).expect("alloc");
    let host: Vec<u8> = if arm.bf16_streams() {
        streams0
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect()
    } else {
        streams0.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    gpu.hip.memcpy_htod(&sa.buf, &host).expect("htod");
    Bufs {
        z,
        pre,
        post,
        res,
        out,
        sa,
        sb,
    }
}

fn free(gpu: &mut Gpu, b: Bufs) {
    for t in [b.z, b.pre, b.post, b.res, b.out, b.sa, b.sb] {
        gpu.free_tensor(t).expect("free");
    }
}

const KERNELS: [&str; 4] = ["gemm", "route", "aggregate", "postmix"];

type Ev = Vec<(usize, hip_bridge::Event, hip_bridge::Event)>;

fn site(gpu: &mut Gpu, arm: Arm, d: &Data, b: &Bufs, cur_is_a: bool, ev: &mut Ev, timed: bool) {
    let (cur, oth) = if cur_is_a {
        (&b.sa, &b.sb)
    } else {
        (&b.sb, &b.sa)
    };
    let rows = d.rows;
    let mut bracket = |gpu: &mut Gpu, k: usize, f: &mut dyn FnMut(&mut Gpu)| {
        if timed {
            let s = gpu.hip.event_create().expect("ev");
            let e = gpu.hip.event_create().expect("ev");
            gpu.hip.event_record(&s, None).expect("rec");
            f(gpu);
            gpu.hip.event_record(&e, None).expect("rec");
            ev.push((k, s, e));
        } else {
            f(gpu);
        }
    };
    bracket(gpu, 0, &mut |g| {
        if arm.bf16_streams() {
            g.gemm_bf16_xbf16_rows(&d.selector, cur, &b.z, ZD, XD, rows, rows >= 512)
                .expect("gemm");
        } else {
            g.gemm_bf16_xf32_tiled_rows(&d.selector, cur, &b.z, ZD, XD, rows)
                .expect("gemm");
        }
    });
    let gamma = Some(StreamGamma {
        weight: &d.gamma,
        eps: 1e-6,
    });
    match arm {
        Arm::U => {
            bracket(gpu, 1, &mut |g| {
                g.stream_mix_route_helmert(&StreamRouteHelmert {
                    streams: cur,
                    z_raw: &b.z,
                    params: &d.params,
                    pre: &b.pre,
                    post: &b.post,
                    res: &b.res,
                    rows,
                    n: N,
                    width: W,
                    post_factor: 2.0,
                    selector_eps: 1e-5,
                    rounding: StreamRounding::Bf16,
                })
                .expect("route");
            });
            bracket(gpu, 2, &mut |g| {
                g.stream_mix_aggregate(&StreamAggregate {
                    streams: cur,
                    pre: &b.pre,
                    norm: gamma,
                    out: &b.out,
                    rows,
                    n: N,
                    width: W,
                    rounding: StreamRounding::Bf16,
                })
                .expect("aggregate");
            });
        }
        Arm::F | Arm::B => {
            bracket(gpu, 1, &mut |g| {
                g.stream_mix_route_aggregate(&StreamRouteAggregate {
                    streams: cur,
                    z_raw: &b.z,
                    params: &d.params,
                    pre: &b.pre,
                    post: &b.post,
                    res: &b.res,
                    norm: gamma,
                    out: &b.out,
                    rows,
                    n: N,
                    width: W,
                    post_factor: 2.0,
                    selector_eps: 1e-5,
                    rounding: StreamRounding::Bf16,
                })
                .expect("route_aggregate");
            });
        }
    }
    bracket(gpu, 3, &mut |g| {
        g.stream_mix_postmix(&StreamPostmix {
            streams: cur,
            branch: &d.branch,
            post: &b.post,
            res: &b.res,
            out: oth,
            rows,
            n: N,
            width: W,
            rounding: StreamRounding::Bf16,
        })
        .expect("postmix");
    });
}

/// Source-level logical bytes per kernel (reads + writes).
fn logical_bytes(arm: Arm, rows: usize) -> [f64; 4] {
    let r = rows as f64;
    let e = if arm.bf16_streams() { 2.0 } else { 4.0 };
    let s = r * XD as f64 * e; // one pass over the stream stack
    let a = r * W as f64 * 4.0; // [rows, width] F32 plane
    let z = r * ZD as f64 * 4.0;
    let ctl = r * (2 * N + N * N) as f64 * 4.0; // pre, post, res
    let gemm = s + z;
    match arm {
        Arm::U => [
            gemm,
            s + z + ctl,
            s + r * N as f64 * 4.0 + a,
            s + a + r * (N + N * N) as f64 * 4.0 + s,
        ],
        Arm::F | Arm::B => [
            gemm,
            s + z + ctl + a,
            0.0,
            s + a + r * (N + N * N) as f64 * 4.0 + s,
        ],
    }
}

fn widen_streams(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    if t.dtype == DType::BF16 {
        gpu.download_raw_bytes(t)
            .unwrap()
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    } else {
        gpu.download_f32(t).unwrap()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(4096);
    let sites: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(50);
    let arms: Vec<Arm> = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("u,f,b")
        .split(',')
        .map(|a| match a {
            "u" => Arm::U,
            "f" => Arm::F,
            "b" => Arm::B,
            other => panic!("unknown arm {other}"),
        })
        .collect();
    let rounds: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);

    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    let mut rng = Lcg(0x5eed_0431);

    let streams0: Vec<f32> = (0..rows * XD)
        .map(|_| bf16_round(rng.next() * 2.0))
        .collect();
    let sel_bits: Vec<u16> = (0..ZD * XD).map(|_| bf16_bits(rng.next() * 0.02)).collect();
    let mut params = vec![0.3f32, 0.2, 0.4, 0.1];
    for _ in 0..(N + N + (N - 1) * (N - 1)) {
        params.push(rng.next() * 0.5);
    }
    let gamma_bits: Vec<u16> = (0..W).map(|_| bf16_bits(1.0 + rng.next() * 0.2)).collect();
    let branch: Vec<f32> = (0..rows * W).map(|_| bf16_round(rng.next())).collect();
    let d = Data {
        rows,
        selector: upload_bf16(&gpu, &sel_bits, &[ZD, XD]),
        params: gpu.upload_f32(&params, &[params.len()]).unwrap(),
        gamma: upload_bf16(&gpu, &gamma_bits, &[W]),
        branch: gpu.upload_f32(&branch, &[rows, W]).unwrap(),
        streams0,
    };

    // Bit identity: every arm against the first arm after 1 site and after `sites`
    // ping-ponged sites (all of pre/post/res/out/new streams).
    type Out = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let snap = |gpu: &Gpu, b: &Bufs, cur_is_a: bool| -> Out {
        gpu.hip.device_synchronize().unwrap();
        let fin = if cur_is_a { &b.sa } else { &b.sb };
        (
            gpu.download_f32(&b.pre).unwrap(),
            gpu.download_f32(&b.post).unwrap(),
            gpu.download_f32(&b.res).unwrap(),
            gpu.download_f32(&b.out).unwrap(),
            widen_streams(gpu, fin),
        )
    };
    let mut reference: Vec<Out> = Vec::new();
    let mut all_ok = true;
    for (ai, &arm) in arms.iter().enumerate() {
        let b = fresh(&mut gpu, arm, rows, &d.streams0);
        let mut cur_is_a = true;
        let mut none = Vec::new();
        let mut got: Vec<Out> = Vec::new();
        for s in 1..=sites {
            site(&mut gpu, arm, &d, &b, cur_is_a, &mut none, false);
            cur_is_a = !cur_is_a;
            if s == 1 || s == sites {
                got.push(snap(&gpu, &b, cur_is_a));
            }
        }
        for (ci, g) in got.iter().enumerate() {
            let at = if ci == 0 { 1 } else { sites };
            if ai == 0 {
                continue;
            }
            let r = &reference[ci];
            let names = ["pre", "post", "res", "out", "streams"];
            let pairs = [
                (&r.0, &g.0),
                (&r.1, &g.1),
                (&r.2, &g.2),
                (&r.3, &g.3),
                (&r.4, &g.4),
            ];
            for (nm, (x, y)) in names.iter().zip(pairs) {
                let bad = x
                    .iter()
                    .zip(y.iter())
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                let nan = y.iter().filter(|v| !v.is_finite()).count();
                let maxd = x
                    .iter()
                    .zip(y.iter())
                    .map(|(p, q)| (p - q).abs())
                    .fold(0f32, |m, v| if v.is_nan() { m } else { m.max(v) });
                println!(
                    "bitcheck arm={} after {at} site(s) {nm}: {} elems, {bad} differ, max|d|={maxd:e}, nonfinite={nan}",
                    arm.name(),
                    x.len()
                );
                all_ok &= bad == 0;
            }
        }
        if ai == 0 {
            reference = got;
        }
        free(&mut gpu, b);
    }
    if !all_ok {
        println!("BITCHECK FAILED");
        std::process::exit(3);
    }

    // Timing: interleave arms across rounds.
    for round in 0..rounds {
        for &arm in &arms {
            let b = fresh(&mut gpu, arm, rows, &d.streams0);
            let mut cur_is_a = true;
            let mut ev: Ev = Vec::new();
            let warm = 5usize;
            for i in 0..(sites + warm) {
                site(&mut gpu, arm, &d, &b, cur_is_a, &mut ev, i >= warm);
                cur_is_a = !cur_is_a;
            }
            gpu.hip.device_synchronize().unwrap();
            let mut tot = [0f64; 4];
            let mut cnt = [0usize; 4];
            for (k, s, e) in &ev {
                tot[*k] += gpu.hip.event_elapsed_ms(s, e).unwrap() as f64;
                cnt[*k] += 1;
            }
            let lb = logical_bytes(arm, rows);
            let mut line = format!("round={round} arm={} rows={rows}", arm.name());
            let mut chain = 0.0;
            let mut nonmatrix = 0.0;
            for k in 0..4 {
                if cnt[k] == 0 {
                    continue;
                }
                let ms = tot[k] / cnt[k] as f64;
                chain += ms;
                if k != 0 {
                    nonmatrix += ms;
                }
                line += &format!(
                    " {}={:.4}ms({:.0}GB/s,{:.1}MB)",
                    KERNELS[k],
                    ms,
                    lb[k] / 1e9 / (ms / 1e3),
                    lb[k] / 1e6
                );
            }
            println!(
                "{line} nonmatrix/site={nonmatrix:.4}ms nonmatrix50={:.2}ms chain/site={chain:.4}ms chain50={:.2}ms",
                nonmatrix * 50.0,
                chain * 50.0
            );
            for (_, s, e) in ev {
                gpu.hip.event_destroy(s).unwrap();
                gpu.hip.event_destroy(e).unwrap();
            }
            free(&mut gpu, b);
        }
    }
}
