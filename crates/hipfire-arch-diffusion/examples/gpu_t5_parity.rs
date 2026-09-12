// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vs host T5 encoder parity, and the T5-encode timing that justifies the
//! whole offload.
//!
//! Compares `t5_gpu::encode` (f16 weights, f32 accumulate, WMMA GEMM) against
//! the f32 host oracle `t5::encode` on identical weights and token ids, and
//! reports the max relative error plus both wall times.
//!
//! Two modes:
//!
//! - **real** (default) — loads `text_encoder_2/` from a diffusers pipe dir.
//!   This is the acceptance run and needs the real T5-XXL checkpoint
//!   (`d_model` 4096, `d_ff` 10240, 24 layers). The host side of the compare
//!   costs ~55 s per prompt at that geometry; budget for it.
//!
//!   ```
//!   cargo run --release --features lab --example gpu_t5_parity \
//!     -p hipfire-arch-diffusion -- /home/user/flux-pipe
//!   ```
//!
//! - **`--synthetic`** — random weights at a tiny geometry, no checkpoint
//!   needed. This is what gates the implementation on a dev box; it exercises
//!   every kernel and every shape rule the real path uses (gated GELU, the
//!   relative bias, the K % 64 LDS GEMM route) at a size that runs in
//!   milliseconds. Geometry is tunable so the same binary can also time a
//!   REAL-WIDTH slice for extrapolation:
//!
//!   ```
//!   # tiny gate
//!   cargo run --release --features lab --example gpu_t5_parity \
//!     -p hipfire-arch-diffusion -- --synthetic
//!   # one real-width layer, for a per-layer cost that scales to 24
//!   cargo run --release --features lab --example gpu_t5_parity \
//!     -p hipfire-arch-diffusion -- --synthetic --layers 1 --d 4096 \
//!     --dff 10240 --heads 64 --n 256 --no-host
//!   ```
//!
//! Tolerance: **rel ≤ 5e-3**. The GPU path holds weights in f16 while the
//! host oracle is f32 throughout, so the two cannot agree to f32 epsilon;
//! 5e-3 clears f16 rounding on a 24-layer residual stack with margin while
//! still catching a miswire (a wrong transpose or operand order lands at
//! rel ~O(1), not 1e-2). Exit 1 on failure.

use hipfire_arch_diffusion::flux::Tensor;
use hipfire_arch_diffusion::t5::{self, T5Config, T5Weights};
use hipfire_arch_diffusion::t5_gpu::{self, GpuT5Weights};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::time::Instant;

const TOL: f32 = 5e-3;

/// Deterministic xorshift64* — the goldens must be reproducible across runs
/// and machines, and pulling a real RNG crate in for a fixture is not worth
/// the dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[-scale, scale)`.
    fn uniform(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32; // [0,1)
        (u * 2.0 - 1.0) * scale
    }

    fn tensor(&mut self, rows: usize, cols: usize, scale: f32) -> Tensor {
        Tensor {
            data: (0..rows * cols).map(|_| self.uniform(scale)).collect(),
            rows,
            cols,
        }
    }
}

struct Args {
    synthetic: bool,
    pipe: PathBuf,
    layers: usize,
    d: usize,
    d_ff: usize,
    heads: usize,
    n: usize,
    /// Skip the host oracle. Only for timing a real-width synthetic slice,
    /// where the host side would take minutes and proves nothing new.
    no_host: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        synthetic: false,
        pipe: PathBuf::from("/home/user/flux-pipe"),
        // Tiny default geometry. d and d_ff are multiples of 64 so the
        // synthetic run takes the SAME LDS GEMM route the real geometry does
        // (4096 / 10240 are both K % 64 == 0) — a fixture that silently fell
        // back to the 16-step kernel would gate a path nobody ships.
        layers: 2,
        d: 128,
        d_ff: 256,
        heads: 4,
        n: 64,
        no_host: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let v = |i: usize| -> usize {
            argv.get(i + 1)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| panic!("expected a number after {}", argv[i]))
        };
        match argv[i].as_str() {
            "--synthetic" => a.synthetic = true,
            "--no-host" => a.no_host = true,
            "--layers" => {
                a.layers = v(i);
                i += 1;
            }
            "--d" => {
                a.d = v(i);
                i += 1;
            }
            "--dff" => {
                a.d_ff = v(i);
                i += 1;
            }
            "--heads" => {
                a.heads = v(i);
                i += 1;
            }
            "--n" => {
                a.n = v(i);
                i += 1;
            }
            other => a.pipe = PathBuf::from(other),
        }
        i += 1;
    }
    a
}

/// Random gated (v1.1) T5 weights at the requested geometry.
///
/// Weight scale is `1/sqrt(fan_in)`: at `d_model` 4096 a unit-scale random
/// weight would drive the pre-softmax logits into the thousands and the
/// residual stream to overflow, and the comparison would be measuring
/// saturation rather than the kernels.
fn synthetic_weights(a: &Args) -> T5Weights {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let d = a.d;
    let d_ff = a.d_ff;
    let hd = d / a.heads;
    let vocab = 256usize;
    let buckets = 32usize;
    let ws = 1.0 / (d as f32).sqrt();
    let ffs = 1.0 / (d_ff as f32).sqrt();
    let config = T5Config {
        d_model: d,
        d_ff,
        d_kv: hd,
        num_heads: a.heads,
        num_layers: a.layers,
        vocab_size: vocab,
        relative_attention_num_buckets: buckets,
        relative_attention_max_distance: 128,
        layer_norm_epsilon: 1e-6,
    };
    let mut w = T5Weights {
        embed: rng.tensor(vocab, d, 1.0),
        rel_bias: rng.tensor(buckets, a.heads, 0.5),
        // Norm scales sit near 1, as trained ones do; centring them on 0
        // would make every layer output ~0 and hide a scale bug.
        final_norm: Tensor {
            data: (0..d).map(|_| 1.0 + rng.uniform(0.1)).collect(),
            rows: d,
            cols: 1,
        },
        q: vec![],
        k: vec![],
        v: vec![],
        o: vec![],
        attn_norm: vec![],
        wi: vec![],
        wi_gate: vec![],
        wo: vec![],
        ffn_norm: vec![],
        config,
    };
    for _ in 0..a.layers {
        w.q.push(rng.tensor(d, d, ws));
        w.k.push(rng.tensor(d, d, ws));
        w.v.push(rng.tensor(d, d, ws));
        w.o.push(rng.tensor(d, d, ws));
        w.wi.push(rng.tensor(d_ff, d, ws));
        w.wi_gate.push(rng.tensor(d_ff, d, ws));
        w.wo.push(rng.tensor(d, d_ff, ffs));
        for slot in [&mut w.attn_norm, &mut w.ffn_norm] {
            slot.push(Tensor {
                data: (0..d).map(|_| 1.0 + rng.uniform(0.1)).collect(),
                rows: d,
                cols: 1,
            });
        }
    }
    w
}

/// Max relative error, using `max(|a|,|b|)` normalised by the tensor's own
/// scale so a near-zero element does not manufacture a huge ratio.
fn max_rel(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs() / scale;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

fn main() {
    let a = parse_args();

    let host: T5Weights = if a.synthetic {
        println!(
            "mode=synthetic layers={} d_model={} d_ff={} heads={} d_kv={} n={}",
            a.layers,
            a.d,
            a.d_ff,
            a.heads,
            a.d / a.heads,
            a.n
        );
        assert_eq!(a.d % a.heads, 0, "d_model must be divisible by heads");
        synthetic_weights(&a)
    } else {
        let src = SafetensorsSource::open(&a.pipe.join("text_encoder_2"))
            .unwrap_or_else(|e| panic!("open {:?}/text_encoder_2: {e:?}", a.pipe));
        let w = T5Weights::load(&src).unwrap_or_else(|e| panic!("T5Weights::load: {e}"));
        println!(
            "mode=real pipe={:?} layers={} d_model={} d_ff={} heads={} d_kv={}",
            a.pipe,
            w.config.num_layers,
            w.config.d_model,
            w.config.d_ff,
            w.config.num_heads,
            w.config.d_kv
        );
        w
    };

    // Token ids: deterministic, in range, and NOT all distinct — a real
    // prompt repeats tokens and ends in a pad run, and the relative-bias
    // indexing is position- not token-driven, so repeats are the honest case.
    let len = if a.synthetic { a.n } else { 256 };
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let ids: Vec<u32> = (0..len)
        .map(|i| {
            if i >= len * 3 / 4 {
                0 // pad tail, as `condition_prompt` builds
            } else {
                (rng.next_u64() % (host.config.vocab_size as u64 - 1)) as u32 + 1
            }
        })
        .collect();
    let mask: Vec<u8> = ids.iter().map(|&t| u8::from(t != 0)).collect();

    let mut gpu = Gpu::init().unwrap_or_else(|e| panic!("Gpu::init: {e:?}"));
    println!("gpu arch={}", gpu.arch);

    let t0 = Instant::now();
    let mut gw = GpuT5Weights::from_host(&mut gpu, &host)
        .unwrap_or_else(|e| panic!("GpuT5Weights::from_host: {e}"));
    let upload_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Warm run: first use of each kernel shape pays HIP JIT, which is not
    // what this example is reporting.
    let warm = t5_gpu::encode(&mut gpu, &mut gw, &host, &ids, &mask)
        .unwrap_or_else(|e| panic!("t5_gpu::encode (warm): {e}"));
    gpu.free_tensor(warm).expect("free warm");

    let t1 = Instant::now();
    let got = t5_gpu::encode_host(&mut gpu, &mut gw, &host, &ids, &mask)
        .unwrap_or_else(|e| panic!("t5_gpu::encode: {e}"));
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;

    println!(
        "upload_ms={upload_ms:.1} gpu_encode_ms={gpu_ms:.2} rows={len} d={}",
        host.config.d_model
    );

    let freed = gw.free_gpu(&mut gpu);
    println!("freed_buffers={freed}");

    if a.no_host {
        println!("SKIP host oracle (--no-host): timing only, no parity verdict");
        return;
    }

    let t2 = Instant::now();
    let (want, _layers) = t5::encode(&host, &ids, &mask);
    let host_ms = t2.elapsed().as_secs_f64() * 1e3;

    let (rel, at) = max_rel(&want, &got);
    println!(
        "host_encode_ms={host_ms:.2} speedup={:.1}x max_rel={rel:.3e} at={at} tol={TOL:.0e}",
        host_ms / gpu_ms.max(1e-9)
    );
    if rel > TOL {
        eprintln!(
            "FAIL: max_rel {rel:.3e} > {TOL:.0e} (element {at}: host {} vs gpu {})",
            want[at], got[at]
        );
        std::process::exit(1);
    }
    println!("PASS");
}
