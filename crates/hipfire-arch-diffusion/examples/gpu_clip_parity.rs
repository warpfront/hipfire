// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vs host CLIP-L text-encoder parity.
//!
//! Compares `clip_gpu::encode` (f16 weights, f32 accumulate) against the f32
//! host oracle `clip::encode` on identical weights and token ids. Checks BOTH
//! outputs, not just one:
//!
//! - `last_hidden_state` `[77, hidden]` — catches a wrong layer body.
//! - `pooled` `[hidden]` — catches a wrong EOS position. Pooling reads the row
//!   at `argmax(input_ids)`, so a mirror that took row 0, the last row, or the
//!   last non-pad row would still match `last_hidden_state` exactly and be
//!   silently wrong in the only vector FLUX actually consumes. The fixture ids
//!   below place the max id in the interior specifically so those three wrong
//!   answers all differ from the right one.
//!
//! Modes and tolerance mirror `gpu_t5_parity`:
//!
//! ```
//! cargo run --release --features lab --example gpu_clip_parity \
//!   -p hipfire-arch-diffusion -- /home/user/flux-pipe
//! cargo run --release --features lab --example gpu_clip_parity \
//!   -p hipfire-arch-diffusion -- --synthetic
//! ```
//!
//! Tolerance **rel ≤ 5e-3** — f16 weights against an f32 oracle. Exit 1 on
//! failure.

use hipfire_arch_diffusion::clip::{self, ClipConfig, ClipWeights};
use hipfire_arch_diffusion::clip_gpu::{self, GpuClipWeights};
use hipfire_arch_diffusion::flux::Tensor;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::time::Instant;

const TOL: f32 = 5e-3;

/// Deterministic xorshift64* (see `gpu_t5_parity` for why not a real RNG).
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

    fn uniform(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
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
    h: usize,
    inter: usize,
    heads: usize,
    n: usize,
}

fn parse_args() -> Args {
    // Tiny default geometry; `h` and `inter` are multiples of 64 so the
    // synthetic run takes the same LDS GEMM route the real 768/3072 does.
    let mut a = Args {
        synthetic: false,
        pipe: PathBuf::from("/home/user/flux-pipe"),
        layers: 2,
        h: 128,
        inter: 256,
        heads: 4,
        n: 77,
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
            "--layers" => {
                a.layers = v(i);
                i += 1;
            }
            "--hidden" => {
                a.h = v(i);
                i += 1;
            }
            "--inter" => {
                a.inter = v(i);
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

fn synthetic_weights(a: &Args) -> ClipWeights {
    let mut rng = Rng(0x5EED_4321_DCBA_0002);
    let h = a.h;
    let inter = a.inter;
    let vocab = 256usize;
    let ws = 1.0 / (h as f32).sqrt();
    let is = 1.0 / (inter as f32).sqrt();
    let config = ClipConfig {
        hidden_size: h,
        intermediate_size: inter,
        num_attention_heads: a.heads,
        num_hidden_layers: a.layers,
        vocab_size: vocab,
        max_position_embeddings: a.n.max(77),
        layer_norm_eps: 1e-5,
        // The FLUX clip_l setting, and the only one the GPU path implements.
        hidden_act: "quick_gelu".to_string(),
    };
    // Norm gammas near 1 / betas near 0, as trained ones are.
    let mut affine = |rng: &mut Rng, n: usize, centre: f32| Tensor {
        data: (0..n).map(|_| centre + rng.uniform(0.1)).collect(),
        rows: n,
        cols: 1,
    };
    let mut w = ClipWeights {
        token_embed: rng.tensor(vocab, h, 1.0),
        pos_embed: rng.tensor(config.max_position_embeddings, h, 0.2),
        final_ln_w: affine(&mut rng, h, 1.0),
        final_ln_b: affine(&mut rng, h, 0.0),
        ln1_w: vec![],
        ln1_b: vec![],
        q_w: vec![],
        q_b: vec![],
        k_w: vec![],
        k_b: vec![],
        v_w: vec![],
        v_b: vec![],
        out_w: vec![],
        out_b: vec![],
        ln2_w: vec![],
        ln2_b: vec![],
        fc1_w: vec![],
        fc1_b: vec![],
        fc2_w: vec![],
        fc2_b: vec![],
        config,
    };
    for _ in 0..a.layers {
        w.ln1_w.push(affine(&mut rng, h, 1.0));
        w.ln1_b.push(affine(&mut rng, h, 0.0));
        w.q_w.push(rng.tensor(h, h, ws));
        w.q_b.push(rng.tensor(h, 1, 0.05));
        w.k_w.push(rng.tensor(h, h, ws));
        w.k_b.push(rng.tensor(h, 1, 0.05));
        w.v_w.push(rng.tensor(h, h, ws));
        w.v_b.push(rng.tensor(h, 1, 0.05));
        w.out_w.push(rng.tensor(h, h, ws));
        w.out_b.push(rng.tensor(h, 1, 0.05));
        w.ln2_w.push(affine(&mut rng, h, 1.0));
        w.ln2_b.push(affine(&mut rng, h, 0.0));
        w.fc1_w.push(rng.tensor(inter, h, ws));
        w.fc1_b.push(rng.tensor(inter, 1, 0.05));
        w.fc2_w.push(rng.tensor(h, inter, is));
        w.fc2_b.push(rng.tensor(h, 1, 0.05));
    }
    w
}

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

    let host: ClipWeights = if a.synthetic {
        println!(
            "mode=synthetic layers={} hidden={} inter={} heads={} head_dim={} n={}",
            a.layers,
            a.h,
            a.inter,
            a.heads,
            a.h / a.heads,
            a.n
        );
        assert_eq!(a.h % a.heads, 0, "hidden must be divisible by heads");
        synthetic_weights(&a)
    } else {
        let src = SafetensorsSource::open(&a.pipe.join("text_encoder"))
            .unwrap_or_else(|e| panic!("open {:?}/text_encoder: {e:?}", a.pipe));
        let w = ClipWeights::load(&src).unwrap_or_else(|e| panic!("ClipWeights::load: {e}"));
        println!(
            "mode=real pipe={:?} layers={} hidden={} inter={} heads={} act={}",
            a.pipe,
            w.config.num_hidden_layers,
            w.config.hidden_size,
            w.config.intermediate_size,
            w.config.num_attention_heads,
            w.config.hidden_act
        );
        w
    };

    // Fixture ids shaped like a real CLIP frame: BOS, body, EOT, pad tail.
    // The EOT id is the LARGEST id and sits in the interior, so `argmax`
    // pooling picks a row that is neither the first, the last, nor the last
    // non-pad — a wrong pooling rule cannot accidentally agree.
    let len = a.n;
    let vocab = host.config.vocab_size as u64;
    let eot = (vocab - 1) as u32;
    let mut rng = Rng(0xBEEF_0000_9876_5432);
    let body_end = len / 2;
    let ids: Vec<u32> = (0..len)
        .map(|i| {
            if i == 0 {
                1 // BOS
            } else if i < body_end {
                (rng.next_u64() % (vocab - 2)) as u32 + 1
            } else if i == body_end {
                eot
            } else {
                2 // pad tail, strictly below every body id
            }
        })
        .collect();
    let mask: Vec<u8> = (0..len).map(|i| u8::from(i <= body_end)).collect();
    assert_eq!(
        ids.iter()
            .enumerate()
            .max_by_key(|(_, &t)| t)
            .map(|(i, _)| i),
        Some(body_end),
        "fixture must put argmax at the interior EOT position"
    );

    let mut gpu = Gpu::init().unwrap_or_else(|e| panic!("Gpu::init: {e:?}"));
    println!("gpu arch={}", gpu.arch);

    let t0 = Instant::now();
    let gw = GpuClipWeights::from_host(&mut gpu, &host)
        .unwrap_or_else(|e| panic!("GpuClipWeights::from_host: {e}"));
    let upload_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Warm run: first use of each kernel shape pays HIP JIT.
    clip_gpu::encode(&mut gpu, &gw, &host, &ids, &mask)
        .unwrap_or_else(|e| panic!("clip_gpu::encode (warm): {e}"));

    let t1 = Instant::now();
    let (got_last, got_pooled) = clip_gpu::encode(&mut gpu, &gw, &host, &ids, &mask)
        .unwrap_or_else(|e| panic!("clip_gpu::encode: {e}"));
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;
    let freed = gw.free_gpu(&mut gpu);
    println!("upload_ms={upload_ms:.1} gpu_encode_ms={gpu_ms:.2} freed_buffers={freed}");

    let t2 = Instant::now();
    let (want_last, want_pooled, _layers) = clip::encode(&host, &ids, &mask);
    let host_ms = t2.elapsed().as_secs_f64() * 1e3;

    let (rel_last, at_last) = max_rel(&want_last, &got_last);
    let (rel_pool, at_pool) = max_rel(&want_pooled, &got_pooled);
    println!(
        "host_encode_ms={host_ms:.2} speedup={:.1}x last_max_rel={rel_last:.3e} \
         pooled_max_rel={rel_pool:.3e} tol={TOL:.0e}",
        host_ms / gpu_ms.max(1e-9)
    );

    let mut bad = false;
    if rel_last > TOL {
        eprintln!(
            "FAIL last_hidden_state: {rel_last:.3e} > {TOL:.0e} (element {at_last}: \
             host {} vs gpu {})",
            want_last[at_last], got_last[at_last]
        );
        bad = true;
    }
    if rel_pool > TOL {
        eprintln!(
            "FAIL pooled: {rel_pool:.3e} > {TOL:.0e} (element {at_pool}: host {} vs gpu {})",
            want_pooled[at_pool], got_pooled[at_pool]
        );
        bad = true;
    }
    if bad {
        std::process::exit(1);
    }
    println!("PASS");
}
