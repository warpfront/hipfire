// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU vs host Qwen3 tap-encoder parity on a REAL FLUX.2 Klein text encoder,
//! and the encode timing that justifies the offload.
//!
//! Compares `qwen3_gpu::encode_taps_host` (f16 weights, f32 accumulate, WMMA
//! GEMM) against the f32 host oracle `qwen3::encode_taps` on identical
//! weights and token ids, per tap, and reports both wall times.
//!
//! ```
//! cargo run --release --features lab --example gpu_qwen3_parity \
//!   -p hipfire-arch-diffusion -- /home/user/comfy-models/klein/FLUX.2-klein-4B \
//!   --tokens 64
//! ```
//!
//! `<pipe_dir>` is a diffusers Klein pipe directory: `text_encoder/` (the
//! Qwen3 text encoder) and `tokenizer/tokenizer.json`. The prompt is the Klein chat
//! template around "a photo of a cat", right-padded to `--tokens` (64 by
//! default) exactly as `encode_klein_prompt` builds a real frame — the
//! production 512-token frame would put the HOST oracle, which is a scalar
//! rayon loop over ~3.7 TFLOP, into the tens of minutes and prove nothing the
//! shorter frame does not. The template itself is only 17 tokens for this
//! prompt, so the default frame is 17 real + 47 pad and its mask exercises
//! the key-padding path end to end. `--mask-all` forces an all-ones mask
//! (every pad attended to, as T5 does) — the two runs bracket both mask
//! states of the same kernel.
//!
//! `--from-host` uploads through `GpuQwen3Weights::from_host` (the eager
//! path) instead of `from_stream` (the production one); both must give the
//! same numbers, since the host RNE f16 conversion matches the device cast.
//!
//! Tolerance: **rel_l2 ≤ 2e-3 per tap**. The GPU path holds weights in f16
//! while the host oracle is f32 throughout, so the two cannot agree to f32
//! epsilon; 2e-3 clears f16 rounding through a 27-layer residual stack with
//! margin while still catching a miswire (a wrong transpose, a missing
//! per-head QK norm, or a RoPE convention flip lands at rel ~O(1)). Exit 1 on
//! failure.

use hipfire_arch_diffusion::klein_prompt::{encode_klein_prompt, KLEIN_PAD_ID};
use hipfire_arch_diffusion::qwen3::{self, Qwen3Plan, Qwen3Weights, KLEIN_TAPS};
use hipfire_arch_diffusion::qwen3_gpu::{self, GpuQwen3Weights};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::time::Instant;

const TOL: f32 = 2e-3;
const PROMPT: &str = "a photo of a cat";

struct Args {
    pipe: PathBuf,
    tokens: usize,
    from_host: bool,
    mask_all: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        pipe: PathBuf::from("/home/user/comfy-models/klein/FLUX.2-klein-4B"),
        tokens: 64,
        from_host: false,
        mask_all: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--from-host" => a.from_host = true,
            "--mask-all" => a.mask_all = true,
            "--tokens" => {
                a.tokens = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| panic!("expected a number after --tokens"));
                i += 1;
            }
            other => a.pipe = PathBuf::from(other),
        }
        i += 1;
    }
    a
}

/// One tap's `[len, d]` slab out of a `[len, taps * d]` buffer.
fn tap_slab(all: &[f32], len: usize, n_taps: usize, ti: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len * d);
    for t in 0..len {
        let base = (t * n_taps + ti) * d;
        out.extend_from_slice(&all[base..base + d]);
    }
    out
}

/// Relative L2 error `||got - want||₂ / ||want||₂` and the max absolute
/// difference. L2 rather than max-rel because a tap is a 2560-wide residual
/// stream whose individual elements straddle zero: one near-zero element with
/// f16-level absolute noise would dominate a max-relative metric while saying
/// nothing about whether the conditioning is right.
fn rel_l2(want: &[f32], got: &[f32]) -> (f32, f32) {
    assert_eq!(want.len(), got.len(), "length mismatch");
    let mut num = 0f64;
    let mut den = 0f64;
    let mut max_abs = 0f32;
    for (w, g) in want.iter().zip(got) {
        let dv = (w - g) as f64;
        num += dv * dv;
        den += (*w as f64) * (*w as f64);
        max_abs = max_abs.max((w - g).abs());
    }
    ((num.sqrt() / den.sqrt().max(1e-12)) as f32, max_abs)
}

fn main() {
    let a = parse_args();
    let text_dir = a.pipe.join("text_encoder");
    let src = SafetensorsSource::open(&text_dir)
        .unwrap_or_else(|e| panic!("open {}: {e:?}", text_dir.display()));
    let plan = Qwen3Plan::detect(&src).unwrap_or_else(|e| panic!("Qwen3Plan::detect: {e}"));
    let c = &plan.config;
    println!(
        "pipe={:?} hidden={} layers={} heads={}/{} head_dim={} inter={} theta={} eps={} vocab={}",
        a.pipe,
        c.hidden,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.intermediate,
        c.rope_theta,
        c.eps,
        c.vocab
    );

    // ── prompt ───────────────────────────────────────────────────────
    let tok_path = a.pipe.join("tokenizer/tokenizer.json");
    let tok = Tokenizer::from_tokenizer_json(&tok_path)
        .unwrap_or_else(|e| panic!("open {}: {e:?}", tok_path.display()))
        .unwrap_or_else(|| panic!("{} missing", tok_path.display()));
    let pad_id = tok
        .special_token_id("<|endoftext|>")
        .unwrap_or(KLEIN_PAD_ID);
    let framed = encode_klein_prompt(&tok, PROMPT, pad_id, a.tokens);
    let len = a.tokens.min(framed.ids.len());
    let ids: Vec<u32> = framed.ids[..len].to_vec();
    let real = framed.mask[..len].iter().filter(|&&m| m == 1).count();
    // `--mask-all` attends over the pads (the T5 convention) instead of
    // masking them; the CPU oracle takes the same mask, so either way this is
    // a parity check of one kernel configuration against its reference.
    let mask: Vec<u8> = if a.mask_all {
        vec![1u8; len]
    } else {
        framed.mask[..len].to_vec()
    };
    println!(
        "prompt={PROMPT:?} frame={len} real_tokens={real} pad_id={pad_id} mask={}",
        if a.mask_all { "all-ones" } else { "klein" }
    );

    // ── host tables ──────────────────────────────────────────────────
    // The GPU encoder needs only the embedding table from the host; the
    // oracle needs everything. In the streaming mode (default) the light set
    // is loaded first so the multi-GB f32 materialisation happens AFTER the
    // device upload and the GPU encode, not alongside them.
    let t0 = Instant::now();
    let light = if a.from_host {
        plan.materialize(&src)
            .unwrap_or_else(|e| panic!("Qwen3Plan::materialize: {e}"))
    } else {
        plan.materialize_light(&src)
            .unwrap_or_else(|e| panic!("Qwen3Plan::materialize_light: {e}"))
    };
    println!(
        "host_load_s={:.1} mode={}",
        t0.elapsed().as_secs_f64(),
        if a.from_host {
            "from_host"
        } else {
            "from_stream"
        }
    );

    let mut gpu = Gpu::init().unwrap_or_else(|e| panic!("Gpu::init: {e:?}"));
    println!("gpu arch={}", gpu.arch);

    let t1 = Instant::now();
    let gw = if a.from_host {
        GpuQwen3Weights::from_host(&mut gpu, &light)
            .unwrap_or_else(|e| panic!("GpuQwen3Weights::from_host: {e}"))
    } else {
        GpuQwen3Weights::from_stream(&mut gpu, &src, &plan)
            .unwrap_or_else(|e| panic!("GpuQwen3Weights::from_stream: {e}"))
    };
    println!("upload_s={:.1}", t1.elapsed().as_secs_f64());

    // Warm run: first use of each kernel shape pays HIP JIT, which is not
    // what this example is reporting.
    let warm = qwen3_gpu::encode_taps(&mut gpu, &gw, &light, &ids, &mask, &KLEIN_TAPS)
        .unwrap_or_else(|e| panic!("qwen3_gpu::encode_taps (warm): {e}"));
    gpu.free_tensor(warm).expect("free warm");

    let t2 = Instant::now();
    let got = qwen3_gpu::encode_taps_host(&mut gpu, &gw, &light, &ids, &mask, &KLEIN_TAPS)
        .unwrap_or_else(|e| panic!("qwen3_gpu::encode_taps_host: {e}"));
    let gpu_ms = t2.elapsed().as_secs_f64() * 1e3;
    let freed = gw.free_gpu(&mut gpu);
    println!("gpu_encode_ms={gpu_ms:.1} freed_buffers={freed}");

    // ── host oracle ──────────────────────────────────────────────────
    let host: Qwen3Weights = if a.from_host {
        light
    } else {
        drop(light);
        let t = Instant::now();
        let w = plan
            .materialize(&src)
            .unwrap_or_else(|e| panic!("Qwen3Plan::materialize: {e}"));
        println!("host_materialize_s={:.1}", t.elapsed().as_secs_f64());
        w
    };
    let t3 = Instant::now();
    let want = qwen3::encode_taps(&host, &ids, &mask, &KLEIN_TAPS);
    let host_ms = t3.elapsed().as_secs_f64() * 1e3;
    println!(
        "host_encode_ms={host_ms:.1} speedup={:.1}x",
        host_ms / gpu_ms.max(1e-9)
    );

    // ── per-tap verdict ──────────────────────────────────────────────
    assert_eq!(want.len(), got.len(), "tap buffer length mismatch");
    let d = plan.config.hidden;
    let n_taps = KLEIN_TAPS.len();
    let mut fail = false;
    for (ti, tap) in KLEIN_TAPS.iter().enumerate() {
        let w = tap_slab(&want, len, n_taps, ti, d);
        let g = tap_slab(&got, len, n_taps, ti, d);
        let (rel, max_abs) = rel_l2(&w, &g);
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        println!("tap {tap}: rel_l2={rel:.3e} max_abs={max_abs:.3e} host_max_abs={scale:.3e}");
        if rel.is_nan() || rel > TOL {
            eprintln!("FAIL: tap {tap} rel_l2 {rel:.3e} > {TOL:.0e}");
            fail = true;
        }
    }
    if fail {
        std::process::exit(1);
    }
    println!("PASS (tol rel_l2 <= {TOL:.0e})");
}
