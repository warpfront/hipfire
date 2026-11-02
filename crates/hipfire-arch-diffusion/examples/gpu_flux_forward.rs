// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-vs-CPU parity for the assembled FLUX.1 MMDiT forward.
//!
//! Builds synthetic weights at a small lab geometry, uploads them via
//! `GpuFluxWeights`, runs the full forward on the GPU (`gpu_forward_parts`)
//! and on the CPU reference (`flux::forward_parts`) with identical inputs, and
//! compares every named intermediate part-for-part: `vec`, `img_in`,
//! `txt_in`, each `double_{b}_{img,txt}`, `single_concat`, and `final`.
//!
//! Correctness gate: per-part max |gpu - cpu| relative to that part's max
//! magnitude, tolerated at 2e-4. The primitives individually matched at
//! ~1e-6; the forward accumulates rounding across blocks, so the tolerance is
//! looser but still far below the bf16 1e-3 the block-parity gate will use.
//! The double-block 2D RoPE path is exercised via a nonzero image grid.
//! Any part over tolerance, or a name/len mismatch, → exit 1.
//!
//! Build + run (GPU required, gpu-lock it):
//! ```
//! cargo run --release --features lab --example gpu_flux_forward -p hipfire-arch-diffusion
//! ```

use hipfire_arch_diffusion::config::FluxDiffusionConfig;
use hipfire_arch_diffusion::flux::{
    forward_parts, FinalAdaLNOrder, FluxForwardInput, FluxWeights, MlpAct,
};
use hipfire_arch_diffusion::flux_gpu::{gpu_forward_parts, install_forward_stream, GpuFluxWeights};
use rdna_compute::Gpu;

/// Per-part relative tolerance. 2e-4 was right while both sides were all-fp32
/// (the primitives matched at ~1e-6 and only cross-block accumulation moved
/// the number). The tuned GPU path now holds `.weight` tables in f16 for the
/// WMMA GEMM, so it rounds at f16 mantissa magnitude — 2^-11 ≈ 4.9e-4 per
/// rounding — and this geometry measures 2.7e-4 to 4.3e-4 against the all-fp32
/// CPU reference. 5e-3 clears that by ~10× while still catching what this gate
/// exists to catch: a miswired transpose or operand order gives rel ~O(1), and
/// a buffer read before it is written gives the same. Same reasoning as the
/// 5e-2 in `gpu_flux_block_parity`, tighter because this geometry is smaller.
const TOL: f32 = 5e-3;

fn lcg(seed: u64, i: u64) -> f32 {
    let mut x = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let frac = (x >> 11) as f64 / (1u64 << 53) as f64;
    (frac * 2.0 - 1.0) as f32
}

fn main() {
    // Lab geometry, chosen so the GPU path this gate covers is the one the
    // real model takes. Every K here is a multiple of 16, which the WMMA GEMM
    // requires (the older `hidden_size: 32` / `pooled_projection_dim: 8` shape
    // gave `vector_in` a K of 8 and made this example abort since the GEMM was
    // wired in). `hidden_size: 64` also makes the block K values multiples of
    // 64, so the LDS-staged GEMM — the route the real forward uses — is the one
    // under test, while `img_in` (K=16) still exercises the 16-step fallback.
    // `axes_dim` sums to `head_dim`, as in the real config.
    let cfg = FluxDiffusionConfig::from_json(&serde_json::json!({
        "hidden_size": 64,
        "num_layers": 2,
        "num_single_layers": 2,
        "num_attention_heads": 4,
        "head_dim": 16,
        "patch_size": 2,
        "guidance_embed_dim": 8,
        "pooled_projection_dim": 16,
        "axes_dim": [4, 6, 6],
        "latent_channels": 4,
        "txt_hidden_dim": 16,
    }))
    .expect("cfg");

    let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels; // 16
    let grid = (2usize, 2usize);
    let n_img = grid.0 * grid.1; // 4
    let n_txt = 4usize;
    let pooled = (0..cfg.pooled_projection_dim)
        .map(|i| lcg(11, i as u64))
        .collect::<Vec<_>>();
    let txt = (0..n_txt * cfg.txt_hidden_dim)
        .map(|i| lcg(22, i as u64))
        .collect::<Vec<_>>();
    let img = (0..n_img * patch_in)
        .map(|i| lcg(33, i as u64))
        .collect::<Vec<_>>();

    let input = FluxForwardInput {
        timestep: 0.7,
        pooled,
        guidance: Some(3.5),
        txt,
        img,
        grid,
        mlp_act: MlpAct::GeluTanh,
        final_order: FinalAdaLNOrder::ShiftScale,
        img_ids: None,
    };

    // CPU reference.
    let cpu = forward_parts(&cfg, &FluxWeights::synthetic(&cfg), &input);
    eprintln!("cpu forward: {} parts", cpu.len());

    // GPU forward (upload synthetic weights).
    let mut gpu = Gpu::init().expect("GPU init failed");
    // This example does not go through `FluxPipeBundle::ensure_gpu`, so it
    // installs the diffusion stream itself — once, before any upload. See
    // `flux_gpu::install_forward_stream`: permanent by design, and unrelated
    // to `HIPFIRE_FLUX_F16_ACT`, which selects the activation layout only.
    install_forward_stream(&mut gpu).expect("install forward stream");
    let host = FluxWeights::synthetic(&cfg);
    let gw = GpuFluxWeights::from_host(&mut gpu, &host, &cfg).expect("upload");
    let gpu_parts = gpu_forward_parts(&mut gpu, &cfg, &gw, &input).expect("gpu forward");
    eprintln!("gpu forward: {} parts", gpu_parts.len());

    // `gpu_forward_parts` runs embedders + every block + the final
    // head in one call, so — unlike the block gate, which only
    // exercises one double + one single block — this is where `embed.*` and
    // `final.*` (the HIPFIRE_PROFILE per-kernel-family attribution) can be
    // proven non-zero without a checkpoint.
    if let Some(table) = hipfire_arch_diffusion::flux_gpu::take_step_profile() {
        println!("== HIPFIRE_PROFILE: gpu_forward_parts per-family (lab geometry) ==");
        for (family, us) in &table {
            println!("  {family:<20} {:8.3} ms", us / 1000.0);
        }
    }

    if gpu_parts.len() != cpu.len() {
        eprintln!(
            "FAIL: part count differ (gpu {} vs cpu {})",
            gpu_parts.len(),
            cpu.len()
        );
        std::process::exit(1);
    }

    let mut fails = 0usize;
    for ((gn, gv), (cn, cv)) in gpu_parts.iter().zip(cpu.iter()) {
        if gn != cn {
            eprintln!("FAIL: name order mismatch: gpu {gn} vs cpu {cn}");
            fails += 1;
            continue;
        }
        if gv.len() != cv.len() {
            eprintln!("{gn}: FAIL len {} != cpu {}", gv.len(), cv.len());
            fails += 1;
            continue;
        }
        let max_want = cv.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-9);
        let max_err = gv
            .iter()
            .zip(cv.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let rel = max_err / max_want;
        if rel > TOL {
            eprintln!("{gn}: FAIL rel={rel:.3e} (max_err={max_err:.3e})");
            fails += 1;
        } else {
            println!("{gn}: ok rel={rel:.3e} ({} elems)", gv.len());
        }
    }

    // Bit-identity anchor. The padded weight pitch (`HIPFIRE_FLUX_WPAD`,
    // Task 9) only moves where a weight row sits in DRAM: the kernel sums the
    // same K elements in the same order, so the forward must be BYTE-identical
    // between the padded and the packed layout. The `rel=` lines above cannot
    // show that — they are printed to three digits — so hash the raw f32 BIT
    // patterns of every part. Per part as well as overall, so a divergence is
    // localised to a stage rather than just detected.
    let fnv = |v: &[f32]| {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    };
    let mut all: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, v) in &gpu_parts {
        let h = fnv(v);
        println!("fnv1a {name:<20} {h:#018x}");
        all ^= h;
        all = all.wrapping_mul(0x0000_0100_0000_01b3);
    }
    println!("fnv1a {:<20} {all:#018x}", "ALL-PARTS");

    let freed = gw.free_gpu(&mut gpu);
    let _ = freed;

    if fails > 0 {
        eprintln!("FAIL: {fails}/{} parts diverged", cpu.len());
        std::process::exit(1);
    }
    println!(
        "PASS: GPU MMDiT forward matches CPU reference ({})",
        cpu.len()
    );
}
