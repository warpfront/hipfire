// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-weight GPU block parity at TRUE FLUX.1-dev
//! geometry, run on the gfx1151 iGPU (HIP device 0, unified-RAM heap).
//!
//! Loads block 0 (one double block + one single block) of the real
//! FLUX.1-dev checkpoint (`/home/user/comfy-models/diffusion_models/
//! flux1-dev.safetensors`, oct 23.8 GB BF16), decodes those tensors to host
//! f32, uploads them to the GPU, runs the GPU double/single block forward on
//! fixture latents, and compares against the CPU reference on the same real
//! weights + latents.
//!
//! This is deliberately BLOCK-SCOPED: it holds only one block's weights on
//! the GPU (~1.4 GB at 3072) plus one host copy, so it fits comfortably in
//! the iGPU's system-RAM-backed heap even though a full-model fp32 lift
//! (~27 GB GPU + 27 GB host) would strain the box's ~50 GB free.
//!
//! Correctness gate: relative tolerance 5e-2. The tuned path keeps the
//! `.weight` tables GPU-resident in f16 (WMMA GEMM operand) and casts K/V to
//! f16 for the DFlash attention, so the output diverges from the all-fp32 CPU
//! reference by f16-rounding magnitude (~1e-2 rel), not the ~1e-6 the old
//! all-fp32 path held. 5e-2 passes f16 noise with margin while still flagging a
//! miswire (wrong transpose / operand order → rel ~O(1)). Any stream/block
//! over tolerance or a length mismatch → exit 1.
//!
//! Build + run (GPU required, gpu-lock it):
//! ```
//! cargo run --release --features lab --example gpu_flux_block_parity \
//!   -p hipfire-arch-diffusion [PATH-TO-flux1-dev.safetensors]
//! ```

use hipfire_arch_diffusion::config::FluxDiffusionConfig;
use hipfire_arch_diffusion::flux::{
    decode_dtype, double_block, single_block, FluxWeights, MlpAct, Tensor,
};
use hipfire_arch_diffusion::flux_gpu::{
    gpu_double_block, gpu_single_block, install_forward_stream, upload_flux_key, GpuFluxWeights,
};
use hipfire_arch_diffusion::manifest::expected_flux_keys;
use rdna_compute::{Gpu, GpuTensor};
use std::collections::HashMap;

const TOL: f32 = 5e-2;

fn dev_cfg() -> FluxDiffusionConfig {
    FluxDiffusionConfig::from_json(&serde_json::json!({
        "hidden_size": 3072,
        "num_layers": 19,
        "num_single_layers": 38,
        "num_attention_heads": 24,
        "head_dim": 128,
        "patch_size": 2,
        "guidance_embed_dim": 256,
        "pooled_projection_dim": 768,
        "axes_dim": [16, 56, 56],
        "latent_channels": 16,
        "txt_hidden_dim": 4096,
    }))
    .expect("dev cfg")
}

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

fn cmp(label: &str, gpu_v: &[f32], cpu_v: &[f32]) -> usize {
    if gpu_v.len() != cpu_v.len() {
        eprintln!("{label}: FAIL len {} != cpu {}", gpu_v.len(), cpu_v.len());
        return 1;
    }
    let max_want = cpu_v.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-9);
    let max_err = gpu_v
        .iter()
        .zip(cpu_v.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let rel = max_err / max_want;
    if rel > TOL {
        eprintln!("{label}: FAIL rel={rel:.3e} (max_err={max_err:.3e})");
        1
    } else {
        println!("{label}: ok rel={rel:.3e} ({} elems)", gpu_v.len());
        0
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/user/comfy-models/diffusion_models/flux1-dev.safetensors".into());
    let cfg = dev_cfg();
    let d = cfg.hidden_size;

    // Block-0 key set (double block 0 + single block 0) from the manifest.
    let keys = expected_flux_keys(&cfg);
    let needed: Vec<_> = keys
        .iter()
        .filter(|k| {
            k.name.starts_with("double_blocks.0.") || k.name.starts_with("single_blocks.0.")
        })
        .collect();
    eprintln!("block-0 manifest keys: {}", needed.len());

    // Read just the block-0 tensors from the real checkpoint via mmap.
    let file = std::fs::File::open(&path).expect("open checkpoint");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap checkpoint") };
    let st = safetensors::SafeTensors::deserialize(&mmap).expect("parse checkpoint");

    let mut host = FluxWeights {
        tensors: HashMap::new(),
    };
    for k in &needed {
        let view = st
            .tensor(&k.name)
            .unwrap_or_else(|e| panic!("missing {}: {e:?}", k.name));
        let data = decode_dtype(view.dtype().to_string().as_str(), view.data())
            .unwrap_or_else(|e| panic!("decode {}: {e}", k.name));
        host.tensors.insert(
            k.name.clone(),
            Tensor {
                data,
                rows: k.rows,
                cols: k.cols,
            },
        );
    }
    eprintln!(
        "host: {} block-0 tensors decoded (BF16→f32)",
        host.tensors.len()
    );

    // Upload block-0 to GPU.
    let mut gpu = Gpu::init().expect("GPU init failed");
    // This example does not go through `FluxPipeBundle::ensure_gpu`, so it
    // installs the diffusion stream itself — once, before any upload. See
    // `flux_gpu::install_forward_stream`: permanent by design, and unrelated
    // to `HIPFIRE_FLUX_F16_ACT`, which selects the activation layout only.
    install_forward_stream(&mut gpu).expect("install forward stream");
    let mut gw = GpuFluxWeights {
        tensors: HashMap::new(),
    };
    for k in &needed {
        let t = host.get(&k.name);
        upload_flux_key(
            &mut gpu,
            &k.name,
            &t.data,
            [k.rows, k.cols],
            &mut gw.tensors,
        )
        .unwrap_or_else(|e| panic!("upload {}: {e}", k.name));
    }
    eprintln!("gpu: {} block-0 tensors uploaded", gw.tensors.len());

    // Fixture latents at true dev geometry, small row counts for CPU speed.
    let grid = (2usize, 2usize);
    let n_img = grid.0 * grid.1; // 4
    let n_txt = 2usize;
    let n_all = n_img + n_txt;
    let img: Vec<f32> = (0..n_img * d).map(|i| lcg(11, i as u64)).collect();
    let txt: Vec<f32> = (0..n_txt * d).map(|i| lcg(22, i as u64)).collect();
    let vec: Vec<f32> = (0..d).map(|i| lcg(33, i as u64) * 0.5).collect();
    let fused: Vec<f32> = {
        let mut v = Vec::new();
        v.extend_from_slice(&txt);
        v.extend_from_slice(&img);
        v
    };

    let g_img = gpu.upload_f32(&img, &[n_img, d]).unwrap();
    let g_txt = gpu.upload_f32(&txt, &[n_txt, d]).unwrap();
    let g_vec = gpu.upload_f32(&vec, &[1, d]).unwrap();
    let g_fused = gpu.upload_f32(&fused, &[n_all, d]).unwrap();

    // CPU block reference on the same real weights.
    let (i_c, t_c) = double_block(&cfg, &host, 0, &img, &txt, &vec, n_img, grid);
    let s_c = single_block(&cfg, &host, 0, &fused, &vec, n_img, grid, MlpAct::GeluTanh);

    // GPU block forward.
    let (i_g, t_g) = gpu_double_block(
        &mut gpu, &cfg, &gw, 0, &g_img, &g_txt, &g_vec, n_img, n_txt, grid,
    )
    .expect("gpu double block 0");
    let s_g = gpu_single_block(
        &mut gpu,
        &cfg,
        &gw,
        0,
        &g_fused,
        &g_vec,
        n_img,
        grid,
        MlpAct::GeluTanh,
    )
    .expect("gpu single block 0");

    // Compare every block stream.
    let mut fails = 0;
    fails += cmp("double_0.img", &i_g, &i_c);
    fails += cmp("double_0.txt", &t_g, &t_c);
    fails += cmp("single_0", &s_g, &s_c);

    let _ = gw.free_gpu(&mut gpu);
    if fails > 0 {
        eprintln!("FAIL: {fails}/3 block streams diverged vs CPU on real weights");
        std::process::exit(1);
    }
    println!(
        "PASS: GPU block parity at real FLUX.1-dev geometry ({})",
        path
    );
}
