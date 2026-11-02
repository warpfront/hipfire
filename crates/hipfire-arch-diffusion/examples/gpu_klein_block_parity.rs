// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-weight GPU/CPU parity for the FLUX.2 (Klein) transformer forward,
//! part by part.
//!
//! Opens a Klein diffusers pipe directory's `transformer/`, parses its
//! `config.json`, and runs BOTH forwards on the same real weights and the
//! same deterministic (LCG) latents:
//!
//! * CPU: `flux::forward_parts` over `FluxPlan::materialize` — every manifest
//!   key decoded to f32 host tables (~15.5 GB at Klein 4B geometry).
//! * GPU: `flux_gpu::gpu_forward_parts` over `GpuFluxWeights::from_stream` —
//!   the product upload path, f16 weight tables at the `weight_pitch` row
//!   pitch.
//!
//! Both return the same named intermediates (`temb`, `img_in`, `txt_in`,
//! `double_{b}_{img,txt}`, `single_concat`, `final`), so the FIRST part over
//! tolerance names the convention that diverged rather than leaving a wrong
//! final latent to bisect.
//!
//! Tolerance: **relative L2 < 5e-3 per part on Klein 4B**, overridable with
//! `TOL_L2=<f32>` (Klein 9B needs `8e-3` — see `TOL_L2`'s doc). The GPU keeps
//! `.weight` tables in f16 (the WMMA GEMM operand) and casts K/V to f16 for
//! attention, so the divergence from the all-f32 CPU reference is f16-rounding
//! magnitude; a miswire (wrong M-slice, swapped modulation chunk, missing
//! RoPE id) lands at rel ~O(1) and cannot hide under it.
//!
//! **This gate cannot see a modelling error the two paths SHARE** — it is
//! hipfire against hipfire. The 2026-09-05 text-RoPE bug read 2.830e-3 here
//! while being a whole missing rotation. `gpu_klein_golden_latent` (ComfyUI
//! as the oracle) is what covers that, and it is the run that has to pass
//! before this one's bar is ever raised for a new model.
//!
//! `--refs` re-runs the whole comparison with explicit `img_ids`: the
//! generated grid at time id 0 PLUS a second, smaller reference grid at time
//! id 10 — the FLUX.2 edit layout. That pass is what proves the device id
//! table is read (a forward that ignored `img_ids` would pass the default
//! pass and fail this one).
//!
//! Build + run (GPU required, gpu-lock it):
//! ```
//! flock /tmp/hipfire-gpu.lock \
//!   cargo run --release --features lab --example gpu_klein_block_parity \
//!   -p hipfire-arch-diffusion -- <pipe-dir> [--grid 8x8] [--txt 16] [--refs]
//! ```
//! On the 9B pipe, prefix `TOL_L2=8e-3`.
//! Exits 1 if any part is over tolerance or the two part lists disagree.

use hipfire_arch_diffusion::config::FluxDiffusionConfig;
use hipfire_arch_diffusion::flux::{
    forward_parts, rope_ids_for_grid, FinalAdaLNOrder, FluxForwardInput, FluxPlan, MlpAct,
};
use hipfire_arch_diffusion::flux_gpu::{gpu_forward_parts, install_forward_stream, GpuFluxWeights};
use hipfire_arch_diffusion::pipeline::latent_rel_error;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::PathBuf;

/// Per-part relative-L2 ceiling, **calibrated on the Klein 4B geometry**
/// (hidden 3072, 5 double + 20 single blocks). See the module doc on why f16
/// weights put the floor around 1e-3 and a miswire around 1e0.
///
/// It is a *default*, overridable with `TOL_L2=<f32>`, because the f16 band
/// is not a constant of the code — it scales with how much accumulation the
/// forward does. Klein 9B is hidden 4096 with 8 double + 24 single blocks, so
/// every GEMM has a 33% longer K and there are 32 blocks instead of 25, and
/// the whole per-part curve shifts up together: measured on the same command
/// and the same deterministic inputs, 4B `--refs` reads 2.189e-3 at `final`
/// while 9B reads 6.539e-3, with the ratio climbing monotonically with depth
/// (1.0 at `img_in`, 2.45 at `single_concat`, 2.99 at `final`) and no step
/// change anywhere — the signature of accumulation, not of a miswire, which
/// would put ONE part at rel ~1e0. The 9B runs use `TOL_L2=8e-3`.
///
/// **8e-3 is rounded up from the measured 6.539e-3, not a derived bracket;
/// the 4B→9B ratio 2.99× is unexplained (a naive f16-accumulation scaling
/// model predicts 1.30×).** That is a real weakness and it is the difference
/// between this knob and `gpu_klein_golden_latent`'s bars, every one of which
/// is a measured ComfyUI-against-ComfyUI bracket. The naive model is
/// `sqrt(depth · hidden)`: `sqrt((32·4096)/(25·3072)) = 1.31`, against a
/// measured 2.99 at `final` (and 2.45 at `single_concat`, so the miss is not
/// confined to the last layer). Until someone explains the gap — a third
/// Klein size would settle it — treat 8e-3 as an operational bar backed by
/// the discrimination controls below, NOT as a number derived from the
/// arithmetic.
///
/// Raising this for a model whose ComfyUI golden gate has NOT been run is how
/// a real error gets absorbed. Do it the other way round: the 9B bar was
/// raised only after `gpu_klein_golden_latent` passed on the 9B pipe against
/// an independent oracle at velocity 4.909e-2 (bar 0.15).
const TOL_L2: f32 = 5e-3;

/// `TOL_L2` unless the environment overrides it — **failing closed**. Read
/// once at the top of `main`, before any argument, checkpoint or GPU is
/// touched, and threaded through, so both passes of a `--refs` run report the
/// same bar and a malformed one costs a millisecond rather than a 4-minute
/// forward.
///
/// Fail-closed matters more here than anywhere else in this file, because
/// both failure modes of the obvious
/// `.ok().and_then(|v| v.parse().ok()).unwrap_or(TOL_L2)` produce a *green*
/// result:
///
/// * a typo (`TOL_L2=8e--3`) silently reverts to the default and the run
///   reports PASS against a bar the operator did not choose;
/// * a dropped minus sign — `TOL_L2=8e3` for the documented `TOL_L2=8e-3` —
///   parses cleanly to 8000.0, and every part in the table passes. A silent
///   false PASS on a parity gate is worse than no gate.
///
/// So: unset → the default; unparseable → exit 1 naming the value; outside
/// `(0, 0.05]` → exit 1 naming the range. The 0.05 ceiling is an order above
/// the largest bar this gate has ever legitimately used (8e-3) and an order
/// below `single_concat`'s own magnitude, so it admits every plausible model
/// while refusing a bar that cannot fail. The resolved value is printed once,
/// so the run's own output records which bar it was judged against.
fn tol_l2() -> f32 {
    /// Upper bound on any bar this gate will accept. See `tol_l2`.
    const TOL_L2_CEILING: f32 = 0.05;
    let Ok(raw) = std::env::var("TOL_L2") else {
        println!("tol_l2 = {TOL_L2:.3e} (default)");
        return TOL_L2;
    };
    let Ok(v) = raw.parse::<f32>() else {
        eprintln!("TOL_L2={raw:?} is not a number");
        std::process::exit(1);
    };
    if !(v > 0.0) || v > TOL_L2_CEILING {
        eprintln!("TOL_L2={v} out of range (0, {TOL_L2_CEILING}]");
        std::process::exit(1);
    }
    println!("tol_l2 = {v:.3e} (from env TOL_L2)");
    v
}

/// The reference-image grid `--refs` appends, and the time id it sits at.
const REF_GRID: (usize, usize) = (4, 4);
const REF_TIME_ID: f32 = 10.0;

/// splitmix64-style deterministic noise in [-1, 1). Reproducible across
/// machines and runs, which is what makes a rel_l2 comparable between the
/// CPU and GPU passes (and between sessions).
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

fn noise(seed: u64, n: usize) -> Vec<f32> {
    (0..n).map(|i| lcg(seed, i as u64) * 0.5).collect()
}

fn parse_grid(s: &str) -> (usize, usize) {
    let (h, w) = s
        .split_once(['x', 'X'])
        .unwrap_or_else(|| panic!("--grid wants HxW, got `{s}`"));
    (
        h.parse().unwrap_or_else(|e| panic!("--grid height: {e}")),
        w.parse().unwrap_or_else(|e| panic!("--grid width: {e}")),
    )
}

/// One CPU-vs-GPU pass over every named part. Returns the number of failures
/// and the worst rel_l2 seen.
fn compare(
    label: &str,
    cpu: &[(String, Vec<f32>)],
    gpu: &[(String, Vec<f32>)],
    tol: f32,
) -> (usize, f32) {
    let mut fails = 0usize;
    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    if cpu.len() != gpu.len() {
        eprintln!(
            "{label}: FAIL part count cpu={} gpu={}",
            cpu.len(),
            gpu.len()
        );
        return (1, f32::INFINITY);
    }
    println!("{label}: {:<22} {:>11} {:>11}", "part", "rel_l2", "max_abs");
    for ((cn, cv), (gn, gv)) in cpu.iter().zip(gpu.iter()) {
        if cn != gn {
            eprintln!("{label}: FAIL part name cpu=`{cn}` gpu=`{gn}`");
            fails += 1;
            continue;
        }
        if cv.len() != gv.len() {
            eprintln!("{label}: {cn}: FAIL len cpu={} gpu={}", cv.len(), gv.len());
            fails += 1;
            continue;
        }
        // `latent_rel_error(a, b)` is (max_abs_error / max|b|, relative L2).
        let (_rel_max, rel_l2) = latent_rel_error(gv, cv);
        let max_abs = gv
            .iter()
            .zip(cv.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let ok = rel_l2 <= tol;
        if rel_l2 > worst {
            worst = rel_l2;
            worst_name = cn.clone();
        }
        println!(
            "{label}: {cn:<22} {rel_l2:>11.3e} {max_abs:>11.3e} {}",
            if ok { "ok" } else { "FAIL" }
        );
        if !ok {
            fails += 1;
        }
    }
    println!("{label}: worst rel_l2 {worst:.3e} at `{worst_name}` (tol {tol:.3e})");
    (fails, worst)
}

fn main() {
    // The bar FIRST — before the argument loop, the checkpoint and the GPU.
    // A malformed `TOL_L2` must cost a millisecond, not a weight upload and a
    // 4-minute CPU forward ending in a PASS against a bar nobody chose.
    let tol = tol_l2();

    let mut pipe_dir: Option<PathBuf> = None;
    let mut grid = (8usize, 8usize);
    let mut n_txt = 16usize;
    let mut refs = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--grid" => grid = parse_grid(&args.next().expect("--grid requires a value")),
            "--txt" => {
                n_txt = args
                    .next()
                    .expect("--txt requires a value")
                    .parse()
                    .unwrap_or_else(|e| panic!("--txt: {e}"));
            }
            "--refs" => refs = true,
            _ if pipe_dir.is_none() => pipe_dir = Some(PathBuf::from(a)),
            _ => panic!("unexpected argument: {a}"),
        }
    }
    let pipe_dir = pipe_dir.unwrap_or_else(|| {
        panic!("usage: gpu_klein_block_parity <pipe_dir> [--grid HxW] [--txt N] [--refs]")
    });

    // ── config ───────────────────────────────────────────────────────────
    let src = SafetensorsSource::open(&pipe_dir.join("transformer"))
        .unwrap_or_else(|e| panic!("open transformer: {e}"));
    let cfg_json: serde_json::Value = serde_json::from_str(src.metadata_json())
        .unwrap_or_else(|e| panic!("transformer config.json invalid: {e}"));
    let cfg_json = cfg_json.get("config").cloned().unwrap_or(cfg_json);
    let cfg = FluxDiffusionConfig::from_json(&cfg_json)
        .unwrap_or_else(|e| panic!("transformer cfg: {e}"));
    assert!(
        cfg.is_flux2(),
        "gpu_klein_block_parity is the FLUX.2 (Klein) harness; {:?} is not",
        cfg.family
    );
    let d = cfg.hidden_size;
    let patch_in = cfg.patch_in();
    let n_grid = grid.0 * grid.1;
    let n_ref = if refs { REF_GRID.0 * REF_GRID.1 } else { 0 };
    let n_img = n_grid + n_ref;
    println!(
        "cfg: hidden={d} layers={}/{} heads={} head_dim={} mlp={} patch_in={patch_in} \
         txt_hidden={} axes={:?} theta={} bias={}",
        cfg.num_layers,
        cfg.num_single_layers,
        cfg.num_attention_heads,
        cfg.head_dim,
        cfg.mlp_width(),
        cfg.txt_hidden_dim,
        cfg.axes_dim,
        cfg.theta,
        cfg.bias,
    );
    println!(
        "inputs: grid {}x{} ({n_grid} tok){} + txt {n_txt} tok = {} rows",
        grid.0,
        grid.1,
        if refs {
            format!(
                " + ref grid {}x{} @ t={REF_TIME_ID} ({n_ref} tok)",
                REF_GRID.0, REF_GRID.1
            )
        } else {
            String::new()
        },
        n_txt + n_img,
    );

    // ── deterministic input ──────────────────────────────────────────────
    let img_ids = if refs {
        // The generated grid at time id 0, then a second, smaller reference
        // grid at a NON-ZERO time id — the FLUX.2 edit layout. Only the
        // explicit-ids path can express this; the derived grid cannot.
        let mut ids = rope_ids_for_grid(grid, 0.0);
        ids.extend(rope_ids_for_grid(REF_GRID, REF_TIME_ID));
        Some(ids)
    } else {
        None
    };
    let input = FluxForwardInput {
        timestep: 0.5,
        pooled: Vec::new(), // FLUX.2 has no pooled conditioning path.
        guidance: None,
        txt: noise(0xA11CE, n_txt * cfg.txt_hidden_dim),
        img: noise(0xBEEF, n_img * patch_in),
        grid,
        mlp_act: MlpAct::default(),
        final_order: FinalAdaLNOrder::default(),
        img_ids,
    };

    // ── GPU weights (product streaming path) ─────────────────────────────
    let plan = FluxPlan::detect(&src, &cfg);
    let mut gpu = Gpu::init().expect("GPU init failed");
    // This example does not go through `FluxPipeBundle::ensure_gpu`, so it
    // installs the diffusion stream itself, once, before any upload.
    install_forward_stream(&mut gpu).expect("install forward stream");
    let t0 = std::time::Instant::now();
    let gw = GpuFluxWeights::from_stream(&mut gpu, &src, &plan, &cfg)
        .unwrap_or_else(|e| panic!("GpuFluxWeights::from_stream: {e}"));
    println!(
        "gpu: {} weight tensors uploaded in {:.1}s",
        gw.tensors.len(),
        t0.elapsed().as_secs_f64()
    );

    // ── CPU weights (f32 host tables) ────────────────────────────────────
    let t0 = std::time::Instant::now();
    let host = plan
        .materialize(&src, &cfg)
        .unwrap_or_else(|e| panic!("FluxPlan::materialize: {e}"));
    println!(
        "cpu: {} host tensors materialized in {:.1}s",
        host.tensors.len(),
        t0.elapsed().as_secs_f64()
    );

    // ── the two forwards ─────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let gpu_parts = gpu_forward_parts(&mut gpu, &cfg, &gw, &input)
        .unwrap_or_else(|e| panic!("gpu_forward_parts: {e}"));
    println!("gpu: forward in {:.1}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let cpu_parts = forward_parts(&cfg, &host, &input);
    println!("cpu: forward in {:.1}s", t0.elapsed().as_secs_f64());

    let label = if refs { "refs" } else { "grid" };
    let (fails, worst) = compare(label, &cpu_parts, &gpu_parts, tol);
    let freed = gw.free_gpu(&mut gpu);
    println!("gpu: freed {freed} weight tensors");
    if fails > 0 {
        eprintln!("FAIL: {fails} part(s) over tolerance (worst rel_l2 {worst:.3e})");
        std::process::exit(1);
    }
    println!("PASS: every part within rel_l2 {tol:.3e} (worst {worst:.3e})");
}
