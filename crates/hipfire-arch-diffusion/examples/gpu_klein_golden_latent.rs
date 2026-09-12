// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! **FLUX.2 Klein end-to-end golden gate**: hipfire's whole denoise loop
//! against ComfyUI's, from byte-identical initial noise.
//!
//! The Klein twin of `gpu_flux_golden_latent`, and it exists for the same
//! reason: `gpu_klein_block_parity` stops at the last block, so it cannot see
//! a wrong final adaLN order, a wrong sigma schedule, a wrong latent
//! convention or a wrong conditioning frame — all of which are upstream or
//! downstream of the blocks and all of which move the IMAGE by O(1) while
//! leaving a per-block tolerance green. What this compares is what the
//! transformer produces at the END of a forward, with the VAE excluded: the
//! step-1 velocity (the hard model bar — see below), the step-1 latent, and
//! the final latent after the whole 4-step loop.
//!
//! Both sides start from the SAME noise, so the comparison is a difference of
//! implementations rather than of random draws. The same seed INTEGER would
//! not do — `scheduler::seeded_gaussian` and torch's `randn` are different
//! generators — so ComfyUI's noise is taken OUT and fed IN here.
//!
//! ## Fixture
//!
//! `<fixture-dir>` holds:
//!   * `init.latent`   — ComfyUI's own seed-7 noise, its `.latent` format.
//!   * `golden.latent` — ComfyUI's final latent from that same noise.
//!   * `step1.latent`  — ComfyUI's x₁, from the same graph with
//!     `SplitSigmas(step=1)` feeding the sampler its first two sigmas. It
//!     carries the `1/(1-σ₁)` factor `inverse_noise_scaling` applies on the
//!     way out; the gate undoes it. This file is what the gate ASSERTS on —
//!     see below.
//!   * `golden.png`    — ComfyUI's decode of `golden.latent`.
//!   * `prompt.txt`    — the prompt, exact bytes (md5 in `meta.json`).
//!   * `meta.json`     — sizes, steps, sampler, schedule, provenance, and
//!     `velocity_regression_bar` (REQUIRED): the per-fixture velocity
//!     regression tripwire, 2× the last measured value.
//!
//! ## The two latent conventions this gate had to settle
//!
//! Both were read out of ComfyUI 0.31's source rather than guessed, because
//! either one guessed wrongly produces a plausible number instead of an
//! error.
//!
//! **1. Channel order.** ComfyUI stores a FLUX.2 latent as
//! `[1, 128, H/16, W/16]` — already 2×2-patched, unlike FLUX.1's
//! `[1, 16, H/8, W/8]`. `comfy/ldm/models/autoencoder.py` (`AutoencoderKL`,
//! `batch_norm_latent=True`) packs it with
//! `rearrange("... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)`, so
//! ComfyUI's channel index is `c*4 + ph*2 + pw`. hipfire's
//! `scheduler::pack_latents` writes the feature index `ch*4 + oh*2 + ow`.
//! The two orders are the SAME, so the conversion is a pure transpose:
//! `packed[(y*J + x)*128 + f] = comfy[f*I*J + y*J + x]`. `PACK_ORDER=alt`
//! runs the competing hypothesis (`(ph*2+pw)*32 + c`) so the choice is
//! measured, not asserted — it lands at rel_l2 ~1 on the latent and turns the
//! decode into noise.
//!
//! **2. Normalization.** The same `AutoencoderKL.encode` applies
//! `F.batch_norm` with the checkpoint's `bn.running_mean` / `bn.running_var`
//! INSIDE the VAE, after the rearrange, and `comfy/latent_formats.py`'s
//! `Flux2` overrides no `scale_factor`, so `process_latent_in`/`_out` are the
//! identity. A ComfyUI FLUX.2 `.latent` is therefore ALREADY NORMALIZED —
//! the same space hipfire's denoise loop runs in. Do NOT put
//! `scheduler::normalize_packed` on the load path; that would apply the
//! BatchNorm a second time. The init-noise assert below is what catches it:
//! a doubly-normalized unit noise does not read mean ~0 / std ~1.
//!
//! FLUX.1 is the opposite on both counts (raw VAE space, scalar
//! scale/shift), which is exactly why this is a separate example rather than
//! a flag on the FLUX.1 one.
//!
//! ## Schedule
//!
//! ComfyUI's `Flux2Scheduler` (`comfy_extras/nodes_flux.py`) is
//! `scheduler::empirical_mu` plus the exponential shift, constant for
//! constant: same `a1/b1/a2/b2`, same `image_seq_len > 4300` branch, and
//! `linspace(1, 0, steps+1)` shifted whole equals hipfire's
//! `linspace_sigmas(steps)` shifted then terminated with 0. `image_seq_len`
//! is `round(W*H/256)` on ComfyUI's side and `n_img` on hipfire's, which are
//! the same 4096 at 1024². The golden must therefore be captured with
//! `Flux2Scheduler` + `SamplerCustomAdvanced`, NOT with plain
//! `KSampler scheduler=simple` — `simple` is an unshifted schedule and would
//! make this gate measure the scheduler instead of the model.
//!
//! ## What this gate asserts on, and why it is NOT the final latent
//!
//! The FLUX.1 gate compares the FINAL latent at `rel_l2 ≤ 0.0935`. That bar
//! is unachievable here, and not because of anything hipfire does. Klein at 4
//! steps is a distilled schedule whose last Euler step alone carries σ from
//! 0.767 to 0, so the image is decided by very few evaluations of the
//! velocity field and a small difference in that field is not averaged away —
//! it is integrated. Measured, ComfyUI against ITSELF with one input changed
//! (the bracket is recorded in the 9b fixture's meta.json):
//!
//!   * re-running the identical graph — byte-identical latent, so the
//!     reference is deterministic and every number below is signal;
//!   * moving the sigma list by one or two float32 ULP — final `rel_l2`
//!     **2.9e-2**;
//!   * loading the same weights as `fp8_e4m3fn` instead of bf16 — final
//!     `rel_l2` **4.2e-1**.
//!
//! A reference that moves 0.42 under a weight-dtype change cannot be matched
//! to 0.0935 by an implementation that necessarily differs by at least a
//! dtype. So the FLUX.1 bar is not applied to the final latent here. Every
//! bound below is instead the measured **fp8 bracket** for its quantity — the
//! spread ComfyUI shows against ITSELF when only the weight dtype changes,
//! which is the smallest difference an independent implementation can have:
//!
//!   1. **the golden decode** (`DECODE_TOL`, 0.05) — hipfire's VAE on
//!      ComfyUI's own golden latent against ComfyUI's own PNG. No
//!      transformer in it, so it isolates the two latent conventions above.
//!   2. **the step-1 VELOCITY** (`VELOCITY_TOL`, 0.15 = the fp8 bracket) —
//!      THE model gate. One forward with nothing fed back and nothing
//!      diluting it: the whole Qwen3 conditioning tower, the MMDiT trunk, the
//!      final adaLN head, the 4-axis RoPE ids and the σ→t convention,
//!      compared against the velocity ComfyUI's own `x₁` implies
//!      (`v_ref = (x₁ − x₀)/Δσ`). This quantity carries a SECOND bar, the
//!      fixture's `velocity_regression_bar` in `meta.json` (2× the last
//!      measured value): the 0.15 ceiling has 3–11× headroom over every
//!      measured number, so on its own it cannot see a 3× regression that is
//!      still "correct". Both bars fail the gate.
//!   3. **the latent after ONE step** (`TOL`, 0.0935) — the diluted view of
//!      the same forward, kept because it is directly comparable with the
//!      FLUX.1 gate's number.
//!   4. **the final latent** (`FINAL_TOL`, 0.42 = the fp8 bracket) — fires
//!      when four steps accumulate more deviation than a whole weight-dtype
//!      change does.
//!
//! `step1.latent` is therefore REQUIRED, not optional: without it neither the
//! velocity nor the one-step bar exists, and the final latent alone cannot
//! discriminate at 4 steps.
//!
//! **Why the velocity and not just the latent.** `x₁ = x₀ + (σ₁-σ₀)·v` with
//! `x₀` byte-identical on both sides, so `Δx₁ = (σ₁-σ₀)·Δv` and at
//! σ₁ = 0.967 only 3.3% of `x₁` is the model's output — the latent bar is 30×
//! weaker than it looks. That is not academic: before the text-RoPE fix
//! (2026-09-05 — Klein rotates its TEXT tokens by token index on axis 3,
//! ComfyUI `txt_ids_dims = [3]`, while hipfire left them unrotated) the
//! step-1 latent read a comfortable **1.635e-2** against this 0.0935 bar
//! while the velocity was **4.385e-1**, three times the fp8 bracket. The
//! diluted bar could not see a whole missing rotation. With the fix both
//! collapse (1.159e-3 / 3.108e-2) and the final latent goes 0.841 → 0.220.
//! One more thing the step-1 latent does NOT catch: a WRONG channel order
//! still passes it (measured: 4.4e-2), because both sides are then permuted
//! the same way and `x₀` dominates. The golden-decode check is what catches
//! that, which is why it is a separate hard gate.
//!
//! ## The edit path (`--image`, `--ref-latent`)
//!
//! With `--image <ref.png>` the same four bars run over the FLUX.2 Klein
//! EDIT path: the reference is decoded by `refimg::load_reference`,
//! VAE-encoded on the device, packed, normalized and appended to the image
//! stream at RoPE time id `10*(i+1)` by the product's own
//! `pipeline::build_ref_tokens` — the example calls that function rather than
//! re-deriving it, so the gate cannot agree with a copy while the shipped
//! path drifts. ComfyUI's side is settled from source the same way the
//! txt2img conventions were (`comfy/ldm/flux/model.py::Flux._forward`):
//!
//!   * reference tokens are `torch.cat([img, kontext], dim=1)` — AFTER the
//!     generated tokens, which is hipfire's order;
//!   * their ids come from `process_img(ref, index=index, ...)` with
//!     `index += params.ref_index_scale` per reference and `ref_index_scale
//!     = 10.0` for `image_model == "flux2"` (`comfy/model_detection.py`), so
//!     reference `i` sits at axis-0 time `10*(i+1)` with axes 1/2 its own
//!     `h`/`w` grid and axis 3 zero — hipfire's `rope_ids_for_grid(grid,
//!     10.0*(i+1))`;
//!   * `model_base.Flux.extra_conds` runs each reference latent through
//!     `process_latent_in`, which for `latent_formats.Flux2` is the identity,
//!     so what `SaveLatent` writes is what the trunk sees.
//!
//! `--ref-latent <ref.latent>` is the cheap check that runs FIRST, before a
//! denoise is spent: hipfire's packed+normalized reference tokens against a
//! ComfyUI `LoadImage → VAEEncode → SaveLatent` of the same PNG, under the
//! same pure transpose as every other latent here. No transformer and no
//! schedule are in it — it isolates the reference VAE encode and the packing
//! — so its bound is neither the fp8 bracket nor
//! `gpu_klein_vae_parity`'s intra-hipfire 5e-3 but the measured **bf16
//! bracket** for this VAE (`REF_TOL`, see the constant). A failure here
//! means every number below it is about the wrong input.
//!
//! **The reference resize is deliberately OUT of the comparison.** ComfyUI's
//! own Klein edit template (`image_flux2_klein_image_edit_4b_distilled`) puts
//! `ImageScaleToTotalPixels(nearest-exact, megapixels=1.0,
//! resolution_steps=1)` in front of `VAEEncode` and then takes the output
//! size from `GetImageSize` of the SCALED image. That rule is not hipfire's:
//! `refimg::target_size` caps AREA at 1024² and never upscales, then floors
//! each side to a multiple of 16, whereas ComfyUI scales to exactly 1 MP in
//! both directions and rounds to `resolution_steps`. A reference that is
//! already a fixed point of BOTH rules — 768×512 is: under the cap, a
//! multiple of 16, and reached by ComfyUI only if its scale node is absent —
//! removes the resize from the loop, which is why the capture graph wires
//! `LoadImage → VAEEncode` directly and the fixture's `ref.png` is 768×512.
//! `--make-ref` and `--dump-snapped` exist to exercise the snap rule
//! separately (see `meta.json`'s `note_resize`).
//!
//! Build + run (GPU required, gpu-lock it):
//! ```text
//! flock /tmp/hipfire-gpu.lock cargo run --release --features lab \
//!   -p hipfire-arch-diffusion --example gpu_klein_golden_latent -- \
//!   <pipe-dir> <fixture-dir> [--image ref.png] [--ref-latent ref.latent] \
//!   [--ref-only] [--dump-snapped snapped.png]
//!
//! # resize helper, no GPU and no pipe needed:
//! cargo run --release --features lab -p hipfire-arch-diffusion \
//!   --example gpu_klein_golden_latent -- --make-ref src.png 768x512 ref.png
//! ```
//! Env:
//!   VELOCITY_TOL=<f32> pass/fail bound on the step-1 VELOCITY (default 0.15,
//!                     the ComfyUI fp8-vs-bf16 bracket) — the model gate
//!   TOL=<f32>         pass/fail bound on the ONE-STEP latent (default 0.0935)
//!   FINAL_TOL=<f32>   bound on the final latent (default 0.42, the fp8 bracket)
//!   PACK_ORDER=alt    use the competing channel order (diagnostic)
//!   DUMP_OURS=<path>  write our final latent as a ComfyUI `.latent`
//!   DUMP_PNG=<path>   write our decoded PNG

use hipfire_arch_diffusion::pipeline::{
    build_ref_tokens, comfy_latent_bytes, condition_prompt, generate_txt2img_steps_gpu,
    latent_rel_error, load_pipe, read_comfy_latent, vae_upscale, FluxPipeBundle, RefTokens,
    Txt2ImgInput,
};
use hipfire_arch_diffusion::refimg::{self, RefImage};
use hipfire_arch_diffusion::scheduler;
use hipfire_arch_diffusion::vae::LatentNorm;
use hipfire_arch_diffusion::vae_gpu;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

/// Pixel-space bound on decoding ComfyUI's own golden latent through
/// hipfire's VAE. Covers the u8 quantization of `golden.png` (~1/255
/// relative on its own) plus the f16 GEMMs in the GPU decoder, which
/// `gpu_klein_vae_parity` measures at rel_l2 5e-3 against the CPU reference.
/// A wrong channel order or a missing normalization lands one to two orders
/// above it.
const DECODE_TOL: f32 = 0.05;

/// The undiluted step-1 VELOCITY bracket, measured ComfyUI-against-ComfyUI:
/// loading the SAME weights as `fp8_e4m3fn` instead of bf16 moves the step-1
/// velocity by rel_l2 ≈ 0.147 (derived from the fp8 run's step-1 latent,
/// recorded in the 9b fixture's meta.json). A pure weight-dtype
/// change is the smallest difference an independent implementation can have,
/// so this is the floor of what the quantity can resolve — and hipfire, at
/// f16 weights with its own attention and GEMM kernels, must land inside it.
/// Rounded up from 0.147 to a stable 0.15.
///
/// **This is a CEILING, not a target.** Every measured value sits 3-11x under
/// it (4B txt2img 3.108e-2, 9B 4.909e-2, 4B edit 1.344e-2), so on its own it
/// would let a real 3x regression land green. The fixture's
/// `velocity_regression_bar` (see [`velocity_regression_bar`]) is the second,
/// per-fixture bar that closes that gap.
const VELOCITY_FP8_BRACKET: f32 = 0.15;

/// The per-fixture VELOCITY REGRESSION TRIPWIRE, read from the fixture's
/// `meta.json` field `velocity_regression_bar`.
///
/// Two bars, two different questions. `VELOCITY_TOL` above asks "is this
/// implementation CORRECT" and its answer is a bracket derived from ComfyUI
/// against itself — it must stay at 0.15 whatever hipfire measures, or it
/// stops being a bracket. This one asks "did this commit make the forward
/// WORSE than the last one did", and its answer is 2x whatever the fixture
/// last measured. It is per-fixture because the measured values differ by 4x
/// across the three (edit 1.34e-2 vs 9B 4.91e-2) and one shared number would
/// be as loose as the ceiling for the tightest of them.
///
/// It is REQUIRED, not optional: a fixture with no bar would silently gate on
/// the ceiling alone, which is the state this exists to end. Failing it is an
/// exit 1 like any other bar — a tripwire nobody has to act on is not a gate
/// — and the fix for a legitimate numerical change is to re-measure and move
/// the number in `meta.json`, in the same commit, with the new sha in
/// `note_velocity_regression_bar`.
fn velocity_regression_bar(meta: &serde_json::Value) -> f32 {
    let v = meta["velocity_regression_bar"].as_f64().unwrap_or_else(|| {
        panic!(
            "fixture meta.json: `velocity_regression_bar` (a number) is required — it is \
             the per-fixture regression tripwire under VELOCITY_TOL, set to 2x the \
             measured velocity, with the provenance in `note_velocity_regression_bar`"
        )
    }) as f32;
    assert!(
        v > 0.0 && v <= 1.0,
        "meta.velocity_regression_bar = {v} out of range (0, 1] — it is a relative-L2 \
         bar, and two independent samples of the same model sit near sqrt(2)"
    );
    v
}

/// The same bracket on the FINAL latent after 4 steps: the fp8 control lands
/// at rel_l2 0.4158, so 0.42. This is a real bound, not the √2 catastrophe
/// bound the gate used before the text-RoPE fix — anything above it is a
/// larger deviation than a whole weight-dtype change accumulated over the
/// distilled schedule.
const FINAL_FP8_BRACKET: f32 = 0.42;

/// Bound on hipfire's packed+normalized REFERENCE tokens against ComfyUI's
/// `VAEEncode` of the same PNG — the **bf16 bracket** for this VAE.
///
/// The obvious number to reach for is `gpu_klein_vae_parity`'s 5e-3, and it
/// is the WRONG CLASS: that is hipfire's f16 GPU decoder against hipfire's
/// own f32 CPU reference, an INTRA-hipfire dtype spread. This quantity is
/// hipfire's f32 encoder against ComfyUI's, and ComfyUI runs this VAE in
/// **bfloat16** — `comfy/sd.py`'s `VAE.working_dtypes` defaults to
/// `[bfloat16, float32]` and the `batch_norm_latent` branch does not
/// override it, so `model_management.vae_dtype` picks bf16 on any device
/// that supports it, and `AutoencoderKL.encode` even casts the BatchNorm's
/// `running_mean`/`running_var` to `z.dtype` before applying them. bf16
/// carries 8 mantissa bits (2⁻⁹ ≈ 2e-3 per value) and the encoder is ~30
/// convolutions deep, so ~1e-2 at the output is the floor, not a defect.
///
/// Measured, three independent times on this VAE across both directions:
/// the reference encode reads **9.712e-3**, and the same VAE compared the
/// other way — hipfire decoding ComfyUI's own latent against ComfyUI's own
/// PNG — reads **7.635e-3** on this fixture and **8.435e-3** on the txt2img
/// one. The bound is 2× the largest of those. It still discriminates by two
/// orders: `PACK_ORDER=alt` takes this same number to **1.400**.
///
/// The residual is broadband, not a convention: the gate prints a
/// border-vs-interior split of the token grid (measured 8.958e-3 border vs
/// 9.797e-3 interior), and a padding, resample-alignment or downsample
/// off-by-one lands on the border while a dtype difference does not.
const REF_TOL: f32 = 0.02;

/// How a ComfyUI FLUX.2 latent's 128 channels map onto hipfire's packed
/// feature index.
#[derive(Clone, Copy, PartialEq, Debug)]
enum PackOrder {
    /// `c*4 + ph*2 + pw` — what ComfyUI's `rearrange` and hipfire's
    /// `pack_latents` both do. The truth; see the module doc.
    ChannelMajor,
    /// `(ph*2 + pw)*C + c` — the plausible alternative, kept so the choice is
    /// a measurement rather than a claim.
    PatchMajor,
}

/// Read a tolerance knob from the environment, **failing closed**.
///
/// The obvious `.ok().and_then(|v| v.parse().ok()).unwrap_or(default)` is a
/// trap on a gate: a typo silently reverts to the default and the run reports
/// a PASS against a bar the operator did not choose, with nothing on stdout
/// saying so. Worse in the other direction — a dropped minus sign turns
/// `1e-1` into `1e1` and every bar passes. So:
///
/// * unset          → `default`, printed as `(default)`;
/// * unparseable    → **exit 1** naming the variable and the value;
/// * outside `(0, ceiling]` → **exit 1** naming the range.
///
/// `ceiling` is 1.0 for every knob here: these are all relative-L2
/// tolerances, and two independent samples of the same model sit near √2, so
/// a bar above 1.0 cannot fail anything that is not already catastrophic.
/// The resolved value is printed once so the run's own output records which
/// bar it was judged against.
fn env_f32(name: &str, default: f32, ceiling: f32) -> f32 {
    let Ok(raw) = std::env::var(name) else {
        println!("{name} = {default:.4e} (default)");
        return default;
    };
    let Ok(v) = raw.parse::<f32>() else {
        eprintln!("{name}={raw:?} is not a number");
        std::process::exit(1);
    };
    if !(v > 0.0) || v > ceiling {
        eprintln!("{name}={v} out of range (0, {ceiling}]");
        std::process::exit(1);
    }
    println!("{name} = {v:.4e} (from env {name})");
    v
}

fn mean_std(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// ComfyUI `[1, C4, I, J]` → hipfire packed `[I*J, C4]`.
///
/// Row-major tokens (`t = y*J + x`) either way; only the feature index
/// differs between the two hypotheses.
fn comfy_to_packed(v: &[f32], c4: usize, i: usize, j: usize, order: PackOrder) -> Vec<f32> {
    let hw = i * j;
    assert_eq!(v.len(), c4 * hw, "comfy latent is not [{c4}, {i}, {j}]");
    let mut out = vec![0f32; hw * c4];
    for f in 0..c4 {
        let dst_f = match order {
            PackOrder::ChannelMajor => f,
            PackOrder::PatchMajor => (f % 4) * (c4 / 4) + f / 4,
        };
        for y in 0..i {
            for x in 0..j {
                out[(y * j + x) * c4 + dst_f] = v[f * hw + y * j + x];
            }
        }
    }
    out
}

/// The exact inverse of [`comfy_to_packed`], so a dumped latent reloads into
/// ComfyUI (and into this gate) as the thing it was.
fn packed_to_comfy(p: &[f32], c4: usize, i: usize, j: usize, order: PackOrder) -> Vec<f32> {
    let hw = i * j;
    assert_eq!(p.len(), hw * c4, "packed latent is not [{hw}, {c4}]");
    let mut out = vec![0f32; c4 * hw];
    for f in 0..c4 {
        let src_f = match order {
            PackOrder::ChannelMajor => f,
            PackOrder::PatchMajor => (f % 4) * (c4 / 4) + f / 4,
        };
        for y in 0..i {
            for x in 0..j {
                out[f * hw + y * j + x] = p[(y * j + x) * c4 + src_f];
            }
        }
    }
    out
}

/// Decode a PACKED, normalized latent through the resident GPU VAE — the
/// same three steps `denoise_and_decode` takes, so this gate cannot decode
/// the golden by a different route than it decodes ours.
fn decode_packed(
    b: &FluxPipeBundle,
    gpu: &mut Gpu,
    packed: &[f32],
    n_img: usize,
    lh: usize,
    lw: usize,
) -> Result<Vec<f32>, String> {
    let cfg = &b.transformer_cfg;
    let packed_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
    let denormed = scheduler::denormalize_packed(packed, n_img, packed_in, &b.meta.latent_norm);
    let scaled = scheduler::unpack_latents(&denormed, n_img, packed_in / 4, lh, lw);
    let gv = b
        .gpu_vae
        .as_ref()
        .ok_or("golden decode: the VAE decoder is not resident (ensure_gpu)")?;
    Ok(vae_gpu::gpu_decode(gpu, gv, &scaled, lh, lw)?.0)
}

/// `WxH` → `(w, h)`.
fn parse_size(s: &str) -> (u32, u32) {
    let (w, h) = s
        .split_once(['x', 'X'])
        .unwrap_or_else(|| panic!("size must be WxH, got {s:?}"));
    (
        w.parse()
            .unwrap_or_else(|e| panic!("size width {w:?}: {e}")),
        h.parse()
            .unwrap_or_else(|e| panic!("size height {h:?}: {e}")),
    )
}

/// `--make-ref <src> <WxH> <out>`: Lanczos3-resize a PNG and write it.
///
/// The fixture's `ref.png` has to come from somewhere, and it has to come
/// from the SAME filter `refimg::prepare_reference` uses — otherwise the
/// reference-latent check would be comparing hipfire's resampler against
/// whatever produced the file, on top of the encode it is meant to isolate.
/// Runs before the pipe is loaded, so it needs neither weights nor a GPU.
fn make_ref(src: &Path, size: &str, out: &Path) {
    let (w, h) = parse_size(size);
    let img = image::open(src)
        .unwrap_or_else(|e| panic!("--make-ref {}: {e}", src.display()))
        .to_rgb8();
    let resized = image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3);
    resized
        .save(out)
        .unwrap_or_else(|e| panic!("--make-ref {}: {e}", out.display()));
    eprintln!(
        "--make-ref: {} ({}x{}) -> {} ({w}x{h}), Lanczos3",
        src.display(),
        img.width(),
        img.height(),
        out.display()
    );
}

/// Write a decoded [`RefImage`] back out as a PNG.
///
/// `refimg::load_reference` may have RESIZED and snapped what it read (a
/// 1000×700 reference becomes 992×688), and ComfyUI has no node that applies
/// hipfire's rule — its Klein template scales to exactly 1 MP instead. So
/// when the snap fires, the only way to encode the same pixels on both sides
/// is to hand ComfyUI the pixels hipfire actually fed its VAE. This writes
/// them.
fn dump_snapped(r: &RefImage, out: &Path) {
    let mut img = image::RgbImage::new(r.width as u32, r.height as u32);
    let plane = r.width * r.height;
    for (x, y, p) in img.enumerate_pixels_mut() {
        for c in 0..3 {
            let v = r.pixels[c * plane + y as usize * r.width + x as usize];
            p[c] = (((v + 1.0) * 127.5).round()).clamp(0.0, 255.0) as u8;
        }
    }
    img.save(out)
        .unwrap_or_else(|e| panic!("--dump-snapped {}: {e}", out.display()));
    eprintln!(
        "--dump-snapped: wrote {}x{} to {}",
        r.width,
        r.height,
        out.display()
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // `--make-ref` is a pure image utility: handled before anything reads a
    // fixture, loads weights or touches the GPU, so it works on a box with
    // no ROCm and no checkpoint.
    if let Some(i) = argv.iter().position(|a| a == "--make-ref") {
        let need = |k: usize, what: &str| -> &String {
            argv.get(i + k)
                .unwrap_or_else(|| panic!("--make-ref needs <src> <WxH> <out> (missing {what})"))
        };
        make_ref(
            Path::new(need(1, "src")),
            need(2, "WxH"),
            Path::new(need(3, "out")),
        );
        return;
    }

    // Tolerance knobs FIRST, before any argument, fixture, checkpoint or GPU
    // is touched: a malformed bar must cost a millisecond, not a 3-minute
    // weight upload followed by a PASS nobody asked for. `env_f32` exits 1 on
    // anything it cannot accept and prints the resolved value either way.
    let tol = env_f32("TOL", 0.0935, 1.0);
    let vel_tol = env_f32("VELOCITY_TOL", VELOCITY_FP8_BRACKET, 1.0);
    let final_tol = env_f32("FINAL_TOL", FINAL_FP8_BRACKET, 1.0);

    let mut positional: Vec<String> = Vec::new();
    let mut images: Vec<PathBuf> = Vec::new();
    let mut ref_latents: Vec<PathBuf> = Vec::new();
    let mut snapped_out: Option<PathBuf> = None;
    let mut ref_only = false;
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        let mut val = |flag: &str| it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match a.as_str() {
            "--image" => images.push(PathBuf::from(val("--image"))),
            "--ref-latent" => ref_latents.push(PathBuf::from(val("--ref-latent"))),
            "--dump-snapped" => snapped_out = Some(PathBuf::from(val("--dump-snapped"))),
            "--ref-only" => ref_only = true,
            other if other.starts_with("--") => panic!(
                "unknown flag {other} (expected --image, --ref-latent, --ref-only, \
                 --dump-snapped, --make-ref)"
            ),
            _ => positional.push(a),
        }
    }
    // A `--ref-latent` with no `--image` compares against nothing; fail loud
    // rather than silently skipping the check the caller asked for.
    assert!(
        ref_latents.len() <= images.len(),
        "--ref-latent given {} time(s) but only {} --image: each reference \
         latent is compared against the reference at the SAME index",
        ref_latents.len(),
        images.len()
    );
    let mut positional = positional.into_iter();
    let pipe_dir = PathBuf::from(
        positional
            .next()
            .unwrap_or_else(|| "/home/user/comfy-models/klein/FLUX.2-klein-4B".to_string()),
    );
    let fixture =
        PathBuf::from(positional.next().unwrap_or_else(|| {
            "crates/hipfire-arch-diffusion/tests/fixtures/klein-golden/4b".into()
        }));
    let order = match std::env::var("PACK_ORDER").as_deref() {
        Ok("alt") | Ok("patch") => PackOrder::PatchMajor,
        _ => PackOrder::ChannelMajor,
    };

    let meta_raw = std::fs::read_to_string(fixture.join("meta.json"))
        .unwrap_or_else(|e| panic!("fixture meta.json: {e}"));
    let meta: serde_json::Value =
        serde_json::from_str(&meta_raw).unwrap_or_else(|e| panic!("meta.json invalid: {e}"));
    // The second velocity bar: fixture-carried, 2x the last measured value.
    // Read here, next to the other required meta fields, so a fixture without
    // one fails before the checkpoint is opened rather than after a 3-minute
    // upload and a 4-step denoise.
    let vel_bar = velocity_regression_bar(&meta);
    let steps = meta["steps"].as_u64().expect("meta.steps") as usize;
    let width = meta["width"].as_u64().expect("meta.width") as usize;
    let height = meta["height"].as_u64().expect("meta.height") as usize;
    // The prompt is the FILE, not a meta.json string: a JSON re-encode of a
    // prompt is a place for whitespace to change without anyone noticing.
    let prompt = std::fs::read_to_string(fixture.join("prompt.txt"))
        .unwrap_or_else(|e| panic!("fixture prompt.txt: {e}"));
    // One trailing newline is a different prompt: it tokenizes differently,
    // conditions differently, and would make this gate compare hipfire's
    // answer to one question against ComfyUI's answer to another — while
    // still producing a plausible number. `$EDITOR` adds it for free.
    assert!(
        !prompt.ends_with('\n') && !prompt.is_empty(),
        "fixture prompt.txt must hold the prompt's EXACT bytes with no \
         trailing newline (meta.json records its md5 and length); got {} \
         bytes ending in {:?}",
        prompt.len(),
        prompt.chars().last()
    );

    let read_latent = |name: &str| -> (Vec<f32>, [usize; 4]) {
        let p = fixture.join(name);
        let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        read_comfy_latent(&bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    };
    let (init_comfy, init_shape) = read_latent("init.latent");
    let (golden_comfy, golden_shape) = read_latent("golden.latent");
    assert_eq!(
        init_shape, golden_shape,
        "fixture latents disagree on shape"
    );

    eprintln!("pipe    : {}", pipe_dir.display());
    eprintln!("fixture : {}", fixture.display());
    eprintln!("prompt  : {prompt:?}");
    eprintln!("size    : {width}x{height}  steps: {steps}  latent {init_shape:?}");
    eprintln!("order   : {order:?}");

    let mut bundle: FluxPipeBundle =
        load_pipe(&pipe_dir).unwrap_or_else(|e| panic!("load_pipe: {e}"));
    assert!(
        matches!(bundle.meta.latent_norm, LatentNorm::BatchNorm { .. }),
        "this gate is FLUX.2 Klein only: the pipe's latent norm is not the \
         BatchNorm one (a FLUX.1 pipe belongs in gpu_flux_golden_latent)"
    );
    assert_eq!(
        bundle.meta.shift_rule,
        scheduler::ShiftRule::Empirical,
        "this gate's golden was captured with ComfyUI's Flux2Scheduler, which \
         is the empirical-mu exponential shift; a pipe on a Fixed shift would \
         measure the SCHEDULE and blame the transformer"
    );

    let mut gpu = Gpu::init().expect("GPU init failed");
    bundle
        .ensure_gpu(&mut gpu)
        .unwrap_or_else(|e| panic!("ensure_gpu: {e}"));

    let up = vae_upscale(&bundle);
    let (lh, lw) = (height / up, width / up);
    let n_img = (lh / 2) * (lw / 2);
    let c4 = bundle.transformer_cfg.patch_in();
    let (comfy_c, comfy_i, comfy_j) = (init_shape[1], init_shape[2], init_shape[3]);
    assert_eq!(
        (comfy_c, comfy_i, comfy_j),
        (c4, lh / 2, lw / 2),
        "ComfyUI latent is [{comfy_c}, {comfy_i}, {comfy_j}] but a {width}x{height} \
         FLUX.2 latent at VAE upscale {up} with a 2x2 patch is [{c4}, {}, {}]",
        lh / 2,
        lw / 2
    );

    let init_packed = comfy_to_packed(&init_comfy, c4, comfy_i, comfy_j, order);
    let golden_packed = comfy_to_packed(&golden_comfy, c4, comfy_i, comfy_j, order);

    // Check the noise BEFORE spending a denoise on it, and fail hard rather
    // than reporting a number. Every way of extracting ComfyUI's noise that
    // does NOT work returns a file of the right name, shape and size — two of
    // them return the ZEROS they were handed (see the FLUX.1 fixture's
    // meta.json `note_noise`, which enumerates four failures) — and denoising
    // from a constant still produces a finished-looking latent. A gate that
    // cannot tell a bad input from a bad implementation sends you debugging
    // the wrong component.
    //
    // The same assert covers the normalization question from the module doc:
    // a ComfyUI FLUX.2 latent is already BatchNorm-normalized, so unit noise
    // reads ~0/~1 here with no conversion. Had it been raw VAE space, or had
    // this path applied `normalize_packed` on top, the std would not be 1.
    {
        let (mean, std) = mean_std(&init_packed);
        eprintln!("init noise: mean {mean:.4}  std {std:.4}  (expect ~0.0 / ~1.0)");
        assert!(
            mean.abs() < 0.05 && (std - 1.0).abs() < 0.05,
            "FIXTURE BROKEN, not the model: init.latent is not unit noise \
             (mean {mean:.4}, std {std:.4}). A std of ~0 means the extraction \
             graph returned its input latent instead of the noise; a std far \
             from 1 means the latent was normalized twice (a ComfyUI FLUX.2 \
             `.latent` is ALREADY BatchNorm-normalized — see the module doc). \
             Regenerate with the fixture's comfy-graph.json \
             and do not compare against it until this line reads ~0.0 / ~1.0."
        );
        let (gm, gs) = mean_std(&golden_packed);
        eprintln!("golden    : mean {gm:.4}  std {gs:.4}");
    }

    // The schedule both sides run. Printed rather than merely used, so a
    // report can put it next to ComfyUI's `Flux2Scheduler` output.
    let pairs = scheduler::sigma_pairs_ruled(steps, bundle.meta.shift_rule, n_img);
    eprintln!(
        "mu      : {:.9}  (image_seq_len {n_img}, {steps} steps)",
        scheduler::empirical_mu(n_img, steps)
    );
    eprintln!(
        "sigmas  : {}",
        pairs
            .iter()
            .map(|(s, _)| format!("{s:.9}"))
            .chain(std::iter::once(format!("{:.9}", pairs[steps - 1].1)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // ---- the edit path -------------------------------------------------
    //
    // Decoded and encoded through the PRODUCT's own functions
    // (`refimg::load_reference`, `pipeline::build_ref_tokens`), not through a
    // copy of them: a copy would agree with itself while the shipped path
    // drifted, which is the one failure a parity gate exists to prevent.
    let refs: Vec<RefImage> = images
        .iter()
        .map(|p| refimg::load_reference(p).unwrap_or_else(|e| panic!("{}", e)))
        .collect();
    for (p, r) in images.iter().zip(&refs) {
        eprintln!(
            "reference: {} -> {}x{} after refimg::target_size (area cap {}, snap {})",
            p.display(),
            r.width,
            r.height,
            refimg::MAX_REF_AREA,
            refimg::REF_MULTIPLE
        );
    }
    if let Some(out) = &snapped_out {
        dump_snapped(
            refs.first()
                .unwrap_or_else(|| panic!("--dump-snapped needs an --image")),
            out,
        );
    }
    // `REF_ENCODE=cpu` runs the same `build_ref_tokens` against the HOST
    // encoder instead of the device one. It exists to answer the first
    // question a failing reference check raises: is the residual hipfire's,
    // or is it between hipfire and ComfyUI? `gpu_klein_vae_parity` bounds
    // hipfire's two encoders against each other at 1e-4 (both f32), so if
    // both land at the same distance from ComfyUI the residual is not a
    // hipfire backend difference.
    let cpu_encode = std::env::var("REF_ENCODE").as_deref() == Ok("cpu");
    let ref_tokens: Vec<RefTokens> = if refs.is_empty() {
        Vec::new()
    } else {
        if cpu_encode {
            eprintln!("REF_ENCODE=cpu: encoding references on the HOST reference encoder");
        }
        let g = if cpu_encode { None } else { Some(&mut gpu) };
        build_ref_tokens(&bundle, g, &refs).unwrap_or_else(|e| panic!("build_ref_tokens: {e}"))
    };

    // The reference-latent check, run BEFORE a denoise is spent on it. This
    // is the edit path's twin of the golden decode: no transformer and no
    // schedule in it, so a failure here is about the INPUT and every model
    // number below would be measuring the wrong thing.
    //
    // ComfyUI's `VAEEncode` for a FLUX.2 VAE already applies the BatchNorm
    // (it lives inside `AutoencoderKL.encode`) and `latent_formats.Flux2`
    // adds no scale factor, so `SaveLatent` writes the same normalized space
    // `build_ref_tokens` ends in. Transpose only — the same conversion as
    // every other latent here.
    let ref_checks: Vec<(usize, f32, f32)> = ref_latents
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let (v, shape) =
                read_comfy_latent(&bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let r = &refs[i];
            let (rc, ri, rj) = (shape[1], shape[2], shape[3]);
            assert_eq!(
                (rc, ri, rj),
                (c4, r.height / (2 * up), r.width / (2 * up)),
                "{} is [{rc}, {ri}, {rj}] but reference {i} is {}x{} pixels, which \
                 at VAE upscale {up} with a 2x2 patch is [{c4}, {}, {}]. The \
                 usual cause is that ComfyUI resized the image and hipfire did \
                 not (its Klein template runs ImageScaleToTotalPixels to exactly \
                 1 MP; refimg::target_size caps AREA at 1024^2 and never \
                 upscales) — capture the reference at a size that is a fixed \
                 point of both rules, or feed ComfyUI the --dump-snapped PNG.",
                p.display(),
                r.width,
                r.height,
                r.height / (2 * up),
                r.width / (2 * up)
            );
            let want = comfy_to_packed(&v, rc, ri, rj, order);
            let (m_ours, s_ours) = mean_std(&ref_tokens[i].packed);
            let (m_ref, s_ref) = mean_std(&want);
            eprintln!(
                "ref latent {i}: ours mean {m_ours:+.4} std {s_ours:.4} | \
                 ComfyUI mean {m_ref:+.4} std {s_ref:.4}"
            );
            // Border-vs-interior split. This is the discriminator between the
            // two causes a nonzero residual can have, and they have opposite
            // fixes. A convention difference in the encoder — padding mode,
            // resample alignment, an off-by-one in the downsample — lands on
            // the EDGE of the grid and leaves the middle alone. A dtype or
            // accumulation difference is broadband and reads the same in
            // both. Without the split, one number cannot tell them apart and
            // a bound gets moved on a guess.
            let ours = &ref_tokens[i].packed;
            let mut sum = |border: bool| -> (f64, f64) {
                let (mut num, mut den) = (0f64, 0f64);
                for y in 0..ri {
                    for x in 0..rj {
                        let edge = y == 0 || x == 0 || y + 1 == ri || x + 1 == rj;
                        if edge != border {
                            continue;
                        }
                        let t = y * rj + x;
                        for f in 0..rc {
                            let d = (ours[t * rc + f] - want[t * rc + f]) as f64;
                            num += d * d;
                            den += (want[t * rc + f] as f64).powi(2);
                        }
                    }
                }
                (num, den)
            };
            let (bn, bd) = sum(true);
            let (inn, ind) = sum(false);
            eprintln!(
                "ref latent {i}: border rel_l2 {:.4e} ({} tokens)  interior rel_l2 {:.4e} ({} tokens) \
                 — a padding/resample convention difference lands on the border; a dtype \
                 difference reads the same in both",
                (bn / bd.max(1e-30)).sqrt(),
                2 * (ri + rj) - 4,
                (inn / ind.max(1e-30)).sqrt(),
                ri * rj - (2 * (ri + rj) - 4)
            );
            let (a, b) = latent_rel_error(ours, &want);
            (i, a, b)
        })
        .collect();

    // `--ref-only` stops here: the reference checks alone, no denoise. It is
    // what makes the SNAP rule testable. `refimg::target_size` floors a
    // 1000x700 reference to 992x688, and no ComfyUI node reproduces that rule
    // — its Klein template scales to exactly 1 MP instead — so the only way
    // to compare the two encoders on a snapped reference is to hand ComfyUI
    // the `--dump-snapped` pixels. That reference then has a different
    // geometry from the fixture's, which would make the model bars compare
    // against a golden captured from a DIFFERENT reference and fail for a
    // reason that has nothing to do with the model. Stopping is the honest
    // answer; reporting those bars would not be.
    if ref_only {
        assert!(
            !ref_checks.is_empty(),
            "--ref-only with no --ref-latent checks nothing (pass the ComfyUI \
             VAEEncode SaveLatent of the same pixels; --dump-snapped writes \
             the pixels hipfire actually fed its encoder)"
        );
        println!("== FLUX.2 Klein reference-latent check (--ref-only; no denoise) ==");
        let mut bad = false;
        for (i, a, b) in &ref_checks {
            let verdict = if *b <= REF_TOL { "ok" } else { "OVER" };
            println!("  ref latent {i}  rel_inf {a:.4e}   rel_l2 {b:.4e}   bound {REF_TOL:.4}  {verdict}");
            bad |= *b > REF_TOL;
        }
        if bad {
            std::process::exit(1);
        }
        return;
    }

    let cond = condition_prompt(&bundle, &prompt, bundle.meta.max_seq)
        .unwrap_or_else(|e| panic!("condition_prompt: {e}"));
    let input = Txt2ImgInput {
        txt_ids: &cond.txt_ids,
        txt_mask: &cond.txt_mask,
        clip_ids: &cond.clip_ids,
        clip_mask: &cond.clip_mask,
        // Empty for txt2img; `--image` makes this the EDIT gate, with the
        // reference tokens appended after the generated ones and held fixed
        // across every step (ComfyUI: `torch.cat([img, kontext], dim=1)`).
        references: &ref_tokens,
        init_latents: Some(&init_packed),
        height: lh,
        width: lw,
        steps,
        mlp_act: FluxPipeBundle::mlp_act_default(),
        prompt_key: None,
    };
    let t0 = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let mut step_ms: Vec<f64> = Vec::new();
    let out = generate_txt2img_steps_gpu(&mut bundle, &mut gpu, &input, &mut |i, n| {
        let ms = last.elapsed().as_secs_f64() * 1e3;
        last = std::time::Instant::now();
        step_ms.push(ms);
        eprintln!("  step {i}/{n}  {ms:.0} ms");
    })
    .unwrap_or_else(|e| panic!("denoise: {e}"));
    let total_s = t0.elapsed().as_secs_f64();

    // ComfyUI hands back ONE latent, so there is no per-step reference to
    // diff against — the FLUX.1 gate's step-1 bisect is not available here.
    // What IS available for free is the shape of our own trajectory, and it
    // separates the two failure modes almost as well: the velocity of a
    // working flow-matching model has std ~1 at every step and a latent whose
    // std falls monotonically from 1 toward the data scale. A velocity std
    // near 0 means the transformer produced no signal (conditioning or
    // modulation), and one far above 1 means the latent scaling is off.
    println!("== per-step trajectory (no ComfyUI reference; shape diagnosis only) ==");
    println!("    k     sigma    sigma'    t_model   |v| mean     |v| std   x_out std      ms");
    for (k, rec) in out.steps.iter().enumerate() {
        let (sigma, sigma_next) = pairs[k];
        let (vm, vs) = mean_std(&rec.noise_pred);
        let (_, xs) = mean_std(&rec.latents_out);
        println!(
            "   {:2}  {sigma:8.6}  {sigma_next:8.6}  {:9.4}  {vm:+10.5}  {vs:10.5}  {xs:9.5}  {:6.0}",
            k + 1,
            rec.t_model,
            step_ms.get(k).copied().unwrap_or(f64::NAN)
        );
    }

    let ours_packed = &out
        .steps
        .last()
        .expect("at least one denoise step")
        .latents_out;
    let (rel_inf, rel_l2) = latent_rel_error(ours_packed, &golden_packed);

    // Bisect the trajectory with the optional one-step reference.
    //
    // The final number alone cannot separate the two causes of a divergence,
    // and they have opposite fixes. A difference already present after ONE
    // step is a convention or conditioning difference — one Euler step is a
    // single forward pass with nothing fed back. A difference that is small
    // at step 1 and large at step 4 is the flow amplifying a small per-step
    // difference, which on a 4-step bf16 schedule it demonstrably does (see
    // the calibration in the 9b fixture's meta.json).
    //
    // The reference needs one correction first. Cutting the sigma list short
    // makes ComfyUI stop at a NON-ZERO sigma, and `CFGGuider.inner_sample`
    // ends with `inverse_noise_scaling(sigmas[-1], .)`, which for flow
    // matching divides by `(1 - sigma)`. At sigma1 ~ 0.967 that is a factor
    // of 30.7 — the saved file reads rms ~30 where the latent is order 1.
    // Undo it with the sigma OUR scheduler used for the same step. The
    // printed rms pair exists so a wrong factor shows up as a scale error
    // instead of quietly inflating rel_l2 into a false model defect.
    let (step1_rel, step1_vel) = {
        let p = fixture.join("step1.latent");
        let b = std::fs::read(&p).unwrap_or_else(|e| {
            panic!(
                "{}: {e} — this fixture file is REQUIRED, because the one-step \
                 latent is what this gate asserts on (the final latent cannot \
                 discriminate at 4 steps; see the module doc). Capture it with \
                 the fixture's comfy-graph.json plus a \
                 SplitSigmas(step=1) between Flux2Scheduler and the sampler.",
                p.display()
            )
        });
        let (g1, _) = read_comfy_latent(&b).unwrap_or_else(|e| panic!("step1.latent: {e}"));
        let s1 = pairs[0].1;
        let g1 = comfy_to_packed(&g1, c4, comfy_i, comfy_j, order);
        let ref1: Vec<f32> = g1.iter().map(|v| v * (1.0 - s1)).collect();
        let rms = |v: &[f32]| {
            (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
        };
        eprintln!(
            "step 1: sigma1 {s1:.6}, leftover-noise factor (1-sigma1) {:.6}; rms ours {:.4} vs ref {:.4}",
            1.0 - s1,
            rms(&out.steps[0].latents_out),
            rms(&ref1)
        );
        // The step-1 LATENT is a diluted view of the step-1 VELOCITY, and the
        // dilution is severe enough that it must be printed, not inferred:
        // `x₁ = x₀ + (σ₁-σ₀)·v` with `x₀` byte-identical on both sides, so
        // `Δx₁ = (σ₁-σ₀)·Δv` and at σ₁ = 0.967 the factor is 0.0326. A
        // 1.6e-2 disagreement on the latent is a 44% disagreement on the
        // velocity. That is not a reason to distrust the number — the fp8
        // control in the findings doc puts a pure weight-dtype change at 15%
        // on the same quantity, so the velocity field at σ = 1 is simply
        // ill-conditioned — but a gate that reported only the diluted number
        // would be claiming 30× more precision than it has.
        let dsigma = pairs[0].1 - pairs[0].0;
        let v_ref: Vec<f32> = ref1
            .iter()
            .zip(&init_packed)
            .map(|(x1, x0)| (x1 - x0) / dsigma)
            .collect();
        let (v_inf, v_l2) = latent_rel_error(&out.steps[0].noise_pred, &v_ref);
        eprintln!(
            "step 1 velocity (undiluted: Δx₁ = (σ₁-σ₀)·Δv, factor {:.6}): \
             rel_inf {v_inf:.4e} rel_l2 {v_l2:.4e}  \
             — ComfyUI-vs-ComfyUI at fp8 weights is ~1.5e-1 on this quantity",
            dsigma.abs()
        );
        (
            latent_rel_error(&out.steps[0].latents_out, &ref1),
            (v_inf, v_l2),
        )
    };

    // Decoding the GOLDEN latent through hipfire's own VAE and comparing to
    // ComfyUI's PNG is the decisive test of both conventions in the module
    // doc, and it costs one decode. It is a closed loop through ComfyUI's
    // encoder-side packing and hipfire's decoder-side unpacking: if the
    // channel order were wrong, or if the latent needed a normalization this
    // path does not apply, the reconstruction is noise, not an image — and it
    // says so WITHOUT the transformer in the picture, so a failure here can
    // never be mistaken for a model defect.
    let golden_img = decode_packed(&bundle, &mut gpu, &golden_packed, n_img, lh, lw)
        .unwrap_or_else(|e| panic!("decode golden: {e}"));
    let comfy_png = refimg::load_reference(&fixture.join("golden.png"))
        .unwrap_or_else(|e| panic!("golden.png: {e}"));
    assert_eq!(
        (comfy_png.width, comfy_png.height),
        (width, height),
        "golden.png is not {width}x{height}"
    );
    let (dec_inf, dec_l2) = latent_rel_error(&golden_img, &comfy_png.pixels);
    // Ours vs ComfyUI's PNG, in pixels. `out.image` is the same `[-1, 1]`
    // channel-major layout `refimg` produces.
    let (img_inf, img_l2) = if out.image.len() == comfy_png.pixels.len() {
        latent_rel_error(&out.image, &comfy_png.pixels)
    } else {
        (f32::NAN, f32::NAN)
    };

    if let Ok(p) = std::env::var("DUMP_OURS") {
        let as_comfy = packed_to_comfy(ours_packed, c4, comfy_i, comfy_j, order);
        std::fs::write(&p, comfy_latent_bytes(&as_comfy, c4, comfy_i, comfy_j))
            .unwrap_or_else(|e| panic!("dump ours {p}: {e}"));
        eprintln!("wrote our final latent to {p}");
    }
    if let Ok(p) = std::env::var("DUMP_PNG") {
        std::fs::write(Path::new(&p), &out.png).unwrap_or_else(|e| panic!("dump png {p}: {e}"));
        eprintln!("wrote our decode to {p}");
    }

    let steady: Vec<f64> = step_ms.iter().skip(1).copied().collect();
    let mean_step = if steady.is_empty() {
        total_s * 1e3 / steps as f64
    } else {
        steady.iter().sum::<f64>() / steady.len() as f64
    };

    println!("== FLUX.2 Klein end-to-end golden latent gate ==");
    let (r1_inf, r1_l2) = step1_rel;
    let (v1_inf, v1_l2) = step1_vel;
    println!("  latent  {golden_shape:?}  steps {steps}  same initial noise on both sides");
    if refs.is_empty() {
        println!("  mode    txt2img (no --image)");
    } else {
        println!(
            "  mode    EDIT, {} reference(s): {}  ({} image tokens + {} reference tokens)",
            refs.len(),
            refs.iter()
                .map(|r| format!("{}x{}", r.width, r.height))
                .collect::<Vec<_>>()
                .join(", "),
            n_img,
            ref_tokens.iter().map(|r| r.n).sum::<usize>()
        );
    }
    for (i, a, b) in &ref_checks {
        println!(
            "  ref latent {i}  rel_inf {a:.4e}   rel_l2 {b:.4e}   bound {REF_TOL:.4}  \
             (hipfire VAE encode + pack + normalize vs ComfyUI VAEEncode — no transformer)"
        );
    }
    println!(
        "  step-1 velocity rel_inf {v1_inf:.4e}   rel_l2 {v1_l2:.4e}   tol {vel_tol:.4}   \
         <- THE GATE (undiluted; ComfyUI-vs-ComfyUI at fp8 weights is ~1.5e-1)"
    );
    println!(
        "  velocity tripwire                    bar {vel_bar:.4e}  (meta.velocity_regression_bar \
         = 2x the last measured value; headroom {:.2}x)",
        vel_bar / v1_l2.max(1e-12)
    );
    println!("  after 1 step   rel_inf {r1_inf:.4e}   rel_l2 {r1_l2:.4e}   tol {tol:.4}   (diluted by (1-sigma1) = 0.0326)");
    println!("  final latent   rel_inf {rel_inf:.4e}   rel_l2 {rel_l2:.4e}   bound {final_tol:.3}  (ComfyUI-vs-ComfyUI at fp8 weights is 4.2e-1)");
    println!(
        "  growth {:.1}x over {steps} steps — the 4-step schedule integrates a per-step \
         difference rather than averaging it",
        rel_l2 / r1_l2.max(1e-9)
    );
    println!("  golden decode  rel_inf {dec_inf:.4e}   rel_l2 {dec_l2:.4e}   bound {DECODE_TOL:.3}  (hipfire VAE vs ComfyUI PNG — the latent-convention check)");
    println!("  our decode     rel_inf {img_inf:.4e}   rel_l2 {img_l2:.4e}   (pixels vs ComfyUI PNG; diagnostic)");
    println!(
        "  {total_s:.2} s total, {mean_step:.0} ms/step steady state (first step dropped: JIT)"
    );

    // The golden-decode number is checked FIRST and on its own bound, because
    // it is the only one of the three that has no transformer in it. If it
    // fails, the latent conversion is wrong and every other number in this
    // run is meaningless — reporting them as a model verdict is how a fixture
    // bug gets filed against the forward pass.
    let mut failed = false;
    // Checked before the model bars for the same reason the golden decode is:
    // it has no transformer in it, so if it fails the reference the trunk saw
    // is not the reference ComfyUI saw and the velocity number below is a
    // measurement of the wrong input.
    for (i, _, b) in &ref_checks {
        if !(*b <= REF_TOL) {
            failed = true;
            println!("FAIL: REFERENCE {i} does not match ComfyUI's encode ({b:.4e} > {REF_TOL}).");
            println!("  No transformer is in this path, so it is the reference INPUT, not");
            println!("  the model. Suspect, in order:");
            println!("  1. the resize — ComfyUI's Klein template runs");
            println!("     ImageScaleToTotalPixels(nearest-exact, 1.0 MP, steps=1) before");
            println!("     VAEEncode and hipfire's refimg::target_size does not (area cap,");
            println!("     never upscale, floor to 16). Capture at a size that is a fixed");
            println!("     point of both, or feed ComfyUI the --dump-snapped PNG.");
            println!("  2. the normalization — a ComfyUI FLUX.2 latent is ALREADY");
            println!("     BatchNorm-normalized, so this compares against the PACKED AND");
            println!("     NORMALIZED tokens, not the raw VAE output.");
            println!("  3. the channel order — rerun with PACK_ORDER=alt and compare");
            println!("     (it takes this number to ~1.4, two orders above the bound).");
            println!("  4. the encoder itself — gpu_klein_vae_parity bounds hipfire's GPU");
            println!("     encoder against its own CPU reference at 1e-4 (measured 2.6e-5,");
            println!("     both f32), and REF_ENCODE=cpu reruns this check on the host");
            println!("     encoder; if the two agree, the residual is not hipfire's.");
            println!("  5. the border-vs-interior line above — if the border is much worse");
            println!("     it is a padding or resample convention, not a dtype.");
        }
    }
    if !(dec_l2 <= DECODE_TOL) {
        failed = true;
        println!("FAIL: the LATENT CONVENTION is wrong, not the model.");
        println!("  Decoding ComfyUI's own golden latent through hipfire's VAE did");
        println!("  not reproduce ComfyUI's own PNG ({dec_l2:.4e} > {DECODE_TOL}), and that");
        println!("  path never touches the transformer. Suspect, in order: the");
        println!("  channel order (rerun with PACK_ORDER=alt and compare), a");
        println!("  double or missing BatchNorm on the load path, and the token");
        println!("  row-major assumption. Ignore every other number above.");
    }

    // The model verdict: the UNDILUTED velocity of one forward pass against
    // ComfyUI's own, on the tightest bound the quantity can carry (the fp8
    // bracket). This is the bar the step-1 latent used to stand in for; it is
    // 30x stronger, because `x₁` is 96.7% `x₀` and `x₀` is byte-identical on
    // both sides. Before the text-RoPE fix (Klein rotates text tokens by
    // token index on axis 3, ComfyUI `txt_ids_dims = [3]`) this number was
    // 4.385e-1 while the step-1 LATENT still read a comfortable 1.635e-2 —
    // the diluted bar could not see a whole missing rotation.
    if !(v1_l2 <= vel_tol) {
        failed = true;
        println!(
            "FAIL: hipfire's step-1 VELOCITY diverges from ComfyUI ({v1_l2:.4e} > {vel_tol})."
        );
        println!("  This is one forward with nothing fed back and nothing diluting it,");
        println!("  so it is a whole-pass CONVENTION, not accumulation. Bisect in order");
        println!("  of cost:");
        println!("  1. position ids — text rows are (0,0,0,l) by TOKEN INDEX on the");
        println!("     Flux2 path (ComfyUI model_detection txt_ids_dims=[3]), image rows");
        println!("     (index, h, w, 0). Leaving text rows unrotated costs 0.44 here.");
        println!("  2. schedule — the sigma line printed above must match ComfyUI's");
        println!("     Flux2Scheduler; pin ComfyUI to that list with ManualSigmas and");
        println!("     diff. An unshifted schedule alone is worth 0.79 of final rel_l2.");
        println!("  3. conditioning — the chatml template, the 512 right-pad, the");
        println!("     [9,18,27] taps and their `tap*hidden` concat order.");
        println!("  4. the trunk — the final adaLN chunk order, the shared modulation");
        println!("     split, the sigma→t_model convention.");
        println!("  5. the |v| std column — a working flow model holds it near 1.");
    }

    // The REGRESSION tripwire, checked after the ceiling and reported
    // separately: a run that clears 0.15 but has doubled since the fixture
    // was captured is not correct-and-fine, it is a change nobody measured.
    // Every value the ceiling has ever seen sits 3-11x under it, so without
    // this bar a 3x degradation of the forward passes green.
    if !(v1_l2 <= vel_bar) {
        failed = true;
        println!("FAIL: the step-1 velocity REGRESSED ({v1_l2:.4e} > {vel_bar:.4e}, the fixture's");
        println!("  velocity_regression_bar). It is still inside the {vel_tol:.4} correctness");
        println!("  ceiling, so this is not a broken convention — it is a forward that moved");
        println!("  since this fixture was captured, and the move is larger than the 2x");
        println!("  margin the bar allows. Either find the change (git bisect against this");
        println!("  gate, cheapest suspects: kernel/dispatch selection, a dtype on the");
        println!("  conditioning path, a GEMM route flag) or, if the new number is correct");
        println!("  and understood, re-measure and move meta.velocity_regression_bar in the");
        println!("  SAME commit, recording the new sha in note_velocity_regression_bar.");
    }

    // The diluted view of the same forward, kept as a second bar because it
    // is directly comparable with the FLUX.1 gate's 0.0935.
    if !(r1_l2 <= tol) {
        failed = true;
        println!("FAIL: hipfire's FIRST STEP diverges from ComfyUI ({r1_l2:.4e} > {tol}).");
        println!("  One Euler step is one forward with nothing fed back, so this is a");
        println!("  whole-pass CONVENTION, not accumulation. Bisect in order of cost:");
        println!("  1. schedule — the sigma line printed above must match ComfyUI's");
        println!("     Flux2Scheduler; pin ComfyUI to that list with ManualSigmas and");
        println!("     diff. An unshifted schedule alone is worth 0.79 of final rel_l2.");
        println!("  2. conditioning — the chatml template, the 512 right-pad, the");
        println!("     [9,18,27] taps and their `tap*hidden` concat order.");
        println!("  3. the trunk — the final adaLN chunk order, the 4-axis RoPE ids,");
        println!("     the shared modulation split, the sigma→t_model convention.");
        println!("  4. the |v| std column — a working flow model holds it near 1.");
    }

    // The final latent still cannot resolve a dtype at 4 steps (module doc),
    // so its bound is the fp8 bracket rather than the FLUX.1 0.0935: it fires
    // when the accumulated deviation exceeds a whole weight-dtype change.
    if !(rel_l2 <= final_tol) {
        failed = true;
        println!("FAIL: the final latent exceeds the fp8 bracket ({rel_l2:.4e} > {final_tol:.3}).");
        println!("  A pure weight-dtype change costs 0.4158 here, so this is a larger");
        println!("  deviation than dtype — with a clean velocity above it means the");
        println!("  divergence is injected at EVERY step, not by a convention. Look at");
        println!("  what the forward does to a state that is still mostly noise (high");
        println!("  sigma), and at the last Euler step, which carries sigma 0.767 → 0.");
    }

    if failed {
        std::process::exit(1);
    }
    println!(
        "PASS: one forward's velocity matches ComfyUI at rel_l2 {v1_l2:.4e} (tol {vel_tol:.4},"
    );
    println!("  the fp8 bracket) and the fixture's {vel_bar:.4e} regression tripwire, the");
    println!("  diluted step-1 latent at {r1_l2:.4e} (tol {tol:.4}),");
    println!("  the latent conventions round-trip at {dec_l2:.4e}, and the 4-step");
    println!("  trajectory stays inside the fp8 bracket at {rel_l2:.4e}.");
}
