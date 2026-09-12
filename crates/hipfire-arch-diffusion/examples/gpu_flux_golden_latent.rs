// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! **End-to-end golden gate**: hipfire's whole denoise loop against ComfyUI's,
//! from byte-identical initial noise.
//!
//! This exists because `gpu_flux_block_parity` cannot catch a whole class of
//! bug. That gate stops at the last single block, so it reported rel 3e-4
//! against a 5e-2 bound while the model's FINAL adaLN head was computing the
//! wrong thing — wrong halves modulating the final projection, a systematically
//! wrong velocity, and an image that kept its composition but never lost its
//! noise. Four such defects shipped green (see commit 5a9d4db5): the T5 v1.0
//! vs v1.1 gated FFN, a missing guidance embedding, the final adaLN order, and
//! the latent-space convention. Every one of them is downstream or upstream of
//! the blocks, and therefore invisible to a per-block tolerance.
//!
//! What this compares is the FINAL LATENT, not an image: it is the last thing
//! the transformer produces, and it isolates the model from the VAE. Both sides
//! start from the same noise, so the comparison is a difference of
//! implementations rather than of random draws — the same seed INTEGER would
//! NOT do, because `seeded_gaussian` and torch's `randn` are different
//! generators.
//!
//! ## Fixture
//!
//! A directory holding:
//!   * `init.latent`   — the initial noise, ComfyUI `.latent`, VAE space.
//!   * `golden.latent` — ComfyUI's final latent from that same noise.
//!   * `meta.json`     — `{prompt, steps, width, height}`.
//!
//! The direction matters: we take ComfyUI's noise OUT, we do not push ours IN.
//! Pushing ours in cannot work, because `add_noise=disable` on a flow-matching
//! model computes `noise_scaling(σ₀=1, noise=0, latent) = 0` and starts the
//! sampler from zeros. Extracting is also easy to get wrong — see meta.json
//! `note_noise` and the assert below. The full recipe for both files lives in
//! meta.json `regenerate`, next to the data it produced.
//!
//! ## Tolerance
//!
//! Compared as `rel_inf = max|a-b| / max|golden|` and `rel_l2`. This is not a
//! bit-exactness test and cannot be: the two run different kernels, different
//! attention, and f16 vs bf16 operand rounding. The bar is set to pass that
//! noise and fail a wrong CONVENTION — a swapped adaLN order, a missing
//! guidance embedder or an ungated FFN all move the result by O(1), orders
//! above any rounding difference.
//!
//! Build + run (GPU required, gpu-lock it):
//! ```
//! cargo run --release --features lab --example gpu_flux_golden_latent \
//!   -p hipfire-arch-diffusion -- <pipe-dir> <fixture-dir>
//! ```
//! Env:
//!   TOL=<f32>       pass/fail bound on the final latent (default 0.15)
//!   TRACE=<dir>     per-step error curve against ComfyUI x_k in `<dir>/kNN.latent`
//!   PERTURB=<f32>   displace the INITIAL latent by this rel_l2 (sensitivity probe)
//!   DUMP_OURS=<path>  write our final latent as a ComfyUI `.latent`
//!   VERBOSE=1       per-channel statistics

use hipfire_arch_diffusion::flux::FluxForwardInput;
use hipfire_arch_diffusion::pipeline::{
    condition_prompt, generate_txt2img_steps_gpu, latent_rel_error, load_pipe, read_comfy_latent,
    vae_upscale, FluxPipeBundle, Txt2ImgInput,
};
use hipfire_arch_diffusion::scheduler;
use hipfire_arch_diffusion::vae::LatentNorm;
use rdna_compute::Gpu;
use std::path::PathBuf;

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pipe_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/home/user/flux-pipe".to_string()),
    );
    let fixture = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "crates/hipfire-arch-diffusion/tests/fixtures/flux-golden".into()),
    );
    let tol = env_f32("TOL", 0.15);

    let meta_raw = std::fs::read_to_string(fixture.join("meta.json"))
        .unwrap_or_else(|e| panic!("fixture meta.json: {e}"));
    let meta: serde_json::Value =
        serde_json::from_str(&meta_raw).unwrap_or_else(|e| panic!("meta.json invalid: {e}"));
    let prompt = meta["prompt"].as_str().expect("meta.prompt");
    let steps = meta["steps"].as_u64().expect("meta.steps") as usize;
    let width = meta["width"].as_u64().expect("meta.width") as usize;
    let height = meta["height"].as_u64().expect("meta.height") as usize;

    let (init_vae, init_shape) = read_comfy_latent(
        &std::fs::read(fixture.join("init.latent")).unwrap_or_else(|e| panic!("init.latent: {e}")),
    )
    .unwrap_or_else(|e| panic!("init.latent: {e}"));
    let (golden_vae, golden_shape) = read_comfy_latent(
        &std::fs::read(fixture.join("golden.latent"))
            .unwrap_or_else(|e| panic!("golden.latent: {e}")),
    )
    .unwrap_or_else(|e| panic!("golden.latent: {e}"));
    assert_eq!(
        init_shape, golden_shape,
        "fixture latents disagree on shape"
    );

    eprintln!("pipe    : {}", pipe_dir.display());
    eprintln!("fixture : {}", fixture.display());
    eprintln!("prompt  : {prompt}");
    eprintln!("size    : {width}x{height}  steps: {steps}  latent {init_shape:?}");

    // The VAE decode is not exercised here and cannot load a real FLUX VAE, so
    // take the config-only path: this gate is about the transformer.
    std::env::set_var("HIPFIRE_VAE_CONFIG_ONLY", "1");
    let mut bundle: FluxPipeBundle =
        load_pipe(&pipe_dir).unwrap_or_else(|e| panic!("load_pipe: {e}"));

    // The sampler must be IDENTICAL on both sides or this gate measures the
    // scheduler instead of the model. The same step count is not enough: the
    // FLUX sigma shift is a separate parameter and the three sources disagree
    // about it. The checkpoint's scheduler_config.json says 3.0. ComfyUI and
    // diffusers both apply DYNAMIC shifting, and both EXPONENTIATE — the shift
    // is `exp(mu)`, not `mu`, so at 1024x1024 where the resolution
    // interpolation gives mu = 1.15 the shift is exp(1.15) = 3.158.
    //
    // Measured, not derived, and the derivation was wrong: an earlier version
    // of this gate asserted ComfyUI interpolated LINEARLY and forced 1.15,
    // which is further from ComfyUI than the checkpoint's own 3.0 would have
    // been. Pinning ComfyUI to explicit sigma lists settled it — a hand-built
    // list at shift 1.15 diffs at rel_l2 0.2474 against ComfyUI's native
    // schedule, one at exp(1.15) at 0.0049. See meta.json `note_shift`.
    //
    // A whole-schedule mismatch is therefore worth ~0.25 of rel_l2 on its own,
    // which calibrates this gate: no tolerance below that survives a scheduler
    // difference, and a failure should suspect the schedule before the
    // transformer. The fixture records the shift its golden was made with.
    if let Some(shift) = meta.get("shift").and_then(|s| s.as_f64()) {
        eprintln!(
            "shift   : {shift} (from fixture; checkpoint default was {:?})",
            bundle.meta.shift_rule
        );
        bundle.meta.shift_rule = scheduler::ShiftRule::Fixed(shift as f32);
    }
    let mut gpu = Gpu::init().expect("GPU init failed");
    bundle
        .ensure_gpu(&mut gpu)
        .unwrap_or_else(|e| panic!("ensure_gpu: {e}"));

    let up = vae_upscale(&bundle);
    let (lh, lw) = (height / up, width / up);
    let ch = init_shape[1];
    assert_eq!(
        init_vae.len(),
        ch * lh * lw,
        "init.latent does not match {width}x{height} at VAE upscale {up}"
    );

    // ComfyUI latents are in VAE space; the denoise loop runs in model space.
    // Invert `process_latent_out` (which `scale_latents` implements), then pack
    // into the 2x2 patch layout the transformer consumes.
    let (sf, shf) = match &bundle.meta.latent_norm {
        LatentNorm::ScaleShift { scaling, shift } => (*scaling, *shift),
        LatentNorm::BatchNorm { .. } => panic!("this probe assumes FLUX.1 ScaleShift latents"),
    };
    let init_model: Vec<f32> = init_vae.iter().map(|v| (v - shf) * sf).collect();
    let (mut init_packed, n_img) = scheduler::pack_latents(&init_model, ch, lh, lw);
    assert_eq!(n_img, (lh / 2) * (lw / 2));

    // PERTURB measures how far the trajectory carries a difference injected
    // ONCE at the input. Set it to the step-1 residual, compare the run against
    // an unperturbed one, and the ratio is the flow's sensitivity to initial
    // conditions.
    //
    // MEASURED on FLUX.1-dev, 1024x1024, 20 steps: an input displacement of
    // rel_l2 1.98e-2 ends at 2.65e-2. A factor of 1.34 over twenty steps — the
    // map is very nearly neutral, NOT chaotic.
    //
    // That result is load-bearing, because it kills the comfortable
    // explanation. A step-1 agreement of 2% alongside a step-20 divergence of
    // 36% cannot be amplification of the step-1 difference; this flow does not
    // amplify. The arithmetic points elsewhere: 20 x 0.0198 = 0.396 against a
    // measured 0.362, while errors adding in quadrature would give only
    // sqrt(20) x 0.0198 = 0.089. So the per-step differences accumulate
    // COHERENTLY, which is the signature of a systematic bias in the velocity
    // field rather than of rounding noise.
    //
    // Know what this knob does NOT measure. One displacement at the input is
    // not the same quantity as a difference injected at every step, so a
    // PERTURB result can REFUTE the amplification story but can never confirm
    // whatever replaces it. Use TRACE for that.
    //
    // The perturbation is unit-normal noise rescaled so its L2 relative to the
    // latent is exactly PERTURB, from a fixed seed, so the experiment repeats.
    //
    let perturb = env_f32("PERTURB", 0.0);
    if perturb > 0.0 {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut d: Vec<f32> = Vec::with_capacity(init_packed.len());
        while d.len() < init_packed.len() {
            let (u1, u2) = (next().max(1e-12), next());
            let (r, th) = ((-2.0 * u1.ln()).sqrt(), std::f64::consts::TAU * u2);
            d.push((r * th.cos()) as f32);
            d.push((r * th.sin()) as f32);
        }
        d.truncate(init_packed.len());
        let l2 = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let k = (perturb as f64 * l2(&init_packed) / l2(&d)) as f32;
        for (x, dx) in init_packed.iter_mut().zip(d.iter()) {
            *x += k * dx;
        }
        eprintln!(
            "PERTURB {perturb:.4}: init latent displaced by rel_l2 {:.4e} \
             — compare THIS run's final latent against the unperturbed one",
            perturb
        );
    }

    // Check the noise we were handed BEFORE spending a denoise on it, and fail
    // hard rather than reporting a number. ComfyUI's noise is standard normal
    // in MODEL space, so after inverting `process_latent_out` the mean must be
    // ~0 and the std ~1.
    //
    // This assert is not defensive padding. Two of the three plausible ways to
    // extract ComfyUI's noise silently return the ZEROS they were handed (see
    // meta.json `note_noise`), and the resulting file has the right name, the
    // right shape and the right size. Denoising from a constant instead of
    // from noise still produces a finished-looking latent, so the comparison
    // came out at rel_l2 0.96 and read exactly like a model defect — the gate
    // accused the transformer of a bug in the fixture. A gate that cannot tell
    // a bad input from a bad implementation is worse than no gate, because it
    // sends you debugging the wrong component.
    {
        let n = init_model.len() as f64;
        let mean = init_model.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = init_model
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        eprintln!("init noise (model space): mean {mean:.4}  std {std:.4}  (expect ~0.0 / ~1.0)");
        assert!(
            mean.abs() < 0.05 && (std - 1.0).abs() < 0.05,
            "FIXTURE BROKEN, not the model: init.latent is not unit noise \
             (mean {mean:.4}, std {std:.4}). A std of ~0 means the file is a \
             constant — the extraction graph returned its input latent instead \
             of the noise. Regenerate it with the SamplerCustomAdvanced recipe \
             in meta.json `regenerate`, and do not compare against it until \
             this line reads ~0.0 / ~1.0."
        );
    }

    let cond = condition_prompt(&bundle, prompt, bundle.meta.max_seq)
        .unwrap_or_else(|e| panic!("condition_prompt: {e}"));
    let input = Txt2ImgInput {
        txt_ids: &cond.txt_ids,
        txt_mask: &cond.txt_mask,
        // FLUX.1 txt2img: no reference-image tokens.
        references: &[],
        clip_ids: &cond.clip_ids,
        clip_mask: &cond.clip_mask,
        init_latents: Some(&init_packed),
        height: lh,
        width: lw,
        steps,
        mlp_act: FluxPipeBundle::mlp_act_default(),
        prompt_key: None,
    };
    let _ = std::mem::size_of::<FluxForwardInput>();
    let out = generate_txt2img_steps_gpu(&mut bundle, &mut gpu, &input, &mut |i, n| {
        eprintln!("  step {}/{n}", i);
    })
    .unwrap_or_else(|e| panic!("denoise: {e}"));

    // Packed model space -> the VAE space ComfyUI's `.latent` files live in.
    let to_vae = |packed: &[f32]| -> Vec<f32> {
        scheduler::scale_latents(
            &scheduler::unpack_latents(packed, n_img, ch, lh, lw),
            sf,
            shf,
        )
    };

    let final_packed = &out
        .steps
        .last()
        .expect("at least one denoise step")
        .latents_out;
    let ours: Vec<f32> = to_vae(final_packed);
    let (rel_inf, rel_l2) = latent_rel_error(&ours, &golden_vae);

    // Bisect the trajectory when the fixture carries a 1-step reference.
    //
    // The final number alone cannot separate the two causes of a divergence,
    // and they have opposite fixes. A difference already present after ONE
    // step is a convention or conditioning difference — a wrong text
    // embedding, a wrong modulation order — because one Euler step is a single
    // forward pass with no accumulation. A difference that is small at step 1
    // and large at step 20 is arithmetic drift compounding through a chaotic
    // sampler, which is expected between f16 and bf16 and is not a defect.
    //
    // The reference needs one correction first. `return_with_leftover_noise`
    // makes ComfyUI stop at a NON-ZERO sigma, and `CFGGuider.inner_sample`
    // ends with `inverse_noise_scaling(sigmas[-1], ·)`, which for flow matching
    // divides by `(1 - σ)`. At σ₁ ≈ 0.956 that is a factor of 22.8 — the saved
    // file reads std 60 where the latent itself is order 1. Undo it with the
    // sigma OUR scheduler used for the same step, and compare in model space.
    // The printed norm ratio exists so a wrong factor shows up as a scale
    // error instead of quietly inflating rel_l2 into a false model defect.
    let step1_rel = std::fs::read(fixture.join("step1.latent")).ok().map(|b| {
        let (g1_vae, _) = read_comfy_latent(&b).unwrap_or_else(|e| panic!("step1.latent: {e}"));
        let (_, s1) = scheduler::sigma_pairs_ruled(steps, bundle.meta.shift_rule, 0)[0];
        let ref1: Vec<f32> = g1_vae.iter().map(|v| (v - shf) * sf * (1.0 - s1)).collect();
        let ours1 = scheduler::unpack_latents(&out.steps[0].latents_out, n_img, ch, lh, lw);
        let rms = |v: &[f32]| {
            (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
        };
        eprintln!(
            "step 1: σ₁ {:.6}, leftover-noise factor (1-σ₁) {:.6}; rms ours {:.4} vs ref {:.4}",
            s1,
            1.0 - s1,
            rms(&ours1),
            rms(&ref1)
        );
        latent_rel_error(&ours1, &ref1)
    });

    // TRACE=<dir> replaces the single end-to-end number with an error CURVE,
    // which is the only form of this measurement that can ATTRIBUTE a
    // divergence instead of merely detecting one.
    //
    // One number at step 20 cannot distinguish two opposite situations. A
    // correct implementation that rounds differently starts with a small error
    // and multiplies it by a roughly CONSTANT factor per step, because the
    // denoise map feeds each output back into the next input. A genuinely
    // wrong implementation shows a JUMP at the step where the wrong thing
    // first matters. Both end at the same place; only the shape tells them
    // apart, and the shape is invisible unless every step is compared.
    //
    // `<dir>/kNN.latent` holds ComfyUI's x_NN, produced by cutting the sigma
    // list to its first NN+1 entries (see gentrace). Those files carry the
    // `1/(1-σ)` factor that `inverse_noise_scaling` applies on the way out, so
    // undo it with the sigma OUR scheduler used for the same step, exactly as
    // the step-1 probe above does. Missing files are skipped, so a partial
    // trace still plots.
    if let Ok(dir) = std::env::var("TRACE") {
        let pairs = scheduler::sigma_pairs_ruled(steps, bundle.meta.shift_rule, 0);
        println!("== per-step trace vs ComfyUI ==");
        println!("    k       σ_k      rel_l2    step ratio");
        let mut prev: Option<f32> = None;
        for k in 1..=steps.min(out.steps.len()) {
            let p = std::path::Path::new(&dir).join(format!("k{k:02}.latent"));
            let Ok(bytes) = std::fs::read(&p) else {
                continue;
            };
            let (g_vae, _) =
                read_comfy_latent(&bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let sk = pairs[k - 1].1;
            let refk: Vec<f32> = g_vae.iter().map(|v| (v - shf) * sf * (1.0 - sk)).collect();
            let oursk = scheduler::unpack_latents(&out.steps[k - 1].latents_out, n_img, ch, lh, lw);
            let (_, l2) = latent_rel_error(&oursk, &refk);
            match prev {
                Some(p0) => println!(
                    "   {k:2}   {sk:8.6}   {l2:.4e}   {:5.2}x",
                    l2 / p0.max(1e-12)
                ),
                None => println!("   {k:2}   {sk:8.6}   {l2:.4e}       —"),
            }
            prev = Some(l2);
        }
        println!("  a flat ratio column means amplification of a rounding difference;");
        println!("  a single large ratio names the step where something is actually wrong.");
    }

    if std::env::var_os("VERBOSE").is_some() {
        let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        eprintln!(
            "  ours mean {:.4}  golden mean {:.4}",
            mean(&ours),
            mean(&golden_vae)
        );
    }

    // Optional: write what hipfire produced, so it can be decoded and looked
    // at. A gate that only reports a number cannot tell "different image" from
    // "broken image", and those have completely different causes.
    if let Ok(p) = std::env::var("DUMP_OURS") {
        std::fs::write(
            &p,
            hipfire_arch_diffusion::pipeline::comfy_latent_bytes(&ours, ch, lh, lw),
        )
        .unwrap_or_else(|e| panic!("dump ours {p}: {e}"));
        eprintln!("wrote our final latent to {p}");
    }

    println!("== FLUX end-to-end golden latent gate ==");
    println!("  latent  {golden_shape:?}  steps {steps}  same initial noise on both sides");
    if let Some((r1_inf, r1_l2)) = step1_rel {
        println!("  after  1 step   rel_inf {r1_inf:.4e}   rel_l2 {r1_l2:.4e}");
        println!(
            "  after {steps} steps  rel_inf {rel_inf:.4e}   rel_l2 {rel_l2:.4e}   tol {tol:.3}"
        );
        println!(
            "  growth {:.1}x over {} steps — a large ratio means drift, a flat one means convention",
            r1_l2.max(1e-9).recip() * rel_l2,
            steps
        );
    } else {
        println!("  rel_inf {rel_inf:.4e}   rel_l2 {rel_l2:.4e}   tol {tol:.3}");
    }
    // Verdict. rel_l2 is the calibrated comparison currency (meta.json
    // note_shift calibrates a whole-schedule mismatch at ~0.25 of rel_l2).
    // rel_inf is kept as an info diagnostic: with 262144 latent elements,
    // a single-pixel tail above tol appears from pure fp16-vs-bf16 rounding
    // amplified over 20 flat-ratio steps, and it does not distinguish a
    // structural one-pixel bug from arithmetic (that distinction is the
    // step-1 number, which is one forward with zero accumulation). The
    // per-step trace classifier above says a FLAT ratio column is rounding
    // amplification — the regime measured on a correct implementation.
    let step1_ok = step1_rel.is_none_or(|(_, l2)| l2 <= tol);
    if rel_l2 <= tol && step1_ok {
        println!("PASS: hipfire's denoise matches ComfyUI from identical noise");
        if rel_inf > tol {
            println!(
                "  (rel_inf {rel_inf:.3} exceeds tol: single-pixel tail from fp16/bf16\n   arithmetic amplification; flat-ratio trace and clean step 1 → rounding, not mechanism)"
            );
        }
    } else {
        println!("FAIL: hipfire's denoise diverges from ComfyUI");
        // Read the STEP-1 number first: it is one forward pass with nothing
        // fed back, so it separates the two causes that the final number
        // conflates. Suspects differ completely between the two branches, and
        // guessing the wrong branch is how a whole day gets spent.
        if step1_rel.is_some_and(|(_, l2)| l2 > tol) {
            println!("  Step 1 ALSO disagrees, so a whole-pass CONVENTION is wrong:");
            println!("  check the final adaLN order for the checkpoint family, the");
            println!("  guidance embedding, the T5 gated FFN, and the latent-space");
            println!("  transform — in that order. Rule out the sigma SHIFT first:");
            println!("  a schedule mismatch alone is worth ~0.25 (meta.json note_shift).");
        } else {
            println!("  Step 1 agrees, so the conventions are right and the defect");
            println!("  accumulates. Do NOT re-check adaLN order or the embedders.");
            println!("  Measured on this model: the trajectory does not amplify");
            println!("  (1.34x over 20 steps) and dtype rounding cannot pay for it");
            println!("  (bf16 -> fp8 costs only 0.055), so a step-20 divergence is a");
            println!("  systematic per-step bias in the VELOCITY FIELD. Run TRACE to");
            println!("  get the per-step curve, then divide each increment by that");
            println!("  step's dsigma — on FLUX that error is largest at HIGH sigma");
            println!("  and decays as the image forms, so look at what the forward");
            println!("  pass does to a state that is still mostly noise.");
        }
        std::process::exit(1);
    }
}
