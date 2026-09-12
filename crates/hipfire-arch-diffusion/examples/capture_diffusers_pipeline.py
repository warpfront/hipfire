#!/usr/bin/env python3
"""Capture a golden txt2img trace from diffusers `FluxPipeline` (tiny fixture).

Fixture capture: runs the REAL diffusers FluxPipeline
(diffusers 0.40) on the tiny diffusers-format pipe fixture, then writes
`golden.json` + `golden.png` containing everything the Rust CPU pipeline
needs to prove parity:

- conditioning: T5 prompt embeds (txt), CLIP pooled (vec), token ids + masks
- geometry: packed latent grid, `latent_image_ids`, `text_ids`, scheduler
  sigmas/timesteps
- denoise: per-step (`timestep`, `latents_in`, `noise_pred`, `latents_out`)
- decode: unpacked+scaled latents (VAE input) and the decoded image tensor
  (VAE output, pre-PIL), plus the postprocessed `golden.png`
- per-encoder intermediate dumps (T5 per-layer, CLIP per-layer, VAE decoder
  per-block) for unit-level debugging of the Rust reimplementation
- a cross-check: the manual loop's decoded tensor vs a full `pipe(...)` run
  (must agree to 1e-5, else the capture is inconsistent and must not be used)

The pipe weights stay OUTSIDE the repo (external fixture, like
tiny-flux-trace): `PIPE_DIR` defaults to
`~/.cache/trace-assets/tiny-flux-pipe`. Component safetensors + config md5s
are recorded in the golden so a drifted fixture is detectable.

Usage: python3 examples/capture_diffusers_pipeline.py [OUT_DIR]
Env: PIPEFLUX_PIPE_DIR to override the pipe location.
Dependency: .trace-venv (torch, diffusers>=0.40, transformers>=4.41).
"""

import base64
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

PIPE_DIR = os.environ.get(
    "PIPEFLUX_PIPE_DIR", os.path.expanduser("~/.cache/trace-assets/tiny-flux-pipe")
)

PROMPT = "a tiny cat sitting on a tiny table"
SEED = 0x0D510
STEPS = 2
HEIGHT = 32
WIDTH = 32
MAX_SEQ = 64


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flat(v: "torch.Tensor") -> list[float]:
    return v.detach().float().reshape(-1).tolist()


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tiny-pipeline-golden"
    os.makedirs(out_dir, exist_ok=True)

    from diffusers import FluxPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    pipe = FluxPipeline.from_pretrained(
        PIPE_DIR, local_files_only=True, torch_dtype=torch.float32
    )
    print("pipe loaded, vae_scale_factor =", pipe.vae_scale_factor)

    # ── component md5s (fixture provenance) ─────────────────────────────
    comp_md5s = {}
    for sub in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        d = os.path.join(PIPE_DIR, sub)
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                comp_md5s[f"{sub}/{fn}"] = md5(p)

    # token ids + masks straight from the tokenizers (pin tokenization too)
    tok = pipe.tokenizer_2(PROMPT, padding="max_length", max_length=MAX_SEQ, truncation=True)
    t5_ids = tok["input_ids"]
    t5_mask = tok["attention_mask"]
    tok1 = pipe.tokenizer(PROMPT, padding="max_length", max_length=77, truncation=True)
    clip_ids = tok1["input_ids"]
    clip_mask = tok1["attention_mask"]

    # ── latent init (same generator semantics the pipeline uses) ────────
    num_channels_latents = pipe.transformer.config.in_channels // 4
    generator = torch.Generator().manual_seed(SEED)
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, HEIGHT, WIDTH, torch.float32, "cpu", generator
    )
    print("packed latents", tuple(latents.shape), "img_ids", tuple(latent_image_ids.shape))

    # ── timesteps (pipeline's own _prepare_timesteps snippet) ───────────
    sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    image_seq_len = latents.shape[1]
    mu = (
        (1.15 - 0.5) / (4096 - 256) * (image_seq_len - 256) + 0.5
    )  # calculate_shift inlined
    from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, STEPS, "cpu", sigmas=sigmas, mu=mu
    )
    sched_sigmas = pipe.scheduler.sigmas.tolist()
    print("mu", mu, "timesteps", [float(t) for t in timesteps])

    # ── T5 / CLIP per-layer intermediates (debug targets for unit tests).
    # Hooks are registered BEFORE encode_prompt so the real encode fills them.
    t5_hook_out: dict[str, torch.Tensor] = {}
    clip_hook_out: dict[str, torch.Tensor] = {}

    def hook_into(d: dict, prefix: str, mod, name: str):
        def fn(_m, _i, o):
            d[prefix + name] = o[0] if isinstance(o, tuple) else o
        return fn

    for i, b in enumerate(pipe.text_encoder_2.encoder.block):
        b.register_forward_hook(hook_into(t5_hook_out, "t5_layer_", b, str(i)))
    for i, b in enumerate(pipe.text_encoder.encoder.layers):
        b.register_forward_hook(hook_into(clip_hook_out, "clip_layer_", b, str(i)))
    vae_hook_out: dict[str, torch.Tensor] = {}
    v = pipe.vae
    v.decoder.conv_in.register_forward_hook(hook_into(vae_hook_out, "vae_conv_in", v.decoder, ""))
    for i, b in enumerate(v.decoder.up_blocks):
        b.register_forward_hook(hook_into(vae_hook_out, "vae_up_", b, str(i)))
    v.decoder.mid_block.register_forward_hook(hook_into(vae_hook_out, "vae_mid", v.decoder, ""))

    # ── conditioning (diffusers semantics: vec=CLIP pooled, txt=T5 hidden).
    # Runs AFTER the T5/CLIP hooks above so per-layer outputs get captured.
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=PROMPT,
        max_sequence_length=MAX_SEQ,
        device="cpu",
    )
    clip_pooled = pooled_prompt_embeds  # (1, 32), vec conditioning
    print("prompt_embeds", tuple(prompt_embeds.shape), "pooled", tuple(pooled_prompt_embeds.shape))

    # ── denoise loop, mirroring pipeline_flux.py exactly ────────────────
    step_records = []
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=None,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        step_records.append(
            {
                "t_model": (t / 1000).item(),
                "t_sched": t.item(),
                "latents_in": flat(latents),
                "noise_pred": flat(noise_pred),
            }
        )
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        step_records[-1]["latents_out"] = flat(latents)
        print(f"step {i}: t={t.item():8.2f} latents {tuple(latents.shape)}")

    # ── unpack + scaling + VAE decode (pipeline's tail) ─────────────────
    latents_unpacked = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    latents_scaled = (latents_unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents_scaled, return_dict=False)[0]
    print("decoded", tuple(image.shape), "min/max", float(image.min()), float(image.max()))

    # ── cross-check vs a full pipeline run (same seed → same latents) ───
    gen2 = torch.Generator().manual_seed(SEED)
    full = pipe(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=0.0,
        generator=gen2,
        max_sequence_length=MAX_SEQ,
        output_type="pt",
    ).images
    # pipeline postprocess for output_type="pt" denormalizes to [0,1]
    manual_denorm = torch.clamp((image.detach() + 1.0) / 2.0, 0.0, 1.0)
    diff = float((manual_denorm - full).abs().max())
    print("manual-vs-full max abs diff (denormalized):", diff)
    assert diff < 1e-5, "manual loop disagrees with full pipeline — capture invalid"

    img = pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]
    img.save(os.path.join(out_dir, "golden.png"))

    # keep vae per-block dumps (hooks fired on that decode)
    vae_layers = {k: flat(v) for k, v in vae_hook_out.items()}
    t5_layers = {k: flat(v) for k, v in t5_hook_out.items()}
    clip_layers = {k: flat(v) for k, v in clip_hook_out.items()}

    golden = {
        "source": "diffusers FluxPipeline (tiny pipe fixture), fp32 CPU",
        "diffusers_version": __import__("diffusers").__version__,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "steps": STEPS,
        "max_seq": MAX_SEQ,
        "vae_scale_factor": pipe.vae_scale_factor,
        "mu": mu,
        "component_md5s": comp_md5s,
        "inputs": {
            "t5_ids": t5_ids,
            "t5_mask": t5_mask,
            "clip_ids": clip_ids,
            "clip_mask": clip_mask,
            "txt": flat(prompt_embeds),
            "vec": flat(clip_pooled),
            "text_ids": flat(text_ids),
            "latent_image_ids": flat(latent_image_ids),
            "scheduler": {
                "num_train_timesteps": pipe.scheduler.config.num_train_timesteps,
                "shift": pipe.scheduler.config.shift,
                "use_dynamic_shifting": pipe.scheduler.config.use_dynamic_shifting,
                "base_image_seq_len": pipe.scheduler.config.base_image_seq_len,
                "max_image_seq_len": pipe.scheduler.config.max_image_seq_len,
                "base_shift": pipe.scheduler.config.base_shift,
                "max_shift": pipe.scheduler.config.max_shift,
            },
            "sigmas": sched_sigmas,
            "timesteps": [float(t) for t in timesteps],
        },
        "steps": step_records,
        "decode": {
            "latents_unpacked": flat(latents_unpacked),
            "latents_scaled": flat(latents_scaled),
            "image": flat(image),
            "image_shape": list(image.shape),
        },
        "encoder_intermediates": {
            "t5": t5_layers,
            "clip": clip_layers,
            "vae": vae_layers,
        },
    }
    with open(os.path.join(out_dir, "golden.json"), "w") as f:
        json.dump(golden, f, indent=1)

    png_b64 = base64.b64encode(open(os.path.join(out_dir, "golden.png"), "rb").read()).decode()
    print(f"wrote {out_dir}/golden.json + golden.png (b64 {len(png_b64)} bytes)")
    print("final image first pixels:", [round(x, 4) for x in flat(image)[:6]])


if __name__ == "__main__":
    main()