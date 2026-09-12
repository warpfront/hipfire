# Image generation (FLUX.1 / FLUX.2 Klein) — local test guide

hipfire generates images from two diffusion families on AMD RDNA GPUs:

| Family | `arch_id` | Conditioning | Default steps | Device memory |
|---|---:|---|---:|---|
| FLUX.1 schnell / dev | 40 | T5-XXL + CLIP-L | 4 / 28 | about 24 GB for the f16 weights (T5 falls back to the host when it cannot upload) |
| FLUX.2 Klein 4B / 9B | 45 | Qwen3 | 4 | about 13 GB / 24 GB |

Both run through the same daemon message (`img_generate`), the same HTTP
endpoint (`POST /v1/images/generations`) and the same CLI (`hipfire img`).
Klein also edits: reference images condition every denoise step without
being denoised.

The daemon loads **HFQ component packs only**. A diffusers pipe directory is
the input of the packer (`hipfire-quantize --flux-pipe`), never a model path.

This guide is the shortest path to a first image on a developer box. The
crate-level details (modules, A/B env flags, gate bars) are in
[`crates/hipfire-arch-diffusion/README.md`](../crates/hipfire-arch-diffusion/README.md).

## Quick start

Five commands from a clean checkout to a first image (Klein 4B, the smaller
model; about 20 minutes, most of it the download):

```bash
cargo build --release
export HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon
huggingface-cli download black-forest-labs/FLUX.2-klein-4B --local-dir ~/models/flux-klein-pipe
./target/release/hipfire-quantize --flux-pipe ~/models/flux-klein-pipe --output ~/models/flux-klein.hfq
./target/release/hipfire img ~/models/flux-klein-transformer.hfq "a red bicycle leaning on a stone wall, photo" --out bike.png
```

Edit that image with a reference (Klein only; `--image` goes before the
prompt):

```bash
./target/release/hipfire img --image bike.png ~/models/flux-klein-transformer.hfq "make the bicycle blue" --out bike-blue.png
```

The sections below explain each step, the FLUX.1 path, the HTTP API, the
tests and what to do when something fails.

## 1. Requirements

- An RDNA3 or RDNA3.5 GPU with the ROCm HIP runtime installed. hipfire
  `dlopen`s HIP at run time, so the build needs no ROCm. RDNA4 (gfx12) is
  refused at load: the FLUX kernels use the gfx11 wave32 WMMA intrinsics.
  Measured: gfx1151 (Radeon 8060S, unified memory; FLUX.1 schnell and
  Klein), gfx1150 (Radeon 890M, Klein only), gfx1100 (RX 7900 XT: kernels
  pass, FLUX.1 schnell does not fit in 24 GB — see § 10).
- Rust stable and disk for the pipe plus its packs: about 90 GB for FLUX.1
  schnell (54 GB pipe + 34 GB packs), 30 GB for Klein 4B.
- `huggingface-cli` (or any HF download tool) for the weights.

## 2. Build

```bash
cargo build --release
# target/release/hipfire, target/release/daemon, target/release/hipfire-quantize
```

The CLI prefers an installed daemon under `~/.hipfire/bin`. Point it at the
fresh build for every command below:

```bash
export HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon
```

## 3. Get the weights

Download a pipe root in the diffusers layout (`transformer/`,
`text_encoder*/`, `tokenizer*/`, `vae/`, `scheduler/`):

```bash
# FLUX.1 schnell (Apache-2.0):
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ~/models/flux-schnell-pipe
# FLUX.2 Klein 4B (Apache-2.0; the 9B is non-commercial):
huggingface-cli download black-forest-labs/FLUX.2-klein-4B --local-dir ~/models/flux-klein-pipe
```

FLUX.1 dev is gated and non-commercial; the same commands work on its pipe.

## 4. Pack

The packer is CPU-only. It converts the weights to F16 (bias, scale and
norm statistics stay F32) and embeds each component's config, the scheduler
config and the tokenizers in the pack headers, so the pipe directory is not
needed after this step. Measured on a 32-core box: about 25 minutes for
FLUX.1 schnell, 12 minutes for Klein 4B.

```bash
./target/release/hipfire-quantize --flux-pipe ~/models/flux-schnell-pipe --output ~/models/flux-schnell.hfq
# -> flux-schnell-transformer.hfq (40), -t5.hfq (41), -clip.hfq (42), -vae.hfq (43)
./target/release/hipfire-quantize --flux-pipe ~/models/flux-klein-pipe --output ~/models/flux-klein.hfq
# -> flux-klein-transformer.hfq (45), -qwen3.hfq (46), -vae.hfq (43)
```

The trunk pack is the model path. The sidecar packs are found next to it by
name (`<base>-t5.hfq` and so on, or the shared `t5-xxl.hfq` / `clip-l.hfq` /
`qwen3.hfq` / `vae.hfq`). Details: [`QUANTIZE.md`](QUANTIZE.md).

## 5. First image

The first run compiles the kernels for your GPU and uploads the weights, so
expect it to take longer than the numbers below.

```bash
./target/release/hipfire img ~/models/flux-schnell-transformer.hfq \
  "a tiny lighthouse on a rock at sunset, photo" \
  --width 512 --height 512 --steps 4 --seed 0 --out lighthouse.png --json
```

The JSON result names the file, size, seed, steps and wall time. The same
seed gives a byte-identical PNG on the same GPU.

Klein txt2img and reference edit (`--image` must come before the prompt):

```bash
./target/release/hipfire img ~/models/flux-klein-transformer.hfq \
  "a red bicycle leaning on a stone wall, photo" --width 512 --height 512 --out bike.png
./target/release/hipfire img --image bike.png ~/models/flux-klein-transformer.hfq \
  "make the bicycle blue" --out bike-blue.png
```

Measured on gfx1151 at 512x512, 4 steps, after warm-up:

| Model | Wall time |
|---|---:|
| FLUX.1 schnell | 17.5 s |
| FLUX.2 Klein 4B | 7.1 s |

## 6. HTTP

```bash
./target/release/hipfire serve 127.0.0.1 11580 --model ~/models/flux-schnell-transformer.hfq --idle-timeout 0 &
curl -s -X POST http://127.0.0.1:11580/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a tiny lighthouse on a rock at sunset","size":"512x512","steps":4,"seed":0,"response_format":"b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 -d > lighthouse.png
```

`n` must be 1 and `response_format` must be `b64_json`. Malformed bodies
return 400 with a message; a text model on this endpoint, or a diffusion model
on `/v1/chat/completions`, is refused.

Reference edit over HTTP is the OpenAI multipart route (Klein pack loaded):

```bash
curl -s -X POST http://127.0.0.1:11580/v1/images/edits \
  -F image=.png -F prompt="make the bicycle blue" -F steps=4 -F seed=0 \
  | jq -r .data[0].b64_json | base64 -d > bike-blue.png
```

The image bytes travel in the request. Neither route accepts a server path,
so an HTTP client cannot make the daemon read a file. Field reference:
[`SERVE.md`](SERVE.md).

## 7. Tests and gates

No GPU, no weights:

```bash
cargo test -p hipfire-arch-diffusion -p hipfire-loader -p hipfire-quantize --lib
```

Kernel parity gates on your GPU (each one compares against a CPU reference
and prints `PASS` per case):

```bash
for e in test_rope_2d_flux test_modulate_f32 test_layernorm_modulate_parity \
         test_qk_rmsnorm_rope_parity test_copy_rows_strided_f32_parity \
         test_gemm_f16_x_f16_wmma_lds_parity test_gemm_wide_lds_parity \
         test_gemm_epilogue_parity test_attention_flux_vt_parity \
         test_attention_text_gqa test_vae_lds; do
  cargo run --release --features lab -p rdna-compute --example $e || break
done
```

Model-level gates read the pipe directory (they need the raw weights for
the CPU oracle) and a GPU; they live under
`crates/hipfire-arch-diffusion/examples/` and are listed with their purpose
in that crate's `Cargo.toml`. The whole-denoise-loop gates compare against
the committed ComfyUI golden latents:

```bash
cargo run --release --features lab -p hipfire-arch-diffusion --example gpu_flux_golden_latent -- \
  ~/models/flux-dev-pipe crates/hipfire-arch-diffusion/tests/fixtures/flux-golden
cargo run --release --features lab -p hipfire-arch-diffusion --example gpu_klein_golden_latent -- \
  ~/models/flux-klein-pipe crates/hipfire-arch-diffusion/tests/fixtures/klein-golden/4b
```

The serve harness runs the HTTP path end to end, with seeded within- and
cross-process byte parity and the fail-closed 400 cases:

```bash
python3 scripts/serve_harness.py --mode images --model ~/models/flux-klein-transformer.hfq \
  --port 11530 --img-width 512 --img-height 512 --img-steps 4
```

## 8. Profiling knobs

`HIPFIRE_IMG_PROFILE=1` prints per-stage wall time; `HIPFIRE_PROFILE=1` adds
a per-kernel-family table per denoise step. Every other `HIPFIRE_FLUX_*` /
`HIPFIRE_VAE_*` flag is an A/B knob whose default is the measured winner;
the table in the crate README lists them.

## 9. Known limits

- One image per request; `euler` (flow-match) is the only sampler.
- FLUX.1 runs f16 activations by default (`HIPFIRE_FLUX_F16_ACT=0` for f32).
- Width and height must be multiples of 16: the VAE factor is 8 and the
  latent is packed in 2x2 patches. Klein never upscales a reference image;
  it is area-capped at 1 MP.
- Noise is hipfire's own seeded generator, not torch-compatible: the same
  seed does not reproduce a diffusers or ComfyUI image. The golden-latent
  gates start from the fixture's captured init latent for that reason.
- The `flux.schnell:1` registry entry points at the public `elphil/flux` Hub
  repo (Apache-2.0); `hipfire pull flux.schnell:1` fetches the packs directly.

## 10. Troubleshooting

- `... is a diffusers pipe directory, which is not loadable` — pack it first
  (section 4) and pass the trunk pack.
- `... needs the t5 and clip sidecar packs next to it` — the sidecars are
  looked up next to the trunk by name; keep the packer's output together.
- `FATAL: hipfire daemon already running` — one daemon per `$HOME/.hipfire`.
  Stop the other one (`hipfire stop`) or run the test with a private
  `HOME=/tmp/hipfire-home`.
- A very slow first image — kernel JIT plus the weight upload. Run once more
  before you read any timing.
- Out of device memory on FLUX.1 — the T5 and CLIP encoders fall back to
  the host (slow, correct), but the transformer itself needs about 24.3 GB
  resident (23.8 GB f16 weights + row padding + activations), so a 24 GB
  card (RX 7900 XT/XTX) fails at the first activation alloc. FLUX.1 needs a
  unified-memory APU (Strix Halo) or a 32 GB+ RDNA3 card; Klein 4B fits.
