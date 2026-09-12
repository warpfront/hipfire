# hipfire-arch-diffusion — FLUX latent diffusion

Latent image diffusion architectures for hipfire: the FLUX.1 MMDiT trunk
(arch 40) and the FLUX.2 Klein trunk (arch 45), with their conditioning
encoders (T5-XXL + CLIP-L, or Qwen3) and the VAE. The crate implements
`ArchModel` and is loaded by `FluxDiffusionCarrier` in `hipfire-loader` from
the HFQ component packs that `hipfire-quantize --flux-pipe` writes; the
daemon serves it through `img_generate`, HTTP `/v1/images/generations` and the
`hipfire img` CLI. Component ids: `docs/architecture-ids.md`.

## Why no `Architecture` impl

The `hipfire_runtime::arch::Architecture` trait is token-stream machinery
(tokenizer, vocab, KV, spec decode). A diffusion trunk is a latent-step
optimizer, so it is a **component**: loadable, refused by text `generate`.
This crate implements `ArchModel` and deliberately not `Architecture`.

## Modules

- `src/config.rs` — `FluxDiffusionConfig`: the FLUX.1 / FLUX.2 transformer
  config surface, with hard shape invariants.
- `src/manifest.rs` — the tensor inventory. **The single place to correct**
  when a checkpoint disagrees; the loader validates against it first.
- `src/flux.rs` — dependency-free f32 MMDiT reference + host weight read.
  `VERIFY-FIXTURE` markers name the convention-level details only a golden
  capture can pin.
- `src/flux_gpu.rs` — GPU-resident weights (streamed f16 upload, `K+64` row
  pitch) and the GPU forward for both families.
- `src/t5.rs`, `src/clip.rs`, `src/qwen3.rs` (+ `*_gpu.rs`) — conditioning
  encoders, host reference and GPU path.
- `src/vae.rs`, `src/vae_gpu.rs` — AutoencoderKL decoder (and the FLUX.2
  encoder for reference images).
- `src/scheduler.rs` — Flow-Match Euler schedule, latent pack/unpack, seeded
  noise (`seeded_gaussian`: deterministic per seed, NOT torch-compatible).
- `src/pipeline.rs` — pipe load (HFQ packs for serving; the diffusers dir
  for the gates), conditioning
  cache, the txt2img / reference-edit request path, PNG postprocess.
- `src/refimg.rs`, `src/klein_prompt.rs` — FLUX.2 reference-image
  preprocessing and the Klein prompt template.
- `src/arch_model.rs` — the `ArchModel` view the loader stores.

## Build / test

```bash
cargo build -p hipfire-arch-diffusion
cargo test  -p hipfire-arch-diffusion
```

The unit tests need no GPU and no weights. The gates under `--features lab`
need a pipe directory and, for the GPU ones, a device:

```bash
# CPU fixture gate on the committed tiny-pipe capture:
cargo run --release -p hipfire-arch-diffusion --features lab --example flux_pipeline_parity \
  --pipe <tiny-flux-pipe> --golden crates/hipfire-arch-diffusion/tests/fixtures/tiny-pipeline
# GPU txt2img vs CPU txt2img on the same seed (byte-identical PNG):
cargo run --release -p hipfire-arch-diffusion --features lab --example gpu_pipeline_parity -- <tiny-flux-pipe>
# Real weights: block parity, then the whole denoise loop vs the ComfyUI golden latent:
cargo run --release -p hipfire-arch-diffusion --features lab --example gpu_flux_block_parity -- <flux1-dev.safetensors>
cargo run --release -p hipfire-arch-diffusion --features lab --example gpu_flux_golden_latent -- <pipe-dir> \
  crates/hipfire-arch-diffusion/tests/fixtures/flux-golden
```

`tests/fixtures/` holds the diffusers capture of the tiny pipe
(`tiny-pipeline/`, produced by `examples/capture_diffusers_pipeline.py`), the
ComfyUI golden latents for FLUX.1 (`flux-golden/`) and FLUX.2 Klein 4B / 9B /
4B-edit (`klein-golden/`, each with the ComfyUI graph that regenerates it), and
the Klein key lists (`klein/`). Each `meta.json` records how the fixture was
made and what its gate asserts.

## Serving

The daemon loads HFQ component packs only. Pack a diffusers pipe once
(`docs/QUANTIZE.md`), then point every command at the trunk pack; the sidecar
packs are found next to it by name.

```bash
hipfire-quantize --flux-pipe <pipe_dir> --output flux-schnell.hfq
hipfire img flux-schnell-transformer.hfq "a tiny cat sitting on a tiny table" --steps 4 --seed 0 --out cat.png
# HTTP:
hipfire serve 127.0.0.1 11580 --model flux-schnell-transformer.hfq &
curl -s -X POST http://127.0.0.1:11580/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a tiny cat","n":1,"size":"1024x1024","steps":4,"seed":0}'
# Harness:
python3 scripts/serve_harness.py --mode images --model flux-schnell-transformer.hfq --port 11530
```

Width and height on the wire are pixel dims; the pipeline validates
divisibility by the VAE scale and even latent dims. `backend` is `gpu` when a
GPU initializes, else `cpu`; an explicit `backend` always wins. The host
`t5::encode` / `clip::encode` reference is the f32 oracle and the fallback when
an encoder cannot be uploaded.

## FLUX.2 Klein (arch 45)

`FluxDiffusionConfig` recognises `_class_name: "Flux2Transformer2DModel"` (or
`model_type: "flux2"`) and maps it to arch 45, daemon name `flux2_mmdit`,
loaded by the same carrier as arch 40. FLUX.2 is a separate forward body:
bias-free linears, one shared modulation vector, SwiGLU MLPs, fused
single-block projections and 4-axis id-table RoPE. Conditioning is Qwen3
(hidden-state taps 9/18/27), the VAE is the 32-channel FLUX.2 decoder whose
latent statistics live in an internal BatchNorm (a ComfyUI FLUX.2 `.latent` is
already normalized and must not be re-normalized on load), and the schedule
is the empirical sigma shift (`ShiftRule::Empirical`). Supported checkpoints:
Klein 4B and 9B.

Klein adds **reference-image editing**: `img_generate` takes an `images`
field of up to four PNG/JPEG images as base64 bytes. `hipfire img --image`
reads the files and sends their bytes; HTTP takes them as multipart file parts
on `/v1/images/edits`. The daemon never opens a client-named path. Each reference is decoded, area-capped at 1 MP and floored to a multiple
of 16 (`refimg::target_size`; hipfire never upscales), VAE-encoded, and
concatenated onto the image token stream, where it conditions every step
without being denoised. The joint sequence is refused above
`pipeline::MAX_ROUTE_TOKENS` (32768). With no `images` the request is plain
txt2img at 1024x1024 by default; otherwise the output defaults to the
reference's size.

## A/B env flags (FLUX GPU path)

Developer knobs, not product configuration: every default is the measured
winner, and each flag exists so the alternative can be benched in one session
against one binary. The accessor's doc comment is the authority on semantics;
this table is the index. The table describes the FLUX.1 path; on a FLUX.2 pipe
`HIPFIRE_FLUX_F16_ACT` is ignored (that forward is f32-activation only) and
`HIPFIRE_FLUX_GUIDANCE` is unreachable (Klein has no guidance embedder).

| flag | default | effect when set |
|---|---|---|
| `HIPFIRE_FLUX_F16_ACT` | on | `=0` runs the f32 activation path. Read once per process — the upload path and the forward must agree, because the f16 path splits `single_blocks.*.linear2.weight` into `.w_attn`/`.w_mlp` at upload and the f32 path keeps the fused tensor. |
| `HIPFIRE_FLUX_GEMM_PIPE` | per-arch table (`Gpu::LDS_PIPE_ON`: gfx1151 on, others off) | `=1` forces the software-pipelined LDS GEMM main loop, `=0` the plain one. Both arms are compiled into the same module and are bit-exact, so an A/B measures only the main loop. A tile with no compiled `_p` twin always gets the plain loop. |
| `HIPFIRE_FLUX_MOD_GEMV` | on | `=0` restores the per-block `mod_linear`/`linear` GEMM route. On, the 76 batch-1 modulation linears per step run as GEMVs into one `ModAll` buffer instead of one 128-row WMMA macro-tile each. Read once per process. |
| `HIPFIRE_FLUX_ATTN` | measured per-arch route (`v2` on gfx1150/gfx1151/gfx1100, else `vt`) | `vt`, `vtk`, `v2` or `v5` pins the FLUX attention route. `v5` is F32-only; asking for F16 through it errors. Any other value is an error, so a typo cannot silently bench the default twice. |
| `HIPFIRE_FLUX_GEMM_LDS` | on | `=0` forces the old 16-step WMMA GEMM instead of the LDS-staged 128x128 macro-tile kernel. The LDS kernel needs `K % 64 == 0` (true of every FLUX.1-dev linear); a ragged K falls back regardless. |
| `HIPFIRE_FLUX_GEMM_WIDE` | on | `=0` pins the fixed 128x128 tile instead of the per-arch measured macro-tile (`_auto`). The per-arch winners differ — gfx1100's best tile is a 27% loss on gfx1151 — so there is no single tile. |
| `HIPFIRE_FLUX_WPAD` | `64` | row pitch, in elements, added to every WMMA-GEMM-consumed FLUX weight at upload (must be a multiple of 16); `=0` uploads everything packed at `K`. Modulation weights, any `K` not a multiple of 64, and the `HIPFIRE_FLUX_GEMM_LDS=0`/`HIPFIRE_FLUX_GEMM_WIDE=0` fallback routes stay packed regardless. Bit-exact with the packed layout; gfx1151 measured 2.677 s/step padded vs 3.275 s/step packed on a 3-step probe. |
| `HIPFIRE_T5_GPU` | on | `=0` sends T5 and CLIP conditioning back to the host `t5::encode`/`clip::encode` f32 reference. That is the numeric-oracle escape hatch, not a normal route: the host encode is 54.9 s per prompt at real geometry. |
| `HIPFIRE_IMG_COND_CACHE` | on | `=0` disables the conditioning cache completely — nothing is stored, conditioning takes the owned path and is freed after the denoise loop. Read once at pipe load. |
| `HIPFIRE_IMG_PROFILE` | off | any value but `0` prints per-stage wall time on the GPU txt2img path. |
| `HIPFIRE_PROFILE` | off | when set, the GPU txt2img path additionally prints a per-kernel-family table for each denoise step, plus a `gap` line (sum of the families vs the step wall time). |
| `HIPFIRE_VAE_PROFILE` | off | any value but `0` prints per-kernel call counts and wall time for the GPU VAE decode. |
| `HIPFIRE_VAE_CONV` | GEMM route | `=direct` pins the naive one-thread-per-output conv kernel and uploads the conv weights f32 to match. The GEMM route is also skipped automatically when some conv's K is not a multiple of 16. |
| `HIPFIRE_VAE_IM2COL_MAP` | `lds` | `c` selects the channel-fastest scalar gather, `p` the pixel-fastest one (~6x worse). The default stages a halo patch through LDS so each input element crosses DRAM once instead of nine times. |
| `HIPFIRE_VAE_IM2COL_TILE` | `64:8:16` (c_tile clamped to divide `c_in`) | `<c_tile>:<th>:<tw>` overrides the im2col workgroup tile. LDS cost is `c_tile*(th+2)*(tw+2)*2` bytes. A malformed value is an error, not a silent fallback. |
| `HIPFIRE_VAE_TRANSPOSE` | naive scatter | `=tiled` selects the 32x32 LDS-tile transpose. The default won on the VAE's skinny shapes (347 vs 524 ms over a decode). |
| `HIPFIRE_VAE_FUSE_NORM` | on | `=0` restores a separate GroupNorm + SiLU pair instead of folding the SiLU into the norm kernel. |
