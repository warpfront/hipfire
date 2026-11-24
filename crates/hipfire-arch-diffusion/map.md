# hipfire-arch-diffusion — map

> **Status:** `active` — FLUX.1 (arch 40) and FLUX.2 Klein (arch 45) txt2img on the GPU, Klein reference-image edit; served through `img_generate`.
> **Layer:** Arch component — hangs off `hipfire-loader` via `Carrier` (`arch_id` 40/45); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Latent image diffusion trunks and their conditioning stack: FLUX.1 MMDiT
(T5-XXL + CLIP-L, AutoencoderKL) and FLUX.2 Klein (Qwen3 taps, 32-channel
VAE). CPU f32 references are the parity oracles for the GPU forwards. Not a
chat model: implements `ArchModel` + `Carrier`, deliberately not
`Architecture`.

## Gotchas

- `axes_dim` must sum to `head_dim`; the config parser enforces it, and the 2D
  RoPE budget in `flux.rs` depends on it.
- The manifest is generated from config — correct a tensor name/shape there,
  not in the loader.
- `VERIFY-FIXTURE` markers in `src/flux.rs`: conditioning assembly, modulation
  chunk order, and 2D RoPE axis→position mapping are conventions that only
  a golden capture can pin. Do not "fix" them from memory.
- A ComfyUI FLUX.2 `.latent` is already BatchNorm-normalized; never
  re-normalize it on load.
- FLUX.2 text tokens carry RoPE ids `(0, 0, 0, l)`, not `(0, l, 0, 0)`.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
