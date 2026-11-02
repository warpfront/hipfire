// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-diffusion: latent image diffusion architectures for hipfire.
//!
//! Scope:
//!
//! 1. **Component contract** — [`FluxDiffusionConfig`] parsed from a
//!    FLUX transformer `config.json` (`src/config.rs`) and
//!    [`arch_model::FluxPipeModel`] implementing the loader's [`ArchModel`]
//!    view (`src/arch_model.rs`). Arch id **40** is the FLUX.1 trunk and
//!    **45** the FLUX.2 Klein trunk; the sidecar ids 41 (T5), 42 (CLIP),
//!    43 (VAE) and 46 (Qwen3) live in the per-component HFQ pack headers
//!    (`docs/architecture-ids.md`).
//! 2. **Tensor manifest** — [`expected_flux_keys`] (`src/manifest.rs`): the
//!    canonical FLUX.1 key list with shapes, validated against the actual
//!    safetensors directory at load time. This is the single place to correct
//!    when a checkpoint disagrees.
//! 3. **CPU block reference** — [`flux`] (`src/flux.rs`): a dependency-free,
//!    obviously-correct f32 reimplementation of the MMDiT double/single block
//!    math (RMSNorm, adaLN-Zero, 2D RoPE, joint attention, GELU-tanh MLP).
//!    It is the parity target for the GPU forward (`src/flux_gpu.rs`).
//! 4. **Full CPU txt2img pipeline** — [`t5`] (T5 encoder conditioning),
//!    [`clip`] (CLIP pooled `vec`), [`scheduler`] (Flow-Match Euler +
//!    latent pack/unpack), [`vae`] (AutoencoderKL decoder), [`pipeline`]
//!    (denoise orchestration + PNG postprocess). Validated against a
//!    diffusers capture of the tiny pipe (≥ 130 dB per stage, byte-identical
//!    PNG) — see `examples/flux_pipeline_parity.rs`. Two
//!    documented diffusers deviations from the shipped BFL architecture are
//!    pinned in this crate (unmasked CLIP conditioning; final-head adaLN
//!    `(scale, shift)` chunk order) — see [`flux::FinalAdaLNOrder`] and
//!    [`clip`].
//! 5. **Qwen3 CPU reference encoder** — [`qwen3`]: the causal-LM text encoder
//!    FLUX.2 Klein conditions on, with hidden-state taps after layers 9, 18,
//!    27 ([`qwen3::KLEIN_TAPS`]) concatenated per token.
//! 6. **Klein prompt template** — [`klein_prompt`]: the ComfyUI chatml
//!    wrapping plus tokenization and right-padding
//!    ([`klein_prompt::encode_klein_prompt`]) that feeds the Qwen3 text encoder.
//! 7. **Reference image preprocessing** — [`refimg`]: decode, area-capped
//!    resize, floor-to-multiple-of-16 snap, and `[-1, 1]` channel-major
//!    mapping for FLUX.2 Klein image-editing conditioning.
//! 8. **FLUX.2 (Klein) pipe** — [`pipeline::load_pipe`] detects the family
//!    from the transformer config and carries the text stack as a
//!    [`pipeline::TextCond`] enum (T5 + CLIP, or the Qwen3 text encoder);
//!    [`pipeline::generate_img_prompt`] is the CPU entry point for both
//!    families and for the Klein reference-image edit path; the GPU route
//!    (`flux_gpu::forward_parts_flux2`, `qwen3_gpu`, the FLUX.2 `vae_gpu`
//!    encoder/decoder) is a separate body, never a FLUX.1 path run against
//!    FLUX.2 weights.
//!
//! Constraint: diffusion trunks are **components, not chat models**. There is
//! deliberately NO [`hipfire_runtime::arch::Architecture`] impl here — that
//! trait's machinery (tokenizer, vocab, KV cache, spec decode, terminal) is
//! token-stream machinery a latent-step optimizer never uses. Arch 40 and
//! 45 must never reach the daemon's text `generate` path.

pub mod arch_model;
pub mod clip;
pub mod clip_gpu;
pub mod config;
pub mod f16_stage;
pub mod flux;
pub mod flux_gpu;
pub mod klein_prompt;
pub mod manifest;
pub mod nn;
pub mod pipeline;
pub mod qwen3;
pub mod qwen3_gpu;
pub mod refimg;
pub mod scheduler;
pub mod t5;
pub mod t5_gpu;
pub mod tokenizer;
pub mod vae;
pub mod vae_gpu;

pub use config::FluxDiffusionConfig;
