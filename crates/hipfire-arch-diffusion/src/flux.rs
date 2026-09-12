// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dependency-free f32 CPU reference for the FLUX.1 MMDiT forward pass,
//! pinned to the ORIGINAL BFL implementation (`black-forest-labs/flux`, commit
//! `87f6fff`, Sept 2024 — the code that produced the shipped FLUX.1-schnell /
//! FLUX.1-dev weights; Apache-2.0).
//!
//! This is the **fixture parity target** for the golden trace: capture the
//! same inputs from the reference implementation (BFL torch code or a
//! diffusers/ComfyUI build of the same architecture), run this reference,
//! compare per block. Written for obvious correctness, not speed — naive
//! loops, no kernel calls, no GPU.
//!
//! ## Source-pinned conventions (BFL `87f6fff`)
//!
//! - **Conditioning is ADDITIVE and 3072-wide**: `vec = time_in(sinusoidal(t))
//!   [+ guidance_in(sinusoidal(g))] + vector_in(pooled)` — three 3072-vectors
//!   summed elementwise. There is no concatenation in FLUX conditioning.
//! - **Timestep embedding is sinusoidal**: 256-dim `[cos, sin]` of
//!   `(t·1000)·theta^(-2i/128)`, fed through `time_in` = Linear(256→3072),
//!   SiLU, Linear(3072→3072) (`in_layer`/`out_layer` keys).
//! - **Block norms are weightless LayerNorms** (subtract mean, eps 1e-6):
//!   `img_norm1/2`, `txt_norm1/2`, `pre_norm`, `norm_final`. QK norms are the
//!   only learned norms: per-head RMSNorm (`head_dim` wide) with
//!   `norm.query_norm.scale` / `norm.key_norm.scale` weights, applied
//!   unconditionally before RoPE.
//! - **Modulation linears take `silu(vec)`** (SiLU lives inside `Modulation`),
//!   6 chunks `(shift, scale, gate)` ×2 per double-block stream, 3 chunks for
//!   single blocks; the MLP chain re-normalizes with `norm2` before its
//!   scale/shift; final head applies SiLU-then-Linear
//!   (`adaLN_modulation.1`) and takes the img stream only.
//! - **2D axial RoPE (standard rotation)**: `(a, b) → (c·a − s·b, s·a + c·b)`
//!   per interleaved pair, axes positions `(0, row, col)` (BFL `prepare()`
//!   img_ids: channel 1 = row, channel 2 = col) with per-axis width from
//!   `axes_dim` and frequencies `theta^(-2i/d_axis)`. Text rows carry zero
//!   positions → identity; image tokens only.
//! - **Fused single-block linears**: `linear1` = qkv + mlp-in in one matrix
//!   (`3d + 4d` wide), `linear2` = attention-proj + mlp-out (`d + 4d`
//!   wide); `img_attn.proj` / `txt_attn.proj` and `txt_in` are bias-free.
//! - Joint attention concatenates **text first, image second**; single blocks
//!   run attention over the same concat and the final head reads the img
//!   slice.

use crate::config::FluxDiffusionConfig;
use crate::manifest::{self, TS_EMBED_DIM};
use hipfire_runtime::model_source::ModelSource;
use std::collections::HashMap;

/// A row-major f32 tensor; `cols == 1` for vectors.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub fn elem(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

/// Host-side FLUX.1 transformer weights, keyed by manifest name.
#[derive(Debug, Clone)]
pub struct FluxWeights {
    pub tensors: HashMap<String, Tensor>,
}

pub(crate) fn synth_val(seed: u64, i: u64) -> f32 {
    // Deterministic LCG + frac(sin) — reproducible, no RNG dep.
    let mut x = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let frac = (x >> 11) as f64 / (1u64 << 53) as f64;
    (frac * 2.0 - 1.0) as f32
}

const SYNTH_SEED: u64 = 0xC0FF_EE00_0000_0040; // "…40" for arch 40

impl FluxWeights {
    /// Deterministic synthetic weights matching the manifest shapes — the
    /// self-parity substrate (no real checkpoint needed).
    pub fn synthetic(cfg: &FluxDiffusionConfig) -> Self {
        let mut tensors = HashMap::new();
        let mut idx = 0u64;
        for key in manifest::expected_flux_keys(cfg) {
            let (rows, cols) = (key.rows, key.cols);
            let n = rows * cols;
            let data: Vec<f32> = (0..n as u64)
                .map(|i| synth_val(SYNTH_SEED, idx + i) * 0.05)
                .collect();
            idx += n as u64;
            tensors.insert(key.name, Tensor { data, rows, cols });
        }
        FluxWeights { tensors }
    }

    pub fn get(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("flux reference: missing weight `{name}`"))
    }
}

// ─── Weight plan: naming, without decoding ─────────────────────────────────

/// One checkpoint tensor contributing to a canonical manifest key, with the
/// `[rows, cols]` it is expected to have.
#[derive(Debug, Clone)]
pub struct SourcePart {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
}

impl SourcePart {
    fn mat(name: String, rows: usize, cols: usize) -> Self {
        Self { name, rows, cols }
    }
    fn vec(name: String, n: usize) -> Self {
        Self {
            name,
            rows: n,
            cols: 1,
        }
    }
}

/// The two naming conventions the same FLUX weights ship under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxLayout {
    /// BFL single-file layout (`double_blocks.0.img_mod.lin.weight`), what
    /// ComfyUI and the official release ship. Keys ARE manifest keys.
    Bfl,
    /// diffusers layout (`transformer_blocks.0.attn.to_q.weight`), which
    /// splits the fused `qkv` / `linear1` projections into separate tensors.
    Diffusers,
    /// FLUX.2 (Klein) diffusers layout. Keys ARE manifest keys except the two
    /// fused joint-attention qkv projections, which diffusers ships split
    /// (`to_q`/`to_k`/`to_v` and `add_q_proj`/`add_k_proj`/`add_v_proj`).
    Flux2Diffusers,
}

/// Canonical manifest key → the checkpoint tensors that build it, in
/// row-concatenation order.
///
/// The point of separating this from the load is that a plan is **pure
/// metadata**: building one reads no tensor bytes and allocates nothing
/// model-sized. That lets the GPU upload path walk the manifest and convert
/// one tensor at a time straight out of the mmap
/// ([`FluxPlan::stage_f16`]) instead of first decoding the whole checkpoint
/// into f32 host tables — ~47 GB at FLUX.1-dev geometry, which is what used
/// to push a 128 GB unified-memory host into swap. The CPU reference path
/// materialises the same tables on demand through [`FluxPlan::materialize`].
#[derive(Debug, Clone)]
pub struct FluxPlan {
    pub layout: FluxLayout,
    parts: HashMap<String, Vec<SourcePart>>,
}

impl FluxPlan {
    /// Sniff which layout `src` uses and build the matching plan.
    ///
    /// Detection is one metadata lookup, so a ComfyUI model tree loads
    /// directly rather than needing a conversion step.
    pub fn detect(src: &dyn ModelSource, cfg: &FluxDiffusionConfig) -> Self {
        if cfg.is_flux2() {
            return Self::flux2_diffusers(cfg);
        }
        if src
            .tensor_info("double_blocks.0.img_mod.lin.weight")
            .is_some()
        {
            Self::bfl(cfg)
        } else {
            Self::diffusers(cfg)
        }
    }

    /// Identity plan: each manifest key is one checkpoint tensor of the same
    /// name and shape.
    pub fn bfl(cfg: &FluxDiffusionConfig) -> Self {
        let parts = manifest::expected_flux_keys(cfg)
            .into_iter()
            .map(|k| {
                let p = SourcePart::mat(k.name.clone(), k.rows, k.cols);
                (k.name, vec![p])
            })
            .collect();
        Self {
            layout: FluxLayout::Bfl,
            parts,
        }
    }

    /// Translate the diffusers key layout into the canonical BFL one the
    /// reference forward reads.
    ///
    /// The two differ only in key names and linear *splitting*:
    ///
    /// | canonical (BFL)        | diffusers                                        |
    /// |------------------------|--------------------------------------------------|
    /// | `img_in`               | `x_embedder`                                     |
    /// | `txt_in`               | `context_embedder`                               |
    /// | `time_in.in_layer`     | `time_text_embed.timestep_embedder.linear_1`     |
    /// | `time_in.out_layer`    | `time_text_embed.timestep_embedder.linear_2`     |
    /// | `vector_in.in_layer`   | `time_text_embed.text_embedder.linear_1`         |
    /// | `vector_in.out_layer`  | `time_text_embed.text_embedder.linear_2`         |
    /// | `double_blocks.{b}.img_attn.qkv` | `attn.to_q` + `to_k` + `to_v` (row-concat)      |
    /// | `double_blocks.{b}.img_attn.proj`| `attn.to_out.0`                                  |
    /// | `double_blocks.{b}.img_mlp.0/2`  | `ff.net.0.proj` / `ff.net.2`                     |
    /// | `single_blocks.{b}.linear1`      | `attn.to_q`+`to_k`+`to_v`+`proj_mlp` (row-concat)|
    /// | `single_blocks.{b}.linear2`      | `proj_out`                                       |
    /// | `final_layer.linear`             | `proj_out`                                       |
    pub fn diffusers(cfg: &FluxDiffusionConfig) -> Self {
        let d = cfg.hidden_size;
        let f = 4 * d;
        let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
        let mut parts: HashMap<String, Vec<SourcePart>> = HashMap::new();
        fn one(
            parts: &mut HashMap<String, Vec<SourcePart>>,
            canon: &str,
            name: &str,
            rows: usize,
            cols: usize,
        ) {
            parts.insert(
                canon.to_string(),
                vec![SourcePart::mat(name.to_string(), rows, cols)],
            );
        }

        // ── top-level ──────────────────────────────────────────────────
        one(
            &mut parts,
            "img_in.weight",
            "x_embedder.weight",
            d,
            patch_in,
        );
        one(&mut parts, "img_in.bias", "x_embedder.bias", d, 1);
        one(
            &mut parts,
            "txt_in.weight",
            "context_embedder.weight",
            d,
            cfg.txt_hidden_dim,
        );
        one(&mut parts, "txt_in.bias", "context_embedder.bias", d, 1);
        one(
            &mut parts,
            "time_in.in_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            d,
            256,
        );
        one(
            &mut parts,
            "time_in.in_layer.bias",
            "time_text_embed.timestep_embedder.linear_1.bias",
            d,
            1,
        );
        one(
            &mut parts,
            "time_in.out_layer.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
            d,
            d,
        );
        one(
            &mut parts,
            "time_in.out_layer.bias",
            "time_text_embed.timestep_embedder.linear_2.bias",
            d,
            1,
        );
        one(
            &mut parts,
            "vector_in.in_layer.weight",
            "time_text_embed.text_embedder.linear_1.weight",
            d,
            cfg.pooled_projection_dim,
        );
        one(
            &mut parts,
            "vector_in.in_layer.bias",
            "time_text_embed.text_embedder.linear_1.bias",
            d,
            1,
        );
        one(
            &mut parts,
            "vector_in.out_layer.weight",
            "time_text_embed.text_embedder.linear_2.weight",
            d,
            d,
        );
        one(
            &mut parts,
            "vector_in.out_layer.bias",
            "time_text_embed.text_embedder.linear_2.bias",
            d,
            1,
        );

        for b in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{b}.");
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_mod.lin.weight"),
                &format!("{p}norm1.linear.weight"),
                6 * d,
                d,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_mod.lin.bias"),
                &format!("{p}norm1.linear.bias"),
                6 * d,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_mod.lin.weight"),
                &format!("{p}norm1_context.linear.weight"),
                6 * d,
                d,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_mod.lin.bias"),
                &format!("{p}norm1_context.linear.bias"),
                6 * d,
                1,
            );

            parts.insert(
                format!("double_blocks.{b}.img_attn.qkv.weight"),
                vec![
                    SourcePart::mat(format!("{p}attn.to_q.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.to_k.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.to_v.weight"), d, d),
                ],
            );
            parts.insert(
                format!("double_blocks.{b}.img_attn.qkv.bias"),
                vec![
                    SourcePart::vec(format!("{p}attn.to_q.bias"), d),
                    SourcePart::vec(format!("{p}attn.to_k.bias"), d),
                    SourcePart::vec(format!("{p}attn.to_v.bias"), d),
                ],
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_attn.proj.weight"),
                &format!("{p}attn.to_out.0.weight"),
                d,
                d,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_attn.proj.bias"),
                &format!("{p}attn.to_out.0.bias"),
                d,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_attn.norm.query_norm.scale"),
                &format!("{p}attn.norm_q.weight"),
                cfg.head_dim,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_attn.norm.key_norm.scale"),
                &format!("{p}attn.norm_k.weight"),
                cfg.head_dim,
                1,
            );

            parts.insert(
                format!("double_blocks.{b}.txt_attn.qkv.weight"),
                vec![
                    SourcePart::mat(format!("{p}attn.add_q_proj.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.add_k_proj.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.add_v_proj.weight"), d, d),
                ],
            );
            parts.insert(
                format!("double_blocks.{b}.txt_attn.qkv.bias"),
                vec![
                    SourcePart::vec(format!("{p}attn.add_q_proj.bias"), d),
                    SourcePart::vec(format!("{p}attn.add_k_proj.bias"), d),
                    SourcePart::vec(format!("{p}attn.add_v_proj.bias"), d),
                ],
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_attn.proj.weight"),
                &format!("{p}attn.to_add_out.weight"),
                d,
                d,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_attn.proj.bias"),
                &format!("{p}attn.to_add_out.bias"),
                d,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_attn.norm.query_norm.scale"),
                &format!("{p}attn.norm_added_q.weight"),
                cfg.head_dim,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_attn.norm.key_norm.scale"),
                &format!("{p}attn.norm_added_k.weight"),
                cfg.head_dim,
                1,
            );

            one(
                &mut parts,
                &format!("double_blocks.{b}.img_mlp.0.weight"),
                &format!("{p}ff.net.0.proj.weight"),
                f,
                d,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_mlp.0.bias"),
                &format!("{p}ff.net.0.proj.bias"),
                f,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_mlp.2.weight"),
                &format!("{p}ff.net.2.weight"),
                d,
                f,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.img_mlp.2.bias"),
                &format!("{p}ff.net.2.bias"),
                d,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_mlp.0.weight"),
                &format!("{p}ff_context.net.0.proj.weight"),
                f,
                d,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_mlp.0.bias"),
                &format!("{p}ff_context.net.0.proj.bias"),
                f,
                1,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_mlp.2.weight"),
                &format!("{p}ff_context.net.2.weight"),
                d,
                f,
            );
            one(
                &mut parts,
                &format!("double_blocks.{b}.txt_mlp.2.bias"),
                &format!("{p}ff_context.net.2.bias"),
                d,
                1,
            );
        }

        for b in 0..cfg.num_single_layers {
            let p = format!("single_transformer_blocks.{b}.");
            one(
                &mut parts,
                &format!("single_blocks.{b}.modulation.lin.weight"),
                &format!("{p}norm.linear.weight"),
                3 * d,
                d,
            );
            one(
                &mut parts,
                &format!("single_blocks.{b}.modulation.lin.bias"),
                &format!("{p}norm.linear.bias"),
                3 * d,
                1,
            );
            parts.insert(
                format!("single_blocks.{b}.linear1.weight"),
                vec![
                    SourcePart::mat(format!("{p}attn.to_q.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.to_k.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.to_v.weight"), d, d),
                    SourcePart::mat(format!("{p}proj_mlp.weight"), f, d),
                ],
            );
            parts.insert(
                format!("single_blocks.{b}.linear1.bias"),
                vec![
                    SourcePart::vec(format!("{p}attn.to_q.bias"), d),
                    SourcePart::vec(format!("{p}attn.to_k.bias"), d),
                    SourcePart::vec(format!("{p}attn.to_v.bias"), d),
                    SourcePart::vec(format!("{p}proj_mlp.bias"), f),
                ],
            );
            one(
                &mut parts,
                &format!("single_blocks.{b}.linear2.weight"),
                &format!("{p}proj_out.weight"),
                d,
                d + f,
            );
            one(
                &mut parts,
                &format!("single_blocks.{b}.linear2.bias"),
                &format!("{p}proj_out.bias"),
                d,
                1,
            );
            one(
                &mut parts,
                &format!("single_blocks.{b}.norm.query_norm.scale"),
                &format!("{p}attn.norm_q.weight"),
                cfg.head_dim,
                1,
            );
            one(
                &mut parts,
                &format!("single_blocks.{b}.norm.key_norm.scale"),
                &format!("{p}attn.norm_k.weight"),
                cfg.head_dim,
                1,
            );
        }

        one(
            &mut parts,
            "final_layer.adaLN_modulation.1.weight",
            "norm_out.linear.weight",
            2 * d,
            d,
        );
        one(
            &mut parts,
            "final_layer.adaLN_modulation.1.bias",
            "norm_out.linear.bias",
            2 * d,
            1,
        );
        one(
            &mut parts,
            "final_layer.linear.weight",
            "proj_out.weight",
            patch_in,
            d,
        );
        one(
            &mut parts,
            "final_layer.linear.bias",
            "proj_out.bias",
            patch_in,
            1,
        );

        Self {
            layout: FluxLayout::Diffusers,
            parts,
        }
    }

    /// Build the plan for a FLUX.2 (Klein) diffusers-format checkpoint.
    ///
    /// Klein's diffusers keys ARE the canonical manifest names (see
    /// [`crate::manifest::expected_flux_keys`]'s FLUX.2 branch) except for
    /// the two fused joint-attention qkv projections, which diffusers ships
    /// split into three tensors each; this plan row-concatenates
    /// `to_q`/`to_k`/`to_v` into `attn.qkv` and
    /// `add_q_proj`/`add_k_proj`/`add_v_proj` into `attn.add_qkv`. There are
    /// no bias tensors and no per-block modulation linears (Klein shares
    /// modulation across every double/single block).
    pub fn flux2_diffusers(cfg: &FluxDiffusionConfig) -> Self {
        let d = cfg.hidden_size;
        let f = cfg.mlp_width();
        let hd = cfg.head_dim;
        let patch_in = cfg.patch_in();
        let mut parts: HashMap<String, Vec<SourcePart>> = HashMap::new();
        fn one(parts: &mut HashMap<String, Vec<SourcePart>>, name: &str, rows: usize, cols: usize) {
            parts.insert(
                name.to_string(),
                vec![SourcePart::mat(name.to_string(), rows, cols)],
            );
        }

        // ── top-level (shared across blocks) ──────────────────────────
        one(&mut parts, "x_embedder.weight", d, patch_in);
        one(&mut parts, "context_embedder.weight", d, cfg.txt_hidden_dim);
        one(
            &mut parts,
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            d,
            manifest::TS_EMBED_DIM,
        );
        one(
            &mut parts,
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            d,
            d,
        );
        one(
            &mut parts,
            "double_stream_modulation_img.linear.weight",
            6 * d,
            d,
        );
        one(
            &mut parts,
            "double_stream_modulation_txt.linear.weight",
            6 * d,
            d,
        );
        one(
            &mut parts,
            "single_stream_modulation.linear.weight",
            3 * d,
            d,
        );

        for b in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{b}.");
            parts.insert(
                format!("{p}attn.qkv.weight"),
                vec![
                    SourcePart::mat(format!("{p}attn.to_q.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.to_k.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.to_v.weight"), d, d),
                ],
            );
            one(&mut parts, &format!("{p}attn.to_out.0.weight"), d, d);
            parts.insert(
                format!("{p}attn.add_qkv.weight"),
                vec![
                    SourcePart::mat(format!("{p}attn.add_q_proj.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.add_k_proj.weight"), d, d),
                    SourcePart::mat(format!("{p}attn.add_v_proj.weight"), d, d),
                ],
            );
            one(&mut parts, &format!("{p}attn.to_add_out.weight"), d, d);
            one(&mut parts, &format!("{p}attn.norm_q.weight"), hd, 1);
            one(&mut parts, &format!("{p}attn.norm_k.weight"), hd, 1);
            one(&mut parts, &format!("{p}attn.norm_added_q.weight"), hd, 1);
            one(&mut parts, &format!("{p}attn.norm_added_k.weight"), hd, 1);
            one(&mut parts, &format!("{p}ff.linear_in.weight"), 2 * f, d);
            one(&mut parts, &format!("{p}ff.linear_out.weight"), d, f);
            one(
                &mut parts,
                &format!("{p}ff_context.linear_in.weight"),
                2 * f,
                d,
            );
            one(
                &mut parts,
                &format!("{p}ff_context.linear_out.weight"),
                d,
                f,
            );
        }

        for b in 0..cfg.num_single_layers {
            let p = format!("single_transformer_blocks.{b}.");
            one(
                &mut parts,
                &format!("{p}attn.to_qkv_mlp_proj.weight"),
                3 * d + 2 * f,
                d,
            );
            one(&mut parts, &format!("{p}attn.to_out.weight"), d, d + f);
            one(&mut parts, &format!("{p}attn.norm_q.weight"), hd, 1);
            one(&mut parts, &format!("{p}attn.norm_k.weight"), hd, 1);
        }

        one(&mut parts, "norm_out.linear.weight", 2 * d, d);
        one(&mut parts, "proj_out.weight", patch_in, d);

        Self {
            layout: FluxLayout::Flux2Diffusers,
            parts,
        }
    }

    /// Every manifest key must be covered, with the manifest shape — checked
    /// before anything is decoded, so a checkpoint with an unexpected layout
    /// fails at load rather than mid-forward.
    pub fn validate(&self, cfg: &FluxDiffusionConfig) -> Result<(), String> {
        let tag = self.tag();
        for key in manifest::expected_flux_keys(cfg) {
            let parts = self.parts(&key.name)?;
            let rows: usize = parts.iter().map(|p| p.rows).sum();
            let cols = parts[0].cols;
            if rows != key.rows || cols != key.cols {
                return Err(format!(
                    "{tag}: manifest key `{}` built as [{rows}, {cols}], manifest [{}, {}]",
                    key.name, key.rows, key.cols
                ));
            }
        }
        Ok(())
    }

    /// The checkpoint tensors behind one canonical key.
    pub fn parts(&self, key: &str) -> Result<&[SourcePart], String> {
        self.parts
            .get(key)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("{}: manifest key `{key}` not built", self.tag()))
    }

    fn tag(&self) -> &'static str {
        match self.layout {
            FluxLayout::Bfl => "flux",
            FluxLayout::Diffusers => "diffusers flux",
            FluxLayout::Flux2Diffusers => "flux2",
        }
    }

    /// Locate one part's bytes, validating dtype-independent shape.
    ///
    /// The BFL layout checks the safetensors `shape` field against the
    /// manifest (a 1-D `[len]` for vectors, a 2-D `[rows, cols]` otherwise),
    /// because there the checkpoint key IS the manifest key and a mismatch
    /// means the wrong model. The diffusers layout row-concatenates several
    /// tensors into one manifest key, so only the element count is meaningful
    /// per part.
    fn locate<'a>(
        &self,
        src: &'a dyn ModelSource,
        part: &SourcePart,
    ) -> Result<(&'a str, &'a [u8]), String> {
        let (info, bytes) = match self.layout {
            FluxLayout::Bfl => src
                .tensor_data(&part.name)
                .ok_or_else(|| format!("flux: tensor `{}` missing from source", part.name))?,
            FluxLayout::Diffusers | FluxLayout::Flux2Diffusers => src
                .tensor_data(&part.name)
                .ok_or_else(|| format!("{}: missing tensor `{}`", self.tag(), part.name))?,
        };
        if self.layout == FluxLayout::Bfl {
            let shape_ok = if part.cols == 1 && info.shape.len() == 1 {
                info.shape[0] == part.rows
            } else {
                info.shape.len() == 2 && info.shape[0] == part.rows && info.shape[1] == part.cols
            };
            if !shape_ok {
                return Err(format!(
                    "flux: tensor `{}` shape {:?} != manifest [{}, {}]",
                    part.name, info.shape, part.rows, part.cols
                ));
            }
        }
        Ok((info.dtype.as_str(), bytes))
    }

    /// Decode one canonical key to an f32 host tensor, row-concatenating its
    /// parts. This is the per-key unit the CPU reference path materialises.
    pub fn tensor(&self, src: &dyn ModelSource, key: &manifest::FluxKey) -> Result<Tensor, String> {
        let tag = self.tag();
        let parts = self.parts(&key.name)?;
        let mut data: Vec<f32> = Vec::with_capacity(key.rows * key.cols);
        for part in parts {
            let (dtype, bytes) = self.locate(src, part)?;
            let decoded = decode_dtype(dtype, bytes)
                .map_err(|e| format!("{tag}: tensor `{}`: {e}", part.name))?;
            let want = part.rows * part.cols;
            if decoded.len() != want {
                return Err(match self.layout {
                    FluxLayout::Bfl => format!(
                        "flux: tensor `{}` decoded {} elements, manifest wants {want}",
                        part.name,
                        decoded.len()
                    ),
                    FluxLayout::Diffusers | FluxLayout::Flux2Diffusers => format!(
                        "{}: tensor `{}` has {} elements, expected {want}",
                        tag,
                        part.name,
                        decoded.len()
                    ),
                });
            }
            data.extend_from_slice(&decoded);
        }
        Ok(Tensor {
            data,
            rows: key.rows,
            cols: key.cols,
        })
    }

    /// Convert one canonical key's parts into `stage` as f16 words, WITHOUT
    /// ever holding an f32 copy of the tensor.
    ///
    /// `stage` is cleared first, so the returned slice is exactly this key.
    /// Bit-identical to `tensor()` followed by the device `(_Float16)` cast
    /// it replaces — see [`crate::f16_stage`].
    pub fn stage_f16<'s>(
        &self,
        src: &dyn ModelSource,
        key: &manifest::FluxKey,
        stage: &'s mut crate::f16_stage::F16Stage,
    ) -> Result<&'s [u16], String> {
        let tag = self.tag();
        let parts = self.parts(&key.name)?;
        stage.clear();
        for part in parts {
            let (dtype, bytes) = self.locate(src, part)?;
            let n = stage
                .push(dtype, bytes)
                .map_err(|e| format!("{tag}: tensor `{}`: {e}", part.name))?;
            let want = part.rows * part.cols;
            if n != want {
                return Err(format!(
                    "{tag}: tensor `{}` has {n} elements, expected {want}",
                    part.name
                ));
            }
        }
        let want = key.rows * key.cols;
        if stage.words().len() != want {
            return Err(format!(
                "{tag}: key `{}` staged {} elements, manifest wants {want}",
                key.name,
                stage.words().len()
            ));
        }
        Ok(stage.words())
    }

    /// Hint that every source tensor behind one canonical key is done with,
    /// so the source can drop its pages. Best-effort by contract.
    pub fn release(&self, src: &dyn ModelSource, key: &str) {
        if let Ok(parts) = self.parts(key) {
            for part in parts {
                src.release_tensor_pages(&part.name);
            }
        }
    }

    /// Decode the WHOLE checkpoint into f32 host tables.
    ///
    /// At FLUX.1-dev geometry this is ~47 GB of host `Vec`. It is the CPU
    /// reference path's input and nothing else — the GPU path streams
    /// (`GpuFluxWeights::from_stream`) and never calls this.
    pub fn materialize(
        &self,
        src: &dyn ModelSource,
        cfg: &FluxDiffusionConfig,
    ) -> Result<FluxWeights, String> {
        self.validate(cfg)?;
        let mut tensors = HashMap::new();
        for key in manifest::expected_flux_keys(cfg) {
            let t = self.tensor(src, &key)?;
            tensors.insert(key.name, t);
        }
        Ok(FluxWeights { tensors })
    }
}

/// Host-side weight load: read every manifest key from a
/// [`ModelSource`] in the canonical BFL layout, decode F32/BF16/F16 to f32,
/// and validate each shape against the manifest. No GPU involved —
/// correctness is testable against a synthetic checkpoint (see the
/// `load_tests` module).
///
/// A thin wrapper over [`FluxPlan::bfl`] + [`FluxPlan::materialize`]: one
/// decode implementation serves both the eager CPU tables and the streaming
/// GPU upload, so the two cannot drift.
pub fn load_weights(
    src: &dyn ModelSource,
    cfg: &FluxDiffusionConfig,
) -> Result<FluxWeights, String> {
    FluxPlan::bfl(cfg).materialize(src, cfg)
}

/// Load FLUX transformer weights from a DIFFUSERS-format checkpoint (keys
/// like `x_embedder.weight`, `transformer_blocks.0.attn.to_q.weight`) and
/// translate them into the canonical BFL layout [`load_weights`] reads —
/// the same `FluxWeights` map, so the CPU reference forward is shared.
///
/// See [`FluxPlan::diffusers`] for the key mapping. Returns an error naming
/// the first manifest key that cannot be built, so a checkpoint with
/// unexpected keys fails at load, not mid-forward.
pub fn load_weights_diffusers(
    src: &dyn ModelSource,
    cfg: &FluxDiffusionConfig,
) -> Result<FluxWeights, String> {
    FluxPlan::diffusers(cfg).materialize(src, cfg)
}

/// Decode a little-endian F32 / BF16 / F16 byte stream to f32.
/// Widen a safetensors tensor to f32.
///
/// Parallel because of scale, not cleverness: loading FLUX.1-dev means
/// widening 11.9e9 BF16 values, and elementwise conversion is embarrassingly
/// parallel with identical results either way.
pub fn decode_dtype(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    use rayon::prelude::*;
    match dtype {
        "F32" => Ok(bytes
            .par_chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(bytes
            .par_chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F16" => Ok(bytes
            .par_chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(format!(
            "unsupported tensor dtype `{other}` (F32/BF16/F16 only)"
        )),
    }
}

/// BF16 → f32: the bf16 bits are the high half of the f32 bits.
fn bf16_to_f32(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

/// IEEE half → f32, hand-rolled so the reference stays dependency-free.
fn f16_to_f32(u: u16) -> f32 {
    let sign = ((u >> 15) & 1) as u32;
    let exp = ((u >> 10) & 0x1F) as u32;
    let man = (u & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // Subnormal: shift the mantissa until it carries into the
            // exponent field, then renormalize.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

// ─── Primitive ops ─────────────────────────────────────────────────────────

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu_tanh(x: f32) -> f32 {
    // GELU tanh approximation, as used by FLUX MLPs.
    0.5 * x
        * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() as f32 * (x + 0.044_715 * x * x * x)).tanh())
}

/// `x` (n×in) → `y` (n×out) with `w.rows == out`, `w.cols == in`.
fn linear(w: &Tensor, b: Option<&Tensor>, x: &[f32], in_dim: usize, n: usize) -> Vec<f32> {
    let out = w.rows;
    assert_eq!(w.cols, in_dim, "linear in-dim mismatch");
    let mut y = vec![0.0f32; n * out];
    for r in 0..n {
        for o in 0..out {
            let mut acc = b.map_or(0.0, |b| b.elem(o, 0));
            for i in 0..in_dim {
                acc += w.elem(o, i) * x[r * in_dim + i];
            }
            y[r * out + o] = acc;
        }
    }
    y
}

/// 1-D MLP with silu between the two linears (MLPEmbedder: in_layer → SiLU →
/// out_layer; used for time_in / vector_in / guidance_in).
fn mlp_embedder(w0: &Tensor, b0: &Tensor, w1: &Tensor, b1: &Tensor, x: &[f32]) -> Vec<f32> {
    let h = w0.rows;
    let mut mid = vec![0.0f32; h];
    for o in 0..h {
        let mut acc = b0.elem(o, 0);
        for i in 0..x.len() {
            acc += w0.elem(o, i) * x[i];
        }
        mid[o] = silu(acc);
    }
    let out_dim = w1.rows;
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = b1.elem(o, 0);
        for i in 0..h {
            acc += w1.elem(o, i) * mid[i];
        }
        y[o] = acc;
    }
    y
}

/// Weightless LayerNorm (mean-subtract + normalize), FLUX's block norm
/// (`nn.LayerNorm(..., elementwise_affine=False, eps=1e-6)`).
fn layernorm(x: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mean = x.iter().sum::<f32>() / d as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|v| (v - mean) * inv).collect()
}

/// FLUX's sinusoidal timestep embedding: `[cos, sin]` of `(t·time_factor) ·
/// max_period^(-2i/dim)` over `dim/2` frequencies. BFL applies `time_factor
/// = 1000.0` internally, so the model input `t` is the RAW scheduler value.
pub(crate) fn timestep_embedding(
    t: f32,
    dim: usize,
    max_period: f64,
    time_factor: f32,
) -> Vec<f32> {
    let half = dim / 2;
    let mut emb = vec![0.0f32; dim];
    let v = t * time_factor;
    for i in 0..half {
        let freq = (-(max_period.ln()) * i as f64 / half as f64).exp() as f32;
        let arg = v * freq;
        emb[i] = arg.cos();
        emb[half + i] = arg.sin();
    }
    emb
}

/// Per-head RMSNorm with learned scale (FLUX QK norm). q/k are `n × heads·hd`
/// row-major with per-token head blocks contiguous, matching the BFL
/// `QKNorm` applied after the `(B L (K H D))` split.
fn qk_rmsnorm(
    q: &[f32],
    k: &[f32],
    n_heads: usize,
    hd: usize,
    q_scale: &Tensor,
    k_scale: &Tensor,
) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-6;
    let mut qo = q.to_vec();
    let mut ko = k.to_vec();
    for (buf, scale) in [(&mut qo, q_scale), (&mut ko, k_scale)] {
        let rows = buf.len() / (n_heads * hd);
        for t in 0..rows {
            for h in 0..n_heads {
                let row = &mut buf[(t * n_heads + h) * hd..(t * n_heads + h + 1) * hd];
                let mut sum = 0.0f32;
                for v in row.iter() {
                    sum += v * v;
                }
                let inv = 1.0 / (sum / hd as f32 + eps).sqrt();
                for (i, v) in row.iter_mut().enumerate() {
                    *v *= inv * scale.elem(i, 0);
                }
            }
        }
    }
    (qo, ko)
}

/// Build the FLUX.1-convention 4-axis RoPE ids `(t, row, col, 0)` for an
/// `n_img_rows`-token image grid, row-major (`t → (row, col)` via
/// `grid.1` as the row stride). Axis 0 (`t`) is the caller-supplied time
/// value; FLUX.1 always passes `0.0` here (BFL `prepare()` img_ids channel 0
/// is always 0). Axis 3 is always 0 — FLUX.1's `axes_dim[3] == 0` rotates
/// nothing there; FLUX.2 Klein uses the same slot for a real 4th axis when
/// the caller supplies explicit `img_ids` instead of this derived grid.
pub fn rope_ids_for_grid(grid: (usize, usize), t: f32) -> Vec<[f32; 4]> {
    let (grid_h, grid_w) = grid;
    let mut ids = Vec::with_capacity(grid_h * grid_w);
    for i in 0..grid_h * grid_w {
        let (row, col) = (i / grid_w, i % grid_w);
        ids.push([t, row as f32, col as f32, 0.0]);
    }
    ids
}

/// Build the FLUX.2 Klein 4-axis RoPE ids for the `n_txt` TEXT tokens:
/// `(0, 0, 0, l)` with `l` the token index. This is ComfyUI's
/// `model_detection` `txt_ids_dims = [3]` for `image_model == "flux2"`, which
/// makes `Flux._forward` fill axis 3 with
/// `linspace(0, context.shape[1] - 1, steps=context.shape[1])` while axes
/// 0/1/2 stay zero. FLUX.1 has `txt_ids_dims = []` — its text ids are all
/// zero (BFL `prepare()`), so this helper is Flux2-only and the FLUX.1 path
/// must not call it.
pub fn text_ids(n_txt: usize) -> Vec<[f32; 4]> {
    (0..n_txt).map(|l| [0.0, 0.0, 0.0, l as f32]).collect()
}

/// 2D axial RoPE with the BFL **standard** rotation (see `flux/math.py`,
/// 2×2 matrix `[[c, −s],[s, c]]`):
/// per interleaved pair `(a, b)` with frequency `theta^(-2i/d_axis)` for axis
/// positions `(0, row, col)` (BFL `prepare()`: img_ids channel 0 = 0,
/// 1 = row/y, 2 = col/x): `(a, b) → (c·a − s·b, s·a + c·b)`.
/// `x` holds `n_img_rows × heads·hd`; `grid (h, w)` maps image row index
/// `t → (row, col)`. Text rows carry all-zero ids → identity (BFL sends
/// `txt_ids = 0`), so only image rows are rotated.
///
/// Thin wrapper over [`rope_ids`] using the derived FLUX.1 grid ids — kept so
/// the FLUX.1 call sites (and the pinned-sum parity test) don't need to
/// change shape; the rotation loop itself lives in `rope_ids` and is shared
/// with the FLUX.2 explicit-ids path.
fn rope_2d(
    x: &mut [f32],
    n_img_rows: usize,
    n_heads: usize,
    hd: usize,
    grid: (usize, usize),
    axes_dim: [usize; 4],
    theta: f64,
) {
    let ids = rope_ids_for_grid(grid, 0.0);
    // `attention` may rope a shorter trailing image run than a full
    // `grid.0 * grid.1` grid (see `attention_image_rows_rope_but_text_rows_do_not_move`);
    // the row-major ids sequence still starts at t=0, so take its prefix.
    rope_ids(
        x,
        n_img_rows,
        n_heads,
        hd,
        &ids[..n_img_rows],
        axes_dim,
        theta,
    );
}

/// 4-axis RoPE, `pos = ids[t]` per image-stream token instead of a derived
/// grid — the FLUX.2 Klein path (explicit reference/time ids). Same rotation
/// arithmetic as `rope_2d` (identical `theta.powf(2p/d_axis)` in f64,
/// `cos()/sin()` cast to f32, same pair order): `rope_2d` is exactly this
/// function called with `rope_ids_for_grid(grid, 0.0)`, so the two paths
/// cannot numerically drift apart.
#[allow(clippy::too_many_arguments)]
fn rope_ids(
    x: &mut [f32],
    n_img_rows: usize,
    n_heads: usize,
    hd: usize,
    ids: &[[f32; 4]],
    axes_dim: [usize; 4],
    theta: f64,
) {
    debug_assert_eq!(ids.len(), n_img_rows);
    let mut pair_regions = [0usize; 4];
    let mut acc = 0usize;
    for (i, d_axis) in axes_dim.iter().enumerate() {
        pair_regions[i] = acc;
        acc += d_axis / 2;
    }
    for t in 0..n_img_rows {
        let pos = ids[t];
        for h in 0..n_heads {
            let base = (t * n_heads + h) * hd;
            for (axis, &d_axis) in axes_dim.iter().enumerate() {
                for p in 0..d_axis / 2 {
                    let angle = pos[axis] as f64 / theta.powf(2.0 * p as f64 / d_axis as f64);
                    let (c, s) = (angle.cos() as f32, angle.sin() as f32);
                    let i = base + 2 * (pair_regions[axis] + p);
                    let (a, b) = (x[i], x[i + 1]);
                    x[i] = a * c - b * s;
                    x[i + 1] = a * s + b * c;
                }
            }
        }
    }
}

/// Joint attention over `n_q` query rows and `n_kv` key/value rows (all
/// `heads × head_dim` per row). RoPE is applied to the trailing `n_q_img`
/// rows of q and the trailing `n_k_img` rows of k (BFL concatenates text
/// FIRST, so image rows sit at the end); scale = `1/√head_dim`.
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    n_heads: usize,
    n_q_img: usize,
    n_k_img: usize,
    grid: (usize, usize),
    axes_dim: [usize; 4],
    theta: f64,
) -> Vec<f32> {
    let d_head = n_heads * head_dim;
    let mut qq = q.to_vec();
    let mut kk = k.to_vec();
    if n_q_img > 0 {
        rope_2d(
            &mut qq[(n_q - n_q_img) * d_head..],
            n_q_img,
            n_heads,
            head_dim,
            grid,
            axes_dim,
            theta,
        );
    }
    if n_k_img > 0 {
        rope_2d(
            &mut kk[(n_kv - n_k_img) * d_head..],
            n_k_img,
            n_heads,
            head_dim,
            grid,
            axes_dim,
            theta,
        );
    }
    attention_core(&qq, &kk, v, n_q, n_kv, head_dim, n_heads)
}

/// Joint self-attention over `n` combined text+image rows using an explicit
/// per-token 4-axis RoPE `ids` table (FLUX.2 Klein) instead of a derived
/// grid. Text-first concat, same as `attention` — but unlike FLUX.1, ALL `n`
/// rows are rotated: Klein's text tokens carry `(0, 0, 0, l)` (ComfyUI
/// `txt_ids_dims = [3]`, see [`text_ids`]), so `ids` covers the text rows
/// too and is expected to have exactly `n` entries.
#[allow(clippy::too_many_arguments)]
fn attention_ids(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    head_dim: usize,
    n_heads: usize,
    ids: &[[f32; 4]],
    axes_dim: [usize; 4],
    theta: f64,
) -> Vec<f32> {
    let mut qq = q.to_vec();
    let mut kk = k.to_vec();
    if n > 0 {
        rope_ids(&mut qq, n, n_heads, head_dim, ids, axes_dim, theta);
        rope_ids(&mut kk, n, n_heads, head_dim, ids, axes_dim, theta);
    }
    attention_core(&qq, &kk, v, n, n, head_dim, n_heads)
}

/// Softmax(qk^T / sqrt(hd)) · v per head, shared by [`attention`] (grid RoPE)
/// and [`attention_ids`] (explicit-ids RoPE) once each has rotated its own
/// `qq`/`kk` copies.
fn attention_core(
    qq: &[f32],
    kk: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    n_heads: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let d_head = n_heads * head_dim;
    let mut out = vec![0.0f32; n_q * d_head];
    for h in 0..n_heads {
        for t in 0..n_q {
            let mut max_s = f32::MIN;
            let mut logits = vec![0.0f32; n_kv];
            for u in 0..n_kv {
                let mut acc = 0.0f32;
                for dd in 0..head_dim {
                    let qi = (t * n_heads + h) * head_dim + dd;
                    let ki = (u * n_heads + h) * head_dim + dd;
                    acc += qq[qi] * kk[ki];
                }
                logits[u] = acc * scale;
                if logits[u] > max_s {
                    max_s = logits[u];
                }
            }
            let mut denom = 0.0f32;
            for u in 0..n_kv {
                logits[u] = (logits[u] - max_s).exp();
                denom += logits[u];
            }
            for dd in 0..head_dim {
                let mut acc = 0.0f32;
                for u in 0..n_kv {
                    let vi = (u * n_heads + h) * head_dim + dd;
                    acc += logits[u] / denom * v[vi];
                }
                out[(t * n_heads + h) * head_dim + dd] = acc;
            }
        }
    }
    out
}

// ─── Forward ───────────────────────────────────────────────────────────────

/// Single-block MLP stream activation. BFL, diffusers and ComfyUI all use
/// GELU-tanh here (diffusers `FluxSingleTransformerBlock.act_mlp`); the knob
/// exists so a future deviation can be pinned without touching the block
/// math. Default keeps the shipped-FLUX GELU-tanh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlpAct {
    #[default]
    GeluTanh,
    Silu,
}

/// Final-head adaLN chunk order. BFL `LastLayer` (and ComfyUI) emit
/// `(shift, scale)` from the 2-chunk linear; diffusers `AdaLayerNormContinuous`
/// emits `(scale, shift)` — REVERSED, another documented diffusers deviation
/// from the shipped FLUX architecture. The golden (diffusers) needs
/// `ScaleShift`; real weights need the BFL `ShiftScale` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FinalAdaLNOrder {
    #[default]
    ShiftScale,
    ScaleShift,
}

/// Everything the reference forward needs besides weights.
pub struct FluxForwardInput {
    /// Timestep, the RAW scheduler value (BFL embeds `t·1000` internally).
    pub timestep: f32,
    /// Pooled text condition, len `pooled_projection_dim`.
    pub pooled: Vec<f32>,
    /// Guidance-scale value (None for schnell / guidance-free).
    pub guidance: Option<f32>,
    /// Text-encoder output, `n_txt × txt_hidden_dim`, row-major.
    pub txt: Vec<f32>,
    /// Patched image latents, `n_img × patch_in`; the reference applies
    /// `img_in` itself.
    pub img: Vec<f32>,
    /// Image-token grid `(height, width)` for 2D RoPE positions.
    pub grid: (usize, usize),
    /// Single-block MLP activation (default BFL GELU-tanh; see [`MlpAct`]).
    pub mlp_act: MlpAct,
    /// Final-head adaLN chunk order (default BFL shift-first; the diffusers
    /// golden selects scale-first — see [`FinalAdaLNOrder`]).
    pub final_order: FinalAdaLNOrder,
    /// Explicit per-image-token 4-axis RoPE ids `(t, row, col, extra)`, one
    /// entry per image-stream token. `None` derives `(0, row, col, 0)` from
    /// `grid` (the FLUX.1 convention; see [`rope_ids_for_grid`]). FLUX.2
    /// Klein reference/edit inputs pass explicit ids so a non-zero time axis
    /// (multi-reference conditioning) can move image tokens in RoPE space.
    pub img_ids: Option<Vec<[f32; 4]>>,
}

/// Run the MMDiT forward; returns the final image stream `n_img × patch_in`
/// (input to the next denoise step after patching back).
pub fn forward(cfg: &FluxDiffusionConfig, w: &FluxWeights, input: &FluxForwardInput) -> Vec<f32> {
    forward_parts(cfg, w, input)
        .into_iter()
        .last()
        .map(|(_, v)| v)
        .expect("forward_parts never returns an empty list")
}

/// Run the MMDiT forward and return the named intermediates produced along
/// the way — `vec`, `img_in`, `txt_in`, `double_{b}_img` /
/// `double_{b}_txt` per double block, `single_concat` after the single
/// blocks, and `final`. The parity harness compares these one-by-one against
/// the golden trace, so the first divergent part identifies the convention.
pub fn forward_parts(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    input: &FluxForwardInput,
) -> Vec<(String, Vec<f32>)> {
    if cfg.is_flux2() {
        return forward_parts_flux2(cfg, w, input);
    }
    let d = cfg.hidden_size;
    let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
    let txt_dim = cfg.txt_hidden_dim;
    let n_img = input.img.len() / patch_in;
    let n_txt = input.txt.len() / txt_dim;
    debug_assert_eq!(input.grid.0 * input.grid.1, n_img);
    let mut parts: Vec<(String, Vec<f32>)> = Vec::new();

    // Conditioning: vec = time_in(sinu(t)) [+ guidance_in(sinu(g))] +
    // vector_in(pooled) — all 3072-wide, added elementwise.
    let te = timestep_embedding(input.timestep, TS_EMBED_DIM, 10000.0, 1000.0);
    let mut vec = mlp_embedder(
        w.get("time_in.in_layer.weight"),
        w.get("time_in.in_layer.bias"),
        w.get("time_in.out_layer.weight"),
        w.get("time_in.out_layer.bias"),
        &te,
    );
    if let Some(g) = input.guidance {
        let ge = timestep_embedding(g, TS_EMBED_DIM, 10000.0, 1000.0);
        let gv = mlp_embedder(
            w.get("guidance_in.in_layer.weight"),
            w.get("guidance_in.in_layer.bias"),
            w.get("guidance_in.out_layer.weight"),
            w.get("guidance_in.out_layer.bias"),
            &ge,
        );
        for i in 0..d {
            vec[i] += gv[i];
        }
    }
    let c = mlp_embedder(
        w.get("vector_in.in_layer.weight"),
        w.get("vector_in.in_layer.bias"),
        w.get("vector_in.out_layer.weight"),
        w.get("vector_in.out_layer.bias"),
        &input.pooled,
    );
    for i in 0..d {
        vec[i] += c[i];
    }
    parts.push(("vec".into(), vec.clone()));

    // Stream embeddings (img_in and txt_in both carry biases).
    let mut img = linear(
        w.get("img_in.weight"),
        Some(w.get("img_in.bias")),
        &input.img,
        patch_in,
        n_img,
    );
    let mut txt = linear(
        w.get("txt_in.weight"),
        Some(w.get("txt_in.bias")),
        &input.txt,
        txt_dim,
        n_txt,
    );
    parts.push(("img_in".into(), img.clone()));
    parts.push(("txt_in".into(), txt.clone()));

    // Double blocks (text-first joint attention, img stream updated in place).
    for b in 0..cfg.num_layers {
        let (a, b_) = double_block(cfg, w, b, &img, &txt, &vec, n_img, input.grid);
        img = a;
        txt = b_;
        parts.push((format!("double_{b}_img"), img.clone()));
        parts.push((format!("double_{b}_txt"), txt.clone()));
    }
    // Single blocks act on the concatenated stream.
    let mut fused: Vec<f32> = txt.clone();
    fused.extend_from_slice(&img);
    for b in 0..cfg.num_single_layers {
        fused = single_block(cfg, w, b, &fused, &vec, n_img, input.grid, input.mlp_act);
    }
    parts.push(("single_concat".into(), fused.clone()));
    let img_only: Vec<f32> = fused[n_txt * d..].to_vec();

    // Final head: SiLU-then-Linear adaLN (2 chunks: shift, scale), weightless
    // norm_final, bias-ful patch projection.
    let adain = linear(
        w.get("final_layer.adaLN_modulation.1.weight"),
        Some(w.get("final_layer.adaLN_modulation.1.bias")),
        &vec.iter().map(|v| silu(*v)).collect::<Vec<f32>>(),
        d,
        1,
    );
    let n = img_only.len() / d;
    let mut h = vec![0.0f32; img_only.len()];
    for t in 0..n {
        let row = &img_only[t * d..(t + 1) * d];
        let normed = layernorm(row, 1e-6);
        for i in 0..d {
            let (shift, scale) = match input.final_order {
                FinalAdaLNOrder::ShiftScale => (adain[i], adain[d + i]),
                FinalAdaLNOrder::ScaleShift => (adain[d + i], adain[i]),
            };
            h[t * d + i] = (1.0 + scale) * normed[i] + shift;
        }
    }
    let out = linear(
        w.get("final_layer.linear.weight"),
        Some(w.get("final_layer.linear.bias")),
        &h,
        d,
        n,
    );
    parts.push(("final".into(), out));
    parts
}

/// Dual-stream block (BFL `DoubleStreamBlock`, commit 87f6fff): per-stream
/// modulation of `silu(vec)` (6 chunks `(shift, scale, gate)` ×2), qkv →
/// per-head QK-RMSNorm, joint attention over the text-first concat, gated
/// residual, re-normalized gated MLP.
#[allow(clippy::too_many_arguments)]
pub fn double_block(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    b: usize,
    img: &[f32],
    txt: &[f32],
    vec: &[f32],
    n_img: usize,
    grid: (usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let n_txt = txt.len() / d;
    let n_kv = n_img + n_txt;
    let silu_vec: Vec<f32> = vec.iter().map(|v| silu(*v)).collect();

    // Per-stream modulation + qkv + QK norm (order-independent).
    let mut q = [Vec::new(), Vec::new()];
    let mut k = [Vec::new(), Vec::new()];
    let mut v = [Vec::new(), Vec::new()];
    let mut modu = [Vec::new(), Vec::new()];
    for (si, (stem, x, n)) in [("img", img, n_img), ("txt", txt, n_txt)]
        .into_iter()
        .enumerate()
    {
        let p = |s: &str| w.get(&format!("double_blocks.{b}.{stem}_{s}"));
        let m = linear(
            p("mod.lin.weight"),
            Some(p("mod.lin.bias")),
            &silu_vec,
            d,
            1,
        );
        modu[si] = m;
        let mut h = vec![0.0f32; n * d];
        for t in 0..n {
            let row = &x[t * d..(t + 1) * d];
            let normed = layernorm(row, 1e-6);
            for i in 0..d {
                h[t * d + i] = (1.0 + modu[si][d + i]) * normed[i] + modu[si][i];
            }
        }
        let qkv = linear(p("attn.qkv.weight"), Some(p("attn.qkv.bias")), &h, d, n);
        let (qq, kk, vv) = split_qkv(&qkv, n, heads, hd);
        let (qqn, kkn) = qk_rmsnorm(
            &qq,
            &kk,
            heads,
            hd,
            p("attn.norm.query_norm.scale"),
            p("attn.norm.key_norm.scale"),
        );
        q[si] = qqn.clone();
        k[si] = kkn.clone();
        v[si] = vv;
    }

    // ONE joint attention over the full text-first concat (BFL: q = cat(txt_q,
    // img_q), one call per double block); the output rows are then split so
    // each stream's proj/gate/MLP operate on its own slices.
    let mut q_all: Vec<f32> = q[1].clone();
    q_all.extend_from_slice(&q[0]);
    let mut k_all: Vec<f32> = k[1].clone();
    k_all.extend_from_slice(&k[0]);
    let mut v_all: Vec<f32> = v[1].clone();
    v_all.extend_from_slice(&v[0]);
    let att = attention(
        &q_all,
        &k_all,
        &v_all,
        n_kv,
        n_kv,
        hd,
        heads,
        n_img,
        n_img,
        grid,
        cfg.axes_dim,
        cfg.theta,
    );
    let d_head = heads * hd;
    let mut out = [img.to_vec(), txt.to_vec()];
    for si in 0..2 {
        let stem = if si == 0 { "img" } else { "txt" };
        let n = if si == 0 { n_img } else { n_txt };
        let start = if si == 0 { n_txt } else { 0 };
        let att_slice = &att[start * d_head..(start + n) * d_head];
        let p = |s: &str| w.get(&format!("double_blocks.{b}.{stem}_{s}"));
        let proj = linear(
            p("attn.proj.weight"),
            Some(p("attn.proj.bias")),
            att_slice,
            d,
            n,
        );
        let (g1, s2, c2, g2) = (
            &modu[si][2 * d..3 * d],
            &modu[si][3 * d..4 * d],
            &modu[si][4 * d..5 * d],
            &modu[si][5 * d..6 * d],
        );
        let mut acc = vec![0.0f32; n * d];
        for t in 0..n {
            for i in 0..d {
                acc[t * d + i] = out[si][t * d + i] + g1[i] * proj[t * d + i];
            }
        }
        // Second modulation chain re-normalizes with the weightless norm2.
        let mut h2 = vec![0.0f32; n * d];
        for t in 0..n {
            let row = &acc[t * d..(t + 1) * d];
            let normed = layernorm(row, 1e-6);
            for i in 0..d {
                h2[t * d + i] = (1.0 + c2[i]) * normed[i] + s2[i];
            }
        }
        let m0 = linear(p("mlp.0.weight"), Some(p("mlp.0.bias")), &h2, d, n);
        let m0g: Vec<f32> = m0.iter().map(|v| gelu_tanh(*v)).collect();
        let m1 = linear(p("mlp.2.weight"), Some(p("mlp.2.bias")), &m0g, 4 * d, n);
        for t in 0..n {
            for i in 0..d {
                out[si][t * d + i] = acc[t * d + i] + g2[i] * m1[t * d + i];
            }
        }
    }
    (out[0].clone(), out[1].clone())
}

/// Single-stream block (BFL `SingleStreamBlock`, commit 87f6fff): 3-chunk
/// modulation of `silu(vec)`, weightless pre-norm, FUSED `linear1` (qkv +
/// mlp-in), per-head QK norm, self-attention on the full concat, fused
/// `linear2` (attn-proj + mlp-out), gated residual.
#[allow(clippy::too_many_arguments)]
pub fn single_block(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    b: usize,
    fused: &[f32],
    vec: &[f32],
    n_img: usize,
    grid: (usize, usize),
    act: MlpAct,
) -> Vec<f32> {
    single_block_parts(cfg, w, b, fused, vec, n_img, grid, act).final_out
}

/// Single-block internals (the parity-bisection dump and the forward share
/// one implementation; the forward only reads `out`).
pub struct SingleBlockParts {
    pub x_mod: Vec<f32>,
    pub att: Vec<f32>,
    pub mlp_g: Vec<f32>,
    pub cat: Vec<f32>,
    /// linear2 raw output, pre gate (bisection only)
    pub out: Vec<f32>,
    /// post-gate residual — the block's actual output
    pub final_out: Vec<f32>,
}

pub fn single_block_parts(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    b: usize,
    fused: &[f32],
    vec: &[f32],
    n_img: usize,
    grid: (usize, usize),
    act: MlpAct,
) -> SingleBlockParts {
    let d = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let f = 4 * d;
    let n_all = fused.len() / d;
    let p = |s: &str| w.get(&format!("single_blocks.{b}.{s}"));
    let silu_vec: Vec<f32> = vec.iter().map(|v| silu(*v)).collect();

    let modu = linear(
        p("modulation.lin.weight"),
        Some(p("modulation.lin.bias")),
        &silu_vec,
        d,
        1,
    );
    let (s1, c1, g1) = (&modu[0..d], &modu[d..2 * d], &modu[2 * d..3 * d]);

    let mut x_mod = vec![0.0f32; n_all * d];
    for t in 0..n_all {
        let row = &fused[t * d..(t + 1) * d];
        let normed = layernorm(row, 1e-6);
        for i in 0..d {
            x_mod[t * d + i] = (1.0 + c1[i]) * normed[i] + s1[i];
        }
    }
    let fused_proj = linear(
        p("linear1.weight"),
        Some(p("linear1.bias")),
        &x_mod,
        d,
        n_all,
    );
    let mut qkv_part = vec![0.0f32; n_all * 3 * d];
    let mut mlp_part = vec![0.0f32; n_all * f];
    for t in 0..n_all {
        let base = t * 7 * d;
        qkv_part[t * 3 * d..(t + 1) * 3 * d].copy_from_slice(&fused_proj[base..base + 3 * d]);
        mlp_part[t * f..(t + 1) * f].copy_from_slice(&fused_proj[base + 3 * d..base + 7 * d]);
    }
    let (q, k, v) = split_qkv(&qkv_part, n_all, heads, hd);
    let (q, k) = qk_rmsnorm(
        &q,
        &k,
        heads,
        hd,
        p("norm.query_norm.scale"),
        p("norm.key_norm.scale"),
    );
    let att = attention(
        &q,
        &k,
        &v,
        n_all,
        n_all,
        hd,
        heads,
        n_img,
        n_img,
        grid,
        cfg.axes_dim,
        cfg.theta,
    );
    let mlp_g: Vec<f32> = match act {
        MlpAct::GeluTanh => mlp_part.iter().map(|x| gelu_tanh(*x)).collect(),
        MlpAct::Silu => mlp_part.iter().map(|x| silu(*x)).collect(),
    };
    let mut cat = vec![0.0f32; n_all * (d + f)];
    for t in 0..n_all {
        cat[t * (d + f)..t * (d + f) + d].copy_from_slice(&att[t * d..(t + 1) * d]);
        cat[t * (d + f) + d..(t + 1) * (d + f)].copy_from_slice(&mlp_g[t * f..(t + 1) * f]);
    }
    let out = linear(
        p("linear2.weight"),
        Some(p("linear2.bias")),
        &cat,
        d + f,
        n_all,
    );
    let mut next = fused.to_vec();
    for t in 0..n_all {
        for i in 0..d {
            next[t * d + i] += g1[i] * out[t * d + i];
        }
    }
    SingleBlockParts {
        x_mod,
        att,
        mlp_g,
        cat,
        out,
        final_out: next,
    }
}

/// Split a `n × 3d` qkv projection into q, k, v (`n × heads × head_dim`).
fn split_qkv(qkv: &[f32], n: usize, heads: usize, hd: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = heads * hd;
    let mut q = vec![0.0f32; n * d];
    let mut k = vec![0.0f32; n * d];
    let mut v = vec![0.0f32; n * d];
    for t in 0..n {
        for i in 0..d {
            q[t * d + i] = qkv[t * 3 * d + i];
            k[t * d + i] = qkv[t * 3 * d + d + i];
            v[t * d + i] = qkv[t * 3 * d + 2 * d + i];
        }
    }
    (q, k, v)
}

// ─── FLUX.2 Klein forward ───────────────────────────────────────────────────
//
// Bias-free throughout (`linear(.., None, ..)`), SwiGLU MLPs (`silu(first
// half) * second half` instead of GELU-tanh), one SHARED `silu(temb)`
// modulation vector feeding three separate linears (double-img, double-txt,
// single) instead of per-block `mod.lin`, fused single-block qkv+mlp
// (`to_qkv_mlp_proj`, width `3d + 2f`) and fused attn-proj+mlp-out
// (`to_out`, width `d + f`), and 4-axis RoPE ids (`img_ids`, default the
// derived `(0, row, col, 0)` grid) instead of the 3-axis FLUX.1 grid. The
// final head is diffusers `AdaLayerNormContinuous(bias=False)`, chunk order
// always `(scale, shift)` — `input.final_order` and `input.mlp_act` are
// FLUX.1-only knobs and are ignored on this branch.
fn forward_parts_flux2(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    input: &FluxForwardInput,
) -> Vec<(String, Vec<f32>)> {
    let d = cfg.hidden_size;
    let patch_in = cfg.patch_in();
    let n_txt = input.txt.len() / cfg.txt_hidden_dim;
    let n_img = input.img.len() / patch_in;
    // The image stream is the generated grid PLUS any reference-image tokens
    // (the FLUX.2 edit path), so `grid` alone only accounts for every token
    // when the caller left the ids implicit. With explicit ids the ids list
    // IS the authority on the token count.
    debug_assert!(
        match &input.img_ids {
            Some(ids) => ids.len() == n_img,
            None => input.grid.0 * input.grid.1 == n_img,
        },
        "flux2 forward: img_ids/grid must cover all {n_img} image tokens"
    );
    let mut parts: Vec<(String, Vec<f32>)> = Vec::new();

    // temb = linear_2(silu(linear_1(sincos(t * 1000)))), no biases.
    let te = timestep_embedding(input.timestep, TS_EMBED_DIM, 10000.0, 1000.0);
    let h1 = linear(
        w.get("time_guidance_embed.timestep_embedder.linear_1.weight"),
        None,
        &te,
        TS_EMBED_DIM,
        1,
    );
    let h1: Vec<f32> = h1.iter().map(|v| silu(*v)).collect();
    let temb = linear(
        w.get("time_guidance_embed.timestep_embedder.linear_2.weight"),
        None,
        &h1,
        d,
        1,
    );
    parts.push(("temb".into(), temb.clone()));
    let stemb: Vec<f32> = temb.iter().map(|v| silu(*v)).collect();
    let mod_img = linear(
        w.get("double_stream_modulation_img.linear.weight"),
        None,
        &stemb,
        d,
        1,
    ); // 6d
    let mod_txt = linear(
        w.get("double_stream_modulation_txt.linear.weight"),
        None,
        &stemb,
        d,
        1,
    ); // 6d
    let mod_single = linear(
        w.get("single_stream_modulation.linear.weight"),
        None,
        &stemb,
        d,
        1,
    ); // 3d

    let mut img = linear(
        w.get("x_embedder.weight"),
        None,
        &input.img,
        patch_in,
        n_img,
    );
    let mut txt = linear(
        w.get("context_embedder.weight"),
        None,
        &input.txt,
        cfg.txt_hidden_dim,
        n_txt,
    );
    parts.push(("img_in".into(), img.clone()));
    parts.push(("txt_in".into(), txt.clone()));

    // The RoPE id table covers the FULL text-first concat: Klein rotates its
    // text rows too, by token index on axis 3 (see [`text_ids`]). `img_ids`
    // is the image half only, so the text half is prepended here.
    let mut ids = text_ids(n_txt);
    ids.extend(
        input
            .img_ids
            .clone()
            .unwrap_or_else(|| rope_ids_for_grid(input.grid, 0.0)),
    );
    for b in 0..cfg.num_layers {
        let (ni, nt) = double_block_flux2(cfg, w, b, &img, &txt, &mod_img, &mod_txt, n_img, &ids);
        img = ni;
        txt = nt;
        parts.push((format!("double_{b}_img"), img.clone()));
        parts.push((format!("double_{b}_txt"), txt.clone()));
    }
    let mut fused = txt.clone();
    fused.extend_from_slice(&img);
    for b in 0..cfg.num_single_layers {
        fused = single_block_flux2(cfg, w, b, &fused, &mod_single, &ids);
    }
    parts.push(("single_concat".into(), fused.clone()));
    let img_only: Vec<f32> = fused[n_txt * d..].to_vec();

    // norm_out: AdaLayerNormContinuous(bias=False): (scale, shift) = chunk2(W silu(temb)).
    let adain = linear(w.get("norm_out.linear.weight"), None, &stemb, d, 1);
    let mut h = vec![0.0f32; img_only.len()];
    for t in 0..n_img {
        let normed = layernorm(&img_only[t * d..(t + 1) * d], 1e-6);
        for i in 0..d {
            let (scale, shift) = (adain[i], adain[d + i]);
            h[t * d + i] = (1.0 + scale) * normed[i] + shift;
        }
    }
    let out = linear(w.get("proj_out.weight"), None, &h, d, n_img);
    parts.push(("final".into(), out));
    parts
}

/// FLUX.2 Klein dual-stream block: shared-`stemb` per-stream modulation (6
/// chunks `shift_a, scale_a, gate_a, shift_m, scale_m, gate_m`), fused qkv
/// per stream, per-head QK-RMSNorm, joint self-attention over the text-first
/// concat with 4-axis RoPE ids, gated residual, SwiGLU-gated re-normalized
/// MLP. Bias-free throughout.
#[allow(clippy::too_many_arguments)]
fn double_block_flux2(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    b: usize,
    img: &[f32],
    txt: &[f32],
    mod_img: &[f32],
    mod_txt: &[f32],
    n_img: usize,
    ids: &[[f32; 4]],
) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.hidden_size;
    let f = cfg.mlp_width();
    let heads = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let n_txt = txt.len() / d;
    let p = |k: &str| w.get(&format!("transformer_blocks.{b}.{k}"));
    // chunk order: shift_a, scale_a, gate_a, shift_m, scale_m, gate_m
    let prep = |x: &[f32], n: usize, m: &[f32], qkv_key: &str, qn: &str, kn: &str| {
        let mut h = vec![0.0f32; n * d];
        for t in 0..n {
            let normed = layernorm(&x[t * d..(t + 1) * d], 1e-6);
            for i in 0..d {
                h[t * d + i] = (1.0 + m[d + i]) * normed[i] + m[i];
            }
        }
        let qkv = linear(p(qkv_key), None, &h, d, n);
        let (q, k, v) = split_qkv(&qkv, n, heads, hd);
        let (q, k) = qk_rmsnorm(&q, &k, heads, hd, p(qn), p(kn));
        (q, k, v)
    };
    let (tq, tk, tv) = prep(
        txt,
        n_txt,
        mod_txt,
        "attn.add_qkv.weight",
        "attn.norm_added_q.weight",
        "attn.norm_added_k.weight",
    );
    let (iq, ik, iv) = prep(
        img,
        n_img,
        mod_img,
        "attn.qkv.weight",
        "attn.norm_q.weight",
        "attn.norm_k.weight",
    );
    let mut q = tq;
    q.extend_from_slice(&iq);
    let mut k = tk;
    k.extend_from_slice(&ik);
    let mut v = tv;
    v.extend_from_slice(&iv);
    let n_all = n_txt + n_img;
    let att = attention_ids(&q, &k, &v, n_all, hd, heads, ids, cfg.axes_dim, cfg.theta);
    let stream = |x: &[f32],
                  n: usize,
                  att_slice: &[f32],
                  m: &[f32],
                  proj: &str,
                  ff_in: &str,
                  ff_out: &str|
     -> Vec<f32> {
        let proj_out = linear(p(proj), None, att_slice, d, n);
        let mut acc = x.to_vec();
        for t in 0..n {
            for i in 0..d {
                acc[t * d + i] += m[2 * d + i] * proj_out[t * d + i];
            }
        }
        let mut h2 = vec![0.0f32; n * d];
        for t in 0..n {
            let normed = layernorm(&acc[t * d..(t + 1) * d], 1e-6);
            for i in 0..d {
                h2[t * d + i] = (1.0 + m[4 * d + i]) * normed[i] + m[3 * d + i];
            }
        }
        let hh = linear(p(ff_in), None, &h2, d, n); // [n, 2f]
        let mut g = vec![0.0f32; n * f];
        for t in 0..n {
            for j in 0..f {
                g[t * f + j] = silu(hh[t * 2 * f + j]) * hh[t * 2 * f + f + j];
            }
        }
        let o = linear(p(ff_out), None, &g, f, n);
        for t in 0..n {
            for i in 0..d {
                acc[t * d + i] += m[5 * d + i] * o[t * d + i];
            }
        }
        acc
    };
    let txt_next = stream(
        txt,
        n_txt,
        &att[..n_txt * d],
        mod_txt,
        "attn.to_add_out.weight",
        "ff_context.linear_in.weight",
        "ff_context.linear_out.weight",
    );
    let img_next = stream(
        img,
        n_img,
        &att[n_txt * d..],
        mod_img,
        "attn.to_out.0.weight",
        "ff.linear_in.weight",
        "ff.linear_out.weight",
    );
    (img_next, txt_next)
}

/// FLUX.2 Klein single-stream block: shared-`stemb` modulation (3 chunks
/// `shift, scale, gate`), FUSED `to_qkv_mlp_proj` (qkv + SwiGLU-mlp-in,
/// width `3d + 2f`), per-head QK-RMSNorm, self-attention on the full
/// text-first concat with 4-axis RoPE ids, FUSED `to_out` (attn-proj +
/// mlp-out, width `d + f`), gated residual. Bias-free throughout.
fn single_block_flux2(
    cfg: &FluxDiffusionConfig,
    w: &FluxWeights,
    b: usize,
    fused: &[f32],
    m: &[f32],
    ids: &[[f32; 4]],
) -> Vec<f32> {
    let d = cfg.hidden_size;
    let f = cfg.mlp_width();
    let heads = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let n_all = fused.len() / d;
    let p = |k: &str| w.get(&format!("single_transformer_blocks.{b}.{k}"));
    let mut x_mod = vec![0.0f32; n_all * d];
    for t in 0..n_all {
        let normed = layernorm(&fused[t * d..(t + 1) * d], 1e-6);
        for i in 0..d {
            x_mod[t * d + i] = (1.0 + m[d + i]) * normed[i] + m[i];
        }
    }
    let width = 3 * d + 2 * f;
    let proj = linear(p("attn.to_qkv_mlp_proj.weight"), None, &x_mod, d, n_all); // [n_all, width]
    let mut qkv = vec![0.0f32; n_all * 3 * d];
    let mut g = vec![0.0f32; n_all * f];
    for t in 0..n_all {
        let row = &proj[t * width..(t + 1) * width];
        qkv[t * 3 * d..(t + 1) * 3 * d].copy_from_slice(&row[..3 * d]);
        for j in 0..f {
            g[t * f + j] = silu(row[3 * d + j]) * row[3 * d + f + j];
        }
    }
    let (q, k, v) = split_qkv(&qkv, n_all, heads, hd);
    let (q, k) = qk_rmsnorm(
        &q,
        &k,
        heads,
        hd,
        p("attn.norm_q.weight"),
        p("attn.norm_k.weight"),
    );
    let att = attention_ids(&q, &k, &v, n_all, hd, heads, ids, cfg.axes_dim, cfg.theta);
    let mut cat = vec![0.0f32; n_all * (d + f)];
    for t in 0..n_all {
        cat[t * (d + f)..t * (d + f) + d].copy_from_slice(&att[t * d..(t + 1) * d]);
        cat[t * (d + f) + d..(t + 1) * (d + f)].copy_from_slice(&g[t * f..(t + 1) * f]);
    }
    let out = linear(p("attn.to_out.weight"), None, &cat, d + f, n_all);
    let mut next = fused.to_vec();
    for t in 0..n_all {
        for i in 0..d {
            next[t * d + i] += m[2 * d + i] * out[t * d + i];
        }
    }
    next
}

// ─── Self-parity fixtures ──────────────────────────────────────────────────

/// Run the full forward with synthetic weights and deterministic inputs;
/// returns the output buffer. Shared by tests and the parity example.
pub fn self_forward(cfg: &FluxDiffusionConfig) -> Vec<f32> {
    let w = FluxWeights::synthetic(cfg);
    let (grid_h, grid_w) = (4usize, 4usize);
    let n_img = grid_h * grid_w;
    let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
    let mut img = Vec::with_capacity(n_img * patch_in);

    let mut x = 0.0f32;
    for i in 0..n_img * patch_in {
        x = (x * 1.0001 + synth_val(SYNTH_SEED ^ 0xABCD, i as u64) * 0.5).sin() * 0.5;
        img.push(x);
    }
    let txt: Vec<f32> = (0..2 * cfg.txt_hidden_dim)
        .map(|i| synth_val(SYNTH_SEED ^ 0x55AA, i as u64) * 0.05)
        .collect();
    let pooled: Vec<f32> = (0..cfg.pooled_projection_dim)
        .map(|i| synth_val(SYNTH_SEED ^ 0x0FF0, i as u64) * 0.1)
        .collect();
    let input = FluxForwardInput {
        timestep: 0.75,
        pooled,
        guidance: if cfg.guidance_embed_dim > 0 {
            Some(1.0)
        } else {
            None
        },
        txt,
        img,
        grid: (grid_h, grid_w),
        mlp_act: MlpAct::GeluTanh,
        final_order: FinalAdaLNOrder::ShiftScale,
        img_ids: None,
    };
    forward(cfg, &w, &input)
}

/// "Run it twice, get identical bytes" — the determinism contract.
#[test]
fn forward_is_deterministic() {
    let cfg = test_cfg();
    let a = self_forward(&cfg);
    let b = self_forward(&cfg);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(*x, *y, "self-parity forward is not bit-deterministic");
    }
}

#[test]
fn forward_output_is_finite_and_shaped() {
    let cfg = test_cfg();
    let out = self_forward(&cfg);
    assert!(!out.is_empty());
    assert_eq!(
        out.len(),
        cfg.patch_size * cfg.patch_size * cfg.latent_channels * 16
    );
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
}

#[test]
fn conditioning_vector_is_additive() {
    // FLUX conditioning has no concatenation: vec = time + pooled (+guidance).
    // Deterministic inputs must feed a 3072-wide `vec` into the blocks; a
    // single-element change in the pooled condition changes the whole output.
    let cfg = test_cfg();
    let w = FluxWeights::synthetic(&cfg);
    let mut base = self_forward(&cfg);
    let _ = &mut base;
    // Rebuild inputs with a perturbed pooled vector.
    let (grid_h, grid_w) = (4usize, 4usize);
    let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
    let mut x = 0.0f32;
    let mut img = Vec::new();
    for i in 0..grid_h * grid_w * patch_in {
        x = (x * 1.0001 + synth_val(SYNTH_SEED ^ 0xABCD, i as u64) * 0.5).sin() * 0.5;
        img.push(x);
    }
    let txt: Vec<f32> = (0..2 * cfg.txt_hidden_dim)
        .map(|i| synth_val(SYNTH_SEED ^ 0x55AA, i as u64) * 0.05)
        .collect();
    let mut pooled: Vec<f32> = (0..cfg.pooled_projection_dim)
        .map(|i| synth_val(SYNTH_SEED ^ 0x0FF0, i as u64) * 0.1)
        .collect();
    pooled[0] += 1.0;
    let out = forward(
        &cfg,
        &w,
        &FluxForwardInput {
            timestep: 0.75,
            pooled,
            guidance: None,
            txt,
            img,
            grid: (grid_h, grid_w),
            mlp_act: MlpAct::GeluTanh,
            final_order: FinalAdaLNOrder::ShiftScale,
            img_ids: None,
        },
    );
    assert_ne!(
        &base[..],
        &out[..],
        "pooled perturbation must change the output"
    );
}

#[test]
fn attention_image_rows_rope_but_text_rows_do_not_move() {
    // The attention path must rotate ONLY the trailing image rows. Regression:
    // roping used to be applied over a heads count derived from position
    // counts, which corrupted text rows whenever a stream had more text rows.
    let (heads, hd) = (1usize, 8usize);
    let (n_q, n_kv) = (6usize, 6usize);
    // 4 text rows first, then 2 image rows (BFL concat order).
    let q: Vec<f32> = (0..n_q * hd)
        .map(|i| (i / hd) as f32 * 0.1 + (i % hd) as f32 * 0.01)
        .collect();
    let v: Vec<f32> = (0..n_kv * hd).map(|i| 1000.0 + (i / hd) as f32).collect();
    let with_rope = attention(
        &q,
        &q,
        &v,
        n_q,
        n_kv,
        hd,
        heads,
        2,
        2,
        (4, 2),
        [2, 3, 3, 0],
        10000.0,
    );
    // Same k (both runs rope the trailing 2 image k rows); only the query
    // roping differs, so text queries must come out byte-identical.
    let img_q_unroped = attention(
        &q,
        &q,
        &v,
        n_q,
        n_kv,
        hd,
        heads,
        0,
        2,
        (4, 2),
        [2, 3, 3, 0],
        10000.0,
    );
    // Text rows (0..4) byte-identical: only the image rows' queries rotate.
    for t in 0..4 {
        for i in 0..hd {
            assert_eq!(
                with_rope[t * hd + i],
                img_q_unroped[t * hd + i],
                "text row {t} moved"
            );
        }
    }
    assert_ne!(
        &with_rope[4 * hd..],
        &img_q_unroped[4 * hd..],
        "image rows must be RoPE'd"
    );
}

#[test]
fn rope_2d_applies_standard_rotation() {
    // Single head, hd=2, with the whole 2-dim budget on axis 1 (row): a row
    // position of 1 then yields exactly a 1-radian rotation. BFL's
    // `flux/math.py` applies the STANDARD rotation [[c,-s],[s,c]], so pair
    // (4,5) by +1 rad must land on (4c-5s, 4s+5c) = (-2.0461454, 6.0673952).
    // Regression: the reference previously applied the conjugate rotation.
    let mut x = vec![0.0f32, 0.0, 4.0, 5.0]; // rows 0 (identity) and 1 (=(4,5))
    rope_2d(&mut x, 2, 1, 2, (2, 1), [0, 2, 0, 0], 10000.0);
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    assert!(
        (x[2] - (4.0 * c - 5.0 * s)).abs() < 1e-6,
        "rotated a = {}",
        x[2]
    );
    assert!(
        (x[3] - (4.0 * s + 5.0 * c)).abs() < 1e-6,
        "rotated b = {}",
        x[3]
    );
    assert_eq!(&x[0..2], &[0.0, 0.0], "position-0 rows must be identity");
}

#[test]
fn timestep_embedding_matches_bfl_formula() {
    // sin/cos of t·1000·max_period^(-2i/128), cos-first then sin-second.
    let emb = timestep_embedding(0.75, 256, 10000.0, 1000.0);
    assert_eq!(emb.len(), 256);
    let f0 = (-(10000.0f64.ln()) * 0.0 / 128.0).exp() as f32;
    assert!((emb[0] - (0.75 * 1000.0 * f0).cos()).abs() < 1e-6);
    assert!((emb[128] - (0.75 * 1000.0 * f0).sin()).abs() < 1e-6);
}

/// Pin captured on the parent commit (ab99ad38, before the Flux2 forward
/// branch existed) via `cargo test -p hipfire-arch-diffusion --lib
/// forward_is_deterministic -- --nocapture` with a temporary
/// `eprintln!("PIN_SUM {}", a.iter().map(|v| *v as f64).sum::<f64>())`
/// inside that test: printed `PIN_SUM 7.701007844880223`.
#[cfg(test)]
const PINNED_FLUX1_SELF_FORWARD_SUM: f64 = 7.701007844880223;

#[test]
fn flux1_forward_is_unchanged_by_the_family_branch() {
    // Pin: the Flux1 self_forward output before this task equals the output after.
    let cfg = test_cfg();
    let out = self_forward(&cfg);
    let checksum: f64 = out.iter().map(|v| *v as f64).sum();
    assert!(
        (checksum - PINNED_FLUX1_SELF_FORWARD_SUM).abs() < 1e-3,
        "{checksum}"
    );
}

/// Shared input builder for the Flux2 forward tests: deterministic sin/cos
/// txt and img streams sized to `cfg` and `grid`, `img_ids: None` (derived
/// grid ids) by default.
#[cfg(test)]
fn flux2_input(cfg: &FluxDiffusionConfig, grid: (usize, usize), n_txt: usize) -> FluxForwardInput {
    let n_img = grid.0 * grid.1;
    FluxForwardInput {
        timestep: 0.75,
        pooled: vec![],
        guidance: None,
        txt: (0..n_txt * cfg.txt_hidden_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect(),
        img: (0..n_img * cfg.patch_in())
            .map(|i| (i as f32 * 0.02).cos())
            .collect(),
        grid,
        mlp_act: MlpAct::GeluTanh,
        final_order: FinalAdaLNOrder::ScaleShift,
        img_ids: None,
    }
}

#[test]
fn flux2_forward_is_finite_shaped_and_deterministic() {
    let cfg = test_cfg_flux2();
    let w = FluxWeights::synthetic(&cfg);
    let grid = (2, 4);
    let n_img = grid.0 * grid.1;
    let input = flux2_input(&cfg, grid, 3);
    let a = forward(&cfg, &w, &input);
    let b = forward(&cfg, &w, &input);
    assert_eq!(a.len(), n_img * cfg.patch_in());
    assert!(a.iter().all(|v| v.is_finite()));
    assert_eq!(a, b);
}

#[test]
fn flux2_explicit_ids_equal_to_the_derived_grid_give_the_same_output() {
    let cfg = test_cfg_flux2();
    let w = FluxWeights::synthetic(&cfg);
    let grid = (2, 3);
    let mut input = flux2_input(&cfg, grid, 3);
    let derived = forward(&cfg, &w, &input);
    input.img_ids = Some(rope_ids_for_grid(grid, 0.0));
    let explicit = forward(&cfg, &w, &input);
    assert_eq!(derived, explicit);
}

#[test]
fn flux2_text_rows_are_rotated_by_token_index() {
    // ComfyUI `model_detection` gives `image_model == "flux2"` the config
    // `txt_ids_dims = [3]`, so `Flux._forward` writes `linspace(0, L-1)` into
    // axis 3 of `txt_ids` — Klein's text tokens ARE positioned. With the
    // all-zero text ids of FLUX.1 the IMAGE-stream output would be invariant
    // under a permutation of the text tokens (attention's softmax over keys
    // is order-free and every text row is embedded identically). It is not.
    let cfg = test_cfg_flux2();
    let w = FluxWeights::synthetic(&cfg);
    let grid = (2, 3);
    let td = cfg.txt_hidden_dim;
    let mut input = flux2_input(&cfg, grid, 3);
    let base = forward(&cfg, &w, &input);
    for i in 0..td {
        input.txt.swap(i, td + i); // swap text tokens 0 and 1
    }
    let swapped = forward(&cfg, &w, &input);
    assert_eq!(base.len(), swapped.len());
    assert_ne!(base, swapped);
}

#[test]
fn flux2_text_ids_are_zero_but_the_token_index_on_axis_3() {
    assert_eq!(
        text_ids(3),
        vec![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0]
        ]
    );
    assert!(text_ids(0).is_empty());
}

#[test]
fn flux2_reference_time_id_changes_the_output() {
    // Same tokens, time axis 10 instead of 0: the 4-axis RoPE must move them.
    let cfg = test_cfg_flux2();
    let w = FluxWeights::synthetic(&cfg);
    let grid = (2, 3);
    let mut input = flux2_input(&cfg, grid, 3);
    input.img_ids = Some(rope_ids_for_grid(grid, 0.0));
    let t0 = forward(&cfg, &w, &input);
    input.img_ids = Some(rope_ids_for_grid(grid, 10.0));
    let t10 = forward(&cfg, &w, &input);
    assert_ne!(t0, t10);
}

#[cfg(test)]
fn test_cfg() -> FluxDiffusionConfig {
    FluxDiffusionConfig::from_json(&serde_json::json!({
        "hidden_size": 32,
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "axes_dims_rope": [4, 4, 8],
        "num_layers": 2,
        "num_single_layers": 1,
        "joint_attention_dim": 8,
        "pooled_projection_dim": 8,
        "latent_channels": 4,
        "patch_size": 1,
    }))
    .unwrap()
}

#[cfg(test)]
pub(crate) fn test_cfg_flux2() -> FluxDiffusionConfig {
    FluxDiffusionConfig::from_json(&serde_json::json!({
        "_class_name": "Flux2Transformer2DModel",
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "axes_dims_rope": [4, 4, 4, 4],
        "num_layers": 2,
        "num_single_layers": 1,
        "joint_attention_dim": 24,
        "in_channels": 8,
        "patch_size": 1,
        "mlp_ratio": 3.0,
        "rope_theta": 2000,
    }))
    .unwrap()
}

/// On-disk fixture writers shared by this crate's load tests — the flux
/// `load_tests` below and the tiny end-to-end pipes in `pipeline.rs`.
///
/// One home for "write a safetensors file by hand" so a fixture for a new
/// component (Qwen3, the VAE, a FLUX.2 transformer) reuses the same writer
/// and the same deterministic [`synth_val`] value stream rather than growing
/// a second, subtly different one.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::{synth_val, FluxPlan, SYNTH_SEED};
    use crate::config::FluxDiffusionConfig;
    use crate::manifest;
    use serde_json::json;
    use std::io::Write;
    use std::path::Path;

    /// One safetensors entry: `(name, little-endian bytes, dtype, shape)`.
    pub(crate) type NamedTensor = (String, Vec<u8>, String, Vec<usize>);

    /// Serialize a minimal safetensors file by hand (8-byte LE header length +
    /// JSON header + concatenated little-endian tensor bytes) so the tests
    /// have no writer dependency. Callers may hand-write one key with a
    /// different (dtype, shape, bytes-per-elem) as long as the file stays
    /// self-consistent — the safetensors reader rejects headers whose byte
    /// counts disagree with their shapes.
    pub(crate) fn write_safetensors(path: &Path, tensors: &[NamedTensor]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        let mut blobs: Vec<(&str, Vec<u8>)> = Vec::new();
        for (name, data, dtype, shape) in tensors {
            let start = offset;
            let end = start + data.len();
            let mut meta = serde_json::Map::new();
            meta.insert("dtype".into(), dtype.clone().into());
            meta.insert(
                "shape".into(),
                serde_json::Value::Array(shape.iter().map(|&s| s.into()).collect()),
            );
            meta.insert("data_offsets".into(), json!([start, end]));
            header.insert(name.clone(), meta.into());
            blobs.push((name.as_str(), data.clone()));
            offset = end;
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_json.as_bytes()).unwrap();
        for (_, blob) in &blobs {
            file.write_all(blob).unwrap();
        }
    }

    /// `n` BF16 words of the deterministic [`synth_val`] stream starting at
    /// `*idx`, scaled the same way [`FluxWeights::synthetic`] scales it, with
    /// `*idx` advanced. bf16 IS the high 16 bits of the f32 pattern, so the
    /// conversion is a truncation.
    pub(crate) fn bf16_blob(seed: u64, idx: &mut u64, n: usize) -> Vec<u8> {
        let data: Vec<u8> = (0..n as u64)
            .flat_map(|i| {
                let bits = (synth_val(seed, *idx + i) * 0.05).to_bits();
                (((bits >> 16) & 0xFFFF) as u16).to_le_bytes()
            })
            .collect();
        *idx += n as u64;
        data
    }

    /// safetensors `shape` for a `[rows, cols]` tensor: 1-D for vectors.
    pub(crate) fn shape_of(rows: usize, cols: usize) -> Vec<usize> {
        if cols == 1 {
            vec![rows]
        } else {
            vec![rows, cols]
        }
    }

    /// Every CHECKPOINT tensor `plan` names for `cfg`, BF16, deterministic.
    ///
    /// Values are arbitrary but reproducible — a fixture built this way
    /// exercises the naming and the row-concatenation order, not numerics.
    pub(crate) fn plan_tensors(cfg: &FluxDiffusionConfig, plan: &FluxPlan) -> Vec<NamedTensor> {
        let mut named: Vec<(String, usize, usize)> = Vec::new();
        for key in manifest::expected_flux_keys(cfg) {
            for p in plan.parts(&key.name).unwrap() {
                named.push((p.name.clone(), p.rows, p.cols));
            }
        }
        // The same source tensor never feeds two manifest keys, but sort and
        // dedup anyway so a future mapping change cannot write a duplicate
        // safetensors entry and fail opaquely.
        named.sort();
        named.dedup();
        let mut idx = 0u64;
        named
            .into_iter()
            .map(|(name, rows, cols)| {
                let data = bf16_blob(SYNTH_SEED, &mut idx, rows * cols);
                (name, data, "BF16".to_string(), shape_of(rows, cols))
            })
            .collect()
    }

    /// A synthetic checkpoint in `plan`'s key layout, plus the `config.json`
    /// `SafetensorsSource::open` requires.
    pub(crate) fn write_plan_checkpoint(
        dir: &Path,
        cfg: &FluxDiffusionConfig,
        plan: &FluxPlan,
        config_json: &serde_json::Value,
    ) {
        std::fs::write(dir.join("config.json"), config_json.to_string()).unwrap();
        write_safetensors(&dir.join("model.safetensors"), &plan_tensors(cfg, plan));
    }

    /// A fresh, empty temp directory named after `tag` and this process.
    pub(crate) fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hipfire-flux-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}

#[cfg(test)]
mod load_tests {
    use super::test_fixtures::{shape_of as st_shape, temp_dir, write_safetensors};
    use super::*;
    use crate::manifest::FluxKey;
    use serde_json::json;
    use std::path::Path;

    fn shape_of(k: &FluxKey) -> Vec<usize> {
        st_shape(k.rows, k.cols)
    }

    /// All manifest keys of `cfg` as BF16 bytes, matching
    /// [`FluxWeights::synthetic`] value-for-value (shared running index).
    fn checkpoint_tensors(
        cfg: &FluxDiffusionConfig,
        rewrite: Option<(&str, &str, Vec<usize>, usize)>,
    ) -> Vec<(String, Vec<u8>, String, Vec<usize>)> {
        let mut idx = 0u64;
        let mut out = Vec::new();
        for k in manifest::expected_flux_keys(cfg) {
            let n = k.rows * k.cols;
            let (dtype, shape, elem_bytes) = match &rewrite {
                Some((name, dt, sh, eb)) if *name == k.name => (dt.to_string(), sh.clone(), *eb),
                _ => ("BF16".to_string(), shape_of(&k), 2),
            };
            let count = shape.iter().product::<usize>();
            let data: Vec<u8> = (0..count as u64)
                .flat_map(|i| {
                    let bits = (synth_val(SYNTH_SEED, idx + i) * 0.05).to_bits();
                    let mut b = vec![0u8; elem_bytes];
                    // bf16 lives in the HIGH 16 bits of the f32 pattern; any
                    // wider dtype (tests only) zero-pads beyond those bytes.
                    for j in 0..elem_bytes {
                        b[j] = if j < 2 {
                            ((bits >> (16 + 8 * j)) & 0xFF) as u8
                        } else {
                            0
                        };
                    }
                    b
                })
                .collect();
            idx += n as u64;
            out.push((k.name, data, dtype, shape));
        }
        out
    }

    /// A synthetic checkpoint in the DIFFUSERS key layout: every source part
    /// the plan names, as its own BF16 tensor. Values are arbitrary but
    /// deterministic — this fixture exists to exercise the naming and the
    /// row-concatenation order, not the numerics.
    fn write_diffusers_checkpoint(dir: &Path, cfg: &FluxDiffusionConfig) {
        super::test_fixtures::write_plan_checkpoint(
            dir,
            cfg,
            &FluxPlan::diffusers(cfg),
            &json!({ "model_type": "flux" }),
        );
    }

    fn write_checkpoint(dir: &Path, cfg: &FluxDiffusionConfig) {
        std::fs::write(
            dir.join("config.json"),
            json!({ "model_type": "flux" }).to_string(),
        )
        .unwrap();
        let tensors = checkpoint_tensors(cfg, None);
        write_safetensors(&dir.join("model.safetensors"), &tensors);
    }

    fn open_source(dir: &Path) -> Box<dyn ModelSource> {
        hipfire_runtime::safetensors_source::SafetensorsSource::open(dir)
            .map(|s| Box::new(s) as Box<dyn ModelSource>)
            .map_err(|e| panic!("safetensors open failed: {e}"))
            .unwrap()
    }

    #[test]
    fn host_load_round_trips_bf16_checkpoint() {
        let cfg = test_cfg();
        let dir = temp_dir("roundtrip");
        write_checkpoint(&dir, &cfg);
        let src = open_source(&dir);
        let loaded = load_weights(&*src, &cfg).unwrap();
        let expected = FluxWeights::synthetic(&cfg);
        assert_eq!(loaded.tensors.len(), expected.tensors.len());
        for (name, t) in &expected.tensors {
            let got = loaded
                .tensors
                .get(name)
                .unwrap_or_else(|| panic!("missing {name} in loaded weights"));
            assert_eq!(
                (got.rows, got.cols),
                (t.rows, t.cols),
                "shape drift in {name}"
            );
            let want: Vec<f32> = t
                .data
                .iter()
                .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
                .collect();
            assert_eq!(got.data, want, "decode mismatch in {name}");
        }
    }

    /// The streaming upload must produce the SAME f16 words the old path
    /// produced by decoding to f32 and casting on the device. Anything else
    /// silently moves the block-parity gates.
    #[test]
    fn streaming_f16_stage_is_bit_identical_to_decode_then_rne() {
        use crate::f16_stage::{f32_to_f16_rne, F16Stage};
        let cfg = test_cfg();
        let dir = temp_dir("stream_bfl");
        write_checkpoint(&dir, &cfg);
        let src = open_source(&dir);
        let plan = FluxPlan::detect(&*src, &cfg);
        assert_eq!(plan.layout, FluxLayout::Bfl);
        plan.validate(&cfg).unwrap();

        let mut stage = F16Stage::new();
        let mut peak_words = 0usize;
        for key in manifest::expected_flux_keys(&cfg) {
            let want: Vec<u16> = plan
                .tensor(&*src, &key)
                .unwrap()
                .data
                .iter()
                .map(|v| f32_to_f16_rne(*v))
                .collect();
            let got = plan.stage_f16(&*src, &key, &mut stage).unwrap();
            assert_eq!(got, &want[..], "staged words differ for {}", key.name);
            peak_words = peak_words.max(got.len());
        }
        // The whole point: the staging buffer never grows past one tensor.
        let model_words: usize = manifest::expected_flux_keys(&cfg)
            .iter()
            .map(|k| k.rows * k.cols)
            .sum();
        assert!(
            peak_words < model_words,
            "staging peak {peak_words} should be one tensor, not the model ({model_words})"
        );
        assert!(
            stage.capacity_bytes() / 2 >= peak_words,
            "staging buffer must hold the largest key it staged"
        );
    }

    /// Same guarantee through the diffusers key mapping, where a manifest key
    /// row-concatenates three or four checkpoint tensors — the staged words
    /// must be the concatenation in plan order.
    #[test]
    fn streaming_f16_stage_matches_the_diffusers_concatenation() {
        use crate::f16_stage::{f32_to_f16_rne, F16Stage};
        let cfg = test_cfg();
        let dir = temp_dir("stream_diffusers");
        write_diffusers_checkpoint(&dir, &cfg);
        let src = open_source(&dir);
        let plan = FluxPlan::detect(&*src, &cfg);
        assert_eq!(plan.layout, FluxLayout::Diffusers);
        plan.validate(&cfg).unwrap();

        // At least one key must actually be a multi-part concatenation, or
        // this test is not testing what it claims to.
        let fused = plan.parts("single_blocks.0.linear1.weight").unwrap();
        assert_eq!(fused.len(), 4);

        let mut stage = F16Stage::new();
        for key in manifest::expected_flux_keys(&cfg) {
            let want: Vec<u16> = plan
                .tensor(&*src, &key)
                .unwrap()
                .data
                .iter()
                .map(|v| f32_to_f16_rne(*v))
                .collect();
            let got = plan.stage_f16(&*src, &key, &mut stage).unwrap();
            assert_eq!(got, &want[..], "staged words differ for {}", key.name);
        }
    }

    #[test]
    fn stage_reports_the_offending_tensor_on_a_bad_dtype() {
        use crate::f16_stage::F16Stage;
        let cfg = test_cfg();
        let dir = temp_dir("stream_baddtype");
        std::fs::write(
            dir.join("config.json"),
            json!({ "model_type": "flux" }).to_string(),
        )
        .unwrap();
        let tensors =
            checkpoint_tensors(&cfg, Some(("img_in.bias", "F64", vec![cfg.hidden_size], 8)));
        write_safetensors(&dir.join("model.safetensors"), &tensors);
        let src = open_source(&dir);
        let plan = FluxPlan::bfl(&cfg);
        let key = manifest::expected_flux_keys(&cfg)
            .into_iter()
            .find(|k| k.name == "img_in.bias")
            .unwrap();
        let mut stage = F16Stage::new();
        let err = plan.stage_f16(&*src, &key, &mut stage).unwrap_err();
        assert!(err.contains("img_in.bias") && err.contains("F64"), "{err}");
    }

    #[test]
    fn host_load_rejects_shape_mismatch() {
        let cfg = test_cfg();
        let dir = temp_dir("badshape");
        std::fs::write(
            dir.join("config.json"),
            json!({ "model_type": "flux" }).to_string(),
        )
        .unwrap();
        let tensors = checkpoint_tensors(&cfg, Some(("img_in.weight", "BF16", vec![17, 4], 2)));
        write_safetensors(&dir.join("model.safetensors"), &tensors);
        let src = open_source(&dir);
        let err = load_weights(&*src, &cfg).unwrap_err();
        assert!(
            err.contains("img_in.weight") && err.contains("[17, 4]"),
            "{err}"
        );
    }

    #[test]
    fn host_load_rejects_unsupported_dtype() {
        let cfg = test_cfg();
        let dir = temp_dir("baddtype");
        std::fs::write(
            dir.join("config.json"),
            json!({ "model_type": "flux" }).to_string(),
        )
        .unwrap();
        let tensors =
            checkpoint_tensors(&cfg, Some(("img_in.bias", "F64", vec![cfg.hidden_size], 8)));
        write_safetensors(&dir.join("model.safetensors"), &tensors);
        let src = open_source(&dir);
        let err = load_weights(&*src, &cfg).unwrap_err();
        assert!(err.contains("img_in.bias") && err.contains("F64"), "{err}");
    }

    #[test]
    fn flux2_plan_fuses_qkv_and_keeps_single_proj_whole() {
        let cfg = test_cfg_flux2();
        let plan = FluxPlan::flux2_diffusers(&cfg);
        assert_eq!(
            plan.parts("transformer_blocks.0.attn.qkv.weight")
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            plan.parts("transformer_blocks.0.attn.add_qkv.weight")
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            plan.parts("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight")
                .unwrap()
                .len(),
            1
        );
        plan.validate(&cfg).unwrap();
    }

    #[test]
    fn f16_widening_matches_known_values() {
        for (u, want) in [
            (0x3C00u16, 1.0f32),
            (0x4000, 2.0),
            (0xC000, -2.0),
            (0x7BFF, 65504.0), // max normal half
            (0x7C00, f32::INFINITY),
            (0x8000, -0.0),
            (0x0001, 5.960_464_5e-08), // min subnormal = 2^-24
        ] {
            let got = f16_to_f32(u);
            assert_eq!(got.to_bits(), want.to_bits(), "f16 0x{u:04X}");
        }
    }
}
