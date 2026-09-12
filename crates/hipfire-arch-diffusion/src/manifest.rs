// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLUX.1 tensor manifest: the canonical safetensors key list with shapes,
//! generated from [`FluxDiffusionConfig`] so it cannot drift from the parse
//! path that produces it.
//!
//! Key names and shapes are pinned to the ORIGINAL BFL checkpoint format —
//! `black-forest-labs/flux` at commit `87f6fff` (Sept 2024, the code that
//! produced the shipped FLUX.1-schnell / FLUX.1-dev weights): dotted module
//! paths (`double_blocks.N.img_attn.qkv.weight`), `img_attn.proj` bias-free,
//! `txt_in` bias-free, weightless LayerNorms (`img_norm1`, `img_norm2`,
//! `pre_norm`, `norm_final`) carrying **no key**, per-head learned QK-norm
//! scales (`img_attn.norm.query_norm.scale`), SiLU-inside-Sequential modules
//! indexed at `.1` (`final_layer.adaLN_modulation.1.weight`), and sinusoidal
//! timestep/guidance embeddings feeding `MLPEmbedder`s (`time_in.in_layer`,
//! `vector_in.in_layer`, each with `out_layer`).
//!
//! This is the single place to correct if the golden-trace pass over the real
//! checkpoint disagrees: the loader
//! validates the safetensors directory against [`expected_flux_keys`] before
//! any weight upload, and the CPU reference consumes shapes from here too.
//!
//! Validation today checks for MISSING keys only; unlisted tensors in the
//! directory are tolerated (checkpoints can carry auxiliary buffers).

use crate::config::FluxDiffusionConfig;

/// T5-XXL encoder hidden width (the `txt_in` projection input).
pub const T5_XXL_HIDDEN: usize = 4096;
/// Sinusoidal embedding width for timestep / guidance (FLUX uses 256).
pub const TS_EMBED_DIM: usize = 256;

/// Width of the fused conditioning vector: timestep 256 + optional guidance +
/// pooled 768. FLUX.1-schnell (guidance dim 0) → 1024; FLUX.1-dev → 1280.
pub fn vector_dim(cfg: &FluxDiffusionConfig) -> usize {
    TS_EMBED_DIM + cfg.guidance_embed_dim + cfg.pooled_projection_dim
}

/// A name + row-major shape. Vectors carry `(len, 1)`.
pub struct FluxKey {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
}

/// Full expected key list for a FLUX.1 transformer, given `cfg`.
///
/// Deterministic ordering: fixed top-level keys, then double blocks
/// (index-sorted), then single blocks, then final. A caller comparing against
/// a real safetensors listing should sort both sides by name.
pub fn expected_flux_keys(cfg: &FluxDiffusionConfig) -> Vec<FluxKey> {
    if cfg.is_flux2() {
        return expected_flux2_keys(cfg);
    }
    let d = cfg.hidden_size;
    let f = 4 * d; // MLP intermediate width is 4×hidden in FLUX blocks
    let hd = cfg.head_dim;
    let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
    let mut keys: Vec<FluxKey> = Vec::new();

    let vec = |name: String, rows: usize| FluxKey {
        name,
        rows,
        cols: 1,
    };
    let mat = |name: String, rows: usize, cols: usize| FluxKey { name, rows, cols };

    // Stream projections. Both carry biases (nn.Linear default) — including
    // txt_in and the attention proj layers (BFL `Flux` / `SelfAttention`).
    keys.push(mat("img_in.weight".into(), d, patch_in));
    keys.push(vec("img_in.bias".into(), d));
    keys.push(mat("txt_in.weight".into(), d, cfg.txt_hidden_dim));
    keys.push(vec("txt_in.bias".into(), d));
    if cfg.guidance_embed_dim > 0 {
        push_embedder(&mut keys, "guidance_in", TS_EMBED_DIM, d);
    }
    // Sinusoidally-embedded timestep and pooled text conditioning.
    push_embedder(&mut keys, "time_in", TS_EMBED_DIM, d);
    push_embedder(&mut keys, "vector_in", cfg.pooled_projection_dim, d);

    // Double blocks: per-stream modulation + qkv + joint attention + MLP.
    // Block norms (img_norm1/2, txt_norm1/2) are affine=False LayerNorms → no
    // keys; attention proj layers carry biases (nn.Linear default).
    for b in 0..cfg.num_layers {
        for stem in ["img", "txt"] {
            let s = |k: &str| format!("double_blocks.{b}.{stem}_{k}");
            keys.push(mat(s("mod.lin.weight"), 6 * d, d));
            keys.push(vec(s("mod.lin.bias"), 6 * d));
            keys.push(mat(s("attn.qkv.weight"), 3 * d, d));
            keys.push(vec(s("attn.qkv.bias"), 3 * d));
            keys.push(mat(s("attn.proj.weight"), d, d));
            keys.push(vec(s("attn.proj.bias"), d));
            keys.push(vec(s("attn.norm.query_norm.scale"), hd));
            keys.push(vec(s("attn.norm.key_norm.scale"), hd));
            keys.push(mat(s("mlp.0.weight"), f, d));
            keys.push(vec(s("mlp.0.bias"), f));
            keys.push(mat(s("mlp.2.weight"), d, f));
            keys.push(vec(s("mlp.2.bias"), d));
        }
    }

    // Single blocks: fused qkv+mlp_in (`linear1`), fused attn-proj+mlp_out
    // (`linear2`), 3-chunk modulation, per-head QK-norm scales. `pre_norm` is
    // an affine=False LayerNorm → no key.
    for b in 0..cfg.num_single_layers {
        let s = |k: &str| format!("single_blocks.{b}.{k}");
        keys.push(mat(s("modulation.lin.weight"), 3 * d, d));
        keys.push(vec(s("modulation.lin.bias"), 3 * d));
        keys.push(mat(s("linear1.weight"), 3 * d + f, d));
        keys.push(vec(s("linear1.bias"), 3 * d + f));
        keys.push(mat(s("linear2.weight"), d, d + f));
        keys.push(vec(s("linear2.bias"), d));
        keys.push(vec(s("norm.query_norm.scale"), hd));
        keys.push(vec(s("norm.key_norm.scale"), hd));
    }

    // Final head: SiLU + Linear adaLN modulation (Sequential → index 1),
    // weightless norm_final, bias-ful patch projection.
    keys.push(mat(
        "final_layer.adaLN_modulation.1.weight".into(),
        2 * d,
        d,
    ));
    keys.push(vec("final_layer.adaLN_modulation.1.bias".into(), 2 * d));
    keys.push(mat("final_layer.linear.weight".into(), patch_in, d));
    keys.push(vec("final_layer.linear.bias".into(), patch_in));

    keys
}

/// `MLPEmbedder(in, hidden)` key block: in_layer (in→hidden) + out_layer
/// (hidden→hidden), both bias-ful.
fn push_embedder(keys: &mut Vec<FluxKey>, name: &str, in_dim: usize, d: usize) {
    keys.push(FluxKey {
        name: format!("{name}.in_layer.weight"),
        rows: d,
        cols: in_dim,
    });
    keys.push(FluxKey {
        name: format!("{name}.in_layer.bias"),
        rows: d,
        cols: 1,
    });
    keys.push(FluxKey {
        name: format!("{name}.out_layer.weight"),
        rows: d,
        cols: d,
    });
    keys.push(FluxKey {
        name: format!("{name}.out_layer.bias"),
        rows: d,
        cols: 1,
    });
}

/// Full expected key list for a FLUX.2 (Klein) transformer, given `cfg`.
///
/// Klein's diffusers checkpoint keys ARE the canonical manifest names here
/// (see [`crate::flux::FluxPlan::flux2_diffusers`]) except for the two fused
/// attention projections. Every key ends in `.weight` — the Klein
/// architecture is bias-free (`cfg.bias == false`) — and the three
/// modulation linears are shared ACROSS every double/single block
/// (`cfg.shared_modulation`), so each appears exactly once at the top level
/// rather than per block.
///
/// Deterministic ordering: fixed top-level keys, then double blocks
/// (index-sorted, img parts then txt parts), then single blocks, then final.
fn expected_flux2_keys(cfg: &FluxDiffusionConfig) -> Vec<FluxKey> {
    let d = cfg.hidden_size;
    let f = cfg.mlp_width();
    let hd = cfg.head_dim;
    let patch_in = cfg.patch_in();
    let mut keys: Vec<FluxKey> = Vec::new();

    let vec = |name: String, rows: usize| FluxKey {
        name,
        rows,
        cols: 1,
    };
    let mat = |name: String, rows: usize, cols: usize| FluxKey { name, rows, cols };

    // Top-level: patch/text embedders, shared timestep embedder, and the
    // three modulation linears (shared across blocks — see doc comment).
    keys.push(mat("x_embedder.weight".into(), d, patch_in));
    keys.push(mat("context_embedder.weight".into(), d, cfg.txt_hidden_dim));
    keys.push(mat(
        "time_guidance_embed.timestep_embedder.linear_1.weight".into(),
        d,
        TS_EMBED_DIM,
    ));
    keys.push(mat(
        "time_guidance_embed.timestep_embedder.linear_2.weight".into(),
        d,
        d,
    ));
    keys.push(mat(
        "double_stream_modulation_img.linear.weight".into(),
        6 * d,
        d,
    ));
    keys.push(mat(
        "double_stream_modulation_txt.linear.weight".into(),
        6 * d,
        d,
    ));
    keys.push(mat(
        "single_stream_modulation.linear.weight".into(),
        3 * d,
        d,
    ));

    // Double blocks: fused joint-attention qkv (img then txt/"add" parts),
    // per-head QK-norm scales, and separate img/txt-context feed-forwards.
    for b in 0..cfg.num_layers {
        let s = |k: &str| format!("transformer_blocks.{b}.{k}");
        keys.push(mat(s("attn.qkv.weight"), 3 * d, d));
        keys.push(mat(s("attn.to_out.0.weight"), d, d));
        keys.push(mat(s("attn.add_qkv.weight"), 3 * d, d));
        keys.push(mat(s("attn.to_add_out.weight"), d, d));
        keys.push(vec(s("attn.norm_q.weight"), hd));
        keys.push(vec(s("attn.norm_k.weight"), hd));
        keys.push(vec(s("attn.norm_added_q.weight"), hd));
        keys.push(vec(s("attn.norm_added_k.weight"), hd));
        keys.push(mat(s("ff.linear_in.weight"), 2 * f, d));
        keys.push(mat(s("ff.linear_out.weight"), d, f));
        keys.push(mat(s("ff_context.linear_in.weight"), 2 * f, d));
        keys.push(mat(s("ff_context.linear_out.weight"), d, f));
    }

    // Single blocks: fused qkv+mlp-in projection, whole attn+mlp-out
    // projection, per-head QK-norm scales.
    for b in 0..cfg.num_single_layers {
        let s = |k: &str| format!("single_transformer_blocks.{b}.{k}");
        keys.push(mat(s("attn.to_qkv_mlp_proj.weight"), 3 * d + 2 * f, d));
        keys.push(mat(s("attn.to_out.weight"), d, d + f));
        keys.push(vec(s("attn.norm_q.weight"), hd));
        keys.push(vec(s("attn.norm_k.weight"), hd));
    }

    // Final head.
    keys.push(mat("norm_out.linear.weight".into(), 2 * d, d));
    keys.push(mat("proj_out.weight".into(), patch_in, d));

    keys
}

/// Check a real directory listing (`(name, element_count)` pairs) against the
/// manifest. Returns the sorted list of missing keys (empty = complete).
pub fn missing_keys(cfg: &FluxDiffusionConfig, listing: &[(&str, usize)]) -> Vec<String> {
    let present: std::collections::HashSet<&str> = listing.iter().map(|(n, _)| *n).collect();
    let mut missing: Vec<String> = expected_flux_keys(cfg)
        .iter()
        .filter(|k| !present.contains(k.name.as_str()))
        .map(|k| k.name.clone())
        .collect();
    missing.sort();
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny_cfg() -> FluxDiffusionConfig {
        FluxDiffusionConfig::from_json(&json!({
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

    #[test]
    fn manifest_complete_for_tiny_cfg() {
        let keys = expected_flux_keys(&tiny_cfg());
        // top-level: img_in w+b (2), txt_in w+b (2), time_in embedder (4),
        // vector_in embedder (4) = 12; 2 double blocks × 2 streams × 12
        // (=2 mod.lin w+b + 2 qkv w+b + 2 proj w+b + 2 norm scales + 2 mlp.0
        // w+b + 2 mlp.2 w+b); 1 single block × 8; final adaLN w+b (2) +
        // linear w+b (2).
        let expected_count = 12 + 2 * 2 * 12 + 8 + 4;
        assert_eq!(keys.len(), expected_count, "key count drifted");
        for k in &keys {
            assert!(!k.name.is_empty());
            assert!(k.rows > 0 && k.cols > 0, "empty shape for {}", k.name);
        }
    }

    #[test]
    fn manifest_detects_missing_keys() {
        let cfg = tiny_cfg();
        let empty: Vec<(&str, usize)> = vec![];
        let missing = missing_keys(&cfg, &empty);
        assert!(!missing.is_empty());
        assert!(missing.contains(&"img_in.weight".to_string()));
        assert!(missing.contains(&"final_layer.linear.weight".to_string()));
    }

    #[test]
    fn manifest_key_count_scales_with_layers() {
        let cfg = tiny_cfg();
        let a = expected_flux_keys(&cfg).len();
        let mut cfg2 = cfg.clone();
        cfg2.num_layers += 1;
        cfg2.num_single_layers += 1;
        let b = expected_flux_keys(&cfg2).len();
        assert_eq!(b - a, 2 * 12 + 8);
    }

    /// Real FLUX.1-dev transformer geometry (BFL `double_blocks.*` /
    /// `single_blocks.*` / `final_layer.*` safetensors layout — the exact
    /// `/home/user/comfy-models/diffusion_models/flux1-dev.safetensors`
    /// header). Guards against the manifest drifting off the shipped
    /// checkpoint: 780 keys, none missing against a complete listing, and a
    /// sliced listing is still caught as incomplete.
    fn real_dev_cfg() -> FluxDiffusionConfig {
        FluxDiffusionConfig {
            family: crate::config::FluxFamily::Flux1,
            hidden_size: 3072,
            num_layers: 19,
            num_single_layers: 38,
            num_attention_heads: 24,
            head_dim: 128,
            patch_size: 2,
            max_sequence_length: 4096,
            guidance_embed_dim: 256,
            pooled_projection_dim: 768,
            axes_dim: [16, 56, 56, 0],
            theta: 10000.0,
            qk_norm: true,
            norm_type: "rms_norm".into(),
            latent_channels: 16,
            txt_hidden_dim: 4096,
            mlp_ratio: 4.0,
            bias: true,
            shared_modulation: false,
        }
    }

    #[test]
    fn manifest_covers_real_dev_checkpoint() {
        let cfg = real_dev_cfg();
        let keys = expected_flux_keys(&cfg);
        // Verified against the real FLUX.1-dev safetensors header: 780 tensors,
        // exact name set (no missing, no extra). Counting here must equal that.
        assert_eq!(
            keys.len(),
            780,
            "real-dev key count drifted from checkpoint"
        );

        // A complete listing (every manifest key present) → nothing missing.
        let full: Vec<(&str, usize)> = keys
            .iter()
            .map(|k| (k.name.as_str(), k.rows * k.cols))
            .collect();
        assert!(
            missing_keys(&cfg, &full).is_empty(),
            "expected complete listing to be complete"
        );

        // Drop one tensor → the manifest flags exactly it.
        let mut sliced = full.clone();
        sliced.retain(|(n, _)| *n != "guidance_in.in_layer.weight");
        let miss = missing_keys(&cfg, &sliced);
        assert_eq!(miss, vec!["guidance_in.in_layer.weight".to_string()]);

        // Guidance geometry: 256-wide sinusoidal embedder in, 3072 out.
        let g = keys
            .iter()
            .find(|k| k.name == "guidance_in.in_layer.weight")
            .expect("guidance_in present");
        assert_eq!((g.rows, g.cols), (3072, 256));
    }

    #[test]
    fn proj_and_txt_in_carry_bias_keys() {
        // BFL `nn.Linear` defaults bias=True everywhere (txt_in, proj layers).
        let keys = expected_flux_keys(&tiny_cfg());
        assert!(keys
            .iter()
            .any(|k| k.name == "double_blocks.0.img_attn.proj.bias"));
        assert!(keys.iter().any(|k| k.name == "txt_in.bias"));
        assert!(keys
            .iter()
            .any(|k| k.name == "double_blocks.0.img_attn.norm.query_norm.scale"));
    }

    #[test]
    fn flux2_manifest_key_count_for_klein_4b() {
        let raw = include_str!("../tests/fixtures/klein/klein-4b-transformer.json");
        let cfg = FluxDiffusionConfig::from_json(&serde_json::from_str(raw).unwrap()).unwrap();
        let keys = expected_flux_keys(&cfg);
        // top 7 + double 5*(4 fused/proj + 4 norms + 4 ff) + single 20*(2 + 2) + final 2
        assert_eq!(keys.len(), 7 + 5 * 12 + 20 * 4 + 2);
        assert!(
            keys.iter().all(|k| !k.name.ends_with(".bias")),
            "flux2 has no biases"
        );
        let qkv = keys
            .iter()
            .find(|k| k.name == "transformer_blocks.0.attn.qkv.weight")
            .unwrap();
        assert_eq!((qkv.rows, qkv.cols), (3 * 3072, 3072));
        let l1 = keys
            .iter()
            .find(|k| k.name == "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight")
            .unwrap();
        assert_eq!((l1.rows, l1.cols), (3 * 3072 + 2 * 9216, 3072));
        let l2 = keys
            .iter()
            .find(|k| k.name == "single_transformer_blocks.0.attn.to_out.weight")
            .unwrap();
        assert_eq!((l2.rows, l2.cols), (3072, 3072 + 9216));
    }

    #[test]
    fn schnell_is_guidance_free_dev_adds_guidance_keys() {
        let schnell = tiny_cfg();
        let mut dev = tiny_cfg();
        dev.guidance_embed_dim = 256;
        assert!(expected_flux_keys(&schnell)
            .iter()
            .all(|k| !k.name.starts_with("guidance_in")));
        assert!(expected_flux_keys(&dev)
            .iter()
            .any(|k| k.name == "guidance_in.in_layer.weight"));
    }

    /// Guards the FLUX.2 Klein 4B manifest against the REAL checkpoint
    /// header on inferno02 (`klein_manifest_check` against
    /// `/home/user/comfy-models/klein/FLUX.2-klein-4B/transformer`, 2026-09-
    /// 04: 149 manifest keys, 169 source parts, 0 missing). The fixture key
    /// list is the sorted `tensor_names()` dump from that real header; every
    /// source part the plan builds must appear in it verbatim.
    #[test]
    fn flux2_manifest_matches_the_real_4b_header() {
        let names: Vec<&str> =
            include_str!("../tests/fixtures/klein/klein-4b-transformer-keys.txt")
                .lines()
                .collect();
        let raw = include_str!("../tests/fixtures/klein/klein-4b-transformer.json");
        let cfg = FluxDiffusionConfig::from_json(&serde_json::from_str(raw).unwrap()).unwrap();
        let plan = crate::flux::FluxPlan::flux2_diffusers(&cfg);
        for key in expected_flux_keys(&cfg) {
            for part in plan.parts(&key.name).unwrap() {
                assert!(
                    names.contains(&part.name.as_str()),
                    "missing in real header: {}",
                    part.name
                );
            }
        }
    }
}
