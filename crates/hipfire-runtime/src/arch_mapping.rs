// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Single source of truth for `model_type` / `general.architecture` → `arch_id`.
//!
//! Why this table exists: three independent `model_type -> arch_id` maps
//! drifted (safetensors_source.rs, quantize/src/pipeline.rs,
//! quantize/src/pipeline_gguf.rs). One silently defaulted to llama (0) on
//! unknown input, another returned UNCLAIMED, the third lacked entries. This
//! module is the sole authority; the other sites call [`lookup_model_type`].
//!
//! The numeric ids are the HFQ `arch_id` stamped into the file header and
//! claimed by `Carrier::claims_arch_id`. Changing any assignment is a
//! wire-format / routing break, so keep them byte-identical.

/// Canonical executable `model_type` (HF) / `general.architecture` (GGUF) →
/// `arch_id`.
///
/// Reserved source-only families live in [`RESERVED_MODEL_TYPE_TO_ARCH_ID`]
/// instead of this executable registry. Keeping the tables separate lets
/// source classification retain a reservation without making reset-core and
/// carrier inventories pretend that a generation route exists.
///
/// Covers the union of every string previously recognised by executable
/// consumers. Strings absent from both tables are *unknown* and must fail
/// closed (not silently become llama 0). The qwen2 entry is intentionally
/// `7` (Qwen2Carrier, loads Q/K/V biases); earlier `hipfire-quantize` builds
/// mapped it to `1` (LLaMA) which dropped those biases — that was a bug and
/// is corrected here. See `safetensors_source.rs` commit 9002d7f8b.
///
/// Sorted by `arch_id` then alphabetically for auditability.
pub const MODEL_TYPE_TO_ARCH_ID: &[(&str, u32)] = &[
    // arch 0 — llama family
    ("llama", 0),
    ("mistral", 0),
    // arch 1 — qwen3 (llama-family loader, no bias)
    ("qwen3", 1),
    // arch 5 — qwen3.5 dense (qwen3.5/qwen3.6 share the same loader, 5 dense / 6 MoE)
    ("qwen3.5", 5),
    ("qwen3.6", 5),
    ("qwen35", 5),
    ("qwen3_5", 5),
    ("qwen3_5_text", 5),
    ("qwen3_6", 5),
    // arch 5 — ornith 1.5 dense (9B). Same loader as qwen3.5 dense (5); a3b MoE variant is 6.
    ("ornith", 5),
    ("ornith-1.5", 5),
    ("ornith1.5", 5),
    ("ornith_1.5", 5),
    // arch 6 — qwen3.5 MoE (explicit model_type strings; the safetensors path also
    // derives 6 from has_experts==true for the qwen3.5/3.6 family)
    ("qwen3_5_moe", 6),
    ("qwen3_5_moe_text", 6),
    ("qwen3moe", 6),
    // arch 6 — ornith 1.5 MoE (35B-A3B). Mirrors registry_gen arch_id_for ornith-1.5 + a3b.
    ("ornith_moe", 6),
    ("ornith-1.5_moe", 6),
    ("ornith1.5_moe", 6),
    ("ornith_1.5_moe", 6),
    ("qwen2", 7),
    // arch 8 — dots.ocr
    ("dots_ocr", 8),
    // arch 9 — deepseek_v4
    ("deepseek_v4", 9),
    // arch 10 — minimax_m2
    ("minimax_m2", 10),
    // arch 11 — lfm2 (dense) + lfm2_moe (MoE); both route to hipfire-arch-lfm2moe/11
    ("lfm2", 11),
    ("lfm2_moe", 11),
    // lfm2_vl is the vision-language variant; it reuses the arch-11 text backbone
    // (hipfire-arch-lfm2moe) plus an embedded SigLIP-2 vision tower + projector.
    ("lfm2_vl", 11),
    // arch 12 — cohere2_moe
    ("cohere2_moe", 12),
    // arch 13 — gemma4 family (dense + MoE unified; text decoder only). The
    // four strings mirror pipeline.rs; gguf's old `starts_with("gemma4")`
    // catch-all is intentionally replaced by this exact list so unknown
    // `gemma4*` variants fail closed instead of silently becoming 13.
    ("gemma4", 13),
    ("gemma4_text", 13),
    ("gemma4_unified", 13),
    ("gemma4_unified_text", 13),
    // arch 14 — muse_glimmer dense (52-layer + ViT)
    ("muse_glimmer", 14),
    ("muse_glimmer_text", 14),
    // arch 15 — maple (Maple-Preview 20B-A1B, natively-ternary 256-expert MoE)
    ("maple", 15),
    // arch 17 — qwen3.5 multi-vector embedding encoder (bidirectional
    // full-attention + causal GDN hybrid; e.g. pplx-embed-v2-late-0.6b).
    // `qwen3_5model` is the bare composite `Qwen3_5Model` architecture
    // string (no LM head) — it must beat the plain `qwen3_5` → 5 substring
    // in `derive_arch_id`'s architectures loop, so that entry checks for it
    // explicitly before the qwen3.5 dense/MoE family test. The
    // `_model_type_` is `qwen3_5` on these checkpoints, indistinguishable
    // from the causal LM — only the architectures field discriminates.
    ("qwen3_5model", 17),
    ("qwen3_5_embed", 17),
    // arch 18 — EmbeddingGemma 2 (google/embeddinggemma-2): single-vector
    // embedding encoder on the Gemma 4 text architecture. `embeddinggemma2`
    // (no underscore) catches the class name `EmbeddingGemma2Model` in
    // `derive_arch_id`'s substring loop; the underscored keys are the
    // `model_type` strings (composite `embedding_gemma2`, text sub-config
    // `embedding_gemma2_text`).
    ("embedding_gemma2", 18),
    ("embedding_gemma2_text", 18),
    ("embeddinggemma2", 18),
    // arch 22 — gemma4 EAGLE drafter (single-block spec-decode head for arch 13)
    ("gemma4_unified_assistant", 22),
    // arch 23 — muse_glimmer DFlash drafter
    ("muse_glimmer_assistant", 23),
    // arch 40 — flux MMDiT diffusion trunk (image-gen component block 40–47;
    // high by design so the sequential primary range 17–19 stays free for
    // future text arches; never a chat-serve trunk — see
    // docs/architecture-ids.md § Image-generation component ids)
    ("flux", 40),
    // arch 45 — flux2 MMDiT diffusion trunk (FLUX.2 Klein; 44 is intentionally
    // spare). The diffusers `_class_name` strings route through the same two
    // keys: `derive_arch_id` matches the table as substrings with the longest
    // key winning, so `Flux2Transformer2DModel` resolves to 45 via "flux2" and
    // `FluxTransformer2DModel` to 40 via "flux".
    ("flux2", 45),
    // arch 19 — Raon-1-mini. The class name `Raon1MiniForCausalLM` does not
    // contain `raon1_mini`, so the class alias is its own key.
    ("raon1_mini", 19),
    ("raon1miniforcausallm", 19),
];
/// Reserved source classifications with no executable carrier or reset owner.
/// These strings remain accepted by [`lookup_model_type`] so packers and
/// safetensors probing can preserve the wire id without publishing a route.
pub const RESERVED_MODEL_TYPE_TO_ARCH_ID: &[(&str, u32)] =
    &[("qwen4_exp", 16), ("qwen4_exp_text", 16)];

/// Look up an `arch_id` for a `model_type` / GGUF `general.architecture`
/// string, including reserved source-only classifications.
///
/// Returns `None` for unknown inputs — callers must fail closed (error naming
/// the unrecognised string and listing `supported_model_types()`).
pub fn lookup_model_type(model_type: &str) -> Option<u32> {
    MODEL_TYPE_TO_ARCH_ID
        .iter()
        .chain(RESERVED_MODEL_TYPE_TO_ARCH_ID)
        .find_map(|(key, id)| (*key == model_type).then_some(*id))
}

/// Resolve an `arch_id` from an HF `architectures` class name
/// (case-insensitive substring, longest key wins) — the same match
/// `derive_arch_id`'s architectures loop performs, exposed so the
/// quantizer can stamp the same id the loader will derive from the dir.
///
/// Returns `None` when no table key is a substring of `arch`; the
/// qwen3.5-family dense-vs-MoE and `qwen3_5model` embedding decisions are
/// NOT made here (they need config context) — callers with a config
/// should prefer `safetensors_source::derive_arch_id`. The quantizer reads it
/// only through [`quantize_arch_id`].
pub fn lookup_architecture(arch: &str) -> Option<u32> {
    let arch_lower = arch.to_lowercase();
    let mut best: Option<(&'static str, u32)> = None;
    for (k, v) in MODEL_TYPE_TO_ARCH_ID
        .iter()
        .chain(RESERVED_MODEL_TYPE_TO_ARCH_ID)
    {
        if arch_lower.contains(k) {
            match best {
                Some((bk, _)) if k.len() <= bk.len() => {}
                _ => best = Some((*k, *v)),
            }
        }
    }
    best.map(|(_, id)| id)
}

/// The `arch_id` the quantizer stamps for an HF config: `model_type` through
/// [`lookup_model_type`] (fail closed on an unknown type), except the bare
/// composite `Qwen3_5Model` class — arch 17, the multi-vector embedding
/// encoder — which shares `model_type: qwen3_5` with the causal LM (5) and is
/// only distinguishable by its `architectures` entry. No other class is
/// consulted: [`lookup_architecture`]'s substring match reads
/// `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`, 6) as
/// dense 5, and resolves classes whose `model_type` is unsupported (e.g.
/// `Qwen3MoeForCausalLM` → 6) instead of failing closed.
pub fn quantize_arch_id<'a>(
    model_type: &str,
    architectures: impl IntoIterator<Item = &'a str>,
) -> Option<u32> {
    if architectures
        .into_iter()
        .any(|class| lookup_architecture(class) == Some(17))
    {
        return Some(17);
    }
    lookup_model_type(model_type)
}

/// Sorted list of every recognised executable or reserved source string, for
/// error messages. Computed from both tables so they cannot drift.
pub fn supported_model_types() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = MODEL_TYPE_TO_ARCH_ID
        .iter()
        .chain(RESERVED_MODEL_TYPE_TO_ARCH_ID)
        .map(|(key, _)| *key)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Human-readable, comma-joined list for `eprintln!` diagnostics.
pub fn supported_model_types_display() -> String {
    supported_model_types().join(", ")
}

#[cfg(test)]
mod tests {
    use super::{lookup_model_type, MODEL_TYPE_TO_ARCH_ID, RESERVED_MODEL_TYPE_TO_ARCH_ID};

    #[test]
    fn qwen4_outer_and_text_model_types_share_reserved_architecture_id() {
        assert_eq!(lookup_model_type("qwen4_exp"), Some(16));
        assert_eq!(lookup_model_type("qwen4_exp_text"), Some(16));
    }

    #[test]
    fn qwen4_is_reserved_and_not_in_executable_table() {
        assert!(MODEL_TYPE_TO_ARCH_ID
            .iter()
            .all(|(model_type, _)| !model_type.starts_with("qwen4")));
        assert_eq!(
            RESERVED_MODEL_TYPE_TO_ARCH_ID,
            &[("qwen4_exp", 16), ("qwen4_exp_text", 16)]
        );
    }

    #[test]
    fn raon1_mini_is_executable_arch_19() {
        use super::lookup_architecture;
        assert_eq!(lookup_model_type("raon1_mini"), Some(19));
        assert_eq!(lookup_architecture("Raon1MiniForCausalLM"), Some(19));
        assert!(MODEL_TYPE_TO_ARCH_ID.contains(&("raon1_mini", 19)));
        assert!(RESERVED_MODEL_TYPE_TO_ARCH_ID
            .iter()
            .all(|(_, id)| *id != 19));
    }

    #[test]
    fn qwen4_near_misses_fail_closed() {
        assert_eq!(lookup_model_type("qwen4"), None);
        assert_eq!(lookup_model_type("qwen4_exp_moe"), None);
    }

    #[test]
    fn qwen3_5_embed_architecture_resolves_to_17() {
        use super::lookup_architecture;
        // The bare composite class is the embedding encoder (17); the causal
        // LM class keeps resolving through the qwen3.5 family logic (5).
        assert_eq!(lookup_architecture("Qwen3_5Model"), Some(17));
        assert_eq!(lookup_architecture("Qwen3_5ForCausalLM"), Some(5));
        assert_eq!(lookup_model_type("qwen3_5model"), Some(17));
        assert_eq!(lookup_model_type("qwen3_5_embed"), Some(17));
        // model_type stays the causal LM's — only architectures discriminates.
        assert_eq!(lookup_model_type("qwen3_5"), Some(5));
    }

    #[test]
    fn embedding_gemma2_resolves_to_18_by_every_route() {
        use super::lookup_architecture;
        assert_eq!(lookup_architecture("EmbeddingGemma2Model"), Some(18));
        assert_eq!(lookup_model_type("embedding_gemma2"), Some(18));
        assert_eq!(lookup_model_type("embedding_gemma2_text"), Some(18));
        // Longest-key discipline: nothing else in the table is a substring of
        // the class name (the gemma4 keys are not — "gemma4" != "gemma2").
        assert_ne!(lookup_architecture("EmbeddingGemma2Model"), Some(13));
    }

    #[test]
    fn quantize_arch_id_reads_architectures_only_for_the_embed_encoder() {
        use super::quantize_arch_id;
        assert_eq!(quantize_arch_id("qwen3_5", ["Qwen3_5Model"]), Some(17));
        assert_eq!(
            quantize_arch_id("qwen3_5", ["Qwen3_5ForConditionalGeneration"]),
            Some(5)
        );
        // model_type stays authoritative everywhere else: the MoE class name
        // substring-matches the dense `qwen3_5` key, and an unsupported
        // model_type must not resolve through its class name.
        assert_eq!(
            quantize_arch_id("qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]),
            Some(6)
        );
        assert_eq!(
            quantize_arch_id("qwen3_5_moe", ["Qwen3_5MoeForCausalLM"]),
            Some(6)
        );
        assert_eq!(quantize_arch_id("qwen3_moe", ["Qwen3MoeForCausalLM"]), None);
        assert_eq!(
            quantize_arch_id("qwen3_next", ["Qwen3NextForCausalLM"]),
            None
        );
        assert_eq!(quantize_arch_id("qwen2", ["Qwen2ForCausalLM"]), Some(7));
        assert_eq!(quantize_arch_id("llama", []), Some(0));
    }
}
