// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-qwen35-embed: the Qwen3.5 **multi-vector embedding**
//! architecture (`arch_id` 17), first shipped by
//! `perplexity-ai/pplx-embed-v2-late-0.6b`.
//!
//! What the model is: a bidirectional Qwen3.5 hybrid encoder. The text
//! backbone is exactly the Qwen3.5 LM family (GatedDeltaNet linear-attention
//! layers alternating with gated full-attention layers, partial 0.25
//! interleaved mRoPE, zero-centered RMSNorm) with two changes — the
//! full-attention layers run **non-causal** (`is_causal: false` in the HF
//! config) and there is no LM head. A SigLIP-2-style tower (the same
//! [`hipfire_arch_qwen35_vl`] tower code, hidden 768) merges each 2×2 patch
//! group into one 1024-d token that replaces an `<|image_pad|>` run in the
//! token stream. A sentence-transformers `Dense` head projects every token to
//! 128 dims, a ColBERT skiplist mask drops punctuation token vectors on the
//! document side, each kept vector is L2-normalized, and queries score
//! against documents by late interaction (MaxSim).
//!
//! What this crate owns:
//! - [`config`]: parse the composite HF `config.json` (text + vision) and
//!   the sentence-transformers sidecar files (`modules.json`,
//!   `config_sentence_transformers.json`, `2_MultiVectorMask/config.json`).
//! - [`weights`]: stream `language_model.*` / `visual.*` tensors (F32/BF16/F16
//!   safetensors dir or a raw-dtype `.hfq`) to the GPU — f16 GEMM weights,
//!   f32 norm vectors with the HF zero-centered `+1.0` bias applied at load,
//!   and the 128-d Dense head out of `1_Dense/model.safetensors`.
//! - [`state`]: per-request GDN scratch (FP32 S matrices + causal-conv ring),
//!   reset between encodes.
//! - [`forward`]: the GPU encoder — embedding gather/host splice, per-layer
//!   bidirectional full attention ([`Gpu::attention_text_f32`] with
//!   `causal = false`) or causal-recurrent GDN (the production qwen35
//!   prefill kernels), final RMSNorm, Dense projection.
//! - [`post`]: skiplist mask, L2 normalize, MaxSim — pure host math, unit
//!   tested without a GPU.
//! - [`vision`]: image → merged 1024-d tokens via the qwen35-vl tower.
//! - [`bundle`]/[`carrier`]: the `ArchModel` bundle + the
//!   `load_qwen35_embed_bundle` entry point the loader's carrier calls.
//!
//! Design notes:
//! - **GDN layers stay causal.** Gated DeltaNet is a left-to-right recurrence;
//!   the HF implementation ignores `is_causal` for linear-attention layers
//!   (the flag only feeds the full-attention mask). Bidirectionality here
//!   matches that: full-attention layers attend over the whole sequence,
//!   GDN layers keep their causal scan.
//! - **Numerics**: activations f32, GEMM weights f16 (WMMA f16×f16 with f32
//!   accumulate), GDN state FP32. Same tradeoff the FLUX T5/Qwen3 GPU
//!   encoders in `hipfire-arch-diffusion` make (~1e-3 relative vs the f32
//!   reference, not bit-exact). The checkpoint ships f32; an f16-quantized
//!   `.hfq` of it loads the same way.
//! - **One sequence per encode; concurrency batches at the engine.** The
//!   forward itself takes a single input — no padding masks: `attention_text_f32`
//!   needs no `key_mask` and GDN needs no cumulative padding mask. Concurrent
//!   `embed` requests are batched one level up by the multi-slot engine
//!   (`serve.multi_slot`, `slot_engine.rs`): the sequences are row-concatenated
//!   into one forward (per-segment GDN state resets, bidirectional attention
//!   bounded by each segment's `n_valid`), bit-exact against running them
//!   serially. The forward API stayed single-sequence; the engine owns the
//!   batch dimension.
//! - **Two source kinds, both supported.** The HF safetensors directory is
//!   the native form; a `.hfq` built from it by `hipfire-quantize` loads too.
//!   The two need different access (`HfqFile`'s native methods, with an owned
//!   fallback once the mmap is gone) and the `.hfq` may carry a compressed
//!   trunk (mq4 -> qt 44, q8 -> qt 3), which is decoded through the runtime's
//!   registry-wide `dequantize_weight_to_f32`. A compressed `.hfq` therefore
//!   saves disk, not VRAM — it serves the same f16-GEMM path as the dir. Use
//!   `examples/compare_tensor --features lab` to check a build's per-tensor
//!   error sits at its format's bit-width level.
//! - **The vision tower can live in host RAM.** It is ~250 MB of the model's
//!   ~740 MB and only an image request touches it, so when the card cannot
//!   hold everything (`HIPFIRE_EMBED_VISION_OFFLOAD`, unset = auto) the
//!   tower's weights are host-mapped and read over PCIe, leaving the trunk
//!   resident. Text-only embedding then runs on a card whose desktop is
//!   holding most of it, instead of failing to load; image requests pay PCIe
//!   reads. `examples/vision_smoke --features lab` exercises the tower alone.

pub mod bundle;
pub mod carrier;
pub mod config;
pub mod forward;
pub mod names;
pub mod post;
pub mod slot_engine;
pub mod state;
pub mod vision;
pub mod weights;

pub use bundle::{EmbedBundle, MarkerIds};
pub use carrier::{
    assemble_input, build_positions, load_qwen35_embed_bundle, load_qwen35_embed_from_source,
    preflight_slot, AssembledInput,
};
pub use config::{EmbedConfig, InputKind};
pub use forward::{encode_embeddings, encode_embeddings_batch, encode_hidden, EncodeSegment};
pub use post::{l2_normalize_rows, maxsim, skiplist_mask, DEFAULT_SKIPLIST_WORDS};

/// `arch_id` this crate claims — see `docs/architecture-ids.md`.
pub const ARCH_ID: u32 = 17;
