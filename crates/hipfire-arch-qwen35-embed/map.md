# hipfire-arch-qwen35-embed — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 17); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

The Qwen3.5 multi-vector embedding encoder (`perplexity-ai/pplx-embed-v2-late-0.6b`): a bidirectional Qwen3.5 hybrid (GatedDeltaNet + `is_causal:false` full attention) with a SigLIP-style tower and a 128-d late-interaction head. It is **not a chat model** — it owns no LM head and is served through the non-chat `embed` verb (`hipfire embed`, `POST /v1/embeddings`), never the generation routes. `arch_id` 17 exists because its `model_type` is shared with the causal LM (5); only the `architectures` field discriminates. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- **Both stateful mixers are per sequence.** A batched encode (`encode_embeddings_batch`) concatenates row blocks and runs attention + GDN per segment; the GDN conv ring and the delta-rule S matrix are consumed in *two separate passes* (conv, then a row-independent L2 norm, then scan), so the dispatch `EncoderOp::ConvSplit` and `EncoderOp::Scan` stages each zero the state they read at every segment's start. Clearing only in the conv pass silently lets segment *k* start from segment *k-1*'s state.
- `weights::gather_hidden` **fills** the buffer it is given (it clears first). A caller assembling several sequences into one host buffer must gather per sequence into its own vector and append, or each sequence overwrites the previous one's rows.
- Padding to a multiple of 64 (`hipfire_dispatch::pipeline::encoder::padded_rows`, next to the eager predicate it serves) is what puts a 256+ token input on the eager attention path; below 256 the Q-tiled SIMT kernel runs and no padding is taken. Padded rows are dropped by the caller, and the attention's `n_valid` key bound is what keeps them out of the softmax.
- The two host-side post-processing modules (skiplist mask, L2 normalize) live in [`EmbedBundle::finalize_embeddings`](src/bundle.rs) and **must** be shared by every wire path — a mask difference is a silently different vector, not an error.
- The slot engine ([`src/slot_engine.rs`](src/slot_engine.rs)) is not the qwen3.5 generation engine: no KV, no sessions, no swap, no scheduler. Its one concurrency primitive is that several sequences share a forward, so its request-collection policy is the whole design.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
