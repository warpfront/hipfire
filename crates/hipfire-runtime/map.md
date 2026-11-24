# hipfire-runtime — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Shared infra — between `saddle-core` and `hipfire-loader` (arch-agnostic infra + `Architecture` trait); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Arch-agnostic orchestration for RDNA inference: HFQ/safetensors readers, tokenizer, LLaMA-shaped shared transformer primitives (`src/llama.rs` — `KvCache`, `WeightTensor`, `EmbeddingFormat`, GEMV dispatch, `llama::` sampler host for `hipfire-arch-llama`), sampling/grammar/frame/EOS/loop-guard, spec-decode primitives (DFlash/DSpark/NGram), `ArchModel`/`Architecture` traits, `ModelState::as_arch_model{,_mut}` sanctioned destructure, and `loader_api`/`safetensors_source` model-source types. The `//!` docs in [`src/lib.rs`](src/lib.rs) define what stays here vs per-arch crates.

## Gotchas

- `hipfire_runtime::llama` is canonical dense LLaMA/Mistral forward + shared transformer types — `hipfire-arch-llama` is only a re-export facade; do not duplicate them.
- `Architecture` is bring-up scaffolding (config/weights/state), not a hot-path `dyn` forward — forward stays statically dispatched on concrete arch types after `load_model`.
- `serve::EmbedRequest.images` / `.audios` are the decoded media lists (container bytes, request order) a slot engine's GPU thread feeds to the arch's own towers; empty means text-only, and an arch without a tower refuses a non-empty list per request rather than ignoring it. The lists are `Vec`, not `Option`, because arch 18 takes several images in one input (arch 17 takes at most one and refuses a longer list).
- LLaMA `KvCache` and GEMV helpers are used by `hipfire-arch-qwen35`'s plain-LLaMA drafter branch — moving them forces an arch-crate dep on this crate as carrier cycle breaker.
- `ep::RootRoutedEpSchedule` derives the named layer collective from the sealed expert contract. Its executor owns root-route transfer, rank traversal, reduction, and residual completion for decode and prefill; family adapters supply operands and arithmetic.
- `session_cache` is the engine-owned Qwen4 prefill-snapshot cache (`memory.session_cache_bytes` / `HIPFIRE_SESSION_CACHE_BYTES`): the engine owns keying, planning, LRU-over-leaves eviction, the memory guard, placement and every copy, while an arch only reports state parts and boundaries through `SessionState`. Pool snapshots are cold-exact, delta row streams, shared across sessions and never depend on live state. The one exception is the *resident* snapshot: the fixed parts at the deepest boundary of the last committed prefill, restored over the rows the live state still holds (no row copy, not limited by the budget, so a 192K+ prompt stays reusable past an 8 GiB pool). Every `begin` that does not restore it drops it; an owner that resets its live state outside the cache must call `SessionCache::live_state_reset` (Qwen4: `Qwen4Bundle::reset`). The Qwen4 bundle's live continuation sits outside this crate (a live turn captures no snapshot). The RC3 Qwen4 prefix/radix cache is gone; `CheckpointPool` stays as the LRU/pin blob storage (used by `SessionCache` and `hipfire-arch-qwen35`) and its dead exact-entry API was removed.
- `examples/eval_hipfire.rs` qwen35 native branches admit via the shared mode-specific `kv_mode::qwen35_native_refusal` guard before cache allocation (neutral `--kv-v q8` spelling, exact gfx1201, single GPU, no adaptive): FP8 admits H24/KV4/D256 dense and H16/KV2/D256 MoE, and both exact shapes are qualified for native FP8 on unset/`auto`; BF16 admits dense H24/KV4/D256 only. Explicit `q8`, unsupported-route fallback, and PP/TP/slots behavior remain unchanged.
- `examples/eval_hipfire.rs` `--scoring-mode` selects kernel admission, not only windowing (qwen35): `prefill-whole` and the scored-region call of split `prefill` capture hidden rows through `forward_prefill_batch`'s verify capture (`DispatchWorkload::SpeculativeVerify`, sequential GDN recurrence, no chunk scan); `prefill-prompt` runs the whole window through `forward_prefill_batch_capture_hidden` (plain prompt-prefill admission, `DispatchWorkload::Standard`, GDN chunk scan where admitted), the mode that measures the production prompt route; `per-token` is single-token decode kernels. Canonical example: a production-route KLD is `eval_hipfire --model <m.hfq> --ref <ref> --output <o.kldseq> --scoring-mode prefill-prompt`; never compare its kldseq with a `prefill-whole` or split-`prefill` one. The existing modes' forward calls are pinned by `eval_common::existing_scoring_modes_source_is_unchanged`.


## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
