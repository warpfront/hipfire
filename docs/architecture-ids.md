# Architecture ID registry

Canonical `arch_id` values used by HFQ headers (`HfqFile::arch_id`), safetensors
directory derivation (`derive_arch_id` in
`crates/hipfire-runtime/src/safetensors_source.rs`, driven by the
`model_type` tables in `crates/hipfire-runtime/src/arch_mapping.rs`), and the
loader carrier registry (`crates/hipfire-loader`). Trait markers come from
`Architecture::arch_id()` in each arch crate.

One crate may cover multiple runtime ids. The trait’s canonical marker is the
family default; the file/dir id on the loaded source is authoritative for
dispatch.

**Routing availability ≠ product admission.** A primary id, source derivation, or
carrier match means the loader can route that artifact — not that any model is
product-admitted. The sole admission owner is [`admissions.yml`](admissions.yml)
(schema v2; exactly one evidence-bound record). Product-use and admission claims
remain fail-closed for every route outside that exact row; source-derived routing capability is independent.

## Primary model ids

| arch_id | Family | Crate | Carrier | Notes |
|---:|---|---|---|---|
| 0 | LLaMA / Mistral | `hipfire-arch-llama` | `LlamaCarrier` | Dense FA. Trait marker `Llama::arch_id() == 0`. |
| 1 | plain Qwen3 | `hipfire-arch-llama` | `LlamaCarrier` | Same crate; `config_from_hfq` branches on metadata. Dir `model_type`/`architectures` qwen3 → 1. |
| 5 | Qwen3.5 / 3.6 / 3.8 dense | `hipfire-arch-qwen35` | `Qwen35Carrier` | Hybrid DeltaNet + dense FFN. Trait marker returns 5. Optional VL via `hipfire-arch-qwen35-vl` under the same ids. |
| 6 | Qwen3.5 / 3.6 / 3.8 MoE (A3B) | `hipfire-arch-qwen35` | `Qwen35Carrier` | MoE expert routing; same crate as 5. |
| 7 | Qwen2 dense | `hipfire-arch-qwen2` | `Qwen2Carrier` | Standalone path so Q/K/V `attention_bias` loads (llama dir path would drop them). Dir `qwen2` → 7. |
| 8 | dots.ocr (Qwen2-VL family) | `hipfire-arch-dots-ocr` | `DotsOcrCarrier` | Vision tower + Qwen2 text decoder fields on `LoadedModel`. |
| 9 | DeepSeek V4 Flash | `hipfire-arch-deepseek4` | `Deepseek4Carrier` | Hyper-Connections, compressed-KV indexer, tail-only RoPE, raw SWA; optional in-trunk / sidecar MTP; EP via `load_model_ep`. |
| 10 | MiniMax-M2 | `hipfire-arch-minimax` | `MinimaxCarrier` | Mixtral-style MoE (GQA, per-layer QK-norm, partial rotate_half RoPE, sigmoid+bias expert route). EP via `load_model_ep`. |
| 11 | LFM2.5 / LFM2.5-MoE | `hipfire-arch-lfm2moe` | `Lfm2MoeCarrier` | Hybrid short-conv + GQA; dense SwiGLU and/or top-4 MoE; tied embeddings. Dir `lfm2` / `lfm2_moe` / `lfm2_vl` → 11. Optional VL via `hipfire-arch-lfm2-vl` (SigLIP2-NaFlex tower + projector — spec `docs/specs/2026-08-27-lfm2-vl-vision-runtime.md`, carrier recipe `docs/lfm2-vl-mq4v2-spec.md`). |
| 12 | Cohere2-MoE (North-Mini-Code) | `hipfire-arch-cohere2moe` | `Cohere2MoeCarrier` | Parallel block, interleaved sliding(RoPE)/global(NoPE) GQA, sigmoid MoE, dense layer-0, tied embeddings. |
| 13 | Gemma 4 (dense text) | `hipfire-arch-gemma4` | `Gemma4Carrier` | Hybrid 5:1 sliding(RoPE θ=10k, hd 256)/full(partial RoPE θ=1e6, hd 512, K=V sharing) GQA, sandwich RMSNorm + `layer_scalar`, gelu_pytorch_tanh SwiGLU, tied lm_head, final logit softcap 30. Dir `gemma4` / `gemma4_text` → 13. Carrier also claims 22. |
| 14 | Muse Glimmer (dense text) | `hipfire-arch-muse-glimmer` | `MuseGlimmerCarrier` | 3:1 sliding(RoPE θ=500k, SWA 2048)/full(**NoPE**, `layer_rope_theta[i]==0`) GQA 32:2 hd 128, sandwich RMSNorm (`rms_norm_eps` 1e-5 pre / `post_norm_eps` 1e-8 post), scale-less QK-norm + `qk_scale_factor` 3.87, `self_attn.gate_proj` gated attention, silu SwiGLU, **untied** lm_head, `output_multiplier` 0.196116 then softcap 20. Dir `muse_glimmer` / `muse_glimmer_text` → 14. Carrier claims 14 only; the arch-23 drafter rides `HIPFIRE_DFLASH_DRAFT`. `pp>1` and `params.drafter` are refused. |
| 15 | Maple-Preview (natively-ternary MoE) | `hipfire-arch-maple` | `MapleCarrier` | 3:1 sliding(RoPE θ=10k, SWA 512)/full(**NoPE**, `nope_on_global_attention`) GQA 16:4 hd 128, **QK-norm applied BEFORE RoPE**, **partial rotary 0.5** (first 64 of 128 dims), MoE on EVERY layer (256 experts top-8, `moe_intermediate_size` 512, no shared experts, softmax→topk→renorm), **clamped SwiGLU** `silu(clamp(gate,max=7))*clamp(up,-7,7)`, **untied** lm_head, embedding tensor is `model.word_embeddings` (not `embed_tokens`). Weights are natively ternary and carried losslessly by qt=51 `MQ2G256LloydU`; `intermediate_size` 4096 and `quantize`/`preaffine` are dead keys. Dir `maple` → 15. |
| 16 | Qwen3.8-Flash-Next / Qwen4 experimental | `hipfire-arch-qwen4` | `Qwen4Carrier` | Hybrid 3:1 GatedDeltaNet / QSA indexed sparse full attention (indexer budget 2,048), 4-stream hyper-connections, PLE n-gram embeddings on layer 2, 512-expert top-10 MoE with a shared expert, one native MTP layer. The carrier claims **HFQM artifacts only** (`!is_dir`): a safetensors dir with `model_type` `qwen4_exp` / `qwen4_exp_text` derives 16 as a *reserved* source classification (`RESERVED_MODEL_TYPE_TO_ARCH_ID`), but no carrier loads a directory. Text-only; vision and PP/TP are refused. **Expert parallelism** (`topology.ep` 2..=8, `HIPFIRE_QWEN4_EP=0` kill switch) is admitted on homogeneous exact gfx1100 or gfx1201 only, with the mesh inside the rank-0 bundle rather than `LoadedModel.ep`; owner-routed prefill is default-on from EP2 through EP8 on both arches and expert staging on gfx1201 only ([`ARCHITECTURE.md`](ARCHITECTURE.md#qwen4-expert-parallel-mesh-flash-next-ep2-through-ep8)). `max_seq` is bounded by the config `context_limit` (native 262,144; `attention.yarn_factor` / `HIPFIRE_YARN_FACTOR` extends it to factor × original, capped at `QWEN4_MAX_CONTEXT` = 1,048,576), default 32,768 when omitted. QSA K/V: `auto` is fp8 on exact gfx1201 and the exact `bf16` state elsewhere; `q8` (Q8_0 blocks of 32) is opt-in on gfx11/gfx12 ([`CONFIG.md`](CONFIG.md#mode-k-and-v)). PLE rows are not uploaded wholesale: they are read per token from the embedded Q8 rows, `<model>.ple-q8` or the 4-bit `<model>.ple-q4` (`memory.ple_format`, `auto` = **q4 on gfx1151**, q8 elsewhere). A `<model>.trunk-overlay` sidecar (zab: GDN Z/alpha/beta) applies by default only on exact gfx1151. The NPU trunk (`fusions.npu_trunk`) defaults to `auto` and arms only on gfx1151 with an admitted device and calibration. Native MTP attaches by default (greedy, and sampled under `speculation.mtp_sampled`) because Qwen4 declares a measured AR floor (`hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR = true`, the producer capability `speculation.mtp = auto` requires; its existing floor is unchanged); `.mq4r` keeps the retained AR route, and under EP only an explicit `--spec mtp` attaches it. |
| 17 | Qwen3.5 multi-vector embedding encoder (`pplx-embed-v2-late-0.6b`) | `hipfire-arch-qwen35-embed` | `Qwen35EmbedCarrier` | **Not a chat model.** A bidirectional Qwen3.5 hybrid: GatedDeltaNet linear-attention layers alternating 1:1 with gated full-attention layers, `is_causal:false` on the full-attention layers (the GDN layers keep their causal scan), partial 0.25 interleaved mRoPE, zero-centered RMSNorm, no LM head. A SigLIP-style tower (hidden 768, patch 16, `spatial_merge_size` 2) merges each 2×2 patch group into one 1024-d token spliced over `<|image_pad|>` runs; a sentence-transformers `Dense` head projects to 128 dims, a ColBERT skiplist drops punctuation token vectors (document side), and each kept vector is L2-normalized. Queries score documents by late-interaction MaxSim. Served through the non-chat `embed` wire verb (`hipfire embed`, `POST /v1/embeddings`), never the generation routes. Resolved by the bare composite class `Qwen3_5Model` — its `model_type` (`qwen3_5`) is shared with the causal LM (5), so only the `architectures` field discriminates (`derive_arch_id` checks `qwen3_5model` before the qwen3.5 family test). Weights are raw f32/f16 (no compressed trunk tier); the quantizer keeps norms, the Dense head and the tower raw, folds `1_Dense/model.safetensors` into the trunk, and auto-includes the vision tower. |
| 18 | EmbeddingGemma 2 (`google/embeddinggemma-2`) | `hipfire-arch-embeddinggemma2` | `EmbeddingGemma2Carrier` | **Not a chat model.** A single-vector embedding encoder on the Gemma-4 text architecture: 24 bidirectional layers (sliding ±512 radius 5:1 full), per-layer head_dim/KV overrides (sliding 4h×256d 2kv, full 4h×512d 1kv MQA), half-split RoPE over the full head dim (θ 10k sliding / 1M full), learned per-head Q/K RMSNorm + scale-less V norm, attention scaling **1.0** (the QK norms replace 1/√d), sandwich RMSNorm with **direct** (non-zero-centered) weights, gated gelu-tanh MLP, projection-only PLE (`ple.per_layer_model_projection` [24·512, 512] on the √512-scaled embeddings + per-layer gated block), `layer_scalar` per layer, final norm → `embedding_projection` to 768 d, sentence-transformers mean pooling + L2 (Matryoshka 768/512/256/128 via the optional `dims` field on the embed request). Served through the non-chat `embed` wire verb; its slot engine is the arch-17 shape (request channel, 2 ms batch window, row-budgeted groups, one batched forward). The tokenizer is a Gemma ▁-BPE and **requires the merge-rank SP-BPE path** (`Tokenizer::enable_sp_bpe_merges`) — the default greedy SentencePiece path is longest-literal, not merge-rank, and diverges from HF `tokenizers` on many ordinary texts (measured 8/29 sampled prompts; it agrees on all 9 parity fixtures, so that count is not a blanket guarantee). Its checkpoint's vision and audio towers ride ONE `.vl` sidecar (963 tensors: the quantizer packages `vision_tower.*`/`audio_tower.*`/`embed_vision.*`/`embed_audio.*` there and leaves the trunk tower-less), loaded only on `vision_mode` `auto`/`on` — `off` is the schema default and a text-only load refuses image/audio input by modality with a pointing error. Media rides the `embed` verb (`images_base64`, `audios_base64`, `videos_base64`; the HTTP route takes the typed `image_url` / `input_audio` / `input_video` content parts). The video leg has no tower of its own — its frames run the vision tower once per frame — so a load whose `.vl` pack is absent refuses image, audio and video by modality with a pointing error, and the `loaded` ack publishes `vl`/`audio`/`video` facts that `/v1/models` mirrors into `input_modalities`. Dir + `.hfq` sources; the quantizer keeps `embedding_projection` raw, and folds the ST prompt sidecars into `metadata.sentence_transformers`. |
| 19 | Raon-1-mini-preview (`KRAFTON/Raon-1-mini-preview`, 28.8B / 4.3B-active MoE) | `hipfire-arch-raon` | `RaonCarrier` | 25 layers, hidden 2560, vocab 259,840 (untied head). 18 RDA layers (gated-delta recurrence, conv kernel 4, full-rank gate, gate lower bound −5.0) and 7 MLA layers (q_lora 576 / kv_lora 512, nope 128 / rope 64 / v 128, output gate) with 4-stream mHC residuals. Layer 0 is a dense MLP (`first_k_dense_replace=1`); the rest are 128-expert top-6 + 1 shared SiTU MoE (sigmoid router, renormalized). Reasoning model (`<think>`, XML `<tool_call>` tools). Served by route `raon_ar`. |

Ids **2–4** are unused in the carrier registry (not claimed). Do not invent
assignments without a carrier + source mapping.

## Sidecar / reserved ids

These are **not** primary `load_model` trunk targets. Carriers must not claim
them for ordinary dispatch.

| arch_id | Role | Where defined | Notes |
|---:|---|---|---|
| 20 | DFlash draft artifact (generic) | `hipfire-quantize` `dflash_convert` (`src/bin/dflash_convert.rs`, `ARCH_ID_DFLASH_DRAFT`); loader draft open checks `draft_hfq.arch_id == 20` | Generic Qwen-family diffusion draft (fc/hidden_norm/norm naming, `dflash` metadata). Loaded as a draft alongside a compatible target, not as a standalone serve model. |
| 21 | Qwen3.5 MTP head | `hipfire-quantize` `mtp_extract` (`src/bin/mtp_extract.rs`, `ARCH_ID_QWEN35_MTP_HEAD`); consumed as `.mq4-mtp` trailer / `.mtp` sidecar on qwen35 trunks | Attached onto arch 5/6 loads (`LoadedModel.qwen35_mtp_head`), not a carrier `arch_id`, and only on an explicit `speculation.mtp = on` / `--spec mtp`: `auto` does not attach it because Qwen35 declares no measured AR floor (`hipfire_arch_qwen35::MTP_MEASURED_AR_FLOOR = false`). Refused before upload unless its metadata `n_embd`/`vocab_size` equal the trunk's hidden size and vocab (`mtp_head::check_mtp_head_for_trunk`). |
| 22 | Gemma4 EAGLE draft | `hipfire-arch-gemma4` `DRAFTER_ARCH_ID`; claimed by `Gemma4Carrier` alongside 13 | `model_type gemma4_unified_assistant` — hidden-state-consuming EAGLE (pre/post projections, KV-shared layers). Loaded via `params.drafter` on arch 13 as an `MtpSpeculator` (`Gemma4Drafter`). Serves tool-free greedy, sampled, penalized and `logprobs` requests; sampled and penalized ones verify by naive sampling (each verify row drawn with the AR sampler and the request's penalty history), so they follow the AR distribution up to batched-verify rounding. **Opt-in** (`HIPFIRE_GEMMA4_EAGLE=1`): batched verify is not bitwise-equal to single-row decode, so greedy output can leave the AR sequence at near-tied logits, and it is not faster than AR on gfx1151. |
| 23 | Muse Glimmer DFlash draft | `hipfire-arch-muse-glimmer` `GLIMMER_DRAFTER_ARCH_ID` | `model_type muse_glimmer_assistant` — 5-layer block-diffusion draft (encoder.fc/output_norm_enc, block 16, target_layer_ids [1,13,25,37,49], mask 201818). Loaded via `HIPFIRE_DFLASH_DRAFT` on arch 14. NOT a `dflash_convert` 20 artifact (different tensor naming: encoder.fc vs fc, separate `config` envelope). The `model_type` table in `arch_mapping.rs` maps `muse_glimmer_assistant` → 23 (the longest `model_type` substring wins, so it beats `muse_glimmer` → 14); HFQ on disk is `arch 23`. |
| 0xFF | Toy / template | `hipfire-arch-toy` | Never ship; daemon must not dispatch. No current carrier claims it. Registry disjointness tests include `0xFF` and assert at most one claimer — they do **not** assert zero claimers / hard-reserved. |
| `u32::MAX` | Unclaimed dir sentinel | `safetensors_source::UNCLAIMED_ARCH_ID` | Emitted for unrecognized `model_type`; no carrier matches → fail closed. |

## Image-generation component ids (40–47)

Diffusion checkpoints are **components, not chat models**: loadable for image
generation, never text-served. The block is deliberately **high** — ids 18–19
remain free for future primary text arches after Qwen4's id 16 and the
qwen3.5 embedding encoder's id 17. Component ids
must never silently collide with that range; the registry disjointness sweep
covers 0..=64, so 40+ is still test-enforced.

The block is **grouped by family, extension-only**: FLUX.1 owns 40–43 (trunk +
its two text encoders + the VAE it shares with FLUX.2), FLUX.2 owns 45–46 (44
is intentionally spare, kept free for a future need), and future families
(SD/SDXL) start at 47. New components get the next free id;
ids are never re-numbered once a pack ships.

Only the trunk ids (40, 45) are claimed by a carrier. The sidecar ids (41, 42,
43, 46) live in the headers of the per-component HFQ packs that
`hipfire-quantize --flux-pipe` writes, and the trunk's carrier resolves the
sidecars next to the trunk file; no carrier claims a sidecar id on its own.

| arch_id | Family | Crate | Carrier | Notes |
|---:|---|---|---|---|
| 40 | Flux MMDiT (checkpoint trunk) | `hipfire-arch-diffusion` | `FluxDiffusionCarrier` | `model_type flux` or `_class_name FluxTransformer2DModel` → 40. Daemon name `flux_mmdit`, served as `img_generate`/`img_progress`/`img_done`, `/v1/images/generations` and `hipfire img`. Refused by text `generate`/`bench_prefill`. |
| 41 | T5-XXL / T5 (sidecar, FLUX.1) | `hipfire-arch-diffusion` | (sidecar) | Text-conditioning encoder for FLUX/SD3, DFlash-sidecar precedent (20/23). |
| 42 | CLIP-L (sidecar, FLUX.1) | `hipfire-arch-diffusion` | (sidecar) | 768-d pooled-text encoder feeding the MMDiT text stream alongside the T5 token sequence. Packed as its own HFQ component (distinct forward contract from T5 — do not reuse 41). |
| 43 | VAE (sidecar) | `hipfire-arch-diffusion` | (sidecar) | Latent encode (img2img) / decode (pixels); not a serve trunk. Shared across FLUX.1 and FLUX.2 (the FLUX.2 `ae`): one sidecar slot serves both trunks. |
| 44 | (reserved spare) | — | — | Kept free; nothing ships on it. |
| 45 | FLUX.2 MMDiT / Klein (checkpoint trunk) | `hipfire-arch-diffusion` | `FluxDiffusionCarrier` | `_class_name Flux2Transformer2DModel` or `model_type flux2` → 45. Qwen3 text encoder (taps 9/18/27), 32-ch FLUX.2 VAE with BatchNorm latent stats, empirical sigma shift, reference-image edit via `img_generate` `images[]`. Daemon name `flux2_mmdit`. |
| 46 | FLUX.2 Klein Qwen3 text encoder (sidecar) | `hipfire-arch-diffusion` | (sidecar) | Standard Qwen3 causal-LM conditioner (model_type `qwen3`, Qwen3-4B geometry); distinct from arch 1 chat `qwen3` — attached only via the FLUX.2 pack, never loaded standalone. |
| 47 | SD/SDXL UNet (checkpoint trunk) | (planned) | (none) | Reserved for `model_type sd` / `sdxl`; nothing ships on it yet. |

## Source namespaces

| Namespace | Origin | Examples |
|---|---|---|
| HFQ header | `HfqFile::arch_id` written at quantize/pack time | Primary table above; sidecars 20/21 |
| Safetensors dir | `derive_arch_id(&config)` (table: `arch_mapping.rs`; longest `model_type` substring wins; `architectures` is checked before `model_type`) | `llama`/`mistral`→0, `qwen3`→1, `qwen3.5`/`qwen3.6`/`ornith` (+`num_experts`>0→6 else 5), `qwen3_5_moe`/`qwen3moe`→6, `qwen2`→7, `dots_ocr`→8, `deepseek_v4`→9, `minimax_m2`→10, `lfm2`/`lfm2_moe`/`lfm2_vl`→11, `cohere2_moe`→12, `gemma4`/`gemma4_text`/`gemma4_unified`/`gemma4_unified_text`→13, `muse_glimmer`/`muse_glimmer_text`→14, `maple`→15, architectures-first `qwen3_5model` (bare composite)→17, `qwen4_exp`/`qwen4_exp_text`→16 (reserved source classification; the carrier claims HFQM only), `gemma4_unified_assistant`→22, `muse_glimmer_assistant`→23, `flux`/`FluxTransformer2DModel`→40, `flux2`/`Flux2Transformer2DModel`→45 |

`Carrier::claims_arch_id(arch_id, is_dir)` may distinguish the two namespaces.
Today’s carriers are disjoint on bare id for the ids they claim; registry unit
tests sweep `0..=64` plus reserved 20 and `0xFF` for both `is_dir` values and
require at most one claimer (disjointness only — not a zero-claimer guarantee
for `0xFF`).

## Dispatch sites (maintainers)

| Concern | Location |
|---|---|
| Load (single arch-dispatch) | `hipfire_loader::load_model` → `REGISTRY` probe (`crates/hipfire-loader/src/lib.rs`) |
| EP load | `hipfire_loader::load_model_ep` (9, 10); Qwen4 EP (2..=8 ranks) is attached by `Qwen4Carrier` inside `load_model` (`attach_qwen4_ep_peers`) |
| Spec target / emitter by id | `hipfire_loader::carrier_for` |
| Early generate routes | `hipfire_loader::generation_early_route` (13 Gemma 4, 14 Muse Glimmer; 22 is excluded) |
| Generate producer route | `hipfire_generate::ar::select_generation_route` (EP first; then 16, 7, 9, 11, 12, 15, 10, 14, 8; then PP; then Qwen/LLaMA DFlash/spec; else AR) and the `generate` match in `crates/hipfire-generate/src/ar.rs` |
| Continuous-batch route | `hipfire_loader::continuous_batch_route` (5/6 Qwen3.5; 16 Qwen4, VMM lanes only) |
| Bench-decode route | `hipfire_loader::bench_decode_route` (9, 11, 5/6, 13, 14, 16; else unsupported) |
| Trait contract | `crates/hipfire-runtime/src/arch.rs`, `arch_model.rs` (`ArchModel`) |

Vision-capable loads: Qwen3.5-VL rides ids **5/6** with optional vision
weights; dots.ocr is id **8**. New vision arches need explicit generate and
HTTP image gating in the daemon, not only a carrier.

## Related

- Crate / data-flow map: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Model catalog (tags ≠ admission): [`MODELS.md`](MODELS.md)
- Docs ownership / truth states: [`INDEX.md`](INDEX.md)
