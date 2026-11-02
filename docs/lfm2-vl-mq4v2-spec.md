# Spec: LFM2.5-VL-3B — `mq4v2` quantization + vision support on arch 11

> **Lifecycle: historical (design and bring-up record).** Retained for
> provenance of the 2026-08 arch-11 LFM2.5-VL carrier work. It is **not**
> current procedure, product authority, or acceptance: the §6 acceptance
> criteria and the §7 file list are the original plan, not a live gate, and
> the `pipeline.rs` / `lfm2moe.rs` line numbers cited below refer to the
> bring-up branch and are not maintained. Current behavior is defined by the
> code (`crates/hipfire-quantize/src/pipeline.rs`,
> `crates/hipfire-quantize/src/model_filter.rs`,
> `crates/hipfire-runtime/src/arch_mapping.rs` `lfm2_vl` → 11,
> `crates/hipfire-arch-lfm2moe/src/lfm2moe.rs` qt 44 → `MQ4G256V2`,
> `crates/hipfire-arch-lfm2-vl/`), by
> [`architecture-ids.md`](architecture-ids.md) (arch 11), by
> [`specs/2026-08-27-lfm2-vl-vision-runtime.md`](specs/2026-08-27-lfm2-vl-vision-runtime.md),
> and by [`VALIDATION.md`](VALIDATION.md) for any VL claim route (route
> existence is not per-checkpoint evidence). The §4 metadata contract is a
> historical statement of the plan; see the sibling record
> [`qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md).

Status at the time: DRAFT (implementation branch `feat/lfm2-vl`; sibling
arch-5 carrier on `feat/qwen35-vl`). The `hipfire-arch-lfm2-vl` crate this
draft proposed as "new" is now in-tree.
Author: engineering
Target checkpoint: `LiquidAI/LFM2.5-VL-3B` (`model_type: lfm2_vl`)

> **As-built on `feat/lfm2-vl` (2026-08-27):** the §2 quantizer recipe is
> in — `embed_tokens` routes through the mq4 bulk branch under mq4 formats
> (tied head covered), the `multi_modal_projector` joins the F16 vision
> group, `has_vision` + pixel-budget metadata are emitted per the shared
> contract (§4 of [`qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md)),
> `lfm2_vl` maps to arch 11 in `arch_mapping.rs`, and the arch-11 loader
> decodes qt 44 (`MQ4G256V2`) with batch allowlists extended.
> **Validated on gfx1101 (2026-08-27):** the requantized
> `lfm2.5-vl-3b-vlval.mq4v2.hfq` loads and generates coherent text on the
> arch-11 carrier (greedy 17×23→"391" smoke, container-isolated run) with
> census 441 F16 / 167 MQ4G256V2 / 99 Q8F16 matching lineage exactly.
> **R1 resolved positively:** the gather path consumes MQ4G256V2 embeds via
> host-dequant (`weight_backend::dequant_f32` qt44 arm) rather than falling
> back to Q8. Image requests fail closed ("model has no vision encoder").
>
> **§3.3–3.4 as-built (later the same day):** the `hipfire-arch-lfm2-vl`
> crate, projector wiring, VL forward and prompt splice are now implemented
> per [`docs/specs/2026-08-27-lfm2-vl-vision-runtime.md`](specs/2026-08-27-lfm2-vl-vision-runtime.md)
> — image turns run end-to-end locally and over HTTP on gfx1101. This spec's
> acceptance §6.4 structural-image clause is thereby met; vision-quality
> parity remains gated behind future `docs/VALIDATION.md` VL rows.

## 0. Goal

Produce a servable `.hfq` for **LFM2.5-VL-3B** where:

1. The **language backbone** (`model.language_model.*`) is quantized to **MQ4G256V2**
   (the `mq4v2` codec), with **`lm_head` and `embed_tokens` also in mq4v2** (not the
   current hardcoded Q8) so the whole text tower is one consistent 4-bit codec.
2. The **vision tower** (`model.vision_tower.vision_model.*`) and **projector**
   (`model.multi_modal_projector.*`) are embedded and loadable, so the engine can
   encode images. Vision weights are stored in **F16** (the engine's vision kernels
   consume F16 matrices; bf16 vision is explicitly unsupported — see §6).
3. The artifact is **loadable** on this box (RDNA3 / gfx1101) — i.e. correctness of
   load + decode to F32 reference is verified. The gfx12-only *batched* mq4v2 prefill
   path is out of scope; we rely on the per-token GEMV fallback that already exists
   for v2 on non-gfx12.

This design deliberately mirrors the **`hipfire-arch-qwen35-vl`** crate, whose vision
tower is a SigLIP-2 ViT — the *same family* as LFM2.5-VL's encoder (SigLIP2 NaFlex).

## 1. Checkpoint tensor inventory (from model.safetensors)

| Prefix | Count | Kind | Target dtype in `.hfq` |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | 1 | [128000, 2048] | **MQ4G256V2** |
| `model.language_model.embedding_norm.weight` | 1 | [2048] | F16 (norm) |
| `model.language_model.layers.{N}.conv.conv.weight` | 22 | [2048,1,3] | MQ4G256V2 (conv handled as 2D) |
| `model.language_model.layers.{N}.conv.in_proj/out_proj.weight` | 22 ea | [6144,2048]/[2048,2048] | MQ4G256V2 |
| `model.language_model.layers.{N}.self_attn.{q,k,v,out}_proj.weight` | 8 ea (layers 13–29) | [..,2048] | MQ4G256V2 |
| `model.language_model.layers.{N}.self_attn.{q,k}_layernorm.weight` | 8 ea | [64] | F16 |
| `model.language_model.layers.{N}.feed_forward.w{1,2,3}.weight` | 30 ea | [10752,2048]/[2048,10752]/[10752,2048] | **MQ4G256V2** |
| `model.language_model.layers.{N}.{ffn_norm,operator_norm}.weight` | 30 ea | [2048] | F16 |
| `model.vision_tower.vision_model.embeddings.patch_embedding.{weight,bias}` | 1 ea | conv [1152,3,?,?] / [1152] | F16 |
| `model.vision_tower.vision_model.embeddings.position_embedding.weight` | 1 | [N,1152] | F16 (or omit if absolute) |
| `model.vision_tower.vision_model.encoder.layers.{n}.self_attn.{q,k,v,out}_proj.{weight,bias}` | 27 ea | [1152,1152]/[1152] | F16 |
| `model.vision_tower.vision_model.encoder.layers.{n}.mlp.fc{1,2}.{weight,bias}` | 27 ea | [4304,1152]/[1152,4304]/[1152] | F16 |
| `model.vision_tower.vision_model.encoder.layers.{n}.layer_norm{1,2}.{weight,bias}` | 27 ea | [1152] | F16 |
| `model.vision_tower.vision_model.post_layernorm.{weight,bias}` | 1 ea | [1152] | F16 |
| `model.multi_modal_projector.linear_{1,2}.{weight,bias}` | 1 ea | [2048,1152]/[2048] (verify) | F16 |

> Note: 22 of 30 transformer blocks use the **conv mixer** (`conv.*`); only layers
> 13–29 add attention. Quantize both mixer variants + FFN to mq4v2.

> **As-built correction (2026-08-26):** the quantizer routes all norms
> (`embedding_norm`, `ffn_norm`, `operator_norm`, `{q,k}_layernorm`) to
> **Q8F16**, not F16 as the table above planned — verified in the census of
> `lfm2.5-vl-3b.mq4v2.hfq` (local build; 99 Q8F16 tensors, 441 F16 =
> vision+projector only). The table reflects the original plan; the artifact
> is authoritative.

## 2. Quantizer changes (`crates/hipfire-quantize`)

### 2.1 Selection
- Reuse the existing `--format mq4v2` → `use_mq4v2` flag (already parsed;
  `pipeline.rs:291` maps `mq4v2|mq4g256v2 → use_mq4v2`). No new flag needed.
- The MoE-v2 guard at `pipeline.rs:1072` excludes `mq6/5/3/2 v2` for `is_moe_like`,
  but **not** `mq4v2` (it is explicitly allowed: "Use legacy mq{..} or mq4/mq4v2/mq4c
  for MoE"). arch 11 is `is_moe_like`, so mq4v2 is already permitted. No change there.

### 2.2 `try_handle_lfm2moe` (arch 11) — `pipeline.rs:3215`
Currently the bulk branch (`use_mq4g256 || use_mq4v2 || use_mq4c`) already emits
MQ4G256V2 for 2D proj/FFN mats **except** `embed_tokens`. Two gaps:
1. **embed_tokens** is excluded (`!name.ends_with("embed_tokens.weight")`) → falls to
   the Q8 tail. **Fix**: when `use_mq4v2` (or `use_mq4g256`), route
   `model.language_model.embed_tokens.weight` through the same MQ4G256V2 packer
   (`quantize_mq4g256v2`, already imported at `pipeline.rs:3340`). Embed is a gather
   in the runtime, so it must be emitted as a *quantized* tensor the loader can map;
   we emit `QuantType::MQ4G256V2` with group_size 256. (If the loader gather path
   cannot consume mq4v2 embed, fall back to keeping embed at Q8 — see §5 risk.)
2. **lm_head**: LFM2-VL uses `model.language_model.embed_tokens` as a *tied* head
   (`tie_word_embeddings` — confirm in config). If tied, there is **no** `lm_head`
   tensor in the checkpoint and the loader reuses `embed_tokens`. Quantizing embed to
   mq4v2 therefore covers lm_head for free. If untied, add an explicit lm_head branch
   mirroring embed.

### 2.3 Vision embedding
- `pipeline.rs:is_vision` already matches `model.vision_tower.*` (line 1677). With
  `--include-vision` set, vision tensors reach `handle_bf16_passthrough` / the generic
  branch. For arch 11 we **do not** use bf16 passthrough (guard rejects arch 11).
  Instead: when `use_mq4v2` **and** `is_vision`, emit vision 2D mats as **F16**
  (`QuantType::F16`, `qt=1`) — lossless from the bf16 source — and norms/biases as
  F16 too. Robust path: add an explicit early `if is_vision { emit_f16(name, raw) }`
  before `try_handle_lfm2moe` so vision never hits the Q8 tail. F16 matrices are what
  the ViT forward expects.

### 2.4 Projector
- `model.multi_modal_projector.*` is not matched by `is_vision` (it doesn't start with
  `vision_tower.`). Add it to the vision/projector predicate OR handle it explicitly:
  emit as F16 (it's a tiny MLP; precision-sensitive). Treat as part of the "vision
  module" group so it is embedded with `--include-vision`.

## 3. Runtime changes (`crates/hipfire-runtime` + `hipfire-arch-lfm2moe` + new `hipfire-arch-lfm2-vl`)

### 3.1 DType plumbing (already present — verify, do not re-add)
- `DType::MQ4G256V2` exists (`llama.rs` usage at 966/1027/1340/1418/1508/1884).
- qt constant: confirm `MQ4G256V2` qt value (comment at 1501 says "qt=44 twin"). The
  loader qt→DType map in `lfm2moe.rs:1044` must gain a `44 => DType::MQ4G256V2` arm
  (currently only 13=MQ4G256). **This is the critical runtime fix.**
- `Lfm2MoeWeights` loader (`lfm2moe.rs`) must accept `MQ4G256V2` for dense proj / conv
  / embed / lm_head: extend the `match gpu_dtype` arms at lines 1656 (conv), and the
  embed/lm_head allowlists (1624–1643) to include `MQ4G256V2`.
- `batch_weight_formats_supported` (1627): add `DType::MQ4G256V2` to dense-proj,
  embed, and lm_head allowlists. (Batching on gfx11 falls back to per-token GEMV for
  v2 — that fallback must exist or we mark this arch decode as per-token only.)

### 3.2 Dense forward dispatch (arch 11)
- The arch-11 `forward`/`weight_gemv` path currently handles `MQ4G256` (qt 13). Add a
  `DType::MQ4G256V2` arm that:
  - uploads the v2 weight blob,
  - rotates the activation with `rotate_x_mq_batched_for` (same FWHT plan, seeds 42/1042
    — see `llama.rs:1508` comment: identical rotation to v1),
  - dispatches to the **v2** GEMM/GEMV launcher (`gemm_hfq4g256_batched_lmhead` v2
    twin for lm_head; the per-token `weight_gemv` v2 variant for layers).
  - On gfx11, route to the per-token GEMV fallback (mirror how v1 does on non-gfx12).

### 3.3 Vision tower (new crate `hipfire-arch-lfm2-vl`, templated off `hipfire-arch-qwen35-vl`)
- Reuse the SigLIP-2 ViT forward skeleton from `qwen35-vl/src/qwen35_vl.rs`:
  - patch embedding (conv) → position embedding → N encoder layers
    (self-attn + MLP + pre/post layernorms) → post-layernorm → pool/extract.
  - **Adaptations for LFM2-VL**:
    - No mRoPE (Qwen uses 3D mrope; LFM2 uses 1D learned positions). Drop `mrope.rs`.
    - Patch size / grid come from `encoder_patch_size` (in config.json) and the NaFlex
      dynamic resolution path; reuse `image.rs::smart_resize`/`extract_patches` logic
      but strip the Qwen-specific mrope tie-in.
    - Embedding layout: `patch_embedding` is a conv; `position_embedding` is absolute
      (SigLIP2). Verify against config (`do_image_splitting`, `downsample_factor`).
- `VisionWeights` struct + `load_vision_weights` + `vision_config_from_hfq` that read
  the embedded F16 tensors by the **exact names** in §1.

### 3.4 Combined VL arch + projector
- New `Lfm2Vl` `Architecture` impl (mirror `qwen35-vl/src/arch.rs`):
  - `arch_id()` returns 11 (reuse LFM2 arch id; the HFQ header carries a `has_vision`
    flag we add to metadata).
  - `config_from_hfq` parses text config **and** builds the vision config when
    `model.vision_tower.*` tensors are present.
  - `load_weights` loads the LFM2 text `Lfm2MoeWeights` **plus** `VisionWeights` plus
    the `multi_modal_projector` MLP.
- `Lfm2MoeBundle` gains an `Option<VisionWeights>` + `Option<ProjectorWeights>`; the
  forward path, when an image is attached, runs `vision_forward` → projector → injects
  visual tokens into the text KV/positions (LFM2 places `<image>` tokens; follow the
  HF chat template contract).

## 4. HFQ metadata
- Add a `has_vision: bool` + `vision_config` blob to the HFQ `metadata_json` emitted by
  the quantizer for `lfm2_vl` inputs, so the loader knows to instantiate the vision
  tower + projector.
- The field set, vision dtype policy (F16), embedded layout, and naming rules are the
  **shared VL artifact contract** — defined once in
  [`qwen35-vl-mq4v2-spec.md`](qwen35-vl-mq4v2-spec.md) §4 (the arch 5 family spec).
  Keep the two in sync; that section is authoritative.
- Add `lfm2_vl` → 11 to `MODEL_TYPE_TO_ARCH_ID` (`arch_mapping.rs:55`) so the
  quantizer/runtime don't need `--arch-id 11` and so `Lfm2Vl` is auto-selected.

## 5. Risks / decisions
- **R1 (embed as mq4v2)**: if the gather path can't consume a quantized embed, keep
  embed at Q8 (the lm_head then is whatever embed maps to). Documented trade-off.
- **R2 (gfx11 v2 batching)**: v2 batched prefill is gfx12-only. On this box we use the
  per-token GEMV v2 path (must exist for correctness; perf is secondary for bring-up).
- **R3 (SigLIP2 NaFlex details)**: LFM2 uses NaFlex dynamic resolution; exact
  resize/grid logic must match HF `transformers` preprocessing. Bring-up uses a fixed
  small image (e.g. 256×256 → 64 visual tokens) to validate the forward end-to-end
  before tuning resolution handling.
- **R4 (bf16 vision impossible)**: vision is F16 by design (engine constraint). No
  bf16 vision in this engine — confirmed in `handle_bf16_passthrough` (arch 5|6 only).

## 6. Acceptance criteria
1. `hipfire quantize <LFM2.5-VL-3B> --format mq4v2 --include-vision --arch-id 11`
   (or auto via `lfm2_vl` mapping) **produces** `LFM2.5-VL-3B.mq4v2` with every
   text 2D mat + embed in `MQ4G256V2`, every vision/projector tensor in F16.
   (Embed tier is R1-dependent: if the gather can't consume a quantized embed,
   the Q8 fallback applies and the trade-off is recorded here.)
2. A structural qt-map census (per-tensor QuantType counts from the quantizer
   summary — not `strings | grep`, which this spec's sibling §2 rule retires)
   confirms bulk mq4v2.
3. The `.hfq` **loads** on this box (gfx1101): `Lfm2Vl::load_weights` succeeds,
   `batch_weight_formats_supported` passes for mq4v2 text tensors, vision/projector
   tensors upload as F16.
4. A text-only forward (no image) returns sane logits (sanity check vs HF reference
   on a 1-token prompt). Image forward is validated structurally (tensor shapes) on
   the first small image; full vision-quality parity is a follow-up.

## 7. Files touched (summary)
- `crates/hipfire-quantize/src/pipeline.rs` — embed→mq4v2; vision/projector→F16;
  `--include-vision` group; `lfm2_vl` metadata.
- `crates/hipfire-runtime/src/arch_mapping.rs` — `lfm2_vl`→11.
- `crates/hipfire-arch-lfm2moe/src/lfm2moe.rs` — qt 44→MQ4G256V2; allowlists.
- `crates/hipfire-arch-lfm2moe/src/forward.rs` (or forward_batch.rs) — v2 dispatch.
- `crates/hipfire-arch-lfm2-vl/` (NEW) — vision tower + VL arch + projector,
  templated from `hipfire-arch-qwen35-vl`.
- `Cargo.toml` (workspace) — register new crate.
