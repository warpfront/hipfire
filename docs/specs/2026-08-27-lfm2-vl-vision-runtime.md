<!-- SPDX-License-Identifier: Apache-2.0 -->

# Narrow spec: LFM2-VL vision runtime bring-up (`hipfire-arch-lfm2-vl`, arch 11)

Status: **branch-implemented on `feat/lfm2-vl` (2026-08-27)** — see the
as-built blockquote below; quality claims beyond the cited smokes remain
fail-closed until a `docs/VALIDATION.md` VL row exists.
Author: engineering (hipfire agent)
Date: 2026-08-27
Parent carrier recipe: [`docs/lfm2-vl-mq4v2-spec.md`](../lfm2-vl-mq4v2-spec.md)
§3.3–3.4 (the unbuilt remainder this record specifies)
Sibling family spec: [`qwen35-vl-mq4v2-spec.md`](../qwen35-vl-mq4v2-spec.md) on `feat/qwen35-vl`
(§4 shared VL artifact contract remains authoritative for metadata/dtypes)
Validation routing: [`docs/methodology/arch-port-validation.md`](../methodology/arch-port-validation.md)
(model-arch forward bring-up loop); `docs/VALIDATION.md` has **no** VL row
yet — VL quality claims stay fail-closed `unknown surface` until a row lands.

> **As-built on `feat/lfm2-vl` (2026-08-27, same day).** All of §2 landed:
> crate `hipfire-arch-lfm2-vl` (11 CPU unit tests green: unshuffle index
> contract, `(dy,dx,c)` patchify, banker's rounding, token-count arithmetic,
> pos-resize identity/upscale/downscale, erf-GELU values, config defaults);
> bundle/loader/carrier wiring incl. `VisionRoute::Lfm2Vl`,
> `has_vision_encoder()`, caps flip with re-pinned registry tests;
> `forward::prefill_embed_step`; `generate_lfm2_vl`. The published artifact's
> patch embedding is the **Linear `[1152,768]`** serialization (R1 resolved —
> conv fallback retained anyway). Deviations from §2.1/R3 accepted: none
> beyond the documented resize-filter residual. Two serve-layer fixes rode
> along per §2.3/§5-note: `gen_start` openers for generate_vl /
> generate_vl_dots_ocr, and the lfm2-VL abort paths emit the canonical
> `aborted`+`aborted_done` terminal pair — a raw custom `aborted` event holds
> the serve admission guard forever (this was THE wedge mechanism behind the
> 2026-08-27 ledger findings b+c for image turns).
>
> Evidence (`bin md5`: daemon `bcc10e45…`, cli `89830c25…`; fixtures
> sha256-pinned to the 2026-08-27 ledger): local-spawn smoke
> text → "391" ✓; doge_napping.png OCR → all six meme captions read
> ("Pointy teeth wow" — one cluster varies from the qwen35 carrier reading);
> scene_2.jpg OCR → YSL / ERMENEGILDO ZEGNA / BOUCHERON ✓ with dense-fine-print
> variance elsewhere (same class as the documented qwen35 "MQUEEN" miss).
> HTTP on swapped `vl-serve-lfm2` (:6900): `/v1/models` lists the catalog,
> alias `lfm2-vl` resolves, text 391 ✓, image turn streams correct captions
> in ~22 s, **and client-disconnect mid-image-turn frees the slot instantly**
> (immediate follow-up answers) — findings (b)+(c) closed for arch 11.
> Bring-up tier only: no perf numbers claimed, no VALIDATION.md rows claimed.
>
> **Audit addendum (2026-08-27 evening).** Branch audited end-to-end; design
> upheld, five hardening fixes landed on top: the visual/projector row-count
> check before splicing is now a release-mode fail-closed error (§2.3 always
> claimed "fails loud"; `debug_assert` compiled out in release), prefill
> client-cancel emits the canonical cancelled pair and returns immediately
> (mirrors the generate_vl hardening on `feat/qwen35-vl` 84fb32dd — no more
> break-and-fall-through past an empty-logits hazard), the done envelope
> carries `prefill_ms` + a full-turn `total_ms` (tower phase included), the
> dead "defensive" wrapper-marker branch in `expand_image_placeholders` is
> gone (§1.5: markers always required), and the daemon's VL-dispatch reset
> guard gained its missing lfm2moe arm (stale "only arch 5|6/8" comment
> corrected; HTTP-dormant because serve resets per-request for
> non-cache-capable lfm2 — defense-in-depth for raw JSONL multi-turn).
> Revalidated container-isolated: text 391 ✓, doge six-caption exact ledger
> match ✓, scene_2 brands ✓, disconnect-mid-image-turn frees the slot with a
> clean follow-up ✓, and an old-vs-new A/B container showed byte-identical
> greedy output (no behavioral drift). Evidence:
> `.codeinsight+research/lfm2-vl-audit-2026-08-27/RESULTS.md`.

## 0. Goal and scope

Implement **vision execution** for `lfm2_vl` artifacts on the arch-11
carrier so that:

1. A `.mq4v2.hfq` quantized with `--include-vision` (already produced and
   census-verified on 2026-08-27: 441 F16 vision/projector tensors +
   `has_vision:true` + pixel-budget metadata inside
   `lfm2.5-vl-3b-vlval.mq4v2.hfq`) loads its SigLIP2-NaFlex tower,
   projector included, onto the GPU at model-load time.
2. An image-bearing request runs encode→project→splice→generate
   end-to-end through `run --image` (local spawn) and the HTTP serve path.
3. Text-only behavior of existing artifacts is byte-unchanged (no vision
   tensors ⇒ vision stays `None`; generation path identical).

Out of scope (fail-closed or follow-up): multi-image requests, DFlash/spec
× VL, CASK eviction × VL (arch 11 has none), video, bf16 vision, VRAM-tiered
tower offload, per-SKU calibration docs.

## 1. Upstream contract (LiquidAI/LFM2.5-VL-3B, HF `transformers` lfm2_vl)

Pinned reference sources (fetched 2026-08-27 to `/tmp/lfm2vl-hf`):
`modeling_lfm2_vl.py`, `image_processing_lfm2_vl.py`,
`processing_lfm2_vl.py`, `configuration_lfm2_vl.py`,
`modeling_siglip2.py`, checkpoint `config.json`,
`chat_template.jinja`.

### 1.1 Checkpoint config facts (`config.json`, verified)

- `model_type: lfm2_vl`; text_config is standard arch-11 LFM2 (30 layers,
  layer_types conv/full_attention mix, hidden 2048, vocab 128000, tied).
- Vision tower: `siglip2_vision_model` NaFlex — hidden 1152, heads 16
  (head_dim 72), layers 27, mlp 4304, patch 16, `num_patches: 256`
  (= 16×16 learned position table), `hidden_act: gelu_pytorch_tanh`,
  `layer_norm_eps: 1e-6`, `vision_use_head: false` (**no pooling head** —
  downstream consumes `last_hidden_state` directly).
- Projector: `downsample_factor: 2`, `projector_hidden_size: 2048`,
  `projector_bias: true`, `projector_hidden_act: "gelu"` (**exact erf-GELU**,
  distinct from the tower's tanh-GELU), `projector_use_layernorm: false`.
- Splitting: `do_image_splitting: true`, `tile_size: 512`, `min_tiles: 1`,
  `max_tiles: 10`, `use_thumbnail: true`, `encoder_patch_size: 16`,
  `min_image_tokens: 64`, `max_image_tokens: 256`,
  `max_pixels_tolerance: 2.0`.
- Tokens/ids: `image_token_id: 124907` (`<image>`), eos `<|im_end|>`
  (124900), bos `<|startoftext|>` region ids 124893/124894.
- Pixel normalization: rescale 1/255 then **IMAGENET_STANDARD**
  mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)` — NOT Qwen's
  `/127.5 − 1`. Resize filter bilinear (+antialias in torch/torchvision).

### 1.2 Preprocessing pipeline (`Lfm2VlImageProcessor`, exact semantics)

Given original `(H, W)`:

1. `smart_resize` with `total_factor = encoder_patch_size × downsample = 32`,
   `min_px = 64·16²·2² = 65 536`, `max_px = 256·16²·2² = 262 144`;
   round-to-factor via Python `round()` (**banker's rounding**) when within
   budget, shrink uses `floor`, grow uses `ceil`. Returns dims divisible by 32.
2. "Too large" test (triggers splitting): rounded-dims pixels exceed
   `max_image_tokens·ps²·ds²·tolerance = 524 288 px`.
3. Split path: target grid `(cols, rows)` = aspect-closest ratio from all
   `(w,h)` with `w·h ∈ [min_tiles..max_tiles]` (ties prefer larger original
   area vs `tile_size²·w·h·0.5`); whole image resized to
   `(512·rows, 512·cols)`, split row-major into tiles; when grid ≠ 1×1 a
   **thumbnail** sized to the step-1 dims is appended LAST.
4. Non-split path: single sub-image resized straight to the step-1 dims.
5. Every sub-image independently: normalize → patchify with layout
   `(dy, dx, channel)` flat per patch (patch vector index =
   `(dy·16+dx)·3 + c`) → `nph×npw` patch rows.

### 1.3 Tower forward (SigLIP2 NaFlex, exact)

Per sub-image with patch grid `(gh, gw)`, rows = `gh·gw`:

```
x[rows]      = W_pe · patches + b_pe            # Linear(768→1152), no conv
pos          = resize(pos_table.reshape(16,16,D), (gh,gw))
             # F.interpolate bilinear, align_corners=False, antialias=True
h            = x + pos
27 × : h += out_proj(attn(LN1(h))) ; h += fc2(tanh_gelu(fc1(LN2(h))))
out          = LN_post(h)
```

Attention: separate q/k/v/out projections **with bias**, scale =
`head_dim^-0.5`, bidirectional softmax in fp32, **no RoPE**, no pooling,
dropout 0. LN eps 1e-6 pre-norm everywhere + final `post_layernorm`.
Padding/masking (NaFlex packs variable-length rows up to 1024 patches) is a
batching artifact: computing a single unpadded sub-image yields identical
rows, so hipfire processes each sub-image unpadded and skips masks entirely.

### 1.4 Projector (`Lfm2VlMultiModalProjector`, exact)

Input: unpad tower output, reshape to `(1, gh, gw, 1152)`. Then
HF's `pixel_unshuffle(f=2)` — note HF's dim names are swapped relative to
the values passed in; replicating their ops verbatim, for block-row `br`,
block-col `bc`, intra offsets `di,dj ∈ {0,1}`, channel `c`:

```
token_vec[di·2304 + dj·1152 + c] = feat[(2br+di)][(2bc+dj)][c]
```

i.e. **columns pair first** (`dj` stride 1152), rows second (`di` stride
2304). Output token count per sub-image = `(gh/2)·(gw/2)`. Then
(optional LN skipped) → `linear_1` (4608→2048, bias) → **exact
erf-GELU** → `linear_2` (2048→2048, bias). Per-sub-image projected outputs
concatenate tiles row-major then thumbnail.

### 1.5 Prompt / frame contract

Chat template is plain LFM2.5 ChatML (`<|im_start|>user\n … <|im_end|>\n`
+ `<|im_start|>assistant\n`) with image content rendered as a single literal
`<image>` in the user turn. The processor expands it, wrapped by special
markers, into placeholder runs of id 124907:

- always: `<|image_start|>` … `<|image_end|>`;
- multi-tile: per tile in row-major order `<|img_row_R_col_C|>` followed by
  256 `<image>` (tokens_per_tile = `(512/16/2)² = 256`), then
  `<|img_thumbnail|>` + `(new_h//32)·(new_w//32)` `<image>`;
- single tile: `<image>` repeated `(new_h//32)·(new_w//32)`.

Features are scattered positionally over ALL 124907 ids under the markers;
language positions are plain 1D (no mRoPE anywhere in this family).

## 2. Runtime design

### 2.1 New crate `crates/hipfire-arch-lfm2-vl` (templated on `hipfire-arch-qwen35-vl`)

- `VisionConfig`: parsed from artifact metadata
  `config.vision_config` (+ top-level keys the quantizer already merged:
  `downsample_factor`, `do_image_splitting`, `tile_size`, `max/min_tiles`,
  `min/max_image_tokens`, `max_pixels_tolerance`, `projector_*`,
  `use_thumbnail`). Defaults hardcoded from §1.1 when absent.
- `VisionWeights { tower weights…, projector linear_1/linear_2 }`: F16
  tensor loads exactly mirroring `qwen35_vl::load_f16_gpu/load_f32_cpu`
  (qt 1 direct; 2/6/7 narrow-or-dequant), reading names
  `model.vision_tower.vision_model.*` (per-layer
  `self_attn.{q,k,v,out}_proj`, `mlp.fc{1,2}`, `layer_norm{1,2}`,
  embeddings + post_layernorm) and `model.multi_modal_projector.linear_{1,2}`.
  Patch-embedding weight accepts BOTH serializations seen in the wild —
  Linear `[1152,768]` or Conv `[1152,3,16,16]` — normalized at load time to
  row order `(dy,dx,c)` matching §1.2 patch vectors.
- `image.rs`: LFM preprocessing ported exactly per §1.2 (LFM-specific —
  differs from qwen35's `smart_resize` in factor 32 / banker's rounding /
  IMAGENET_STANDARD norm / split+tiles+thumbnail). Returns the ordered
  sub-image list `(pixels CHW, gh, gw, kind)` + the exact expansion token
  counts so the prompt builder and the forward CANNOT disagree.
- `vision_forward(gpu, w, cfg, &sub_images) -> Vec<f32>`: runs the tower
  kernels per sub-image (`gemm_f16[_wmma_mb8]`, `layernorm_batched`,
  `vit_attention_f32`, `gelu_tanh_f32`, `add_inplace_f32`, `bias_add_f32`,
  `transpose_f32` — all existing; gfx1101 validated by the qwen35 work),
  pos-table resize + pixel-unshuffle on CPU (small data, exact-index unit
  tested), projector linears on GPU with the **erf-GELU applied on host**
  between them (exact `erff` without adding a kernel family; ≤ ~23 MB
  round-trip worst case). Concatenates per-sub-image outputs.
- `free_gpu`: full teardown of every uploaded tensor (drained on unload —
  pointer-keyed silent-corruption class, AGENTS.md §2B).
- Unit tests (CPU-only): pixel-unshuffle index formula, patchify layout
  `(dy,dx,c)`, smart_resize rounding/bounds incl. banker's rounding,
  token-count arithmetic (single + split + thumbnail), antialiased
  downsample mass conservation, config parse from a metadata fixture.

Known deviation, accepted for bring-up: position-embedding resize uses
separable triangle-filter antialias only on downscales (upscale path is
plain bilinear, identical to torch). Conv-resize filters differ slightly
from torchvision — same class as qwen35's CatmullRom-vs-PIL residual
(rel-L1 ~0.002, below model sensitivity measured there).

### 2.2 Bundle + loader wiring

- `spec_impl::Lfm2MoeBundle` gains
  `pub vision: Option<hipfire_arch_lfm2_vl::VisionWeights>` +
  `pub vision_config: Option<…VisionConfig>` (crate dep
  `hipfire-arch-lfm2moe → hipfire-arch-lfm2-vl`; acyclic — mirrors how
  `Qwen35Bundle` carries `qwen35_vl` types).
- `carriers.rs` `Lfm2MoeCarrier::load` Hfq arm: probe
  `model.vision_tower.vision_model.embeddings.patch_embedding.weight`
  BEFORE trunk consume (single-pass HFQ rule, mirrored verbatim from the
  qwen35 arm incl. reclaim-on-trunk-failure); stash into the bundle after
  `load_lfm2moe_bundle` returns. Dir source stays text-only.
- `arch_model.rs free_gpu`: destructures + frees vision when present.
- `caps.supports_images` flips to `true` on `Lfm2MoeCarrier` (text-only
  checkpoints still refuse images at the daemon gate via
  `has_vision_encoder() == false` — same declared-capability tradeoff the
  qwen35 carrier makes today).
- `loader::VisionRoute` gains `Lfm2Vl`; route table maps `11 => Lfm2Vl`.
- New `LoadedModel::has_vision_encoder()` covering qwen35-vl, dots-ocr AND
  lfm2 bundles; the daemon gate switches to it (old qwen-typed accessors
  unchanged).

### 2.3 Forward seam + generation

- `forward.rs`: new public
  `prefill_embed_step(cfg, weights, state, gpu, embedding: &[f32], position)`
  = bounds-check + htod pos + `memcpy_htod` the embedding row into
  `state.h` (replacing the `embedding_lookup_dispatch` call of
  `prepare_retained_decode_inputs`) + the ordinary decode body. No changes
  to any existing function's behavior.
- `hipfire_generate::vision::generate_lfm2_vl`: structurally a copy of
  `generate_lfm2moe`'s proven shape — **stream-contract opener first**
  (`emit_gen_start` before ANY error/GPU work; see §5 note), cross-turn
  cold reset, per-token prefill, host sampling loop, terminal
  commit handshake — with three insertions:
  1. preprocess (§1.2) → capacity estimate → vision_forward BEFORE prompt
     build (mirrors generate_vl ordering);
  2. prompt built by rendering the normal jinja ChatML with user content
     `"<image>" + question`, then expanding the single 124907 id in place
     into the §1.5 marker structure (marker ids resolved from the
     tokenizer; missing markers fail closed);
  3. prefill walks the expanded sequence; positions holding 124907 are fed
     via `prefill_embed_step` with successive projected-token rows, all
     other positions via the ordinary decode path.
  Spec-decode/DFlash/batch routes untouched (VL requests never enter them;
  matches qwen35-vl today).
- Dispatch fixes riding along (root cause found during this work): the
  daemon-side stream contract requires the FIRST event on a request stream
  to be `gen_start`; `generate_vl` / `generate_vl_dots_ocr` never emitted
  one — the exact failure signature recorded 2026-08-27 ("vision forward
  completes but no response bytes", wedged slot until restart). This spec
  adds the opener to those two bodies as well so the sibling containers
  stop stalling over HTTP; qwen35-side revalidation belongs to
  `feat/qwen35-vl`.

### 2.4 Daemon

`main.rs` VL gate/dispatch: `has_vl` becomes
`m.has_vision_encoder()`; `match vision_route` gains
`VisionRoute::Lfm2Vl => hipfire_generate::vision::generate_lfm2_vl(...)`.
The existing single-slot reset-on-VL-dispatch guard applies unchanged to
arch 11 (it resets conv-state + KV through the bundle reset already wired
for the text path).

## 3. Validation plan (route: model-arch forward bring-up, Tier P shape)

No GPU oracle exists on this box for LFM2.5-VL (no ROCm host python), so
parity takes the same tier the 2026-08-26/27 sessions used — structural +
behavioral smoke, explicitly NOT a VALIDATION.md claim route:

1. CPU unit tests green (§2.1 list) — `cargo test -p hipfire-arch-lfm2-vl`.
2. Load smoke on gfx1101 (container-isolated): vision tensors upload,
   `has_vision_encoder()` true, text-only answer still correct.
3. Image smoke, greedy: fixed fixtures reused from
   `.codeinsight+research/vl-validation-2026-08-27/` (doge_napping.png —
   six meme captions must be read including the disputed
   "Pointy teeths wow"; scene_2.jpg luxury-brand OCR). These captions are
   content-level ground truth independent of which carrier reads them.
   Eyeball rule applies (AGENTS.md §0: read the decoded text).
4. Math probe ("17×23") through the SAME image-free-and-image-bearing
   prompts used 2026-08-27 for lineage comparability.
5. HTTP serve smoke on the swapped container: OpenAI-style chat completions
   with `image_url` data URI on :6900 — stream opens with `gen_start`,
   tokens flow, slot releases after disconnect mid-turn (regression probe
   against finding (c) of the 2026-08-27 ledger).
6. Text-only regression: greedy 17×23 → "391" must survive byte-identical.

Evidence (binary md5s, container cmd, transcripts) lands in a fresh dated
folder under `.codeinsight+research/`; perf numbers are NOT claimed from
any of this.

What would justify future VALIDATION.md rows (recorded, not claimed here):
a dumped HF-reference stage-parity fixture for the tower (Gap-3 tooling
exists in qwen35-vl behind `HIPFIRE_VL_DUMP_DIR`), and an image-bearing
battery run per [`qwen35-vl-mq4v2-spec.md`] §5.

## 4. Risks / decisions

- R1 patch-embed serialization ambiguity (Linear vs Conv kernel in the
  published safetensors): resolved at load time by shape inspection; both
  fold to the same `(dy,dx,c)` contraction. If neither shape appears, load
  fails loudly — never silently transposed.
- R2 exact-erf GELU placement: host pass between projector linears chosen
  over a new HIP kernel (kernel-family risk ≫ 23 MB transfer cost).
  Measurable drift only if HF's ACT2FN["gelu"] were tanh-based, which the
  pinned source disproves.
- R3 pos-embed antialias fidelity (§2.1 deviation): bounded by qwen35's
  measured filter sensitivity; if caption smoketests degrade, the upgrade
  path is a torchvision-exact area/triangle pyramid, isolated in one pure
  function.
- R4 single-slot HTTP robustness: gen-start opener fixes the observed
  stall class; disconnect-reaping for the single-slot server remains an
  open engine issue (not introduced by VL) and is tracked separately.
- R5 memory: tower+projector F16 ≈ 450 M params ≈ 0.9 GB device + patch
  scratch ≤ few MB at 1024-patch grids — fits the 12.9 GB UMA box next to
  the 2.3 GB artifact and KV budget used on 2026-08-27.

## 5. Files touched

- NEW `crates/hipfire-arch-lfm2-vl/` (lib, config, image, tower+forward,
  tests) — workspace member registration in root `Cargo.toml`.
- `crates/hipfire-arch-lfm2moe/{Cargo.toml,src/spec_impl.rs,
  src/forward.rs,src/arch_model.rs}` — dep + bundle fields +
  `prefill_embed_step` + free_gpu.
- `crates/hipfire-loader/src/{lib.rs,carriers.rs}` — route variant,
  `has_vision_encoder()`, VL detect/load/reclaim, caps flip.
- `crates/hipfire-generate/src/{Cargo.toml,vision.rs}` — cargo dep,
  `generate_lfm2_vl`, gen-start openers for the two existing VL bodies.
- `crates/hipfire-daemon/src/main.rs` — gate + dispatch arm.
- Docs: this file; parent spec as-built blockquote;
  `docs/architecture-ids.md` row-11/dir-map VL note.

— end of record
