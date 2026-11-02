# Spec: Qwen3.5-VL — `mq4v2` quantization + vision serving on arch 5 (family spec)

> **Lifecycle: historical (family design and bring-up record).** Retained
> for provenance of the 2026-08 arch-5 VL carrier work. It is **not**
> current procedure, product authority, or acceptance: the §5 validation
> plan and §7 file list are the original plan, "as-built" notes are dated
> snapshots, and PR/issue references (#312, #610) describe state at the
> time. Current behavior is defined by the code
> (`crates/hipfire-arch-qwen35-vl/`, `crates/hipfire-quantize/src/pipeline.rs`,
> `crates/hipfire-quantize/src/model_filter.rs`,
> `crates/hipfire-quantize/src/vision_sidecar.rs`, `crates/hipfire-loader/`
> admission), by [`architecture-ids.md`](architecture-ids.md), by
> [`specs/2026-08-27-qwen35-vl-vision-serve.md`](specs/2026-08-27-qwen35-vl-vision-serve.md),
> and by [`VALIDATION.md`](VALIDATION.md) for VL claim routes (route
> existence is not per-checkpoint evidence). The §4 "shared contract" is
> retained as the plan the carriers were built against; the `.vl` sidecar
> packaging question it leaves open has since gained a sidecar
> implementation in the quantizer (`vision_sidecar.rs`), so do not read §4
> as the current packaging decision.

Status at the time: DRAFT (branch `feat/qwen35-vl`; sibling arch-11 carrier
on `feat/lfm2-vl`. Supersedes the fatter `feat/vl` working copy, archived as
tag `archive/feat-vl`)
Author: engineering
Sibling record: [`lfm2-vl-mq4v2-spec.md`](lfm2-vl-mq4v2-spec.md) — the arch 11 carrier
(LFM2.5-VL). §4 of this document was the shared artifact contract for both.

## 0. Goal and scope

Produce and validate the **canonical VL test vehicle** on the arch 5 carrier:

1. A single `.hfq` of the §1 primary instance with the text trunk in
   **MQ4G256V2** (qt=44, `--format mq4`), the vision tower + spatial merger
   in **F16**, and `has_vision` metadata — no hand-spliced post-pass.
2. Serve it end-to-end with image input (local `run --image` and the HTTP
   `/v1/chat/completions` image path that landed with #312) and validate the
   shared VL artifact contract of §4 against a real artifact.
3. Use it as the fixture model for the VL validation routes (§5) — the
   evidence source for future `docs/VALIDATION.md` VL rows.

**This is a family spec, not a model spec.** Any Qwen3.5-family VL checkpoint
(`model_type: qwen3_5*`) that passes the §1 config audit uses the §2 recipe
unchanged. Fine-tunes of the family ride the same carrier and the same recipe;
they do not get their own spec rows in this document.

**Out of scope** (§6): arch 6 (MoE) VL — qt44 MoE kernels are pending
upstream PR #610; MTP/DFlash × VL; multi-image and multi-turn vision over
HTTP; bf16 vision; per-fine-tune SKU work.

## 1. Family scope, primary instance, config audit

- **Carrier:** `hipfire-arch-qwen35-vl` — production, claims arch_id 5/6 +
  vision tower. This spec exercises arch 5 (dense) only.
- **There is no first-party Qwen3.5-VL.** Qwen3.5 is text-only (0.8B/2B/4B/
  9B/27B/35B-A3B); the VL line is Qwen3-VL — a different architecture
  generation this carrier does not claim. Every `model_type: qwen3_5` VL
  checkpoint is a **family fine-tune** (vision tower added onto a Qwen3.5
  base by a third party). Instances are therefore chosen from available
  fine-tunes, audited per the table below.
- **Family vision-sourcing convention: the tower ships separately.**
  Qwen3.5-family VL sources distribute the vision encoder as a **separate
  module** (`mmproj`-style GGUF — e.g. `mmproj-Ornith-1.5-9B-F16.gguf`),
  not inside the text trunk. The hipfire build step therefore takes *two*
  inputs — text checkpoint + separate vision module — and **embeds the vision
  tensors F16 into the single `.hfq`/`.mq4` artifact**. Both as-built local
  artifacts follow this (censuses below); the separate-module source and
  embedded artifact are complementary, not competing layouts.
- **Primary instance (as-built, verified 2026-08-26):**
  the local `ornith-1.5-9b.mq4` build — arch_id 5, 760 tensors, census **486 F16
  / 249 MQ4G256V2 / 25 Q8F16**, vision embedded; sha256
  `dde2d142…99cb` **matched the appliance pin** in the local registry
  cache at the time. Local container images existed: `hipfire-ornith:mq4v2`
  (model baked, 28.3 GB) and `local/hipfire-ornith:dynamic-pool` (model
  bind-mounted). Validation ran through this appliance — no new pull, no new
  quant. The engine's vision path was previously validated on a 9B-VL family
  build — `benchmarks/vision/comparison-2026-05-23.md` remains the
  prompt/fixture baseline for §5.
- **As-built tier pattern (record for §2):** trunk 2D mats → qt44; untied
  `lm_head` → **qt44** (540 MB at [248320, 4096]); `embed_tokens` → Q8-class
  (qt=3); norms/biases/A_log + vision tower + projector → F16. A second
  local artifact (a 4B family fine-tune, 450 F16 / 248 qt44 / 25 Q8F16,
  tied embeddings at Q8-class) confirms the pattern; tied models have no
  lm_head tensor, so the Q8 embed *is* the head.

**Per-checkpoint config audit** (run before quantizing any family member;
fine-tunes drift config — do not assume the base model's values):

| check | where | against |
|---|---|---|
| `model_type` ∈ `qwen3_5` family → arch 5 auto-mapping | config.json | `arch_mapping.rs` (`qwen3_5` → 5) |
| `tie_word_embeddings` | text config | loader head path: tied = no `lm_head` tensor; confirm the arch-5 load aliases embed→head (explicit tied handling in-tree today is lfm2/dots-ocr/glimmer) |
| vision tower dims (depth / hidden / out_hidden / patch / spatial_merge / num_position_embeddings) | vision config | read by `vision_config_from_hfq` — parameterized, no hardcodes expected; verify no 1152/27 assumptions leak |
| `deepstack_visual_indexes` | vision config | empty = simple config; non-empty needs the deepstack path |
| `mrope_section` + `mrope_interleaved` | rope_parameters | crate `DEFAULT_MROPE_SECTION` `[11,11,10]` + interleaved handling — mismatch degrades quietly (Ornith design-doc warning) |
| preprocessor min/max pixels | preprocessor_config.json | `VISION_MAX_PIXELS = 2_000_000` default (`image.rs`), `HIPFIRE_VL_MAX_PIXELS` override, ~4.1 MP LDS ceiling |

## 2. Quantize recipe

```
hipfire quantize <Qwen3.5-9B-VL safetensors> --format mq4 --include-vision
```

- `--format mq4` → qt=44 (canonical `mq4v2`); arch 5 resolves from
  `model_type` — no `--arch-id` needed.
- Emission (existing arch-5 pipeline behavior, proven by the Ornith SKU):
  2D text projections/FFN → MQ4G256V2; `embed_tokens` / `lm_head` → the
  existing default tier (Q8); norms, biases, vision tower (`model.visual.*`),
  spatial merger → F16. Single artifact — **supersedes the hand-spliced
  `qwen3.5-9b.mq4-q8head-vision-f16-spliced.hfq`** (v1 MQ4, AWQ+GPTQ body).
- **Post-quantize audit (mandatory):** tensor census + param accounting —
  `quantized + f16 == total` params, zero missing norms/biases. This is the
  #610 defect-1 lesson: a silently-dropped F16 tail produces plausible sizes
  and an unloadable model. A structural qt-map dump (per-tensor QuantType
  counts) replaces any `strings | grep` acceptance check.

## 3. Runtime (verify-only — no new carrier work expected)

The arch 5 VL path is production. Verification items, in order:

1. Load + §2 census audit on the target GPU (this box: gfx1101).
2. The §1 tied/untied head path for this checkpoint's config.
3. `vision_config_from_hfq` round-trip against the 9B tower config.
4. `run --image` smoke on one fixture, then the §5 battery.

As-built note (2026-08-27): items ran green through `run --image`, but the
HTTP serve arm of item 4 failed live — VL entry points predated the stream
contract and delivered zero bytes over `/v1/chat/completions`; the same
probe found a permanent slot wedge after mid-encode client disconnects. Both
fixed and verified; record:
[`docs/specs/2026-08-27-qwen35-vl-vision-serve.md`](specs/2026-08-27-qwen35-vl-vision-serve.md)
(engine touch below reflects it).
5. Known engine debt this vehicle exercises (from
   `docs/plans/vision-pipeline-cleanup-326.md`): multi-turn image-state reset
   defense, per-image GPU alloc count, and the missing VL output coherence
   gate — §5.4 is that gate's first instance.

## 4. Shared VL artifact contract (arch 5 and arch 11)

Single source of truth for both VL specs; the arch 11 spec's §4 defers to
this section.

- **Metadata:** HFQ `metadata_json` gains `has_vision: bool` + a versioned
  `vision_config` blob: tower params of §1, projector/merger shape, and the
  model's pixel budget (min/max pixels). The loader instantiates the vision
  tower + merger when `has_vision` is set. Pixel budgets belong here, not in
  global constants — `VISION_MAX_PIXELS` stays as the engine ceiling.
  **Status: emitted as of 2026-08-26; reshaped on `feat/qwen35-vl`
  (2026-08-27)** — the quantizer sets `has_vision: true` when a vision-module
  tensor is emitted, and merges the source `processor_config.json`
  pixel-budget keys additively into the already-carried `config.vision_config`
  object (the location `vision_config_from_hfq` reads); no second top-level
  schema under the same name. An earlier pre-split build wrote a flat
  top-level `vision_config`, and older lineage artifacts (e.g.
  `ornith-1.5-9b.mq4`, built by the ornith branch) predate the emission:
  vision detection there keys off `architecture` + the embedded config
  blob. Loaders must not require either key on lineage artifacts.
- **Vision dtype policy:** vision tower + projector/merger tensors are stored
  F16 (bf16 sources convert exactly in mantissa within fp16 range; overflow
  must fail closed, not saturate). mq4v2 vision tensors are
  **format-illegal**, not merely undesired: SigLIP-family dims (1152, 4304)
  violate the `K % 256 == 0` constraint. Do not "extend" this without a
  padding rule.
- **Layout:** vision tensors are embedded in the `.hfq`/`.mq4` (what
  `load_vision_weights(hfq, …)` consumes) — this is the **as-built appliance
  convention**, now with two local artifact precedents. The qwen-family
  *source* convention is a separate vision module (mmproj-style), merged at
  build time; the standalone `.vl` sidecar distribution variant (PR #610) is
  the same idea at publish time and remains an **open packaging decision**:
  if kept, it must be attached/spliced at pull-or-load time and the registry
  needs a structured `vl` field — until decided, ship embedded-only artifacts.
- **Naming / ladder:** tags follow the ladder rules (`mq4` = qt44 alias).
  VL `min_vram_gb` must include the F16 tower + image-peak activations. The
  ladder's bpw honesty rule needs a VL amendment (text-tower bpw + declared
  F16 vision adder) — tracked as a `ladder.md` follow-up, not blocking here.
- **embed/lm_head tier is a per-family recipe decision, not a format
  constraint:** arch 5 keeps the Q8 default (a small VL model's tied head is
  its output layer — the OvisOCR2 sensitivity lesson); the arch 11 spec
  elects mq4v2 embed explicitly. Either is contract-legal.

## 5. Validation plan (this artifact is the fixture)

1. **Load/census** (§3.1) — gfx1101, structural qt-map recorded.
2. **Text parity:** qt44 parity is covered by the landed `mq4v2_*` examples
   on master; the Q8 head path is unchanged production. Nothing new owed
   beyond running them once at this shape.
3. **Vision parity:** F16 tower forward vs an HF `transformers` reference
   dump for this checkpoint (`benchmarks/vision/dump_hf_reference.py`
   precedent; dump-and-diff, pixel inputs pinned by hash).
4. **Image coherence battery:** the 6 committed fixtures in
   `benchmarks/vision/images/`, the two fixed prompts (desc / ocr) from the
   2026-05-23 comparison, greedy, temp 0, `/no_think`. Compare against that
   comparison's F16-spliced baseline and its external-engine arms. **Read the
   outputs** — the eyeball rule; tight variance on image batteries is a
   warning sign, not a pass.
5. **Fixture discipline:** record image SHA-256 + prompt md5 + artifact
   sha256 with every result. A re-encoded image is the multimodal equivalent
   of editing a bench prompt.
6. **Text-only regression:** `hipfire bench` vs the text qwen3.5:9b line to
   confirm the VL-bearing artifact's text path did not regress.
7. On pass, propose the `docs/VALIDATION.md` rows this evidence justifies
   (vision-tower parity route; image-bearing coherence battery route) —
   VL claims stay fail-closed "unknown surface" until those rows land.

   As-built 2026-08-27: both routes landed on `feat/qwen35-vl` in
   [`docs/VALIDATION.md`](VALIDATION.md)'s Claim → route map ("VL
   vision-tower forward numerical parity"; "VL image-bearing serve
   semantics"), after the image-bearing battery ran green over HTTP on the
   NuExtract3 fixture (see
   [`specs/2026-08-27-qwen35-vl-vision-serve.md`](specs/2026-08-27-qwen35-vl-vision-serve.md)).
   Route existence is not per-checkpoint evidence: every VL claim still
   runs under its named route with its own identity hashes.

## 6. Out of scope

- **Arch 6 (MoE) VL**: qt44 grouped/indexed MoE kernels are pending PR #610;
  this spec's vehicle is dense.
- **MTP / DFlash × VL**: short, structured VL outputs; revisit only with a
  measured need.
- **Multi-image / multi-turn vision over HTTP** (phase 2 of
  `docs/plans/completions_vision.md`).
- **bf16 vision** (engine constraint) and **quantized vision towers** (§4).
- **Per-fine-tune SKU specs**: any family fine-tune that passes §1 + §2 is
  servable; its publication is a SKU campaign (registry entry + evidence
  file), not a spec change.

## 7. Files touched (as-built)

- **Engine: generate-layer only** (`crates/hipfire-generate/src/vision.rs`,
  2026-08-27): VL entry points gained the `gen_start` stream opener and
  `generate_vl` gained client-cancel polls emitting the canonical cancelled
  terminal pair; the same-day audit pass routed `generate_vl` decode
  emission through the shared typed contract (EosFilter →
  ThinkOutputRouter → `emit_visible_token`/`emit_reasoning_token`, both
  emission sites + finish flush, honest `started_in_think`) and fixed the
  dots.ocr decode-loop abort terminal — no carrier, kernel, or dispatch
  change; see
  [`docs/specs/2026-08-27-qwen35-vl-vision-serve.md`](specs/2026-08-27-qwen35-vl-vision-serve.md).
- `crates/hipfire-quantize/src/pipeline.rs` — the recipe items the shared
  contract needs: arch-11 `embed_tokens` routes through the mq4 bulk branch
  when an mq4 format requests it (tied head covered for free); the
  `multi_modal_projector` MLP joins the vision group; `has_vision` +
  pixel-budget metadata emission per §4 as-built. `has_vision` latches
  only after every name-based skip gate (include-prefix, gemma4 text-only)
  so the flag always means "vision tensors actually emitted".
- `crates/hipfire-quantize/src/model_filter.rs` — projector joins the
  never-quantize predicate (towers were already there).
- Carrier items for the arch-11 sibling (`arch_mapping.rs` `lfm2_vl` → 11,
  `hipfire-arch-lfm2moe` qt-44 decode + batch allowlists, arch-11 embed
  routing) live on `feat/lfm2-vl`, not here — this branch touches no
  engine/arch crate. See [`lfm2-vl-mq4v2-spec.md`](lfm2-vl-mq4v2-spec.md).
- Registry entry for the published artifact — optional, at publish time.
- Explicitly **not** here: appliance/container images and quantizer error
  measurement — separate concerns from this spec.
