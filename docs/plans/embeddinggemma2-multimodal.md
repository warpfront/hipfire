# EmbeddingGemma-2 multimodal bring-up — design contract

Ground truth: transformers 5.19.0 `models/embedding_gemma2/*` (towers = `models/gemma4/*`,
same `gemma4_vision`/`gemma4_audio` archs) + the cached checkpoint
`google/embeddinggemma-2` (config, `processor_config.json`, safetensors header).
The trunk forward, pooling, L2 and MRL truncation do NOT change: multimodality is
input assembly only.

## Checkpoint facts (verified from the safetensors header)

- `vision_tower.patch_embedder.input_proj.weight [768,768]` (no bias),
  `patch_embedder.position_embedding_table [2,10240,768]`; 16 encoder layers
  (sandwich norms, q/k_norm [64], qkv/o `[768,768]`, mlp 3072, all `.linear.weight`),
  **no final tower norm**; pooling_kernel_size 3; ≤280 soft tokens/image.
- `audio_tower`: conv subsample [128,1,3,3]/[32,128,3,3]+norms, `input_proj_linear
  [1024,1024]`; 12 conformer layers (macaron FFN 4096 with clipped-linears min/max
  scalars, lconv1d depthwise 5 + linear_start [2048,1024] + linear_end [1024,1024],
  attention per_dim_scale [128], relative_k_proj [1024,1024], chunked attention
  chunk 12 / left 13 / right 0, logit cap 50, residual_weight 0.5); `output_proj
  [1536,1024]+bias`. Audio soft tokens are UNCAPPED: `audio_ms_per_token: 40` and
  processor `audio_seq_length: 280` describe the 11.2 s training window but the
  float32 reference emits M = 375 for a 15 s clip — no waveform truncation,
  no M clamp.
- Embedders `embed_vision.embedding_projection [512,768]`,
  `embed_audio.embedding_projection [512,1536]` — **plus a scale-less pre-projection
  RMSNorm** (no weight tensor; eps = tower `rms_norm_eps`).
- Composition (modeling_embedding_gemma2.py:744 `forward`): placeholder ids
  (258880 image / 258884 video / 258881 audio) are embedded as pad then
  **masked_scatter**-overwritten with projected tower rows; sequence positions and
  RoPE positions are ordinary; invalid audio frames are dropped before scatter, so
  placeholder count == valid frame count. Mean pool covers ALL rows (1_Pooling
  `include_prompt: true`).
- Processor framing: the TEXT carries the `<|image|>` marker (id 258880) once
  per image — the processor validates the count and refuses a mismatch
  (`validate_inputs`); each marker expands to `boi + 258880×N + eoi`
  (`replace_image_token`), N = `patch_h·patch_w // 9` ≤ 280 dynamic (1024² →
  256); audio: `boa + audio_token×M + eoa`, M = valid subsampled frames
  (frontend `pad_to_multiple_of=128`, two stride-2 subsamples), UNCAPPED —
  verified M values: 0.1 s → 3, 0.5 s → 13, 1.0 s → 25, 2.0 s → 50,
  11.2 s → 280, 15 s → 375. The only invariant is `M == audio_tower mask sum`
  (the HF assert); never truncate the waveform or clamp the placeholder
  count. Audio clip scalars are real
  calibrated per-layer values on this checkpoint — `use_clipped_linears` must
  actually clamp. Wire convenience mirrors ST: an image content part whose
  text lacks the marker appends the marker at the part position; inline
  markers win.

## A. Sidecar (`.vl` sibling, arch 18)

One sidecar carries all tower groups (llama.cpp mmproj model): `vision_tower.*` (210),
`embed_vision.embedding_projection.weight`, `audio_tower.*` (751),
`embed_audio.embedding_projection.weight`. HFQM package stamped arch_id 18; discovery
reuses the existing chain (`params.vision` → `HIPFIRE_VISION_SIDECAR` → `.vl` sibling).

Dtype rule (qwen35 sidecar convention): 2-D matrices and conv kernels F16; every 1-D
vector (norms, biases, min/max clip scalars, per_dim_scale, layer_scalar) F32;
`position_embedding_table` F32 (lookup table; exactness at gather).

Quantizer (arch 18): write the sidecar always (it is part of the model), keep the
trunk skip. `vision_mode off` / absent sidecar = text-only loads exactly as today.

## B. Arch crate

- `EmbedGemmaBundle.vision: Option<VisionTower>` (Phase 2 adds `.audio`).
  Loaded when the sidecar resolves; `None` ⇒ image inputs refused with the existing
  pointing error text.
- New `image.rs`: `Gemma4ImageProcessor` port (decode → resize → patches →
  `(patches, gh, gw, n_soft_tokens)`). Decode with the same image crate the VL arches
  use; decompression-bomb guard per lfm2-vl.
- New `vision.rs`: tower forward (f32 activations, f16 WMMA GEMMs via the `Enc`
  dispatch) ending in embedder pre-norm + projection → `[N,512]` rows ready to splice.
- `AssembledInput` gains `media_spans: Vec<MediaSpan { start_row, len, kind }>`;
  `assemble_input` expands each bare `<|image|>` id (258880) in the token
  stream into `[255999, 258880×N, 258882]` (marker-count mismatch = pointing
  error, HF-parity). `ids.len()` counts the expansion — row budgets and the
  8192 cap see the expanded length.
- `EncodeInput` gains the per-span projected rows; the gather step writes them over
  the placeholder rows (positions/RoPE untouched — HF masked_scatter semantics).

## C. Request surface

- `EmbedRequest.image: Option<Vec<u8>>` → `images: Vec<Vec<u8>>` (Phase 2 adds
  `audios`); one decode point per route (base64 → bytes at the boundary).
- Wire (llama.cpp #29556 shape): `/v1/embeddings` accepts `input: string | string[] |
  [{content: [{type:"text"|"image_url"|"input_audio", ...}]}]`; content arrays are
  one input each; legacy string forms keep working; malformed shapes rejected with a
  migration-style pointing error. JSONL embed verb gains the same fields; CLI
  `hipfire embed --image <path>` (repeatable, appended after text).
- Slot engine: tower forward per image on the engine's GPU thread before the batched
  text encode; no cross-request caching of tower outputs (embeddings stay stateless).
- Caps: carrier declares `supports_images: true`; per-instance gate is
  `bundle.vision.is_some()`. No new `VisionRoute` variant (eg2 is embed-only; the
  generate-side VL routes do not apply).

## D. Oracles (acceptance)

- Reference generator v2: image cases (sizes across the ≤280 range, non-multiple
  pooling grids, tiny, cap-hitting) and audio cases (durations spanning 3 → 375
  soft tokens) through float32 CPU transformers 5.19.0 in `.venv`; ids asserted bit-exact,
  embeddings within the text legs' rel-L2 bands.
- Parity/multislot/quant scripts: vision (then audio) legs; multislot bit-identity
  with images in the batch; tower tensors in the quant bands.
- Gates: `scripts/pre-commit-ci.sh origin/beta` green; ratchet raise for daemon
  growth is expected and declared; crate maps regenerated.

## Phase order

Phase 1 vision (A+B+C+D vision legs) → Phase 2 audio (caps bit, `audio.rs`, wire
`input_audio`) → Phase 3 video → Phase 4 productization (registry sidecar
pairing, docs, VALIDATION route, perf checkpoint).

### Phase 4 status

Phases 1–3 are landed and verified: the parity oracle passes on all 21
reference cases plus the refusal contract, the multislot route is bit-identical
with mixed media in one forward, and `pre-commit-ci.sh` is green. Docs carry the
formula-level ground truth ([`embeddinggemma2-tower-spec.md`](embeddinggemma2-tower-spec.md))
with HF file:line references.

- **Registry entry: landed as a local-only row.** Publishing was never a
  prerequisite: `repo: ""` is the registry's local-only shape (`pull` refuses
  with "place \<file\> in \<models dir\>" instead of 404ing), which is how the
  other local-only rows ship (the Flash-Next trunk tier is no longer a row: it
  is the default overlay on the stock tags). `embeddinggemma-2:q8f16` (default,
  also the `embeddinggemma-2` alias) and `embeddinggemma-2:mq4v2` declare their
  `<trunk>.vl` pack in the `vision` slot with the real digests, and
  `scripts/registry_gen.py` gained the arch-18 family mapping. Verified end to
  end from `hipfire quantize <snapshot> --format q8f16 --install --register
  embeddinggemma-2:q8f16` → `hipfire embed embeddinggemma-2:q8f16 --image/--audio/--video`
  (all modalities within the tier's band). Publishing on HuggingFace would only
  change `pull` from "place the file yourself" to "download it".
- **Perf checkpoint: deliberately deferred.** No throughput or latency claim is
  made for these legs (the changelog's entry states parity, exactness and the
  quant bands only), and an append-only checkpoint with nothing to back would
  be fixture-bound noise. If a media-leg perf claim is ever made, it owes the
  `docs/perf-checkpoints/README.md` protocol: a committed fixture per modality,
  ≥3 fresh processes per arm, prompt/precision md5s, and the `check_embeddinggemma2_*.py`
  oracle re-run alongside.

### Phase 3 video — grounded decision: dedicated path, a frame list on the wire

The video leg cannot reuse the image token: `replace_video_token`
(`models/embedding_gemma2/processing_embedding_gemma2.py:244-263`) emits — with
`add_timestamps=False`, which this checkpoint's video processor declares — the
string `"".join([boi + video_token×S + eoi] * num_frames)`: **one
`boi + video_token(258884)×S + eoi` block per frame, concatenated with no
timestamps and no separators**. `S = num_soft_tokens_per_video[video]` is the
**per-FRAME** soft-token count, `patches.shape[1] // pooling_kernel_size²`
computed BEFORE the pad-to-`max_patches`
(`video_processing_embedding_gemma2.py:287`). The TEXT carries one `<|video|>`
per VIDEO (not per frame) and `validate_inputs`
(`processing_embedding_gemma2.py:216-222`) refuses `sum(markers) != len(videos)`;
the model scatters the projected rows over the 258884 positions
(`modeling_embedding_gemma2.py:801-818`, `masked_scatter`). Image ids (258880)
and the image cap would be wrong.

Video geometry is `processor_config.json:video_processor`
(`EmbeddingGemma2VideoProcessor`, `video_processing_embedding_gemma2.py:173-195`),
NOT `vision_config`: `max_soft_tokens 140` (its class default is 70 — the
CHECKPOINT's sidecar value wins; `max_patches = 140·3² = 1260`),
`patch_size 16`, `pooling_kernel_size 3`, `max_frames 32`, `fps 1`,
`overflow_strategy "uniform"`, `do_sample_frames true`, `add_timestamps false`,
`resample` bicubic, `rescale_factor 1/255`, `do_normalize false` with
mean 0 / std 1 (identity either way), `num_frames` absent (passing it raises in
HF's sampler).

**Wire representation.** `input.videos_base64` is a list of videos, each a list
of base64 **frames** (PNG/JPEG, one entry per frame, in display order, ≥1
frame); the caller hands a decoded clip at its native rate and **hipfire applies
the sampler itself** — `EmbeddingGemma2VideoProcessor.sample_frames`
(`video_processing_embedding_gemma2.py:316-360`): a decoded frame list carries
no capture rate, so the `fps`-based pass is skipped exactly as HF skips it
(`metadata.fps is None` → `fps = None`, warning), `indices = arange(total)`, and
then the `max_frames` budget: ≤ 32 frames are all kept, more are reduced with
`np.linspace(0, total-1, 32, dtype=int)` (the `"uniform"` strategy). A caller
therefore does not have to select frames to match HF — and the reference
generator hands HF the SAME full frame list, so both sides run the same rule on
the same pixels. `overflow_strategy "truncate"` and `do_sample_frames false` are
implemented too (keep the first `max_frames` / keep everything), since both are
exact and cheap; `add_timestamps true` is REFUSED (the `mm:ss ` prefixes are
text this leg does not synthesize), as is a `num_frames` key HF itself raises
on.

Frames of one clip must share a size: HF stacks them into one
`(T, H, W, C)` array (`video_utils.convert_pil_frames_to_video`), so ragged
frames cannot be represented — refusing them by name is HF-faithful. Bytes that
are not a PNG/JPEG container are refused with the `ffmpeg` frame-extraction to
run, mirroring the WAV-only rule the audio leg uses for containers.

Each frame then goes through the SAME image pipeline the vision leg already
uses (one aspect-preserving resize per video, since every frame has the same
shape; the uint8 bicubic resize transcribed in `image.rs`), with the video
processor's own `max_soft_tokens = 140` patch budget, and the tower runs each
frame's real patches as an independent sequence — HF's tower batches frames as
`(total_frames, max_patches, patch_pixels)` with per-frame bidirectional
attention and padding keys masked (`modeling_gemma4.py:2003-2062`,
`attention_mask = ~padding_positions`), so no frame can read another. hipfire
reuses `VisionTower::encode_image` per frame (the whole `.vl` vision tower: no
new weights, no new tower code) and emits **one `MediaSpan` per frame** — each
frame's block is spliced exactly like an image's, which is also HF's layout.

Request surfaces: JSONL/daemon `videos_base64`, CLI
`--video FRAME[,FRAME…]` (one clip per flag, repeatable), HTTP
`{"type":"input_video","input_video":{"data":[<base64 frame>,…]}}` lowered to
`videos_base64`, always beside the existing `<|video|>` marker rule (one marker
per video; the count is validated with HF's semantics).
