# hipfire-arch-embeddinggemma2 — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 18); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

EmbeddingGemma 2 (`google/embeddinggemma-2`): a 24-layer **bidirectional** Gemma-4-family single-vector embedding encoder — sliding/full hybrid attention with per-layer geometry overrides, half-split RoPE, projection-only PLE, mean-pool + L2 `embedding_projection` head with Matryoshka truncation. **Not a chat model**: no LM head, served only through the non-chat `embed` verb (`hipfire embed`, `POST /v1/embeddings`), sequential or multi-slot. The checkpoint's Gemma-4 **vision and audio towers** both ride the arch-18 `.vl` sidecar and are spliced as projected rows over their placeholder runs by [`src/carrier.rs`](src/carrier.rs); a bundle without a sidecar is text-only and refuses each modality by name ([`src/carrier.rs`](src/carrier.rs)'s `NO_VISION_TOWER` / `NO_AUDIO_TOWER`). The default tokenizer's greedy SentencePiece path diverges from HF on this vocabulary, so arch 18 opts into the exact SP-BPE merge path (`Tokenizer::enable_sp_bpe_merges`). See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

<!-- hand-written: what not to do here; traps a reader will hit. -->

- **The image resize is torchvision's uint8 bicubic kernel, transcribed — not `image`'s `CatmullRom`, and not Pillow's resampler either.** HF's processor resizes through the torchvision backend (`Gemma4ImageProcessor` → `tvF.resize(..., BICUBIC, antialias=True)`), i.e. `torch.nn.functional.interpolate` on a **uint8** tensor, which ATen serves from its own separable fixed-point kernel (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`: double-precision Catmull-Rom weights, normalized, quantized to **int16** at a dynamic precision, then an integer multiply-accumulate with a `1 << (p-1)` bias — width pass first into a uint8 intermediate). `image`'s `CatmullRom` runs the passes the other way with an f32 intermediate (up to 25/255 off on noise), and Pillow's resampler uses 22-bit int32 weights (1 LSB off on 6% of a smooth gradient's pixels). `crate::image`'s `aten_bicubic_resize` is a line-for-line transcription; its unit test pins torch's own bytes and asserts the `image`-crate result *differs*, so do not "simplify" it back.
- **The vision tower's RoPE head-dim permutation is load-bearing.** HF's table is `cat([h, h, w, w])` and `apply_multidimensional_rope` rotates the two *halves* with `rotate_half` (pairs `(j, j + head_dim/4)`); the `rope_2d_halfsplit_f32` kernel pairs `(d, d + head_dim/2)`. `vision::axial_perm` is the map between them — a permutation of Q/K rows can never show up as a shape error, only as a wrong embedding, and `vision::rope_tables_plus_perm_reproduce_the_hf_rotation` is the differential test that pins it against HF's own rotation.
- **The audio token count `M` is the mask-subsample count, not `nf/4` and not `ceil(ms/40)`.** The frontend pads the waveform to `pad_to_multiple_of = 128` samples FIRST (0.5 s → 8 064, which is the difference between 50 frames / M 13 and 49 / 12), then frame validity rides through two `mask[:, ::2]` halvings. `audio::audio_token_count` is the one implementation; the unit test pins the reference's 0.1/0.5/1/2/11.2/15/30 s values. It is UNCAPPED (no `audio_seq_length` clamp); the only bound is the kind's sequence limit, enforced by `carrier::assemble_input`'s room accounting.
- **The audio subsample conv stack and the blocked attention run on the HOST** (exact f32), the GEMMs on the GPU. Both are ~1 % / 0.2 % of the layer's MACs and the attention is pure index arithmetic; if profiling ever says otherwise, the device kernel has to reproduce the chunked window, the `rel_shift` skew and the softcap exactly (`audio_tower::attention`'s unit tests are the contract).
- **The audio attention's mask is INVERTED — reproduce the inversion, do not "fix" it.** The reference defines the mask as `valid_key ∧ 0 ≤ q − kv < attention_context_left − 1` (`create_bidirectional_mask` + `sliding_window_mask_function((12, 0))`), but applies it with the Bool-mask idiom `masked_fill(attention_mask.logical_not(), −1e9)` while the generator forces `attn_implementation="eager"` — and HF's `eager_mask` (`masking_utils.py` L541) turns that boolean mask into an ADDITIVE one first. The slots that survive are therefore the COMPLEMENT: `in_mask && !(valid_key && window)`, with NO query-validity term (a dumped 4-D row is `valid_key ∧ window` even for the padded-tail rows). Consequences that look like bugs and are not: a query reads its FUTURE keys, `R[0]` is the only relative-position row that can reach a survivor, and a first-block query can have no survivor at all (the softmax is then uniform over the block context, whose out-of-sequence values are zeros). The `sdpa` interface does NOT invert and disagrees with the reference by 3e-2…1.4e-1, so an implementation that matches a standalone tower run is wrong.
- **`examples/audio_lab.rs` is the audio parity instrument** (`--features lab`): frontend + tower for one WAV straight from the checkpoint (safetensors dir or `.vl` pack, no daemon, no text tower), dumping the log-mel and the `[M, text_hidden]` rows; `--layers N` truncates the tower for a layer-by-layer bisect against HF's `Gemma4AudioModel` — which is only the reference when built with `attn_implementation="eager"`.
- **The tower runs the whole subsampled sequence and returns its first `M` rows.** `SubSample::forward` hands back every `t1` token (not just the `M` valid ones) and `attention` takes both lengths: `steps` is the sequence the reference's block-context mask covers, `rows` its valid prefix. Because the mask is inverted a valid row KEYS INTO THE PADDED TAIL and a tail row's own output becomes the next layer's key/value row, so truncating to `M` before the forward drops real keys — ~15 % rel-L2 on the clips where `M < steps` (roughly one clip length in four; `examples/audio_lab.rs` repros it with 1 409 samples). `attention` fills the rest with `attention_invalid_logits_value` (the config's value, not a constant).
- **The clamp bounds are real checkpoint scalars.** `use_clipped_linears` is true here and all 10 clippable linears per layer read four 0-d bounds; the `gradient_clipping` clamps (1e10 in f32) are kept for reference parity though they are inert. A sidecar missing an audio group while the config declares an `audio_config` is a LOAD ERROR, never a silent text-only load.

- **The video leg is the IMAGE tower with the VIDEO processor's budget — and one `MediaSpan` per FRAME.** HF's `replace_video_token` (with `add_timestamps=false`) emits one `boi + video_token(258884)×S + eoi` block per frame, concatenated with no separators, so a clip's frames are NOT one contiguous placeholder run: `carrier::assemble_input` pushes one span per frame (each frame splices exactly like an image), and `AssembledInput::media` holds the images first (`n_images`) then every clip's frames, because image and video spans interleave in `ids` and each needs its own cursor. Frames are resized against `video_frontend.max_soft_tokens` (the checkpoint's 140 → 1 260 patches), NOT `vision_config`'s 280 (2 520) — mixing them up shifts every per-frame soft count and the whole prompt. The sampler is HF's (`video::sample_indices`), and it runs over the frames the CALLER supplied: a decoded frame list has no capture rate, so the `fps` pass is skipped (exactly HF's `metadata.fps is None` branch) and only `max_frames`+`overflow_strategy` decide. `patch_size`/`pooling_kernel_size` must still agree with `vision_config` (the patch embedder is built for them) — `frame_vision()` refuses otherwise.

## Crate map

<!-- crate-map:generated:begin -->

_Generated by `scripts/check-crate-maps.py` from the tree — do not edit inside the markers._

### Modules and exported symbols

- [`src/audio.rs`](src/audio.rs)
  - `audio_token_count`
  - `AudioInput`
  - `AudioPlan`
  - `decode_and_plan`
  - `decode_wav`
  - `frames`
  - `mels`
  - `n_tokens`
  - `plan`
  - `process_audio`
  - `ProcessedAudio`
- [`src/audio_tower.rs`](src/audio_tower.rs)
  - `AudioTower`
  - `encode_audio`
  - `free_gpu`
  - `load_audio_tower`
  - `sidecar_has_tower`
  - `synthetic`
- [`src/bundle.rs`](src/bundle.rs)
  - `EmbedGemmaBundle`
  - `finalize_embedding`
  - `kind_prompt`
  - `MRL_DIMS`
- [`src/carrier.rs`](src/carrier.rs)
  - `assemble_input`
  - `AssembledInput`
  - `encode_assembled`
  - `encode_media`
  - `load_embed_gemma_bundle`
  - `load_embed_gemma_from_source`
  - `load_embed_gemma_with_vision`
  - `media_rows`
  - `MediaKind`
  - `MediaSpan`
  - `NO_AUDIO_TOWER`
  - `NO_VIDEO_TOWER`
  - `NO_VISION_TOWER`
  - `preflight_slot`
  - `resolve_slot_vision_path`
  - `text`
- [`src/config.rs`](src/config.rs)
  - `apply_sbert_config`
  - `apply_st_config`
  - `attn`
  - `AUDIO_PAD_TO_MULTIPLE_OF`
  - `AudioConfig`
  - `AudioFrontendConfig`
  - `config_from_metadata`
  - `context_size`
  - `conv_freq_out`
  - `DEFAULT_DOCUMENT_LENGTH`
  - `DEFAULT_DOCUMENT_PROMPT`
  - `DEFAULT_QUERY_LENGTH`
  - `DEFAULT_QUERY_PROMPT`
  - `embed_scale`
  - `EmbedConfig`
  - `ffn_dim`
  - `fft_bins`
  - `frame_vision`
  - `frames`
  - `from_processor_json`
  - `InputKind`
  - `kind_length`
  - `kind_prompt`
  - `kv_dim`
  - `LayerAttn`
  - `LayerType`
  - `lconv_dim`
  - `max_patches`
  - `OverflowStrategy`
  - `padded_samples`
  - `parse_audio_config`
  - `parse_config`
  - `parse_vision_config`
  - `patch_pixels`
  - `ple_scale`
  - `position_axis_stride`
  - `q_dim`
  - `rope_theta`
  - `side_mult`
  - `subsample_input_dim`
  - `SUPPORTED_SOFT_TOKENS`
  - `valid_frames`
  - `validate`
  - `VideoFrontendConfig`
  - `VisionConfig`
  - `window_bites`
- [`src/forward.rs`](src/forward.rs)
  - `attention_pv_tiles`
  - `attention_qk_tiles`
  - `encode_embeddings`
  - `encode_embeddings_batch`
  - `encode_hidden`
  - `EncodeInput`
  - `EncodeSegment`
  - `MediaRows`
  - `text`
- [`src/image.rs`](src/image.rs)
  - `aspect_ratio_preserving_size`
  - `MAX_DIMENSION_PIXELS`
  - `position`
  - `process_image`
  - `process_rgb8`
  - `ProcessedImage`
- [`src/lib.rs`](src/lib.rs)
  - `ARCH_ID`
  - `audio`
  - `audio_tower`
  - `bundle`
  - `carrier`
  - `config`
  - `forward`
  - `image`
  - `names`
  - `post`
  - `slot_engine`
  - `video`
  - `vision`
  - `weights`
- [`src/names.rs`](src/names.rs)
  - `numel`
  - `read`
  - `read_tower`
  - `release`
  - `release_tower`
  - `required_text_tensors`
  - `tensor_candidates`
  - `TensorBytes`
  - `tower_has`
  - `tower_tensor_candidates`
- [`src/post.rs`](src/post.rs)
  - `l2_normalize`
  - `l2_normalize_rows`
  - `mean_pool_rows`
  - `mrl_truncate`
- [`src/slot_engine.rs`](src/slot_engine.rs)
  - `EmbedEngineConfig`
  - `EmbedSlotEngine`
  - `from_slot_config`
  - `spawn`
- [`src/video.rs`](src/video.rs)
  - `process_video`
  - `sample_indices`
- [`src/vision.rs`](src/vision.rs)
  - `encode_image`
  - `encode_images`
  - `free_gpu`
  - `load_vision_tower`
  - `sidecar_has_tower`
  - `synthetic`
  - `VisionLayerWeights`
  - `VisionTower`
- [`src/weights.rs`](src/weights.rs)
  - `EmbedWeights`
  - `free_gpu`
  - `gather_hidden`
  - `layer_type_summary`
  - `LayerWeights`
  - `load_weights`
  - `refuse_unsupported_quant_types`
  - `source_has_ple`

### Dependencies (from `Cargo.toml`)

#### path

- `hip-bridge`
- `hipfire-config`
- `hipfire-runtime`
- `rdna-compute`

#### external

- `half`
- `image`
- `serde_json`

#### dev

—

#### build

—

### Reverse dependencies

Workspace crates with a path dependency on this crate:

- `hipfire-generate`
- `hipfire-loader`

<!-- crate-map:generated:end -->
