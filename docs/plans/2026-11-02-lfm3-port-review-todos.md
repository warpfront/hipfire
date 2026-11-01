# LFM3 port — pre-PR review follow-ups (actionable task list)

> **Source:** pre-PR review of branch `lfm3-port` (27 commits, base `beta`, HEAD `dd6286e`).
> **Status (2026-11-02):** all 23 tasks resolved across four commits on
> `lfm3-port` — `d1d1fc0` (batch 1: quick wins + omni/streaming contract,
> 14 IDs), `3f1834f` (ROPE-1), `7c95d23` (batch 2: numerics + hardening),
> `e9498cb` (batch 3: cleanup). GPU verify (gfx1151, LFM3-1.7B q8 via
> `serve_harness.py` battery) ran coherent through the new zero-tail RoPE
> kernel; `finish_reason=length` surfaces through the CLI. The merge gate
> (ROPE-1, PROMPT-1, STREAM-1) is satisfied. Each task is self-contained and
> independently pickable.
> **Line numbers** are as of `dd6286e` and will drift — re-grep the named symbol if a
> reference is stale.
> **Convention:** `P0` = must fix before merge (correctness/blocking), `P1` = correctness or
> contract defect (should fix before merge), `P2` = robustness/quality, `P3` = nit.

## Priority summary

| ID | Pri | Title | Primary location(s) |
|----|-----|-------|---------------------|
| ROPE-1 | P0 | Full-attention partial RoPE does not match LFM3 zero-tail scheme | `hipfire-arch-lfm3/src/forward.rs`, `kernels/src/rope_partial_halfsplit.hip` |
| PROMPT-1 | P0 | Corrupt prompt literal `IMDTextWriter` in Omni route | `daemon.rs`, `depth_smoke.rs` |
| STREAM-1 | P0 | Omni streaming loop has no step budget → GPU OOB | `daemon.rs` |
| SPLICE-1 | P1 | Silent vision/audio splice fallback → ungrounded output | `daemon.rs` |
| VL-RESIZE-1 | P1 | LFM3-VL resized with Qwen pixel budget (not LFM3's) | `hipfire-arch-qwen35-vl/src/image.rs`, `daemon.rs` |
| ASR-STR-1 | P1 | Streaming ASR drops lookahead+f tail, no `is_final` | `hipfire-arch-lfm3-audio/src/lfm3_audio.rs` |
| AUDIO-NAME-1 | P1 | Omni audio tower not detected (name mismatch) | `hipfire-loader/src/carriers.rs`, `hipfire-quantize/src/main.rs` |
| ARCHID-1 | P1 | `derive_arch_id` omits `lfm3_omni` | `hipfire-runtime/src/safetensors_source.rs` |
| FINISH-1 | P1 | LFM3 `done` omits `finish_reason` → truncation says "stop" | `daemon.rs` |
| SUPPRESS-1 | P1 | Omni text argmax lacks HF token suppression | `hipfire-arch-lfm3-omni/src/omni.rs` |
| DTYPE-1 | P1 | BF16/F16/HFQ4 dtype handling landmines | `lfm3-omni/src/depth.rs`, `codec.rs` |
| EOS-1 | P2 | Audio-eos action diverges from HF (stop vs continue-to-text) | `hipfire-arch-lfm3-omni/src/omni.rs` |
| EPS-1 | P2 | Audio projector LayerNorm eps `1e-6` should be `1e-5` | `hipfire-arch-lfm3-audio/src/lfm3_audio.rs` |
| REPPEN-1 | P2 | Repetition penalty compounds per duplicate | `hipfire-arch-lfm3-omni/src/depth.rs` |
| LEAK-1 | P2 | `pos_buf` DeviceBuffers never freed (+ codec leak on error paths) | `lfm3.rs`, `lfm3_audio.rs`, `codec.rs`, `daemon.rs` |
| THINK-1 | P2 | `gen_start.started_in_think` computed before framing | `daemon.rs` |
| GRAPH-1 | P2 | New kernels use raw `launch_kernel` (graph-capture hazard) | `attention.rs`, `norm.rs` |
| GRID-1 | P2 | `vit_attention_gqa_f32` gridDim.y/LDS ceiling | `attention.rs`, `kernels/src/vit_attention_gqa_f32.hip` |
| PANIC-1 | P2 | Loader helpers `panic!` on missing tensor | `hipfire-arch-lfm3-vl/src/lfm3_vl.rs` |
| MODE-1 | P2 | `tts_mode` silently defaults to audio; `omni_stream_prefill` default mode | `daemon.rs`, `hipfire-arch-lfm3-omni/src/omni.rs` |
| DOC-1 | P2 | arch-ids doc + registry entry not updated for arch 13 | `docs/architecture-ids.md`, `registry/models.json` |
| TEST-1 | P2 | Weak smoke assertions / no unit tests for new daemon fixes | examples, `daemon.rs` |
| DEAD-1 | P3 | Dead code + leftover diagnostics | multiple |
| NITS-1 | P3 | Misc: empty-prompt underflow, doc drift, merge identity copy, irfft cost | multiple |

---

## P0 — blocking

### ROPE-1 — Full-attention partial RoPE does not match LFM3's zero-tail scheme

**Files:** `crates/hipfire-arch-lfm3/src/forward.rs:258-270`, `crates/hipfire-arch-lfm3/src/config.rs:50-55`, `kernels/src/rope_partial_halfsplit.hip:39`
**Ground truth:** `/data/liquid-ai/LFM3-1.7B/modeling_lfm3.py:95-118` (`compute_default_rope_parameters`), plus the plan's own flagged item at `docs/plans/2026-10-31-lfm3-port.md:59-63`.

**Symptom.** The `full_attention` layers (partial-rotary factor 0.5) rotate the wrong dims with 2× the frequencies vs HF:
- HF: `inv_freq[i] = base^(-i/head_dim)` for `i∈[0,32)`, zero-padded to 64, then `emb = cat(freqs,freqs)` with a **full-length** `rotate_half` ⇒ rotates pairs `(d, d+64)` for `d∈[0,32)` (dims `{0..31, 64..95}`), freqs `{1.0, 0.806, 0.649, …}`.
- hipfire: `n_rot = partial·head_dim = 64`, kernel `freq = base^(-2i/n_rot)`, pairs `(i, i+32)` ⇒ rotates dims `{0..63}`, freqs `{1.0, 0.649, 0.422, …}` = HF's every-other frequency.

No value of `n_rot` makes the current kernel match (n_rot=128 would rotate dims 32..63 that HF leaves un-rotated). The `sliding_attention` layers (full rotary, n_rot=128) **do** match HF — only the 6 full-attn layers are affected.

**Evidence it is real.** Numeric diff reproduced analytically (see review); a byte/numeric parity test vs the HF reference was owed by the plan and does not exist (no rope unit test anywhere). End-to-end output is still coherent (model tolerates 6/24 layers), so this was not caught by the happy path.

**Fix options.**
1. Add a zero-tail partial-rotary kernel: pair `(d, d + head_dim/2)` for `d ∈ [0, rope_angles)`, `freq = base^(-2d/head_dim)`, leave the rest pass-through; route `full_attention` to it when `partial_rotary_factor < 1.0`. (Generalizes to qwen35 partial 0.25 if that model also uses the zero-tail layout — verify before swapping its dispatch.)
2. Or permute/inject at quantize time (heavier, affects all checkpoints).

**Verify.** Numeric parity of the rotated Q/K for one full-attention layer vs the HF reference at a few positions, on head_dim=128 / partial=0.5 (pin FP32 + `HIPFIRE_DETERMINISTIC=1` if byte-comparing). Add a unit test asserting the dim-pair/frequency mapping.

---

### PROMPT-1 — Corrupt prompt literal `IMDTextWriter`

**Files:** `crates/hipfire-runtime/examples/daemon.rs:19770`, `crates/hipfire-loader/examples/depth_smoke.rs:53`
**Also update the stale comments** referencing the same garble: `daemon.rs:19174, 19178, 19345` (` IMDA`).

**Symptom.** `format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n IMDTextWriter")`. `IMDTextWriter` is not a token; LFM3's control tokens are ` thinking` (124901) / `` (124902). The function's own comment (`daemon.rs:19741-19743`) and the validated examples (`omni_tts.rs:72`, `omni_stream_equiv.rs:19`) use `...assistant\n thinking`. Introduced/replaced in `e8b582b`. The shipped Omni TTS route therefore feeds BPE junk where the control token belongs.

**Fix.** Restore `"\n<|im_start|>assistant\n thinking"` in both files (and fix the ` IMDA` comments). Confirm against `/data/liquid-ai/LFM3-Omni-2B/tokenizer.json` added tokens.

**Verify.** Re-run the daemon Omni TTS path (or `omni_tts`) and confirm the prompt tokenizes to the intended control-token ids (124901/124902) and `started_in_think=false` still matches the framing.

---

### STREAM-1 — Omni streaming loop has no step budget → GPU OOB

**Files:** `crates/hipfire-runtime/examples/daemon.rs:19849` (loop), `:19781` (`max_steps`), `:19837` (codec cache sizing), `crates/hipfire-arch-lfm3-omni/src/codec.rs:919-921`.

**Symptom.** `while !stream_state.finished` has no total-step cap. `finished` is set only on audio-eos (AUDIO) or `<|im_end|>` (INTERLEAVED); pure AUDIO mode never reaches eos (plan admits this) and `mode` defaults to `"audio"` (`daemon.rs:5979`). So `{"tts":true,"stream":true}` never terminates: unbounded KV/`history` growth; `CodecStreamingCache` is sized for ≤ `max_steps` (≤200) frames but chunks are 12 frames, and `codec_decoder_forward` has no bounds guard ⇒ out-of-range KV writes (silent corruption/GPU fault), no terminal `done`, leaked tensors.

**Fix.** Carry a remaining total-step budget from `max_steps` into `omni_stream_step`; break/finish when exhausted (emit `done`). Optionally reject `stream:true` in AUDIO mode (only INTERLEAVED terminates naturally).

**Verify.** `{"tts":true,"stream":true,"mode":"interleaved"}` and `mode:"audio"` both terminate with a `done`; no codec KV/position exceeds `max_seq`.

---

## P1 — correctness / contract

### SPLICE-1 — Silent vision/audio splice fallback

**Files:** `daemon.rs:~19202-19213` (VL), `daemon.rs:~19506-19515` (audio).

**Symptom.** If the anchor token window isn't found (or the prompt is empty), the route prints a warning and continues with an un-spliced prompt; nothing asserts the prompt later contains the `n_visual_tokens` / `n_audio_tokens` placeholders, so generation "succeeds" with **zero** modality embeddings and returns an ungrounded answer.

**Fix.** Hard-error (`emit_active_attempt_error`/`write_error` + return) when the anchor is missing or the placeholder count != expected. Add an assert before prefill.

**Verify.** Unit-test a rendered prompt that lacks the anchor → route errors; normal path unchanged.

---

### VL-RESIZE-1 — LFM3-VL resized with Qwen pixel budget

**Files:** `crates/hipfire-arch-qwen35-vl/src/image.rs:69-70`, `daemon.rs` (VL preprocess call ~`19090/19121`).
**Ground truth:** `/data/liquid-ai/LFM3-VL-2B/processor_config.json` → `size = {shortest_edge: 65536, longest_edge: 16777216}`; `image_processing_lfm3_vl.py:83`.

**Symptom.** Shared `preprocess_dynamic_image` hardcodes `min_pixels=3136, max_pixels=1_003_520` (Qwen's). Any image `<65 536 px` (HF upscales) or `>1.0 M px` (hipfire downscales) changes the patch grid / visual-token count. The plan's "byte-identical reuse" claim is false; the two validated images fell inside the overlap.

**Fix.** Parameterize `smart_resize`/loader with min/max pixels (dots-ocr's `image.rs` already does) and pass LFM3's values for arch 13 VL.

**Verify.** Pin `(H,W)->(grid_h,grid_w)` for boundary sizes (e.g. 224×224, 200×200, 1600×1200) vs HF; numeric byte-diff on one out-of-overlap image.

---

### ASR-STR-1 — Streaming ASR drops lookahead + utterance tail

**Files:** `crates/hipfire-arch-lfm3-audio/src/lfm3_audio.rs:1141-1174`; `AudioEncoderStreamingCache` (`:1105-1128`).
**Ground truth:** `modeling_lfm3_audio.py:161-213`.

**Symptom.** HF `stream()` keeps `cache.features` (7 lookahead frames + left context), uses `num_tokens = (num_frames + (7 if is_final else 0))//8`, and finalizes the tail. Rust computes mel per chunk (drops ≤399 trailing samples), keeps only `num_frames - num_frames%8` (permanently discards ≤7 frames), and has no `is_final` flush → short clips can yield **0 tokens**; boundary tokens have no left context. `asr_stream_smoke`'s 19→18 token delta is this bug.

**Fix.** Carry leftover samples + mel frames in the cache, add `is_final`, mirror HF `offset`/`num_tokens` bookkeeping (or buffer the whole stream and call the non-streaming path).

**Verify.** Streaming == batch over a non-multiple-of-8 chunk split; short (<80 ms) clip emits its tail token; add the `is_final` case to `asr_stream_smoke`.

---

### AUDIO-NAME-1 — Omni audio tower not detected (name mismatch)

**Files:** `crates/hipfire-loader/src/carriers.rs:1469` (detection), `crates/hipfire-arch-lfm3-audio/src/lfm3_audio.rs:122` (`audio_config_from_hfq`), `crates/hipfire-quantize/src/main.rs:4805-4806` (F16-keep list).

**Symptom.** `Lfm3Carrier` detects audio via `encoder.downsample.conv_in.weight`, but the Omni artifact names it `model.audio_encoder.downsample.*` / `model.audio_projector.*` (verified: 170+6 tensors in `/data/liquid-ai/LFM3-Omni-2B/model.safetensors`). ⇒ `audio:false`, "model has no audio encoder" for the Omni model despite the tower being present; and the quantizer's F16-keep list doesn't match either, so those tensors get Q8'd.

**Fix.** Accept both prefixes and both config shapes (`config.preprocessor`+`config.encoder.lfm` vs `audio_encoder_config`). If Omni audio **input** is intentionally out of scope, document it explicitly rather than reporting the capability as absent.

**Verify.** Load the Omni HFQ → `loaded` reports `audio:true`; a WAV request splices audio tokens.

---

### ARCHID-1 — `derive_arch_id` omits `lfm3_omni`

**File:** `crates/hipfire-runtime/src/safetensors_source.rs:263` (mapping) and the stale comment above it.

**Symptom.** Maps `"lfm3" | "lfm3_vl" | "lfm3_audio" => 13` but not `"lfm3_omni"` (its `config.json` `model_type` is exactly that), while the quantizer maps `lfm3_omni`→13. Safetensors-dir load derives the unclaimed sentinel → "no carrier"; only the pre-quantized HFQ works.

**Fix.** Add `"lfm3_omni"`; refresh the "phased later" comment. Extend the `derive_arch_id` test.

**Verify.** `derive_arch_id(&json!({"model_type":"lfm3_omni"})) == 13`; dir-load the Omni checkpoint succeeds.

---

### FINISH-1 — LFM3 `done` omits `finish_reason`

**Files:** `daemon.rs:19014` (lfm3), `:19387` (vl), `:19687` (audio), `:19955` (omni stream), `:18715` region (lfm2moe).

**Symptom.** `pending_done` has no `finish_reason`; the loop breaks indistinguishably on `generated_count >= max_tokens` (length) and on EOS (stop). The CLI defaults a missing value to `"stop"` (`hipfire-cli/src/main.rs:4906`), so token-cap truncation is always reported as a natural stop.

**Fix.** Track the break cause; set `finish_reason = "length"` on the cap, `"stop"` otherwise (mirror the main generate path).

**Verify.** A max_tokens-capped LFM3 request returns `finish_reason:"length"` through the CLI/OpenAI surface.

---

### SUPPRESS-1 — Omni text argmax lacks HF token suppression

**Files:** `crates/hipfire-arch-lfm3-omni/src/omni.rs:236-245, 461-463`.
**Ground truth:** `modeling_lfm3_omni.py:1255-1265` (suppress set built from mode/eos ids, excluding `<|text_end|>`).

**Symptom.** HF forces the mode/EOS markers to `-inf` during text sampling; Rust's `max_by` is unguarded, so the text slot can emit `125012`/`125015`/EOS ids, desyncing the state machine and feeding a wrong embedding.

**Fix.** Mask the HF suppress set (at minimum `audio_start`, `mixed_start`, text eos ids except `text_end`) before the argmax in both text branches; centralize so interleaved-batch and streaming stay in sync.

**Verify.** Force a case where a suppressed token has the max logit → it is not chosen.

---

### DTYPE-1 — BF16/F16/HFQ4 dtype handling landmines

**Files:** `hipfire-arch-lfm3-omni/src/depth.rs:279-280, 294, 411`; `hipfire-arch-lfm3-omni/src/codec.rs:205-215, 221-225`.

**Symptom (latent today).** `bytes_to_f16` treats qt=16 (BF16) as raw F16 (no conversion) — used for `host_shared_codebook_embedding`; `depth.rs:294` computes element counts wrong for HFQ4; codec loaders decode `"F16"` as BF16. Current artifacts happen to dodge all three, but any dtype change silently yields noise.

**Fix.** Use `hipfire_runtime::safetensors_source::{source_bytes_to_f16_stream, source_bytes_to_f32_vec}`; add correct HFQ4 `numel` handling.

**Verify.** Load with a genuine F16 vs BF16 tensor and compare dequant output.

---

## P2 — robustness / quality

### EOS-1 — Audio-eos action diverges from HF
**File:** `hipfire-arch-lfm3-omni/src/omni.rs:130-134, 199-204, 406-411, 427-432`.
HF (`modeling_lfm3_omni.py:1734-1745`) replaces the frame with eos codes, feeds the embedding, switches to TEXT and continues; Rust `break`s. Decide: match HF or document the TTS-stop deviation. (The `.any()` test is correct; the `depth.rs:775-776` doc "in every codebook" is wrong.)

### EPS-1 — Audio projector LayerNorm eps
`lfm3_audio.rs:207`: `1e-6` → `1e-5` (HF `nn.LayerNorm(in_features)` default; `modeling_lfm3_audio.py:222`). One-line.

### REPPEN-1 — Repetition penalty compounds
`depth.rs:725-735`: dedup history (or snapshot raw logits) so a code repeated in the 64-frame window is penalized once (HF gather→where→scatter is idempotent).

### LEAK-1 — Unfreed `pos_buf` + codec error-path leaks
`lfm3.rs:905`, `lfm3_audio.rs:852`, `codec.rs:488` never free the 4-byte `pos_buf` DeviceBuffer (inherited from the lfm2moe template). Plus `daemon.rs:19871/19909/20058-20065` leak `codec_weights`/`codec_cache` on omni error returns, and `:19930` swallows the tail-flush error via `if let Ok`.

### THINK-1 — `gen_start.started_in_think` before framing
`daemon.rs:18760` uses `max_think_tokens != 1`; correct for Jinja but wrong for the `HIPFIRE_JINJA_CHAT=0`/Jinja-failure `Plain` fallback (would mislabel the whole answer as `reasoning_content`). Derive from the chosen frame (cf. `render_tail_opens_think`).

### GRAPH-1 — Raw `launch_kernel` on new kernels
`attention.rs:6374` (`vit_attention_gqa_f32`), `norm.rs:4029/4083` (`gelu_erf_f32`, `conv2d_gated_f32`) → migrate to `launch_maybe_blob` (AGENTS.md graph-capture dangling-kernarg trap). Pre-existing pattern, but `vit_attention_gqa_f32` is reused by the depth decoder.

### GRID-1 — `vit_attention_gqa_f32` gridDim.y/LDS ceiling
`attention.rs` launches `grid=[n_heads, N, 1]`, `LDS=(N+block)*4`. `gridDim.y` caps at 65535 and LDS ~64 KB (N≲16k). Safe at ~3920 today, but fixing VL-RESIZE-1 (16.7 M-px budget → N≈65536) trips it. Flatten N onto `gridDim.x` or tile.

### PANIC-1 — Loader panics on missing tensor
`hipfire-arch-lfm3-vl/src/lfm3_vl.rs:268, 273, 287, 299` `panic!` instead of returning `Err` (daemon crash on a partially-quantized HFQ). Convert to `Err`.

### MODE-1 — Silent mode defaults
`daemon.rs:19763` + `:5979`: any `tts_mode` ≠ `"interleaved"` becomes `"audio"` (typo silently selects pure TTS) → error on unknown. `omni.rs:349-353`: `omni_stream_prefill` silently defaults to AUDIO for any non-`mixed_start` last token while `omni_generate` (`:88-92`) errors → make consistent.

### DOC-1 — Registry/doc drift for arch 13
- `docs/architecture-ids.md`: primary-id table (`:31`) and `derive_arch_id` row (`:54`) — add 13 → `lfm3*`.
- `registry/models.json`: no LFM3 entry (the plan's wiring section asked for one); regenerate `registry/v1.json`.
- `admissions.yml` untouched (routing ≠ admission — confirm intent).

### TEST-1 — Tests
- `hipfire-arch-lfm3-omni` and `hipfire-arch-lfm3-vl` have **zero** unit tests.
- No daemon test pins the regression fixes (gen_start ordering, whitespace-EOS guard, abort transcript).
- `asr_stream_smoke` can't catch ASR-STR-1; `omni_stream_equiv` doesn't bound frame counts.

---

## P3 — nits

### DEAD-1 — Dead code / leftover diagnostics
`HIPFIRE_OMNI_EOS_DIAG` (`depth.rs:645`, `std::env::var` per `depth_forward`), `HIPFIRE_OMNI_DEBUG`/`hidden_cksum` (`omni.rs:28-38` + 4 sites), `let _ = (q_dim, kv_dim, cb_dim)` (`codec.rs:314`), `MelConfig::ENCODER_STRIDE`/`num_frames`, `AudioEncoderStreamingCache::{mel_n_mels,enc_hidden}`, `codec.rs:387` `max_seq` (unused), VL `conv_kernel_size`/`video_token_id`/`ProjectorConfig.bias` (parsed, unhonored), `let is_lfm3` (used only for an `eprintln!`).

### NITS-1 — Misc
- `omni.rs:117/177/366` `prompt_tokens.len() as u32 - 1` underflows on empty prompt → `saturating_sub`.
- `lfm3_vl.rs:644-663` 2×2 merge "rearrange" is an identity copy (drop the download/upload round-trip); HF does a plain reshape.
- `codec.rs:709-736` `irfft` rebuilds 641×1280 twiddle tables per STFT frame → hoist per decode.
- `wav.rs` doc says 16-bit-only (code handles 16/24/32-int/32-float); `WAVE_FORMAT_EXTENSIBLE`/8-bit rejected; `chunks_exact` drops a partial trailing frame.
- `depth.rs:704` `randperm` modulo bias.
- `emit_qwen_ar_cancelled` name is now generic (LFM cancellation emitter).
- Comment/doc fixes: `depth.rs:775-776` eos "every codebook"; stale `arch.rs` "Placeholder" docs; `wav.rs` header.

---

## Suggested landing order

1. **Quick wins (one-liners), can land together:** PROMPT-1, EPS-1, ARCHID-1, FINISH-1, MODE-1, DOC-1.
2. **Tighten the streaming/omni contract:** STREAM-1, SPLICE-1, SUPPRESS-1, LEAK-1, EOS-1.
3. **Loader/model-mapping:** AUDIO-NAME-1, PANIC-1, DTYPE-1.
4. **Numerics (needs parity checks):** ROPE-1 (block on this before merge), VL-RESIZE-1, ASR-STR-1, REPPEN-1.
5. **Hardening/tests:** GRAPH-1, GRID-1 (gates VL-RESIZE-1), TEST-1, DEAD-1, NITS-1.

**Merge gate:** ROPE-1, PROMPT-1, STREAM-1 must be resolved (or explicitly waived) before merge.

---

## Round 2 — residual findings from the post-fix verification (2026-11-03)

A verification pass over the five fix commits (d1d1fc0..db4a3a8) surfaced seven
residuals; all addressed in the round-2 commit(s):

1. **STREAM-1 budget under-count (medium, KV overflow)** — the streaming loop
   subtracted a full `chunk_steps` from the budget while always generating a
   full chunk, so total frames = ceil(max_steps/chunk)·chunk (max_steps=200,
   chunk=12 → 204 frames = 1224 decoder tokens vs a 1200-token cache; worst at
   small max_steps). Fixed twice: the daemon now clamps the last chunk to
   `min(chunk_steps, steps_left)`, and `codec_decoder_forward` gained a
   defense-in-depth guard that refuses to write past the KV cache's
   `physical_cap` instead of corrupting GPU memory.
   *GPU-verified (gfx1151): max_tokens=13 → exactly 13 frames (pre-fix: 24),
   max_tokens=200 → 155 frames, natural eos; streaming + batch `done` events
   carry the right `finish_reason`.*
2. **GRID-1 multi-tile path untested (medium-risk)** — the tiled
   online-softmax rewrite only changes behavior for N > 2048 and nothing
   exercised it. New GPU-gated integration test
   `crates/rdna-compute/tests/vit_attention_gqa_multitile.rs` runs
   `vit_attention_gqa_f32` at the VL GQA shape (16Q/8KV, head_dim 64) for
   N=2048 (tile boundary) and N=3000 (partial tail) against an f64 CPU
   reference softmax: max|gpu−cpu| ≈ 1.3e-6. *(Gracefully skips without an AMD
   GPU, same convention as `pp_parity.rs`.)*
3. **ASR streaming O(n²) CPU + unbounded host memory (perf)** —
   `audio_forward_streaming` accumulated every raw sample and recomputed the
   mel + downsampler over the whole stream per call. Now bounded with ZERO
   output change: the cache keeps only the sample tail from
   `frames_computed·hop` (each mel frame depends only on its own window —
   no global normalization) and computes only NEW frames; the downsampler
   runs on a bounded window starting at a multiple of 8 (token t depends only
   on mel frames [8t−8, 8t+8); three k3s2p1 convs = ±8-frame receptive field).
   Bit-identity of the window is pinned by
   `downsampler_window_bit_identity` (real LFM3-Audio-2B fixture weights,
   `assert_eq!` — including the final-flush partial token). Memory is now
   O(1) in stream length (sample tail < n_fft samples + ≤32 retained mel
   frames).
   *GPU-verified: `asr_stream_smoke` 19==19 tokens vs batch at 4 chunk sizes
   (1.6k/8k/16k/24k samples), all finite, `is_final` flush closes the stream.*
   Note: run-to-run token hashes differ even on the OLD implementation — the
   encoder's Q8-KV stochastic rounding is nondeterministic per run (the same
   property documented for the codec decoder), so end-to-end bit comparison is
   not a meaningful gate here; the downsampler-window unit test is the exact
   gate.
4. **FINISH-1 missed the omni batch done events (minor)** — the text-only
   (`audio_codes.is_empty()`) and audio-result batch `done` events defaulted
   to "stop" in the CLI. Both now emit `finish_reason` (`stop` on a natural
   eos/im_end/text_end finish, `length` on max_steps exhaustion), matching the
   streaming path's contract. *GPU-verified: batch TTS run reports
   `stop` (169 frames, natural eos) and `length` (max_tokens=13 cap).*
5. **codec non-finite waveforms (robustness)** — `exp(log_mag)` in the iSTFT
   overflows f32 at log_mag ≳ 88.7 (observed ~12% of codec_smoke batch decodes
   pre-fix). The polar construction now clamps log-magnitude to
   [−120, 80] (far above any legitimate speech spectrum) and maps NaN
   log-magnitude/phase to silence, so every decoded waveform is finite.
   `codec_smoke` now asserts finiteness on EVERY waveform (batch runs 1+2 and
   every streaming chunk) and uses a NaN-aware max|diff| (the old `f32::max`
   fold silently swallowed NaNs). *GPU-verified: 4× codec_smoke PASS, zero
   non-finite samples.*
6. **argmax_text_token NaN overclaim (nit)** — the old `bv >= v` comparison
   adopted a NaN whenever it was the current element, so a NaN in the LAST
   non-suppressed position still won (the old test passed only because its
   NaN wasn't last). NaNs are now skipped explicitly; new tests cover the
   last-position NaN and the all-NaN fallback.
7. **`RopeParams::n_rot` doc (nit)** — no longer says "dims actually rotated";
   documents that on the zero-tail partial path the rotated dims are the
   first `n_rot/2` pairs (d, d + head_dim/2), with only the full-rotary
   halfsplit path being a contiguous prefix.

Landing: round-2 fix commit(s) on `lfm3-port`. The merge gate (ROPE-1,
PROMPT-1, STREAM-1) now stands fully closed — STREAM-1's budget arithmetic is
clamped and guarded rather than merely documented.
