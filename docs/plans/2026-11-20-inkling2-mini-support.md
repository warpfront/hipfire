# Inkling-2-Mini support plan (planning only — no code changed)

Status: plan only (2026-11-20, release day for `thinkingmachines/Inkling-2-Mini`).
Upstream model was released the same day; re-verify every cited fact before
building. Prior research notes were treated as leads and spot-checked against
this checkout; where I could not verify, I say so.

Promoted 2026-11-20 from the gitignored `docs/plans/local/` scratch copy to the
tracked `docs/plans/` path for upstream review on `feat/inkling2-mini-support`
(base `origin/beta` `9e410448e`). Live verification results and resolved
decisions are recorded in §8; the rest is the original plan unless marked.

## 0a. Reference implementation (merged — verified 2026-11-20)

llama.cpp PR #32694 ("Inkling 2 Mini", author pcuenca, reviewed offline by
ngxson, merged by CISC) **merged 2026-11-20T22:01:08Z** as merge commit
`29b52d398b294509746ec0a3f28948a3107c0213` (9 commits, 13 files, +631/−123:
Conversion, Separate model_type, gate_rank, Base model, Loop graph, is_varloop,
naming from #32588, Rename to varying, master merge). Companion PR #32695
(danielhanchen, "looped arch", Draft) covers GGUF quant/shard follow-ups —
still open, do not rely on it. Reference files on master, read verbatim:

- `src/models/inkling2.cpp` (base arch forward), `src/models/inkling2-loop.cpp`
  (loop subclass), `conversion/inkling2.py` (converter + loop validation),
  arch/tensor plumbing in `src/llama-arch.*`, `gguf-py/gguf/*`.
- Published GGUFs: `ggml-org/Inkling-2-Mini-GGUF` (+ experimental
  `TML-d24-varloop-research-GGUF` for `is_varloop=true`).
- Quants: the converter's `_dequant_expert_chunk` **is** the NVFP4/MXFP8
  dequant reference (§8) — the earlier "no NVFP4-specific dequant math in the
  merged diff hunks" note was a sampling error, not a fact about the merged
  tree. Non-expert tensors and all loop/controller tensors convert F32.

Verified against the merged source (corrects §0 where noted):

- `layer_phys(il) = il < n_phys ? il : core_start + (il-n_phys) % (core_end-core_start)`;
  `layers[il] = layers[layer_phys(il)]` (structural sharing, per-slot KV/conv
  state); `n_slot = n_phys + (n_passes-1)*(core_end-core_start)` = 48, refused
  past `LLAMA_MAX_LAYERS`. Upstream validation (mirror this, don't invent
  stricter): fixed loop ≥ 2 passes (varying ≥ 1), `0 < start < end ≤ n_phys`,
  start/end block-aligned to `n_layers_per_attn_res_block`, `gate_rank ≥ 1`.
- Controllers: `gate = RMSNorm(cur)‖RMSNorm(msg)` (eps **1e-6**),
  `down [2·n_embd, rank]`, `gain/forget [rank, n_embd]`,
  `out = cur − tanh(F·gate)·cur + tanh(G·gate)·msg`. 4 controllers = Q/K/V
  inputs + FFN (LOOP_CTRL_Q..FFN); SWA looped layers control all three
  attention inputs, then per-input norm; FFN message applied at MoE input.
- Cross-pass query packing (converter-enforced): global `q`,`kv` each
  `[2·core_len, …]`; per core layer `i` the GGUF `loop_msg_query [n_embd,4]`
  stacks `[q[2i], kv[2i], kv[2i+1], q[2i+1]]` → [Q,K,V,FFN]. Validate shapes
  at ingest (`q.shape[0] == 2*(end-start)`, `kv == q`), fail closed.
- `mix()`: concat `[cur, block…]` → RMSNorm 1e-6 → softmax scores at
  `1/sqrt(n_embd)` → weighted sum. Bank = previous pass's blocks after the
  first core block + its output; prelude restored each pass (loop-graph
  confirmed, including the skip-first-core-block detail).
- Relative bias, corrected: R is projected to `n_rel` dims/head
  (`attn_rel [n_embd, n_rel·n_head]`), multiplied by the **`[n_rel, n_extent]`**
  table (`attn_rel_bias`), sink logit subtracted **before** gather, then
  `get_rows` by distance. (Plan draft had the projection direction vague;
  the table is `[n_rel, n_extent]`, not `[d_rel, extent]`-projected-R.)
- `clip()`: optional per-channel scale, row L2 with eps 1e-6, factor clamped
  [0,1]. Q clips with `attn_q_clip` scale, K scaleless, R with
  `attn_rel_clip` scale.
- `tau = 1 + alpha·log(clamp((pos+1)/floor, ≥1))`, multiplies Q **and** R on
  full layers only. Token embeds RMSNormed; image embeds deliberately not.
- Conv state is **one unified recurrent state** of width
  `max_layers(2·n_embd_k_gqa + 2·n_embd)` holding K, V, attn-out, FFN-out
  windows at offsets 0, `n_embd_k_gqa`, `2·n_embd_k_gqa`,
  `2·n_embd_k_gqa+n_embd`; post-conv scale `1/sqrt((dynamic?2:1)·n_conv+1)`.
  SWA layers: static k/v convs; full layers: input-dependent taps via
  `k/v_conv_proj [n_embd, n_conv·n_head_kv]`.
- `attn_res_query`: absent on layer 0; SWA layers 1 query, full layers 3
  (Q/K/V inputs differ). Output: mix with `output_res_query` → norm → head
  in F32 → softcap → **unpadded slice**; loader refuses unless tokenizer size
  == unpadded vocab (200,058, not 201,024 — size embeddings accordingly).
- MoE exact: `LLM_FFN_SITU`, sigmoid gating, `expert_weights_scale`
  (route_scale 8), norm eps 1e-20, gate_up interleaved
  `[n_embd, 2·n_ff, n_exp]`, gate bias optional (`TENSOR_NOT_REQUIRED`).
- No RoPE (`rope_pattern.fill(false)`); hybrid iswa+rel KV memory with a
  per-layer SWA pattern. Full-attention visits = 8, sliding = 40 (matches the
  third-party otis catalog note "8 global and 40 sliding-window layers" —
  independent cross-check of the 48-visit schedule).
- **block_count trap confirmed real**: unsloth GGUFs written with 48 blocks
  fail to load on the merged build (llmctl pins ggml-org's GGUF for exactly
  this reason). GGUF must carry 24 physical + loop metadata; hipfire must
  stamp physical count + loop config and never unroll.
- `is_varloop` exists upstream (varying flag, min 1 pass, research GGUF
  only). Mini is `is_varloop=false`; Phase 1 refuses varying (same
  fail-closed posture as upstream's invalid-config throws).
- Re-verified verbatim from `inkling2.cpp`, `inkling2-loop.cpp` and
  `conversion/inkling2.py` at the full merge SHA (§8): the controller, `mix()`,
  `clip()`, `convolve()`, relative-bias gather, `tau`, MoE and output bullets
  above; the prelude save / skip-first-core-block bank; and the cross-pass
  query packing. Extra ingest details and two corrections are in §8.
- Two HF controller namings accepted upstream: Mini's
  `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight` and the varloop
  checkpoint's `{q,k,v,mlp}_controller.{down,gain,forget}_proj`. Ingest
  targets Mini's; note the variant.

## 0. What the model is (verified 2026-11-20)

Fetched live: `config.json`, `hf_quant_config.json`,
`model.safetensors.index.json`, `chat_template.jinja` from
`thinkingmachines/Inkling-2-Mini`. Geometry below was re-checked against the
live `config.json` on 2026-11-20; no discrepancies found (§8).

- `model_type: inkling2_loop`,
  `architectures: ["Inkling2LoopForConditionalGeneration"]`,
  text sub-config `inkling2_loop_text`. Multimodal (23-layer CLIP-style ViT,
  patch 20, projector 4096). 26.4B total, ~3.4B active (5.4B effective with 3
  loop passes).
- **Loop**: 24 physical layers; 0–5 once, 6–17 three times, 18–23 once =
  **48 visits/token**. Each visit keeps its **own KV and short-conv state**;
  weights are shared across passes. `loop_config`: core 6–18, 3 passes,
  gate_rank 64, `is_varloop: false`.
- Per core layer: 4 controllers (`attn,k,v,mlp` × `down,gain,forget`,
  rank 64) + globals `middle_cycle_cross_pass_queries`,
  `middle_cycle_cross_pass_kv_queries`. Controller math (from lead, NOT yet
  verified against HF transformers source — P0 oracle must confirm):
  `h=RMSNorm([cur;msg]); out=(1-tanh(F·h))·cur + tanh(G·h)·msg`.
- Cross-pass messages (`mix()` over `[earlier block outputs…, cur]` with a
  learned query); cross-layer attn residual `attn_res.attn_queries.0..22`,
  `mlp_queries.0..23`, `output_query`, in blocks of 6
  (`n_layers_per_attn_res_block=6`).
- Attention: 24Q/8KV, head_dim 128, 5 sliding (1024) + 1 full per 6-block,
  **no RoPE**; learned relative logits (`d_rel=16`, `rel_extent=1024`),
  learned sink, output gate, dynamic K/V convs (kernel 4), QK + relative L2
  clips (22.627…, 8.0), value RMSNorm, log-scaled positions on full layers
  (`tau=1+0.1·log(max(1,(pos+1)/8192))`), final softcap 30.
- MoE: 128 experts, top-8, intermediate 1152, gate bias, `route_scale=8`,
  SiTU softcaps 4/25. No shared experts.
- Quantization in the published checkpoint: `quantization_config` is
  `{"quant_method":"modelopt","quant_algo":"NVFP4"}`; 40 NVFP4 tensors (block
  1×16, E4M3 scales + `amax`/`outer_scale_denominator` companions), 8 MXFP8
  tensors (block 32×32), 993 BF16 + 74 FP32. **No original pre-quant BF16
  checkpoint is published**, but ggml-org publishes a dequantized
  `Inkling-2-Mini-BF16.gguf` (52,226,509,504 B, sha256 `7074c976…`) and
  `imatrix.gguf` (90,752,032 B, sha256 `553662ef…`). The BF16 GGUF is a dequant
  oracle/reference only — GGUF **weight** input to the quantizer was removed in
  0.5.0; `imatrix.gguf` remains accepted. Weight path is still quant→quant
  (see §5).
- Tokenizer: BPE ByteLevel + Llama-3-style split regex, 201,024 vocab
  (200,058 unpadded), custom message/content/thinking/tool tokens; chat
  template takes `reasoning_effort` (0.0–0.99, default 0.9) and renders tool
  calls as strict canonical JSON.
- `mtp.safetensors` (636 MB): 3-layer DSpark draft, markov rank 256, block 8,
  reads decoder visits 5/23/41/48. This is **not** hipfire's Qwen3 5-layer
  dense DSpark (arch 20) — new drafter semantics.

## 1. Recommended architecture

**Visit mapping lives in the new arch crate; physical weights stored once;
all stream state sized per-visit (48).** Concretely:

- New crate `crates/hipfire-arch-inkling2/` (copy `hipfire-arch-toy`
  skeleton): `config.rs` (loop geometry + validation), `inkling2.rs`
  (bundle: `weights[24]` physical + `state[48]` visits), `forward.rs` with
  `loop_layer(visit)->phys`, `loop_visit(visit)->pass`,
  `loop_n_extra()=24`, and a `VISIT_SCHEDULE: [u8; 48]` table built from
  `(core_start, core_end, num_passes)` and **validated with upstream's rules
  (fixed ≥ 2 passes, `0 < start < end ≤ 24`, block-aligned to 6, `rank ≥ 1`,
  `n_slot = 48`); any violation refuses the load**; `carrier.rs` (thin, calls
  into the crate per the
  Tier-B rule that `Carrier::load` returns `LoadedModel`).
- Weight aliasing, not duplication: the 12 core layers' tensors are uploaded
  once; visits index them. KV caches, short-conv states, controller scratch,
  and the cross-pass prelude/bank buffers are per-visit (48× seq, 1× params).
- Loader routing: new primary `arch_id` **24** and MTP draft sidecar **25**
  (educated guesses, cheap to change later — §8). On this base (`origin/beta`
  `9e410448e`) 0–19 are claimed (18 EmbeddingGemma2, 19 Raon), 2–4 stay
  untouched per `architecture-ids.md`, 20–23 are sidecars, 40+ diffusion; 24 is
  the first free primary id and 25 the first free sidecar after 23. Naming:
  crate `hipfire-arch-inkling2`, carrier `InklingCarrier`, text route
  `InklingAr` / later `InklingSpec`. `model_type` rows `inkling2_loop`, `inkling2_loop_text`
  (+ `inkling2_vision` classified as vision sidecar, not a trunk) in
  `crates/hipfire-runtime/src/arch_mapping.rs`; class-substring row for
  `Inkling2LoopForConditionalGeneration` via the longest-key-wins loop in
  `derive_arch_id` (already table-driven — no other routing edit needed).
- Generate: new `GenerationRoute::InklingAr` (+ `InklingSpec` later) in
  `hipfire-generate/src/ar.rs`, `generate_inkling` in `dense.rs`; **no**
  batch/slot/PP/EP/Redline in Phase 1.

**Rejected alternative — unroll to a 48-layer pseudo-model** (each visit its
own weights, standard 48-layer forward): simpler dispatch but ~2× weight VRAM
on a 26B model (breaks 24 GB cards), silently redefines every tensor identity
imatrix/calibration tooling keys on, and walks straight into the
block_count 48-vs-24 trap a llama.cpp PR comment already flagged. The alias
table costs one indirection per visit and keeps every other tool honest.
A second rejected option — host-side loop orchestration calling a 24-layer
forward 3× per token — would thrash KV/conv state in and out per pass with
no fusion headroom; keep the schedule inside the forward.

**New-vs-reused operator split** (§4 has the kernel list): reuse MoE
top-8/sigmoid routing + sliding-window + sink + softcap + short-conv
*scaffolding* from cohere2moe/ds4/gemma4/qwen35; write new code only for the
loop schedule/state, the relative-bias gather, L2 clips, controller fuse,
`mix()`, and SiTU-softcap MoE activation.

## 2. Phased plan (correctness provable at each step)

### Phase 0 — CPU oracle + ingest probe (no GPU correctness claims yet)

1. Port the merged `inkling2.cpp`/`inkling2-loop.cpp` forward to a throwaway
   CPU oracle (prefer a `hipfire-arch-inkling2` `#[cfg(test)]` module so it
   lives with the crate) covering: visit schedule (`layer_phys`, prelude/bank
   save-restore incl. the skip-first-core-block detail), one controller step
   (eps 1e-6, tanh gates), one `mix()` step, one relative-bias gather with
   pre-gather sink subtract, one dynamic-conv step (unified state + offsets +
   post-scale). Parity target: HF transformers (preferred) or the merged
   llama.cpp master build (stable now — merge commit `29b52d398b294509746ec0a3f28948a3107c0213`, no longer a moving
   PR branch) on 2–3 fixed prompts; assert per-visit hidden states, not just
   final logits.
2. Ingest probe: extend `dequant.rs` with NVFP4 (1×16) and 2-D-block MXFP8
   (32×32) readers (§5), dequant one expert tensor of each kind, and compare
   against the GGUF-converted F32 weights elementwise. **Nibble order and
   `amax/denominator` semantics are guilty until proven innocent — the merged
   PR's diff shows no NVFP4-specific dequant math to copy.**
3. Trap tests first: 48-vs-24 block_count (refuse unrolled 48-block inputs —
   this exact failure already bricks unsloth GGUFs on merged llama.cpp),
   `is_varloop=true` refusal, schedule-mismatch refusal, cross-pass query
   shape (`[2·core_len,…]`) refusal, NaN-scale refusal.

### Phase 1 — smallest useful text-only slice (AR, capped context, no vision/MTP)

Files (Tier A–D of the toy checklist; Tier E–F deferred except ids/registry):

- `crates/hipfire-arch-inkling2/{config,inkling2,forward,carrier,arch,arch_model}.rs`
  + `Cargo.toml`; `scripts/layering.txt` + `crates/*/map.md` seed
  (`check-crate-maps.py`).
- `crates/hipfire-runtime/src/arch_mapping.rs`: `("inkling2_loop",24)`,
  `("inkling2_loop_text",24)` (+ class alias if needed).
- `crates/hipfire-quantize/src/{dequant,pipeline,model_filter}.rs`: NVFP4 +
  32×32-MXFP8 ingest (§5), keep-F16/BF16 lists for
  controllers/queries/norms/biases/scales, per-expert split for the
  pre-split `mlp.experts.w13/w2_weight` tensors (DeepSeekV4/MiniMax precedent,
  not the stacked-3D Qwen3.5 path).
- `crates/hipfire-loader/src/{lib.rs (REGISTRY line),carriers.rs (InklingCarrier)}`,
  `crates/hipfire-generate/{Cargo.toml,src/ar.rs (route),src/dense.rs (generate_inkling)}`.
- `crates/hipfire-loader/src/carriers.rs` caps: `supports_images=false`,
  no MTP floor (`mtp_measured_ar_floor=false`), no batch/PP/EP.
- Config surface: none new except reading loop geometry from checkpoint; cap
  `max_seq` well below 1M initially (suggest 32768; KV is 48 visits deep —
  §6). Refuse vision tensors/tokens fail-closed in Phase 1.
- Tokenizer + template: load 201k BPE via the existing tokenizer path
  (precedents: 259k raon, 262k cohere — low risk, but verify the Llama-3
  split regex and every custom special token); port `chat_template.jinja`
  semantics (`reasoning_effort` default 0.9, strict tool-JSON) to the
  arch-default template arm, behind the standard override chain.
- Tests/gates: CPU oracle parity (Phase 0) → `serve_harness.py battery +
  chain` on 2–3 pinned text prompts (eyeball output; tight-stddev suspicion
  rule applies) → `redline_daemon_harness.py` once new kernels land →
  `scripts/pre-commit-ci.sh` + `cargo test -p railgun-cert` on the final
  commit. Pin KLD (`.kldseq` SHA) for the text-only route as the bitwise bar
  for all later optimization.

### Phase 2 — vision (follows Phase 1 acceptance)

- ViT tower (23-layer CLIP-style, patch 20) + projector as an embedded tower
  in the same crate (precedent: `hipfire-arch-qwen35-vl`, `dots-ocr`;
  embedding RMSNorm on text tokens but **not** on image tokens — encode as a
  tested branch, not a comment). `vision_route` + daemon image gating +
  `supports_images=true`.
- Oracle: image+text parity on 2–3 fixed images before any perf claim.

### Phase 3 — DSpark/MTP draft (follows Phase 1; independent of Phase 2)

- New sidecar id **25** (educated guess, §8; **not** 20/21/22/23 — claimed with
  incompatible tensor namings) + `mtp_extract`-pattern packer for
  `mtp.safetensors`; 3-layer dense +
  markov-rank-256 draft forward; `MtpSpeculator` attach behind explicit
  opt-in only (no measured AR floor → `auto` stays AR, same posture as
  Qwen35). Verify sampled-distribution behavior explicitly; greedy
  byte-parity vs AR is the entry gate.

### Phase 4 — perf, calibration, admission

- imatrix-graded expert tiers via the existing `HIPFIRE_MOE_GRADED`
  machinery (collect **after** Phase 1 decodes correctly — the Phase-1 ship
  is uniform MQ4G256V2 + Q8 router); prefill WMMA for the small-expert
  geometry only with a served A/B per `lessons-0.4.5.md` §3.
- `registry/models.json` card + `registry_gen.py` → `v1.json`; admission row
  in `docs/admissions.yml` (curated, evidence-bound — a card is not
  admission); `docs/{ARCHITECTURE,MODELS,architecture-ids}.md`, `CLAUDE.md`
  id list, `docs/env-vars.md` only if a knob is added (avoid new knobs).

## 3. New GPU kernels/operators (and what each adapts)

| # | Operator | Adapt from |
|---|---|---|
| 1 | Visit-schedule dispatch + per-visit KV/conv state index | new (llama.cpp `loop_layer/visit` shape); state containers mirror `qwen35` per-layer state ×48 visits |
| 2 | Controller fuse: `RMSNorm([cur;msg]) → rank-64 down → tanh gain/forget blend`, 4×/core-layer | new fuse; RMSNorm piece from `rmsnorm_batched`, GEMM pieces from existing MoE/dense GEMV paths |
| 3 | Cross-pass `mix()`: softmax over block outputs + cur with learned query | new small kernel (host-composed in Phase 0/1, fuse in Phase 4) |
| 4 | Relative-bias gather: R→`n_rel`/head × `[n_rel,n_extent]` table, sink subtracted pre-gather, distance gather | new; no hipfire precedent (T5 bucketing is FLUX-encoder-only). Validate `loop_msg_query [n_embd,4]` packing `[q2i,kv2i,kv2i+1,q2i+1]` at ingest |
| 5 | Q/K + relative L2 clip to thresholds | new elementwise; pattern-match `logit_softcap_f32` |
| 6 | Value RMSNorm (no gamma) + full-layer Q/K input norms | reuse `rmsnorm_batched` (+ ones-weight trick from glimmer scale-less QK-norm) |
| 7 | Dynamic K/V conv (k=4, `weight_proj`+`static_w`) + attn/FFN-out short convs over one unified recurrent state (width `max(2·n_embd_k_gqa+2·n_embd)`, fixed offsets, post-scale `1/sqrt((2?2:1)·n_conv+1)`) | adapt `execute_gated_short_conv` / `conv1d_silu` (qwen35 GDN) + raon conv-4 precedent |
| 8 | `tau` log-scale multiply on Q,R (full layers) | fuse into attention prologue (one-liner, no kernel) |
| 9 | Output-gate sigmoid multiply; sliding-1024 mask + sink-softmax attention, NoPE | adapt `attention_dflash_sliding_f32` + ds4 `attn_sink` plumbing |
| 10 | MoE: sigmoid+bias top-8 (no renorm), `×route_scale 8`, SiTU(4/25) experts, 128×1152 | routing from cohere2moe (identical 128/top-8/sigmoid/no-renorm); SiTU-softcap activation new (maple clamped-SiLU is the nearest pattern) |
| 11 | Final softcap 30 | reuse `execute_softcap` (gemma4 precedent) |

Phase-1 license to be slow: host-compose #3, run #4/#2 unfused-but-correct,
fuse only in Phase 4 behind the KLD pin.

## 4. Weight-ingest path (exact)

- NVFP4 1×16 (`w13/w2_weight` layers 0–19): **verified against the merged
  converter (§8)** — low nibble first (`packed & 15`) then high (`packed >> 4`)
  through the E2M1 table `[0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, …]`, × per-16 E4M3
  scale, then × `amax / outer_scale_denominator` (F32 scalars, per-expert and
  global respectively). No extra `weight_scale` factor: `amax/d` **is** the
  outer scale. Strict: refuse E4M3-NaN codes, non-finite amax (must be ≥ 0) or
  denominator (must be > 0), and non-finite products (same posture as
  `dequantize_mxfp8_e8m0_strict`).
- MXFP8 **1×32, not 32×32** (layers 20–23): **correction from the merged
  converter (§8)** — it requires the stored block
  `(rows/scale_rows, cols/scale_cols) == (1, 32)` for MXFP8 (`(1, 16)` for
  NVFP4); E4M3 values × UE8M0 `2^(byte-127)` per 32 columns. The earlier
  "2-D 32×32 block" reading was wrong: the existing `[1,block_k]`-only
  `dequantize_mxfp8_e8m0_strict` needs **no 2-D generalization**, just
  `block_k = 32`. If `hf_quant_config.json` advertises
  `weight_block_shape [32,32]`, that is a metadata conflict the converter only
  warns about — trust the stored tensor/scale shapes.
- BF16/F32 (controllers, queries, norms, biases, `q/k_l2_clip.scale`,
  `rel_l2_clip.scale`, vision, embed, unembed, MTP): existing `to_f32`
  paths; add keep lists (`model_filter.rs`, MiniMax/DS4 precedent) so
  small controller/query tensors are never requanted.
- Target: **requant to hipfire's two focus quants — MQ4 (qt=44 `MQ4G256V2`)
  and MQ4L (qt=52 `MQ4G256V2Lloyd`)**, not the broader MFP4 family; no new `qt`
  (`qt-register.txt` untouched; native NVFP4/MXFP8 runtime types out of scope).
  MQ4L is the same 136 B V2 group plus a per-tensor `lloyd_levels` F32[16]
  codebook; quantizer flag `mq4v2-lloyd` (aliases `mq4l`, `mq4v2l`,
  `mq4g256v2l`, `mq4g256v2-lloyd`); product tiers `mq4l-xt` / `mq4l` /
  `mq4l-pro`. Phase-1 default: uniform MQ4G256V2
  (+Q8 router/bias-keep); MQ4L is the first quality upgrade once Lloyd
  calibration is available; graded experts only in Phase 4 with imatrix.
- Calibration/imatrix: imatrix collection needs a running model, so Phase 1
  ships uniform quality and collects imatrix from the Phase-1 build; the
  graded recipe lands in Phase 4. Expect a quality gap vs the NVFP4 original
  on the small 1152-wide experts — state it, don't hide it.

## 5. Risks, unknowns

1. **Quant→quant**: no BF16 source exists; both NVFP4→MQ4 and MXFP8→MQ4 lose
   information the original never had to lose. Mitigation: strict dequant
   parity first, uniform-MQ4 quality measured before any graded claim.
2. **48-visit cost model**: core layers run 3×/token; prefill is ~2.5× a
   24-layer model and KV is 48 visits deep — the 1M `model_max_length` is
   not servable on day one (cap + refuse loudly).
3. **Upstream residual drift**: base support is merged and stable
   (`29b52d398b294509746ec0a3f28948a3107c0213`); only the quant/shard follow-up
   (#32695, Draft) still moves.
   The NVFP4/MXFP8 dequant semantics are now pinned from the merged converter
   (§8); what remains is the ingest-parity probe, not the semantics.
4. **MTP head is alien**: markov-rank-256 DSpark matches no hipfire drafter;
   budget it as a mini-arch, not a config flag.
5. **Template surface**: `reasoning_effort` + strict tool-JSON + custom
   thinking/tool tokens — template bugs are silent-wrong, so parity-fixture
   them like kernels.

## 6. Decisions (resolved 2026-11-20)

1. **Arch ids**: primary **24** for `inkling2_loop(_text)`, MTP draft sidecar
   **25**. Educated guesses, explicitly cheap to change later (no artifact has
   shipped on either). 24 is the first free primary id on this base, 25 the
   first free sidecar after 23; ids 2–4 remain untouched.
2. **Quant posture**: accept quant→quant requant for Phase 1, focused on
   **MQ4 (qt=44)** and **MQ4L (qt=52)**, with a stated quality gap vs the NVFP4
   original. Do not hold Phase 1 for a BF16 release and do not fund native
   NVFP4/MXFP8 runtime types. The published dequantized BF16 GGUF and
   `imatrix.gguf` are evidence/calibration inputs, not a BF16 training source.
3. **Scope caps**: text-only AR at `max_seq` 32768 as the Phase-1 ceiling;
   vision, MTP and 1M context stay deferred. Accepted.
4. **Destination**: this plan is developed on `feat/inkling2-mini-support` off
   `origin/beta` and is intended to land in an upstream PR to `beta`.

## 7. Effort (honest, calendar time for one engineer with hardware access)

- Phase 0 (oracle + ingest probe): 3–5 days.
- Phase 1 (text-only AR, capped, gated): 3–5 weeks (ingest ~1wk, crate +
  forward ~2wk, attention/conv kernels ~1–2wk, validation/templates ~1wk).
- Phase 2 (vision): 2–3 weeks. Phase 3 (DSpark draft): 3–4 weeks (mini-arch).
  Phase 4 (graded quant + fusion + admission evidence): 2–4 weeks.
- Full support ≈ 10–16 weeks after Phase 1; Phase 1 alone is the only
  committable slice.

## 8. Verification log (2026-11-20, live sources)

Re-checked from live sources before any build; "unverified" items below are
carried forward from the original plan and remain Phase-0 tasks.

| Claim | Source | Result |
|---|---|---|
| Geometry: 24 physical layers, loop core 6–18 ×3 passes, gate_rank 64, `is_varloop=false`, 24Q/8KV hd 128, `d_rel` 16, rel_extent 1024, QK/rel clips 22.627/8.0, `final_logit_softcapping` 30, MoE 128/top-8/1152/`route_scale` 8 + gate bias, sconv kernel 4, SWA 1024, 6-layer attn-res blocks, unpadded vocab 200058, MTP 3 layers/markov 256/block 8/target visits 5/23/41/48, `model_max_length` 1048576 | live `thinkingmachines/Inkling-2-Mini/config.json` | confirmed |
| `reasoning_effort` default 0.9, strict canonical tool JSON, custom roles/thinking/tool tokens | live `chat_template.jinja` | confirmed |
| `quantization_config` = modelopt / NVFP4 | live `config.json` | confirmed |
| llama.cpp support merged | GitHub API PR #32694 | merged 2026-11-20T22:01:08Z by CISC; full merge SHA `29b52d398b294509746ec0a3f28948a3107c0213`; 9 commits, 13 files, +631/−123. Short `29b52d3` did not resolve via GitHub web/API here — use the full SHA. |
| GGUF fixtures published | `hf models list` | both repos exist: `ggml-org/Inkling-2-Mini-GGUF` (BF16 52,226,509,504 B sha256 `7074c976…`, IQ4_NL, Q8_0, `imatrix.gguf` 90,752,032 B, mmproj) and `ggml-org/TML-d24-varloop-research-GGUF` (BF16/IQ4_NL/Q8_0/imatrix/mmproj). |
| Arch ids 24 / 25 free | `docs/architecture-ids.md` on `origin/beta` `9e410448e` | 18 (EmbeddingGemma2) and 19 (Raon) are now claimed; 24 is first free primary, 25 first free sidecar after 23. |

### Merged-source re-verification (2026-11-20)

Read verbatim from `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`
and `conversion/inkling2.py` at the full merge SHA:

| Mechanism | Result |
|---|---|
| `control()` | confirmed: `gate = RMSNorm(cur,1e-6) ‖ RMSNorm(msg,1e-6)`, down `[2·n_embd, rank]`, gain/forget `[rank, n_embd]`, `out = cur − tanh(F·gate)·cur + tanh(G·gate)·msg`. |
| `mix()` | confirmed: concat `[blocks…, cur]` → RMSNorm 1e-6 → scores at `1/√n_embd` → weighted sum; bank = previous pass's `blocks[prelude_len+1..]` plus its output, prelude restored each pass (skip-first-core-block detail confirmed). |
| `clip()`, `convolve()`, `tau`, output | confirmed: row L2 + 1e-6 with factor `threshold/norm` clamped [0,1]; unified conv state offsets 0 / `n_embd_k_gqa` / `2·n_embd_k_gqa` / `2·n_embd_k_gqa+n_embd`, `out = cur + conv`, post-scale `1/√((dynamic?2:1)·n_conv+1)`; `tau = 1+α·log(max(1,(pos+1)/floor))` on Q and R of full layers; F32 head → softcap → unpadded slice. |
| Relative bias | confirmed: R projected to `n_rel`/head × the `[n_rel, n_extent]` table, sink logit subtracted **before** the distance gather. |
| **NVFP4 dequant** | **resolved**: E2M1 low nibble first, then high, table `[0,.5,1,1.5,2,3,4,6,-0.0,…]`, × E4M3 scale per 16 columns, then × `amax/d` (F32 scalars; amax ≥ 0, denominator > 0; E4M3-NaN and non-finite results refused). There is **no separate `weight_scale` factor**. |
| **MXFP8 block shape** | **corrected to 1×32** (stored block `(1,32)`, NVFP4 `(1,16)`), E4M3 × UE8M0 `2^(byte−127)`. No 2-D-block decoder is needed; a `weight_block_shape [32,32]` in `hf_quant_config.json` is only a warning. |
| Loop validation | confirmed, plus one extra rule: `n_layer_nextn == 0` is required; `n_slot = 48 ≤ LLAMA_MAX_LAYERS`. |
| Ingest transforms | static conv weights are `flip(0).T` at conversion; `attn_res.attn_queries.{i}` maps to block `i+1`; `mlp.gate.bias` is the router `e_score_correction_bias`; F32 keep list covers conv/clip/res_query/attn_rel_bias and every loop tensor. The text converter skips vision and MTP tensors (matches the Phase-1 refusal). |

Still oracle-owned by Phase 0: elementwise dequant parity against the
published BF16 GGUF, the 48-vs-24 `block_count` trap on unsloth GGUFs,
HF-transformers numerical parity of the controllers, and the template surface.
Source confirmation is not numerical proof.

### Phase-1 file map (checked on `origin/beta` `9e410448e`)

All paths named in §1–§4 exist on this base: the toy skeleton is
`crates/hipfire-arch-toy/{lib,arch,arch_model,carrier,toy_model}.rs` + `Cargo.toml`
+ `map.md` (note: it has **no** `config.rs` / `forward.rs` — those two are new
files, not copies); the quantize, runtime, loader and generate files are all
present; and `scripts/{serve_harness.py,redline_daemon_harness.py,pre-commit-ci.sh,check_fixture.sh,check-crate-maps.py,check-attribution.py}`
exist.

Two ingest consequences of the source reading:

- `dequant.rs` already has `dequantize_mxfp8_e8m0_strict(weight, scale, block_k)`
  with 2-D weight/scale, shape `[rows, cols/block_k]`, block `[1, block_k]` and
  strict E4M3-NaN / E8M0-255 / non-finite checks. The merged converter's MXFP8
  is exactly `block_k = 32` — **no new 2-D decoder** (the §4 correction).
- NVFP4 has no existing decoder: `dequantize_e2m1_ue8m0_to_f32` pairs E2M1 with
  **UE8M0**, not E4M3 scales, and `dequantize_e4m3_f32scale_to_f32` is the
  MiniMax block-128 path. The 1×16 E2M1 + E4M3 + `amax/d` combination needs one
  new strict function plus a `dequant.rs` dispatch arm for a 3-D scale and the
  two F32 companions.

`derive_arch_id` does longest-substring matching over `model_type` and
`architectures` in `arch_mapping.rs`, so the §1 routing rows are sufficient —
no other routing edit is needed. Phase-1 `serve_harness` prompts should be
committed files under `benchmarks/prompts/` (several suitable fixtures already
exist); do not use `/tmp`.

Environment notes: downloads go to `D:\hipfire-models`; `hf` is authenticated;
`codegraph` is absent in this worktree, so exploration uses grep/read.
