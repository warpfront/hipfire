# Inkling-2-Mini implementation log (2026-11-21)

Working branch: `feat/inkling2-mini-support` off `origin/beta` (`9e410448e`),
in worktree `C:\Users\Jan\AppData\Local\Temp\opencode\hipfire-inkling2`.
A **draft PR** is open for visibility only:
https://github.com/warpfront/hipfire/pull/1192 (fork branch
`raufaser:feat/inkling2-mini-support` → `warpfront:beta`). It is explicitly
**not ready for review or merge** — the model cannot be served yet. Pushing to
`warpfront` directly is not permitted (no write access), so the PR comes from
the `raufaser` fork. All commits carry an `Agent: hy4-flash-free/<seat>` trailer;
`python3 scripts/check-attribution.py --range origin/beta..HEAD --require` passes.

The planning document is
[`docs/plans/2026-11-20-inkling2-mini-support.md`](2026-11-20-inkling2-mini-support.md).
A synced copy of that plan is kept at
`C:\Users\Jan\Development\hipfire\docs\plans\local\2026-11-20-inkling2-mini-support.md`
(gitignored scratch, for VS Code visibility).

## Environment

- GPU: AMD Radeon RX 7900 XTX (gfx1100), ROCm 10.1 for Windows at
  `C:\Users\Jan\Development\therock-dist-windows-gfx110X-all-10.1.0`.
- Checkpoint (19 GB) downloaded to
  `D:\hipfire-models\Inkling-2-Mini\` (single safetensors + `mtp.safetensors`,
  tokenizer, configs). All downloads go under `D:\hipfire-models`.
- llama.cpp reference at merged PR #32694, commit
  `29b52d398b294509746ec0a3f28948a3107c0213`, in a detached worktree
  `C:\Users\Jan\AppData\Local\Temp\opencode\llama-inkling2`. The user's own
  `C:\Users\Jan\Development\llama.cpp` checkout was left untouched.
- `codegraph` is unavailable; use grep/read. `cargo fmt` is aliased away; use
  `rustfmt --edition 2021 --config skip_children=true <files>` directly.

## Done (commits on the branch, oldest first)

1. `873e248d1` — plan promoted from gitignored scratch to tracked `docs/plans/`;
   live HF config/chat-template/GGUF-repo facts verified; decisions resolved.
2. `4635c5c8f` — dequant semantics verified from the merged llama.cpp source:
   NVFP4 nibble order and `amax/d`, and the MXFP8 **1×32** stored-block
   correction (plan previously said 2-D 32×32).
3. `71b0dd722` — Phase-1 file map checked against this `origin/beta` base.
4. `50f87f38f`, `85e06e5b1`, `895eab551`, `73fd57f39` — CI baseline note
   (Windows-host failures reproduce on a clean base worktree), GGUF-header
   verification (`block_count=24` + loop keys), quant metadata pin, risk wording.
5. `22adb30b7` — **implementation**: `dequantize_nvfp4_amax_denominator` in
   `crates/hipfire-quantize/src/dequant.rs`, strict, reusing E2M1/E4M3 helpers;
   4 unit tests.
6. `de59c12c0` — new crate `crates/hipfire-arch-inkling2`: `LoopConfig` with the
   24-physical/48-visit schedule, upstream-matching validation traps, provisional
   arch ids 24/25, and CPU oracle helpers (`rmsnorm`, `clip`, `tau`); 9 tests.
7. `4bca3e36a` — two **ignored** real-checkpoint ingest probes in `dequant.rs`:
   NVFP4 and MXFP8 expert 0 from the downloaded model, dtype/shape/stats printed
   and finiteness asserted. NVFP4 first-16 values match an independent Python
   port of the merged converter exactly. MXFP8 confirms `[1,32]` scale shape.
8. `e58931a63` — oracle operators `matvec`, `control`, `mix`, `convolve_static`
   with tests; 16 tests pass in the crate.

## Decisions already made

- Primary arch id **24**, MTP draft sidecar **25** — educated guesses, cheap to
  change; nothing has shipped on either.
- Quant focus **MQ4 (qt=44 MQ4G256V2)** and **MQ4L (qt=52 MQ4G256V2Lloyd)**;
  no new `qt`, no native NVFP4/MXFP8 runtime types.
- Phase-1 scope: text-only AR, `max_seq` 32768; vision/MTP/1M context deferred.
- PR only after implementation is verified.

## Test commands

```bash
cargo test -p hipfire-arch-inkling2
cargo test -p hipfire-quantize --bin hipfire-quantize dequant::tests
# real-checkpoint probes (needs the downloaded model):
cargo test -p hipfire-quantize --bin hipfire-quantize dequant::tests -- --ignored --nocapture
python3 scripts/check-attribution.py --range origin/beta..HEAD --require
python3 scripts/check-crate-maps.py --check
```

## Added after the initial log (same session)

- `82c28c81a`, `585ff6485` — **quantizer pipeline ingest** in
  `crates/hipfire-quantize/src/pipeline.rs`: index the 48 packed expert
  companion sets (`_scale`, `_amax`, `_outer_scale_denominator`), dequantize
  NVFP4/MXFP8 to F32, split each 3-D stacked tensor per expert, apply the w13
  gate/up regrouping, and emit per-expert `...experts.{E}.gate_up_proj` /
  `down_proj` MQ4G256 tensors. Also fixed a pre-existing MQ4V2 assertion for
  non-2-D tensors and the packed-vs-logical param accounting.
- `803d54324` — routing rows in `crates/hipfire-runtime/src/arch_mapping.rs`
  for `inkling2_loop`, `inkling2_loop_text` and the class name → id 24; reserved
  ids 24/25 in `docs/architecture-ids.md`.
- `7d469f132`, `6fade9c02` — CPU oracle additions: `relative_bias`, `situ`,
  `sigmoid_topk` (no renorm), `softcap`. Crate now has 20 passing tests.

**Smoke validation:** an end-to-end `--include-prefix model.llm.layers.0.`
quantization run completed and wrote a 556 MB HFQ to
`D:\hipfire-models\inkling2-mini-smoke.hfq` with per-expert MQ4G256 weights.
Run it with:
```bash
HIPFIRE_HOME="D:\hipfire-models\inkling2-home" \
  cargo run -q -p hipfire-quantize --bin hipfire-quantize -- \
  --input "D:\hipfire-models\Inkling-2-Mini" \
  --output "D:\hipfire-models\inkling2-mini-smoke.hfq" \
  --format mq4v2 --arch-id 24 --include-prefix "model.llm.layers.0."
```
(`HIPFIRE_HOME` isolates an unrelated stale config key in the default user
home.) **Full-model quantization completed** with the release binary:
`D:\hipfire-models\inkling2-mini-mq4.hfq` — 13.9 GB, arch id 24, **6742
tensors** (428 F16, 6144 per-expert MQ4G256, 170 MQ4G256V2). Log:
`D:\hipfire-models\inkling2-quant-full-release3.log`. The MTP sidecar is
excluded from the trunk. An independent Python header/index reader
(`C:\Users\Jan\AppData\Local\Temp\opencode\hfq_inspect.py`) confirms the arch
id, tensor count and per-expert names
(`...experts.{E}.gate_up_proj.weight` / `down_proj.weight`).

Fixes needed along the way (all committed): companion suffix is `_scale`
(not `_weight_scale`); MQ4V2 `m` must be 1 for non-2-D tensors; packed-vs-logical
param accounting; MXFP8 experts must be dequantized per 2-D slice; expert names
need `.weight`; `mtp.safetensors` must be excluded.

## Numeric oracle (2026-11-21)

A working reference oracle is built and producing golden outputs:

- **llama.cpp build**: `D:\hipfire-models\llama-inkling2-build\bin\llama-cli.exe`
  (build `b12600-29b52d398`, CPU backend), built from the merged PR #32694
  source at `C:\Users\Jan\AppData\Local\Temp\opencode\llama-inkling2`. CMake /
  Ninja came from VS BuildTools; RC / `mt.exe` from the Windows SDK
  (`...\Windows Kits\10\bin\10.0.26100.0\x64`) must be on PATH for linking.
- **Oracle model**: `D:\hipfire-models\Inkling-2-Mini\inkling2-f32.gguf`
  (104 GB F32), converted with `convert_hf_to_gguf.py` from the downloaded
  checkpoint using the merged converter.
- **Golden outputs**: `D:\hipfire-models\inkling2-golden\MANIFEST.txt` — four
  committed prompts with md5, decoded thinking text, and the exact flags
  (`-n 48 --temp 0 -ngl 0 --no-warmup -st`). Regenerate with
  `C:\Users\Jan\AppData\Local\Temp\opencode\make_golden_v2.sh`.
  The model is a reasoning model: output begins `[Start thinking]`. Note the
  48-token budget is mostly consumed by the thinking phase, so these are
  **thinking-phase** parity targets, not final answers.

This oracle is the numeric target for the CPU-forward port: future work should
compare per-visit hidden states / logits against it, not just final tokens.

## Research cross-check (2026-11-21, three parallel reports)

Read `inkling2-research-transformers.md`,
`inkling2-research-transformers-ninfer-all.md` and
`inkling2-research-transformers-vllm.md` in `docs/plans/local/`. All three
independently agree with the llama.cpp source I verified. Actions taken:

- **Bug fixed**: MoE top-8 sigmoid weights ARE renormed by their sum (eps
  `expert_weights_norm_eps = 1e-20`) then multiplied by `route_scale = 8`.
  Confirmed in authoritative `llama-graph.cpp::build_moe_ffn` lines
  2518-2543. `sigmoid_topk` and its tests corrected; added
  `attention_softmax_with_sink`. 23 oracle tests pass.
- **Sink-vs-bias equivalence confirmed**: llama.cpp subtracts `sinks` from
  the bias table before gather and passes a zero sink to `build_attn`;
  transformers concatenates `[scores, sink]` and softmaxes jointly. These are
  algebraically identical. My `relative_bias` (subtract before gather) +
  `attention_softmax_with_sink` (zero sink) implementation is correct.
- **Geometry confirmed**: 20 sliding + 4 full layers (full = 5, 11, 17, 23 —
  one per 6-layer AttnRes block); `log_scaling_n_floor = 8192` (checkpoint
  value wins over vLLM's 128000 default); 48 visits; 24 physical layers.
- **SiTU confirmed**: `mlp_gate_softcap=4` / `mlp_up_softcap=25` ARE the SiTU
  `beta` / `linear_beta`; my `situ` op matches.
- **Flag for the forward port**: transformers PR #51050 threads a
  `cache_offset` so each pass uses deep-copied per-pass KV + conv caches —
  hipfire must allocate **48** cache slots, not 24 (my `LoopConfig::n_slot()`
  already computes 48).

**Discrepancy to NOT "fix"**: one report claims llama.cpp has no
`[q[2i], kv[2i], kv[2i+1], q[2i+1]]` packing. That is only true of the
transformers *runtime* tensor. The llama.cpp **converter**
(`conversion/inkling2.py:modify_tensors`) does exactly that `torch.stack`
per core layer to produce each `loop_msg_query [n_embd,4]`. My plan §0a is
correct; leave it.

## CPU forward complete (2026-11-21, continued session)

The CPU reference forward is now **complete end-to-end** in `forward.rs`:
- `decode_layer` (SWA attention primitive), `decode_moe` / `decode_moe_lazy`
  (sigmoid top-8 + SiTU + fp32 combine), `decode_visit` (full layer: AttnRes
  mix, controllers, attention, MoE, convs, residuals), `pack_message_queries`,
  `forward_token` (all 48 visits with per-visit state, prelude/core×3/post,
  cross-pass messages from the previous pass's bank), `lm_head_logits`
  (GEMV + softcap 30 + slice to 200058).
- **Real-artifact verification**: full 48-visit forward + LM head on the 14 GB
  artifact produces **200058 finite logits** (argmax 196997, max 14.66) in a
  ~4-minute debug run. Lazy top-8 expert decode cuts expert decodes from 6144
  to ~384 per token (OOM before; now fine). F32 decode is cached in
  `weights.rs`.
- 26 unit tests + 6 ignored real-artifact tests (all verified manually).

**Parity investigation (2026-11-21, continued):** built a C API logit dumper
(`D:\hipfire-models\dump_logits.cpp/.exe`) against the merged llama.cpp and
compared oracle F32-GGUF logits for token 0 against the CPU forward on the MQ4
artifact (`compare_logits.py`). Result: r≈0.19, argmax disagrees — **parity not
achieved**. Root-cause analysis narrowed it down:
- NVFP4 dequant is exact vs the converter (diff 4e-10).
- qt44 (MQ4G256V2) f32 decode correlates **0.995** with the F32 GGUF but SQNR
  is only **20 dB** per weight (mse 4.3e-6, signal 4.3e-4) — meaningful loss.
- With discrete top-8 MoE routing, small router-logit changes flip expert
  selection, and errors compound across 48 visits. This plausibly explains a
  large logit divergence for **uniform, uncalibrated MQ4** — exactly the
  "quant→quant quality gap" risk in the plan. This is a hypothesis, not proven.
- Three genuine forward bugs were found and fixed during this: full-layer
  `static_w` conv flip/transpose, 3-row AttnRes mix for full layers, sink
  projection using inputs[0]; and RMSNorm raw-weight convention. None closed
  the gap alone.

To separate quantization from forward correctness: the forward is now generic
over `WeightSource` (`Inkling2Weights` for the MQ4 artifact, `GgufF32` for the
oracle's exact F32 weights). `GgufF32` reads from `D:\hipfire-models\gguf-f32.bin`
(a 98 GB flat export of the F32 GGUF made by `gguf_to_f32_bin.py`) on demand via
seek, mapping HFQ names/expert layout to GGUF names (expert-last axis, down
transposed). The `dump_raw_logits_token0` test runs the same 48-visit forward on
those F32 weights; compare its output to `oracle-logits-tok0.txt` with
`compare_logits.py` (point it at `cpu-raw-logits-tok0.txt`). If parity holds on
F32, the MQ4 gap is quantization/imatrix; if not, there are still forward bugs.

Remaining simplifications that could still bite: dynamic K/V conv taps on full
layers (static weights only), multi-token KV cache (single-token primitive).
The practical fix for the quality gap is imatrix-based MQ4L calibration, which
this CPU forward can now drive.

## GPU milestone (2026-11-21, continued session)

**arch 24 now loads on the real gfx1100 GPU.** Verified end-to-end with the
debug CLI + release daemon:
- `hipfire list` shows `inkling2-mini.mq4.hfq 13.93 GB`.
- `hipfire run inkling2-mini.mq4.hfq "hi" --max-tokens 1` loads the artifact on
  gfx1100 (25.8 GB VRAM), prints
  `hipfire load summary: inkling2-mini.mq4.hfq (inkling2)` and
  `[inkling2] 2304d 48L 200058 vocab`, then returns a clean fail-closed
  `[internal retryable=false] inkling2: generation is not implemented yet`.
- This followed the vendored carrier spec: `arch.rs` / `arch_model.rs` /
  `carrier.rs`, `Inkling2Carrier` in REGISTRY, `arch_label`, `bench_decode_route`,
  `reset_core` inventory, and a clean arch-24 guard at `ar.rs::generate`
  (replaces an LLaMA-downcast panic).

The GPU VRAM plan and kernel inventory research docs are vendored under
`docs/plans/inkling2-research/`.

## Parity bisect status (2026-11-21 — resolved)

The validated Python reference (vendored `ref-visits.md`) hits the llama.cpp
oracle at r=1.0000 and dumps per-visit hiddens + sub-components. Using it, the
Rust `decode_visit` was debugged **operator by operator** on the exact F32
weights via `GgufF32` (the `.bin` carries file-order dim labels but torch-order
bytes; the shim reverses dims at read).

**Result (F32 weights, token 0, against the llama.cpp oracle):**

- All **48 visits** match: `max|d| <= 1.8e-4`, `mean|d| <= 2e-5`, `corr 1.0000`
  (`compare_visits.py` over `D:\hipfire-models\rust-visits-tok0`).
- **Final hidden** (output AttnRes mix + final norm): `max|d| 1.4e-5`,
  `corr 1.0000000000` vs `ref-visits/tok0/final_hidden.npy`.
- **Token-0 logits: r = 0.9999999968**, mean|d| 4.2e-4, max|d| 1.6e-3,
  **argmax 31274 == oracle**. That is fp32 accumulation-order noise, nothing
  structural.
- **MQ4 artifact (the real quant gap):** r = 0.9907, mean|d| 0.330,
  **argmax 31274 == oracle**. Uniform MQ4 costs ~0.33 logit units at token 0
  and does not move the argmax — far better than the 20 dB-SQNR/routing-flip
  hypothesis predicted. MQ4L is therefore a *quality* track, not a
  correctness blocker.
- `dump_visits_f32`, `dump_raw_logits_token0` and `dump_cpu_logits_token0` are
  the commands; capture a fresh baseline before touching `forward.rs` again.
  `dump_cpu_logits_token0` needs `HIPFIRE_HOME=D:\hipfire-models\inkling2-home`
  (the default user home carries an unrelated unknown config key that
  `hipfire-config` refuses).

The earlier r≈0.19 was **not** quantization — it was four independent bugs in
the pass>0 / output path, and none of them was visible until a cross-pass
message is first applied at visit 18:

1. **Visit→physical-layer schedule** (the dominant one). `layer_phys` took
   llama.cpp's extended-layer index while its only caller passes the visit
   slot; on `usize` the shared-core visit 18 underflowed and ran layer 16, and
   the post layers 42–47 ran 12–17. `layer_phys` is now visit-indexed and
   `visit_schedule()` derives from it.
2. **Duplicated softmax state.** `states_for_mix` already held `prev_out`, and
   `mix()` appends its `cur` — so `prev_out` entered the message softmax twice.
   Signature: messages corr 0.987 (close, not wrong-looking).
3. **K/V message queries.** `GgufF32` classified the two cross-pass globals
   with `ends_with("_queries")`, true for both, so the kv global returned Q/FFN
   rows. Predicate is now `!ends_with("kv_queries")` (covers both checkpoint
   spellings).
4. **Silently skipped output mix.** `GgufF32` had no mapping for
   `model.llm.attn_res.output_query`, and `forward_token` wrote the final
   AttnRes mix as `if let Ok(...)`, so the LM head read an *unmixed* hidden and
   nothing failed. Cost 3.5 absolute on the final hidden (9% of its std) while
   every per-visit state looked perfect — the one bug the per-visit trace
   could not see. Fixed the mapping, and the final mix, the per-layer FFN mix
   and both L2-clip scales are now `?` (fail closed) instead of silently
   optional: upstream always applies them.

Earlier bugs found and fixed while getting visit 0/5 to the oracle (all still
required): per-head clip, GQA block grouping `h//3`, MoE residual base, final
output mix, conv tap order (tap `k-1` = current), AttnRes query off-by-one
(HFQ `attn_queries.0` feeds block 1, not 0), `decode_visit` loading conv
weights raw instead of via `load_conv`, full-layer dynamic K/V convs, and the
`static_w` double-flip between HFQ (verbatim, needs flip) and GgufF32 (already
converted). **29 unit tests pass.**

Debug tooling added along the way (all ignored tests, `--ignored`):
`dump_visits_f32` (per-visit trace), `dump_raw_logits_token0`,
`visit18_debug_dump` (any-visit component capture via `forward_token`'s
`debug` hook), `visit18_message_dump` / `visit18_control_dump` (rebuild one
message/control from the trace without a full forward), and
`cmp_visit.py` / `compare_visits.py` in the temp dir.

## Current end state (2026-11-21 — parity closed, GPU forward next)

**Branch** `feat/inkling2-mini-support` (fork `raufaser`), draft PR
https://github.com/warpfront/hipfire/pull/1192. Clean tree,
attribution + crate-map checks pass.

Done:
- Working quantizer + 13.9 GB arch-24 HFQ artifact (6790 tensors).
- Typed config, 24/48 loop schedule, validation traps, HFQ weight index.
- CPU reference forward: all 48 visits, controllers, cross-pass messages,
  dynamic convs, MoE, LM head.
- **Full forward parity on F32 weights**: all 48 visits ≤1.8e-4, final hidden
  1.4e-5, token-0 logits r=0.9999999968 with the oracle's argmax. The MQ4
  artifact is r=0.9907 with the same argmax, i.e. the quant gap at token 0 is
  small. See the parity section above.
- **Loader/carrier skeleton**: arch 24 loads on the RX 7900 XTX (gfx1100,
  25.8 GB VRAM) via `hipfire run`; `generate` returns a clean fail-closed
  "inkling2: generation is not implemented yet" (no panic).
- Research vendored under `docs/plans/inkling2-research/` (6 docs: 3 math
  cross-checks, tensor mapping, reference visits, calibration, carrier spec,
  VRAM plan, kernel inventory).

Not done: GPU forward/kernels, generation route (`GenerationRoute::InklingAr`),
multi-token KV cache, served validation, MQ4 forward parity.
**The model loads but cannot generate yet.**

**Artifact fix (2026-11-21):** the quantizer's generic `.scale`-sibling rule
had silently dropped the 48 L2-clip tensors (`q_l2_clip.scale` /
`rel_l2_clip.scale`) because they are bare learned parameters with no
`.weight` companion. Fixed: `.scale` is only treated as an FP8 sibling when a
real `<stem>.weight` exists. Re-quantized artifact now has 6790 tensors
(+48): 144 controllers, 2 cross-pass, 48 clips, 3072 gate_up + 3072 down.
The forward primitive needs these clip scales, so this was a real blocker.

A CPU forward **primitive** now exists (`forward.rs`): single-token SWA
attention using decoded f32 weights + oracle ops, with a unit test and a
real-layer-0 finite-output test. It is explicitly a scaffold — MoE, loop
schedule (messages/controllers/mix), full-layer dynamic convs, multi-token KV
cache and LM head are NOT yet implemented (tracked in the module doc).

Layer-type norm names in the loaded artifact (for the forward port): SWA
layers use `attn_norm.weight`; full layers (5, 11, 17, 23) use
`q_input_norm.weight` / `k_input_norm.weight` / `v_input_norm.weight`. Full
attention uses dynamic K/V convs (`k_sconv.static_w` + `k_sconv.weight_proj`);
SWA uses static `k_sconv.weight`. Per-expert tensors are
`...mlp.experts.{E}.{gate_up_proj,down_proj}.weight`.

## Next steps, in order (for the next session — split-friendly)

1. ~~**Re-measure full-model parity**~~ — **done**, see the parity section.
   F32 forward is clean through all 48 visits and the LM head.
   Remaining measurement: MQ4 (quantized artifact) forward parity, which is
   the quant-gap number, not a forward check.
2. **GPU forward** (largest slice, can be a separate worker): implement device
   weight upload + per-visit state (VRAM plan: 48 visit slots, conv width 6656,
   SWA-aware KV) and the 13 operators mapped in `inkling2-research/kernel-inventory.md`.
   Most reuse existing `rdna-compute` HIP kernels (rmsnorm, dynamic conv,
   attention sliding+sink, SiTU MoE, softcap); NEW are controller fuse, mix,
   rel-bias gather, L2 clip. Gate each operator against the CPU oracle
   (`forward_token` with the `debug` hook captures any visit's components).
3. **Generation route** (depends on 2): add `GenerationRoute::InklingAr` and a
   `generate_inkling` producer; remove the arch-24 guard in `ar.rs::generate`.
   No batch/slot/PP/EP/Redline in Phase 1.
4. **MQ4L calibration** if step 1 shows a quant gap: follow
   `inkling2-research/calibration-plan.md` (Lloyd needs no imatrix; imatrix only
   for AWQ/graded tiers; ≥12% E4M3 gain gate per family).
5. **Validation**: `serve_harness.py battery` + `chain`; capture a KLD/`.kldseq`
   pin; served A/B on gfx1100.
6. PR #1192 stays **draft** until 1–5 are done and verified.

## Split suggestion
Step 1 is self-contained and fast — do it first in any session. Steps 2 and 4
are independent research/implementation tracks that can run in parallel
workers, feeding results back via `docs/plans/local/`. Step 3 requires step 2.

## Parallel tracks in flight (branch base: `feat/inkling2-mini-support`)

None of these are merged into this worktree; they were built in isolation so
they could not disturb the parity bisect.

- **`hipfire-kernels` worktree** (`feat/inkling2-kernels` @ `6f08889`): the four
  genuinely NEW operators as new `rdna-compute` files — `inkling2_control`,
  `inkling2_mix`, `inkling2_rel_bias`, `inkling2_clip` — each with an F32
  reference, goldens, randomized differentials and property tests (22 pass,
  clippy clean). HIP bodies are inert source strings: no GPU execution was
  possible on that host. Integrate before the GPU forward, then gate each
  against `forward_token`'s `debug` capture.
- **MQ4L expert path** (`D:/mq4l`, `feat/mq4l-expert-path`): quantizer expert
  branch uses the Lloyd encoder + `lloyd_levels` sidecar + per-expert AWQ
  Route A; `quant_mq.rs` maps Inkling names to imatrix keys (zero misses on
  bartowski); family table now judges experts. Gate PASS: layer-0 smoke →
  `MQ4G256V2L:263 + F32:263`, `mlp_down` 14.16% / `mlp_gate` 14.13% (bar 12%).
  Its own caveat: **runtime decode for `MQ4G256V2L` is not wired** — an MQ4L
  artifact can be produced and gated but not served. 48 `bin`-test failures
  are pre-existing (diffed identical).
- **qt52 runtime decode** (handed to another agent, not started here): its
  blockers are concrete — `lloyd_lut::apply_lloyd_centering` refuses
  `K % 256 != 0` and the Inkling `down_proj` is K=1152; `expert_capabilities`
  has no qt52 expert kernel (`QT_MQ4_LLOYD` is a reserved tag); and
  `hipfire-quantize/src/dequant.rs` has no qt52 arm, so the CPU forward cannot
  yet read an MQ4L artifact. Not on the critical path until the GPU forward
  exists.

## GPU forward — concrete implementation brief (next slice)

Everything below is verified against the tree; it is the structure the next
session should build, in this order. Do NOT land a partial version of this:
1 and 2 are only useful together, and a half-built bundle was already thrown
away once.

**New file `crates/hipfire-arch-inkling2/src/gpu.rs`**, wired from
`arch.rs::load_weights` (upload) and `arch_model.rs::free_gpu` (free every
allocation; cf. PR #566).

Per physical layer (×24), following `hipfire-arch-cohere2moe/src/cohere2moe.rs`
(`Cohere2MoeLayerWeights` + `wt_from_raw` at `:224`):

- attention projections: `wq_du` [3072,2304], `wk_dv`/`wv_dv` [1024,2304],
  `wr_du` [384,2304], `gate_attn` [3072,2304], `wo_ud` [2304,3072]
- `attn_sink_proj` [24,2304]; `rel_logits_proj.proj` (F16 table, forward wants
  `[n_rel=16, n_extent=1024]`); `q_l2_clip.scale` [3072]; `rel_l2_clip.scale` [384]
- norms: SWA layers `attn_norm` [2304]; full layers `q/k/v_input_norm`; all
  layers `mlp_norm`
- convs: `k_sconv`/`v_sconv` static `[C,4]` (+ `weight_proj` on the 4 full
  layers), `attn_sconv`, `mlp_sconv`
- controllers ×4 (`attn`,`k`,`v`,`mlp`) × (`down` [64,4608], `gain`/`forget`
  [2304,64]) — F16 keep
- `attn_res_query` ([2304] SWA / [3,2304] full), `ffn_res_query` [2304]
- MoE: router `mlp.gate.weight` [128,2304], 128 × (`gate_up` [2304,2304] +
  `down` [1152,2304]) MQ4/MQ4V2, plus the `[128]` device pointer arrays the
  grouped path wants (`MoeFfn` precedent)

Globals: `embed`, `embed_norm`, `norm`, `unembed`, `attn_res.output_query`,
`middle_cycle_cross_pass_queries`/`_kv_queries` (or the 12 per-layer
`loop_msg_query` tensors), 200058-slice on the head.

**Per-visit state, 48 slots** (VRAM plan: q8 KV 0.61 GiB at 32768, conv 3.8 MB,
weights 13.1 GiB, total ≈14.5 GiB on 24 GiB):
- KV: SWA visits keep 1024 rows, full visits 32768 (slot count from
  `vram-plan.md` §2); allocate by mask, never 48×32768
- unified conv state per visit: width `2·1024 + 2·2304 = 6656`, 3 taps
- keyed by **visit**, not layer — an ×24 allocation silently aliases passes

**Operator → existing kernel** (from `inkling2-research/kernel-inventory.md`,
with the routing row corrected): rmsnorm `norm.rs:420`; linear `weight_gemv`;
conv `norm.rs:5461/9211/9470` (drop the SiLU); attention
`attention.rs:16769` + the DS4 sink idiom; SiTU `ffn_activation.rs` `Situ`;
softcap `norm.rs:8898`. The four NEW ones are the kernels worktree files
(`hipfire-kernels`: `inkling2_control.rs`, `inkling2_mix.rs`,
`inkling2_rel_bias.rs`, `inkling2_clip.rs` — uncommitted there, F32 reference +
inert HIP source + 22 tests): integrate them as the device implementations land.

**Gating order** (each step is a checkpoint, not a guess):
1. upload + free smoke: load arch 24 and return VRAM to baseline on unload;
2. one visit (phys 6, pass 0) vs the CPU oracle — `forward_token`'s `debug`
   hook gives every component of that visit for the diff;
3. all 48 visits vs the CPU trace (`visit_out` per visit);
4. token-0 logits vs `D:\hipfire-models\oracle-logits-tok0.txt`
   (bar: r > 0.999, argmax match, as the F32 CPU path now achieves);
5. only then `GenerationRoute::InklingAr` + `generate_inkling`, and remove the
   arch-24 guard in `ar.rs::generate`.

## Known issues / risks

- Quant→quant: no original pre-quant BF16 checkpoint exists; the published BF16
  GGUF is a dequant oracle, not a training source. Expect a quality gap on the
  1152-wide experts.
- `pre-commit-ci.sh origin/beta` fails on this Windows host for pre-existing
  reasons (cp1252 decoding, missing `os.killpg`, no MSVC `cl`, POSIX-only crates
  `hipfire-xdna`/`redline`, rustfmt path-length). These reproduce on a clean
  base worktree; re-run on Linux CI before any PR.
- The dequant probes cover one expert each of NVFP4 and MXFP8, not every tensor.
- `hf_quant_config.json` advertises MXFP8 `[32,32]` while the stored block is
  `(1,32)`; trust the stored tensor/scale shapes (the converter only warns).
