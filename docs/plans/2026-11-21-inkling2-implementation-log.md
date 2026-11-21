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

## Parity bisect status (2026-11-21 — updated at intermediate stop)

The validated Python reference (vendored `ref-visits.md`) hits the llama.cpp
oracle at r=1.0000 and dumps per-visit hiddens + sub-components. Using it, the
Rust `decode_visit` was debugged **operator by operator** on the exact F32
weights via `GgufF32` (the `.bin` carries file-order dim labels but torch-order
bytes; the shim reverses dims at read). Latest measured result:

- **visit 0** (SWA): every intermediate (`x_in`, `q_inp`, `k/v_inp`, `q/k/v/r_raw`,
  `k/v_conv`, `q/k/v/r_clip`, `vc`, `sinks`, `rel_bias`, `attn_heads`, `wo_out`,
  `cur_attn`, `ffn_mix`, `ffn_normed`, `router_logits`, selected experts,
  `moe_raw`, `moe_conv`, `visit_out`) matches the reference to **~1e-5**.
- **visit 5** (first full-attention layer, dynamic conv): every intermediate
  matches to **~1e-5** after the last fix (`D:\hipfire-models\ref-visits\tok0\`
  has `ref_visit5_*.txt` and `rust_visit5_*.txt`; `cmp_visit5.py` diffs them).
- Bugs found and fixed this session: per-head clip, GQA block grouping `h//3`,
  MoE residual base, final output mix, conv tap order (tap `k-1` = current),
  AttnRes query off-by-one (HFQ `attn_queries.0` feeds block 1, not 0),
  `decode_visit` loading conv weights raw instead of via `load_conv`, full-layer
  dynamic K/V convs, and the `static_w` double-flip between HFQ (verbatim,
  needs flip) and GgufF32 (already converted). **28 unit tests pass.**

**Not yet re-measured after the visit-5 fix**: the full 48-visit trace. The last
`dump_visits_f32` run was interrupted while capturing; its late-visit numbers
are stale and must not be quoted. Re-run `dump_visits_f32` +
`compare_visits.py` to get the true full-model parity curve. Full-logit parity
on the MQ4 artifact is still expected to show the quantization gap.

## Current end state (2026-11-21 — intermediate stop)

**Branch** `feat/inkling2-mini-support` (fork `raufaser`), draft PR
https://github.com/warpfront/hipfire/pull/1192. ~50 local commits, clean tree,
attribution + crate-map checks pass.

Done:
- Working quantizer + 13.9 GB arch-24 HFQ artifact (6790 tensors).
- Typed config, 24/48 loop schedule, validation traps, HFQ weight index.
- CPU reference forward: all 48 visits, controllers, cross-pass messages,
  dynamic convs, MoE, LM head. Visit 0 and visit 5 verified to ~1e-5 against
  the validated oracle; full-model re-measurement pending (above).
- **Loader/carrier skeleton**: arch 24 loads on the RX 7900 XTX (gfx1100,
  25.8 GB VRAM) via `hipfire run`; `generate` returns a clean fail-closed
  "inkling2: generation is not implemented yet" (no panic).
- Research vendored under `docs/plans/inkling2-research/` (6 docs: 3 math
  cross-checks, tensor mapping, reference visits, calibration, carrier spec,
  VRAM plan, kernel inventory).

Not done: GPU forward/kernels, generation route (`GenerationRoute::InklingAr`),
multi-token KV cache, full-model parity re-measurement, served validation.
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

1. **Re-measure full-model parity** (quick, unblocks everything): run
   `cargo test -p hipfire-arch-inkling2 dump_visits_f32 -- --ignored` then
   `python3 <temp>/compare_visits.py`. Visit 0 and 5 already match to ~1e-5;
   expect the whole curve near 1.0 on the F32 path. Then run `dump_raw_logits_token0`
   and compare to `oracle-logits-tok0.txt` — this finally separates forward
   correctness from MQ4 quantization loss.
2. **GPU forward** (largest slice, can be a separate worker): implement device
   weight upload + per-visit state (VRAM plan: 48 visit slots, conv width 6656,
   SWA-aware KV) and the 13 operators mapped in `inkling2-research/kernel-inventory.md`.
   Most reuse existing `rdna-compute` HIP kernels (rmsnorm, dynamic conv,
   attention sliding+sink, SiTU MoE, softcap); NEW are controller fuse, mix,
   rel-bias gather, L2 clip. Gate each operator against the CPU oracle.
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
