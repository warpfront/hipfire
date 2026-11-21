# Inkling-2-Mini implementation log (2026-11-21)

Working branch: `feat/inkling2-mini-support` off `origin/beta` (`9e410448e`),
in worktree `C:\Users\Jan\AppData\Local\Temp\opencode\hipfire-inkling2`.
**No PR is open** — PR is the goal only after the implementation is done and
verified. All commits carry an `Agent: hy4-flash-free/<seat>` trailer;
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

## Current end state (2026-11-21)

Done: a working quantizer path and a 13.9 GB arch-24 HFQ artifact;
typed config parsing; a 21-test CPU-oracle crate; a built reference oracle with
reproducible golden outputs. Not done: the loader bundle, GPU forward, and
generation route — the artifact cannot be served yet.

## Next steps, in order

1. **DONE** — NVFP4/MXFP8 quantizer ingest (see above).
2. **DONE** — CPU oracle scalar/operator set: `rmsnorm`, `clip`, `tau`,
   `matvec`, `control`, `mix`, `convolve_static`, `relative_bias`, `situ`,
   `sigmoid_topk`, `softcap` (21 tests). Not yet covered: full attention with
   softmax/sink, the dynamic-conv tap path, and the MoE expert combine.
3. **Arch crate wiring**: `arch.rs`, `arch_model.rs`, `carrier.rs`, `forward.rs`
   mirroring `hipfire-arch-toy`; register `arch_mapping.rs` rows for
   `inkling2_loop` / `inkling2_loop_text` → 24 and the class-substring row;
   add `InklingCarrier` to `hipfire-loader/src/lib.rs::REGISTRY`.
4. **Generation route**: `GenerationRoute::InklingAr` in `hipfire-generate/src/ar.rs`
   and `generate_inkling` in `dense.rs`. No batch/slot/PP/EP/Redline in Phase 1.
5. **Tokenizer/template**: load the 201k BPE; port `chat_template.jinja`
   (`reasoning_effort` default 0.9, strict canonical tool JSON).
6. **Validation**: `serve_harness.py battery` and `chain` on committed prompts;
   capture a KLD/`.kldseq` pin for the text-only route; GPU smoke on gfx1100.
7. Only then consider the upstream PR.

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
