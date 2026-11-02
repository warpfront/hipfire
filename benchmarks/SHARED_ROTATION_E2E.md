# Final shared-rotation E2E validation

Production source changes are limited to the configuration schema, FeatureFlags, and DFlash projection dispatch. All three routes default on, with independent opt-outs: HIPFIRE_DRAFT_SHARED_QKV_ROTATION=0, HIPFIRE_DRAFT_SHARED_FFN_ROTATION=0, and HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION=0. Eligibility remains restricted to gfx1100, MQ4V2, compatible scratch, token-row batch 2-16, and no AWQ sidecar or capture/replay recording. Other calls preserve the existing dispatch. No target-model changes, new kernels, weight formats, KV allocations, profile timers, hash readbacks, or failed attention probes are included.

A disables all three shared rotations; B enables all three. Each of the five canonical long-decode prompts runs two ABBA blocks (eight complete measured generations). Each process warms up twice, emits up to 8192 tokens with context limit 16384, then idles ten seconds. Target is Qwen3.8-27B MQ4 with the existing DFlash draft and q8 KV, on GPU1 via ROCR_VISIBLE_DEVICES=1.

Primary metric is the native bench decode_tok_s from an uninstrumented daemon. Report median and min/max for four A and four B runs, plus text hash, tau and cycle parity. A text/tau/cycle mismatch must be discussed before attributing E2E changes to the optimization. Token collection and SHA256 computation occur in the external benchmark harness, not inside GPU forward execution.

The CLI harness is the rebuilt research CLI (same config schema, saved output and two-warmup support); the daemon is compiled from this clean final worktree and isolated at /tmp/hipfire-q38-dflash-final-20261009-artifact/daemon. Source diff and both binary hashes are saved per experiment. Do not rebuild the CLI during the run. Initial launch directories without complete results reflect CLI artifact/version skew and are excluded from statistics.

Command:

```bash
python3 scripts/bench_dflash_shared_rotation_e2e.py \
  --cli ../hipfire-q38-dflash-20261007/target/release/hipfire \
  --daemon /tmp/hipfire-q38-dflash-final-20261009-artifact/daemon \
  --cases benchmarks/gemma4_eseries_long_decode.json \
  --out benchmarks/results/qwen38-shared-rotation-final-e2e-v3-abba-20261009
```

The command above requires the research CLI benchmark harness, not an unmodified upstream CLI. Its host-only patch is archived as benchmarks/shared_rotation_bench_harness.patch against beta 90d90306d for reproduction in a separate checkout; it is not applied to this PR's production CLI. The daemon has no added profiling instrumentation. The 40-run ABBA evidence was collected against beta 90d90306d, before rebasing the production changes onto 740818f37.

| Scenario | Output tokens | A decode tok/s median | B decode tok/s median | Delta |
|---|---:|---:|---:|---:|
| KV-cache guide | 3257 | 61.50 | 61.85 | +0.57% |
| SGLang guide | 2762 | 59.45 | 59.60 | +0.25% |
| Story | 1753 | 30.40 | 30.50 | +0.33% |
| Snake Python | 2100 | 119.95 | 120.20 | +0.21% |
| Tetris Python | 5566 | 113.85 | 114.15 | +0.26% |

All 40 generations finished normally; output hashes, tau and cycle counts matched within each scenario. These small E2E differences overlap run variation and are not evidence of a statistically significant serving speedup.

Separate GPU-event projection-group measurements (rotation plus existing GEMMs, not individual GEMM timing) showed noise QKV 253.08 -> 193.21 us (1.31x), gate/up 332.69 -> 304.53 us (1.09x), and incremental context KV 1.246x-1.251x across the five scenarios. Shared inputs reduce rotation counts from three to one for QKV and two to one for gate/up and context KV; weight GEMMs remain unchanged.

After rebasing onto beta 740818f37, the official serving battery passed five cases with omitted flags and five with explicit flags set to 1; compared outputs, finish reasons, generated lengths and tau matched. Reproduce this default-on check using the unmodified upstream CLI/daemon built from this branch:

```bash
python3 scripts/verify_dflash_shared_rotation_default.py \
  --out benchmarks/results/qwen38-shared-rotation-default-check
```

Successful raw evidence is in qwen38-shared-rotation-final-e2e-v3-abba-20261009 and qwen38-shared-rotation-default-v2-battery-20261009. Their metadata records artifact hashes and tested revisions. Failed preflight/artifact-skew attempts are not part of the submitted evidence.
