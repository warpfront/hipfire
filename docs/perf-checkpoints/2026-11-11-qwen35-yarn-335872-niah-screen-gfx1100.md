# 2026-11-11 — Qwen3.5/3.8 YaRN 335872 long-context load + NIAH screen (gfx1100)

**Lifecycle:** `historical` — fixture-bound measurement, not a current default,
baseline, or admission decision.

## Fixture

- GPU: gfx1100 (7900 XTX, 24 GB), Windows host, ROCm 10.1 / HIP 7.16
- Model: `qwen3.8-27b.mq3-pro` (13,184,433,152 B, registry
  `qwen3.8:27b-mq3-pro`; local md5 `b171dd618eecf1fe6aed2c1dc5eef4dc`)
- Binary: `feat/qwen35-yarn-rope` (hipfire.exe md5
  `ef1695b75976b74a1abad320cddb8d3c`, daemon.exe md5
  `a98b13715362fc4ce5bbb283a85f3622`); the NIAH runs used the
  pre-refusal-fix build (same kernel/config bytes; the fix only adds a
  load-time refusal that the AR-only NIAH path never reaches)
- YaRN: `HIPFIRE_YARN_FACTOR=1.28125` → context limit
  262144 × 1.28125 = 335872
- KV: q8/q8, legacy backend (automatic; Windows VMM not certified)
- Prompt: `.codeinsight+research/niah_prompt.txt` (1,385,632 chars,
  **290,996 prompt tokens** per `hipfire bench --json`, md5
  `15d5655796496208f668ae42e4c0dd1c`); needle
  `ZEBRA-42-QUARTZ` at the 95% text mark (~276K tokens, past the 262144
  native limit)

## Measured

- Load: 7.4 s, own 25,505 MiB (vs 22,919 MiB at native 262144 — the KV
  cache sized for 335872 rows), `max_seq: 335872 (bound=user)`, no
  admission error.
- `hipfire bench` (noslots, stateless, 1 run, 0 warmups, max-tokens 40):
  prefill 197.8 tok/s (ttft 1,471.4 s over 290,996 tokens), decode
  5.8 tok/s, `spec_route: null` (spec decode refused under YaRN — v1
  boundary, AR only).
- Serve path `/v1/chat/completions` (greedy, `temperature 0.0`,
  max-tokens 40, no thinking): answer `ZEBRA-42-QUARTZ` — the exact
  needle string embedded at ~276K tokens, past the 262144 native limit.
  Elapsed 1,486 s (prefill-dominated).

## Not measured

- KLD vs BF16 teacher past the native context (requires the pinned
  teacher fixtures at >262K rows — out of scope for this screen).
- Any perf comparison against the native-262K incumbent (YaRN rescales at
  all positions; the two are different routes by design).

## Disposition

Screen evidence that the YaRN-extended load, KV sizing, 291K-token prefill
and AR decode path work end to end on gfx1100. Not a quality claim for
>262K positions; not a perf claim (single run, no interleaving, no CIs).
