# gfx1201 Qwen3.8 multi-row verifier on the composed v0.3.1 beta head — 2026-09-13

**Lifecycle:** `historical`

**Disposition:** maintainer re-measurement of the #748 route on the composed
beta head (after #755 sealed MoE landed), with an AR anchor the original
checkpoint lacked. Evidence under this fixture only; not a product baseline,
admission, or `docs/BENCHMARKS.md` claim.

Amends (does not modify)
[`2026-09-11-gfx1201-qwen38-multirow-verifier.md`](2026-09-11-gfx1201-qwen38-multirow-verifier.md).

## Fixture identity

- Host: 4× AMD Radeon AI PRO R9700, exact `gfx1201`, HIP `7.15`; one GPU used
  (`HIP_VISIBLE_DEVICES=0`), held under `flock /tmp/hipfire-gpu.lock`.
- Tree: beta merge head `03b55ef40` (#755 `d9b05c454` + #748 + the
  retained/PM4 recording exclusion), built from source in
  `/home/kaden/.omp/wt/pr748-review`.
- `hipfire` MD5 `6f2676b4e60ce732502677687366b80a`; daemon MD5
  `202d02d054f51125aee6e6ed4d8afb4a`.
- Target: `qwen3.8-27b.mq4-xt`; SHA-256
  `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`;
  MD5 `e45d15bfe0c9a87132697101d17cbed6`.
- Draft: `qwen3.8-27b-dflash.mq4v2.hfq` (registry `qwen38-27b-dflash-mq4.hfq`
  identity); SHA-256
  `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`;
  MD5 `013395583cd04206c8aa68f4d061983d`. Loaded windowed at its declared
  W=2048, all layers sliding (`HIPFIRE_DFLASH_WINDOW` / `HIPFIRE_DFLASH_CTX_CAP`
  unset).
- Prompt: `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`; MD5
  `b4d0b63cddcac872648ddf3cdd92cac2`; 21,550 tokens after the chat scaffold.

## Method

`hipfire bench <target> --backend noslots --workload stateless --runs 1
--max-tokens 200 --prompt-file <prompt> --kv-mode q8 --kv-backend vmm --json`,
one fresh process per sample, `memory.max_seq=65536`, `HIPFIRE_VERIFY_GRAPH=0`,
greedy, isolated `HIPFIRE_HOME`, parent `HIPFIRE_*` scrubbed. One unrecorded
warmup, then declared order `ar, off, on, on, off, off, on, ar, ar`. Arms:
`ar` = `--spec off`; `off` = `--spec dflash` +
`HIPFIRE_FA_PERTOKEN_MIN_CTX=0`; `on` = `--spec dflash` + binary default
(`4096`). Every DFlash sample was gated on `drafter=dflash`, τ > 1, the
windowed-load line, and absence of any AR-fallback marker.

## Result

| arm | decode samples (tok/s) | median | τ / cycles |
|---|---|---:|---|
| AR, no draft | 32.6, 32.6, 32.6 | 32.6 | — |
| DFlash, established batched route | 26.3, 26.3, 26.3 | 26.3 | 1.80 / 71 |
| DFlash, R4/R8 multi-row route | 39.3, 39.3, 39.3 | 39.3 | 1.80 / 71 |

- Multi-row vs batched: +49.4%. Identical to the pre-#755 measurement of the
  same binary lineage on 2026-09-13 (`/tmp/pr748-bench-home/results/20260913T070753Z`,
  same three medians), so #755 did not move this route.
- **AR anchor:** at 21,550 tokens the established batched DFlash route is a
  19.3% loss against plain AR. The multi-row route turns long-context DFlash
  on gfx1201 into a +20.6% net win over AR. The original checkpoint's +37.6%
  compares against a baseline that is slower than not speculating.
- Identical samples within an arm reflect `hipfire bench` rounding to 0.1 tok/s
  under deterministic greedy decode with graph capture off; τ and cycle counts
  match the original checkpoint exactly.

Coherence (VALIDATION.md serve route, route on, pre-#755 head `5c6db516b`):
`serve_harness.py --mode session` on `benchmarks/prompts/session_coding.json`
(MD5 `c0d470288bde3f1e54e4bba04da8f8a2`), `max_seq=131072`, registry sampling,
seed 7: 8/8 turns `stop`, contexts 7.1K→36.9K with prefix-cache hits on every
turn, zero empty/runaway/attractor, decoded text read. The harness's literal
`dedupe` retrieval substring was absent on two turns (model named its
functions `hash_file`/`find_duplicates`); not a coherence failure.

Not claimed: Redline/PM4 parity (route is excluded under recording), any
other architecture, KV mode, or head dimension.
