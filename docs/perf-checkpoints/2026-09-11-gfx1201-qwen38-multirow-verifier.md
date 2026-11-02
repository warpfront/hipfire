# gfx1201 Qwen3.8 multi-row verifier — 2026-09-11

**Lifecycle:** `historical`

**Disposition:** measured gfx1201 extension evidence; not a product baseline or
admission decision.

## Question and scope

Test whether the exact-gfx1100 R4/R8 Q8 verifier route landed through
AlpineQ's #741 can be admitted unchanged on exact `gfx1201`. This candidate
changes only the Rust admission and launcher predicates, their tests, and the
environment-variable documentation. It does not modify the R4/R8 kernel
source, arithmetic, partials layout, reducer, or its existing AlpineQ/Kaden
attribution.

Current-beta base: `c88a1ba0dc3fb2104ab78e6cb65b3067973e41ac`.

## Fixture identity

- Host: `X570`, AMD Radeon AI PRO R9700, exact `gfx1201`, wave32, 34.2 GB
  reported VRAM, HIP `7.14`.
- Target: `qwen3.8-27b.mq4-xt`; SHA-256
  `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`;
  MD5 `e45d15bfe0c9a87132697101d17cbed6`.
- Draft: `qwen38-27b-dflash-mq4.hfq`; SHA-256
  `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`;
  MD5 `013395583cd04206c8aa68f4d061983d`.
- Prompt: `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`; MD5
  `b4d0b63cddcac872648ddf3cdd92cac2`; 21,550 tokens after the Qwen chat
  scaffold.
- Tested `hipfire` MD5: `a93fd1cef69d2a6158ed7af69c07c675`.
- Tested daemon MD5: `a9ed52a245c3f8531fdddd8e207e6216`.
- Dedicated micro-harness MD5: `a20c510dbe0d68a3084dc90f0c1e76b4`.

## Product-path A/B

One unrecorded warmup preceded six native-serve fresh processes in declared
order `off,on,on,off,off,on`. Both arms used `max_seq=65536`, Q8 VMM KV,
DFlash, greedy sampling, thinking disabled, `HIPFIRE_VERIFY_GRAPH=0`, a
ten-second DPM warmup, and 200 generated tokens. The only arm difference was
`HIPFIRE_FA_PERTOKEN_MIN_CTX=0` versus `4096`.

| route | decode samples (tok/s) | median | delta |
|---|---|---:|---:|
| established batched | 24.6, 23.7, 23.3 | 23.7 | — |
| multi-row R4/R8 | 32.6, 32.6, 32.5 | 32.6 | +37.6% |

Every recorded process emitted 200 tokens in 71 cycles with `tau=1.80`,
reported request-level `drafter=dflash`, and ended at the intentional length
cap. All six decoded outputs were byte-identical, MD5
`b501ab0e0102889bd63537f2006d4f61`; none was empty or an attractor. The text
was manually inspected and was a coherent, incomplete-at-the-cap description
of the prompt's codebase.

## Kernel screen

The dedicated oracle used 100 timed iterations per cell, 24 query heads, four
KV heads, tile 128, and compared R4/R8 against
`attention_flash_q8_0_tile_batched`.

| head dim | context | R4 speedup | R8 speedup | worst relative error |
|---:|---:|---:|---:|---:|
| 256 | 2,048 | 1.24x | 1.36x | 3.276e-7 |
| 256 | 4,096 | 1.54x | 1.80x | 2.911e-7 |
| 256 | 20,676 | 1.78x | 2.14x | 3.378e-7 |
| 256 | 32,768 | 2.08x | 2.49x | 4.222e-7 |
| 128 | 4,096 | 1.55x | 1.77x | 4.765e-7 |
| 128 | 20,676 | 1.91x | 2.27x | 3.410e-7 |
| 128 | 32,768 | 1.93x | 2.34x | 4.042e-7 |

The worst relative error across the matrix was `4.765e-7`, against the
oracle's `1e-3` limit. The existing conservative `>4096` production threshold
was retained rather than tuned as part of this architecture extension.

Radiowave/clang resource reporting on gfx1201:

| entry point | VGPR | SGPR | private/scratch | occupancy (waves/SIMD) |
|---|---:|---:|---:|---:|
| R4, head dim 128 | 67 | 46 | 0 | 16 |
| R8, head dim 128 | 116 | 63 | 0 | 12 |
| R4, head dim 256 | 104 | 44 | 0 | 12 |
| R8, head dim 256 | 185 | 61 | 0 | 8 |

## Correctness and route validation

- `test_kernels`: 16 passed, 0 failed, 0 skipped on gfx1201.
- `test_kernelsQA --expected-arch gfx1201`: 16 passed, 0 failed, 0 skipped.
- `hipfire-arch-qwen35` unit tests: 193 passed, 4 ignored.
- `rdna-compute` unit tests: 244 passed.
- Native `serve_harness.py battery`: five of five prompts ended normally with
  coherent decoded code, reasoning, factual, prose, and instruction answers;
  no empty, runaway, attractor, or retrieval failures.
- A resident-daemon LongBench-v2 hard-30 soak used the pinned dataset SHA-256
  `839a19be0b3b1c801a0ca58d388996faff34704e34dd196d54346adf17a1dce9`,
  its `prompt_think` variant (20,462--30,258 input tokens), Q8 VMM KV,
  DFlash window 2,048, `reasoning_effort=xhigh`, temperature 1.0, and a
  65,536-token output ceiling. All 30 requests stopped naturally with zero
  runtime errors, empty outputs, open-think terminals, or attractors; the
  longest output was 14,940 tokens. The model scored 17/30 (quality context,
  not a performance or parity claim). Median daemon-reported prefill/decode
  rates were 465.8/59.1 tok/s and median DFlash tau was 3.71. The tested
  daemon SHA-256 was
  `b5a3e3c785ec530914b45e028748910afab022eeac261aee8b1ef2db4efc6d77`;
  corrected raw-result SHA-256 was
  `a21c6020f9726e00adfe1c699b6f1a0f3e4d553277c4adec8336ec2d60c6e01a`.
  Ordinal 12 was rerun from its exact recorded prompt MD5
  `d56605216e66071163ba70f8f08118b0` after the scratch analyzer was fixed to
  preserve a legal U+2028 inside the JSON record; the pinned dataset itself
  was byte-correct and unchanged.
- Crate-map generation, environment-document scan, changed-file formatting,
  serve-harness self-test, and diff whitespace checks passed.

The full battery's first short-context request paid cold kernel setup and is
not a performance claim. The long-context A/B above is the claim-bearing
product-path evidence.

The multi-row route remains excluded during graph capture and retained/PM4
replay, so this checkpoint makes no Redline performance or parity claim. Other
architectures, KV formats, head dimensions, tree/independent batches, and
unsupported row counts continue to fail closed to the established batched
route.
