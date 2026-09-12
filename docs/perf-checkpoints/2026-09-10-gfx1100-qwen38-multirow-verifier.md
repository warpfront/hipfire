# gfx1100 Qwen3.8 multi-row verifier — 2026-09-10

**Lifecycle:** `historical`

**Disposition:** measured candidate evidence; not a product baseline or admission decision.

## Question and scope

Measure a gfx1100-only Q8 flash-attention kernel in which one wave owns four
or eight verifier rows and shares each KV scan across those rows. Only the
attention step changes; projections remain batched. The production admission
predicate is exact `gfx1100`, Q8 KV, head dimension 128 or 256, sequential
non-tree batches of 4–32 rows, logical context above 4096, and graph capture
off. Unsupported shapes retain the established batched route.

Current-beta base: `b8092f7c7fe0eb3dabccc28e8993ee10c3465fc6`.

## Fixture identity

- Host: `odin`, Radeon RX 7900 XTX, `gfx1100`, wave32, 24,560 MiB reported
  VRAM, HIP `7.2.53211-9999`.
- Target: `qwen3.8-27b.mq4-xt`, 14,980,361,216 bytes; SHA-256
  `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`;
  MD5 `e45d15bfe0c9a87132697101d17cbed6`.
- Draft: `qwen38-27b-dflash-mq4.hfq`, 1,209,603,072 bytes; SHA-256
  `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`;
  MD5 `013395583cd04206c8aa68f4d061983d`.
- Prompt: `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`, 75,251
  bytes; MD5 `b4d0b63cddcac872648ddf3cdd92cac2`; 21,550 tokens after the
  Qwen chat scaffold.
- Tested daemon MD5: `512fccca7c7189559048a7aba17cb6c1` (SHA-256
  `4775d1225c71db6d8c1b717e59a62ff1145f74f9ee433462f8376db99f19bfb3`).
- Dedicated micro-harness MD5: `0a032cfcbf525c9fa3565a7b55ca6d88`.

## Product-path A/B

Six native-daemon fresh processes ran in declared order
`off,on,on,off,off,on`. Both arms used `max_seq=65536`, Q8 KV, DFlash,
greedy sampling, `HIPFIRE_VERIFY_GRAPH=0`, a ten-second DPM warmup, and 200
generated tokens. The only arm difference was
`HIPFIRE_FA_PERTOKEN_MIN_CTX=0` versus `4096`. A separate unrecorded candidate
probe populated the shared JIT cache before the series.

| route | decode samples (tok/s) | median | delta |
|---|---|---:|---:|
| established batched | 33.4, 32.6, 33.4 | 33.4 | — |
| multi-row R4/R8 | 46.4, 42.9, 46.3 | 46.3 | +38.6% |

All six samples produced 200 tokens in 69 cycles with `tau=1.88`, no daemon
errors, and byte-identical decoded output MD5
`b501ab0e0102889bd63537f2006d4f61`. A separately built, warmed current-beta
daemon produced 33.5 tok/s with the same token count, cycles, tau, and output
MD5. The first baseline invocation was discarded as cold-JIT (11.2 tok/s).

Raw local discovery artifact: `/mnt/data4/claude-scratch/20260831-hipfire-tp2/evidence-beta/xt-ab/results.jsonl`,
MD5 `1b037e814fd00caae4b9ca6d919f3a30`.

## Kernel screen

The dedicated oracle used the Qwen3.8-27B shape (24 query heads, four KV
heads, head dimension 256), 100 timed iterations per cell, and compared every
output against `attention_flash_q8_0_tile_batched`.

| context | R4 speedup | R8 speedup |
|---:|---:|---:|
| 2,048 | 0.98x | 0.72x |
| 4,096 | 1.45x | 2.01x |
| 20,676 | 1.99x | 2.40x |
| 32,768 | 1.94x | 2.28x |

Worst relative output error was `4.222e-7` against a `1e-3` limit. The
head-dimension-128 screen at context 8192 measured 1.33x for R4 and 2.03x for
R8, with worst relative error `3.419e-7`.

Radiowave inspection for head dimension 256 reported:

| entry point | VGPR | SGPR | VGPR spills | SGPR spills | private bytes |
|---|---:|---:|---:|---:|---:|
| `attention_flash_q8_0_rows4_d8` | 106 | 41 | 0 | 0 | 0 |
| `attention_flash_q8_0_rows8_d8` | 186 | 58 | 0 | 0 | 0 |

## Correctness and route validation

- `test_kernels`: 16 passed, 0 failed, 0 skipped on the RX 7900 XTX.
- `hipfire-arch-qwen35` unit tests: 193 passed, 4 ignored.
- `rdna-compute` unit tests: 242 passed.
- Canonical-XT serve battery: five of five coherent responses passed recall,
  empty, runaway, and attractor checks.
- crate-map generation, env-doc scan, changed-file formatting, fmt-bomb, and
  diff whitespace checks passed.

The canonical kernel-bucket Redline PM4 arm is blocked on both the clean base
and candidate by the same current-beta limitation:
`gemv_mq4g256v2_residual: GFX10/GFX11 PM4 dispatch does not yet support
scratch (private=32, dynamic_callstack=false)`. Before that refusal, both
lanes reported the same capture identities: prefill-128 hash
`bdc60fd56c3670e7`, prefill-512 hash `9111b45dd02bfe5d`, and decode hash
`92ede73d35f4a51f`. This record does not claim a PM4 pass; the new route is
itself excluded during graph/retained capture and falls back to the batched
kernel there.

## Interpretation

The result supports review of this narrowly gated gfx1100 candidate. It does
not transfer to other architectures, KV formats, graph/PM4 routes, prompts,
or drafts. It also does not solve draft-target disagreement: this fixture's
`tau=1.88` is unchanged, so sufficiently low-tau workloads may still favor
plain autoregressive decode.
