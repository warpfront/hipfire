# gfx1100 FA2 split-KV verifier with HipGraph and Redline/PM4 — 2026-10-05

**Lifecycle:** `historical`

**Disposition:** measured opt-in candidate evidence; not a product default,
current baseline, or admission decision.

## Question and scope

Measure an exact-gfx1100 Q8 DFlash verifier route that keeps the packed-KV
front end and replaces the established R4/R8 attention step with an FA2
split-KV S8 partial/merge back end. Then determine whether the same fixed-grid
route can be safely retained by HipGraph and Redline/PM4.

The candidate is default-off behind
`HIPFIRE_GFX1100_FA2_SPLIT_VERIFY=1`. It fails closed unless the target is
dense Qwen H24/NKV4/HD256, the verify batch is 4..32, Q8 KV is active, and the
live logical context exceeds 4,096. Capture additionally requires precompiled
kernels and fully materialized fixed-address Q16 scratch. Redline product
admission retains its existing B=16, single-GPU, Q8-state, and route guards.

Source base: `d5305333d4c609848f09ce1cd98502bb2ee2fe82` (`warpfront/beta`).

## Fixture identity

- Host GPU: Radeon Pro W7900, exact `gfx1100`, HIP 7.15, GPU0.
- Target: `qwen3.8-27b.mq4-xt`, SHA-256
  `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`,
  MD5 `e45d15bfe0c9a87132697101d17cbed6`.
- Draft: `qwen38-27b-dflash-mq4.hfq`, SHA-256
  `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`,
  MD5 `013395583cd04206c8aa68f4d061983d`.
- Prompt: `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`, 21,550
  actual tokens, MD5 `b4d0b63cddcac872648ddf3cdd92cac2`.
- Product settings: Q8 VMM KV, DFlash, greedy sampling, 200 output tokens,
  `max_seq=65536`.

## Fresh-process product A/B

Six graph-off fresh processes ran in declared order
`off,on,on,off,off,on`, with one unrecorded warmup before each measured run.
The baseline is the established gfx1100 R4/R8 verifier; the candidate changes
only the split-verifier route.

- CLI MD5: `9bbfbbaac68ed262867a6e7136485081`.
- Daemon MD5: `f0c79a77e2cb1b019ee58bbae96de113`.

| route | decode samples (tok/s) | median | tau / cycles | delta |
|---|---|---:|---:|---:|
| established R4/R8 | 43.5, 42.6, 42.5 | 42.6 | 1.97 / 67 | — |
| FA2 split-KV S8 | 48.5, 48.3, 47.8 | 48.3 | 1.97 / 67 | +13.38% |

The unchanged tau and cycle count isolate the speedup to verifier execution,
not improved draft acceptance.

## Kernel screen

Batch 16 used 10 warmups and 30 measured repetitions per cell.

| logical context | established R4/R8 | split-KV S8 | speedup |
|---:|---:|---:|---:|
| 8,192 | 368.80 us | 177.36 us | 2.079x |
| 20,676 | 828.45 us | 410.72 us | 2.017x |
| 32,768 | 1,369.01 us | 648.21 us | 2.112x |

S1 was bit-identical to direct FA2. S8 relative L2 against direct FA2 was
about `2.46e-4`, `2.50e-4`, and `2.58e-4` respectively; cosine was about
`0.999999970`. The direct and partial kernels compiled at 254/255 VGPR, the
merge at 18 VGPR, with no spills or private scratch.

## HipGraph validation

The warpfront-beta graph-validation binaries were CLI MD5
`9bbfbbaac68ed262867a6e7136485081` and daemon MD5
`f0c79a77e2cb1b019ee58bbae96de113`.

- B=16 capture retained 706 launch blobs.
- Graph off/on decoded at 4.8/4.9 tok/s in this separate cold
  `serve_harness.py` comparison, with `tau=2.06` and 200 generated tokens.
- Graph off/on transcripts were byte-identical, MD5
  `7b8dc5b28daef60f803fe2a466c888b2`.

The two beta runs used the same prompt MD5 and request MD5
`8a54e9aa236f678362f89864bf000125`. Both ended inside the model's hidden
thought channel and therefore reported `RUNAWAY,EMPTY`; their rates are not a
performance claim and the run is route/output parity evidence, not
answer-quality evidence. The fresh-process native bench above remains the
performance measurement.

## Redline/PM4 validation

The route-scoped daemon harness exercised 12 consecutive B=16 windows from
positions 8,176 through 8,352 across HipAuto, capture-safe direct HIP,
recorded HIP, and PM4.

- Result: PASS; backend `pm4_ib`.
- Tape: 711 launches/dispatches, 18 unique typed AQL kernel contracts,
  one packet, one queue, and one phase; dispatch count matched launch count.
- Retained route: ready; 17 successful replays; zero contract, preparation,
  or replay failures.
- All four arms agreed exactly in every window on tokens, argmax, hidden
  staging/ring, final hidden, logits, active KV hashes, GDN intermediates,
  recurrent state after forward, and recurrent state after rollback.
- Five interleaved timing windows: HipGraph median 46.413 ms, PM4 median
  43.630 ms, delta -2.783 ms (-6.00%); p95 delta -2.652 ms.

The five-window PM4 timing is a retained-route smoke measurement, not a
standalone headline throughput claim.

## Validation and interpretation

- `test_kernels`: 17 passed on the W7900.
- `rdna-compute`: 438 passed, 12 ignored.
- `hipfire-arch-qwen35`: 247 passed, 23 ignored.
- `hipfire-generate`: 54 passed.
- `scripts/no-gpu-ci.sh`: Rust/main gates passed. Its Python stage reported
  21 failures and 541 passes because `scripts/hw-gate/select.py` shadows the
  standard-library `select` module when `scripts/hw-gate/review.py` imports
  `subprocess`. This reproduces on the unchanged beta files; this PR does not
  touch `scripts/hw-gate`, `tests`, or `scripts/no-gpu-ci.sh`.
- Crate maps, lifecycle/env inventory, changed-file formatting, and diff
  whitespace checks passed.

These results support review of the narrowly gated gfx1100 candidate and its
HipGraph/Redline retention contracts. They do not transfer to other GPUs, KV
formats, model shapes, prompts, drafts, or sampling modes. The route remains
default-off, and low draft-target agreement can still make DFlash slower than
plain autoregressive decode.
