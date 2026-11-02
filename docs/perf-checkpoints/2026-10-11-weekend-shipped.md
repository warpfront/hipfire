# Weekend of 2026-10-10/11: measurements behind five shipped changes

**Lifecycle:** `historical`

**Date:** 2026-10-11. All runs took place on 2026-10-10 or 2026-10-11.

**Disposition:** these are measured local results for changes that are already
on `beta`. The record gives the fixture, method and every number that the
public summaries of these changes quote. It does not admit or change any
default and does not update a product benchmark. Nothing here transfers to
another model, quant, GPU, prompt set, route or method without a new record.
Numbers marked **derived** are arithmetic on the measured values beside them.

Sections:

1. Sampled requests on the exact VMM batch route
2. Smaller DFlash lane state: 8 resident lanes at C8
3. Flash-Next q4 PLE rows on Strix Halo
4. ACO-compiled MQ4 GEMV versus hipcc (PeaceMaker lift)
5. `/v1/messages` and `/v1/responses` live smoke

---

## 1. Sampled requests on the exact VMM batch route

**Change:** `serve.batch_sampled`, default on. It merged to `beta` as the series
ending `d0e1a0c35` (feature commit `a0c4c31f7`). The runs used the pre-merge
branch build of the same series, daemon md5 `b1e615ba707786ea89e23d1201acfff6`.

### Fixture

- **GPU:** one Radeon AI PRO R9700 (gfx1201, 32 GB).
- **Model:** `qwen3.8-27b.mq4-xts` (14,987,185,152 B).
  - MTP mode uses `qwen3.8-27b.mtp`.
  - DFlash mode uses `qwen3.8-27b-dflash.mq4v2`, with
    `HIPFIRE_CB_VERIFY_CHUNK128=1` set. This is a developer variable that is
    off by default.
- **Serve:** `hipfire serve` with `serve.vmm_batch`, `continuous_batch_size = 8`.
- **Workload:** the BetterBench corpus, 48 requests built from 29 prompts.
  - Sampling: T 0.7, top-p 0.95, top-k 20, wire seed 42.
  - Thinking off; `max_tokens` 256.
- **Client:** a closed-loop concurrent HTTP client that runs outside the tree,
  with K = 1 (C1) or K = 8 (C8) clients. Each cell was one fresh
  `hipfire serve` process.
- **Metric:** aggregate output tok/s (`bb_aggregate_tps`).

### Throughput

| mode | C1 tok/s | C8 tok/s |
|---|---:|---:|
| MTP | 58.808 | 186.930 |
| DFlash | 95.899 | 163.177 |

Rounded, these are 58.8 → 186.9 and 95.9 → 163.2. Each value is one process.
C2 and C4 were measured only on an earlier build of the branch, so they are not
part of this record.

### Identity: each C8 request against the same request at C1

Text, tokens and finish were compared per request.

| mode | identical | requests that ran batched at C8 |
|---|---:|---:|
| MTP | 48/48 | 47 of 48 |
| DFlash | 48/48 | 48 of 48 |

### Distribution oracle: sampled MTP versus AR

The oracle is `sampled_mtp_penalties_match_ar_distribution`, run once on the
same GPU, model and MTP head.

- **Trials:** 2,000 per arm, 8 tokens each, with a think-closed render.
- **Sampling:** T 0.7, top-p 0.8, top-k 20, repeat penalty 1.1 over a 128-token
  window, presence 0.5, frequency 0.2.
- **Cases:** `animals` ("Name ten random animals…") and `integers` ("Write
  twenty random integers…").
- **Tests:** 18 chi-square tests, 8 token positions plus one (t1, t2) joint per
  case, at α = 0.001.
- **Result:** PASS.
  - Every test has df 2–39 and an effective sample of 1,079–1,850. The
    degeneracy floor is 100.
  - The smallest p is 0.0151 (`animals`, token 3).
  - MTP acceptance: `animals` 7,843 of 14,092 drafts, `integers` 6,319 of
    17,841.
  - Wall time 4,690 s.

## 2. Smaller DFlash lane state: 8 resident lanes at C8

**Change:** `dc9f47925`. The runs used a pre-merge build of the same change,
daemon md5 `d09d9f76bd0cfc13ed84c550ddcfe6c1`. The base was `beta` `662c7b21d`,
daemon md5 `c60e5b8fbebfabbaaf52ce5fb163d10a`.

### Fixture

- **GPU:** one R9700 (gfx1201, 32 GB).
- **Models:** `qwen3.8-27b.mq4-xts` + `qwen3.8-27b-dflash.mq4v2`.
- **Serve:** `serve.vmm_batch`, `continuous_batch_size = 8`, greedy
  (temperature 0).
- **Workloads:**
  - `long8`: 8 prompts of 3,780–4,354 tokens, 640 output tokens each.
  - `xl8`: 8 prompts of 7,651–8,316 tokens, 512 output tokens each.
- **Client:** the same closed-loop client as section 1, at K = 8.

### Results

| item | base `662c7b21d` | new |
|---|---:|---:|
| DFlash state per lane | 1,038,090,240 B | 805,306,368 B |
| DFlash state per lane, **derived** (GB = 10⁹ B) | ≈ 1.04 GB | ≈ 0.81 GB (0.75 GiB) |
| `long8`: lanes resident at K = 8 | 7 + 1 queued | 8 |
| `xl8`: lanes resident at K = 8 | 6 + 2 queued | 8 |
| `long8` aggregate output tok/s | 92.82 | 101.54 |
| `xl8` aggregate output tok/s | 70.70 | 73.15 / 69.21 (two processes) |

- At K = 8, the text of every cell equals K = 1.
- **`long8`:** one process per arm. The new arm's first process (62.90 tok/s)
  compiled kernels on first use (JIT), so it was discarded and the second
  process is reported.
- **`xl8`:** the two new runs bracket the base, so the result is no change
  beyond process noise.

## 3. Flash-Next q4 PLE rows on Strix Halo

**Change:** `662c7b21d`, which makes the `.ple-q4` sidecar the default on gfx1151.
The in-load fill followed in `3a9ae4961`.

### Fixture

- **Hardware:** Strix Halo (gfx1151, 128 GB unified memory).
- **Model:** `qwen3.8-flash-next-gptq3.mq4` (125,288,544,792 B), the same
  artifact in every run of this section. With the Q8 rows, its WT2 logits are
  `6018690d…`.
- **Serve:** `hipfire serve`.
- **Prompts:** three 8K-token prompt classes, each measured steady, i.e. after
  the prewarm and one warm request:
  - Ciru: real text, Ciru protocol;
  - natural prose;
  - agentic.

### Prefill tok/s, steady

| prompt | Q8 rows (old default) | q4 rows (new default) | change, **derived** |
|---|---:|---:|---:|
| Ciru 8K | 1,598.8 | 1,828.4 | +14.4 % |
| natural (prose) 8K | 1,582.2 | 1,840.2 | +16.3 % |
| agentic 8K | 1,654.8 | 1,832.7 | +10.8 % |

- **Q8 column:** the store-off arm of the hot-row store A/B at `1bd278a2e`.
  Store off is the shipped Q8 default. Each value is the median of 3 fresh
  processes:
  - Ciru: 1,618.7 / 1,598.8 / 1,539.5;
  - natural: 1,515.5 / 1,582.2 / 1,585.3;
  - agentic: 1,650.7 / 1,654.8 / 1,661.5.
- **q4 column:** one process per prompt class at `662c7b21d`, after the default
  15 GiB hottest-first prewarm.

### Quality

**WT2 KLD (8,160 tokens):**

| rows | KLD | top-1 agreement |
|---|---:|---:|
| Q8 | 0.103252 | 0.8763 |
| q4 | 0.104865 | 0.8718 |

- **Derived:** 0.104865 − 0.103252 = +0.001613 ≈ +0.0016.
- The q4 KLD was measured by a diagnostic build that re-quantizes each decoded
  row exactly as the q4 table stores it (`233cb86ac`). The real `.ple-q4`
  sidecar's WT2 logits are byte-equal to that emulation's (`23526fdf…`).

**Task check, TC70-84:** 7 seeds (123–129), Ciru tool protocol (T 0.7, top-p
0.8, top-k 20, presence 1.5, no think), MTP off, `max_seq` 262144, one fresh
serve process per run.

| | q4 emulation | Q8 |
|---|---:|---:|
| total, 7 seeds | 534 | 544 |
| per seed | 70–80 | 73–83 |

- Mean difference: −0.43 points/30 per seed, 95 % CI [−1.6, +0.75].
- Every paired test has p ≥ 0.56.
- No detectable difference. The CI excludes a q4 loss larger than ~1.6
  points/30.

### Table size, derived

- **q4 rows:** 90 B per row = 5 blocks × (2 B f16 scale + 16 B labels), and each
  row holds 160 weights (5 × 32).
- **Q8 rows:** 170 B per row = 5 blocks × (32 B + 2 B scale).
- **Weight count:** the 28.8 GB q4 table holds 28.8 × 10⁹ / 90 = 320 × 10⁶ rows,
  and 320 × 10⁶ × 160 ≈ 51.2 × 10⁹ weights.
- **Cross-check:** the ~54.4 GB Q8 table at 170 B per row gives the same 320 × 10⁶
  rows.

## 4. ACO-compiled MQ4 GEMV versus hipcc (PeaceMaker lift)

**Change:** `80b819f89`, followed by `1ad7a54e8`. This is a developer-only
experiment, not a product route.

### Fixture

- **GPU:** one R9700 (gfx1201).
- **Kernel:** `gemv_mq4g256v2_residual`, M = 5120, K = 6144, grid 5120 × 1 × 1,
  block 32 × 1 × 1.
- **Inputs:** captured at a real `qwen3.8-27b.mq4-xts` decode call site (launch 6,
  position 513).
  - Weight slice sha256 `81399d75f09fb226030f0dc836b0bb46bfc2c955d4008ffedf8cabf42386ac0a`.
  - x sha256 `afa96e019383b5f252d7a1eb77e634dba1328898ea0974a2efebff85351b4621`.
- **Odd/tail shapes:** two more cases at M = 5119, K = 5888, cut from a prefix of
  the real inputs. That makes 3 shapes in all.

**Arms:**

- **hipcc:** the inventory recipe object, sha256 `bd87c3e5…`.
- **PeaceMaker native twin.**
- **ACO body on HIP:** the ACO body lifted by PeaceMaker and run under HIP.
  - The 5,456-byte ACO body sits unchanged behind a 28-byte HSA prologue.
  - Candidate object sha256 `9768c2a5…`; ACO binary sha256 `ff89b057…`;
    SPIR-V sha256 `a5515733…`.
- **Original RADV/Vulkan executable.**

The oracle is the `pm_aco_oracle` example, executable md5
`b5e788d0fc6a99180155126c69b6a524`.

### Byte comparison

The full mode compares the weight, input and output buffers, guard frames
included:

> 16,719,872 + 32,768 + 28,672 = 16,781,312 B (≈ 16.8 MB) per check.

| check | count | differing bytes |
|---|---:|---:|
| identity checks over all arm/mode pairs (eager and HIP graph) | 22 | 0 in every check |
| guard, VMM tail and graph-replay checks | 45 | all pass |
| planted corruptions caught (negative controls) | 25 of 25 | — |

The planted corruptions:

- wrong weight bytes or nibbles;
- wrong input bytes;
- flipped output bits;
- corrupted guard bytes;
- a tampered Vulkan reference.

### Timing (warm cache, one session)

- **Protocol:** 20 warm-up launches, then 5 samples × 200 launches per arm,
  timed with one event pair per sample.
- **Median time per launch:**

| arm | µs per launch |
|---|---:|
| hipcc | 18.243 |
| ACO body on HIP | 21.743 |
| native PeaceMaker twin | 21.857 |

- **Ratios:** ACO/hipcc 1.192, ACO/twin 0.995.
- hipcc is faster on this kernel.
- This is a kernel diagnostic, not product speed.

## 5. `/v1/messages` and `/v1/responses` live smoke

**Change:** `0c75e05fa`, followed by `8a5cfb539`.

- **Binaries:** built from the adapter branch before it was committed. hipfire
  md5 `6a3d056281127d9540908e062767c9f4`, daemon md5
  `5f494264bc970e93efc046d712bd3e8f`.
- **GPU:** one R9700 (gfx1201).
- **Model:** `qwen3.5:4b`.
- **Requests:** 12 in total.
  - Messages: text, stream, tool, tool round trip, stop sequence, error.
  - Responses: text, stream, tool, tool round trip, error.
  - One chat-completions control.
- **Result:** 10 requests returned HTTP 200, and both error cases returned the
  expected HTTP 400. `max_tokens` missing gave the Anthropic error shape, and
  `previous_response_id` was refused.
- **Equivalence:** the request-equivalence unit test (each adapter hands the
  daemon the same `generate` request as the matching chat-completions call)
  ships with the change.
