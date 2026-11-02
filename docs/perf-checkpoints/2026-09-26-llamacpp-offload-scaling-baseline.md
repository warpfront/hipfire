# llama.cpp partial GPU offload — cost per spilled layer on gfx1201 — 2026-09-26

**Lifecycle:** `historical`

**Disposition:** exploratory baseline for
[`2026-09-26-gfx1201-gpu-offload-scaling.md`](2026-09-26-gfx1201-gpu-offload-scaling.md)
— the same host, the same GPU, the same 128/64 decode shape, measured on a
different engine, so the hipfire `memory.gpu_layer_budget` curve has an
independently-produced reference. It is **not** a product baseline, not an
admission, and not a `docs/BENCHMARKS.md` claim. It owes a claim-scoped
[`scripts/serve_harness.py`](../../scripts/serve_harness.py) pass on the engine
it describes (llama.cpp) before being restated as a performance claim.

The hipfire numbers quoted below are **transcribed from the 2026-09-26 record,
not re-measured here**. Nothing in that record is modified.

## What this establishes

1. **The two engines put the host-side work in different places.** hipfire's
   spilled layers keep executing on the GPU and read their weights from
   host-mapped RAM over PCIe — every spilled byte crosses the link once per
   token. llama.cpp's non-offloaded layers execute on the **CPU** backend and
   read weights from system RAM; the only per-token traffic across PCIe is the
   hidden state, a few KB, at one contiguous boundary.
2. **So the two curves are bounded by different resources**, and both curves
   are near the ceiling of their own resource (see "Ceilings"): hipfire at
   79–86% of a measured 27.1 GB/s link, llama.cpp at 82–93% of a measured
   50–58 GB/s DRAM read (50.0–54.2 GB/s on the file-backed arm, which is the
   access pattern its CPU layers actually use).
3. **On this host the DRAM-bound design is ~1.9× cheaper per spilled byte**
   (llama.cpp 45.2 / 46.4 GB/s marginal vs hipfire 21.4 / 23.4 GB/s), and
   1.1–1.6× faster at matched spilled-layer counts across the whole partial
   range, for both model sizes. hipfire is faster only at zero spill.

## Fixture identity (measured)

- Host: 1× AMD Radeon RX 9070 XT, `gfx1201`; 17,095,983,104 B VRAM. Same host
  as the hipfire record, so the two are directly comparable.
- CPU: AMD Ryzen 7 7800X3D, 8c/16t. RAM 29,661 MB total; THP `[always]`.
  **Swap had 14.2 GB in use** during these runs, and an unrelated
  `baloo_file` indexer was doing sustained disk I/O: **load average was
  5.4–11.0 throughout**. This is a shared desktop, not a quiesced bench host;
  see "Limits".
- **Engine:** llama.cpp @ `5cf3a35287163f79a302a15db66f0fa386d94d10`
  ("llama-grammar: fix numeric truncation for token_id parsing (#29382)"),
  build number 11169. Backends built: `GGML_VULKAN=ON` (RADV, GFX1201),
  `GGML_CPU=ON`, `GGML_CUDA=OFF`, `GGML_HIP=OFF` — i.e. **the GPU side is
  Vulkan/radv, not ROCm**, which is a genuine method difference from the
  hipfire record that must be carried by any comparison.
- **Binaries (the fixture identity that matters):**
  `llama-bench` SHA-256 `7a38e93aff0eaf8c4891d8e36a327044b969c3ad294c96a6ddd96a6025b725f4`;
  `llama-cli` SHA-256 `66897146b4d14fbb0ce83f5b772bc51df9d9ee7d4d76389585cd7a0474ea41cf`;
  `llama-server` SHA-256 `63161354cfa7de3e72be4f891052e08a0c956cd107868a3133ca2cd43d06cf2f`.
- Prompt: `benchmarks/prompts/gpu_offload_probe.txt`, SHA-256
  `b6eddc54931a1daa28c32fc8381a72d982eded30093ddf13a6a71049a695066f` — the
  **same committed prompt file** the hipfire record used. Used for the text
  gates only; the throughput sweep uses llama-bench's synthetic tokens (see
  "Method").
- Models (unsloth GGUFs, the Q4_K family — **not** comparable in format to the
  hipfire MQ4 records, so every cross-engine claim below is stated per spilled
  byte or per spilled layer, never as a raw tok/s delta alone):

| model | bytes | SHA-256 | n_layer | tensor payload | layer-indexed | always-resident |
|---|---:|---|---:|---:|---:|---:|
| `Qwen3.5-2B-Q4_K_S.gguf` | 1,217,757,440 | `56eee7b85a2023ba393c5f9a5a5e372e3645e7925a13adb4d13723fc5bc14124` | 24 | 1,206,795,520 | 789,609,728 (65.4%) | 417,185,792 (34.6%) |
| `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | 32 | 5,669,554,176 | 4,263,053,312 (75.2%) | 1,406,500,864 (24.8%) |

`benchmarks/gguf_bytes.py` produces the byte columns; its `tensor_payload`
matches llama-bench's own reported `model_size` exactly for both files
(1,206,795,520 and — via the same code path — 5,669,554,176), so the parser is
cross-checked against the engine's own accounting rather than trusted alone.

**The layers are not byte-uniform** — 4 distinct per-layer sizes for the 2B
(29,509,632 – 35,312,256 B), 6 for the 9B (117,999,616 – 142,394,112 B). The
hipfire record could use a single per-layer divisor because its layers were
uniform; here a count-based fit and a byte-based fit are reported side by side.

## The two knobs do not count the same quantity

This is the trap in comparing the two curves, and it was resolved by audit, not
by reading documentation:

| engine | knob | what it counts | host side |
|---|---|---|---|
| hipfire | `memory.gpu_layer_budget=N` | repeating layers **kept on the GPU** | prefix, `n_layers − N` layers |
| llama.cpp | `-ngl N` | VRAM layer **slots**, and the slot count **includes the output layer** | prefix, `n_layers + 1 − N` repeating layers |

Verified from llama.cpp's own per-layer device assignment lines
(`load_tensors: layer N assigned to device …`), for a 24-layer model:

| flag | repeating layers in VRAM | output slot | host repeating layers |
|---|---:|---|---:|
| `-ngl 1` | 0 | GPU | 24 |
| `-ngl 4` | 3 | GPU | 21 |
| `-ngl 25` | 23 | GPU | 1 |
| `-ngl 99` | 24 | GPU | 0 |
| `-dev none` | 0 | CPU | 24 |

So `-ngl k` and `budget k` are **not the same split**, and the two knobs count
opposite ends of one split: hipfire counts layers retained, llama.cpp counts
VRAM slots (one of which is not a repeating layer at all). The host side is a
prefix in both engines — that part does match. The sweep therefore drives the
flag as `-ngl n_layer + 1 − spilled` so both engines land on the same
spilled-prefix counts, and every recorded point carries the loader's own host
layer list; nothing is inferred from the flag value.

Two further flag traps found on this build:

- **`-ngl 0` is not "no VRAM offload" in the sense one wants.** The audit shows
  it does leave 0 layers in VRAM, but it also *enables the auto-fit path*
  (`common_params_fit_impl: getting device memory data for initial parameters`,
  `projected to use 1044 MiB of device memory vs. 1604`). Setting `-ngl` to an
  exact non-zero value does not. The explicit all-host endpoint used here is
  `-dev none` ("don't offload"), never `-ngl 0`.
- **`--fit` adjusts arguments to fit device memory** and is on by default in
  `llama-cli` (off by default in `llama-bench`, via `-fitt`). Every audit load
  is run with `--fit off` so the placement the audit reads is the placement the
  timings were taken with.

## Method

- llama-bench, `-p 128 -n 64` — the same 128-prompt / 64-generated shape as the
  hipfire record. llama-bench's own warmup runs are left enabled; each point is
  a **fresh llama-bench process**, `-r 5` samples, median reported, all 5
  samples retained in the per-point JSON and the CSV's `samples_ts`.
- Threads: llama-bench's default (`-t 8`, i.e. the 8 physical cores). Not
  pinned by the harness; recorded per point.
- Points per model: fully resident, every `n_layer/8` spilled, fully spilled,
  plus the explicit `-dev none` endpoint (10 points). Total sweep wall clock:
  2B 56 s, 9B 231 s.
- Every point's split is audited from a separate verbose `llama-cli` load
  (with stdin closed and `-st`; without those llama-cli enters conversation
  mode and blocks on stdin forever, which is how an earlier probe run silently
  burned a 300 s timeout).
- **Reproducibility check:** the 2B sweep was run twice end to end. Fully
  resident 234.8 → 241.4 tok/s (+2.8%), full spill 46.7 → 47.5 (+1.7%), i.e.
  the curve reproduces to ~3% run-to-run. Only the second run is tabulated.
  The 9B was run once.

## Results

Decode tok/s, median of 5. `MB/token` is the **per-token** host read, computed
per point from that point's actual host layer list — not `file_size/n_layers`,
and not the raw residency sum (see "Byte accounting" for why the difference
matters).

### Qwen3.5-2B-Q4_K_S (24 layers)

| `-ngl` | spilled (of 24) | host MB/token | output slot | decode tok/s | ms/token |
|---:|---:|---:|---|---:|---:|
| 25 | 0 | 0.0 | GPU | 241.4 | 4.14 |
| 22 | 3 | 105.9 | GPU | 166.5 | 6.01 |
| 19 | 6 | 203.1 | GPU | 111.7 | 8.95 |
| 16 | 9 | 300.2 | GPU | 91.8 | 10.90 |
| 13 | 12 | 397.3 | GPU | 76.6 | 13.06 |
| 10 | 15 | 498.5 | GPU | 66.4 | 15.07 |
| 7 | 18 | 595.6 | GPU | 56.0 | 17.85 |
| 4 | 21 | 692.6 | GPU | 50.9 | 19.64 |
| 1 | 24 | 789.6 | GPU | 47.5 | 21.06 |
| `-dev none` | 24 | 1206.8 | **CPU** | 33.0 | 30.33 |

### Qwen3.5-9B-Q4_K_M (32 layers)

| `-ngl` | spilled (of 32) | host MB/token | output slot | decode tok/s | ms/token |
|---:|---:|---:|---|---:|---:|
| 33 | 0 | 0.0 | GPU | 84.2 | 11.88 |
| 29 | 4 | 559.2 | GPU | 38.5 | 26.00 |
| 25 | 8 | 1079.6 | GPU | 26.4 | 37.92 |
| 21 | 12 | 1599.9 | GPU | 20.6 | 48.49 |
| 17 | 16 | 2133.1 | GPU | 16.6 | 60.14 |
| 13 | 20 | 2652.4 | GPU | 13.9 | 71.80 |
| 9 | 24 | 3171.6 | GPU | 12.3 | 81.49 |
| 5 | 28 | 3704.9 | GPU | 10.8 | 92.82 |
| 1 | 32 | 4263.1 | GPU | 9.5 | 105.12 |
| `-dev none` | 32 | 5097.4 | **CPU** | 8.8 | 114.03 |

The last row of each table is the only point where the output projection moves
to the host as well. It is a **different placement regime** and is not pooled
into the regressions below; it is listed to price the output projection
separately: for the 2B, `-ngl 1` (47.5 tok/s) and `-dev none` (33.0 tok/s) have
*identical* repeating-layer bytes on the host (789.6 MB/token) and differ only
in where the output projection runs.

## Regression

Token time against spilled layer count, over the 9 points with the output slot
on the GPU:

| model | fit (ms/token) | R² | fully-resident intercept, measured |
|---|---|---:|---:|
| 2B | `t = 4.261 + 0.7253·n_host` | 0.99640 | 4.14 ms (241.4 tok/s) |
| 9B | `t = 13.927 + 2.8494·n_host` | 0.99886 | 11.88 ms (84.2 tok/s) |

Token time against per-token host **bytes** (the fit that stays meaningful with
non-uniform layers):

| model | fit (ms/token) | R² | marginal host-side rate |
|---|---|---:|---:|
| 2B | `t = 4.158 + 0.0221·MB_host` | 0.99651 | **45.2 GB/s** |
| 9B | `t = 13.615 + 0.0216·MB_host` | 0.99917 | **46.4 GB/s** |

Two models, two different layer counts, two different per-layer byte sizes,
**~2.4% apart** on the implied host-side rate. As in the hipfire record, the
byte-fit rate and the full-spill endpoint rate are algebraically the same
quantity at full spill, not independent corroboration; only the linearity and
the agreement between the two models are assumption-free.

Interval-by-interval, the single slope hides real scatter, and the scatter is
much larger for the smaller model:

| model | interval marginal rate | per-layer marginal | mean layer bytes |
|---|---|---:|---:|
| 2B | 33.0 – 68.6 GB/s (8 intervals) | 0.7253 ms | 32.90 MB |
| 9B | 39.6 – 53.6 GB/s (8 intervals) | 2.8494 ms | 133.22 MB |

The 2B's 8 intervals move nearly equal byte counts (97.0–101.2 MB each) at
rates that alternate between ~34 and ~69 GB/s. A host layer's cost is therefore
**not purely byte-proportional** at this size: CPU-side attention/recurrence
work and per-layer launch/sync overhead ride along with the weight read. The
9B's intervals are tight (±15%), which is what makes its single-slope fit
trustworthy and the 2B's merely indicative.

### How these fits and R² are computed

Plain **unweighted** OLS in linear space, per model, over the 9 points whose
output slot is on the GPU (`benchmarks/offload_scaling_fit.py`, `ols()`):

- `slope = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²`, `intercept = ȳ − slope·x̄`,
  `R² = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²`
- `x` = host-resident layer count, or MB/token for the byte fit;
  `y` = ms/token = 1000 / decode tok/s. One **median** per point — the 5
  underlying samples are kept in the CSV but are not propagated into the fit.

Because each point is a median with its own spread, the weighting choice was
checked rather than assumed:

| model | unweighted slope | `w = 1/stdev²` slope | leave-one-out slope |
|---|---:|---:|---|
| 2B | 0.7253 ms/layer (45.4 GB/s) | 0.7297 (+0.6%) → 45.1 GB/s | 0.7156–0.7471 → 44.0–46.0 GB/s |
| 9B | 2.8494 ms/layer (46.8 GB/s) | 2.8265 (−0.8%) → 47.1 GB/s | 2.7946–2.8646 → 46.5–47.7 GB/s |

Both perturbations are small against the 1.9× cross-engine gap that the fits
are being used to size.

Two cautions on reading the R² column:

- **R² is a goodness-of-fit number, not an error bar.** With `x` spread across
  the full range, SS_tot is dominated by the trend, so a high R² coexists with
  large per-interval scatter: the 2B's `R² = 0.9964` sits on top of intervals
  that range 33–69 GB/s. The interval table, not R², is what bounds the claim.
- The hipfire record's `R² ≈ 0.9999` is **quoted from that record, not
  recomputed** — its fitting procedure is not stated in that file — so R²
  values are deliberately not compared across the two records. Only regimes,
  slopes and implied rates are.

## Byte accounting (and the divisor error this record had to avoid)

The hipfire record's lesson applies with a different twist. Two rules decide
what counts as a per-token host read:

- **Host layers count in full** — a GEMV reads the whole weight tensor.
- **The output projection counts in full only when its slot is on the CPU.**
  Otherwise llama.cpp runs it on the GPU.
- **The input embedding never counts in full.** Only one row (one token's
  embedding) is looked up per token. `token_embd.weight` is 417 MB for the 2B
  and 572 MB for the 9B — 35% and 10% of their payloads — and a naive
  "resident bytes" divisor would add all of it to every host layer count. The
  sweep records that raw sum in `host_bytes_total`; the per-token figure in the
  tables above is recomputed from the audited host layer list, and
  `host_bytes_total` must **not** be used as the divisor.

The rule is self-consistent across arms that move the same bytes with different
placement: at `-ngl 1` the 2B sustains 789.6 MB/token at 21.06 ms (37.5 GB/s),
and at `-dev none` 1206.8 MB/token at 30.33 ms (39.8 GB/s) — the second arm
pays 417 MB/token more and moves more bytes per second, so the accounting is
explaining the difference rather than hiding it.

## Ceilings: the link for one engine, DRAM for the other

`benchmarks/host_bw_probe.c` (streaming read, 4 GiB anonymous buffer, 1e9 B per
GB), **with page-fault accounting so the arm proves what it measured**. Two
sessions, both recorded in
`benchmarks/results/llamacpp_offload/host_bw_probe.txt`:

| arm | threads | session A (loadavg 5.4–11) | session B (loadavg 3.3) | major faults |
|---|---:|---:|---:|---:|
| anon | 1 | 44.0 – 46.8 | 47.3 / 48.8 | 0 |
| anon | 8 | 48.7 / 49.7 / 51.9 | 51.6 – 54.9 | 0 |
| anon + `MADV_HUGEPAGE` | 8 | 48.9 / 53.9 | 52.3 – 62.0 | 0 |
| anon | 16 | 51.4 – 53.1 | 53.9 / 55.6 | 0 |
| **the 5.29 GiB 9B GGUF itself, mmap'd** | 8 | 50.0 / 50.9 / 50.9 / 51.4 | 54.2 / 54.0 / 52.3 | 0 |

So the host DRAM streaming-read ceiling on this box is **~50–58 GB/s** (52–60%
of the 96 GB/s DDR5-6000 2-DIMM theoretical), and the file-backed arm — the
pattern llama.cpp's host layers actually read through — is **50.0–54.2 GB/s**.
llama.cpp's offload sustains 45.2–46.4 GB/s of *useful* weight traffic against
that: **82–87%** of the anon 8-thread arm, **83–93%** of the file-backed arm.
The page-size hypothesis is again refuted in both sessions: `MADV_HUGEPAGE` and
the file-backed mapping both land inside the spread of the anonymous arms.

Two probe results deserve their own lines because they are how the ceiling
almost got measured wrong:

- An early file-arm run read **1.7 GB/s** — that is NVMe, not DRAM. It happened
  while 8 GiB of freshly zeroed anonymous buffers were resident and
  `baloo_file` was saturating the disk, which evicted the mapping's pages. The
  instrumented re-run reports `majflt=0` at 50–51 GB/s. Reporting 1.7 GB/s as
  "the host ceiling" would have inverted this record's conclusion.
- Wider or faster RAM remains a real lever **here**, unlike for hipfire. The
  hipfire record concluded "faster or wider RAM buys nothing for offload" and
  that is correct for its design (the link binds before DRAM is reached), but
  the CPU-executed design is on the DRAM side of that boundary: it is at
  82–93% of a 50–58 GB/s ceiling, whereas a healthy Gen5 x16 link would be
  ~55 GB/s practical and the hipfire path's *observed* link is 27.1 GB/s.

## Cross-engine comparison (same host, same shape)

hipfire columns are transcribed from the 2026-09-26 record; llama.cpp columns
are measured here. Both are decode tok/s, median of 5, 128 prompt / 64
generated.

| spilled | 2B hipfire | 2B llama.cpp | ratio | 9B hipfire | 9B llama.cpp | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 250.3 | 241.4 | 0.96× | 101.0 | 84.2 | 0.83× |
| 3 / 4 | 127.6 | 166.5 | 1.30× | 35.2 | 38.5 | 1.09× |
| 6 / 8 | 86.2 | 111.7 | 1.30× | 21.4 | 26.4 | 1.23× |
| 9 / 12 | 64.6 | 91.8 | 1.42× | 15.4 | 20.6 | 1.34× |
| 12 / 16 | 51.9 | 76.6 | 1.48× | 12.0 | 16.6 | 1.39× |
| 15 / 20 | 43.6 | 66.4 | 1.52× | 9.8 | 13.9 | 1.42× |
| 18 / 24 | 37.8 | 56.0 | 1.48× | 8.3 | 12.3 | 1.48× |
| 21 / 28 | 32.9 | 50.9 | 1.55× | 7.2 | 10.8 | 1.50× |
| 24 / 32 | 29.5 | 47.5 | 1.61× | 6.4 | 9.5 | 1.49× |

| quantity | hipfire 2B | llama.cpp 2B | hipfire 9B | llama.cpp 9B |
|---|---:|---:|---:|---:|
| marginal cost per spilled layer | 1.247 ms | 0.7253 ms | 4.589 ms | 2.8494 ms |
| mean per-layer bytes | 30.44 MB | 32.90 MB | 114.89 MB | 133.22 MB |
| **marginal GB/s per spilled byte** | **24.4** | **45.4** | **25.0** | **46.8** |
| implied rate, full-spill definition | 21.4 GB/s | 36.4 GB/s | 23.4 GB/s | 40.6 GB/s |
| measured host-side ceiling | 27.1 GB/s (PCIe) | 50.0–54.2 GB/s (DRAM, file-backed) | 27.1 GB/s | 50.0–54.2 GB/s |
| share of that ceiling | 79–86% | 83–93% | 79–86% | 83–93% |

Reading it:

- **hipfire wins only at zero spill** (−4% on the 2B, −17% on the 9B, i.e. the
  9B MQ4 trunk is genuinely faster than the 9B Q4_K_M at full residency; not a
  like-for-like format comparison, so this is an observation, not a verdict).
- **llama.cpp wins at every partial point**, and the margin grows with spill —
  because it moves *more* bytes per spilled layer (its Q4_K layers are 8%/16%
  larger) at *half* the cost per byte. Per byte, llama.cpp is 1.86× / 1.87×
  cheaper.
- Both engines are pinned near the ceiling of their own resource — llama.cpp at
  83–93% of its file-backed DRAM ceiling, hipfire at 79–86% of its PCIe link.
  The 2× difference in outcome is the ~2× difference between a 27.1 GB/s link
  and 50–54 GB/s of DRAM, not an implementation-quality gap.

**Practical consequence for hipfire's `gpu_layer_budget`:** on this host the
knob's design choice — keep the *compute* on the GPU and stream weights over
PCIe — is what costs it the partial-offload range. The hipfire record's own
most actionable open item (the link running Gen4-class on a Gen5 x16 slot) is
exactly the right lever: closing that gap would roughly double the hipfire
offload ceiling and would roughly cancel this comparison. Until then, a
CPU-executed fallback (llama.cpp's design, or hipfire's per-token GEMV CPU path
on gfx10/gfx906) is the better partial-offload strategy on this box.

## Correctness gate

Offload is where a broken or misplaced weight could still emit plausible text,
so the decoded text was compared directly. Unlike the hipfire record, **byte
identity is not the expectation**: llama.cpp's partial offload changes which
backend executes a layer, so the numerics change even when everything is
correct. What is checked is completion + coherence + *measured* divergence.

Route: `llama-server` + `/completion` (raw prompt bytes, no chat template), and
separately `/v1/chat/completions` (model template, thinking off) — the latter is
the route the hipfire record's `hipfire run` eyeball corresponds to.
`llama-cli` was rejected for this check because its load banner and progress
spinner go to stdout and the spinner's frame count depends on load time; a byte
diff over that reports divergence at the first spinner frame instead of at the
first differing token. Temperature 0, `seed=42`, `n_predict=200`, fresh server
per arm. Arms: `-ngl 99` (all GPU), a mid split (2B `-ngl 13`, 9B `-ngl 17`),
`-dev none` (all host).

| model | route | arms | text | divergence |
|---|---|---|---|---|
| 2B | raw | gpu / mid / host | all on-task Python | gpu vs mid **0 B** common prefix; gpu vs host **0 B** |
| 2B | chat | gpu / mid / host | all on-task Python | 208 B / 542 B common prefix |
| 9B | raw | gpu / mid / host | all on-task Python | 419 B / 419 B common prefix |
| 9B | chat | gpu / mid / host | all on-task Python | **gpu vs mid byte-identical (862 B)**; gpu vs host 210 B |

- Every arm exited 0 and returned non-empty text; all four arms per model
  produce a correct sliding-window `longest_substring` function when read.
- **Contrast with hipfire's gate, which came back byte-identical:** llama.cpp's
  offload is *not* numerically transparent. Two arms differ from the first
  token; the 9B chat route happens to agree exactly at 16 spilled layers and
  diverges at 32.
- **One arm degenerated.** 2B raw at `-ngl 13` (12 spilled) produced a
  repetition attractor — "The function should use a sliding window approach.
  The function should use a dictionary…" for the whole 200-token budget, while
  the same model at `-ngl 99` and `-dev none` produced clean code. This is
  exactly the failure the hipfire record's "always eyeball the decoded output"
  rule exists to catch: it is coherent, non-empty, exit-0 text that no
  length-or-throughput check would flag. It is reported, not smoothed over.
  n=1 arm, n=1 seed: it is evidence that greedy output is placement-sensitive,
  not a rate.

## Assumptions and limits

1. **Different quant format and different GPU stack.** MQ4 vs Q4_K, ROCm vs
   Vulkan. Per-byte and per-layer comparisons are fair; raw tok/s deltas are
   not, and the arm comparison table is only valid at matched *spilled-layer
   counts*, which is hipfire's axis.
2. `t_vram/layer = intercept/n_layer` assumes the fully-resident time spreads
   evenly across layers, exactly as in the hipfire record; violations make the
   reported rate an over-estimate, so 45.2 / 46.4 GB/s are floors.
3. Bytes/token = host layer bytes + the output projection when it is
   host-resident. Each host byte is assumed to cross once per token; not
   verified with a hardware counter.
4. The marginal rate is **weight-equivalent**, not a raw memory rate: CPU-side
   layers also carry their attention/recurrence compute and their KV (in
   llama.cpp's layer split the KV for a layer lives with the layer), and the
   exchange of hidden state across the CPU↔GPU boundary is bundled in. The 2B's
   33–69 GB/s interval scatter is that effect showing up.
5. **Shared, non-quiesced desktop.** `baloo_file` was doing sustained disk I/O
   and load average ran 5.4–11.0 during the sweeps. It demonstrably evicted
   page-cache-resident data in one probe run. The within-point stdev is tight
   (median 1.3% for the 2B, 2.7% for the 9B), and the 2B curve reproduced to
   ~3% across two full sweeps, but between-point state is *not* controlled the
   way the hipfire record's host was. A re-run on an idle machine is owed before
   any of these numbers are promoted.
6. `vram_used_mb` in the CSVs sums both AMD GPUs (the discrete card plus the
   Raphael iGPU), so it is not a clean VRAM reading for this model and is not
   used in any conclusion.
7. MemAvailable spread across each sweep (376 MB / 1092 MB) and page-cache
   spread (611 MB / 1408 MB) are recorded per point in the CSVs. As in the
   hipfire record, MemAvailable is the metric that *cannot* see mmap'd
   file-backed weights; page cache is the one that moves.

## Reproduction

```bash
# sweep (one fresh llama-bench process per point; placement audited per point)
./benchmarks/llamacpp_offload_sweep.sh ~/.lmstudio/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_S.gguf
./benchmarks/llamacpp_offload_sweep.sh ~/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf

# fits, per-token byte accounting, interval rates, memory spreads
python3 benchmarks/offload_scaling_fit.py \
  benchmarks/results/llamacpp_offload/Qwen3.5-2B-Q4_K_S \
  benchmarks/results/llamacpp_offload/Qwen3.5-9B-Q4_K_M

# per-layer byte table (cross-checked against llama-bench's model_size)
python3 benchmarks/gguf_bytes.py <model.gguf>

# text gate on both routes
python3 benchmarks/llamacpp_offload_coherence.py <model.gguf> --route raw
python3 benchmarks/llamacpp_offload_coherence.py <model.gguf> --route chat

# host-side ceiling (fault-instrumented; a majflt>0 arm is disk, not DRAM)
gcc -O3 -march=native -pthread -o /tmp/host_bw_probe benchmarks/host_bw_probe.c
/tmp/host_bw_probe --gib 4 --threads 8
/tmp/host_bw_probe --gib 0.25 --threads 8 --file <model.gguf>
```

Raw per-point JSON, CSVs (with the 5-sample arrays), audit load logs and
`meta.json` for both models are under
`benchmarks/results/llamacpp_offload/<model-stem>/`.
