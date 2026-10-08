# Offload pass-back headroom — CPU-exec spill idle time and host-DRAM contention — 2026-10-01

**Lifecycle:** `historical`

**Disposition:** exploratory headroom reading for `memory.offload_exec=cpu` (partial
GPU offload; design and coverage contract in
[`../plans/partial-gpu-offload-design.md`](../plans/partial-gpu-offload-design.md)
§ 6.2.1). It is **not** a product baseline, not an admission, and not a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim. It answers one design question,
not a throughput question: *during a CPU-executed spill, how much of the wall is
the GPU idle, and do the two engines contend for one host-memory bottleneck?*
The rate columns are context for the idle columns, not results in their own right.

## What this establishes

1. **The GPU is idle for most of a CPU-exec spill's wall, and the share grows
   with the CPU's share of the split.** 9B (32 layers), `gpu_offload_probe`
   prompt, `--spec off --backend noslots`: 77.7% of decode wall runs inside
   CPU-executed steps at 8/32 spilled, 89.8% at 16/32, 92.4% at 20/32, 94.6% at
   24/32; 27B-mq3-xt (64 layers) 66.0% at 8/64. The reading is a *lower bound* on
   GPU idle fraction — a CPU step is a host sync point (blocking D2H, the
   multiplication, an H2D), prefill never enters the seam, and the default stream
   carries all GPU work.
2. **The CPU's host-DRAM stream and the GPU's PCIe read of the same host RAM
   contend only mildly.** In the probe below the CPU retained 86–94% of its
   alone byte rate while the GPU pulled 28 GB/s over the link unchanged, for a
   combined 75–78 GB/s — ~1.4× the CPU-alone rate and ~90% of this host's
   DDR5-5200 dual-channel peak. So a spilled step split across the two engines
   is bounded by host DRAM, not by either engine.
3. **Layer granularity cannot capture any of it.** Consecutive decoder layers are
   serially dependent through the residual stream, so moving a whole layer from
   CPU to GPU swaps who idles; it does not overlap anything. The only shape that
   spends idle GPU time is a *within-step* split (independent output rows, or the
   independent projections that share one input). *(This item is reasoning, not
   measurement.)*
4. **The GPU-only route is only ~20% slower per byte than the CPU route at this
   spill size** (9B 8/32: `cpu` 27.6 vs `pcie` 23.8 tok/s), which is why a split
   has anything to trade at all.

## Fixture identity (measured)

- Host: 1× Radeon RX 9070 XT `gfx1201`, 16304 MB, PCIe **4.0 x16** (`~32 GB/s`
  × 16 current and max), ROCm/HIP 7.2; Ryzen 7 7800X3D (8c/16t), 28 GB RAM.
  Full desktop session; 1-min load 2.1–4.8 during the arms (own work decaying),
  which the repeat runs below show the idle *ratio* does not track.
- Source HEAD `a89ed0a8e9d8dc7a22d4e6dcbacd57bf2abfad74` (v0.4.0 promotion).
- `target/release/daemon` md5 `7f115e384f4fe1521d02dc57fbd02745`;
  `target/release/hipfire` md5 `40549294f6eed802f141a883e65f46a8`.
- `~/.hipfire/models/qwen3.5-9b.mq4` — 5,313,750,016 B, md5
  `296092bf1e6a45d78c1acf815eb93366`, sha256
  `ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad`.
  **This is the registry artifact** (`qwen3.5:9b` → `hipfire-models/qwen3.5-9b` /
  `qwen3.5-9b.mq4`; `registry/v1.json`, generated 2026-09-30, pins this sha256 and
  size from the HF LFS API), and the HF repo has not moved since commit
  `41125e498` "v3-awq-f1 mq4 (overwrite plain, hidden AWQ)", 2026-06-19. It is
  **not** the 9B the CPU-exec records use (`31a8d8dc7603226801b08d8319015602` /
  sha256 `829a84c7…`, 5,297,456,128 B), and that file is in **no registry entry**
  and no HF revision — the other ~5.3 GB 9B files are `qwopus:9b`
  (5,320,221,696 B) and `qwen3.5:9b-hf4` (4,771,464,192 B). So it was a local
  quant, and the record's 9B anchors cannot be re-pulled. Both arms here are
  measured fresh on the registry artifact, which is why these numbers stand on
  their own digest and are not comparable to the 2026-09-27 deltas.
- `~/.hipfire/models/qwen3.8-27b.mq3-xt` — 11,777,616,896 B, md5
  `80bb9198e6a565fc006b2ae1b7c89eca`, sha256
  `3e04fc8db80bda557b965ec60ac876cf2500fced7f340624f3fcbeae134af5c5` (matches
  the handoff table; 64 layers, every projection qt 49 `Mq3G256V2`).
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, md5
  `5835c71e471849b4a72e1dc8e39695e7`.
- Flags: `--spec off --runs 1 --warmups 1 --max-tokens 128 --backend noslots
  --workload stateless --prompt-file <above> --json`, plus
  `HIPFIRE_GPU_LAYER_BUDGET=<budget>`, `HIPFIRE_OFFLOAD_EXEC={cpu,pcie}`,
  `HIPFIRE_CPU_EXEC_TRACE=1`. (The 9B multi-split sweep used `--max-tokens 96`.)
- Every `cpu` point printed `cpu exec: N/N spilled layers fully covered;
  uncovered quants: none` and `0 host-mapped steps still on GPU`, so the arm is
  fully on-CPU and the idle reading is not diluted by steps that silently stayed
  on the link.

## Method

`HIPFIRE_CPU_EXEC_TRACE=1` now also prints, on the *global* step count's doubling
schedule, `cpu exec: idle N% — window ending at step S covers K steps: …ms wall,
…ms on CPU (…us CPU/step)`. `N` is the share of that window's wall spent inside
CPU-executed steps, timed entry-to-return in `cpu_exec::finish_step` (the
blocking D2H, the GEMV, the H2D). Windows rather than a cumulative ratio because
`--runs N` decodes N times inside one process. The quote below is the last window
of each run; reproduce with `grep "cpu exec: idle"`. Windows that straddle the
prefill→decode or warmup→measured-run boundary read low (34–43% in the raw
capture) because prefill is GPU-side and never enters the seam, so it dilutes
whatever window contains it — the last window is the one to quote, and a
cumulative ratio would be wrong for the same reason.

The contention probe is `data-2026-10-01-offload-idle-headroom/probe.rs`
(`cargo run --release -p hipfire-dispatch --example offload_split_probe`, run
from a scratch copy): one 205.5 MB `hipHostMalloc`-mapped buffer (`Mq4G256`,
m=75559 k=5120), 60 reps per phase — (A) `hipfire_cpu::gemv` over it, rayon;
(B) `memcpy_htod` of the same bytes to a device buffer, the same host→device DMA
the `pcie` arm's kernels cause; (C) both, CPU on a spawned thread. The buffer is
205.5 MB — 6× this host's 32 MB L3 — so this is DRAM traffic, not cache, and both
engines stream the *same* buffer because a row split would read the same weight
tensor.

## Results

### CPU-step share of decode wall (last window of the run)

| model | budget | split | decode tok/s | ms/token | CPU-step share |
|---|---:|---|---:|---:|---:|
| 9B | 24 | 24 resident / 8 offloaded | 27.6 | 36.2 | **77.3%** |
| 9B | 24 | repeat (2 runs, load 3.3 / 4.8) | 27.5 / 27.4 | 36.4 / 36.5 | **77.8% / 77.7%** |
| 9B | 16 | 16 / 16 | 15.8 | 63.3 | **89.8%** |
| 9B | 12 | 12 / 20 | 13.4 | 74.6 | **92.4%** |
| 9B | 12 | repeat | 13.2 | 75.8 | **92.6%** |
| 9B | 8 | 8 / 24 | 11.1 | 90.1 | **94.6%** |
| 27B mq3-xt | 56 | 56 / 8 | 14.7 | 68.0 | **66.0%** |
| 9B | 32 | resident control | 110.5 | 9.1 | — |
| 27B mq3-xt | 64 | resident control | 42.8 | 23.4 | — |

Rate controls at the same fixture: 9B budget 24 `pcie` 23.8 tok/s (vs `cpu` 27.6);
27B budget 56 `pcie` 14.7 (vs `cpu` 14.7, a tie). Per-step CPU cost is flat in
spill size — 441–458 µs/step on the 9B across 8→24 spilled — so the token cost
scales with the number of spilled layers, which is why the idle share tracks it.

**Repeat pass, fresh interleaved processes** (`--runs 1` per process, so every
rate below is n=1 with arm-internal stdev 0.0 — these are exploratory, not
admission-grade):

| arm | decode tok/s | CPU-step share |
|---|---:|---:|
| 9B 24/8 `cpu` (2nd) | 27.5 | 77.8% |
| 9B 24/8 `cpu` (3rd) | 27.4 | 77.7% |
| 9B 24/8 `cpu` (4th, load 4.9) | 27.3 | 78.1% |
| 9B 24/8 `pcie` (2nd, load 4.6) | 23.0 | — |
| 27B 56/8 `cpu` (2nd, load 5.6) | 14.6 | 66.3% |

The idle fractions reproduce to ≤0.5% across host load 2.1–5.6, which is the
point: the ratio is what this record rests on, and it does not track load the way
an absolute rate does. **The 27B `pcie` repeat hung** — see the precondition
below.

### CPU stream vs the GPU's PCIe read of the same host RAM (4 fresh processes)

| run | A cpu alone | B gpu alone | C cpu (concurrent) | C gpu (concurrent) | C combined | CPU retention |
|---|---:|---:|---:|---:|---:|---:|
| 1 (cold) | 42.0 GB/s | 27.5 GB/s | 47.2 GB/s | 28.4 GB/s | 75.6 GB/s | 112% |
| 2 | 53.9 | 27.3 | 46.6 | 28.2 | 74.8 | 86% |
| 3 | 51.9 | 27.4 | 46.8 | 28.1 | 74.9 | 90% |
| 4 | 52.4 | 27.3 | 49.4 | 28.2 | 77.5 | 94% |

Run 1's A is a cold-clock outlier (its 112% retention is the artifact). In the
warm runs the CPU gives up 6–14% while the GPU runs at 102–103% of its solo rate,
so the two streams are close to additive up to ~75–78 GB/s. B's 27.3 GB/s agrees
with the link rate the 2026-09-26 baseline measured (27.1 GB/s) and with the
`pcie` arm's own effective spilled-byte rate (9B 8/32: 888 MB in ~33.6 ms after
the resident portion ≈ 26 GB/s), so the proxy is the mechanism, not a stand-in.

## Host capacity is a precondition, not an afterthought

The second 27B `pcie` arm **hung**, and it was a hang, not slowness: the daemon
sat at `loading layer 45/64` for 17 minutes at 99% CPU (one core), RSS 9.2 GB,
with the host at **0 GB free / 12 GB page-cache** — the same signature as the
2026-09-27 amendment's budget-48 stall at layer 62/64 with 0 GB free. It was
killed; the box recovered to 13 GB available immediately. The 27B arms' upstream
precondition is therefore *budget ≤ 56 **and** free host RAM*, checked before the
run, exactly as that amendment says. Every 27B number above came from a run that
started with ≥11 GB available; the hung one is not a data point and is not in any
table.

## Reading

- A spilled step is currently a host-sync-point-shaped serial region; handing a
  fraction of its rows to the GPU on a second stream would convert part of the
  idle fraction measured above into work. The probe bounds what that can buy: at
  most ~1.4× the spilled-byte throughput before the host DRAM ceiling, and less
  once per-step D2H/H2D and launch overhead are added.
- The idle share is *not* the win. It is the room. The token-level win is
  `idle_share × (1 − 1/overlap_factor)` minus the extra sync/launch cost of the
  split. For 9B 8/32 (77.7% idle, 1.4× ceiling) that is ≈ +25–30% at best;
  nothing here says it is reachable, only that the ceiling is not zero and is
  not obviously smaller than the user's stated "30% would still be an
  optimization" bar.
- Mostly-CPU splits are the *better* case for this scheme, not the worse one:
  idle share rises to ~95% at 24/32, and the probe says the CPU is not DRAM-bound
  until it is streaming ~52 GB/s alone. The risk there is per-step overhead and
  the second-stream join, not contention.
- This feature remains a capacity-for-speed trade. Nothing here changes the
  correctness contract or moves a byte out of VRAM.

## Not measured

- **Every rate here is n=1 per process** (`hipfire bench --runs 1`) with
  arm-internal stdev 0.0, so the *rate* columns are exploratory, not
  admission-grade — a rate claim needs ≥3 fresh interleaved processes per arm and
  the `samples` arrays. The idle fractions and the probe are this record's
  evidence; the rates are context.
- The probe's phase B is a raw `memcpy_htod` — an *upper bound* for the GPU's
  share of a split, because the real `pcie` arm reads host-mapped weights inside
  a kernel (SM occupancy and access pattern differ from a DMA copy). It is used
  because it agrees with the arm's measured effective rate (27.3 vs ~26 GB/s),
  which is what makes it a bounded proxy rather than a stand-in. A split's real
  throughput must be validated end-to-end, not extrapolated from B.
- Any end-to-end split implementation — no such mode exists; this record is the
  measurement that gates building one.
- A two-stream split's real cost: the D2H for the CPU half still drains the
  default stream today, so an implementation needs a non-draining read or a
  second stream, and that cost is entirely unmeasured here.
- Gate/serve (`--backend slots`) and prefill — both bypass `execute_steps`
  (`forward_batch_slots` → `dense_ffn_body_slots`), so neither the idle reading
  nor any split would apply to them.
- 27B splits past 8/64: `hipHostMalloc` bytes are pinned and unreclaimable, and
  32 spilled layers of this file is ~5 GB the host cannot take back (the
  amendment to the 2026-09-27 record already stalled at 16 spilled / ~2.5 GB).
- Decoded text: `hipfire bench` sets reasoning off by construction, so it is a
  rate/idle instrument here, not a coherence one. The `cpu` route's coherence
  gate remains the 2026-09-27 serve-harness read.
