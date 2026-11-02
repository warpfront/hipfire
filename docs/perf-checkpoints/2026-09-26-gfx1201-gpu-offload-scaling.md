# gfx1201 partial GPU offload — cost per spilled layer — 2026-09-26

**Lifecycle:** `historical`

**Disposition:** exploratory characterization of the `memory.gpu_layer_budget`
knob on a single consumer gfx1201 card. It is **not** a product baseline, not a
`docs/BENCHMARKS.md` claim, and not comparable to the 4× R9700 records already in
this ledger — different host, CPU, and memory subsystem. It owes a
claim-scoped `scripts/serve_harness.py` pass before being restated as a
performance claim.

## Fixture identity (measured)

- Host: 1× AMD Radeon RX 9070 XT, `gfx1201`; 17,095,983,104 B VRAM (15.92 GiB).
  HIP runtime `7.2`; `rocm-smi` 4.0.0 / ROCm-LIB 7.8.0.
- CPU: AMD Ryzen 7 7800X3D, 8c/16t, 1 NUMA node.
- Memory: DDR5-6000 CL30 (operator-stated). **Channel count unverified** —
  `dmidecode` needs root. The 96 GB/s figure below is the 2-DIMM *theoretical*
  spec, not a measurement.
- PCIe: `0000:03:00.0`, `current_link_speed = 32.0 GT/s`, `current_link_width = x16`
  — Gen5 x16 per sysfs (unprivileged read). **But see "Link under-delivers": the
  measured throughput is Gen4-class.**
- THP: `/sys/kernel/mm/transparent_hugepage/enabled = [always] madvise never`.
- Tree: `feat/partial-gpu-offload` @ `9bb599229499d55f128913c4e7cbb0c529ad2b31`,
  clean of tracked modifications.
- **Binaries (the fixture identity that matters):** `hipfire` SHA-256
  `ed032d4f381b751e5f460c1d6a96326b926ac040720abfe04858e23c6cf6c855`; `daemon`
  SHA-256 `b52aa3f2d90713a1bda5194bf1a2fdcdd712ae95d07916d77117a5a0b3eb7010`.
  Both sweeps below were measured with **these** binaries.

  ### Reproducing the binaries

  The build is byte-reproducible per source state: two independent
  `cargo build --release` runs at this HEAD produced `ed032d4f`/`b52aa3f2` both
  times. A reader who rebuilds at `9bb5992` will get these digests.

  The sweep was **not** first run on these binaries. The original measurement
  used `hipfire cbe86149…` / `daemon 86807e57…`, built 16:40:47 — sitting on top
  of older incremental artifacts, ~9 minutes before `9bb5992` was committed
  (16:49:57) with a test-only change to `crates/rdna-compute/src/dispatch.rs`
  (a VRAM-reclaim polling loop and its assertion message, inside `#[test]` code
  not compiled into `--release`). Those digests are **not** regenerable from this
  tree, which is why the sweep was re-run rather than merely re-labelled.
  Re-running reproduced the curve to within 2% (2B fully-resident 255.4 → 250.3
  tok/s; full spill 29.7 → 29.5), so the correction changed no conclusion.
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, SHA-256
  `b6eddc54931a1daa28c32fc8381a72d982eded30093ddf13a6a71049a695066f`;
  59 tokens, md5 `5835c71e471849b4a72e1dc8e39695e7` — byte-identical at every
  point of both sweeps.
- Models: `qwen3.5-2b.mq4` SHA-256 `bb386f7b…77badc6a`;
  `qwen3.5-9b.mq4` SHA-256 `829a84c7…f1260b3`.

Method: `hipfire bench --spec off --backend noslots --workload stateless --json`,
5 runs / 2 warmups per point, 128 prompt / 64 generated tokens, fresh daemon per
point with the previous one stopped and reaped first.

## Control surface

`memory.gpu_layer_budget` (env `HIPFIRE_GPU_LAYER_BUDGET`), resolved once per
process at load, counts layers kept **on** the GPU, mapping to
`i_gpu_start = n_layers − budget` — the length of the **prefix** spilled to
host-mapped RAM (`crates/hipfire-arch-qwen35/src/qwen35/config.rs:978`).

`Full` (the unset default) is deliberately **silent**; only that case returns
`None` from `residency_report` (`config.rs:1052`). So a missing `partial
offload:` line means the budget never reached the daemon, not that nothing
spilled. Every point below carries the daemon's reported `i_gpu_start`, and the
harness asserts it equals the requested split.

## Results

Decode tok/s, median of 5:

| spilled (2B, of 24) | decode | | spilled (9B, of 32) | decode |
|---:|---:|---|---:|---:|
| 0 | 250.3 | | 0 | 101.0 |
| 3 | 127.6 | | 4 | 35.2 |
| 6 | 86.2 | | 8 | 21.4 |
| 9 | 64.6 | | 12 | 15.4 |
| 12 | 51.9 | | 16 | 12.0 |
| 15 | 43.6 | | 20 | 9.8 |
| 18 | 37.8 | | 24 | 8.3 |
| 21 | 32.9 | | 28 | 7.2 |
| 24 | 29.5 | | 32 | 6.4 |

The two datasets are distinct: 24 vs 32 layers, **no shared decode value**, and
fully-resident points 2.5× apart. They resemble each other in shape only.

### Host memory is not a confound — but the obvious metric cannot show it

`ram_available_mb` (MemAvailable) spread only **281 MB** across the 9B sweep,
which would read as "no host pressure". That is a **blind spot, not a result**:
the HFQ model file is `mmap`'d (`memmap2::Mmap::map`, `hfq.rs:515`), so spilled
weights are **file-backed page cache, not anonymous memory**, and do not reduce
MemAvailable.

Recording page cache alongside it settles the question:

| sweep | MemAvailable spread | page cache (`Cached`) spread |
|---|---:|---:|
| 2B (spills 730 MB) | 253 MB | see CSV |
| 9B (spills 3,676 MB) | 281 MB | **4,069 MB** |

The metric that can actually see spilled weights moved 4 GB, while the one that
cannot moved 281 MB. Host memory is therefore demonstrably *not* a confound —
but only because it was measured with a metric that responds. An earlier draft
of this record asserted the same conclusion from MemAvailable alone; that
assertion was unsupported and is corrected here.

## Regression

Token time is **linear in spilled-layer count**, intercept landing on the
fully-resident compute time:

| model | fit (ms/token) | R² | t₀ measured |
|---|---|---|---|
| 2B | `t = 4.13 + 1.247·n` | 0.99985 | 4.17 ms |
| 9B | `t = 10.01 + 4.589·n` | 0.99996 | 10.11 ms |

R² ≈ 0.9999 on both: the cost is additive per layer with no interaction term.
This is the assumption-free core result, and it is why there is no cheap
partial-offload region (2B has lost 49% of decode after 3 of 24 layers).

## Deriving the host-read rate — and the divisor error

`c` is a **marginal** cost, not a host-read time: spilling layer *i* replaces a
VRAM GEMV with a host GEMV, so `c = t_host − t_vram`, and
`t_host = c + t₀/n_layers`.

The divisor is the sum of `data_size` over tensors that **have a layer index**,
not the file size. `embed_tokens` / `lm_head` / `norm` carry no layer index and
are always device-resident (`serve_engine.rs:227`), so they never cross the host
boundary per token. Measured from the HFQ tensor index
(`benchmarks/hfq_bytes.py`):

| model | tensor payload | layer-indexed | always-resident | true per-layer | naive file/n_layers |
|---|---:|---:|---:|---:|---:|
| 2B | 1,270,848,128 | 730,499,712 (57.5%) | 540,348,416 (42.5%) | **30.44 MB** | 52.95 MB (+74%) |
| 9B | 5,297,456,128 | 3,676,414,976 (69.4%) | 1,621,041,152 (30.6%) | **114.89 MB** | 165.55 MB (+44%) |

With the correct divisor:

| model | per-layer | t_host/layer | implied rate | % of *observed* ceiling |
|---|---:|---:|---:|---:|
| 2B | 30.44 MB | 1.419 ms | **21.4 GB/s** | 79% |
| 9B | 114.89 MB | 4.902 ms | **23.4 GB/s** | 86% |

**That column is an upper bound on code headroom, not an efficiency score.** Its
denominator is the *observed* host-boundary ceiling, which is itself ~2× below
spec. The shortfall between 23.4 GB/s and a healthy Gen5 x16 (~55 GB/s) cannot
be attributed to code from this data. See "Why '15–22% code headroom' is not a
claim this data supports" before reading anything into the percentages.

**An earlier draft reported 34–38 GB/s. Retracted.** It came from the naive
divisor, was 44–74% too high, and was *above what this link can physically
deliver* — which is what exposed it. It also produced a spurious
"size-dependent ceiling" (43.3 vs 36.2 GB/s); with the VRAM baseline removed the
two models agree within noise, and with the correct divisor both sit *below* the
measured hardware ceiling.

At full spill `t = t₀ + n·c = n·t_host`, so a byte-derived rate and the
fit-derived rate are algebraically the same quantity — not independent
corroboration. Only the linearity and the plateau are assumption-free.

**Assumptions, load-bearing and stated:**

1. `t_vram/layer = t₀/n_layers` assumes the fully-resident time spreads evenly
   across layers. If layers overlap or are unequal this is an **upper** bound on
   `t_vram/layer`, making `t_host/layer` an underestimate and the rate an
   **over**estimate. 21.4 / 23.4 GB/s are floors, not ceilings.
2. Bytes/token = layer-indexed tensor bytes × n_spilled, i.e. each spilled byte
   crosses once per token. Not independently verified; a re-read would raise the
   byte count and lower the rate.

## Is the ceiling the transport, or the code?

**The transport. How much is left for the code cannot be determined from this
data.** Settled by a direct control measurement
(`benchmarks/link_bw_probe.cpp`, 1 GiB pinned host buffer):

| path | GB/s |
|---|---:|
| device read of **host-mapped** memory | **27.1** |
| `hipMemcpy` D2H | 28.1 |
| `hipMemcpy` H2D | 28.3 |
| device read of VRAM (control) | 617.9 |

The VRAM control is what makes the host numbers meaningful: 618 GB/s is ~97% of
the 9070 XT's 640 GB/s spec, so the kernel saturates properly and 27.1 GB/s is
the link, not the probe. A first run during a saturating `cargo build --release`
read 27.3 / 28.5 / 28.3 / 621.2 GB/s; the values are load-independent to within
1%, so both runs are recorded. **Probe-time load average: 2.44 (1 min),
3.07 (5 min), 4.02 (15 min) at the first run** — decaying, not saturated, and
the re-run on a quieter machine reproduced it.

The offload path's 21.4–23.4 GB/s is **79–86% of the link's observed ceiling**.

### Why "15–22% code headroom" is not a claim this data supports

That percentage uses the observed 27.1 GB/s as its denominator, and **the
observed ceiling is itself the anomaly**: it is Gen4-class on a link that
advertises Gen5 x16 (see below). If the link is degraded, the entire 14–21%
shortfall is explained by the link and the code is at or near optimal. If the
link is healthy and something else caps the probe, the shortfall could be mostly
code. **These two explanations are not separable from throughput measurements
on this host.**

So the defensible statement is: *offload sustains 21.4–23.4 GB/s, at or below a
host boundary that is itself ~2× below spec.* The code-side gap is **bounded
above** by 14–21% and its true value is unknown. Any figure below that bound is
speculation.

### The page-size hypothesis is dead (measured, not assumed)

`benchmarks/thp_bw_probe.cpp`, three device-read arms at 1 GiB, with smaps
characterising the actual backing:

| arm | AnonHugePages | GB/s |
|---|---:|---:|
| A `hipHostMalloc` + Mapped (today's offload path) | 1024 MiB | 27.2 |
| B anon mmap + `MADV_HUGEPAGE` | 1024 MiB | 27.3 |
| C anon mmap, no madvise | 274 MiB | 26.8 |

Despite `THP = [always] madvise never`, `hipHostMalloc` **already** backs its
buffer with ~1 GiB of anonymous huge pages, and all three arms land within
1.5%. There is no page-size win available: `madvise(MADV_HUGEPAGE)` on the host
weight buffer would buy ~0.4%. This hypothesis was raised on the strength of the
incorrect 34–38 GB/s figure and is recorded here as **tested and refuted**.

This also falsifies the premise the hypothesis rested on: "pinned
`hipHostMalloc` buffers are normally 4K-paged" is **false on this host** — the
allocation carries 1024 MiB of `AnonHugePages`. That premise was a general
expectation about pinned memory, not a measurement, and this box does not
satisfy it.

*Provenance:* an initial run of this probe overlapped a saturating
`cargo build --release`, so its absolutes were taken under memory-bandwidth
contention (26.4 / 27.2 / 27.1 GB/s). The A-vs-B-vs-C comparison remained valid
— a uniform load depresses all arms equally — and the table above is a re-run on
an idle machine. The conclusion is unchanged under both.

### Link under-delivers

Three independent host-touching paths — SM-driven mapped read, D2H, H2D — all
land at **27–28.3 GB/s**. That is Gen4-class throughput on a link sysfs
advertises as 32 GT/s x16 (63.0 GB/s raw, ~50–55 GB/s practical for large-TLP
streaming). The link binds offload regardless of software, and the discrepancy
is itself unexplained: candidates include a Gen4-electrical slot, lane
downtraining under sustained load, or a driver/APS interaction. **This is the
most actionable open item in this record** — it caps offload on this machine by
roughly 2× versus the advertised link.

### On the memory subsystem

DDR5-6000 2-DIMM theoretical is 96 GB/s against a measured 27.1 GB/s
host-boundary ceiling, so DRAM is not binding — the transport is, before DRAM is
reached. CL30 is CAS latency and these are bulk streaming reads, so latency
does not bind and tighter timings would not move the number. The 7800X3D's 3D
V-Cache is likewise irrelevant: the GPU's reads traverse the root complex and
PCIe, not the CPU L3.

**Practical consequence:** faster or wider RAM buys nothing for offload. The
levers are VRAM capacity (avoid spilling) and link generation or width — not
memory speed or CAS latency.

## Correctness gate

Offload is where a broken weight prefix could still emit plausible tokens, so
text was compared directly. `hipfire run`, greedy `-t 0`, 200 tokens,
`thinking=off` (the persisted key was removed afterwards; `config.toml` verified
back to having no reasoning keys), same prompt file:

- fully-resident (`HIPFIRE_GPU_LAYER_BUDGET=24`): 769 bytes, exit 0
- fully-spilled (`HIPFIRE_GPU_LAYER_BUDGET=0`): 769 bytes, exit 0
- **byte-identical**, and correct coherent `longest_substring` output.

Both runs gated on `exit==0 && bytes>0`. An earlier attempt produced two empty
files and a vacuous "IDENTICAL"; that is not counted as a pass.

`bench --json` carries no generated text, so the eyeball necessarily goes
through `hipfire run`, whose prompt arrives as `"$(cat file)"` rather than
`--prompt-file`. Content is equivalent but the routes are not interchangeable
for tok/s; no tok/s from the eyeball is reported here.

## What would settle the remaining questions

- Diagnose the Gen4-class link on a Gen5 x16 slot (board lane routing, negotiated
  speed under sustained load, driver ASPM state). Worth more than any further
  offload micro-optimization.
- Confirm the read-once assumption with a hardware counter (reads retired vs
  bytes) during one offloaded decode step; closes assumption 2.

## Reproduction

```bash
./benchmarks/gpu_offload_sweep.sh qwen3.5-2b.mq4 --runs 5 --warmups 2
./benchmarks/gpu_offload_sweep.sh qwen3.5-9b.mq4 --runs 5 --warmups 2
python3 benchmarks/plot_offload_sweep.py
python3 benchmarks/hfq_bytes.py ~/.hipfire/models/qwen3.5-2b.mq4
hipcc -O3 -o /tmp/link_bw_probe benchmarks/link_bw_probe.cpp && /tmp/link_bw_probe $((1<<30)) 8
hipcc -O3 -o /tmp/thp_bw_probe   benchmarks/thp_bw_probe.cpp   && /tmp/thp_bw_probe   $((1<<30)) 6
```

Per-point JSON and `points.csv` land under
`benchmarks/results/gpu_offload/<model-stem>/` (scratch, not committed); figures
render to `offload_sweep_<model>.png`.
