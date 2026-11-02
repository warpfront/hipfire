# CPU-exec offload — independent reproduction + 27B decoded-text parity — gfx1201 — 2026-09-27

**Lifecycle:** `historical`

**Amends / links to:** [`2026-09-27-gfx1201-cpu-exec-offload.md`](2026-09-27-gfx1201-cpu-exec-offload.md)
(the base record; unchanged — this file is an independent reproduction and a
completion of that record's own "open items", not a correction of its numbers)
and [`2026-09-27-gfx1201-cpu-exec-offload-27b-amendment.md`](2026-09-27-gfx1201-cpu-exec-offload-27b-amendment.md).
Same host, same binaries as those records unless stated.

**Disposition:** exploratory measurement record for `memory.offload_exec=cpu`
(the CPU-exec stage of partial GPU offload; design + coverage contract in
[`docs/plans/partial-gpu-offload-design.md`](../plans/partial-gpu-offload-design.md) § 6.2.1).
**Not** a product baseline, not an admission, not a `docs/BENCHMARKS.md` claim.

This record exists to (a) independently reproduce the base record's headline on
the binaries actually present in this checkout — confirming the effect is real
and not a one-pass artifact — and (b) close that record's §9 "open items", most
importantly the 27B decoded-text parity read, which its author never completed
(every attempt hit the reasoning-budget gate).

## Fixture identity (measured in this run)

- Host: 1× AMD Radeon RX 9070 XT, `gfx1201`, 16304 MB total VRAM, ROCm/HIP 7.2.
  CPU: AMD Ryzen 7 7800X3D (8c/16t), 28 GB RAM, THP `[always]`.
- Source tree HEAD at measurement time: **`7133b3c328ae53a0e9590063dc3d95d5b71c32d0`**
  (one commit past the base record's build). Binaries are **byte-identical** to
  the base record's claimed build, so every anchor in that record still applies:
  `target/release/daemon` md5 `904368995cddb8cda8e82a7f8d31ae96`;
  `target/release/hipfire` md5 `030f080ce4c3d68ca038a614059d21b2`.
- Model: `~/.hipfire/models/qwen3.5-9b.mq4`, md5
  `31a8d8dc7603226801b08d8319015602` (32 layers, 114.9 MB/layer).
- Model: `/path/to/models/qwen3.8-27b.mq3-xt`, md5 `80bb9198e6a565fc006b2ae1b7c89eca`
  (11,777,616,896 B; 64 layers; **every one of its 497 projections is qt 49 /
  `MQ3G256V2`** — covered by the cpu path with no canonical host decoder).
- Prompt: `benchmarks/prompts/humaneval_3_below_zero.txt`, md5
  `37c5aad9f9efe93b5c47f27256bdf149`.
- Flags: `--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots
  --workload stateless --prompt-file <probe|humaneval> --json` (rate);
  `run … --spec off --temp 0 -n 1024` (decoded-text). Offload via
  `HIPFIRE_GPU_LAYER_BUDGET` / `HIPFIRE_OFFLOAD_EXEC`; arms interleaved
  `pcie, cpu, …`, one fresh process per run.

## What this confirms

### 1. Arm A reproduced on current binaries: pcie 21.7 / cpu 25.1 tok/s (+15.7%)

9B mq4, budget 24 → `i_gpu_start=8` (8 of 32 layers host-mapped), three fresh
processes per arm interleaved. Decode medians of the three per-run medians:

| arm | per-run decode tok/s | median |
|---|---|---|
| `pcie` | 19.1 / 21.7 / 21.8 | **21.7** |
| `cpu` | 25.0 / 25.1 / 26.3 | **25.1 (+15.7%)** |

The base record measured +11.6% (21.6 / 24.1) under an otherwise-idle host; this
reproduction is +15.7% under a host whose load averaged ~3–5.7 for much of the
pass (KDE Baloo indexing + GUI). The two arms move the same byte count per
token; the cpu arm reads from host DRAM, the pcie arm across the link — so a
genuine win is expected and this reproduces it.

**Contamination canary.** The `pcie` arm uses no CPU threads, so under a
contended box it would be the *stable* one while the `cpu` arm degraded (the
exact failure mode that produced the base record's superseded 3–6× "cpu slower"
readings). Here it is the reverse: `pcie` medians sit at 21.7 — within ~1% of
the base-record anchor despite load climbing to ~5.7 — while `cpu` is tightly
clustered (25.0–26.3) and clearly ahead. That pattern (pcie tracks its anchor,
cpu stable-and-faster) is the signature of a real win under noise, not a
contaminated run. No arm read anywhere near the retracted ~3 tok/s floor.

**Capacity parity (§4.C).** `vram_free_before_mb` = 16182 across all six arms;
runtime `gpu.vram_free_mb` = 11066–11122 (identical within noise). This feature
moves no bytes out of VRAM, so it must not appear to — confirmed.

**Coverage.** Every arm printed `partial offload: 24 resident / 8 offloaded,
i_gpu_start=8`. The three `cpu` arms additionally printed
`cpu exec: 8/8 spilled layers fully covered; uncovered quants: none` and
`hipGraph capture disabled (CPU-executed steps present)` — the spilled qt13
projections ran on the CPU and graph capture correctly disabled rather than
capturing a mixed pcie/cpu graph.

### 2. 27B decoded-text parity — byte-identical output (closes base §9 "never done")

The base record's §9 listed the 27B decoded-text read as never completed: every
attempt hit `daemon error: … open think span at end of generation` and released
no text. This run closed it without a reasoning-config change, by giving the
model far more tokens than its think block needs (`-n 1024 --temp 0 --spec off`),
so no read was gate-failed.

- Budget **56** → `i_gpu_start=8` (8 of 64 layers host-mapped). Footprint-safe:
  ~1.2 GB pinned `hipHostMalloc`; the base amendment's budget-48 stalls were
  host-capacity, and at this run MemAvailable was ~20 GB.
- Coverage: `partial offload: 56 resident / 8 offloaded, i_gpu_start=8` and
  `cpu exec: 8/8 spilled layers fully covered; uncovered quants: none` — all eight
  spilled **qt 49** projections executed on the CPU.
- Greedy (`--temp 0`) completion of the `humaneval_3_below_zero.txt` prompt,
  `pcie` vs `cpu`, compared byte-for-byte:

```
diff pcie-vs-cpu → (empty)
byte sizes: pcie=610  cpu=610
```

The two arms produced **byte-identical** output — zero first-divergence offset,
to the last byte. This is stronger than the base record's 9B finding (which was
a near-tie: 723 characters agreeing then a single inserted newline). At 27B with
every spilled projection on the CPU, `cpu` and `pcie` are not merely
llama.cpp-level close — they are identical for this completion.

## Supplementary (in flight at time of writing)

A focused pcie-controlled **2B crossover bracket** (spill 0/3/6 = budgets
24/21/18, pcie+cpu ×3 interleaved) was launched to independently confirm the base
record's §6 claim that "the cpu arm *loses* on the 2B at every spill ≥ 3" — i.e.
to pin *where* the arms cross rather than assert a single direction. Its numbers
are recorded in the follow-up amendment below once it finished; they do not alter
§1–2 above, which stand on their own.

## What this run did NOT measure (state in any report built on it)

- The slots/serve path (`forward_batch_slots` → `dense_ffn_body_slots`) — batched
  GEMM kernels that never enter `execute_steps`; a `serve_harness` number shows
  almost none of this feature.
- Prefill rate (GPU-side batched kernels; the seam cannot touch it).
- Formats outside `CpuQuant` staying on PCIe by design (`Q8HFQ`, `MQ8G256`, the
  HFP4/MFP4 family, PARO) — not present in these two fixtures.
- The retained-replay (Redline) route under `offload_exec=cpu` (refused at load,
  per the base record §"Correctness gate").
