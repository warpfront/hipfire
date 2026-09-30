# Amendment — CPU-exec qt 49 AVX2 kernel + per-shape trace semantics — 27B mq3-xt — 2026-09-27

**Lifecycle:** `historical`

**Amends / links to** (all **unchanged**; this file records what they could not
— the kernel the sweep record named as the next lever, and the trace-semantics
correction that makes their per-step numbers readable):

- [`2026-09-27-gfx1201-cpu-exec-offload-reproduction.md`](2026-09-27-gfx1201-cpu-exec-offload-reproduction.md)
  — independent reproduction on the pre-change binaries, including the 27B
  decoded-text parity read this file reproduces on the post-change binary (same
  610 B, same emptiness of the diff) and the 9B arm the sweep record left out.

- [`2026-09-27-cpu-exec-spill-sweep-2b-9b-27b.md`](2026-09-27-cpu-exec-spill-sweep-2b-9b-27b.md)
  — the 27B leg's `cpu` 3.1 tok/s at budget 56 is reproduced here exactly, and its
  GEMV-per-step attribution is corrected below. Its `pcie` 10.8 is the
  cold-first-process-of-a-budget reading the record itself flags (a single
  interleaved point here reads 10.9 too); the steady interleaved `pcie` rate at
  this budget is 13.2, on both binaries.
- [`2026-09-27-gfx1201-cpu-exec-offload-27b-amendment.md`](2026-09-27-gfx1201-cpu-exec-offload-27b-amendment.md)
  — its `m=12288 k=5120 → 5.70 ms/step` line is a cumulative mean (see
  "Correction"), not that shape's cost; the coverage recipe and the qt 49
  real-tensor oracle are unchanged and re-run here.

**Disposition:** exploratory record for `memory.offload_exec=cpu`
([design](../plans/partial-gpu-offload-design.md) § 6.2.1; runbook
[`../methodology/cpu-exec-offload-benchmark-handoff.md`](../methodology/cpu-exec-offload-benchmark-handoff.md)).
**Not** a product baseline, **not** an admission, **not** a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim. Not comparable across
host/model/quant/GPU/prompt/method.

## What changed (source)

Uncommitted working tree on
`7133b3c328ae53a0e9590063dc3d95d5b71c32d0` (the sweep record's HEAD, "drift = 2
docs-only commits" then):

1. **`simd::mq3g256v2_row_dot_avx2`** — a hand-written AVX2 + FMA + F16C row dot
   for qt 49 (`crates/hipfire-cpu/src/simd/x86.rs`): the group's 8 B fp16
   `[s0 z0 s1 z1]` header widens with one F16C convert, and each 3-byte chunk of
   its 96 B payload is one broadcast + one `vpsrlvd` + one mask + one `cvt` + one
   `fmadd` (the eight 3-bit codes are contiguous lanes of one 24-bit
   little-endian field, so no byte-at-a-time unpack), accumulated per 128-element
   half as `s·Σ(c·x) + z·Σx`.
2. **Per-format SIMD dispatch** — `simd::row_dot_enabled(q, requested)` replaces
   the `q == CpuQuant::Mq4G256` gate in `gemv`/`gemm`, so the decision is the
   format's own feature requirement (AVX2+FMA for qt 13, AVX2+FMA+F16C for
   qt 49) rather than a hard-coded format.
3. **`HIPFIRE_CPU_EXEC_TRACE` is per shape** — the seam accumulates
   `(calls, d2h_ns, gemv_ns, h2d_ns)` per step shape and prints a line at the
   shape's first call and at every doubling of its call count, each line's means
   being over *that shape's* calls. Before this it printed a single process-wide
   cumulative mean on every shape's line.

## Correction — the prior per-step numbers are cumulative means

The old trace divided the process-wide accumulated GEMV time by the process-wide
step count, then printed that same figure on every shape's first-sighting line.
Reproduced here with the pre-change binary
(`single_round_old_*.json`, `old_cpu_trace.err`), the whole output of a 64-token
run is eight lines:

```
m=10240 k=5120 | 1 steps on CPU | gemv=7.52ms
m=6144  k=5120 | 2 steps on CPU | gemv=5.66ms
m=48    k=5120 | 3 steps on CPU | gemv=3.82ms      <-- "m=48 costs 3.82-5.31 ms"
m=5120  k=6144 | 5 steps on CPU | gemv=3.10ms
m=17408 k=5120 | 6 steps on CPU | gemv=4.53ms
m=5120  k=17408| 8 steps on CPU | gemv=6.06ms
m=12288 k=5120 | 25 steps on CPU | gemv=5.81ms     <-- the 5.70 ms "per-step" line
m=1024  k=5120 | 26 steps on CPU | gemv=5.61ms
```

Every line is the mean of the *first N steps of the process* (`N` = the counter
on the same line), so the figures differ because the counter differs, not because
the shapes cost that much. The sweep record's "m=48 (0.25 MMAC) takes 5.31 ms
(~0.05 GMAC/s) while m=10240 takes 10.37 ms" therefore does not show a fixed
per-step floor; the three pairs are three snapshots of one rising average. The
scalar arm's *actual* per-shape cost is in the microbench below, and its actual
per-token cost is accounted for in the A/B: `cpu` 3.10 tok/s at budget 56 is
322 ms/token against the `pcie` arm's 75.8 ms/token for the same bytes — a
**+246 ms/token** CPU-side overhead: ~4.8-5.0 GB/s effective on
the spilled weights (1.18-1.24 GB over 246 ms), which is the scalar decode
(8.5-9.4 GMAC/s aggregate over 16 rayon threads, below) — not a copy and not a
fixed floor.

## Microbench — scalar vs AVX2, the real step shapes

`crates/hipfire-cpu` `gemv_with_simd(q, …, Some(false|true))` over synthetic
weights with non-power-of-two group headers (so the two summation orders genuinely
differ), one row per shape, median of 5-20 calls, 16 rayon threads, warm. Harness
kept with this record (`kernel_probe.rs`); two runs agree within ~10 %.

| shape (m × k) | bytes | scalar | AVX2 | speedup | AVX2 rate |
|---|---:|---:|---:|---:|---:|
| 17408 × 5120 | 36.2 MB | 9.44 ms | **0.94 ms** | 10.1× | 95 GMAC/s · 38.6 GB/s |
| 5120 × 17408 | 36.2 MB | 9.58 ms | **1.01 ms** | 9.5× | 88 GMAC/s · 35.9 GB/s |
| 12288 × 5120 | 25.6 MB | 6.72 ms | **0.67 ms** | 10.0× | 94 GMAC/s · 38.2 GB/s |
| 10240 × 5120 | 21.3 MB | 5.55 ms | **0.50 ms** | 11.2× | 105 GMAC/s · 42.8 GB/s |
| 6144 × 5120 | 12.8 MB | 3.72 ms | **0.33 ms** | 11.4× | 96 GMAC/s · 39.1 GB/s |
| 5120 × 6144 | 12.8 MB | 3.42 ms | **0.40 ms** | 8.7× | 80 GMAC/s · 32.3 GB/s |
| 1024 × 5120 | 2.1 MB | 0.60 ms | **0.084 ms** | 7.2× | 63 GMAC/s · 25.4 GB/s |
| 48 × 5120 | 0.1 MB | 0.048 ms | **0.024 ms** | 2.0× | latency-bound |
| **hypothetical per token** (each of the 8 shapes once per layer × 8 layers) | 1176.5 MB | 307 ms | **31.5 ms** | 9.7× | 37.3 GB/s |

The row is a *model* — the real per-token step mix is not uniform (see the trace
section below). It is the right order for the measured old `cpu` arm all the same:
that arm cost ≈298 ms/token above the zero-spill baseline, which is what a
decode-bound arm looks like, and the scalar column here is ≈307 ms for the same
bytes. The AVX2 kernel is within ~16 % of the byte rate
its qt 13 sibling reaches on this host on the same shapes (37.3 vs 44.3 GB/s per
token; 32-43 vs 54-58 GB/s per shape), against the base record's 44.5 GB/s
kernel-ceiling measurement (taken on `Mq4G256`). It streams 1.31× fewer bytes per element than qt 13 (0.406 vs 0.531
B/elem) yet lands at GMAC *parity* with it (95-105 vs 99-110 GMAC/s), so what is
left on the table is ALU (a broadcast + shift + mask + convert per eight
elements), not bytes — and 20 % of bytes is not the 9.7× this change was worth.

## End-to-end A/B — budget 56 (8 of 64 layers spilled)

`hipfire bench /path/to/models/qwen3.8-27b.mq3-xt --spec off --runs 5 --warmups 2
--max-tokens 64 --backend noslots --workload stateless --prompt-file
benchmarks/prompts/gpu_offload_probe.txt --json`, `HIPFIRE_GPU_LAYER_BUDGET=56`,
one fresh process per point, round-robin `cpu, pcie` × `old, new` × 3 rounds,
daemons killed by path between points, host otherwise idle (load 6-17 during the
points — the `cpu` arm's own 16 rayon threads account for most of it).

| arm | per-process decode medians (tok/s) | median-of-medians | vs old `cpu` |
|---|---|---:|---:|
| `cpu`, pre-change (`90436899…`) | 3.1, 3.1, 3.1 | **3.10** | — |
| `cpu`, post-change (`e39c3adb…`) | 15.2, 15.2, 15.2 | **15.20** | **+390 %** |
| `pcie`, pre-change | 13.2, 13.2, 13.2 | 13.20 | — |
| `pcie`, post-change | 13.2, 13.2, 13.2 | 13.20 | ±0 % |

- **The control is flat**: the `pcie` route never executes the CPU kernel, and it
  reads 13.20 tok/s on both binaries to the digit. The whole delta is the `cpu`
  arm. (The sweep record's 10.8 — and the 10.9 of this session's very first
  single-round `pcie` point — is the cold first-process-of-a-budget reading that
  record documents; 13.2 is what the interleaved steady state settles at.)
- **The `cpu` arm flips from a loss to a win** at this spill: −76.5 % → **+15.2 %**
  versus `pcie` (3.10 → 15.20 vs 13.20 tok/s). Context, from the sweep record's
  zero-spill control: fully resident is 40.7 tok/s (24.6 ms/token), so the spilled
  work costs ≈41 ms/token after the fix against ≈298 ms before — ~29-30 GB/s
  effective over the 8 layers' 1.18-1.24 GB. The `pcie` arm moves the same bytes
  in ≈51 ms/token, i.e. ~24 GB/s against its 27.1 GB/s measured link rate.
- Within-process spreads are 0.1 tok/s or less on every arm (`--spec off`, greedy,
  no τ) — the tightness is expected here and is not a spec-decode signal.
- Capacity parity holds: `vram_free_mb` 4346-4402 across all 12 points at the same
  budget (allocator noise), `vram_free_before_mb` identical.

Whole per-point JSON, both trace captures and the two drivers:
[`data-2026-09-27-cpu-exec-mq3-avx2/`](data-2026-09-27-cpu-exec-mq3-avx2/).

## Per-shape steady state in-model (new trace, 15,872 steps on CPU)

`HIPFIRE_CPU_EXEC_TRACE=1`, same fixture/budget, `--runs 5 --max-tokens 64`
(`new_cpu_trace.err`). Steady-state lines (last doubling reached, 512-4096 calls
per shape), all shapes `Mq3G256V2`, `0 host-mapped steps still on GPU` on every
line:

| shape (m × k) | d2h | gemv | h2d |
|---|---:|---:|---:|
| 17408 × 5120 | 0.05 ms | **0.91 ms** | 0.05 ms |
| 5120 × 17408 (rotated+residual) | 0.11 ms | **0.96 ms** | 0.07 ms |
| 12288 × 5120 | 0.08 ms | **0.70 ms** | 0.05 ms |
| 10240 × 5120 | 0.08 ms | **0.62 ms** | 0.04 ms |
| 5120 × 6144 (rotated+residual) | 0.08 ms | **0.42 ms** | 0.06 ms |
| 6144 × 5120 | 0.02 ms | **0.38 ms** | 0.04 ms |
| 1024 × 5120 | 0.03 ms | **0.12 ms** | 0.05 ms |
| 48 × 5120 | 0.03 ms | **0.05 ms** | 0.04 ms |

Summed over the eight shapes at their last printed call counts (12,800 of the
process's 15,872 CPU steps) the run spent **7.48 s in GEMV and 1.42 s in
D2H+H2D** — 0.585 ms of GEMV and 0.111 ms of copies per step on average.

Two cautions on turning that into a per-token figure:

- **The step mix is not "8 shapes × 8 layers".** The measured call ratios are
  16 : 8 : 8 : 8 : 4 : 4 : 1 : 1 (`17408×5120`, `48×5120`, `5120×6144`,
  `5120×17408`, `1024×5120`, `6144×5120`, `10240×5120`, `12288×5120`), so any
  uniform model over-counts. The process's ≈9.3 s of GEMV over the bench's 320-448
  generated tokens (5 runs of 64, ± the 2 warmup runs) is **≈21-29 ms/token**,
  against 65.4 ms/token of wall time at 15.2 tok/s — with the zero-spill control's
  24.6 ms/token of GPU work and the copies accounting for most of the rest.
- **The copies are ~19 % of the GEMV** (0.111 vs 0.585 ms per step), which is the
  same ratio the [base record](2026-09-27-gfx1201-cpu-exec-offload.md) flagged as
  above the 10 % threshold for its pre-decided mitigation (batching the copy pair
  per layer). That lever is still unexercised.

The cold first lines (`calls=1`) are 1.27 ms (`10240×5120`) and 1.02 ms
(`17408×5120`) against 0.61 / 0.91 ms at steady state — the effect that inflated
every cumulative mean in the earlier records.

## Decoded text — greedy `cpu` vs `pcie` on the post-change binary

```
HIPFIRE_GPU_LAYER_BUDGET=56 HIPFIRE_OFFLOAD_EXEC={cpu,pcie} hipfire run \
  /path/to/models/qwen3.8-27b.mq3-xt "$(cat benchmarks/prompts/humaneval_3_below_zero.txt)" \
  -t 0 -n 3072 --no-stream
```

Fresh process per arm, same prompt bytes (md5
`37c5aad9f9efe93b5c47f27256bdf149`), greedy, on the new binary. The `cpu` arm
reports `partial offload: 56 resident / 8 offloaded, i_gpu_start=8` and
`cpu exec: 8/8 spilled layers fully covered; uncovered quants: none`, so all eight
spilled layers' projections ran through the qt 49 AVX2 kernel in the decode loop.

**Both completions are 610 bytes and byte-identical** — shared prefix 610 of 610,
i.e. no divergence within this completion (`cpu.txt` / `pcie.txt` in the data
directory). It is a natural stop, not a token-budget stop, and the thinking block
is stripped from the released text.

This **reproduces** the [reproduction record](2026-09-27-gfx1201-cpu-exec-offload-reproduction.md)
§ 2, which read the same prompt, the same spill and the same 610 B on the
pre-change binary (`-n 1024`; this run used `-n 3072` and stops naturally at the
same completion). The added value here is that the qt 49 **AVX2 kernel** keeps
that parity: the numbers changed at the seventh digit and the greedy tokens did
not move.

That result is *not* a byte-identity claim for the feature: the contract is
coherence plus a measured divergence, and the 9B read in
[the base record](2026-09-27-gfx1201-cpu-exec-offload.md) diverged by one
whitespace token after 723 characters (~190 tokens) under the same greedy
settings. A ~150-token completion with no flip is a coherent read, not proof that
the paths agree forever.

## Numerical evidence

- `cargo test -p hipfire-cpu` and `--release`: 25 passed, including the f64
  oracle for `Mq3G256V2` (1e-4 relative), the AVX2-vs-scalar bound (1e-5, with
  the group headers patched to non-power-of-two values so the two summation
  orders actually differ), and the per-format dispatch truth table.
- Real-tensor production-launcher parity on this fixture (the oracle of record for
  a format with no canonical `dequantize_to_f32` arm), re-run on the new binary:

  ```
  HIPFIRE_PARITY_EXTRA_MODEL=/path/to/models/qwen3.8-27b.mq3-xt HIPFIRE_PARITY_EXTRA_QT=49 \
    cargo test -p hipfire-arch-qwen35 --release --test gpu_gemv_parity -- --ignored
    qwen3.8-27b.mq3-xt qt=49 q_proj  m=12288 k=5120  max_abs=5.364e-7 rel=1.195e-7
    qwen3.8-27b.mq3-xt qt=49 q_proj  m=12288 k=5120  max_abs=9.537e-7 rel=3.530e-7
  ```

  2 passed, 0 failed. Worst per-format row `Mq3G256V2` 3.815e-6 abs / 3.530e-7 rel
  (synthetic) against the 1e-4 tolerance; every other format's row is unchanged
  from the amendment's table to the printed digits (qt 44 8.931e-8, qt 47
  1.933e-7, qt 48 2.857e-7, qt 20 6.652e-7), so the SIMD dispatch change moved
  nothing outside the two formats it targets and the qt 13 row (9.319e-7 rel,
  now on its existing AVX2 kernel).

## Fixture identity

| artifact | digest |
|---|---|
| source HEAD | `7133b3c328ae53a0e9590063dc3d95d5b71c32d0` + this change, uncommitted |
| `target/release/daemon` (new, **the binary every number here was measured on**) | `e39c3adb85f608db1439b128471764e8` |
| `target/release/daemon` (rebuild of the final tree) | `df4eacc0cf9fd47bd95e30e4311c1836` — see the digest note |
| `target/release/hipfire` (new) | `57a0cf5795a8ce376d274eba78bc733a` |
| `target/release/daemon` (pre-change) | `904368995cddb8cda8e82a7f8d31ae96` |
| `target/release/hipfire` (pre-change) | `030f080ce4c3d68ca038a614059d21b2` |
| `/path/to/models/qwen3.8-27b.mq3-xt` | `80bb9198e6a565fc006b2ae1b7c89eca` (11,777,616,896 B) |
| `benchmarks/prompts/gpu_offload_probe.txt` | `5835c71e471849b4a72e1dc8e39695e7` (59 tokens) |

**Digest note.** One edit landed after the measurement: a doc comment on
`prepare_activation` in `cpu_exec.rs` (an earlier `let … else` reflow predates
it). Rebuilding with that comment removed reproduces `e39c3adb` **exactly**, and
restoring it lands on `df4eacc0` again on a second build — so the build is
byte-reproducible for a fixed source, and the daemon md5 is simply not stable
across comment-only edits (line numbers feed the `.llvm.<hash>` symbol suffixes
the linker emits; `hipfire` itself, which does not link this module, stayed at
`57a0cf57` throughout). The `.text` sections of the two daemons are byte-identical
— `objcopy -O binary --only-section=.text`, 33,516,602 B, md5
`a766f371c150cc961eed1a65ffd4971e` — so the measured artifact and the final tree
are the same code. Compare `.text`, not the binary md5, when adjudicating a
digest mismatch on this record.

Host: 1× RX 9070 XT `gfx1201` (16304 MB), ROCm/HIP 7.2, Ryzen 7 7800X3D
(8c/16t), 28 GB RAM, THP `[always]`, full desktop session (load 6-17 during the
points, self-inflicted by the `cpu` arm's rayon threads; the interleaved 2×2
design is what makes the arms comparable under it).

**ISA note (measured, 2026-09-27).** This part is Zen 4 and *does* report
AVX-512F — `is_x86_feature_detected!("avx512f") == true`, confirmed by a runtime
probe and not only by `/proc/cpuinfo`. The base record's "AVX-512 is not
available on this host (Zen 3)" is wrong on both counts and should not be
repeated.

**The consequence is a design rule, not a licence.** The qt 49 kernel is gated on
AVX2 + FMA + F16C (the qt 13 kernel on AVX2 + FMA) — exactly the features
[`row_dot_enabled`](../../crates/hipfire-cpu/src/simd/mod.rs) detects per format —
and that is a portability choice, not a consequence of this box's ISA. The
project's own record history already contains a host without AVX-512: `k9lin` is
a **Ryzen 9 3900X (Zen 2, AM4, 24 threads)** in
[`2026-07-26-gfx1201-retained-pm4-reference.md`](2026-07-26-gfx1201-retained-pm4-reference.md)
§ host table, and Zen 2 has AVX2 + FMA + F16C with no AVX-512 at all;
`hipfire-runtime/src/cpu_router.rs` likewise reasons about a Zen 2 in the same
CPU-side role. So "AVX-512 is available on the measured host" is **not** a reason
to specialise these kernels: an AVX-512 row dot without its own
`is_x86_feature_detected!` gate would look like a pure win in a benchmark run on
this 7800X3D and be a `SIGILL` on any host like `k9lin`. Any such path must bring
its own detection arm (and its own scalar fallback) or not exist.

## Not measured (state these with any citation)

- **The slots/serve body and prefill** — batched GEMM kernels that never enter
  `execute_steps`; this change cannot move them (prefill medians here are 93-96
  tok/s on all 12 points, and the 59-token prompt makes that number launch-bound).
  The decoded-text read above is the sequential route only.
- **The other V2 formats.** qt 44 (`Mq4G256V2`), qt 47, qt 48 and qt 50 are the
  same gap as qt 49 was — still scalar. A model that is all-qt 44 pays the 27B's
  pre-change price; the fix is the same shape of kernel.
- **Budget 48 (16 spilled)** on this fixture — the earlier record stalled in load
  there, and nothing in this change alters the footprint.
- **A second prompt/genre for the text read.** One completion is not a genre
  sweep; the divergence contract needs a per-genre read before any claim about
  agreement rate.
