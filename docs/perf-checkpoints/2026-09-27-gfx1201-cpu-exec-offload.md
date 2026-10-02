# CPU-executed projection ops for host-mapped offload — gfx1201 — 2026-09-27

**Lifecycle:** `historical`

**Disposition:** exploratory record for `memory.offload_exec=cpu` (the CPU-exec
stage of partial GPU offload; design and coverage contract in
[`docs/plans/partial-gpu-offload-design.md`](../plans/partial-gpu-offload-design.md)
§ 6.2.1). It is **not** a product baseline, not an admission, and not a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim. Product claims live in
`BENCHMARKS.md`; validation routes in [`docs/VALIDATION.md`](../VALIDATION.md).

This record exists because the feature's first end-to-end measurement said the
CPU arm was ~6× *slower* than the PCIe arm, and that number was wrong: those runs
shared the host with a full-workspace `cargo build --release` and a second
benchmark sweep. Everything below was re-measured with the host otherwise idle.
The superseded readings are named in "Superseded" rather than deleted, because
the way they were produced is the reusable lesson.

## What this establishes

1. **The CPU arm beats the PCIe arm at the same spilled count on this host.**
   9B mq4, 8 of 32 layers spilled: **24.1 vs 21.6 tok/s decode (+11.6%)**,
   three interleaved fresh processes per arm. At 16 of 32 spilled: **14.0 vs
   11.0 (+27%)**. The two arms move the same byte count per token; the CPU arm
   reads it from host DRAM, the PCIe arm across the link.
2. **The CPU side is at ~80% of its own ceiling, and the ceiling is the kernel,
   not the design.** Per-step steady state (trace, warm): GEMV 0.34–0.49 ms,
   D2H 0.04 ms, H2D 0.05 ms. Per token that is 72 steps × ≈0.53 ms ≈ 38 ms
   against a measured 41.5 ms/token, i.e. the wall time is fully accounted for
   by the per-step work. The same kernel streams a 195.8 MB host-mapped weight
   buffer at **44.5 GB/s** with 16 rayon threads (44.8 GB/s over a plain heap
   `Vec`), so the remaining headroom is the per-step copy pair plus small-shape
   inefficiency, not codegen. AVX-512 is not available on this host (Zen 3).
3. **The CPU path is deterministic and thread-count-insensitive.** Two fresh
   processes with 16 rayon threads produced byte-identical text, and so did
   `RAYON_NUM_THREADS=1`: the per-row accumulation order does not depend on the
   pool, so the device/CPU divergence cannot be explained by CPU-side scheduling.
4. **The end-to-end divergence is a near-tie on a whitespace token, not drift.**
   Greedy, 9B, 8 spilled: the `pcie` and `cpu` texts agree for **723 characters
   (≈190 tokens)** and then differ by a single inserted newline — `pcie` emits
   `left += 1` then `        char_set.add(s[right])`, `cpu` emits the same two
   lines with a blank line between them — and both continue with the same next
   construct. A wrong
   decode of a projection would change identifiers and structure, not add a
   blank line. After the flip the histories differ, so the completions differ
   (2734 vs 2329 chars) — expected under greedy decode, and the reason the
   contract for this feature is llama.cpp-level coherence plus a *measured*
   divergence instead of byte-identity.
5. **Per-step numeric fidelity is 6.7e-7 relative, measured per format.**
   `crates/hipfire-arch-qwen35/tests/gpu_gemv_parity.rs` compares the production
   launcher against `hipfire_cpu::gemv` on identical bytes: worst case 6.7e-7
   (`Mq3G256Lloyd`, real 2B tensor) against a 1e-4 tolerance, several formats
   bit-exact. Separately, `hipfire_cpu::dequant_group` reproduces the canonical
   decoder bit-for-bit over 1695 real tensors of the local fixtures
   (`crates/hipfire-runtime/tests/cpu_quant_cross_check.rs`).
6. **Two silent bugs were found by this measurement path, not by unit tests.**
   Recorded here because they are the case for end-to-end greedy text as the
   acceptance gate for this feature:
   - the per-channel **AWQ** divide was missing (the launcher divides the
     activation *inside the rotation*; the fixtures carry 138 sidecars) → a
     per-channel scale error on every projection;
   - `Prerotated` inputs were rotated a second time → a silent `R²`.
   Both produce plausible-looking activations, so per-tensor parity alone could
   not see them: it used the same convention on both sides.

## Fixture identity (measured)

- Host: 1× AMD Radeon RX 9070 XT, `gfx1201`, 16304 MB total VRAM, ROCm/HIP 7.2.
- CPU: AMD Ryzen 7 7800X3D (8c/16t), 28 GB RAM, THP `[always]`.
- Model: `~/.hipfire/models/qwen3.5-9b.mq4`, md5
  `31a8d8dc7603226801b08d8319015602` (5,297,456,128 B payload).
- `target/release/daemon` md5 `6b2f85588aac1abdbd233e415044eb12`;
  `target/release/hipfire` md5 `7075cd546b1ce61d39a36d59ce149cba`.
- Prompt: `benchmarks/prompts/humaneval_3_below_zero.txt`, md5
  `37c5aad9f9efe93b5c47f27256bdf149` (129 tokens — a decode measurement, not a
  prefill one).
- Offload: `HIPFIRE_GPU_LAYER_BUDGET=24` → `i_gpu_start=8`, i.e. 8 of 32 layers
  host-mapped; `HIPFIRE_OFFLOAD_EXEC` = `pcie` | `cpu`; `HIPFIRE_VERIFY_GRAPH=0`.
- Load line on the `cpu` arm:
  `cpu exec: 8/8 spilled layers fully covered; uncovered quants: none`, and the
  trace's coverage counter reads `0 host-mapped steps still on GPU`.

## Method

`hipfire bench qwen3.5:9b --spec off --runs 3 --warmups 2 --max-tokens 128
--backend noslots --workload stateless --prompt-file <prompt> --json`, fresh
process per run, arms interleaved `pcie, cpu, pcie, cpu, …`, nothing else
running on the host (no build, no second sweep). Decode medians of the three
per-run medians. The `noslots` backend is the path this feature accelerates: the
slots/serve body uses batched GEMM kernels that never enter `execute_steps`.

## Results

### Interleaved arms, 8 of 32 layers spilled (`--max-tokens 128`)

| round | `pcie` decode tok/s | `cpu` decode tok/s |
|---|---|---|
| 1 | 21.6 | 24.1 |
| 2 | 21.5 | 24.0 |
| 3 | 21.9 | 24.3 |
| **median** | **21.6** | **24.1 (+11.6%)** |

`vram_free_mb` is identical across arms (11066 / 11094 / 11122, `vram_free_before_mb`
16182) — this feature moves no bytes out of VRAM and must not appear to.
Effective spilled-byte rate at this count (888 MB/token): `cpu` ≈ 23 GB/s,
`pcie` ≈ 17 GB/s (the link's measured 27.1 GB/s in
[`2026-09-26-llamacpp-offload-scaling-baseline.md`](2026-09-26-llamacpp-offload-scaling-baseline.md),
minus per-step and sampling overhead).

### Spill curve (`--max-tokens 64`, one interleaved pass)

| spilled layers | `pcie` tok/s | `cpu` tok/s | `cpu` ms/token |
|---|---|---|---|
| 0 (budget 32) | — | 65.7 | 15.2 |
| 2 (budget 30) | — | 54.1 | 18.5 |
| 4 (budget 28) | — | 38.4 | 26.0 |
| 8 (budget 24) | 19.7 | 26.1 | 38.3 |
| 16 (budget 16) | 11.0 | 14.0 | 71.4 |

### Steady-state per-step cost (`HIPFIRE_CPU_EXEC_TRACE=1`, warm)

| step shape (m × k) | D2H | GEMV | H2D |
|---|---|---|---|
| 8192 × 4096 | 0.04 ms | 0.98 ms | 0.06 ms |
| 4096 × 4096 | 0.03 ms | 0.64 ms | 0.04 ms |
| 4096 × 12288 | 0.04 ms | 0.49 ms | 0.05 ms |
| 12288 × 4096 | 0.04 ms | 0.41 ms | 0.04 ms |
| 1024 × 4096 | 0.04 ms | 0.45 ms | 0.05 ms |
| 32 × 4096 | 0.05 ms | 0.44 ms | 0.04 ms |

(Running means at the first observation of each shape; the first row is still
warming. Copies are ~0.09 ms/step ≈ 18% of the GEMV — above the 10% threshold at
which the offload design's pre-decided mitigation, batching the copy pair per
layer, becomes the next lever.)

### Kernel ceiling (out-of-model, same host)

AVX2 `Mq4G256` row dot via `hipfire_cpu::gemv`, 16 rayon threads, warm:

| storage | bytes | time | rate |
|---|---|---|---|
| plain heap `Vec` | 195.8 MB | 4.37 ms | 44.8 GB/s (84 GMAC/s) |
| `hipMalloc`-mapped (host-mapped, as the offload path allocates) | 195.8 MB | 4.40 ms | 44.5 GB/s (84 GMAC/s) |

Host-mapped and heap storage are indistinguishable, so page size/TLB is not a
factor; 1-thread rate on the same shapes is ~11 GMAC/s, i.e. rayon scales ~5× on
8 cores / 16 threads.

### Correctness gate

- `cargo test -p hipfire-cpu --lib`: 24 tests, including per-format expectation
  tables generated from the canonical decoder, `rotate_x` against an
  independent Walsh-Hadamard oracle, AWQ divide, and SIMD-vs-scalar ≤1e-5.
- `cargo test --release -p hipfire-runtime --test cpu_quant_cross_check`: 1695
  tensors, quant types {1,3,8,13,15,20}, bit-identical.
- `cargo test -p hipfire-arch-qwen35 --release --test gpu_gemv_parity -- --ignored`:
  8 formats, real + synthetic + AWQ + pre-rotated arms, worst 6.7e-7 against
  1e-4.
- `serve_harness.py --mode battery` and `--mode chain` on the `cpu` arm with the
  spill: 5/5 turns each, `runaway=0 empty=0 attractor=0 retrieval_miss=0`
  (chain shows prefix-cache hits at 318/602/686/840 tokens), avg decode
  24.5 / 25.4 tok/s.
- `redline_daemon_harness.py` on the `pcie` arm with the spill: `decode
  stable=True launches=427 kernels=20 median=21.7 tok/s`, AQL/PM4 shadow
  `exact=True gdn_frame_exact=True` → the seam's fusion-skip branch and the
  capture guard do not disturb the retained-replay route. On the `cpu` arm the
  same harness fails the load with the documented
  `memory.offload_exec=cpu conflicts with the retained-replay (Redline) backend`
  refusal, which is the intended behaviour rather than a route replaying stale
  activations.
- Zero-diff guard: stock-parent build ≡ new build ≡ spill+`pcie`, byte-identical
  text (2B 1343 chars, 9B 2734 chars).

## Superseded

An earlier pass on the same host reported `cpu` 3.1 tok/s versus `pcie` 18.9–21.4,
and a second pass 5.6 tok/s, i.e. "the CPU path is 3–6× slower". Both were
measured while a full-workspace release build and a second arm sweep were
running on the same 16 threads; they are not measurements of the feature. The
observation generalizes: on this host a background compile costs the CPU arm
several hundred percent and the GPU arm almost nothing, so arm interleaving
alone is not sufficient — the host must be otherwise idle.

## Reproduction notes

/`tmp` artifacts from these runs (`/tmp/f_*.json`, `/tmp/det_*.log`,
`/tmp/harness_*.json`, `/tmp/redline_*_spill.json`, `/tmp/timing_cpu.json`) are
discovery pointers, not durable: the fixture identity, flags, and method above
are what a re-measurement needs. The arm drivers (`cpu_exec_ab.sh`,
`cpu_exec_ab2.sh`) and the kernel microbenchmark (`kernel_probe.rs`) are kept
with this record's data directory; both are ~30 lines (`hipfire bench` with the
flags above; `hipfire_cpu::gemv` in a loop over a `Vec<u8>`).
