# Dense 27B co-op prototype: down and gate_up, GPU + NPU (IEF15) on Halo

> **Historical research note: superseded by the 0.4.2 serving route.** This file records a killed dense 27B co-op prototype (verdict in section 8) from the 2026-10 NPU research cycle. It is not current guidance, and any figure in it is a dated fixture-bound measurement or an explicitly labelled historical estimate, not a product claim. The shipped 0.4.2 NPU route is the Flash-Next gfx1151 trunk input projection described in [npu-trunk.md](npu-trunk.md): `HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` (default `auto`, effectively `off` on every architecture except gfx1151), calibration schema 4, BF16-direct only. There is **no I32 serving contract** (no int32 partial, combine or fold exists in the product route). The out-projection program exists but is **not wired** into serving; `qsa.o` and the GDN o-projection are **not shipped**. Environment-variable names and flag syntax below are historical proposals or experiment switches, not supported configuration.

2026-10-03, research branch `npu/coop-down27`. Model `qwen3.8-27b.mq4-xts` sha256 `3e38ccba…f6f8ae` (re-verified on the measurement host).
Code: `coop27/` (out-of-tree prototype; hipfire read-only, incumbent kernel bytes vendored from the 0.4.1 hipfire beta checkout at 798d63b3a,
`native/gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151.hxaco` sha256 `2395edab…778d2b`; PM-native glue kernels assembled
by peacemaker, `native/build.sh`). Windows: `tools/npu/npu-coop27-*.sh`; run logs are not included here.
Labels: **M** measured, **E** estimate.

## 1. What was built
* **GPU columns, incumbent route, byte-identical (M).** The certified PM V2B ADD (down) / gate_up SiLU entries run on a
  feature window by pointer offset (`A + c0*row_bytes`, `Y + c0*4`), the true row stride `M` and a trimmed row-tile grid
  (`M` is only the Y stride in `iu4_v2b.rs`). 0 mismatches vs the full launch for every window tested (down 256..1536 NPU
  columns, gate_up 1024..5120 NPU hidden), NPU region untouched.
* **NPU columns: IEF15 N0 kernel** (`Ief15Gemm`), leading window; gate_up as J commands of `8192 x 2048 x 5120` (g | u
  rows of 1024 hidden each, 256-hidden aligned, shared A stream).
* **Glue (GPU, PM-native):** D/b sidecars (`ief_db`/`ief_db2`), A-stream pack (`ief_pack`, v2 `ief_pack2`+`ief_tail`: one
  pass over Xq, LDS transpose), down epilogue `RN(old+Y)` (`ief_addc`), gate_up epilogue `ief_silu` = the incumbent SiLU
  op DAG copied verbatim from the emitter (byte-equal to the incumbent on its own g/u, M).
* **Transport:** userptr pages (NPU BO + `hipHostRegister`), in-process 5-way alias check (incl. GPU re-read after a
  foreign write) before any NPU command; final config `hipExtHostRegisterCoarseGrained`. Sync: CPU relay (no IEF15 ring
  body exists); relay overhead ≤1.3 µs (M).

## 2. Correctness (M, every co-op process, first and last verified round)
GPU-owned columns byte-identical to GPU-alone (0 of 33.5 M down / 117–134 M gate_up); NPU C byte-equal to the CPU IEF15
oracle; NPU-column outputs equal `RN(old + Y_oracle)` (down) / incumbent SiLU DAG of the oracle g,u (gate_up); D/b and
A stream byte-equal to the CPU encoders. Error vs GPU-alone (layer output, not logits — the full model was not wired
because neither arm is positive):

| arm | pre-epilogue NRMSE (NPU cols) | output NRMSE NPU cols | output NRMSE all cols | output max abs |
|---|---:|---:|---:|---:|
| down n=1024, L2 | 3.12e-5 | 4.37e-7 | 1.96e-7 | 2.10e-5 |
| down n=1024, L42 | 2.24e-5 | 1.97e-6 | 8.79e-7 | 7.99e-5 |
| gate_up J=1 / 2 / 3, L2 | 1.62e-5 | 1.39e-5 / 1.45e-5 / 1.45e-5 | 3.2e-6 / 4.5e-6 / 5.5e-6 | 4.9e-5 / 8.6e-5 / 8.6e-5 |

## 3. Per-layer table (M, L2 real weights + a4-ec real A4 tiled to 8192 tokens, fclk pinned)
GPU window time vs columns: down full 17.54; NPU cols 256 19.04 (gs 0), 512 16.51, 768 17.15, 1024 14.22, 1280 14.77,
1536 13.12 ms. gate_up full 36.45; NPU hidden 1024 34.33, 2048 32.20, 3072 30.13, 4096 28.04, 5120 25.87 ms.
NPU IEF15 alone (userptr): down n 512/768/1024/1280/1536 = 7.21/10.27/13.00/15.41/18.44 ms (20.3–23.8 TOPS);
gate_up 8.7–9.3 ms per 2048-feature command.

Co-op, best transport (coarse) + glue v2, fresh processes (`w8b`, `w8c`, `w8a`, `w9`):

| arm | GPU-alone ms | co-op ms | delta ms/layer (each process) | parts (median) |
|---|---:|---:|---|---|
| down n=1024 | 18.10–18.30 | 19.46–19.52 | +1.22 / +1.29 / +1.28 (L2), +1.36 (L42) | prep 2.9, window conc 15.4 (α 1.04), NPU 14.4, addc 1.21 |
| gate_up J=1 | 38.05–38.23 | 38.66–38.70 | +0.65 / +0.43 / +0.60 | prep 1.28, window 36.6 (α 1.02), NPU 10.6, SiLU 0.79 |
| gate_up J=2 | 37.73–38.12 | 38.17–38.47 | +0.39 / +0.44 / +0.23 | window 35.3 (α 1.05), NPU 20.6, SiLU 1.72 |
| gate_up J=3 | 37.37–37.82 | 37.64–37.86 | +0.04 / +0.20 / +0.03 (v1 SiLU); +0.20 / +0.29 / +0.27 (v2 SiLU) | prep 1.29, window 33.7 (α 1.08–1.10), NPU 30.5 (26.5 alone), SiLU 2.74 |

History of the same points: default registration + v1 glue: down +3.24 ms, gate_up J=3 +2.24; coarse: gate_up J=3 +0.19.

**Step (64 calls each, baseline 6810.69 ms, E from the per-layer deltas):** down alone +78…+87 ms (+1.2%), gate_up
alone +2…+19 ms (≈0, +0.0…0.3%), both: their sum plus the design switch (N0 M: +1.02/+1.20 ms per alternating submit
on one NPU, 2 switches per layer ⇒ ~+140 ms) — all non-negative. **No arm yields a step gain.**

## 4. Mapping matrix (M, `maps-49cb304.log`)
All 16 configs (4k / THP / hugetlb 2M / 1G × hipHostRegister flags 0 / Portable / Mapped / CoarseGrained): amdxdna
import accepted, all alias verdicts ok, raw PM gpu-bw on the shared pages 238–241 read / 220–227 write GB/s. Page size
has no effect; coarse-grained speeds the glue 15–33% (pack 3.93→2.61 ms, addc 1.14→1.00, SiLU 1.20→0.82). The
remaining glue gap is kernel shape, not the memory path. A register-renamed low-VGPR SiLU (`ief_silu2`, 72 VGPRs,
byte-exact) is not faster (1.00 vs 0.98 ms): occupancy is not its limit.

## 5. CPU as the glue device (M, `w9-9737f4b.log`)
CPU STREAM (32 threads, AVX-512 NT): 112 read / 122 write / 149 copy GB/s alone; **31 / 52 / 34 GB/s under a concurrent
GPU read load** (GPU 226 GB/s). CPU pack (byte-equal `pack_in`): down 1.7 + 3.3 ms, gate_up 0.85 + 1.7 ms, alone; CPU
addc (exact) 12 ms (naive). Under GPU load the CPU pack scales by ~3× (E) and exceeds the NPU's ~3 ms slack at gate_up
J=3, so the NPU would wait; CPU glue does not take work off the critical path. SiLU stays on the GPU (not
CPU-reproducible). Not integrated.

## 6. Two NPU contexts / switching
One context switching designs: N0 M +1.02 / +1.20 ms per alternating submit. Two contexts on disjoint 16-tile
partitions: the probe with the M2 single-core design failed (that design is one-shot; resubmission on its context timed
out at 3 s, no wedge, V9 health PASS); the fixed fresh-context probe (`--mode parts`) was not run. Unanswered; moot while
both arms are non-positive.

## 7. Path back (design brief, E)
1. **NPU DDR traffic** (now the largest cost: α 1.08–1.10 at gate_up J=3, NPU 30.5 vs 26.5 ms alone). Per gate_up
   command (8192×2048×5120, MW 32, NW 8; A base 22.3 MB, B base 5.57 MB, C 67.1 MB):

   | tiling | A | B | C | total | NPU ms/cmd (E: max(issue 7.6, bytes/55 GB/s)) | J=3 traffic GB/s | α (curve 2.2/5.9/11.3% @ 5/16/47) |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | N0 today (M) | 178.3 | 178.3 | 67.1 | 423.7 | 8.8 alone / 10.2 conc (M) | 41.6 | 1.08 (M) |
   | B resident per (column, nw), nw-outer (V9-style, 87 KB/column) | 178.3 | 5.6 | 67.1 | 251 | ~7.6–8.0 | ~31 | ~1.086 |
   | A resident per mw, mw-outer | 22.3 | 178.3 | 67.1 | 268 | ~7.6–8.0 | ~33 | ~1.09 |
   | blocked: 3 nw of B (261 KB) + A(mw) double-buffered (174 KB) per memtile, 3 passes | 66.9 | 5.6 | 67.1 | 140 | ~7.6 | ~17.5 | ~1.062 |
   | K-split, i64 partial merge | +268 MB i64 spill | | | worse | — | — | rejected |

   Blocked tiling at gate_up J=3: window 31.1×1.062 = 33.0 ⇒ co-op ≈ 1.3 + 33.0 + 2.7 = 37.0 vs 37.6 (**−0.6 ms/layer,
   −40 ms/step, −0.6%**, E); J=4 (NPU 4×8.0) ≈ 1.3 + max(29.7, 32.0) + 3.6 = 36.9 (−0.7). Memtile budget for the
   blocked map (512 KiB minus the V8 ring/C buffers) is unverified.
2. **A direct-read by BDs** (removes the GPU pack): new shim/memtile design, 3 iteration segments per (wave, column) for
   E > 64, codes+tail BD pair, 3072 task pushes per down submit; down best case ≈ −0.4…−1.8 ms/layer (E, §W2 note).
3. Epilogue fusion into the incumbent window kernel (no separate C read pass) would save ≤0.8 ms (down) / ≤2.7 ms
   (gate_up J=3) of tail (E, upper bound), the largest single glue lever left.

## 8. Verdict
**KILL** wiring either arm into hipfire as an opt-in route now: measured step delta is +1.2% (down) and +0.0…0.3%
(gate_up J=3), with exact numerics and full transport bandwidth. Revisit only with an NPU kernel that cuts its DDR
traffic ≥3× (blocked residency) plus a fused GPU epilogue; together E ≈ −1…−2%/step on Halo, still far from +10%.
