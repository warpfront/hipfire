# CPU-exec offload spill sweep — 2B / 9B / 27B-mq3-xt, probe prompt

**Lifecycle:** `historical`
**Date:** 2026-09-27
**Method:** `docs/methodology/cpu-exec-offload-benchmark-handoff.md` §4B. Flags:
`--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots --workload
stateless --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json`.
Env per point: `HIPFIRE_GPU_LAYER_BUDGET=<b> HIPFIRE_OFFLOAD_EXEC={pcie,cpu}`,
one fresh daemon process per point, interleaved pcie/cpu × 3 rounds per budget.
27B model path: `/path/to/models/qwen3.8-27b.mq3-xt`.
**Not an admission, not a BENCHMARKS.md claim, not comparable across
host/model/quant/GPU/prompt/method.**
**Not measured:** slots/serve path, prefill claims, decoded-text reads for these
points (arm-D read stands), formats that stayed on PCIe (none anywhere —
every cpu point covered, `uncovered quants: none` throughout).

## Identity

| artifact | digest |
|---|---|
| source HEAD | `7133b3c328ae53a0e9590063dc3d95d5b71c32d0` (handoff pin `924d7fd8`; drift = 2 docs-only commits, no code) |
| `target/release/daemon` md5 | `904368995cddb8cda8e82a7f8d31ae96` (same as handoff) |
| `target/release/hipfire` md5 | `030f080ce4c3d68ca038a614059d21b2` (same as handoff) |
| `qwen3.5-2b.mq4` md5 | `9ed6628f2df83ef4b1c062afd4a85bfb` |
| `qwen3.5-9b.mq4` md5 | `31a8d8dc7603226801b08d8319015602` |
| `qwen3.8-27b.mq3-xt` md5 | `80bb9198e6a565fc006b2ae1b7c89eca` (11,777,616,896 B) |
| `gpu_offload_probe.txt` md5 | `5835c71e471849b4a72e1dc8e39695e7` (59 tokens) |
| host | 1× RX 9070 XT gfx1201, ROCm/HIP 7.2, Ryzen 7 7800X3D, full desktop session |

**Host-load caveat (read first).** Per user instruction this box runs a full
desktop environment — clean data is not on offer. 2B/9B legs ran at load
4.5–8.2 (driver stdout; a stray `gh pr view` spun at 50–75% CPU for part of the
run). 27B-leg pre-loads recorded per point in `load.log` (3.3–6.6, except
b48_pcie_1 at 13.8). Direction of every leg-level claim below survives this;
absolute tok/s values inherit the caveat. qt49 parity oracle passed before the
27B leg (2 passed, 0 failed).

## Data

Whole per-point bench JSON + daemon stderr + 27B load log:
`data-2026-09-27-cpu-exec-sweep/{2b,9b,27b}_*.json|.err`, `load.log`.
Residency: every point shows the requested split (`i_gpu_start` = spilled
count); every cpu point at spill > 0 shows `N/N spilled layers fully covered;
uncovered quants: none`. Zero-spill cpu points (2b_b24, 9b_b32) correctly show
no coverage line (`offload_exec=cpu but nothing is spilled`); 27b_b64_cpu_1 the
same. `hipGraph capture disabled` appears on cpu points with spill (expected —
CPU-executed steps present), never on pcie points.

## Results — decode_tok_s median-of-medians (per-process medians in brackets)

### 2B (24 layers) — cpu LOSES at every spill ≥ 3

| budget (spilled) | pcie | cpu | delta |
|---|---|---|---|
| 24 (0) | 258.5 [123.3, 259.1, 258.5] | 258.4 [258.7, 258.4, 258.3] | tie (control ✓) |
| 21 (3) | 131.1 [81.6, 131.4, 131.1] | 99.3 [99.3, 100.0, 99.0] | pcie +32% |
| 18 (6) | 62.6 [62.6, 62.6, 89.3] | 65.5 [64.7, 65.5, 65.8] | tie (≈+5% cpu) |
| 15 (9) | 67.5 [50.5, 67.5, 67.5] | 48.5 [48.5, 48.5, 48.7] | pcie +39% |
| 12 (12) | 54.3 [42.5, 54.3, 54.3] | 38.7 [38.5, 38.9] | pcie +40% |

cpu arm matches the 2026-09-26 anchors almost exactly (259/100/65/50/40 vs
258/99/66/49/39). pcie arm agrees except budget 18 (62.6 vs anchor 91 — though
one of its three processes read 89.3). Verdict: no crossover on 2B; ~30 MB/layer
never amortizes the per-step copy+sync. Matches handoff expectation.

### 9B (32 layers) — cpu WINS at every spill point

| budget (spilled) | pcie | cpu | delta |
|---|---|---|---|
| 32 (0) | 103.9 [103.9, 104.0, 103.8] | 104.0 [104.0, 104.0, 103.8] | tie (control ✓) |
| 28 (4) | 30.1 [×3] | 40.8 [41.1, 40.5, 40.8] | cpu **+35%** |
| 24 (8) | 21.9 [×3] | 25.9 [×3] | cpu **+18%** (arm D: +15.7% ✓) |
| 20 (12) | 15.7 [14.2, 15.7, 15.7] | 19.0 [19.0, 18.9, 19.0] | cpu **+21%** |
| 16 (16) | 12.3 [11.3, 12.3, 12.3] | 14.9 [15.0, 15.0, 14.8] | cpu **+21%** (anchor 11.0/14.0 ✓) |

Crossover lies between 0 and 4 spilled (~115 MB/layer wins immediately).
Budget-24 leg reproduces arm D independently (21.9/25.9 vs 21.6/25.0).

### 27B mq3-xt (64 layers, all qt 49) — cpu LOSES HARD (one process per point)

| budget (spilled) | pcie | cpu | delta |
|---|---|---|---|
| 64 (0) | 40.7 (rerun; first process 23.3 cold — see below) | 40.7 | tie (control ✓) |
| 56 (8) | 10.8 | 3.1 | cpu **−71%** |
| 48 (16) | 7.0 (pre-load 13.8, load-suspect) | 1.6 | cpu **−77%** |

Why: qt 49 (`Mq3G256V2`) has only the scalar host decoder — no AVX2 kernel.
Trace (`HIPFIRE_CPU_EXEC_TRACE=1`, budget 56) shows GEMV 4.1–10.4 ms/step
(qkv m=5120 k=6144: 4.30 ms; gate m=10240 k=5120: 10.37 ms) vs 0.6–2.1 ms for
Mq4 on 9B, while copy+sync stays ≈0.1 ms. The spill cost being dodged (PCIe) is
smaller than the decode cost being assumed (scalar). This is the handoff-§7
predicted lever (AVX2 kernel for Mq3G256V2), not a defect — report the numbers,
do not chase them in a sweep. Coverage was 8/8 and 16/16 with
`0 host-mapped steps still on GPU` on all 8 step shapes, so the slow arm is
fully on-CPU and correctly measured.

## Systematic effects observed (apply to all three legs)

1. **First-process-per-budget cold effect on the pcie arm.** The first daemon
   process at each new budget reads low on *both* decode and prefill:
   2b_b24_pcie_1 (123.3/3367.8 vs 259/5850), 2b_b21_pcie_1 (81.6/2295.7 vs
   131/3200), 2b_b15_pcie_1 (50.5/1497.7 vs 67.5/1830), 2b_b12_pcie_1
   (42.5/1253.9 vs 54.3/1450), 27b_b64_pcie_1 (23.3/440.5 vs rerun 40.7/556.9).
   lives on the GPU path: kernel compile/DPM settle). Headline
   deltas above use median-of-medians over all three (defensible either way
   since the direction never hinges on the cold point), but anyone re-cutting
   this data should drop the first pcie process per budget.
2. **Zero-spill controls tie on all three models** (258.5/258.4, 103.9/104.0,
   40.7/40.7) — the `offload_exec` switch itself costs nothing when nothing is
   spilled.
3. **Prefill is flat across arms within a budget** except via effect (1) —
   no independent prefill smell.
4. **Capacity parity holds.** `gpu.vram_free_mb` identical across arms at the
   same budget on 2B/9B (e.g. 9b_b24: 11094/11066/11122 family, matching pairs
   per round); 27B b48 exact (5664/5664), b56 differs by 28 MB (4374/4402 —
   same 28 MB wobble as arm-D round 3), b64 varies run-to-run 3084–3140 at full
   residency (allocator noise, both arms). No systematic VRAM movement from the
   feature.

## Decode samples (per-process medians; full 5-run arrays in the JSON)

- 9b_b28: pcie [30.1×3], cpu [41.1, 40.5, 40.8] — tightest spread of the sweep.
- 27B points are single-process; spreads within a process are ~0.0
  (e.g. b56_cpu_1 [3.1×5]) — suspiciously tight per AGENTS.md §0 rule 3, but
  there is no τ here (`--spec off`, greedy) and no text to eyeball on bench;
  the coherence read for the cpu path is the arm-D `run` comparison
  (byte-identical, see arm-D checkpoint).
