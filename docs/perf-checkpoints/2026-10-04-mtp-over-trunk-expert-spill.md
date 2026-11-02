# MTP over trunk-expert spill — performance report

**Lifecycle:** historical. Date: 2026-10-04. Branch:
`feature/qwen35-moe-offload`. Measurements taken on the pre-commit build
identified by daemon md5 `0339b7c1ab05…` (`hipfire --version` reported
`build commit: 98f6452a` / `source commit: 89098c29f` / `source/build:
MISMATCH`, dirty tree) — reproduce by binary md5, not by commit. Fixture:
`ornith-1.5-35b-a3b.mq4` + `ornith-1.5-35b-a3b.mtp` (only local pair with
an MTP head). GPU: gfx1201 (RX 9070 XT), HIP 7.2, 16 GiB VRAM. Battery:
`benchmarks/prompts/mtp_genre_battery.json` (md5
`c6311934e3426ad0b64c7adf9e2c56be`, 9 rows derived from `~/mtp-bench.py`
PROMPTS). Harness: `scripts/serve_harness.py --mode battery --thinking off
--sampling greedy --seed 7 --max-tokens 192`, `HIPFIRE_MOE_EXPERT_BUDGET=auto`
(expert-tier spill). Aggregation below is the arithmetic mean of the
harness per-prompt averages (the `DONE` line).

Terminology: the MTP head is **device-resident in every arm** (loads via
`gpu.upload_raw`, VRAM reserve charged). What spills is the **trunk's routed
experts**. "Over a spill" = GPU-resident MTP verifying against a trunk with
host-mapped experts. There is no "spilled MTP" configuration.

## Result (9-prompt greedy battery, same binary/placement/seed)

Daemon md5 `0339b7c1ab05…`, `hipfire` CLI md5 `4c1579ea38f8…` for all four
arms. Model md5s: trunk `ornith-1.5-35b-a3b.mq4` `f6cb95300d29…`, head
`ornith-1.5-35b-a3b.mtp` `0cf41c5a3643…`. `HIPFIRE_MOE_EXPERT_BUDGET=auto`
in all arms; exec mode is the only placement-adjacent difference.

| arm | avg decode tok/s | avg prefill tok/s | τ range |
|---|---|---|---|
| AR, `offload_exec=pcie` (default) | 45.1 | 262.3 | — |
| MTP on, `pcie` | 34.9 | 186.4 | 0.80–1.51 |
| AR, `offload_exec=cpu` | 42.2 | 76.8 | — |
| MTP on, `cpu` | 26.3 | 69.0 | 0.87–1.51 |

Prefill means are outlier-skewed (pcie-AR row `0b86c16d`, ctx=775, reads
805.0 tok/s against 157–247 elsewhere). Medians: prefill pcie-AR 188.3 /
pcie-MTP 158.6 / cpu-AR 78.2 / cpu-MTP 68.9; decode pcie-AR 45.1 /
pcie-MTP 34.4 / cpu-AR 42.3 / cpu-MTP 26.7. Qualitative result is unchanged
either way: the MTP prefill gap (~30 mean, ~30 median under pcie) closes
under cpu (~8 mean, ~9 median), while the decode gap persists.

Single merge_sort prompt (md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`,
daemon md5 `32c2495e`): MTP decode 10.7 vs AR 44.6, prefill 147.2 vs
170.2, τ=1.33. No fault (SIGABRT/GPU fault) on any arm. Text: 6/8
byte-identical rows on the 8-row run; recorded divergences at char 47
(merge_sort `> 1` vs `<= 1`) and char 359 (row4 paraphrase) — divergence
recorded, lossless not proven.

## Why MTP regresses under pcie (measured + structural)

Batched verify confirmed live (`HIPFIRE_DEBUG_BATCH=1`: every K+1 verify
`result=true, all_layers_ok=true` — no per-token fallback). Per decode
cycle at τ≈1.3, K=3: K serial compressed-serial head forwards (block
forward + lm_head GEMV + argmax + D2H each) + 1 batched trunk verify over
K+1 rows (each row re-streaming host-mapped routed experts over PCIe) +
possible partial-accept repair (tape replay or full trunk replay). AR pays
1 trunk row per token. ~3× the PCIe expert bytes per accepted token before
head/lm_head/repair costs — the committed `auto`-degrades policy rests on
this measurement, not on a model of the link.

## cpu is slower than pcie for both AR and MTP

Direct answer: no — `offload_exec=cpu` never beats `pcie` without MTP on
this fixture. AR decode 42.2 vs 45.1 (pcie ~7% ahead), AR prefill 76.8 vs
262.3 mean (78.2 vs 188.3 median, pcie ~2.4–3.4× ahead). Strict win for
pcie on both axes, not a tradeoff.

The cpu-AR arm is the discriminating control. Decode under `cpu` barely
moves for AR (42.2 vs 45.1) but prefill collapses (76.8 vs 262.3): the CPU
splice costs prefill, not decode. Caveat: the decode comparison may be
capture — cpu arms log `hipGraph capture disabled`, pcie arms had capture
on — so let the prefill gap (2.4× median, far too large for capture)
carry the conclusion. MTP under `cpu` still loses decode 26.3
vs 42.2 (~38%) — so the MTP decode gap persists with **zero PCIe expert
traffic** (splice engaged: `18/18 spilled layers fully covered by the CPU
seam; uncovered quants: none`). The decode gap is therefore not (only)
PCIe re-reads; the K serial head forwards + verify + repair overhead is
real and exec-independent in sign, larger under cpu (−23% pcie, −38% cpu).
Prefill under `cpu` is close (69.0 vs 76.8), so the prefill gap *was*
mostly PCIe — decode gap was not.

SIMD is not the story: AVX2+FMA+F16C kernels + rayon (`hipfire-cpu/src/simd`,
`gemv.rs:88`), present on the 7800X3D. Structural cost per MoE layer in the
splice (`qwen35/prefill.rs:5091-5131`): 4 full-tensor D2H copies +
CPU GEMV + 1 full-tensor H2D, per layer per chunk. **But the d2h/gemv/h2d
split is unmeasured on this path**: `moe_cpu_experts_batched`
(`cpu_exec.rs:459`) increments neither `CPU_STEPS` nor calls `trace_step`
(only the single-GEMV paths at :261/:334 do), so
`HIPFIRE_CPU_EXEC_TRACE=1` is structurally silent here — confirmed
(coverage line printed, zero trace lines). "Transfer-bound" is a
code-level hypothesis, not a timed result; closing it needs a `trace_step`
call in the batched splice. Further confound: `hipGraph capture disabled
(CPU-executed steps present)` — the cpu arms are not capture-equivalent to
the pcie arms.

## llama.cpp comparison

llama MTP is the same algorithm (seed from target hidden → K head steps →
one batched verify → accept prefix, `common/speculative.cpp:1390-1828`)
but different wiring: (1) the draft is a second *context* on the same
model object (`llama_init_from_model(model_tgt)`,
`LLAMA_CONTEXT_TYPE_MTP`) rather than a separately loaded head — placement
inherited, no reserve bookkeeping; (2) per-step draft is a nextn-layer
decode, not a full block forward. And llama `-ncmoe` maps experts onto
`ggml_backend_cpu_buffer_type()` (`llama-bench.cpp:1282-1285`) — a **CPU
execution tier**, not a device-alias mapping. So "llama doesn't regress" is
not evidence about PCIe; the fair hipfire counterpart is `offload_exec=cpu`.

Batch-gating verified on AMD, not just CUDA: `ggml-hip/` contains only
`CMakeLists.txt`, but it compiles the CUDA sources verbatim
(`file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")`,
`ggml-hip/CMakeLists.txt:63`), so `ggml_backend_cuda_device_offload_op`
(`ggml-cuda.cu:5749`, `get_op_batch_size >= op_offload_min_batch_size`,
default 32 at `:5923`) is live on gfx1201. Per-op routing — decode on CPU,
prefill-size batches back to GPU over the link — ships on AMD exactly as on
NVIDIA. hipfire's process-global `offload_exec` (`cpu_exec.rs:67-72`,
`LazyLock<bool>`, no batch awareness) is what diverges from both.

## Caveats

n=1 per prompt per arm (no ≥3-run distribution — do not cite a ratio as a
constant); single-prompt MTP decode swung 10.7→29.2 tok/s across runs;
`runaway=1` flags are the 192-token cap on a thinking model, not an MTP
defect. Raw logs: `results-offload/genre9-{mtp,ar,cpu-mtp,cpu-ar}.json`
(+ serve transcripts).
