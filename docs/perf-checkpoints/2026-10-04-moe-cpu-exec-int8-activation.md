# MoE CPU splice: int8-activation path is a null — the splice is weight-bandwidth-bound

**Lifecycle:** historical. Date: 2026-10-04. Third record in the
`offload_exec=cpu` series; links to the unchanged
[`2026-10-04-moe-cpu-exec-decode-only-routing.md`](2026-10-04-moe-cpu-exec-decode-only-routing.md)
and [`2026-10-04-moe-cpu-exec-batched-expert-gemv.md`](2026-10-04-moe-cpu-exec-batched-expert-gemv.md).

Binary: `hipfire` md5 `8fef0cbf2d7915e3f2217e1bc2eb6841`, daemon md5
`2a4895aa2fea0591ffbbf0a352849ced`. Fixture, prompt (`merge_sort`, md5
`253c7ac50857fe6d0e10fb0d2c5e35c0`), host and method as in the batched-GEMV
record.

## What was tried

llama.cpp's quant dot uses an **integer** accumulate over **int8** activations
(`_mm256_maddubs_epi16` + `_mm256_madd_epi16`, `ggml/src/ggml-cpu/arch/x86/quants.c:701`,
helpers at `:122/:229`, activations from `quantize_row_q8_1` at `:400`) — no
AVX-512 anywhere in the file. hipfire's CPU kernel instead decodes every 4-bit
code to f32 and FMAs against f32 activations (~3–4× the inner-loop ops and 4×
the activation traffic). hipfire already owns the int8 block format
(`block_i8_128`, shared with the gfx12 A8 prefill MMQ), so the V2 decode splice
was moved onto it: `block_i8_128` (commits `3d077947f`), the AVX2
`mq4v2_i8_group_dot` (`98adeb58f`), and the `moe_cpu_experts` wiring.

## Result — no measurable win

A/B on one binary, `HIPFIRE_MOE_CPU_I8` 0/1, load ~8.4:

| path | gu ms/layer | dn ms/layer | decode tok/s |
|---|---|---|---|
| f32 | 0.33 | 0.17 | 51.8 |
| int8 | 0.31 | 0.19 | 51.7 |

The ~3–4× inner-op reduction moves nothing. The in-situ gemv split is a mean
over thousands of calls, so this is not run-to-run noise: the kernel is **not
op-bound**.

## Why — the splice streams weights at ~26 GB/s

Per host layer the splice reads 13.3 MB of expert weights (8 × 1.11 MB gate_up +
8 × 0.56 MB down). At ~0.50 ms of GEMV that is **~26 GB/s**, against:

- ~46 GB/s a plain single-threaded sequential read of a `malloc`'d buffer gets on
  this box, ~55 GB/s with 2+ threads (measured, `membw`);
- ~43 GB/s the **dense** cpu arm appears to reach on the same host-mapped memory
  (qwen3.5-9b, 24/32 layers spilled: cpu 11.0 vs pcie 8.3 tok/s) — so the memory
  itself is not the wall.

The host-mapped experts are `hipHostMalloc(_, HIP_HOST_MALLOC_MAPPED)` and, with
`HSA_USERPTR_FOR_PAGED_MEM` unset, **GTT-backed** (`dispatch.rs:4020`, "0 = GTT,
capped by /sys/module/ttm/parameters/pages_limit"); `AnonHugePages` is ~0.9 GB
of a 28 GB box, so they are 4K-paged. But forcing
`HSA_USERPTR_FOR_PAGED_MEM=1` changed nothing (51.8 tok/s), so the GTT/paged
distinction is not the cause either.

Net: neither the arithmetic (int8) nor the allocation class moved it. The
remaining ~2× between 26 GB/s and the ~46–55 GB/s the box sustains is the open
question; it is a property of the read path/pattern, not of the dot.

## Disposition — reverted

The int8 path was landed behind `HIPFIRE_MOE_CPU_I8` and then **reverted**: it
is the llama.cpp arithmetic and is bit-tested against the f32 kernel and the
`block_i8_128` contract, but it is perf-neutral on this fixture and costs an
activation-quantization quality step for no gain, so it does not meet the bar to
land. The three commits (`3d077947f` `block_i8_128`, `98adeb58f` the AVX2 group
dot, `8d84c5854` the wiring) are undone; this record is kept as the evidence of
the null and of *why* the arithmetic was not the lever. Raw logs:
`results-offload/i8*-moe-cpu.*`.
