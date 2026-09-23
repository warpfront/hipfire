# HIPa Kernel Assembly Experiments

Date: 2026-05-09
Worktree: `/home/kaden/ClaudeCode/autorocm/hipfire/.worktrees/HIPa`

## Objective

Profile the existing HIP kernels, inspect generated ISA for the hot loops on
gfx1100/gfx1151/gfx1201, try plausible inner-loop rewrites, and keep only
changes that survive correctness and throughput validation.

## Retained Change

- `attention_flash_q8_0_tile.hip`: fixed the Q8 flash tile path for
  `head_dim=256` and graph-captured partial stride.
  - Added `max_tiles` kernel argument.
  - Writes partials with max-capture stride:
    `(h * max_tiles + tile_id) * (2 + head_dim)`.
  - Handles `head_dim / 128` halves for QK and V accumulation.
  - Validated on hiptrx gfx1201 with forced Q8 flash:
    `gen_tok_s=35.4`, `prefill_tok_s=527.7`.
  - Current local canonical gate:
    `ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ./scripts/coherence-gate-dflash.sh`
    passed with no hard errors, report
    `/tmp/coherence-dflash-20260509-050617.md`.

## Failed Or Rejected Experiments

### Residual GEMV zero-point factoring

- Target: `gemv_hfq4g256_residual.gfx1100.hip`.
- Existing ISA already has no spills/scratch and clean FMA/cvt sequence.
- Factored-zero-point variant compiled to 85 VGPR, no spills, but was slower
  in synthetic 4096x4096 residual GEMV and was not byte-exact
  (`505/4096` divergent vs GEMM reference).
- Result: removed.

### Residual GEMV inline FMA assembly

- Target: `gemv_hfq4g256_residual.gfx1100.hip`.
- Scratch helper:
  `asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));`
- ISA compiled to 18 VGPR, 22 SGPR, no private segment/scratch, occupancy 16.
  The low VGPR count came from serializing the dequant/FMA chain rather than
  preserving the compiler's high-ILP schedule.
- Correctness failed immediately:
  `test_gemm_hfq4g256_residual 4096 4096 1` reported
  `2026/4096` divergent elements (`row=0`, bit drift
  `0xbfbb2f52` vs `0xbfbb2f4c`).
- Result: reverted. Restored source rerun passed byte-exact for the same
  command.

### Residual GEMV multirow

- Target: row batching for `gemv_hfq4g256_residual.gfx1100.hip`.
- ISA:
  - R2: 96 VGPR, no spills.
  - R4: 96 VGPR plus 2 VGPR spills/private segment.
  - R8: 148 VGPR.
- Result: not worth routing; pressure and spills exceed likely locality win.

### gfx1201 non-residual GEMV route

- Existing `gemv_hfq4g256.gfx1201.hip` compiles to 43 VGPR/18 SGPR/no spills
  and uses `s_prefetch_data`.
- HFQ QA passed on hiptrx gfx1201.
- `bench_hfq_family` was mixed/noisy and 27B MQ4 model-level decode stayed at
  `35.3-35.4 tok/s` baseline vs `35.3 tok/s` candidate.
- Result: removed.

### `fused_rmsnorm_mq_rotate` hand-asm candidacy

- Compiled with `hipcc -O3 --offload-arch=gfx1100/gfx1151/gfx1201 -S`.
- ISA is already clean:
  - gfx1100: 22 SGPR, 36 VGPR, private segment 0, occupancy 16.
  - gfx1151: 22 SGPR, 36 VGPR, private segment 0, occupancy 16.
  - gfx1201: 24 SGPR, 36 VGPR, private segment 0, occupancy 16.
- The compiler emits the expected `ds_swizzle_b32` butterfly.
- Result: no retained rewrite; manual asm has little obvious headroom.

### `fused_silu_mul_mq_rotate` fast exp

- Baseline 9B MQ4 local gfx1100 runs:
  - `129.6`, `129.3`, `129.6 tok/s`.
- Replacing `expf` with `__expf` removes the extra `v_ldexp_f32` range
  handling in ISA, but does not improve model-level throughput:
  - candidate runs after JIT warmup: `129.0`, `129.5 tok/s`.
- Result: reverted; numerical risk without measured model-level benefit.

## Current Headroom Estimate

The retained evidence points to bandwidth/launch/algorithmic limits more than
missing scalar assembly polish. A true 20% local win on one hot kernel does not
map to 20% decode unless that kernel dominates. For the current MQ4 decode hot
set, durable end-to-end hand-asm upside looks low single digit, with a generous
6-12% only if multiple GEMV/projection kernels all get real local wins without
increasing register pressure.

## NPU Follow-Up

Given the diminishing returns above, HIPa also added Strix/XDNA NPU probe and
compute-smoke plumbing:

- Local crate module: `crates/hipfire-runtime/src/npu.rs`.
- Examples: `strix_npu_probe`, `strix_npu_compute_smoke`.
- CLI diag classifier: `cli/diag_pure.ts`.
- Smoke wrapper: `scripts/strix-npu-smoke.sh`.
- hipx validation with raised memlock:
  `ready=true`, `compute_validated=true`, `daemon_usable=true`,
  `device_name=RyzenAI-npu5`, `avg_npu_time_us=271`.
- Current isolated hipx validation copy:
  `/home/kaden/hipfire/HIPa-validate-current`.
- gfx1151 iGPU smoke:
  - Device: `Radeon 8060S Graphics`, `gfx1151`.
  - Reason for the retained compiler workaround:
    ROCm 7.2.2 on hipx has missing HIP version metadata and the
    runtime JIT resolves device math (`max`, `rsqrtf`, `expf`, `logf`,
    `fabsf`, `floorf`, `fmaxf`, `fminf`, `powf`, `cosf`, `sinf`) to host
    declarations unless the prelude provides OCML-backed device overloads.
  - Validation-only env prelude first proved the workaround:
    `gen_tok_s=46.7`, `bw_gib_s=230.8`, `avg_ms=21.34`, `p50_ms=21.33`.
  - Retained source change: `crates/rdna-compute/src/compiler.rs` now injects
    that math prelude automatically only for `arch == "gfx1151"`.
  - Current automatic-prelude run after clearing `.hipfire_kernels/gfx1151`
    and rebuilding without `HIPFIRE_HIPCC_EXTRA_FLAGS`:
    `gen_tok_s=46.7`, `bw_gib_s=230.9`, `prefill_tok_s=576.3`,
    `avg_ms=21.33`, `p50_ms=21.32`.
- Current NPU summary from the isolated build:
  - Normal user: `ready=false`, `compute_validated=true`,
    `daemon_usable=false`, `device_name=RyzenAI-npu5`,
    `avg_npu_time_us=256`; advice correctly requests higher memlock.
  - Raised memlock: `ready=true`, `compute_validated=true`,
    `daemon_usable=true`, `device_name=RyzenAI-npu5`,
    `avg_npu_time_us=238`.
