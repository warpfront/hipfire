# gfx1201 iu8 prefill arm — parked STATUS (2026-09-21)

Branch: `gfx1201-iu8-arm` @ `6be650494` + WIP (commit: "WIP gfx1201 iu8 prefill arm (parked; impure trace, no gates)").
Not pursued further for now. Resuming owner: see "Next" below.

## What landed
- `HIPFIRE_IU8_PREFILL` / `kernel.iu8_prefill`: opt-in, exact-gfx1201-only
  (`crates/rdna-compute/src/feature_flags.rs`, `crates/hipfire-config/src/lib.rs`;
  regression test `iu8_prefill_opt_in_gfx1201_only` passes).
- New kernel `kernels/src/gemm_mq4g256v2_residual_mmq.gfx12.hip`: single-wave
  16-row tile on `wmma_i32_16x16x16_iu8_w32_gfx12`, V2 dual-fp16 headers per
  128-K half, same 7-arg ABI as the RDNA3 MMQ kernel, in-kernel M/N masking.
  Lacks vs the iu4 twins: **no symfold twin**, K16 (not K32), per-32 X only.
- `gemm_mq4g256v2_mmq_prequant_gfx12` + set/add wrappers (module
  `gemm_mq4g256v2_residual_mmq_gfx12`); iu8 branch checked FIRST in all four
  gfx12 launchers (`gemm_{qkvza,qkv,gate_up,hfq4g256_residual}_*_gfx12_mq4v2`).
- Chunk scan needs no gate change (flag+arch+dtype gated, not iu4/fp8 gated).
- Debug probe `crates/rdna-compute/examples/probe_iu8.rs` (verifies flag live;
  remove before any real commit).
- Route findings: no pre-existing MMQ-iu8 path on gfx1201 (RDNA3 `#if` + hard
  Err); fp8v2 needs all four family flags + v2 tile + symfold (symmetric V2.5
  artifact `qwen3.8-27b.mq4-xt`).

## Numbers so far
- KLD c1 (`IU8=1 IU4=0` + `GFX12_MQ4V2_FP8_{GATEUP,RESID,QKVZA,QKV}=0`,
  Card-B base env): **0.034611** (1023 tok, 53 tok/s) — reference only,
  **trace impure, do not quote as the iu8 arm**.
- c2/c24, TTFT, matrix, daemon rocprof, battery, `REPORT.md`: NOT DONE.

## Next (when resumed)
1. Rename the gfx12 entry symbols (e.g. `_gfx12` suffix) — current names
   collide with the RDNA3 module so the trace can't attribute MMQ time.
2. Re-trace c1/c2; find which residual call-site still emits
   `residual_wmma_fp8_gfx12_s2bt8/bt4` + fp8 producers (~7% in c1 trace).
   Suspects: second residual call-site (lm_head batched path, MoE/shared-expert
   down), batch<64 tails, module-cache aliasing.
3. Run c2/c24 KLD, TTFT (`ttft_5900.txt`, 8x2), matrix (pp512/8192), in-daemon
   rocprof at pp8192 per `wt-fp8sym/scratch-fp8gap/REPORT.md` recipe, battery.
4. Write `scratch-i8/REPORT.md` (env/gates file:line, c2/c24, TTFT, matrix,
   per-kernel table, iu8-vs-iu4 gaps). Baselines: iu4 c24 0.0860 / 1,672 ms;
   fp8v2 c24 0.0494 / 2,618 ms.
5. FoldFree2 is queued for card-B after this arm — message before daemon runs.
