# Lloyd MMQ A4 staged twin: final on-model A/B (gfx1100, 27B mq4l)

**Lifecycle:** `historical`. Fixture-bound measurement under the exact method
below. Not a current default, baseline, or admission decision.

## Fixture

- GPU: gfx1100 (RX 7900 XTX), HIP 7.16, ROCm 10.1 toolchain
  (`--rocm-device-lib-path=<therock>/lib/llvm/amdgcn/bitcode`)
- Model: `qwen3.8-27b.mq4l` (MQ4G256V2-Lloyd, qt=52), md5
  `d020f248ca52258b1aff623c084719b6`
- Binaries: `hipfire 226ae11d2e6fddb5833f0f3ca28db2e`,
  `daemon 4ceda09f399cd95e3c45a0801f41277c` (release, branch
  `feat/gfx1100-mq4v2-lloyd-prefill`, staged-only tree: packed path excised,
  single tile128/36864 B variant, unified module name)
- Route under test: `HIPFIRE_LLOYD_MMQ_A4=1` (default OFF); OFF leg is the
  shipped Lloyd-X128 route. Exact gfx1100, eager-only, KV q8 legacy.

## Method

`hipfire bench <model> --matrix --pp 2048 --tg 32 --runs 1 --warmups 1
--backend noslots --workload stateless --json`, 6 fresh processes,
interleaved off/on (off, on, on, off, off, on). Synthetic matrix prompts:
byte-identical across legs by construction.

## Results (pp2048 prefill medians)

- OFF (X128): 1050.3 / 1048.2 / 1047.2 tok/s (median 1048.2)
- ON (staged twin): 1218.9 / 1217.4 / 1218.2 tok/s (median 1218.2)
- **Delta: +16.2%.** Decode tied (~41 tok/s both legs).
- Binding: every ON round binds residual `mmq_i8_lloyd_full_add` + fused
  `gate_up_silu_mmq_i8_lloyd` (staged module, cache-hit, no recompile);
  every OFF round binds `x128_lloyd` / `mmq_ds4_x128`, zero twin binds.
- Reproduces the pre-cleanup +15.9% (1050.2/1048.6/1046.4 vs
  1217.6/1215.0/1216.3): the cleanup (packed excision, const rename, module
  unify, comment-only TU edits) changed no numerics.

## Exactness / quality (unchanged, cited)

- Twin-vs-CPU-oracle rel_l2 ≈ 8e-8 all shapes (residual + gate_up, N=128/512/2048).
- KLD WT2 q8-prefill 24-chunk gfx1100: Lloyd 0.0325 vs uniform-X128 0.0400;
  uniform-iu4 pin 0.0693 reproduced byte-exact (validates method).
- 0.8B hybrid (residual-only route): +11.3% pp512; byte-identical greedy
  output twin vs X128 (temp 0).

## Disposition

Perf evidence for PR #1029. Serve-battery eyeball + redline/PM4 parity still
owed per VALIDATION.md before merge; pre-commit-ci green except pre-existing
failures (unformatted foreign T0 harness, workspace clippy warnings outside
this diff, all triaged).
