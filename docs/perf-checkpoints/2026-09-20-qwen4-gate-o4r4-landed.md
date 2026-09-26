# Qwen4 QT44 grouped gate/up — gfx1151 O4×R4 tile — 2026-09-20

**Lifecycle:** `historical`

**Disposition:** landed for the Qwen4 `m == 1280 && k == 2560` gate/up shape on
`gfx1151` only. Not a G5 admission, not retained-replay admission, and not a
cross-architecture claim. Other shapes keep the SIMT arm, other arches keep the
WMMA arm.

## Change

New exact-shape kernel
`kernels/src/gemm_mq4g256v2_moe_grouped_top10_o4_r4.gfx1151.hip`
(SHA256 `e932e7bb1cc53e327814dc5b59746217e8ec8fe82d7296b2ab408f6e52e58b02`),
grid `[ceil(M/4), ceil(grouped_rows/4), 1]`, block 32, four adjacent output rows
× four route slots with 4 chains × 4 slots × 4 rows = 64 accumulators — the same
accumulator budget, per-element K order, chain (`g & 3`) association,
`(c0+c1)+(c2+c3)` fold and full-wave shuffle offsets as the O2×R8 arm it
replaces. Launcher `gemm_mq4g256v2_moe_grouped_top10_o4_r4_gfx1151` and registry
const `GEMM_MQ4G256V2_MOE_GROUPED_TOP10_O4_R4_GFX1151_SRC` in `rdna-compute`; the
superseded O2×R8 launcher was removed and its kernel source/const kept as the
retained bitwise reference. No format, routing, metadata, or topology change.

## Evidence

* **Flag equivalence.** The production recipe (`--genco --offload-arch=gfx1151
  -O3 --no-offload-compress`, no `-ffast-math`) reproduces the shipped kernel-cache
  blob's `.text`/`.rodata`/`.note` byte-identically (`343754572384952b…`,
  `392bc0203ce188b8…`, `e48f253ebe1eb088…`); a `-ffast-math` control differs
  (`08028df9022a3214…`). The landed file compiles to the measured arm's `.text`
  (`9cc285efbff92a72…`).
* **Bitwise equality.** Focused driver (`moe_gate_o4r4_compare`, SHA256
  `4492cc0374f7eb67…`): 23/23 cases, every logical `(slot,row)` cell raw-bit
  equal, `first_mismatch = -1`, `max_abs_delta = 0`, sentinel tiles untouched,
  negative slots exactly `+0.0`, `grouped_rows` tail untouched, odd M covered.
* **Isolated kernel time** (10 warmups, 5 alternating event pairs): production
  shape `8.2356 → 7.9787 ms` (+3.12 %); all-live tile `16.3201 → 15.0343 ms`
  (+7.88 %); `one_valid_slot` `5.0816 → 4.7815 ms` (+5.90 %); cache-resident
  weight probe `5.7954 → 6.2613 ms` (−8.04 %).
* **ISA.** production: VGPR 116 / SGPR 64 / 0 spills / 0 private, 4162 static
  instructions; O4×R4: VGPR 145 / SGPR 53 / 0 spills / 0 private, 3764 static
  instructions. X `b128` loads per `(block, group, lane)` 16 → 8 for the same 144
  MACs.
* **Whole-model oracle** (landed tree, fresh kernel caches):
  `../../.codeinsight+research/qwen4/pp500-kernel-check/o4r4-oracle/result.json` and
  `../../.codeinsight+research/qwen4/pp500-kernel-check/o4r4-boundary/result.json` both `status=pass`,
  `comparison.exact_bits=true`, `ids_equal=true`, prefill logits `max_abs=0.0`,
  logits SHA256 `e9d2fdc41f9b1f31…` (291-token) and `9643dfc8b6978c14…`
  (513-token / 511-chunk) identical to the recorded pre-landing baselines.
* **Product path** (fresh processes, 11 runs each, retained sample 10, prompt
  `benchmarks/prompts/glimmer_prefill_256.txt` MD5
  `973900074bfd15d4adeeecdff3359082`): interleaved same-session A/B with the
  O2×R8 routing restored in an otherwise identical tree (control `hipfire`
  SHA256 `c1eceaae202a6993…`, MD5 `b360ca7afef65adf…`) gives control
  `184.1 / 184.6 / 184.3` tok/s versus landed (`hipfire` SHA256
  `0f0cbeed75fa2532…`, MD5 `c0857bfb0eefef08…`) `191.4 / 190.9 / 190.8` tok/s,
  a further reproduction of `190.0 / 189.7 / 190.2`. Median 184.3 → 190.9
  (+3.6 %), first-token latency 1576.6–1581.0 ms → 1520.7–1533.8 ms.

## Interpretation recorded with the evidence

`prod_n291_xhot` (X footprint collapsed to a cache-resident 80 KB) leaves the
production arm unchanged, while `prod_n291_whot` (weights made L2 resident)
speeds it up 30 %: on this shape the exposed cost is expert-weight DRAM traffic,
not X bytes. The O4×R4 arm wins because it halves the X load instructions per
`(block, group)` at an unchanged 144 MAC issue, and loses ~8 % once weights are
cache resident because the decode/weight-load work per block doubles. The
modelled ~45 % L1-traffic reduction therefore lands as ~3 % wall time here.

Detailed tables, raw samples, rejected variants and artifact hashes:
[`moe-gate-o4r4/report.md`](../../.codeinsight+research/qwen4/pp500-kernel-check/moe-gate-o4r4/report.md).
