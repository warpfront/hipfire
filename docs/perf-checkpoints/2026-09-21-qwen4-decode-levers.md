# Qwen4 (Qwen3.8-Flash-Next) decode — QSA selection ranking, live-row BF16 boundaries, unrolled bf16 GEMV (gfx1151) — 2026-09-21

**Lifecycle:** `historical`

**Disposition:** landed as measured local deltas on `gfx1151`. Not a G5
admission, not a retained-replay certification, not a cross-architecture
result, and not a product speed-floor update.

## Fixture

- Host `halo`, gfx1151, 128 GB UMA, HIP 7.2 via `~/.hipfire/rocm-merged`.
- Model `~/.hipfire/models/qwen3.8-flash-next.mq4r`, size `178001249816`,
  sha256 `7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`
  (byte-identical to the `model_sha256` in the branch's retained-route proof).
- Base commit `fd3426a43`, branch `feat/qwen38-flash-next`.
- Daemon md5 per arm: baseline `9d60cb4251aef530c33541942f36aa38`,
  l1 `f758f4d483d713888f6bfe84f082b89f`, l3 `a805ebbc370aee841529ca54d8e48aae`,
  l4 `70b714a508b24bc3070df1dec6080ade`.

## Method

`bench_decode` through the resident daemon (prime the context, then time
single-token AR forwards), one fresh process per arm, retained PM4 route,
`HIPFIRE_GRAPH=0 HIPFIRE_AR_GRAPH=0`, private `HOME` and private
`HIPFIRE_KERNEL_CACHE` per arm, `HIPFIRE_DPM_WARMUP_SECS=10`, 4 runs per arm,
interleaved arms, run 0 discarded (PLE cold start + first-use kernel JIT).
Without the DPM warmup the same binary spanned ±10% between runs; with it,
repeat arms agree to 0.2%. Raw per-run samples, the driver, the sweep script
and the scratch oracles live in `.codeinsight+research/qwen4/lever-ab/`
(`RESULTS.md`, `proto-*.json`, `probe-*.json`).

## Change

1. **`indexed_attention_select_f32_batched`** (`kernels/src/tensor_ops.hip`)
   scored its blocks in parallel and then ranked them with a selection sort
   owned by thread 0 — `O(chosen x blocks)` on one lane while 255 lanes idled.
   At ctx 2000 that is 70 ms/token of a 150 ms token (12 calls, 5.8 ms each).
   Each live block now computes its own rank under (score desc, index asc),
   which is the permutation the selection sort produced. Live scores are
   `>= 0` (`fmaxf` against 0, summed per head), so the order is strict, the
   ranks distinct, and the old `-FLT_MAX` sentinel cannot collide with a real
   score. The causal tail is unchanged.
2. **Decode MoE source-BF16 boundaries**
   (`crates/hipfire-dispatch/src/pipeline/mod.rs`) rounded the *whole*
   512-row prefill-cap scratch buffer for a single-row decode step: 7 of the
   15 per-layer `bf16_round_trip_f32` calls (336 of 720 per token) carried 512x
   the live data and cost ~3.5 of the kernel's 3.92 ms/token. Each boundary is
   now bound to the extent its consumer reads — one router row (`p.n_exp`), the
   shared-expert selector's single scalar (with its sigmoid), one `hidden` row
   for the combine and shared-down outputs, and the live `k_top` slots of
   `topk_weights`. `build_moe_decode` binds `batch_size: 1`, so these stages are
   single-row by construction.
3. **`gemv_bf16_xf32`** (`kernels/src/gemv_bf16_xf32.hip`) unrolls its K loop
   eight steps. Each sub-step keeps the original per-chain add order and `i`
   only increases, so every output row is bit-identical; the point is eight
   independent `dwordx4` weight loads in flight per lane instead of one, which
   matters for the low-`m` decode shapes (hc down `320x10240` at 124 GB/s,
   `(2560,6144)` at 128 GB/s, against 218-227 GB/s for the many-block shapes).

## Evidence

Bit-exact parity, not tolerance:

- `qwen4_probe` (scratch oracle: 17 teacher-forced positions + a 291-token
  prefill + 5 single-token decode steps, sha256 of the logits row(s) and of the
  full recurrent/attention/PLE state): `baseline` vs `l1` and vs `l1+l3+l4` are
  identical on every field, including the decoded ids
  (`1414, 488, 3766, 7701, 310`) and the prefill argmax (`760`, logits sha
  `e9d2fdc41f9b1f31…`, the recorded 291-token fixture digest).
- `cargo test -p rdna-compute --lib batched_select` on gfx1151: the new ranking
  path matches the retained serial symbol byte-for-byte over 168 randomized
  (compress, heads, dim, rows, blocks, budget) combinations, plus two analytic
  cases pinning (score desc, index asc) and the causal tail.
- `gemv_probe` (scratch, 12 production decode shapes): the unrolled variants
  are bitwise-equal to the production kernel on every output row.

Decode, warm medians of runs 1..3 (us/token):

| arm | ctx 128 | ctx 2000 | tok/s @2000 |
|---|---|---|---|
| baseline | 71691.3 / 71566.1 | 149941.1 | 6.67 |
| l1 (select) | 72145.1 / 71174.7 | 81364.1 | 12.29 |
| l3 (+ live-row boundaries) | 64752.6 / 65095.6 / 64855.8 | 75439.5 | 13.25 |
| l4 (+ unrolled GEMV) | 60967.2 / 60876.9 | 71524.9 | 13.98 |

- ctx 128: −9.4% for l3, −14.9% for l4 (baseline drift across rounds 0.2%;
  every l3/l4 run is below every baseline run).
- ctx 2000: −45.7% (l1), −7.3% (l3), −5.2% (l4) → **149.9 → 71.5 ms/token,
  6.67 → 13.98 tok/s**.
- l1 at ctx 128 is inside the noise, as the profile predicts: the select kernel
  costs 0.93 ms/token there and 70 ms at ctx 2000. The Qwen4 serving cap is
  `max_seq = 2048`, so the context term is the whole `[128, 2048]` range.

## Not claimed / not done

- **Dense-trunk quantization is not attempted.** The other 72% of ctx-128
  device time is 773 `gemv_bf16_xf32` calls moving 8.94 GB/token of BF16 dense
  weights at ~180 GB/s (ceiling 223.9 GB/s). Cutting it needs a regenerated
  ~178 GB artifact (the producer emits BF16 dense), lowering support for
  quantized dense projections (`project_bf16_batch` refuses non-BF16 by
  construction: `UnsupportedVariant("non-bf16-stateful-projection")`,
  `crates/hipfire-dispatch/src/pipeline/layer_ops.rs`), and an acceptance move
  from byte parity to a tolerance route. That is a program, not a lever, and no
  partial subset is reachable in-session because the weights live inside the
  artifact.
- **Retained-route certification of the changed artifacts was not re-proved.**
  Every arm in this harness logs `certified_artifacts=0/16` with
  `unknown_launches=2749` (identically for baseline and candidate, so the
  *deltas* stand) while still reporting `qwen4 retained body prepared:
  transport=pm4 dispatches=2845`. An earlier decode-profile session on the same
  checkout did report `certified_artifacts=16/16`,
  `certified_launches=2845 fallback_launches=0 unknown_launches=0`, so the
  certified route exists here; the harness environment is what differs.
- **Adjacent architectures were not measured.** The unrolled GEMV body is
  shared with other users of `gemv_bf16_xf32` (arch_id 12 Cohere2-MoE among
  them). The rewrite is bit-identical and only ever issues the same loads in
  fewer, wider iterations, but its perf on gfx1100/gfx1201/gfx906 is unmeasured;
  only gfx1151 was exercised here.
