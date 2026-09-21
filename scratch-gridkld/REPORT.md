# iu4 shift-fold artifact and route quality matrix

## Result

No power-of-two grid clears both production route budgets today. The full-pow2 artifact passes the iu4 c24 budget after row-global A (`0.071194 <= 0.10`) but fails fp8v2 at `0.071194 > 0.05`. Pow2h is the closest common-grid candidate: with row-global A it scores iu4 `0.055408` and independently measured fp8v2 `0.055408`, so it misses the hard fp8v2 budget by **0.005408**. The shipped non-pow2 control remains the only grid in this table that clears both arms (`0.081199` iu4 and `0.045510` fp8v2), but it cannot supply the power-of-two scale contract required by shift-fold.

Grid verdict: **KILL both current pow2 artifacts as a dual-arm shipping grid; pending QAT run 8, retain pow2h as the cheapest path to a shippable replacement.** No fp8 kernel or loader change is required: the only remaining blocker is fp8v2 quality.

## Method

The independent decisive cells were run on card-E (`GPU-05f92432f2312a0e`) from branch `gfx1201-grid-kld`, base `6be650494`, with RowGlobalA commits `2977acb21` and `1b0e6a1bc` cherry-picked as `c5104c207` and `f71b29a7e`. The evaluator was built with:

```text
cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet
```

Runs used `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`, fp8 KV, q8 V, prefill scoring, `HIPFIRE_A4_ROWGLOBAL=1`, and `HIPFIRE_IU4_PREFILL=0`. c2 uses `--max-chunks 2`; c24 uses `--max-chunks 24`. Raw logs and score binaries are beside this report.

## Quality matrix

Budgets are iu4 c24 `<= 0.10` and fp8v2 c24 `<= 0.05` (hard). Bold rows are the independent card-E measurements from this worktree. Other numeric rows are the established campaign controls supplied with the assignment or recorded by the preceding card-C evaluator runs; they are included to make the grid decision explicit rather than presented as new card-E measurements. A dash means the non-decisive c2 control was not supplied or rerun.

| artifact | row-global A | compute route | c2 | c24 | c24 gate |
|---|:---:|---|---:|---:|---|
| shipped `sym-a035.qat-r5s100` | off | iu4 | 0.064864 | 0.081199 | pass iu4 |
| shipped `sym-a035.qat-r5s100` | off | fp8v2 | — | 0.045510 | pass fp8v2 |
| shipped `sym-a035.qat-r5s100` | on | iu4 | 0.037143 | 0.045510 | pass iu4 |
| shipped `sym-a035.qat-r5s100` | on | fp8v2 | — | 0.045510 | pass fp8v2 |
| full-pow2 `sym-pow2-a035` | off | iu4 | — | 0.105319 | **fail iu4** |
| full-pow2 `sym-pow2-a035` | off | fp8v2 | — | 0.070638 | **fail fp8v2** |
| full-pow2 `sym-pow2-a035` | on | iu4 | 0.058773 | 0.071194 | pass iu4 |
| **full-pow2 `sym-pow2-a035`** | **on** | **fp8v2** | **0.058773** | **0.071194** | **fail fp8v2** |
| pow2h `sym-pow2h-a035` | off | iu4 | 0.071897 | 0.091338 | pass iu4 |
| pow2h `sym-pow2h-a035` | off | fp8v2 | — | 0.057072 | **fail fp8v2** |
| pow2h `sym-pow2h-a035` | on | iu4 | 0.046252 | 0.055408 | pass iu4 |
| **pow2h `sym-pow2h-a035`** | **on** | **fp8v2** | **0.046252** | **0.055408** | **fail fp8v2 by 0.005408** |

The independent full-pow2 fp8v2 result confirms `0.071194`. The independent pow2h fp8v2 result establishes the run-8 baseline at c2 `0.046252`, c24 `0.055408`. Row-global A is an int4 activation-sidecar change; fp8 activation packing is already row-wide, which is why the fp8 rows do not gain a distinct packing mode from this flag.

## FP8 compatibility of pow2-marked artifacts

The fp8v2 route can read both pow2 artifacts without a compatibility change. The marker describes a restriction on the values already stored in the ordinary MQ4V2 f16 scale field; it does not introduce a new tensor type, payload layout, or kernel ABI.

- Direct metadata inspection confirms `mq4v2.pow2scale=1` on full-pow2 and `mq4v2.pow2scale=2` on pow2h; both also carry `mq4v2.symmetric=1`.
- The HFQ loader parses `mq4v2.symmetric` at `crates/hipfire-runtime/src/hfq.rs:642-646`. There is no runtime or compute consumer of `mq4v2.pow2scale`; a repository-wide source search finds no such key or field.
- Load dispatch propagates only the symmetric bit to each device at `crates/hipfire-runtime/src/model_load.rs:425-428`.
- FP8 v2 dispatch selects its symmetric fold from that symmetric bit at `crates/rdna-compute/src/gemm.rs:9377-9380`, then passes the ordinary weight-buffer pointers to the unchanged v2 ABI (`gemm.rs:9531-9556`).
- The kernel documents the serialized 136-byte MQ4V2 group as two packed f16 `(scale, zero)` headers plus nibble payload at `kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:22-25`. Its v2 path loads the f16 scale directly from that header and converts it to f32 at lines 645-657, then consumes it in the half-group fold at lines 879-883.

Therefore a pow2-marked artifact is legal for both iu4 and fp8v2. The fp8 kernels simply observe f16 scales whose values happen to be powers of two. Pow2h QAT only needs to recover at least `0.005408` c24 KLD on the fp8v2 route while preserving the existing symmetric MQ4V2 layout and pow2 scale constraint; no loader, dispatch, or kernel-compatibility work is on the critical path.

## Evidence

- `full-pow2-rowglobal-fp8v2-c2.log` / `.bin`: `0.058773`.
- `full-pow2-rowglobal-fp8v2-c24.log` / `.bin`: `0.071194`.
- `pow2h-rowglobal-fp8v2-c2.log` / `.bin`: `0.046252`.
- `pow2h-rowglobal-fp8v2-c24.log` / `.bin`: `0.055408`.
