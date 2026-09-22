# gfx1201 fp8-KV long-prefill cliff

## Fixture and conclusion

On local card-B (`GPU-e475645fe0200397`, `gfx1201`) with the 27B QAT model `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`, fp8 KV, automatic VMM, `HIPFIRE_GRAPH=1`, the 32,768-position admission ceiling excludes *both* whole-chunk Q-resident/packet attention and segmented route-N FA2 from deeper prefill. The remaining fp8 flash-tile plus reduce fallback causes a discrete route change; the huge latency at ~50k is **not** explained by quadratic growth on the same FA2 kernel. The exact reason the fallback's latency escalates so sharply between ~41k and ~50k was not isolated independently; this patch instead restores the already-present FA2 route over the model's 262,144-position range. No kernel source or other KV mode changes.

Deterministic request uses the repository `benchmarks/longctx/niah/niah_16k.jsonl` filler repeated five times plus 14,442 characters and the fixture question (raw tokenizer: 57,019 tokens; prompt MD5 `0c939a51f79435d6aa9fd01987dda078`; JSON request MD5 `fbc7edca1797fc5dc02350a211da4af7`). The 66k run uses six repetitions plus 4,727 characters (raw tokenizer: 66,019; prompt MD5 `6de1e44b30652884d345d24690082c7e`). Both request one token, temperature zero, thinking disabled. Local request/trace artifacts are under `scratch-deepctx/`; neither generated prompts nor profiler CSV are committed.

## Predicate audit

`qwen35/prefill.rs` computes `max_ctx_len = start_pos + n` for eager prefill. With admitted `n=8192`, `commit_stride=512` and exact H24/KV4/D256 native fp8, the `packet_runs` predicate formerly required `max_ctx_len <= 32768`. At `start_pos=24576`, the next 8,192-row chunk ends exactly at 32768 and qualifies; at 32768 it ends at 40960 and fails. Failure means 16 sequential 512-row FA attention steps per full chunk, not one whole-chunk step. `families/attention.rs` independently limited Q-resident, packet and segmented route-N fp8 to `max_ctx_len <=32768`. Beyond the cap **each** 512-row step falls through to `attention_flash_fp8_e4m3_tile_batched` plus `attention_flash_asym_reduce_batched`. The `rdna-compute/src/attention.rs` native fp8 launchers also reject >32768. On gfx1201, the Q-resident flag has priority and is enabled by default; packet is the next enabled arm; route N serves eligible segmented rows. The 4096 scalar/flash crossover does not admit FA2. A final partial chunk is inherently segmented, but its 512-row steps can still use Q-resident after the fix.

| 8,192-row span | Old host route | New host route |
| --- | --- | --- |
| 0..8192 | whole Q-resident | whole Q-resident |
| 16384..24576 | whole Q-resident | whole Q-resident |
| 24576..32768 | whole Q-resident (inclusive boundary) | whole Q-resident |
| 32768..40960 | 16 segmented fp8 flash tile + reduce | whole Q-resident |
| 40960..49152 | 16 segmented fp8 flash tile + reduce | whole Q-resident |
| 49152..57344 (full 66k case) | 16 segmented fp8 flash tile + reduce | whole Q-resident |

Changed only these matching ceilings to `262_144`: `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` whole-chunk gate; three fp8 arms in `crates/hipfire-dispatch/src/families/attention.rs`; three native-fp8 launch guards in `crates/rdna-compute/src/attention.rs`. Other gating (architecture, shape, tree, batch/run size, quantization flags) is untouched. Packet/Q-resident query ownership is fixed 512-row runs in grid-z; native KV uses 1,032-byte token rows and 64-bit address arithmetic. Q scratch scales with batch, not context; no 32k-sized scratch or LDS allocation is assumed in those FA2 launches. The model reports `max_seq=262144` and VMM `physical_cap=262144`; the new upper bound stops at its advertised context rather than opening an unbounded route.

## Observed timings and trace

Temporary per-chunk timing was compiled into separate before/after binaries, then removed from this patch. Both ran the same card and model; first chunk includes startup/JIT. Before: 0..8192 `9.035s`, 8192..16384 `0.517s`, 16384..24576 `2.486s`, 24576..32768 `2.780s`, 32768..40960 `4.015s`. After that, the baseline reached the VMM mapped-prefix transition `42674->50803` and did not complete its next 8192-row chunk in >120s; I interrupted the request at 2m33s (no HTTP response). The after binary completed the longer 66k request, HTTP 200 in `42.234s`: 0..8192 `9.038s`, 8192..16384 `0.495s`, 16384..24576 `2.486s`, 24576..32768 `2.780s`, 32768..40960 `3.120s`, 40960..49152 `3.434s`, 49152..57344 `3.748s`, 57344..65536 `4.064s`, final 494 rows `4.382s` (includes last-token work). The identical shallow timings and smoothly increasing deep full-chunk times distinguish the fix from a generic initialization change. These are single local runs, not statistical speedup estimates.

A final-source **profiled 57k** request completed HTTP 200 in `55.718s` using `rocprofv3 --kernel-trace`; `scratch-deepctx/after-trace/hiptrx/1718051_kernel_trace.csv` contains 20,497 dispatches. The default `attention_fp8_e4m3_fa2_gqa_qresident_gfx1201` appeared 96 times at grid-z=16 (six complete 8192-row chunks, 16 FA layers; 5,517.085 ms summed GPU kernel time) and 256 times at grid-z=1 (16 segments of the non-multiple tail across 16 FA layers; 1,991.265 ms summed). `attention_flash_fp8_e4m3_tile` and `attention_flash_q8_0_reduce` each appeared only 16 times, for the one-token decode after prefill, not for deep prefill. A separate rocprof attempt on the stalled baseline was interrupted before profiler flush and yielded no CSV; the old fallback symbol is established by the exhaustive host predicate chain and a direct fallback launch oracle, **not** by a captured before-serve kernel trace.

A direct gfx1201, start=32768, 8192-row fp8-KV oracle trace at `scratch-deepctx/oracle-trace/hiptrx/1697649_kernel_trace.csv` recorded one packet attention launch at `108.429ms`, 16 segmented route-N attention launches at `282.007ms` total plus 16 pre-conversions at `1.010ms`, and one Q-resident at `91.507ms`; fallback sampled on eight deepest rows took `12.239ms` tile + `0.355ms` reduce. This synthetic single-layer trace proves launch identities and arithmetic checks, **not** the wall time of a full-model request.

The oracle compared *all* 50,331,648 f32 outputs for the deep whole packet against 16 segmented 512-row route-N launches, with **zero bit mismatches**. Q-resident/packet versus the production fallback is not bit-exact (FA2 quantizes Q to E4M3, unlike the fallback); over 8 deep rows ×24 heads×256 dimensions (49,152 outputs), packet versus fallback had max absolute difference `0.000133132`, RMSE `0.000031425`; Q-resident versus fallback had max absolute difference `0.000140932`, RMSE `0.000034581`, all finite. Thus the route is numerically close, but no bit-exact claim is made for the default Q-resident versus the old fallback.

## Regression checks

Scoped `cargo build --release -p hipfire-daemon --bin daemon` succeeded; `git diff --check` was clean. The fp8 WT2 prefill scorer with `--max-chunks 24`, `--kv-mode fp8 --kv-v q8 --scoring-mode prefill` completed 24,552 scored tokens in 74.2s and produced slice-mean KLD **0.083278**, mean NLL 1.886010, PPL 6.5930 (`scratch-deepctx/wt2-fp8-c24.bin`). One initial scorer attempt failed `hipMalloc` while my profiled daemon still held 23.5GB VRAM; after releasing that *same* daemon, the rerun passed. That initial capacity collision was not a kernel regression.

The existing gfx1201 baseline guard verified automatic fp8 VMM but **did not meet** its `3620 tok/s` strict pp8192 process floor in two independent fresh attempts. First process-1 samples `[3619.9, 3618.0, 3615.0]`, median **3618.0**; retry samples `[3616.9, 3617.4, 3611.8]`, median **3616.9**. Decode medians were 36.47 and 36.43 tok/s (reference 36.5). The guard intentionally stops at the first under-floor process, so there is no three-process PASS. Both misses are within 0.09% of the floor; no guard threshold was loosened and no pass is claimed. Raw guard evidence: `scratch-deepctx/guard/` and `scratch-deepctx/guard-retry/`. The paired comparison below resolves attribution, not the guard's literal failed result.

### Paired shallow-path comparison

To distinguish an actual shallow regression from card clock/thermal drift, ran one untimed baseline warmup process, then A1/B1/B2/A2 on card-B in one session, each fresh isolated HOME, exact guard flags (`--matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`, `HIPFIRE_GRAPH=1`). A uses the read-only `wt-vmmdefault` binary at source HEAD `f2d0217e5` (that commit differs from the fix's parent `a3531e9` only in `scratch-vmmdefault/REPORT.md`); B uses this branch's built binaries. Every process log independently verified `KV cache: Fp8 vmm`. The discarded warmup measured pp8192 3672.70 tok/s.

| Process | pp512 median tok/s | pp8192 median tok/s | decode median tok/s |
| --- | ---: | ---: | ---: |
| A1 baseline | 3377.00 | 3655.30 | 36.49 |
| B1 fix | 3331.90 | 3659.60 | 36.52 |
| B2 fix | 3344.00 | 3668.70 | 36.55 |
| A2 baseline | 3347.30 | 3666.80 | 36.56 |

All four measured pp8192 medians clear 3620. The B mean is **3664.15** vs A **3661.05** tok/s (+0.085%); no consistent B slowdown, despite the preceding unpaired guard misses. This is evidence of run-to-run variation, not a claim that the strict scripted guard passed. The changed eager `max_ctx_len` is computed as `start_pos+n` (`prefill.rs:11905-11920`); for the widened pp8192 shallow bench it is 8192 rather than VMM's 262144 physical capacity, so neither the prefill whole-run guard nor the dispatch's three fp8 guards change truth value. Capture mode can use physical capacity, but this measured path reports widened `commit_stride=512` and is eager. Per-process JSON and daemon logs are in `scratch-deepctx/abba/summary.json` and adjacent files.
