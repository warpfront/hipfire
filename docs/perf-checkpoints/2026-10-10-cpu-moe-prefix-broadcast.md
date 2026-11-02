# CPU MoE MQ3/MQ6 prefix-broadcast decoding — 2026-10-10

**Lifecycle:** historical. **Disposition:** provisional keep, fixture-bound measured evidence. Not a product admission, portable speedup, current default, or proof that the entire CPU splice is compute-bound.

Implementation: `105dbb717` (`perf(cpu-simd): fold 3/6-bit prefix removal into vector decode`). Comparison retains the adaptive Rayon policy documented in [the 2026-10-09 checkpoint](2026-10-09-cpu-moe-adaptive-row-grain.md). No scheduling or GPU/CPU transfer change.

## Decision and computational evidence

Keep the focused integer-decoding change: both candidate batches had higher trace-disabled end-to-end throughput than both original-decoder batches. All six candidate samples exceeded all six baseline samples. Completion lengths and diagnostic latency ranges vary, so the measured magnitude remains qualified.

The changed function is `crates/hipfire-cpu/src/simd/x86.rs::codes8`:

- **3-bit:** broadcast the 32-bit field containing one preceding byte directly from memory, then shift lanes by `[8,11,14,17,20,23,26,29]`. This replaces the old scalar load, right shift, register transfer and broadcast. The prefix is discarded and every code is unchanged.
- **6-bit:** load two 32-bit fields covering four-code/24-bit halves and their preceding bytes. Shift each half's lanes by `[8,14,20,26]`. This replaces scalar assembly of a 48-bit word and extraction of its upper half.
- **4-bit:** unchanged, including MQ4G256's specialized 16-byte nibble expansion. Its single 32-FMA chain is a separate constraint, not the same decoding defect.

`objdump -Cd --disassemble=... target/release/daemon` confirmed that MQ3-Lloyd now uses memory `vpbroadcastd` followed by `vpsrlvd`, without the per-chunk scalar `shr`/`vmovd`. MQ6 uses memory `vmovd`, lane insertion/shuffle and vector shifts, without the old per-chunk scalar stitching. LLVM also fully unrolled MQ3-Lloyd's 32-chunk loop: its body grew from approximately 220 to 900 bytes. That is a code-generation consequence of the same source change, not an independently isolated optimization.

**[INFERENCE]** Decoder instruction overhead contributes to this fixture's GU/DN latency. Neither assembly nor throughput establishes a global compute-versus-memory classification. CPU PMU measurement was unavailable: `perf` was absent and `perf_event_paranoid=2`. Exact selected GU/DN quant mixtures are not reported by the existing trace: `quant=` is the first selected expert's GU format, not a per-projection histogram. Consequently, no per-format timing or precise bandwidth rate is inferred from those labels.

The [earlier int8-activation null](2026-10-04-moe-cpu-exec-int8-activation.md) is relevant historical memory-path evidence, but tested a V2/qt44 route, not this graded qt13/15/20 fixture. It does not prove that the current route is bandwidth-bound. Weight traffic, decode throughput, cache misses and latency hiding remain possible joint constraints.

## Pointer bounds and arithmetic contract

`read_unaligned()` permits unaligned access; it does not establish bounds. For the changed decoders, `p` points into a payload after a nonempty group header, never to an allocation's first byte:

- 3-bit reads `[p-1, p+3)`.
- 6-bit reads `[p-1, p+3)` and `[p+2, p+6)`.
- The first prefix is inside an 8-byte header, or the 16-byte MQ3-Lloyd codebook. Later prefixes are already inside the payload. Neither read passes the chunk's end, including the final group of the final row.

The unsafe helper documents the same-allocation prefix precondition. Both generic consumers enforce a nonzero payload offset for 3/6-bit formats with compile-time assertions. A regression test checks every possible code in every lane against all 256 prefix-byte values, using an exact-size prefix-plus-chunk buffer.

All floating-point FMAs, chain assignments, horizontal reductions, group accumulation, affine correction, activation preparation and row-major output layout are unchanged. MQ4 is not reassociated in this checkpoint. Obsolete `load_le` 3/6-bit arms were removed. Single-token and multi-row paths use the same decoder.

## Fixture and method

- CPU: AMD Ryzen 7 7800X3D, default 16 logical Rayon workers; `RAYON_NUM_THREADS` unset.
- GPU: RX 9070 XT, gfx1201, HIP 7.2.
- Target: `qwen3.6:35b-a3b`, local `qwen3.6-35b-a3b.mq4p`; MD5 `f9b5b13eb24ecbe4b39f3a3f4a297645`.
- Prompt: `benchmarks/prompts/merge_sort_thinking_off.txt`; MD5 `253c7ac50857fe6d0e10fb0d2c5e35c0`.
- Environment: `HIPFIRE_OFFLOAD_EXEC=cpu`, `HIPFIRE_MOE_EXPERT_BUDGET=24`, `HIPFIRE_GRAPH=0`, `HIPFIRE_REPLAY_BACKEND=hip`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIP_VISIBLE_DEVICES=0`; thinking off.
- Command: `bash .codeinsight+research/cpu-exec-debug-run.sh`, unchanged. The generation command is `hipfire run --spec off --max-tokens 256 --kv-mode q8 --kv-backend vmm -j` with the committed prompt.
- Scoped `max_seq=4096`; restoration to `200000` verified by the runner. All runs reported 16/16 spilled layers CPU-covered and no host-mapped expert work left on the GPU.
- All performance measurement and analysis delegated to `hipfire-benchmark-runner`; no replacement harness or concurrent benchmarks.
- B-C-B-C design: three fresh-process baseline invocations, three candidate, three restored-baseline, three restored-candidate. Each invocation contains a diagnostic trace process and a fresh trace-disabled control, flock-serialized.
- Control `tok_s` is generated tokens divided by prefill-plus-decode wall time, excluding model load/spawn/terminal handshake. It is the end-to-end selection metric here, not a decode-only rate. Trace-enabled `tok_s` is not used for selection.

The mq4p recipe inventory is 51 MQ6, 77 MQ4 and 128 MQ3-Lloyd experts per layer, with matching GU/DN tiers (`scripts/gen_tier_map.py`; `docs/investigations/2026-08-04-a3b-lowbit-quality.md`). This inventory is not the routing-weighted selected mix.

## Measurements

GU/DN/total are diagnostic milliseconds per splice: median of each invocation's trace lines. The pooled values are medians of those per-invocation medians, not throughput measurements. The trace uses a print gate, not an exhaustive per-token log.

| Arm | Trace-off control samples, tok/s, execution order | Median tok/s | GU ms | DN ms | Total splice ms |
|---|---|---:|---:|---:|---:|
| Original decoder, initial | 48.6, 49.7, 46.2 | 48.6 | 0.270 | 0.140 | 0.670 |
| Prefix decoder, initial | 50.2, 52.0, 50.8 | 50.8 | 0.250 | 0.130 | 0.645 |
| Original decoder, repeat | 47.1, 48.4, 50.0 | 48.4 | 0.270 | 0.150 | 0.680 |
| Prefix decoder, repeat | 50.7, 50.8, 50.9 | 50.8 | 0.240 | 0.130 | 0.640 |
| Original decoder, pooled, n=6 | All original samples above | 48.5 | 0.270 | 0.140 | 0.675 |
| Prefix decoder, pooled, n=6 | All prefix samples above | 50.8 | 0.245 | 0.130 | 0.640 |

Runner-computed pooled changes: **+2.3 tok/s (+4.7%)**, GU **−9.26%**, DN **−7.14%**, total splice **−5.19%**. Diagnostic ranges overlap; these medians do not imply separation of every splice or a universal improvement.

### Every invocation

Stamps below are `20261010-HHMMSS`.

| Stamp | Arm | Control tokens | Control tok/s | GU ms | DN ms | Total ms |
|---|---|---:|---:|---:|---:|---:|
| 035214 | Original initial | 166 | 48.6 | 0.270 | 0.140 | 0.670 |
| 035404 | Original initial | 166 | 49.7 | 0.250 | 0.140 | 0.655 |
| 035536 | Original initial | 166 | 46.2 | 0.280 | 0.140 | 0.690 |
| 040206 | Prefix initial | 166 | 50.2 | 0.270 | 0.140 | 0.690 |
| 040335 | Prefix initial | 187 | 52.0 | 0.250 | 0.130 | 0.645 |
| 040506 | Prefix initial | 162 | 50.8 | 0.240 | 0.120 | 0.640 |
| 040819 | Original repeat | 151 | 47.1 | 0.280 | 0.150 | 0.690 |
| 040948 | Original repeat | 166 | 48.4 | 0.240 | 0.130 | 0.645 |
| 041107 | Original repeat | 156 | 50.0 | 0.270 | 0.150 | 0.680 |
| 041409 | Prefix repeat | 166 | 50.7 | 0.250 | 0.130 | 0.640 |
| 041543 | Prefix repeat | 160 | 50.8 | 0.240 | 0.130 | 0.640 |
| 041716 | Prefix repeat | 166 | 50.9 | 0.235 | 0.120 | 0.630 |

Completion lengths vary from 151 to 187 tokens despite unchanged requests, affecting prefill amortization. A secondary, post-hoc check restricted to 166-token controls has baseline samples 46.2/48.4/48.6/49.7 (median 48.5, n=4) and candidate 50.2/50.7/50.9 (median 50.7, n=3). Its direction agrees, but it is not a selection rule or a replacement for all twelve primary samples.

The user reported higher CPU fan activity with the rewritten decoder than the original. Record this as a qualitative thermal/activity observation, not measured CPU utilization, power or PMU evidence. No energy-efficiency claim is made.

## Provenance and durable evidence

| Arm | CLI MD5 | Daemon MD5 |
|---|---|---|
| Original initial | `c068a289d2e3ad8ff83cb6f3354e4075` | `9595741f965d6aa8266fe78428b0c162` |
| Original repeat | `b64efa93867c7ef39c4b37224c0dde2c` | `9595741f965d6aa8266fe78428b0c162` |
| Prefix initial and repeat | `b64efa93867c7ef39c4b37224c0dde2c` | `34fad9e1adaa0e6c204228363e4ea483` |

The original repeat daemon byte-matches the original baseline. The restored candidate daemon byte-matches the first candidate after dead-helper/test-buffer cleanup. CPU GEMV executes in the daemon; the CLI has no matching MQ3-Lloyd kernel in its disassembly. Both executables' actual identities are retained rather than requiring an irrelevant CLI hash change.

[Consolidated durable receipt](data-2026-10-10-cpu-moe-prefix-broadcast.json) preserves every control/trace generation's content, token count, rate and finish reason, per-invocation diagnostic medians, binary identities, fixture and source artifact paths. Local raw invocation files remain discovery pointers: `.codeinsight+research/cpu-exec-debug-20261010-<stamp>.{json,control.json,trace.log}`. The consolidated source was `.codeinsight+research/cpu-gemv-prefix-broadcast-20261010.summary.json`.

## Correctness and exercised runtime

- `cargo test -p hipfire-cpu --lib`: **36 passed**, 0 failed.
- `cargo test --release -p hipfire-cpu --lib`: **36 passed**, 0 failed, repeated after candidate restoration/cleanup.
- `cargo test --release -p hipfire-dispatch moe_splice_tests -- --nocapture`: **4 passed**, 0 failed, repeated after restoration/cleanup.
- `cargo build --release --bin hipfire --bin daemon`: successful for candidate, original repeat and restored candidate.
- A removed throwaway correctness probe produced identical before/after output-bit fingerprints for all 28 formats, seven weight rows, four activation rows and K=256/512/2048/12288 with non-power-of-two metadata. The permanent integer-code regression independently covers all MQ3/MQ6 code values and prefix bytes.
- Existing shared-job tests retain bitwise comparison against independent per-token GEMV across mixed quant tiers, 0–4 activation rows, tail blocks, stack/heap job storage and 1/2/4/8 workers. Existing graded/uniform splice tests exercise multi-row grouping.
- The actual CLI/daemon CPU-offload path completed all runs with coherent merge-sort Python and `finish_reason=stop`. The final control emitted 166 tokens at 50.9 tok/s; its generated-content MD5 was `dff503e84f0d6046bb7c6749ea43afb8`. Output was read, not accepted solely from rates.

This is AR performance evidence. CPU-deferred end-to-end MTP remains guarded as documented in the unchanged 2026-10-09 record; neither loader/runtime guards nor speculation behavior changed. Multi-row arithmetic is exercised, not end-to-end MTP performance. No GPU numeric kernel or replay implementation changed.

## Limits and next computational target

The retained result demonstrates a useful decoder improvement on this exact fixture, not that CPU GEMV is at its hardware limit. A sharper bottleneck claim requires CPU counters and a real selected-format histogram. MQ4's one-chain dot is the next visible computational constraint; hiding it with independent-row interleaving preserves exact order, while splitting its accumulator changes rounding and requires explicitly relaxed numerical acceptance plus a separate measured comparison. Neither is included in this checkpoint.
