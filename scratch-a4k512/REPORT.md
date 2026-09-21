# Single-pass coarse IU4 activation-scale experiment

## Verdict

**KILL.** The implementation meets the single-pass and performance constraints, but the quality premise does not survive the producer ownership boundary. `HIPFIRE_A4_GROUP=512` can coarsen only the RMSNorm/FWHT producer without changing the existing launch geometry; that mixed K512/K128 route moves shipped-artifact IU4 c24 from **0.081199 to 0.089196**, the wrong direction. The experimental uniform K256 route is also worse at c24 (**0.085473**). Neither result frees quality budget for the proposed power-of-two weight-scale package.

The performance result is nevertheless decisive: this is not the row-global producer-cost failure. The corrected producer subtotal is **42.481 us/token** versus **39.413 us/token** for K128, paired pp8192 is **-0.667%**, TTFT is **+0.287%**, and paired tg128 is **+0.045%**. The candidate is killed on quality, not speed or correctness.

Source milestone: `e70ccde8f` (`add single-pass coarse int4 activation scales`). Base: `6be650494`.

## What was inherited from RowGlobalA

Read-only references: branch `gfx1201-rowglobal-a`, commits `2977acb21`, `1b0e6a1bc`, report commit `c1f774b8b`, and `scratch-rowa/REPORT.md`.

That report supplied two working premises:

1. It attributed a large quality win to one scale for the entire activation row (c24 0.045510 versus 0.081199 for shipped K128) and reported that offline per-K512 measurement equaled row-global.
2. Its implementation writes the producer result to an activation plane and launches a second reduction/quantization pass because no grid-wide barrier exists. The reported extra full-plane traffic moved producers from about 39.5 to 61.3 us/token and killed prefill/TTFT.

The present implementation instead retains a complete scale window in one wave's registers, reduces its amax once, and directly writes the usual K128 payloads and sidecars. It adds no activation-plane materialization, no second kernel, and no second pass.

## Existing ownership: where K512 does and does not fit

| Consumer input emitted by fused producer | Producer / launch ownership | Complete contiguous K512 in one workgroup? | `HIPFIRE_A4_GROUP=512` behavior |
|---|---|---:|---|
| qkvza, gate/up, qkv projections | `fused_rmsnorm_mq_rotate.hip`; one 256-thread workgroup per row, eight waves loop over the full K row | **Yes** | One wave owns and retains each K512 window, reduces one amax, emits four K128 blocks with that scale |
| down projection | `fused_silu_mul_mq_rotate*.hip`; one 32-thread workgroup owns one K256 slice (`grid.x = K / 256`) | **No** | Remains K128 |
| linear-attention output projection | `gated_norm_mq_rotate_quant.gfx12.hip`; one 64-thread workgroup owns one K256 slice / two heads | **No** | Remains K128 |
| full-attention output projection | `mq_rotate_x_i4.hip` and its sigmoid variant; one 32-thread workgroup owns one K256 slice | **No** | Remains K128 |

A K512 scale across the last three families would require communication between adjacent workgroups or redundant rereads. Neither is part of their existing launch geometry, so those paths deliberately stay K128 as required.

For diagnosis only, `HIPFIRE_A4_GROUP=256` coarsens every fused IU4 producer. K256 is exactly one already-owned slice for the three small-workgroup families and is also a register window in RMSNorm. It therefore remains single-pass too.

## Implementation

- `HIPFIRE_A4_GROUP` defaults to 128. Supported experimental values are 256 and 512; unsupported values and incompatible K dimensions fall back to 128.
- `block_i4_128_quant.hip` now has a fixed-scale emitter and a wave-local register-window helper. Payload quantization and integer sums remain the ordinary K128 format; only the repeated sidecar scale changes.
- `fused_rmsnorm_mq_rotate.hip` assigns a complete K256/K512 window to one wave, keeps final FWHT values in registers, performs one wave amax reduction, and writes two/four K128 blocks.
- Separate kernel source constants and unique symbols (`*_k256`, `fused_rmsnorm_mq_rotate_awq_i4_gfx12_k512`) prevent cache aliasing and make route proof explicit.
- `HIPFIRE_A4_GROUP=512` selects K512 only for RMSNorm; other families select their baseline symbols. `HIPFIRE_A4_GROUP=256` selects unique K256 symbols for all four fused producer families.
- No format change, GEMM change, alias, compatibility shim, or model symlink change was made.

## Quality

Shipped artifact remained `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r5s100.hfq`. Direct `eval_hipfire` probes used the built runtime and the candidate flag in the same process, so they are not subject to daemon configuration replacement.

| Activation granularity / route | IU4 c2 KLD | IU4 c24 KLD | FP8v2 c2 KLD | FP8v2 c24 KLD |
|---|---:|---:|---:|---:|
| K128 default | 0.064864 | 0.081199 | 0.037143 | 0.045510 |
| K512 where owned, otherwise K128 | 0.071641 | 0.089196 | 0.037143 | 0.045510 |
| uniform K256 diagnostic | 0.063732 | 0.085473 | 0.037143 | 0.045510 |

The FP8 measurement was made with the IU4 prefill route disabled; changing `HIPFIRE_A4_GROUP` does not affect its row-wide packer. It remains inside the hard 0.05 budget. IU4 K512 remains below the existing 0.10 ceiling, but it regresses the shipped baseline by 0.007997 and therefore creates no budget for a coarser weight-scale grid. K256 improves c2 by 0.001132 yet regresses c24 by 0.004274, so it also fails the requested quality bar.

Artifacts: `iu4-k128-{c2,c24}.bin`, `iu4-mix-k512-{c2,c24}.bin`, `iu4-k256-{c2,c24}.bin`, and `fp8v2-{c2,c24}.bin`.

## Route proof and daemon configuration

The daemon installs a `ProcessConfig`; after that point an ambient long-tail environment variable is not authoritative. The valid daemon runs therefore used the persistent process key `developer.a4_group` with string value `512` or `256`, which maps to `HIPFIRE_A4_GROUP`, and reset the key after the final battery.

The first `prof-k512/` and `prof-k256/` attempts used only the ambient variable. Their symbols were baseline, so those attempts are invalid and excluded from every number in this report.

Corrected rocprof traces:

- `prof-k128/bench_kernel_stats.csv`: 736 + 256 `*_iu4_*_symfold` GEMMs; baseline fused producer symbols.
- `prof-k512-config/bench_kernel_stats.csv`: the same 736 + 256 IU4 GEMMs, no FP8 GEMM/packer route, 256 calls to the unique `fused_rmsnorm_mq_rotate_awq_i4_gfx12_k512`, and baseline symbols for SiLU, gated norm, and sigmoid/rotate.
- `prof-k256-config/bench_kernel_stats.csv`: unique `*_k256` symbols for RMSNorm, SiLU, gated norm, and sigmoid/rotate.

The traces are exact pp8192 replays (`replay-iu4.jsonl`, `replay-k512.jsonl`, and `replay-k256.jsonl`). Decode is expected to retain the baseline symbols because the fused batched producer route is not used there.

A late independent route audit by RowGlobalCheap found that RowGlobalA's 0.037143-quality trace was an FP8 fallback from a stale binary, not grouped IU4. With a route-fixed current binary, that audit measured row-global c2 0.118151 and K512 c2 0.072867. The original quality premise is therefore not treated as proven. This experiment's direct quality results are distinct from the 0.037143 FP8 value, and its corrected daemon traces explicitly contain the IU4 symfold GEMMs and unique candidate producer symbols.

## Producer cost

Producer subtotal matches the campaign definition: fused SiLU, RMSNorm, gated-norm and sigmoid producers plus GDN prep, deinterleave, standalone RMSNorm, and rotate; total duration is divided by 16,384 profiled input tokens.

| Route | Producers (us/token) | Delta vs K128 |
|---|---:|---:|
| K128 | 39.4125 | — |
| mixed K512/K128 | 42.4808 | +7.785% |
| uniform K256 diagnostic | 39.1751 | -0.602% |

The K512 RMSNorm producer itself moves from 7.2323 to 10.3238 us/token (+42.745%), while the other producer rows are essentially unchanged. The aggregate is still far below the approximately 61.3 us/token row-global result and demonstrates that no hidden second pass was introduced.

## Decode-first and paired performance gates

All valid daemon rows used card-B UUID `GPU-e475645fe0200397`, the dedicated `ab1` home/cache, the built candidate daemon, graph mode, warmup, and explicit process configuration. No GPU index was used.

Decode-first gate:

| Route | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| K128 | 3324.5 | 36.521726 |
| mixed K512/K128 | 3304.9 | 36.520409 |
| Candidate delta | -0.590% | -0.0036% |

Paired matrix, control/candidate order reversed in pair 2:

| Pair | Route | pp512 | pp8192 | tg128 |
|---|---|---:|---:|---:|
| 1 | K128 | 3366.5 | 3593.9 | 36.450102 |
| 1 | K512 | 3333.5 | 3560.3 | 36.484229 |
| 2 | K512 | 3334.7 | 3554.0 | 36.447676 |
| 2 | K128 | 3346.6 | 3568.2 | 36.449333 |
| mean | K128 | 3356.55 | 3581.05 | 36.449717 |
| mean | K512 | 3334.10 | 3557.15 | 36.465953 |
| delta | candidate / control | **-0.669%** | **-0.667%** | **+0.045%** |

Artifacts: `tg-k128.json`, `tg-k512.json`, and `matrix-{k128,k512}-pair{1,2}.json`.

## TTFT

5,909-token prompt, eight measured runs after two warmups:

| Route | Median TTFT (ms) | Prompt throughput (tok/s) |
|---|---:|---:|
| K128 | 1680.7914 | 3515.6058 |
| mixed K512/K128 | 1685.6167 | 3505.5420 |
| Candidate delta | **+0.287%** | **-0.286%** |

Artifacts: `ttft-k128.json`, `ttft-k512.json`.

## Serve battery

Candidate route, thinking disabled: 5/5 `finish=stop`, zero runaway, empty, attractor, or retrieval-miss results. All decoded text was inspected: the merge is valid, the distance is 210 miles, the factual answer is exactly three sentences, the story is exactly four sentences, and the instruction answer is exactly five numbered lines.

The harness summary block below is pasted verbatim from `battery-full.txt`:

````text
### RUN qwen3.8:27b-mq4-xt|off|battery  kv=auto sampling={'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0} seed=None ###
  [code]t1  finish=stop   ctx=84     cached=0      gen=305  (think 103/ans 46w) prefill=67.8ms/1239.6tok/s decode=36.4tok/s tau=None | '```python\ndef merge_sorted(a, b):\n    """Merge two sorted lists into one sorted list."""\n '
  [reason]t2  finish=stop   ctx=95     cached=0      gen=144  (think 12/ans 79w) prefill=62.5ms/1519.0tok/s decode=36.4tok/s tau=None | 'Step 1: Find the distance for the first part of the trip.\n\nDistance = Speed × Time  \nFirst'
  [factual]t3  finish=stop   ctx=65     cached=0      gen=199  (think 103/ans 47w) prefill=58.0ms/1120.9tok/s decode=36.5tok/s tau=None | "The seasons on Earth are caused primarily by the tilt of Earth's axis relative to its orbi"
  [prose]t4  finish=stop   ctx=73     cached=0      gen=308  (think 169/ans 58w) prefill=57.9ms/1261.3tok/s decode=36.4tok/s tau=None | 'Old Maren found a glass bottle still sealed after the storm, its label smeared by salt. Wh'
  [instruct]t5  finish=stop   ctx=71     cached=0      gen=123  (think 43/ans 46w) prefill=58.0ms/1223.4tok/s decode=36.5tok/s tau=None | '1. Use clear and consistent naming for variables, functions, and modules.\n2. Write small f'
[qwen3.8:27b-mq4-xt|off|battery DONE] turns=5 runaway=0 empty=0 attractor=0 retrieval_miss=0 avg_prefill=1272.8tok/s avg_decode=36.4tok/s
  prompt_md5=43ca0d15712d3dfb777b51ae76d8fd5f request_md5=b45624e909a1eb3f40f63272c1f41355 step=
  prompt_md5=640e0fd4f55996cb175a422f0a12cef5 request_md5=98908f629ef79dce205ca9aea163085b step=
  prompt_md5=8f66b4c97988825bd8e7840aaf44357e request_md5=9502b5d5a6579d96c0a00e350be5375f step=
  prompt_md5=8fe0ad36f61bcf4992cc9df81cdf3817 request_md5=cfbd00094fb95f078fad91980cc566bf step=
  prompt_md5=8bed8e2d056dc1d47dccae9d32dbecf4 request_md5=ae84c5a3746b4b9a8247c6601ff2687c step=
  daemon_binary_md5=af78862f7895bc27442072e921dcf88f path=/home/kaden/ClaudeCode/warpfront/wt-a4k512/target/release/daemon
````

Full decoded answers from `battery.json`:

````text
[code]
```python
def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0

    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1

    result.extend(a[i:])
    result.extend(b[j:])

    return result
```

[reason]
Step 1: Find the distance for the first part of the trip.

Distance = Speed × Time  
First distance = 60 mph × 2.5 hours  
First distance = 150 miles

Step 2: Find the distance for the second part of the trip.

Second distance = 40 mph × 1.5 hours  
Second distance = 60 miles

Step 3: Add the two distances together.

Total distance = 150 miles + 60 miles  
Total distance = 210 miles

Final answer: **210 miles**

[factual]
The seasons on Earth are caused primarily by the tilt of Earth's axis relative to its orbital plane. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times of the year. This variation in sunlight angle and day length produces the changing seasons.

[prose]
Old Maren found a glass bottle still sealed after the storm, its label smeared by salt. When she held it to the lighthouse lamp, a folded note appeared inside. The handwriting was her own, dated thirty years before she had ever climbed the tower. She opened it, and the first line read: Do not turn on the light.

[instruct]
1. Use clear and consistent naming for variables, functions, and modules.
2. Write small functions that each perform a single responsibility.
3. Add meaningful comments to explain intent and non-obvious decisions.
4. Handle errors explicitly and avoid silent failures.
5. Keep dependencies minimal and well documented.
````

## Gate accounting

| Gate | Required | Result |
|---|---|---|
| IU4 c24 | Material improvement while remaining ≤ 0.10 | **Fail:** 0.089196, worse than 0.081199 |
| FP8v2 c24 | ≤ 0.05 | Pass: 0.045510 |
| tg128 | No worse than -1.5% | Pass: +0.045% paired (-0.0036% decode-first) |
| pp8192 | No worse than -1% | Pass: -0.667% |
| TTFT | No worse than +2% | Pass: +0.287% |
| Battery | 5/5 stop, sane decoded text | Pass |
| Single pass | No activation-plane second pass | Pass |

**Final decision: KILL the K512 activation-scale package.** Preserve the implementation commit only as an experimental reference; do not ship the flag or use these results to justify the power-of-two weight-scale cutover.
