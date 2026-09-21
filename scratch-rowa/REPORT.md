# gfx1201 row-global A4 activation-scale gate

## Verdict

**The quality gain is real, but the current two-pass implementation is a performance KILL and must not ship.** Keep `HIPFIRE_A4_ROWGLOBAL` default-off. The implementation proves that one RTN scale per token is substantially better than per-K128 for rotated MQ4V2 activations, but producer geometry forces the guarded path through an F32 write/re-read plus standalone quantizer. Paired daemon gates regress pp512 by 14.6%, pp8192 by 9.4%, and 5909-token TTFT by 10.7%. Decode is unchanged.

The shipped artifact's row-global IU4 quality reaches c24 0.045510, below the IU4 0.10 budget and equal to its FP8v2 result. The pow2h artifact reaches IU4 c24 0.055408, below the IU4 budget; its FP8v2 c24 is also 0.055408 and therefore still fails the independent FP8v2 hard budget of 0.05.

## Artifact and route matrix

All values are KLD. `K512` and `row` were measured independently for IU4. FP8v2 already uses one row-wide activation scale, so the A4 granularity flag does not alter that route; its measured result is repeated across columns to make the requested ladder explicit.

| Artifact | Route | K128 c2 | K128 c24 | K512 c2 | K512 c24 | row c2 | row c24 |
|---|---|---:|---:|---:|---:|---:|---:|
| shipped `56e4a67c…` | IU4 | 0.064864 | 0.081199 | 0.037143 | 0.045510 | 0.037143 | 0.045510 |
| shipped `56e4a67c…` | FP8v2 | 0.037143 | 0.045510 | 0.037143 | 0.045510 | 0.037143 | 0.045510 |
| pow2h `febf7057…` | IU4 | 0.071897 | 0.091338 | 0.046252 | 0.055408 | 0.046252 | 0.055408 |
| pow2h `febf7057…` | FP8v2 | 0.046252 | 0.055408 | 0.046252 | 0.055408 | 0.046252 | 0.055408 |

The exact prior screen was reproduced on pow2h: IU4 row c2/c24 = 0.046252/0.055408. It was not an activation-precision bypass: the corrected production path profiles `quantize_int4_mmq_grouped` followed by the IU4 symfold GEMMs, and the quantized layout retains signed int4 nibbles plus one repeated `d=amax/7` header per K128 block. K512 equaling row on both artifacts is plausible because FWHT-rotated rows are flat enough that the remaining group distinction is not quality-visible at this evaluation resolution.

## Activation difference

A deterministic GPU oracle compared dequantized `block_i4_128` output for K128 and row-global RTN over N=512, K=5120:

- relative RMS: **0.0948547531**
- compared values: 2,621,440
- non-finite values: 0

This confirms that the quality equality with FP8v2 is not caused by byte-identical activations or a skipped quantizer.

## Implementation

Commits:

- `2977acb21` — guarded grouped int4 quantizer and coarse-scale routing
- `1b0e6a1bc` — keep coarse activation requests on IU4 instead of falling through to FP8

`HIPFIRE_A4_ROWGLOBAL=1` selects `group_k=K`; default remains K128. `HIPFIRE_A4_GROUP_K=512` exists only for the measured ladder. The grouped kernel launches one 256-thread workgroup per `(token, K-window)`, reduces one amax, uses RTN with `d=amax/7`, clamps to signed `[-8,7]`, and repeats the scale in each K128 header expected by the consumer.

Producer geometry explains the cost:

- RMSNorm/FWHT already assigns one workgroup to a full row and loops over K; a row reduction can remain single-workgroup.
- SwiGLU/FWHT uses one workgroup per 256-K slice and therefore splits a row across workgroups.
- rotate/sigmoid producers use one workgroup per `(token, 256-K slice)` and split a row.
- gated-norm/FWHT uses one workgroup per 256-K slice/two heads and splits a row.

There is no legal grid-wide barrier inside these ordinary launches. The guarded implementation therefore disables K128-emitting IU4 sidecars, emits the existing F32 rotated row, then launches the grouped quantizer. A shippable implementation would need redesigned one-workgroup-per-row producers or a deliberate reduction/quantization multi-pass; simply changing the current sidecar reduction is incorrect.

## Performance gates

Card-B, shipped artifact, graph on, FP8 KV. Each matrix row is the mean of two independently paired run medians; each invocation used 3 measured runs after warmup.

| Metric | K128 baseline | row-global | Delta |
|---|---:|---:|---:|
| pp512 | 3358.10 tok/s | 2867.05 tok/s | **-14.62%** |
| pp8192 | 3582.50 tok/s | 3246.70 tok/s | **-9.37%** |
| tg128@128 | 36.4618 tok/s | 36.4381 tok/s | -0.06% |
| TTFT, 5909 prompt tokens | 1670.238 ms | 1848.491 ms | **+10.67%** |
| TTFT prompt throughput | 3537.819 tok/s | 3196.662 tok/s | **-9.64%** |

The N=512 ungraphed device-event profile proves the intended route and attributes the producer cost. Row-global activation producers plus grouped quantization total approximately **61.3 us/token**, versus the stated **39.5 us/token** baseline: **+21.8 us/token (+55.2%)**. `quantize_int4_mmq_grouped` alone accounts for 10.5 ms over 512 tokens (20.5 us/token); loss of the in-register fused producer sidecars accounts for the remainder.

## Battery

Final captured run: 5/5 `finish=stop`; runaway=0, empty=0, attractor=0, retrieval_miss=0; average prefill 1182.2 tok/s and decode 36.3 tok/s.

Decoded visible text, verbatim:

### code

```python
def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    merged = []
    i = j = 0

    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1

    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged
```

### reason

Step 1: First part of the trip  
\[
60 \text{ mph} \times 2.5 \text{ hours} = 150 \text{ miles}
\]

Step 2: Second part of the trip  
\[
40 \text{ mph} \times 1.5 \text{ hours} = 60 \text{ miles}
\]

Step 3: Total distance  
\[
150 + 60 = 210 \text{ miles}
\]

**Final answer: 210 miles**

### factual

The seasons on Earth are caused by the tilt of Earth's axis, which is inclined at about 23.5 degrees relative to its orbital plane. As Earth orbits the Sun, this tilt causes each hemisphere to receive more direct sunlight during part of the year and less direct sunlight during another part. The hemisphere tilted toward the Sun experiences summer, while the hemisphere tilted away experiences winter.

### prose

Old Man Callum climbed the wet rocks at dawn, expecting only kelp and broken shells. Instead, his lantern caught the polished curve of a brass key no bigger than his thumb. Its teeth were shaped like the spiral of his lighthouse beam, and its loop bore the faded name of his vanished brother. For the first time in ten years, Callum did not light the lamp; he held the key to his lips and waited for the sea to answer.

### instruct

1. Use clear, descriptive names for variables, functions, and modules so intent is obvious at a glance.
2. Keep each function short and focused on a single responsibility to minimize cognitive load.
3. Write automated tests that document expected behavior and guard against regressions.
4. Remove dead code and add comments only where the "why" is non-obvious, never the "what."
5. Enforce consistent formatting and style with a linter or formatter to eliminate unnecessary noise.

## Verification evidence

- Release build passed for `hipfire-runtime` profile harness, `hipfire-daemon`, and `hipfire-cli`.
- Corrected N=512 profile observed 256 `quantize_int4_mmq_grouped` launches and 496 IU4 symfold GEMM launches; no FP8 projection fallback.
- The exact screen, full KLD ladder, paired matrix, TTFT gate, activation rel-RMS, and final five-case battery were exercised on the assigned gfx1201 cards by UUID.
