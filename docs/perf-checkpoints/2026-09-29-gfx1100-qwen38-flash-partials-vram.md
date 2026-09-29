# gfx1100 Qwen3.8 flash-partials VRAM A/B

**Lifecycle:** historical

**Date:** 2026-09-29

**Disposition:** supports the short-context buffer-sizing change only; this is
not a general performance baseline or an admission decision.

## Claim

For the Qwen3.8-27B gfx1100 Q8-VMM route at `max_seq=8192`, sizing the shared
flash-attention partials buffer from the independent single-token decode and
batched-prefill tile geometries reduces that allocation from 101,449,728 to
50,724,864 bytes (a reduction of 50,724,864 bytes, or 48.4 MiB) without a
measurable prefill-throughput loss.

The byte count is deterministic allocation accounting, not an inference from
whole-process `rocm-smi`. Whole-process readings varied by more than this
allocation delta across fresh loads and are therefore not used as the memory
claim.

## Fixture

- GPU: AMD Radeon Pro W7900, exact `gfx1100`, HIP 7.15
- model: `qwen3.8-27b.mq4-xt`
- model SHA-256: `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`
- model MD5: `e45d15bfe0c9a87132697101d17cbed6`
- baseline: `beta@1621a00900f46dc1f102945e0046387193fdc2a1`
- candidate: `3aeb3c92f822b4f34fa892b55dbbe5e5be5d6a90`
- baseline `hipfire` MD5: `6cbff69d559b878bb1e7baf46739c0fc`
- baseline daemon MD5: `62b8dcbcc8136919fc4cc3e79fc8c639`
- candidate `hipfire` MD5: `0798942d249966e341bdfd544c297303`
- candidate daemon MD5: `8fa22c8b1e2415e11097027c8437366c`
- prompt: `benchmarks/prompts/ttft_5900.txt`, 5,909 tokens
- prompt MD5: `ed720348b81a19fab64d4783c75c1ae3`
- route: Q8 KV, VMM allocator, AR (`--spec off`), `max_seq=8192`
- method: fresh process per sample, one measured run after ten discarded
  warmups, three samples per arm, `HIPFIRE_VERIFY_GRAPH=0`

Both arms used their own pinned `HIPFIRE_DAEMON_BIN`. The candidate was also
prewarmed once in an unrecorded process before collecting its three samples,
matching the already-warm kernel-cache state of the baseline arm.

## Results

| Metric | clean beta samples | candidate samples | Median delta |
|---|---:|---:|---:|
| Prefill tok/s | 1684.6, 1694.6, 1682.0 | 1685.0, 1695.1, 1684.5 | 1684.6 -> 1685.0 (+0.02%) |
| Decode tok/s | 37.2, 37.0, 36.9 | 37.1, 36.8, 36.6 | 37.0 -> 36.8 (-0.54%) |
| Wall tok/s | 18.4, 18.4, 18.3 | 18.4, 18.4, 18.3 | 18.4 -> 18.4 |
| TTFT ms | 3507.6, 3486.9, 3513.1 | 3506.8, 3485.9, 3507.9 | 3507.6 -> 3506.8 |

The decode difference is below the project's +/-2% speed-gate tolerance and
this allocation change is outside the decode hot path.

## Allocation derivation

For 24 heads, head dimension 256, and 8,192 physical KV rows:

```text
legacy = 16 * 24 * ceil(8192 / 32)  * (2 + 256) * 4
       = 101,449,728 bytes

candidate = max(
    1  * 24 * ceil(8192 / 32)  * (2 + 256),
    32 * 24 * ceil(8192 / 128) * (2 + 256)
) * 4
          = 50,724,864 bytes
```

An attempted 16-row batched capacity was rejected because it materially
regressed prefill. The 32-row default is restricted to exact gfx1100 when the
Q8 decode tile is smaller than the batched-attention tile. Explicit
`HIPFIRE_FLASH_PARTIALS_BATCH` values remain exact; equal-tile long-context
gfx1100 and other architectures retain the existing 16-row default.

## Correctness

The final candidate passed the user-facing `serve_harness.py` battery on the
same gfx1100 with Q8 VMM and `max_seq=8192`: 5/5 turns completed at `stop`,
with zero empty outputs, runaways, token attractors, or retrieval misses. The
five decoded responses were manually inspected and were coherent. The serve
log confirmed effective `kv_backend=vmm`.
