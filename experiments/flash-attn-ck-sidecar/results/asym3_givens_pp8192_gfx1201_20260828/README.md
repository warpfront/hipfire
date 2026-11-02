# Asym3-Givens D256 gfx1201 validation

Hardware and software:

- AMD Radeon AI PRO R9700, exact `gfx1201`
- ROCm 7.14
- Qwen3.6-27B MQ4, SHA-256 `86a5f80f...42dc`
- Asym3-Givens K and Q8 V cache
- PP8192, batch 1, no speculative decoding
- HipFire base: `upstream/master@aaf5e3211`

The sidecar was built as an exact gfx1201 artifact. The build script selected
the CK `gfx12` generator family and produced 13 FP16 D64/D128/D256 forward
sources. Reusing the gfx11 generator produced a linkable artifact but failed
closed at runtime with `CK found no matching forward kernel`; that configuration
is not supported.

Raw-ABI GPU smoke:

| Cell | Max abs | Mean abs |
| --- | ---: | ---: |
| F32/Asym3-Givens/Q8 GQA D256 causal | `6.110966e-05` | `1.009872e-05` |

Unsupported Givens D512 and FWHT D256/D512 cells remained recognized but
fail-closed. Q8 D256 is not published by the gfx1201 artifact and the raw ABI
rejects that cell consistently with the Rust selector.

Production Qwen prefill command:

```bash
BUILD=0 GPU_ID=0 KV_MODE=asym3 PREFILL=8192 RUNS=3 SLEEP_SECS=10 \
MODEL=$HOME/.hipfire/models/qwen3.6-27b.mq4 \
SIDECAR=$PWD/experiments/flash-attn-ck-sidecar/build-gfx1201/libhipfire_flash_attn_ck.so \
./scripts/bench_ck_q8_prefill_ab.sh
```

Results:

| Arm | Samples (tok/s) | Median |
| --- | --- | ---: |
| Native | `265.7, 611.1, 604.7` | `604.7` |
| CK | `875.5, 867.5, 860.1` | `867.5` |

The first native sample contains one-time kernel compilation; retaining it does
not change the three-run median. CK improves the median by `43.46%`. Both arms
produce next token `248046`. Decode smoke remains neutral (`23.5` versus `23.7`
tok/s for the one-token diagnostic).
