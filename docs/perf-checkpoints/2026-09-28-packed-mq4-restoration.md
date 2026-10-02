# Packed MQ4/MQ4V2 restoration: experimental FFN path

Base: beta `da629b75d19afebaed91551325d489bc502db3ae`.
Source lineage: unmerged PR #616 (`473efc835`); existing attribution preserved.

Stage 1 restored the gfx1100 FFN gate/up packed path; stage 2 below extends it
to down. This is not the complete #616 stack. No end-to-end speedup is claimed. Enable explicitly with
`HIPFIRE_GFX1100_PACKED_MQ4_PREFILL=1`; the default remains off.

## Contract

- Exact gfx1100, uniform MQ4 or MQ4V2, gate/up shape 17408 x 5120,
  positive batch divisible by 256. Other shapes, Lloyd, mixed formats, and
  capture/recording fall back to existing routes.
- Checkpoint bytes are unchanged. MQ4 uses a float scale/zero per 256;
  MQ4V2 uses separate half scale/zero pairs per 128. The V2 kernel applies
  the zero correction separately to each half.
- Activation scratch is the existing 144-byte DS4 layout, not the separate
  136-byte V2 INT8 activation layout. Each GEMM requantizes its input.
- Gate/up output remains separate F32. Down and decode are unchanged.
- Each projection uses its own AWQ sidecar. Rotated inputs are shared only
  when both sidecars are absent; pointer identity is not used as scale equality.

## Numerical oracle

The standalone HIP oracle uses a CPU weight decoder, irregular nibbles,
mixed-sign activations, different per-token/per-half activation scales,
different V2 half headers, multiple output tiles, and both set/add epilogues.
Inputs deliberately have exactly representable Q8 values and half metadata:
this isolates layout/indexing/zero-correction errors from quantization error.
It is not a random-input error bound or an end-to-end quality gate.

```sh
hipcc --offload-arch=gfx1100 -O3 kernels/test_packed_mq4_oracle.hip -o /tmp/packed-oracle-v1
hipcc --offload-arch=gfx1100 -O3 -DPACKED_MQ4_V2=1 kernels/test_packed_mq4_oracle.hip -o /tmp/packed-oracle-v2
# Use a free GPU and scripts/gpu-lock.sh; set the matching ROCm library path
# if the system loader does not expose libamdhip64.so.7.
HIP_VISIBLE_DEVICES=1 /tmp/packed-oracle-v1 5120
HIP_VISIBLE_DEVICES=1 /tmp/packed-oracle-v2 5120
HIP_VISIBLE_DEVICES=1 /tmp/packed-oracle-v1 17408
HIP_VISIBLE_DEVICES=1 /tmp/packed-oracle-v2 17408
```

W7900/gfx1100: all eight set/add comparisons passed, each checking 65,536
outputs against CPU reference, maximum absolute error 0.

Rust admission tests: one rdna-compute test and one qwen35 routing test passed.
Release daemon/CLI with `flash-attn-ck` built successfully; existing workspace
warnings remain. Full serving, retained-replay validation and matched timing
are still required before promotion.

## Initial serving smoke (not performance evidence)

Prompt: `benchmarks/prompts/adaptive_kv_long_prefill.txt`, MD5
`7ef3f606d2826a08e161bbeb4848b11e`. Greedy AR, Q8 KV, max_seq=4096.

- Qwen3.6-27B MQ4: 936 input / 256 generated tokens, coherent review text,
  no empty output or attractor; stopped at the configured length cap, not EOS.
  A cold single run is not a throughput baseline.
- Local Qwen3.8-27B MQ4XT was confirmed from its tensor index to contain
  qt44 projections. Its first run used only `--thinking off`; the native
  Qwen reasoning contract ignored that legacy setting and exhausted the
  output cap inside an open think span. This is **not a passing smoke**.
  Reruns must explicitly pass `--max-think-tokens 1`, and require the
  `[packed-mq4] admitted format=mq4v2` marker to rule out fallback-only tests.
- Explicit-cap MQ4V2 rerun: the marker confirmed 17408 x 5120 x 512.
  936 input / 1024 generated, no reasoning, no stream error, empty output,
  or attractor. Text remained readable but reached the length cap; this is
  only a fluency smoke, not factual accuracy or a quality-neutrality claim.
  In particular, the generated discussion conflates Bloom-filter false
  positives with invalidation behavior and is not a correctness oracle.
- Explicit-cap MQ4 rerun: the marker confirmed 17408 x 5120 x 512;
  936 input / 876 generated, natural stop, no thinking, empty output,
  attractor, or stream error. The model completed the requested review.

Local artifact SHA256 (the MQ4XT is an older local artifact, **not** the current
canonical acceptance pin):

```
86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc  qwen3.6-27b.mq4
9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7  qwen3.8-27b.mq4-xt
2c2c96ffb3e8db54bcbe9f045046f0008694930e6db48789a3a6ae6892eed87a  daemon
04140655654393a1eb2becdbdd28d661f31db4134d6a14fd6a4c184b931a5843  hipfire
```

After acquiring the GPU lock, run the following per model (use distinct home,
logs, and output names). This is smoke only, not the timing protocol:

```sh
HIP_VISIBLE_DEVICES=1 HIPFIRE_GFX1100_PACKED_MQ4_PREFILL=1 \
HIPFIRE_PREFILL_CHUNK_ROWS=512 python3 scripts/serve_harness.py \
  --model /absolute/path/to/model \
  --speculation off --thinking off --max-think-tokens 1 --sampling greedy \
  --kv q8 --max-seq 4096 --max-tokens 1024 --mode battery \
  --prompt-file benchmarks/prompts/adaptive_kv_long_prefill.txt \
  --home /tmp/packed-smoke-isolated --port 11793 \
  --serve-warm-timeout-secs 600 --serve-log /tmp/packed-smoke-serve.log \
  --out /tmp/packed-smoke.json
```

Raw retained smoke artifacts: `benchmarks/results/20260928-packed-mq4-stage1/`.

## Stage 2: FFN down integration

The opt-in now also handles down (5120 x 17408) for uniform MQ4/MQ4V2.
It requires separate F32 gate/up and a residual epilogue, flushes any deferred
residual before adding, and uses the down weight's own AWQ sidecar. Prepared
H planes and tensor-parallel partial output remain on their existing routes.
QKV, attention output, decode and checkpoint formats remain unchanged.

Two qwen35 route tests pass (gate/up and down). The standalone oracle above
already checks the exact same K17408 add kernel for both formats; it does not
substitute for the integrated serving test.

Reproducible integration smoke:

```sh
bash scripts/test-packed-mq4-smoke.sh benchmarks/results/NEW-RUN 1
```

The runner refuses existing output directories, runs both formats sequentially,
writes original serve/JSON/harness outputs directly, and hashes them afterward.
Model, binary and source identities are recorded. Logs are not hand-edited.
Runtime validation results will be recorded separately from these raw files.

Stage 2 run: `benchmarks/results/20260928-packed-mq4-stage2/`.
Both formats logged gate/up set and down add with the expected shapes.
MQ4 generated 942 tokens and stopped naturally; MQ4V2 generated 1024 and
hit the output cap. Both produced readable review text with no stream error,
empty output, or detected attractor, and no reasoning tokens. Text differs
from stage 1: these tests do not establish bit-exact output or quality parity.
All six raw JSON/serve/harness files passed their post-run SHA256 checks.
The cap-limited MQ4V2 result remains incomplete, not a completed-task pass.
No timing claim: these are fresh-home, first-request smoke runs without the
matched warmup/baseline protocol. Required Redline and broader quality gates
are still outstanding.
