# V2.5 pow2-scale KLD screen (MQ4V2 symmetric, per-128 `d` = power of two)

Branch `mq4v2-pow2-scale`, worktree `wt-pow2`. Impl commit `8a761e75b`.

## Flag diff

- `cli.rs`: new `--mq4v2-pow2-scale` (`requires = "mq4v2_symmetric"`; clap
  rejects it without `--mq4v2-symmetric`, verified via `--help` probe).
- `quant_fwht.rs`: new `quantize_mq4g256v2_symmetric_pow2` wrapper; impl gains
  `pow2_scale: bool`. When on, the symmetric branch replaces the 4-multiplier
  ladder over `(amax/7.5)*0.5` with pow2 candidates
  `{2^floor(log2(amax/7.5)), 2^ceil(log2(amax/7.5))}` built as exact f16 powers
  of two (`(biased_exp << 10)`, zero mantissa; biased exp outside 1..=30
  skipped, clamped to nearest normal pow2 if both fall out). Same MSE
  selection over the 128-group, zero = `f16(-8d)`. Final code assignment loop
  shared. Asym (`false,false`) and plain symmetric (`true,false`) call sites
  unchanged → byte-identical.
- `pipeline.rs`: new `MQ4V2_POW2_SCALE` `OnceLock` (kept as `OnceLock`: value
  comes from runtime CLI args), dispatch in `quantize_mq4g256v2_selected`,
  `--format mq4v2` + `--mq4v2-symmetric` validation, stderr banner, and
  metadata `mq4v2.pow2scale = 1` alongside `mq4v2.symmetric`.

## Artifact

Recipe: `--input /home/kaden/qcal/parents/qwen3.8-27b --format mq4v2 --tier xt
--imatrix .../Qwen3.8-27B-imatrix.gguf --awq-alpha 0.35 --mq4v2-symmetric
--mq4v2-pow2-scale`.

- Path: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2-a035.hfq`
- Size: 14987185152 bytes (14987.2 MB)
- sha256: `e25d2ba8f27f5b6b2e1c0e46db72678b8f95a2a86e44f47a1868301f3f53c230`
- Census: F16 801 / MQ4G256V2 497 / Q8F16 49 (same tensor split as shipped sym).
- Metadata: `mq4v2.symmetric = 1`, `mq4v2.pow2scale = 1` (parsed from HFQ header).

## Mantissa check (`scratch-pow2/scan_mantissa.py`)

HFQ container parsed directly (magic, metadata JSON via brace scan, tensor
index, qt44 blobs at 136 B/group, scales at offsets 0 and 4):

- qt44 tensors 497, total groups 100085760, sampled 3469316 scales strided
  (<=4096 groups/tensor), 0 zero-scales skipped as degenerate.
- Non-zero f16 mantissa: **0**. `zero != -8*d`: **0**. → PASS.

## KLD c24 (prefill scoring, kv fp8/q8, ref `qwen3.8-27b.ref_wt2.bin`)

| route | pow2 artifact | shipped sym baseline | delta |
|---|---|---|---|
| iu4 (default) | **0.105319** (NLL 1.922086, PPL 6.8352) | 0.0860 | +0.0193 |
| fp8v2 (`HIPFIRE_IU4_PREFILL=0`) | **0.070638** (NLL 1.897186, PPL 6.6671) | 0.0491 | +0.0215 |

Sanity pins (c2, fp8v2): pow2 0.058107 vs shipped 0.040442 (delta +0.0177,
same direction/magnitude as c24 → harness stable, cost is real).

Raw score files: `scratch-pow2/iu4-c24.bin`, `scratch-pow2/fp8v2-c24.bin`,
`scratch-pow2/fp8v2-c2.bin`, `scratch-pow2/fp8v2-c2-shipped.bin`.

## Reading

Restricting `d` to pow2 costs ~+0.02 KLD on both routes at c24 with zero
kernel changes (fold-free hypothesis still untested — needs the GEMM-side
fold measurement from the receipt). Whether +0.02 is acceptable is a
main-agent call; the artifact and both numbers are above.
