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

---

# Follow-up: pow2-half (`m * 2^e`, m in {1, 1.5}) + fold-honest MSE

Impl commit `f640ae2` (unpushed; same branch `mq4v2-pow2-scale`).

## Flag diff

- `cli.rs`: new `--mq4v2-pow2-scale-half` (`requires = "mq4v2_symmetric"`,
  `conflicts_with = "mq4v2_pow2_scale"`; both guards probed via `--help`
  invocations).
- `quant_fwht.rs`: new `e4m3_rne()` (RNE into the fp8 E4M3 grid, 3 mantissa
  bits) + `quantize_mq4g256v2_symmetric_pow2_half` wrapper; impl gains
  `pow2_half: bool`. BOTH pow2 modes now use fold-honest MSE selection AND
  final code choice: per candidate `d = m * 2^e`, all 16 codes are scored
  under `recon = e4m3((code-8) * m) * 2^e` (exhaustive per-element argmin;
  for m = 1 the fold is the identity since integers -8..7 sit exactly on the
  e4m3 grid). Half mode: 4 candidates (floor/ceil octave x {1, 1.5}) as exact
  f16 (`(biased << 10) | (m == 1.5 ? 0x200 : 0)`); zero = `f16(-8d)` (exact
  for both m). Ladder (asym + plain symmetric) paths untouched.
- `pipeline.rs`: `MQ4V2_POW2_HALF` static, dispatch, validation, banner, and
  metadata `mq4v2.pow2scale = 2`.

## Artifact

- Path: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2h-a035.hfq`
- Size: 14987185152 bytes (14987.2 MB, same as pow2/sym)
- sha256: `febf70570354ccf8867ae2c7330da0dd80b4263e3ce2603f724abd62eccaf7c8`
- Metadata: `mq4v2.symmetric = 1`, `mq4v2.pow2scale = 2`.
- Quantize wall time 864 s (vs 299 s for full pow2 — exhaustive 16-code
  search over up to 4 candidates).

## Mantissa check

- Same scanner, `half` mode (allowed mantissas 0x000/0x200): 497 qt44
  tensors, 100085760 groups, 3469316 scales sampled, 0 bad mantissa,
  0 bad zero. → PASS.

## KLD (same harness/flags as c24 screen)

| artifact | route | chunks | KLD | NLL | PPL |
|---|---|---|---|---|---|
| sym-pow2h | fp8v2 | 24 | **0.057072** | 1.883551 | 6.5768 |
| sym-pow2h | iu4 (pin) | 2 | **0.071897** | 2.269302 | 9.6726 |

Context: shipped sym fp8v2 c24 0.0491; full-pow2 fp8v2 c24 0.070638; full-pow2
fp8v2 c2 0.058107; shipped fp8v2 c2 0.040442.

## Reading

Half recovers most of the full-pow2 cost on the fold-capable route: +0.0080
over baseline vs +0.0215 for strict pow2. The remaining gap is the honest
folded error (1.5x products that round in e4m3, e.g. 10.5 -> 10), i.e. close
to the best a fold-free fp8 kernel can do with a 2-bit mantissa choice.
