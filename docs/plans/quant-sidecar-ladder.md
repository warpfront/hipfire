# Quant sidecar ladder: one pinned artifact, every lower tier from a sidecar

**Status:** shelved idea (2026-10-04). Not scheduled; revisit after the Flash-Next quality work.
**Owner:** Kaden Schutt.

## Problem

Every quant of a model ships as its own multi-GB artifact (`mq4`, `mq4-xt`, `mq4-xts`, `mq3`, ...). A
user who wants to try another quality point downloads the model again, and the registry keeps N copies
of almost the same weights.

## Idea

Ship **one pinned artifact per bit-width ceiling** (for example `mq4` or `mq6`). Every lower tier
(`mq6 → mq5 → mq4 → mq3 → mq2`) is produced **at load time** from a small **tier sidecar**. The user
downloads one model and picks its quality inside the engine (`hipfire config`), not by downloading
another artifact.

## Requirements

1. **The sidecar carries a BF16-calibrated solve, not a requant of the parent's codes.** A tier's codes
   and scales are what a GPTQ/AWQ solve against BF16 calibration would produce for that tier. Rounding
   the parent's codes onto a coarser grid (quant of a quant) is explicitly out: it is lossy in a way
   the offline solve is not.
2. **Load is decode only.** Applying a sidecar solves nothing. Output is deterministic, bit-exact to
   the standalone artifact that solve would have produced, and independent of GPU float behaviour.
3. **Hash-bound.** A sidecar names its parent artifact by SHA-256 and refuses any other parent. The
   expanded tier has its own recorded hash, checked like a normal artifact.
4. **Compact.** The sidecar stores only the information the parent's codes don't already determine.

## Encoding

Given the pinned parent code for a weight, the BF16-solved lower-tier code is nearly determined: usually
the matching coarser bucket. What is left is rounding direction plus GPTQ error-feedback flips. The
sidecar stores:

- per-group scales (and zero points where the format has them) for the tier;
- the conditional residual `code_tier | code_parent`, entropy-coded per group, with an exception list
  for codes outside the predicted bucket.

### Size depends on how the ladder is solved

| Solve | Residual | Notes |
|---|---|---|
| Independent GPTQ per tier | ~0.5 bit/weight [estimate] | For a ~235B-weight model like Flash-Next, ~15–20 GB per tier. That beats a standalone mq3 (~95 GB) but is not small. |
| **Joint nested solve** from BF16 | scales + sparse exceptions [estimate] | Solve the parent and every lower tier together. Constrain lower codes to nest under the parent's codes, and run GPTQ error feedback per tier inside that constraint. Most lower codes become exact truncations. Prior art: Any-Precision LLM (2024), Matryoshka Quantization (2025). |

The size estimates are unmeasured; a quantizer run on a real model would give the actual residual entropy.

## Same-width recipe tiers collapse

`base` / `xt` / `xts` at the same bit width can't be derived from each other's codes, and at equal size
users want the best recipe. Keep a second same-width artifact only when it changes speed or route. For
example, symmetric experts enable the IU4 fast route on gfx1151 and asymmetric experts do not.

## Open questions

- **Load cost:** expanding ~125 GB on the GPU should take seconds to a minute, but every cold load pays it
  unless the expanded tier is cached, and caching gives the disk savings back.
- **Format:** container layout for tier sidecars (HFQ section vs separate file), and registry schema
  (`parent_sha256`, tier list, expanded hashes).
- **Floor:** MQ2 is currently refused as broken. The ladder ends at MQ3 until the Lloyd MQ2/MQ3 work
  lands.
- **Quantizer:** a joint nested GPTQ solve does not exist in `hipfire-quantize`. It is the main build
  cost.
