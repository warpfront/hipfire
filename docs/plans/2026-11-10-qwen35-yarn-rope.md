# Qwen3.5/3.8 YaRN RoPE Scaling

## Summary

Wire the existing `attention.yarn_factor` / `HIPFIRE_YARN_FACTOR` operator
override into the Qwen3.5/3.8 dense family (arch 5/6), enabling context
extension beyond the checkpoint's native `max_position_embeddings` via
YaRN ("NTK-by-parts") RoPE frequency scaling.

Reference user case (llama.cpp flags this must express):
`--rope-scaling yarn --rope-scale 1.282 --yarn-orig-ctx 262144 --ctx-size 335872`

## What already existed (reused, not rebuilt)

- **Resolution engine:** `crates/hipfire-runtime/src/rope.rs` —
  `rope_scaling(text_config)` reads `rope_parameters`/`rope_scaling` from
  the checkpoint, applies the operator override, returns
  `RopeScaling::{None, Yarn(Yarn)}`.
- **Kernel-side machinery:** `crates/rdna-compute/src/rope.rs` —
  `Rope { theta, scaling }`, `RopeYarn { freq_scale, corr_low, corr_high,
  mscale }` matching `kernels/src/rope_yarn.h`. `RopeYarn::OFF` is the
  identity (freq_scale 1.0, corr 0, mscale 1.0).
- **Working reference:** `crates/hipfire-arch-qwen4/src/config.rs` is the
  only prior production qwen-family caller.

## What this PR adds

### Kernel changes

All qwen35 RoPE kernels gain a trailing `RopeYarn yarn` by-value parameter
(16 bytes). The `cosf`/`sinf` pair is replaced with `rope_yarn_sincos(angle,
pair, yarn, &sin, &cos)` from `kernels/src/rope_yarn.h` (already in-tree,
used by qwen4 and tensor_ops kernels).

Kernels modified:

| File | Kernel(s) |
|------|-----------|
| `rope_partial_halfsplit.hip` | `rope_partial_halfsplit_f32` |
| `rope_partial_halfsplit_headgrid.hip` | `rope_partial_halfsplit_f32_headgrid` |
| `rope_partial_interleaved.hip` | `rope_partial_interleaved_f32` |
| `rope_partial_halfsplit_batched.hip` | `rope_partial_halfsplit_batched_f32` |
| `rope_partial_interleaved_batched.hip` | `rope_partial_interleaved_batched_f32` |
| `qwen35_fa_prep.gfx1100.hip` | `qwen35_fa_prep_gfx{1100,1151,1201}` |
| `qwen35_fa_kvwrite.gfx1100.hip` | `qwen35_fa_prep_kvwrite_gfx1100` |
| `qwen35_fa_prep_batched.gfx1100.hip` | batched gfx1100+gfx1151 |
| `qwen35_fa_prep_batched.gfx1201.hip` | batched gfx1201 (3 variants) |

The `rope_yarn.h` header is prepended to each kernel source via `concat!`
+ `include_str!` in the Rust constants (same pattern as `tensor_ops.hip`).

### Rust launcher changes

Six launch wrappers in `rdna-compute` gain a `yarn: &RopeYarn` parameter:
`rope_partial_interleaved_f32`, `qwen35_fa_prep_gfx1100`,
`qwen35_fa_prep_kvwrite_gfx1100`,
`rope_partial_halfsplit_capture_f32_batched`,
`qwen35_fa_prep_batched_gfx11{00,51}`, `qwen35_fa_prep_batched_gfx1201`.
Each pushes `yarn.push(&mut blob)` after the existing parameters.

### Config + dims plumbing

- `Qwen35Config` gains `rope_scaling: RopeScaling` (populated from
  `hipfire_runtime::rope::rope_scaling(&text_config)` at parse time) and
  `max_position_embeddings: usize` (from the checkpoint JSON, default 262144).
- `HybridDims` gains `rope_scaling: RopeScaling` and a `rope_yarn()`
  helper that lowers `Rope { theta, scaling }.yarn(n_rot)`.
- `program.rs::hybrid_dims()` passes the scaling through.

### All call sites updated

Every caller of the modified launchers passes either the model's actual
yarn (from `dims.rope_yarn()` / `config::rope_yarn(config)`) or
`RopeYarn::OFF` (for families without YaRN: maple, minimax, llama/dspark,
mrope paths, MTP head, sandwich prefix cache).

### Admission

`load_qwen35_pp` (the PP carrier) rejects `max_seq > context_limit()` with
the same style of message as the qwen4 admission. The single-GPU path sizes
KV to `max_seq` directly; the RoPE handles extended positions correctly.

### v1 boundary: speculative decode refused

When `rope_scaling` is `Yarn(_)`, DFlash and MTP are refused at load:
the draft's position arithmetic diverges from the target's under rescaled
RoPE. `dflash_mode=on` fails the load; `auto` logs and serves AR.

The refusal lives on **both** qwen35 load paths: the serve_engine path
and the single-GPU carrier path (`hipfire-loader::finish_qwen35_load`,
what `hipfire run` uses). The carrier guard was added after a measured
miss: the serve-only refusal left the carrier arming DFlash under
`HIPFIRE_YARN_FACTOR=2.0` on qwen3.8-27b.mq4.

## Safety case: default path bit-exact

`RopeYarn::OFF` (`freq_scale=1.0, corr_low=0, corr_high=0, mscale=1.0`)
makes both `if` guards in `rope_yarn_sincos` false:

```c
if (yarn.freq_scale != 1.0f) { ... }  // skipped
*sine = sinf(angle);
*cosine = cosf(angle);
if (yarn.mscale != 1.0f) { ... }      // skipped
```

This is the same computation as the previous direct `sinf`/`cosf` calls.
The FMA contraction in the subsequent rotation code is unchanged. Existing
bit-exact quality pins (AGENTS.md §5) must stay untouched; if any pin
moves, that is a bug in the None-path handling, not a re-pin.

## YaRN is NOT bit-exact (by design)

YaRN rescales low-frequency bands at ALL positions, not just beyond the
native context. Long-context quality must be evaluated (KLD vs BF16
teacher, needle-in-a-haystack), not assumed.

## Validation plan

Per `docs/VALIDATION.md` for a kernel+dispatch+generation change:

1. `scripts/redline_daemon_harness.py` — stable capture, AQL contract,
   HIP/PM4 parity (keep JSON report).
2. `scripts/serve_harness.py` battery + chain on the exact model/settings.
3. Eyeball decoded output (statistical gates alone are not acceptance).
4. Quality pins unchanged (None-path bit-exact).
5. Long-context KLD vs teacher past native context (YaRN path).

## Out of scope

- llama.cpp→hipfire profile translation (user handles via llama-swap).
- New config keys (`attention.yarn_factor` already exists).
- Non-YaRN scaling types (longrope etc. — `rope.rs` refuses them).
- MoE/A3B under YaRN (refused; do not implement).
- mrope (VL positions) + YaRN (orthogonal; mrope path untouched).
- Prefix cache + YaRN (sandwich path passes `RopeYarn::OFF`; document as
  a limitation).
