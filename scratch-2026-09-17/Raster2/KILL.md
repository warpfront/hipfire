# KILL.md — Unit 3: kernarg preload (Raster2)

Verdict: **KILL. Lever is inert on this toolchain — no codegen change through
any spelling, so no rig delta is possible. Tree left clean (no commit).**

## Target (pp8192 trace, top launch-count kernels)
`profile_prefill_qwen35 qwen3.8-27b.mq4-xt --prefill 8192 --kv-mode fp8`
(full table: `pp8192-census.log`), ordinal 2, HIP 7.15 / clang 23.0git:

| rank | kernel | calls | share |
|---|---|---|---|
| 1 (count) | `gated_delta_net_q8_batch_seq` | 768 | 13.3% |
| 1 (time) | `gemm_mq4g256v2_residual_mmq_iu4_full_set` | 368 | 39.3% |
| 2 (time) | `gemm_mq4g256v2_residual_mmq_iu4_full_add` | 128 | 18.7% |
| 4 | `fused_silu_mul_mq_rotate_awq_i4_gfx12_batched` | 64 | 8.1% |
| 5 | `fused_rmsnorm_mq_rotate_awq_i4_gfx12_batched` | 128 | 5.3% |

iu4 entries = 10 kernarg dwords (3 ptr + M,K,N,add); both share module
`gemm_mq4g256v2_residual_mmq_iu4_gfx12` (probe module `kx3_iu4_shipped_mod`).

## Screen matrix (all on gfx1201, ordinal 2)

| # | mechanism | result |
|---|---|---|
| 1 | `-mllvm -amdgpu-kernarg-preload-count=10` via `module_flags_for` (repo JIT, iu4 module) | compiles; `.text` **byte-identical** to no-flag build; only 96B of `.dynsym`/hash metadata differ. Occupancy unchanged (3/CU). Log: `preload10-B1.*` |
| 2 | `-mllvm --amdgpu-kernarg-preload --amdgpu-kernarg-preload-count=10` (enable+count, repo JIT, iu4) | `.text` **byte-identical** to screen 1. Log: `preload-en-B1.*` |
| 3 | hipcc toy kernel, counts 3/4/5/16, single- and double-dash, full + `--cuda-device-only` | prologue keeps `s_load_b32 sN, s[0:1], off` in all builds |
| 4 | direct backend (`clang --target=amdgcn-amd-amdhsa -mcpu=gfx1201` on device IR) ± flags | outputs **byte-identical** (`cmp`: IDENTICAL) |
| 5 | `-amdhsa-code-object-version=6` + flags | no change |
| 6 | `__attribute__((amdgpu_kernarg_preload))` | `warning: unknown attribute 'amdgpu_kernarg_preload' ignored` — does not exist |
| 7 | `-mllvm -help-hidden` | options `--amdgpu-kernarg-preload[++count]` exist; pass named `amdgpu-preload-kernargs` is registered but never fires (not observable: release LLVM has no `-debug-only`) |

ISA proof: `s_load_b96 s[12:14], s[0:1], 0x18` + `s_load_b128 s[4:7], s[0:1], 0x0`
persist in every build; no `.amdhsa_user_sgpr_kernarg_preload_*` notes emitted.

## Rig rows (kx3 probe, both shapes — all within run-to-run spread ±0.3%)

| arm | gate_up iu4 (us) | residual iu4 (us) |
|---|---|---|
| base ×3 (`base-A.rig.log`) | 6641.4 / 6649.0 / 6661.9 | 3017.0 / 3018.1 / 3029.8 |
| count-only (`preload10-B1.rig.log`) | 6654.4 | 3019.0 |
| enable+count (`preload-en-B1.rig.log`) | 6650.7 | 3014.6 |

Δ ≈ 0, explained by identical machine code. No paired-gate run warranted
(nothing to promote; would burn two pairs to re-prove 0).

## Conclusion
The `-amdgpu-kernarg-preload*` options are accepted but inert in ROCm/HIP
7.15's clang 23 for these kernels (pass registered, never fires; attribute
spelling absent). Applies equally to the other top-4 kernels (same backend).
Revisit only with a toolchain bump + a re-screen showing a `.text` diff.
No commit; `compiler.rs` reverted to base (verified `git status` clean
except the untracked probe copy).
