# Unit 1 — iu4 raster remap coverage (Raster2, verify-only)

Base: `e044e3b96` (mq4-lloyd) in wt-raster2, branch `gfx1201-raster2`.
GemmRaster2 owns the raster commit in wt-raster (`gfx1201-gemm-raster`, G8T0
leading); this worktree rebases onto it when it lands. No kernel edits here.

## Claim
The grouped-tile remap lives in the single shared gfx12 body
`gemm_mq4g256v2_residual_mmq_iu4_gfx12_body`, so every iu4 launch inherits it
by construction. Verified by code inspection (no behavior change in this unit).

## Funnel (all paths converge)
- Model roles → prepared wrappers (`crates/rdna-compute/src/gemm.rs`):
  - gate_up → `gemm_gate_up_mq4g256v2_wmma_iu4_prepared` (:31068) → set ×2
  - qkv → `gemm_qkv_mq4g256v2_wmma_iu4_prepared` (:29853) → set ×3
  - qkvza → `gemm_qkvza_mq4g256v2_wmma_iu4_prepared` (:29289) → set ×2–4
  - residual (wo/down) → `gemm_mq4g256v2_residual_wmma_iu4_prepared` (:32672),
    `gemm_hfq4g256_residual_wmma_gfx12_mq4v2` (:32331),
    `gemm_mq4g256v2_residual_wmma` (:32513) → add
- Wrappers → ONE private fn `gemm_mq4g256v2_mmq_prequant_iu4` (:19570):
  - `gemm_mq4g256v2_mmq_set_prequant_iu4` (:19751) → private(add=false)
  - `gemm_mq4g256v2_mmq_add_prequant_iu4` (:19763) → private(add=true)
- Private fn → module `gemm_mq4g256v2_residual_mmq_iu4_gfx12`
  (`_full_set`/`_full_add`, :19602-19607; gfx1151/generic fallback separate).
- All 3 gfx12 entries call the shared body:
  - `gemm_mq4g256v2_residual_mmq_iu4` (:687) → body<add?>
  - `..._full_add` (:701) → body<true>; `..._full_set` (:711) → body<false>.
- Raster remap site = body head (`rs`/`bs` from blockIdx, :283-284).

## Census cross-check (pp8192, qwen3.8-27b-mq4-xt, kv-mode fp8)
`profile_prefill_qwen35`, full table in `pp8192-census.log`:
- `iu4_full_set` 368 calls = gate_up 2×64 + qkv 3×16 + qkvza 4×48 ✓
- `iu4_full_add` 128 calls = down 64 + wo 64 ✓
Every GEMM launch in the trace goes through the shared body. No orphan site.

## Baseline rig rows (ordinal 2, same-GPU reference for post-rebase compare)
kx3 probe, `base-A.rig.log` (3 runs, medians):
- gate_up iu4: 6641.4 / 6649.0 / 6661.9 us
- residual iu4: 3017.0 / 3018.1 / 3029.8 us

## Follow-up — MOOT 2026-09-19
GemmRaster2 killed the raster line (KILL.md in wt-raster scratch: G8T0 kernel
win −10.7% but end-to-end transfer +0.3–1.0%, below ship bar; no commit will
land, kernel reverted to HEAD). No rebase pending; coverage proof above stands
as the unit deliverable, baseline rig rows retained as reference.
