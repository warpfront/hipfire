# Mfp4Diag receipt — qt24 MFP4G32 KLD collapse on gfx1201: diagnosed + fixed

Worktree wt-mfp4 @ f72a5add5 (branch mfp4-fp8). GPU ord2. Eval:
`HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1`,
`--kv-mode q8 --kv-v q8 --scoring-mode prefill`. Logs in this dir.

## Root cause (single defect, not a format problem)
`crates/hipfire-dispatch/src/pipeline/mod.rs` `dispatch_fused`
(`GemvMfp4G32Fused` arm, pre-fix lines 3809–3821) called
`gpu.gemv_mfp4g32_with_rotate(buf, x, y, …)` — a raw-x API that FWHT-rotates
internally — on `params.x`, which under the `run(Prerotated)` contract is
ALREADY rotated (`run_input(Raw)` rotates via RotationFamily;
`llama::weight_gemv` rotates before `run`). Result: every MFP4G32 GEMV on
HasWmma archs (gfx11/gfx12) computed FWHT(FWHT(x)) vs FWHT(W) = silent
garbage. Non-WMMA archs never hit it (fused key requires HasWmma).
Introduced with the dispatch-unification fused variant (b80a2261d lineage);
the same double-rotation trap from #393 was documented for `run_auto` in
`llama.rs:1344-1356` but re-introduced one layer down.
Fix (commit f72a5add5): route to `gemv_mfp4g32_prerotated` (no second rotate).

## Oracles
- Host round-trip (L0 gate_proj, indep. python decoder vs 42/1042-rotated
  parent): relL2 0.116638, tail-1% mean-err/refmax 0.042 → quantizer SANE.
- Split: PREFILL_BATCHED=0 (decode GEMV) KLD 12.362254 == per-token KLD
  12.362254 bit-identical; prefill-WMMA run KLD 11.88 (same garbage class) →
  common cause = rotation/GEMV dispatch, NOT the 4 WMMA prefill kernels.
- HFP4G32 (no rotation) on PRE-FIX binary: wt2_1 KLD 0.054287 → codec, loader
  (qt21/24 AoS), GEMV decode, and all 4 gfx12 WMMA prefill kernels PROVEN.
- Post-fix MFP4 wt2_1 KLD 0.050535 → rotation fix proven end-to-end.

## KLD table (fixed binary; md5s below)
| arm   | wt2_1  | wt2_2  | wt2_24 | ag_24  |
|-------|--------|--------|--------|--------|
| mfp4  | 0.0505 | 0.0579 | 0.0816 | 0.2518 |
| mfp4+awq | 0.0399 | 0.0481 | 0.0615 | 0.1777 |
| hfp4  | 0.0543 | 0.0606 | 0.0812 | 0.2368 |
| hfp4+awq | =hfp4 | =hfp4 | =hfp4 | =hfp4 |
(AG domain runs higher on every arm — domain effect, not a defect.)

## Artifacts (md5)
- plain mfp4 `qwen3.8-27b.mfp4g32.hfq` 7dc374d5a5dc072e8d83c45b94ee7cbe
- mfp4+awq `qwen3.8-27b.mfp4g32.awq.hfq` 53e64a02630fd50224a31f02d3609289
- plain hfp4 `qwen3.8-27b.hfp4g32.hfq` f640dc8e5ba8905c2db0a35c30fe6270
- hfp4+awq: byte-IDENTICAL to plain hfp4 (same md5) — AWQ unwired for HFP4G32
  (no pre-scale branch in pipeline.rs hfp4 dense arm; HFP4G32 absent from
  `supports_awq_sidecar`; no x/s-without-FWHT runtime path). Needs a new arm;
  out of scope here.

## Notes
- The route never worked for dense qwen3_5 on gfx1201: what was missing was
  Prerotated-contract routing in the fused MFP4 GEMV path (now fixed).
- Pre-fix garbage rows kept as evidence: eval_mfp4_gemvforced_wt2_1.log,
  eval_mfp4_pertoken_wt2_1.log (both KLD 12.362254).
- mfp4 plain wt2_24/ag_24 first attempt OOM'd at layer-57 load (transient;
  identical retry EXIT=0). AWQ sidecars attach; x/s divide verified by
  mfp4+awq wt2_1 0.0399 < plain 0.0505.
