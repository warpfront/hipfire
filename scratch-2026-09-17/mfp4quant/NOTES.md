# MFP4Quant evidence log — qt24 MFP4G32 baseline (E2M1 + UE8M0 + f16 row + FWHT)

Worktree: /home/kaden/ClaudeCode/warpfront/wt-mfp4 @ 775ac0612 (branch mfp4-fp8)
GPU ordinal: 2 (ROCR_VISIBLE_DEVICES=2), HOME=/home/kaden/.hipfire-homes/ab2

## Recipe
- Command: ./target/release/hipfire-quantize --input /home/kaden/qcal/parents/qwen3.8-27b/ --output /home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mfp4g32.hfq --format mfp4
- Format: MFP4G32 qt24 (quantize_mfp4g32_2d, FWHT signs 42/1042, RTN, no AWQ, no imatrix)
- Gap vs MQ4V2 fixture recipe (CLOSED by Kaden ruling — AWQ now wired, see below).
  Original gap: AWQ alpha 0.55 pre-scale NOT applied; imatrix NOT consumed; no Lloyd
  iterations. MFP4 dtypes not in DType::supports_awq_sidecar.
- Quality variants in-family (break exponent folding, NOT used): mfp4l/qt32 (Lloyd
  codebook), mfp4p/qt33 (E4M3 scales). E8 variants (qt34/35, GPTQ/LSQ/AWLS) differ.

## Artifacts
- plain: /home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mfp4g32.hfq
  md5 7dc374d5a5dc072e8d83c45b94ee7cbe; elapsed 3m18.404s;
  census 497 MFP4G32 / 49 Q8F16 / 305 F16; 15046.8 MB
- awq: /home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mfp4g32.awq.hfq
  (--format mfp4 --imatrix .../Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55) — PENDING (bg_6)

## Code changes (branch mfp4-fp8, this worktree)
- quantize/pipeline.rs: mfp4 dense arm AWQ pre-scale (mirrors MQ4V2 arm; byte-identical
  to plain when --awq/--imatrix absent).
- rdna-compute/dispatch.rs: MFP4G32 in supports_awq_sidecar (loader attach; rotate
  kernels already route on awq_scale.is_some()).
- dispatch-tests/dtype.rs: allow-list test updated (scoped test passes).

## KLD (tree eval_hipfire bb2c25cb, gfx1201 ord2, q8kv, prefill scoring)
- plain wt2_1:  KLD 11.884398 NLL 14.031357 PPL 1240911.75 (out d0a1b2be)
- plain wt2_2:  KLD 11.809814 NLL 13.944962 PPL 1138203.32 (out 8115807a)
- plain wt2_24: KLD 12.266993 NLL 13.967512 PPL 1164162.25 (out 9459a3d4)
- plain ag_24:  KLD 12.634560 NLL 14.004310 PPL 1207798.54 (out 024de61e)
- awq wt2_1:    KLD 11.944646 NLL 14.088305 PPL 1313630.31 (out 365c56ed)
- awq wt2_2:    KLD 11.892109 NLL 14.028958 PPL 1237938.77 (out c34c5dfa)
- awq wt2_24:   KLD 12.328932 NLL 14.025926 PPL 1234190.13 (out 8b06557f)
- awq ag_24:    PENDING (bg_3)
- Bar: 0.0744. Verdict: COLLAPSE both arms (PPL ~1.2M).
- Route proof: fused hfp4g32 WMMA prefill GEMMs JIT-compiled fresh
  (gemm_gate_up_hfp4g32_wmma_gfx12, gemm_hfp4g32_residual_wmma_gfx12);
  AWQ arm compiled *_rotate_awq twins (sidecars attached, x/s live).
- CPU roundtrip (L0 gate_proj): relL2 0.1166 vs FWHT-rotated parent —
  quant matches spec; kernel LUT/nibble/UE8M0 math verified identical.
