# Plan: DSpine drafter support (sidecar arch 24)

Status: planned (design + validation plan; engine work follows in this branch).

Author: Jerry-Lee661. Target model pair: `tencent/Qwen3.8-27B-DSpine-b16` (draft) + Qwen3.8-27B dense (target, arch 5). Validation hardware: single RX 7900 XTX (gfx1100).

## Motivation

DSpine (Tencent, [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)) is a 5-layer block-diffusion drafter that adds an **adjacent causal injection** pass on top of the DFlash block-diffusion skeleton: each draft layer reads the target hidden state through a per-layer read projection into a vocab-sized codebook, and writes a gated message (sigmoid-gated predecessor value) back into the next layer's input, with the last write refined by a serial sender-table chain over the top-K head candidates.

On Qwen3.8-27B the upstream card reports acceptance length 4.03-7.20 (vs 4.11-5.69 for DFlash2 on the same battery) and 2.71x-3.84x end-to-end throughput vs AR at concurrency 8. hipfire's qwen35 DFlash target machinery already carries the pieces the drafter needs (target hidden taps at `target_layer_ids`, DeltaNet snapshot/tape for the hybrid 3:1 attention target), so the missing surface is the drafter itself.

## Proposed shape

- **New sidecar arch_id 24** (`DSPINE_DRAFT`), following the 20/21/22/23 sidecar precedents; not a primary trunk id. Converter subcommand `dspine_convert` (naming aligned with `dflash_convert` / `mtp_extract`).
- Source: `tencent/Qwen3.8-27B-DSpine-b16` safetensors (4.05 GB BF16, 2.03B params; no own embed/lm_head — reuses the target's, hard-validated vocab 248320 / hidden 5120). Output: `.hfq` sidecar in **mq4** (primary, reuses the existing MQ draft GEMV machinery) and **bf16** (comparison tier; needs a draft-side bf16 GEMV path).
- Mechanism ported from the upstream Python reference (Triton slow path; the sm_90a CUDA fast kernels are out of scope and get Triton-equivalent HIP kernels):
  - per-layer bridge: read projection → anchor-row codebook lookup / non-anchor L2-normalized projection → packed [gate; predecessor_gate; value] message → `sigmoid(q + sender_prev + b) * value_prev` → write projection fused with RMSNorm;
  - last-write serial refinement: precomputed sender tables TG [V,R] / TSV [V,2R], block-wise K=16 candidate scoring chain (`serial_table` mode, `first_exact = n_layers-1` only);
  - config contract preserved: terminal disabled, independent write mode, causal draft block, BF16 activations, head_topk power-of-two in [2,64].
- **Attach style**: tentatively the arch-23 precedent (`HIPFIRE_DFLASH_DRAFT`-style env attach onto arch 5 loads); `params.drafter` (arch 22 precedent) is the alternative. Final choice in review.
- Registry entry: `qwen3.8:27b-dspine` (draft tag; the draft artifact rides next to the target artifact).

## Scope (first cut)

- Single GPU (gfx1100), qwen3.5-family dense targets only, `serial_table` second-pass mode only (converter refuses other modes), greedy + sampled chain verification. No DDTree, no VMM lanes, no EP/PP, no SPS/DSpark dynamic verification and no ReplaySSM-style state replay beyond what the existing qwen35 DFlash path already does. Concurrency admitted per the existing DFlash admission surface.

## Validation plan (per VALIDATION.md / arch-port-validation.md)

| Tier | What |
| --- | --- |
| C (kernel numeric) | the 4 new bridge kernels + serial-table chain vs CPU reference via `test_kernels` example, gfx1100 |
| P (parity) | hidden-state parity of the draft forward vs the upstream PyTorch reference on a tiny oracle (`gen_tiny_oracle.py` pattern), plus KLD vs BF16 teacher on pinned WT2/code24 windows following the `raon_oracle` template; and an end-to-end KLD/acceptance comparison against the llama.cpp DSpine port (Ajay9o9 `dspine` branch, d493f841) built on the same host as an independent oracle |
| S (speed) | gfx1100 prefill/decode vs committed floor (`tests/speed-baselines/gfx1100` exists); drafter-added decode vs same-card MTP and DFlash2 arms, per `perf-benchmarking.md` |
| U (serve) | `serve_harness.py` battery/chain with a new dspine mode (modeled on the `battery-dflash` variants), reasoning off |

Non-goals for this PR: no `admissions.yml` row (routing-capable, product-refused, consistent with most arches); no Redline claims; no multi-GPU.

## References

- Tencent/DSpine (Apache-2.0): https://github.com/Tencent/DSpine — `sglang_dspine/models/dspine.py` (DSpineCodeBridge), `kernels/dspine.py` (Triton), `runtime/worker.py` (DSpineDraftSampler)
- Weights: https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16 (config hard-contracts: terminal disabled, independent write, block-causal, `target_layer_ids=[1,16,31,46,61]`, mask 248070)
- Independent llama.cpp implementation: https://github.com/Ajay9o9/llama.cpp/tree/dspine; upstream PR https://github.com/ggml-org/llama.cpp/pull/31941
- Upstream card numbers: https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16#results

## Progress

- [x] Design + validation plan (this doc)
- [x] Semantics map (docs/plans/2026-11-12-dspine-semantics-map.md)
- [ ] `dspine_convert` (mq4 + bf16)
- [ ] arch-24 sidecar carrier + drafter load path
- [ ] bridge kernels (Tier C)
- [ ] serial refinement (Tier C)
- [ ] parity + KLD oracle (Tier P)
- [ ] gfx1100 end-to-end: acceptance length vs DFlash2/MTP arms (Tier S/U)
