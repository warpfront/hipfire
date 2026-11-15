# Examples archive

Lab examples removed from the tree in the 0.5.0 consolidation. Each one was
named nowhere outside its own crate's `examples/` folder (no doc, CI job,
script, matrix row or `## Unreleased` CHANGELOG entry), was not the
byte-parity oracle of a live kernel or route, and was not a PeaceMaker twin,
ledger, release, eval or bench tool.

Git history keeps them. To restore one, check out its file at the listed
commit (the last commit that touched it) and re-add its `[[example]]` block
(most were gated on `required-features = ["lab"]`):

```sh
git show <sha>:<path> > <path>
```

## hipfire-arch-cohere2moe

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `longctx_probe` | `crates/hipfire-arch-cohere2moe/examples/longctx_probe.rs` | `9a09e55236d2` | 159 | Cohere2MoE: pin where the long-context forward collapses into a degenerate <PAD> / single-token attractor (~30K tokens). |

## hipfire-arch-deepseek4

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `ep_dspark_topology_probe` | `crates/hipfire-arch-deepseek4/examples/ep_dspark_topology_probe.rs` | `a0fca0d6db3f` | 496 | Admission probe for the asymmetric gfx1201 topology: target TP3 on devices 0..2, dedicated DSpark drafter on device 3. |

## hipfire-arch-gemma4

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `dump_gemma4_hidden_states` | `crates/hipfire-arch-gemma4/examples/dump_gemma4_hidden_states.rs` | `bf6ec9ddf240` | 144 | Dump per-layer hidden states from the Gemma 4 dense forward for offline comparison against the HF transformers oracle (arch bring-up validation). |
| `verify_overlap_gemma4` | `crates/hipfire-arch-gemma4/examples/verify_overlap_gemma4.rs` | `6563dff7f0e0` | 239 | Reproduce the non-zero-position overlap used by Gemma 4 EAGLE verify. |

## hipfire-arch-llama

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `qwen3_dspark_load_smoke` | `crates/hipfire-arch-llama/examples/qwen3_dspark_load_smoke.rs` | `3c16c169462a` | 136 | Load only the Qwen3-8B DSpark sidecar HFQ and verify every body layer and DSpark global tensor lands on the GPU. |

## hipfire-arch-muse-glimmer

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `verify_batch_glimmer` | `crates/hipfire-arch-muse-glimmer/examples/verify_batch_glimmer.rs` | `bf6ec9ddf240` | 122 | Batched verify parity harness for Muse Glimmer. |

## hipfire-arch-qwen2

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `profile_verify` | `crates/hipfire-arch-qwen2/examples/profile_verify.rs` | `ced1f71be757` | 121 | Per-kernel profile of qwen2 (arch_id=7) block-parallel spec-decode verify. |

## hipfire-arch-qwen35

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `test_qwen35_load_multi` | `crates/hipfire-arch-qwen35/examples/test_qwen35_load_multi.rs` | `bf6ec9ddf240` | 111 | Stage 4 smoke: load qwen3.5 weights distributed across 2 GPUs via qwen35::load_weights_multi. |
| `test_qwen35_state_multi` | `crates/hipfire-arch-qwen35/examples/test_qwen35_state_multi.rs` | `d52ddaf3bae0` | 238 | Stage 4 smoke: build the full multi-GPU state stack on a real model: Qwen35ScratchSet (per-device), KvCache::new_gpu_fwht3_capped_multi (per-layer K/V on band-owning… |

## hipfire-arch-qwen35-embed

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `inspect_source` | `crates/hipfire-arch-qwen35-embed/examples/inspect_source.rs` | `c53b6c1fd999` | 54 | Source diagnostics for arch 17: print how a tensor name resolves through each read path (tensor_data borrowed, tensor_data_pread / _vec owned) for a dir or an .hfq. |

## hipfire-arch-qwen4

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `qwen4_cost_prompts` | `crates/hipfire-arch-qwen4/examples/qwen4_cost_prompts.rs` | `4612a5d28322` | 606 | Offline cost-curve fixtures, generated from corpus messages, NOT byte-identical replicas of the 22-cell benchmark. |
| `qwen4_rowtile_capture` | `crates/hipfire-arch-qwen4/examples/qwen4_rowtile_capture.rs` | `a26defedb9ab` | 1547 | Live eager rowtile operand producer (Flash-Next serve loader capture). |

## hipfire-ds4-parent

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `ds4_parent_attn_mixed_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_attn_mixed_smoke.rs` | `bf6ec9ddf240` | 671 | Gate 5 attention lift smoke: run parent attention on one layer of each compress_ratio class against the real checkpoint. |
| `ds4_parent_attn_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_attn_smoke.rs` | `113c668b9684` | 290 | Gate 4 parent-attention smoke: load layer 0 (compress_ratio == 0) without experts and run parent_attention_swa over 16 rows at start_pos = 0. |
| `ds4_parent_compressor_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_compressor_smoke.rs` | `bf6ec9ddf240` | 858 | Gate 5 parent-compressor smoke: load layers 2..4 (ratio-4 then ratio-128) without experts and run parent_compressor_forward over 16 rows at start_pos = 0 for both the… |
| `ds4_parent_gpu_linear_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_gpu_linear_smoke.rs` | `bf6ec9ddf240` | 562 | Gate 3 slice GpuLinear — first runtime validation of gemm_bf16_mfma_gfx942 plus parent/linear.rs residency path. |
| `ds4_parent_head_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_head_smoke.rs` | `bf6ec9ddf240` | 501 | Gate 5 head smoke: embed + hc_head + final RMSNorm + lm_head on real weights. |
| `ds4_parent_indexer_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_indexer_smoke.rs` | `bf6ec9ddf240` | 498 | Gate 5 parent-indexer smoke: load layer 2 (first compress_ratio == 4) without experts and run parent_indexer_forward over 16 rows. |
| `ds4_parent_layer_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_layer_smoke.rs` | `bf6ec9ddf240` | 340 | Gate 4 parent-layer canary: load layer 0 with experts and run parent_layer_forward_traced over 16 rows at start_pos = 0. |
| `ds4_parent_moe_smoke` | `crates/hipfire-ds4-parent/examples/ds4_parent_moe_smoke.rs` | `bf6ec9ddf240` | 337 | Gate 4 parent-MoE smoke: load one real layer with experts and run parent_moe_forward over 16 rows of plausible activations. |
| `ds4_parent_seq_plog` | `crates/hipfire-ds4-parent/examples/ds4_parent_seq_plog.rs` | `bf6ec9ddf240` | 108 | Sequential parent forward (token-by-token) for long-seq diagnostics. |

## hipfire-generate

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `attn_wmma_row_probe` | `crates/hipfire-generate/examples/attn_wmma_row_probe.rs` | `c73864832f2c` | 147 | Row-invariance microtest for the batched Q8_0 prefill attention kernels (split-prefill investigation). |

## hipfire-loader

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `dspark_qwen3_1window` | `crates/hipfire-loader/examples/dspark_qwen3_1window.rs` | `0c208697b006` | 152 | Task-10 gate: load a Qwen3-8B target with its DSpark sidecar and drive one spec window through Speculator::prefill. |
| `qwen4_slab_d2_capture` | `crates/hipfire-loader/examples/qwen4_slab_d2_capture.rs` | `a26defedb9ab` | 628 | Same-input D2/all-GPU slot measurements and artifact-backed CPU fixtures. |

## hipfire-quantize

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `mq35_vq_fit` | `crates/hipfire-quantize/examples/mq35_vq_fit.rs` | `3243261ea933` | 494 | MQ3.5GL VQ fit & evaluation on real post-FWHT weights. |
| `mq_family_mse` | `crates/hipfire-quantize/examples/mq_family_mse.rs` | `3243261ea933` | 941 | MQ family MSE harness — rotated-domain quality vs bpw. |
| `poly_fit` | `crates/hipfire-quantize/examples/poly_fit.rs` | `3243261ea933` | 513 | Polynomial codebook fit on real post-FWHT weights. |

## hipfire-runtime

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `a13_repro` | `crates/hipfire-runtime/examples/a13_repro.rs` | `9f71309bfc75` | 134 | Minimal A13 concurrent-adversarial repro: submit N requests to a 2-slot engine, log every event per request, per-request recv timeout so a wedge is attributable to… |
| `check_think_tokens` | `crates/hipfire-runtime/examples/check_think_tokens.rs` | `37b36c8babce` | 42 | Quick check: what do <think> / </think> tokenize to in the Qwen3.5 tokenizer? |
| `debug_dequant` | `crates/hipfire-runtime/examples/debug_dequant.rs` | `37b36c8babce` | 106 | Debug: compare dequantized tensor values against a simple check. |
| `debug_hfq` | `crates/hipfire-runtime/examples/debug_hfq.rs` | `37b36c8babce` | 62 | Quick debug: load HFQ, run one forward pass, print logits stats. |
| `debug_hfq2` | `crates/hipfire-runtime/examples/debug_hfq2.rs` | `37b36c8babce` | 81 | Quick HFQ load + forward debug (variant of debug_hfq). |
| `dflash_smoke` | `crates/hipfire-runtime/examples/dflash_smoke.rs` | `37b36c8babce` | 142 | Load a converted DFlash draft .hfq and run a single forward pass with random inputs. |
| `ds4_tp_longctx_capacity` | `crates/hipfire-runtime/examples/ds4_tp_longctx_capacity.rs` | `0cb934112171` | 565 | DeepSeek V4 TP compressed-cache capacity probe. |
| `dump_jinja_render` | `crates/hipfire-runtime/examples/dump_jinja_render.rs` | `3c8e6141c544` | 60 | Dump the daemon's JinjaChatFrame render of an HFQ's embedded chat_template, for byte-comparison against HF apply_chat_template. |
| `ep_row_gather_probe` | `crates/hipfire-runtime/examples/ep_row_gather_probe.rs` | `38ce67a4407c` | 913 | K0c probe for hipfire_runtime::ep_row_gather (Flash-Next expert-parallel row all-gather) over --ranks GPUs. |
| `hfq_manifest` | `crates/hipfire-runtime/examples/hfq_manifest.rs` | `0990e3d491dd` | 83 | Print an HFQ container's tensor manifest and per-quant-type census. |
| `inspect_gemma4_hfq` | `crates/hipfire-runtime/examples/inspect_gemma4_hfq.rs` | `bf6ec9ddf240` | 67 | Print Gemma 4 HFQ tensor names/shapes. |
| `load_gguf` | `crates/hipfire-runtime/examples/load_gguf.rs` | `37b36c8babce` | 94 | Test: parse a GGUF file, print model config and tensor info. |
| `measure_launch` | `crates/hipfire-runtime/examples/measure_launch.rs` | `37b36c8babce` | 41 | Measure raw kernel launch overhead by launching a trivial kernel many times. |
| `moe_c8_roofline` | `crates/hipfire-runtime/examples/moe_c8_roofline.rs` | `9dfcc0e85eaf` | 465 | Roofline microbench of the routed-expert decode kernels at the EP2 C8 AR shape, on the REAL expert tensors of a .mq4-xts artifact, all VRAM-resident. |
| `ple_reuse_probe` | `crates/hipfire-runtime/examples/ple_reuse_probe.rs` | `bf6ec9ddf240` | 283 | CPU-only PLE sidecar reuse probe. |
| `ple_row_compare` | `crates/hipfire-runtime/examples/ple_row_compare.rs` | `5bce1517475e` | 604 | CPU-only PLE table row-payload comparison: are the *encoded row bytes* of two ple-table sidecars (HFPLEHP1 v3) identical, row for row in canonical logical row order… |
| `probe_wmma` | `crates/hipfire-runtime/examples/probe_wmma.rs` | `37b36c8babce` | 70 | Probe WMMA thread-to-output mapping empirically. |
| `profile_forward` | `crates/hipfire-runtime/examples/profile_forward.rs` | `37b36c8babce` | 50 | Profile one forward pass: time each operation category. |
| `profile_layers` | `crates/hipfire-runtime/examples/profile_layers.rs` | `3ee13a451fca` | 616 | Per-layer profiling of hipfire inference. |
| `render_chat_template` | `crates/hipfire-runtime/examples/render_chat_template.rs` | `3c8e6141c544` | 78 | Render a model's upstream HF chat_template via JinjaChatFrame and print the result to stdout. |
| `render_cohere2moe` | `crates/hipfire-runtime/examples/render_cohere2moe.rs` | `3c8e6141c544` | 156 | Repro for the Pi multi-turn agentic derail: render a conversation that has an assistant turn WITH tool_calls (+ a tool result) through the Cohere tool_use template,… |
| `test_e4_engine_repro` | `crates/hipfire-runtime/examples/test_e4_engine_repro.rs` | `9f71309bfc75` | 146 | Temporary audit probe: submit the E4 enum/const schema request directly to SlotEngine and watch events + per-token mask decisions. |
| `tmp_fa2a_g2` | `crates/hipfire-runtime/examples/tmp_fa2a_g2.rs` | `bf6ec9ddf240` | 501 | THROWAWAY Slice-A G2 harness. |
| `tmp_iu4_attrib` | `crates/hipfire-runtime/examples/tmp_iu4_attrib.rs` | `bf6ec9ddf240` | 125 | Throwaway iu4 attribution runner: JITs a TU copy from disk and times the N=512 gate_set case exactly like tmp_iu4_gfx12_oracle -- TIME. |

## rdna-compute

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `host_kernel_access` | `crates/rdna-compute/examples/host_kernel_access.rs` | `e3066e7396ab` | 228 | Can a shader read host-mapped memory — and specifically the quantized path? |
| `test_mfp4e8_grouped_wave64x4_gfx942` | `crates/rdna-compute/examples/test_mfp4e8_grouped_wave64x4_gfx942.rs` | `bf6ec9ddf240` | 212 | gfx942 native-wave64 qt35 projection candidate channel. |
| `tmp_scratch_leak_probe` | `crates/rdna-compute/examples/tmp_scratch_leak_probe.rs` | `bf6ec9ddf240` | 133 | Scratch-leak probe (gate 1 of the grow_scratch_buffer leak fix). |
| `tp_peer_push_probe` | `crates/rdna-compute/examples/tp_peer_push_probe.rs` | `127dbb74eac3` | 1284 | Lab microbench: in-kernel peer-VRAM push + system-scope flag collective against the incumbent TP2 tp_graph_signal_reduce_gfx1100 collective, on two exact gfx1201 cards. |
| `wmma_zero_sign_probe` | `crates/rdna-compute/examples/wmma_zero_sign_probe.rs` | `b2e67a8ab9ca` | 134 | gfx11 WMMA zero-sign probe: output element 0 = A[0][0]*B[0][0] + masked partners A[0][k>=1] * (+0) + C. |

## redline-rocr

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `pm4_ib_smoke` | `crates/redline-rocr/examples/pm4_ib_smoke.rs` | `dce28e65d31d` | 37 | Minimal redline PM4 indirect-buffer submit smoke. |

## saddle-lab

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `a3b_load_check` | `crates/saddle-lab/examples/a3b_load_check.rs` | `77a471b33efa` | 92 | Phase 1 smoke test: load Qwen3.5-35B-A3B (or any qwen3_5_moe HFQ) end-to-end and report success/failure. |
| `gpus_smoke` | `crates/saddle-lab/examples/gpus_smoke.rs` | `a0fca0d6db3f` | 111 | Smoke test for Gpus orchestrator: init_uniform layer split, peer enable, boundary_copy + wait_boundary round-trip. |
| `infer_qwen3` | `crates/saddle-lab/examples/infer_qwen3.rs` | `d52ddaf3bae0` | 495 | Run inference on a .hfq (hipfire-quantized) model. |
| `test_qwen35_load` | `crates/saddle-lab/examples/test_qwen35_load.rs` | `77a471b33efa` | 159 | Minimal test: load Qwen3.5 HFQ, parse config, print layer types. |
| `test_qwen35_loadQA` | `crates/saddle-lab/examples/test_qwen35_loadQA.rs` | `77a471b33efa` | 185 | QA mirror for Qwen3.5 HFQ loading and config validation. |

## va-bridge

| Example | Path | Last commit | LOC | Purpose |
|---|---|---|---|---|
| `parse_fixtures` | `crates/va-bridge/examples/parse_fixtures.rs` | `492a7e40f265` | 34 | Throwaway probe (experiment/vcn-jpeg): parse the 5 committed fixtures for VA submission without touching the GPU. |
