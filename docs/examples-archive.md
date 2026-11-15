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
