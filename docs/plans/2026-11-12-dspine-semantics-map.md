# DSpine → hipfire semantics map (M2)

Companion to [2026-11-12-dspine-drafter.md](2026-11-12-dspine-drafter.md). Maps every component of the upstream reference (`sglang_dspine` 0.2.0, pinned to SGLang 0.5.22) to its hipfire counterpart. Sources: upstream source read (models/dspine.py, kernels/dspine.py, runtime/worker.py, csrc/) and the HF artifact `tencent/Qwen3.8-27B-DSpine-b16` (config + safetensors header), facts archived 2026-11-12.

## 1. Artifact & converter contract (`dspine_convert`)

| Upstream | hipfire |
| --- | --- |
| safetensors, 71 tensors, 4.05 GB BF16, no shards | `dspine_convert` subcommand (rust, sibling of `dflash_convert`/`mtp_extract`) → HFQ sidecar, `arch_id = 24` (header, not dir-derived); `model_type` mapping entry `dspine_draft` → 24 in `arch_mapping.rs` |
| 5 DFlash decoder layers (55 tensors: q/k/v/o + q/k_norm[128] + MLP 17408), fc [5120, 25600], hidden_norm [5120], norm [5120] | same tensor roles as the DFlash8B artifact the arch-20 loader already consumes → decoder weights ride the existing DFlash draft-load path (MQ4 quant default; bf16 tier stored raw) |
| `selected_prefix_bridge.*` 11 tensors + 4 I64 version scalars | new sidecar section in the HFQ metadata blob (raw bf16; codebook [248320,1024] ≈ 508 MB stays bf16 in both tiers — it is lookup-table data, not GEMM weight) |
| config hard contracts (runtime raises otherwise) | converter refuses (fail-closed, mirror wording of existing converter refusals): `terminal != disabled` (out.weight must be all-zero), `write_mode != independent`, `draft_block_causal != true`, dtype != bf16, `head_topk` not power-of-two in [2,64], `second_pass.mode != "serial_table"`, `first_exact != n_layers-1`, any version scalar != 1, presence of Domino/DFlash2 `prefix_gru`/`embed_proj`/`candidate_selector` tensors |
| no tokenizer files; vocab 248320 == target | no tokenizer in sidecar; draft borrows the target's |

Known deviation to flag in review: upstream **refuses quantized weights** (BF16-only runtime contract). hipfire's mq4 tier quantizes the decoder/MLP GEMM weights (bridge tensors stay bf16); the bf16 tier is the upstream-equivalent arm. Tier P validation therefore runs the bf16 arm against the PyTorch reference (structure gate), and the mq4 arm gets its own KLD gate (quant-noise tier, `raon_oracle` pattern).

## 2. Load time (carrier + target pairing)

| Concern | Plan |
| --- | --- |
| Sidecar attach | arch-23 precedent: env attach onto an admitted arch-5 (qwen3.5 dense) load; `params.drafter` (arch-22) alternative — final call in review |
| Target pairing checks (mirror arch-21's `check_mtp_head_for_trunk` shape) | trunk arch 5; `trunk.vocab == codebook rows (248320)`; `trunk.hidden == 5120`; target lm_head rows == 248320 and width == 5120 (draft borrows embedding + lm_head, BF16, unquantized — hard requirement from the sampler); `target_layer_ids` within trunk layer range |
| Refusal surface | follows `docs/load-refusal-wording.md`: each contract violation = one pointing refusal before upload |
| Registry | `qwen3.8:27b-dspine` draft tag riding beside the target artifact |

## 3. Drafter forward (per block of T=16)

Forward = 5 DFlash decoder layers with a code-bridge injection at every layer (`selected_prefix_feature_layers = [1..5]`). Draft consumes target **embeddings** (input_embeds) + target hidden taps at [1,16,31,46,61] via fc.

| Upstream stage | hipfire counterpart | Notes |
| --- | --- | --- |
| decoder layers (block-causal, sliding 8192, q/k_norm, silu MLP) | existing DFlash draft decoder machinery in `hipfire-runtime/src/dflash.rs` (`DflashWeights` load + layer forward) — reused nearly as-is | decoder weight shapes are identical to DFlash8B modulo dims; sliding-window causal mask over the 16-row block |
| `prepare_read` (Triton): x=hidden(+residual), RMS stats | HIP kernel 1 `prepare_read` (element-wise + row reduction) | outputs raw [m,5120] bf16, rms [m] f32 |
| projected = normalized @ `read_weight[layer]` (end layer) / `early_read_weights[i]` (layers 1-4) | GEMV/GEMM via existing MQ/BF16 draft GEMM dispatch (read weights are small: [1024,5120] ×4 +1) | |
| `normalize_codes` (Triton): anchor row ← `codebook[anchor_id] * sqrt(1024)`; others L2-normalized to sqrt(C) | HIP kernel 2 `normalize_codes` (gather by anchor token id + L2 rows) | anchor = first real token id of the block (the seed position); random row access into the 508 MB codebook → keep codebook GPU-resident bf16 |
| packed = codes @ concat(gate, predecessor_gate, prev) → [m,1536] = (query, sender, value) × 512 | single GEMM with a pre-concatenated weight (build the concat at load time) | |
| `feature_message` (Triton): `gate = sigmoid(query + sender_prev + bias)`, `message = gate * value_prev`; block-first row message = 0 | HIP kernel 3 `feature_message` (row-neighborhood dependency inside the block — pure element-wise over shifted rows, no reduction) | sender_prev/value_prev come from the *previous row's* packed segments |
| delta = message @ `feature_write_weights[layer]` [5120,512]; `feature_finish` (Triton): `output = raw + rms*delta` on non-first rows; next layer's input RMSNorm fused | HIP kernel 4 `feature_finish` (GEMV + fused epilogue) | write projection fused with the next input RMSNorm exactly as upstream (next layer enters with `input_normed=True`) |
| last layer stashes (raw, rms, packed) into `last_write_stash` | keep an equivalent stash buffer for the sampler | consumed by serial refinement |

## 4. Last-write serial refinement (sampling)

| Upstream | hipfire counterpart | Notes |
| --- | --- | --- |
| Precomputed sender tables at load: `TG [V,R] = bf16(lm_head·norm_w) @ W_last` (≈254 MB), `TSV [V,2R] = [G_p; A] @ (sqrt(C)·codebook)` (≈508 MB) | build at load on GPU; VRAM budget +0.76 GB on top of draft (4.05 GB bf16 / ~1.1 GB mq4) — fits 24 GB beside the 27B target | lm_head borrowed from target; `norm_w` = final norm weight |
| greedy: `topk_ids` (two-stage chunk top-k over head logits [n,248320]) → `serial_table` kernel ([n,K] selection over sender-id chain) → `serial_chain` walk of 16 rows, K=16 | HIP kernels 5-7: `head_topk`, `serial_table`, `serial_chain` | head-logits GEMM [n,5120]×[5120,248320] is the perf-critical GEMM of the drafter; score = base + rms·(sigmoid(q + TSV_gate[sid] + gate_b) · TSV_val[sid] @ TG[cand]); sid chains from the anchor row |
| sampled: `serial_rs_table` in fp32 (TF32 off; recomputes exact message/write/norm/head per candidate pair) → [n,K,K] transition table; `serial_sample_chain` does top-k/top-p/min-p chain sampling and emits q values | HIP eager path mirroring the same math in f32 accumulation (no TF32 equivalent on HIP to disable; keep f32 accumulators explicitly); q values feed the existing DFlash rejection-sampling verifier | only runs on sampled rounds; upstream keeps greedy path in CUDA graph — out of scope for v1 (no hipGraph) |
| confidence output (DFlash SPS hook) | not in v1 (SPS/DSpark dynamic verification is a non-goal) | |

## 5. Target integration (qwen3.5 dense, arch 5)

| Need | Status in hipfire |
| --- | --- |
| hidden taps at `target_layer_ids` [1,16,31,46,61] | exists — DFlash path already extracts `DflashTargetParts` at `target_layer_ids` |
| target **embedding** output for draft input_embeds | new tap (embeddings exist on `LoadedModel`; expose the pre-layer-0 embedding rows for the block) — small plumbing item |
| hybrid 3:1 GDN/full target state during verify | exists — `DeltaNetSnapshot` / `GdnTape` machinery in `dflash_spec.rs`; this is why upstream needs ReplaySSM and hipfire does not |
| verify | existing qwen35 `spec_step_dflash` block-parallel verify + rejection sampling for the sampled arm; greedy verify unchanged |

## 6. Numerics contract (preserved from upstream)

- BF16 activations end-to-end in the drafter; bridge tensors bf16 even in the mq4 tier.
- serial refinement scored path: f32 accumulators (upstream forces TF32 off; HIP has no TF32 to disable — state f32 accumulation explicitly in the kernel).
- codebook rows are unit-L2 scaled by sqrt(C); anchor rows bypass projection entirely (identity from codebook).
- The four version scalars must be validated == 1 at load (converter + loader double-check).

## 7. Concurrency & admission stance

- Upstream 27B optimized path caps concurrency at 8 (CUDA-graph greedy path). hipfire v1: no hipGraph, existing DFlash admission surface governs concurrency; no new cap introduced in the PR (document the difference in the plan doc).
- No `admissions.yml` row (routing-capable, product-refused), per the plan doc non-goals.

## 8. Work breakdown (M3/M4, from this map)

| Step | Content | Est |
| --- | --- | --- |
| M3 | `dspine_convert` (safetensors → HFQ arch-24; mq4 + bf16 tiers; converter refusals per §1) | 1.5-2 d |
| M4a | sidecar load path: metadata, bridge tensor placement, target-pair checks, refusal surface, registry tag | 1.5-2 d |
| M4b | bridge HIP kernels 1-4 + decoder-trunk reuse wiring (Tier C via `test_kernels`) | 2-3.5 d |
| M4c | refinement kernels 5-7 (greedy) + fp32 sampled path (Tier C; then Tier P) | 1.5-2.5 d |
| M5 | Tier P tiny-oracle parity + KLD (raon_oracle template) + llama.cpp-fork oracle; Tier S/U on gfx1100 | per contract |

## 9. Open questions for review

1. Attach style: env (arch 23) vs `params.drafter` (arch 22) — leaning env.
2. mq4 tier numerics: is KLD-gated mq4 acceptable for a first merge, or bf16-only until Tier P matures? (bf16 tier exists either way.)
3. Embedding tap: cleanest surface on `LoadedModel` for the pre-layer-0 rows (new accessor vs reusing an existing hook).
4. Codebook + sender tables (~1.3 GB combined bf16) — GPU-resident in v1; any objection vs host-paged?
