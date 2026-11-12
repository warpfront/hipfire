# Decode levers — full lens on Qwen3.6-35B-A3B (gfx1151)

Read-only audit, 2026-05-20. Companion to
[`hipengine-perf-comparison.md`](./hipengine-perf-comparison.md). That doc owns
the top-5 (graph replay, GDN chunkwise prefill, multi-rotation fusion, AWQ
pack8, AOTriton split-K). This doc widens the lens to **every** decode-phase
lever hipEngine uses that hipfire today does not (or does worse).

## Summary

- **35 distinct levers identified** (excluding graph-replay re-enable, which has
  its own row but the full analysis lives in the prior comparison report).
- **Top-3 quick wins (≤1 day each, NOT graph-replay):**
  1. **Device-side argmax + persistent `lm_out_index` token scalar**
     (replaces today's `download_f32(logits) → CPU argmax` and the per-step
     `memcpy_htod(pos_i32)`). Unlocks graph capture even before the GDN drift
     bug is fixed. ~+8-15% decode and a *prerequisite* for graph replay.
  2. **F32 → FP16 activation dtype through the layer body** (hidden, gate,
     attn_out, ffn_hidden, etc. — hipfire is F32 everywhere; hipEngine is FP16
     end-to-end with one BF16 cast only at lm-head input). Halves glue-kernel
     bandwidth on dim=4096 vectors. ~+10-15% decode.
  3. **Pre-allocated decode scratch arena** (replace the 294 `alloc_tensor` /
     `free_tensor` calls per token with a once-reserved scratch struct, the way
     hipEngine's `reserve_full_attention_scratch` / `reserve_moe_c1_scratch`
     work). Eliminates allocator-induced sync points. ~+3-8% decode and another
     graph-capture prerequisite.
- **Biggest cumulative-headroom phase:** the **GDN decode block**
  (LinearAttention QKV-Z proj → conv1d → QK-norm/scale → beta/alpha → GDN
  state update → out-norm/gate → out-proj). hipfire dispatches ~15 kernels
  through here; hipEngine packs it into ~5 (including 2 mega-fused: the
  `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel` does
  qk-norm + beta + decay + recurrent + out-RMSNorm + z-silu-gate in **one
  kernel**, and the `qwen35_head_rmsnorm_partial_rotary_position` does
  q-norm + k-norm + partial-RoPE in **one kernel** for full attention). At 30
  layers (12 LA + 18 FA per the model card) this swing alone is plausibly
  +15-25% decode independent of graph replay.

Phase tags below match the original brief. "Impact%" is rough expected decode
%; sign included; `?` when unclear. Effort: XS ≤2h / S ≤1d / M ≤1w / L >1w.

## Levers

| Phase | Lever | What it does | hipEngine impl | hipfire today | Effort | Impact% | Notes |
|---|---|---|---|---|---|---|---|
| LinearAttention QK norm + scale | Fused L2-norm(Q) + L2-norm(K) + scale fused into the GDN mega-kernel | hipEngine has **no separate QK-norm kernel** — the GDN kernel reads conv_out, computes Q/K L2-norm + scale inline, then does the recurrent step. See `kernels/hip_gfx1100/linear_attn/gdn.hip:54-147`. | `kernels/hip_gfx1100/linear_attn/gdn.hip:90-147` (L2 sum/reduce + scale) | hipfire has `gpu.fused_qk_l2_norm_scale_f32` (`qwen35.rs:3093-3100`) as a *separate* kernel before `gated_delta_net_q8`. | M | +3-5% | Subsumed by GDN mega-fusion (next row). Standalone gain limited; real win is combining both. |
| LinearAttention beta/alpha gating | beta-sigmoid + alpha-(decay = expf(-expf(a_log)*softplus(...))) computed **inside** the GDN kernel (one fmuladd path) | hipEngine does not have a separate "sigmoid_alpha_gate" kernel; beta and decay are scalar-per-head, computed once at line 148-149 of `gdn.hip`. | `kernels/hip_gfx1100/linear_attn/gdn.hip:148-149` | `gpu.fused_sigmoid_alpha_gate_f32(&beta_out, &alpha_out, &dt_bias, &a_log, n_v_heads)` `qwen35.rs:3071-3073` — already fused into one kernel, but separate from GDN. | S | +0.5-1% | Already fused on hipfire; only dispatch elimination via GDN merge. |
| Gated DeltaNet hidden-state update | **Mega-kernel** `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel` (296 LOC): QK-L2-norm + beta + decay + recurrent state update + per-value L2 (norm_partial) + RMSNorm (rsqrt) + z·silu output gate | One kernel does what hipfire does in 4-5 separate kernels. F32 state, FP16 in/out for z and out. | `kernels/hip_gfx1100/linear_attn/gdn.hip:54-245` (entire kernel) | `gpu.gated_delta_net_q8` (`qwen35.rs:3133`) → then *separate* `gpu.gated_norm_f32` (`qwen35.rs:3153`). Out-RMSNorm and z-gate are a second launch. | L | +5-10% | The single largest decode-glue fusion opportunity. Two kernels → one, but more importantly: eliminates a full-vector `[1, v_dim]` round-trip through global memory. **Has correctness risk** — same kernel suspected of the graph-replay drift bug. Fix that bug first. |
| Gated DeltaNet hidden-state update | Q8 KV state (recurrent_state in F32, no quant) | hipEngine keeps `recurrent_state` in F32 (`gdn.hip:62 float* recurrent_state`). hipfire offers Q4/Q8/F32 (`qwen35.rs:3133-3144`). | `gdn.hip:62, 156-225` (F32 reads) | Same: hipfire offers more aggressive quant. | n/a | n/a | Already a hipfire feature, not a lever. |
| LinearAttention output norm + gate + projection | Out-RMSNorm + z·silu folded into GDN mega-kernel | See GDN row. Skips a `gated_norm_f32` launch entirely. | `gdn.hip:226-244` | Separate `gpu.gated_norm_f32` (`qwen35.rs:3153`). | — | — | Same lever as GDN mega-fusion above. |
| FullAttention output gate (sigmoid·mul) | See gate-fused decode above | — | — | — | — | — | Subsumed. |
| LM head GEMV | W8A16 lm-head with FP16 input, F32 logits (`w8a16_linear_bf16_f32_out`) | One kernel does the W8A16 GEMV; lm-head weights are INT8 with FP32 scale, input is BF16 (cast from FP16 via `fp16_to_bf16`), output is F32 logits ready for argmax. | `runner.py:3227-3239` | hipfire computes lm-head via `weight_gemv` and then `download_f32(logits)` (`qwen35.rs:3016`) — full F32 logits download. | S | +1-3% | Subsumed by device-argmax (top-3). |
| Sampling (greedy argmax) | Two-stage device argmax: `lm_head_argmax_stage1_blocks` + `argmax_f32` ; writes result into a *device* int64 scalar `lm_out_index` that survives across replay. | Stage 1: per-block argmax into `lm_block_values`/`lm_block_indices`. Stage 2: final reduce into `lm_out_index` (i64). No host roundtrip. | `runner.py:3240-3251`; helpers at `kernels/hip_gfx1100/linear/lm_head.py:152` | hipfire **downloads full logits vector** to host (`qwen35.rs:3016 download_f32(&logits_gpu)`), then argmaxes on CPU (`runtime/llama.rs:4386 argmax`). | S | +5-10% | **Top-3 quick win.** Hipfire already has `argmax_f32` GPU kernel (`dispatch.rs:18174`) — just unused on the decode path. ~150k F32 elem D2H copy eliminated per token + the implicit sync. **Prerequisite for graph capture.** |
| Sampling | Top-p / non-greedy on GPU | Out of scope for hipEngine's MoE path; only greedy is documented for the comparison rows. | n/a (eager AR run greedy) | hipfire has CPU top-p / nucleus sampling (`sampler.rs:55-119`). | — | — | **Already richer on hipfire.** No lever — hipEngine doesn't sample-top-p in the perf-claimed path; comparison is greedy-vs-greedy. |
| hipGraph capture + replay | Shape-bucketed graph: per-`(context, num_splits, chunk_size)` graph instance | hipEngine recomputes `num_splits = max(1, (position + replay_span + chunk_size - 1)//chunk_size)` per capture (`runner.py:1500`). For long-running gens, multiple graphs are captured per context-length bucket. | `runner.py:1500, 2624` | hipfire's graph path uses one capture (per qwen35.rs:3811-3813 logic). | M | +1-3% at long context, ~0 at short | Future work; useful only after #1 lands. |
| hipGraph capture + replay | In-graph token recording: `record_i64_scalar_indexed` writes each replayed token to a device-side ring buffer for correctness gates without host sync | hipEngine optionally captures a `record_steps` device int64 buffer and stamps each token id into it; host reads only at end of N replays. | `runner.py:1503-1508, 1579-1587` (record_i64_scalar_indexed) | N/A — hipfire reads tokens host-side every step. | XS-S | +0.5-1% | Implementation overhead is small; eliminates the per-step "what token did we just emit?" host probe. Only relevant under graph replay. |
| KV cache layout (decode bandwidth) | INT8-per-token-head storage **with no BF16 shadow** | Cache stores INT8 [layers, pages, block, kv_heads, head_dim] plus per-(row, kv_head) FP16 scale; the decode kernel `paged_attn_decode_int8_gqa_splitk` loads INT8 directly. | `docs/KVCACHE.md:124-158` | hipfire has `kv_cache_write_q8_0` (`qwen35.rs:3221`) — Q8 KV. | — | — | **Already implemented.** Asym3 KV is the hipfire-specific bandwidth lever (3-bit, smaller than Q8); confirm the bench actually uses `--kv-mode q8` and consider asym3 if quality holds. |
| KV cache layout | Q8 KV per-(token, head) scale vs per-channel scale | hipEngine uses per-(row, kv_head) FP16 scale (`KVCACHE.md:124-128`). | `paged_kv_write.hip` | hipfire `kv_cache_write_q8_0` — confirm scale granularity matches; same shape works. | — | — | Verify only; no lever expected. |
| Speculative decoding / MTP | MTP target-attached draft + chain verifier | hipEngine plan exists (`docs/MTP.md`), best measured row is `0.699× AR`. Currently **deferred** for the A3B MoE. | `docs/MTP.md:58` | N/A — hipfire has DFlash for 27B dense but no MTP for A3B. | n/a | -30% to +20% | **Skip.** η < 1 on this MoE per `SPECULATIVE-DECODE.md:283-365`. MoE expert union explodes with B>1; spec decoding does not win on Qwen3.6-35B-A3B. Do not pursue. |
| Speculative decoding / MTP | DFlash block draft on A3B | Same as MTP — fundamentally limited by MoE expert-union per `SPECULATIVE-DECODE.md:404-461`; break-even acceptance needs A ≥ 0.9 (impossible). | `docs/SPECULATIVE-DECODE.md:283-417` | N/A | n/a | -30% to +10% | **Skip.** Same reasoning. The hipEngine team concluded "Accept this is not a speculation model" (`SPECULATIVE-DECODE.md:816`). |
| Cross-cutting: activation dtype | **F32 → FP16 across the layer body** (hidden, residual, intermediate, attn_out, ffn_out) | hipEngine: hidden FP16 (`runner.py:2729 hidden_nbytes = hidden_size * FP16.itemsize`), one BF16 cast at lm-head only (`runner.py:3219-3226`). | hipfire: F32 throughout — `let x = gpu.alloc_tensor(&[dim], DType::F32)` (`qwen35.rs:2993`) and every `tmp`/`o`/`attn_out`/`ffn_out` is F32. | L | +10-15% | **Top-3 #2.** Halves bandwidth on every dim=4096 vector touch. Risk: numerical drift in RMSNorm and the GDN recurrent state. Coherence gate mandatory. |
| Cross-cutting: allocator | **Pre-allocated scratch arena** (`reserve_full_attention_scratch`, `reserve_moe_c1_scratch`) | hipEngine allocates per-layer scratch *once* and reuses; no per-token allocations. | hipfire calls `gpu.alloc_tensor` 294 times in `qwen35.rs` (decode forward path). Per-token: ~14 allocs in DeltaNet block × 12 layers + ~11 in FA block × 18 layers + MoE allocs ≈ 300+/token. | M | +3-8% | **Top-3 #3.** Allocator may issue implicit syncs depending on impl. Also a **prerequisite for graph capture** — hipGraph cannot capture allocs. Pair with lever F32→FP16 (smaller arena). |
| Cross-cutting: memcpy_dtod for Q/K/V split | hipfire issues 3× `memcpy_dtod_at` to split conv_out into Q/K/V (`qwen35.rs:3088-3090`); hipEngine fuses the split into the GDN kernel via base-offset pointer math | The GDN mega-kernel reads conv_out at offsets `q_base`, `k_base`, `value_offset` (`gdn.hip:88-90`) — no copy needed. | `gdn.hip:88-90, 172` | `gpu.hip.memcpy_dtod_at(&q_part.buf, ...)` × 3 per LA layer per token (`qwen35.rs:3088-3090, 3120-3121`) | XS | +1-2% | 12 LA layers × 3 memcpy = 36 needless memcpys/token. Trivial fix when GDN merge happens; also doable independently by passing offsets. |
| Cross-cutting: position_buf | hipfire writes the position scalar via `memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())` (`qwen35.rs:3035-3037`) per step | hipEngine has `position_buf` and `context_buf` as persistent device i64 scalars updated by `set_decode_position_i64` / `advance_decode_position_i64` (`runner.py:3183-3201`). | `runner.py:3183-3201` | `qwen35.rs:3035-3037` H2D per step | XS | +0.5-1% | Eliminates the per-step H2D copy + implicit sync. Tied to graph capture prerequisites. |
| Cross-cutting: launch-bounds re-tune | hipEngine kernels honor `__launch_bounds__(128, 4)` on pack8/selected and `(32, *)` on compact-WMMA/fusedW4; thread-count knobs cannot bypass these (D4.4 audit at `OPTIMIZE.md:318`) | All hot kernels have explicit launch bounds; threads parameters at the wrapper level are clamped. | unknown for hipfire — needs audit per `gfx-kernel-metadata` skill | XS-S | +0-2% | Per LESSONS-LEARNED:904-918: runtime thread-count knobs must honor launch_bounds, or you trigger spills. Worth a one-time audit pass on hipfire's MoE/AWQ/Marlin-K-equivalent decode kernels. |

## Phase coverage check

| Phase | Covered |
|---|---|
| Embedding gather | yes (2 levers) |
| Input layernorm | yes |
| LinearAttention QKV-Z projection | yes (2 levers) |
| LinearAttention conv1d (decode) | yes |
| LinearAttention QK norm + scale | yes (in GDN mega) |
| LinearAttention beta/alpha gating | yes (in GDN mega) |
| Gated DeltaNet hidden-state update | yes (GDN mega-kernel) |
| LinearAttention output norm + gate + projection | yes (in GDN mega + out-proj) |
| FullAttention QKV projection | yes (2 levers) |
| FullAttention RoPE | yes (head-RMSNorm+RoPE mega) |
| FullAttention KV cache write | yes (already implemented note) |
| FullAttention flash attention decode | yes (split-K GQA) |
| FullAttention output gate | yes (gate-fused decode) |
| FullAttention output projection | yes (2 levers) |
| Post-attention layernorm | yes |
| MoE router GEMV | yes |
| MoE top-k + softmax + weight normalization | yes (already implemented note) |
| MoE shared-expert gate | yes |
| MoE shared-expert gate_proj/up_proj/down_proj | yes (2 levers) |
| MoE routed-expert dispatch | yes (already implemented note) |
| MoE routed-expert gate+up GEMV | yes (already implemented note) |
| MoE routed-expert SiLU+mul | yes (dual-rotate) |
| MoE routed-expert down_proj | yes (already implemented note) |
| MoE residual accumulation | yes |
| LM head GEMV | yes |
| Sampling | yes (2 levers) |
| hipGraph capture + replay | yes (3 levers) |
| KV cache layout | yes (3 levers) |
| Speculative decoding / MTP | yes (both ruled out) |

## Citations table

Key hipEngine files referenced:

- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip:54-245` — GDN mega-kernel
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/moe/router.hip:361-509` — router cooperative producer
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/runtime/state.py:22-26` — device position/token primitives
- `/tmp/hipengine-survey/docs/OPTIMIZE.md:248-273` — decode Amdahl table (877 dispatches/token; bucket shares)
- `/tmp/hipengine-survey/docs/KVCACHE.md:124-158` — INT8 KV layout (already implemented on hipfire)
- `/tmp/hipengine-survey/docs/SPECULATIVE-DECODE.md:283-417` — MoE speculation impossibility (η<1)
- `/tmp/hipengine-survey/docs/LESSONS-LEARNED.md:967-981` — tiny c=1 projections are launch-bound + fusion pitfalls
- `/tmp/hipengine-survey/docs/LESSONS-LEARNED.md:991-996` — graph replay state contract (device tokens, positions)

Key hipfire files referenced:

- `crates/hipfire-arch-qwen35/src/qwen35.rs:2820-2902` — MoE decode hot path (router + topk + experts)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:2981-3019` — `forward` (CPU-side argmax + logits download)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3048-3263` — full decode forward (per-layer dispatch chain)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:3765-3825` — MoE graph capture disabled (drift bug)
- `crates/rdna-compute/src/dispatch.rs:18133-18200` — argmax_f32 / argmax_f32_batched GPU kernels (built but unused on decode)
- `kernels/src/gated_delta_net_q8.hip` — GDN decode (not fused with norm/gate)
