# Inkling-2-Mini GPU kernel inventory (gfx1100 first)

Spec only — no code. Derives operator semantics from
`docs/plans/2026-11-20-inkling2-mini-support.md` §0a/§3 (the
`crates/hipfire-arch-inkling2/src/forward.rs` CPU reference does not exist
in this checkout — the arch crate itself is absent — so there is no local
forward.rs to cite; Phase-0 oracle is the authority before build).
Geometry: 24 phys layers, core 6–18 ×3 passes = 48 visits; 24Q/8KV hd128;
SWA-1024 5-of-6 + full; d_rel 16, rel_extent 1024; clips 22.627/8.0;
tau `1+0.1·log(max(1,(pos+1)/8192))`; sconv k4; MoE 128/top-8/1152
sigmoid, route_scale 8, SiTU(4,25), eps 1e-20; final softcap 30;
controller rank 64, eps 1e-6.

Reused kernels are gfx1100-validated paths; gfx1151 usually identical
(unified-memory residency, not new kernels); gfx1201 notes per row.

## Summary table

| # | Operator | Verdict | Reuse source |
|---|---|---|---|
| 1 | Visit-schedule dispatch + per-visit KV/conv index | NEW (host table + index math) | state containers mirror qwen35 per-layer state; no kernel |
| 2 | Controller fuse (concat-RMSNorm → down-64 → tanh gain/forget blend, 4×/core-layer) | NEW fuse; pieces reused | `rmsnorm_batched` (`crates/rdna-compute/src/norm.rs:420`); GEMM pieces from MoE/dense GEMV paths |
| 3 | `mix()` softmax over block outputs + cur, learned query | NEW small kernel; host-composed in Phase 1 | row-softmax pattern `attention_rows_softmax_f16` (`crates/rdna-compute/src/attention.rs:3582`); compressor pool `compressor_softmax_pool_f32` (`:18680`) as shape reference |
| 4 | Rel-bias gather (R proj → n_rel/head × `[n_rel,n_extent]` table, sink-subtract pre-gather, distance gather) | NEW | no hipfire precedent (T5 bucketing is FLUX-encoder-only); DS4 sink-subtract idiom only as pattern (`kernels/src/deepseek4_attn_swa.hip:5,87,124`) |
| 5 | Q/K+rel L2 clip (per-channel scale, row-L2, factor ∈[0,1], eps 1e-6) | NEW elementwise | pattern-match `logit_softcap_f32` (`crates/rdna-compute/src/norm.rs:8898`); fused-QK-norm precedent `fused_qk_l2_norm_scale_f32` (`crates/rdna-compute/src/gemm.rs:29761`) |
| 6 | Value RMSNorm (no gamma) + full-layer Q/K input norms | REUSE | `rmsnorm_batched` (`norm.rs:420`) + glimmer ones-weight trick (`crates/hipfire-arch-muse-glimmer/src/forward.rs:13-14,263`) |
| 7 | Dynamic K/V conv (k4 via weight_proj + static_w) + attn/FFN-out short convs over unified recurrent state | ADAPT | `conv1d_silu_split_f32` (`norm.rs:5461`), `dynamic_causal_conv_f32` (`norm.rs:9211`), `dynamic_conv_f32` (`norm.rs:9470`), lane variant `dynamic_causal_conv_f32_lanes` (`crates/rdna-compute/src/draft_lane_conv.rs:84`); qwen35 GDN conv precedent |
| 8 | Tau scale (×Q,R on full layers) | REUSE (fold into prologue, no kernel) | scalar multiply inside attention prologue; no cite needed |
| 9 | Sliding-1024 + sink-softmax attention, NoPE | REUSE + small adapt | `attention_dflash_sliding_f32` (`attention.rs:16769`); sink pattern `deepseek4_attn_swa.hip:87,124`; NoPE precedent cohere2moe full layers (`crates/hipfire-arch-cohere2moe/src/forward.rs:174,188,589`) |
| 10 | Output-gate sigmoid multiply | REUSE (elementwise) | `hc_pre_post_sigmoid_scale_f32` (`attention.rs:20349`) or host-composed; trivial |
| 11 | MoE sigmoid top-8, no renorm, ×route_scale 8, eps 1e-20 | REUSE routing; NEW combine-scale detail | cohere2moe routing (sigmoid, no renorm: `forward.rs:18`, `config.rs:89`); sealed/prefill topk `moe_topk_renorm_k8` (`crates/rdna-compute/src/gemv.rs:15243`), batched (`:16572`); dispatch `MoeFamily` route_scale/norm_topk_prob (`crates/hipfire-dispatch/src/families/moe.rs:264,841`) |
| 12 | SiTU(4,25) experts, 128×1152, gate-bias | REUSE (already in tree) | `FfnActivation::Situ` (`crates/rdna-compute/src/ffn_activation.rs:77,350`), rows/grouped entries (`:369-381`), dispatch (`:412,544`), scalar oracle (`:1389-1416`); maple clamped-SiLU is the nearest *pattern*, not the impl |
| 13 | Final softcap 30 | REUSE | `logit_softcap_f32` (`norm.rs:8898`; gemma4 precedent = same fn) |

gfx1151: all REUSE rows carry over unchanged; NEW rows need no new code
either (same HIP kernels, APU residency/headroom only). gfx1201: routing
and SiTU have gfx1201 rotate-i8 variants (`ffn_*_rotate_i8_gfx1201`,
`ffn_activation.rs:385-397`); conv/QK-norm has a gfx1201 kernel
(`kernels/src/conv1d_silu_split_qknorm.gfx1201.hip:12`); attention stays
the F32 sliding path until a served A/B says otherwise.

## Per-kernel sections

### 1. Visit-schedule dispatch + per-visit KV/conv state index — NEW
- Spec: `VISIT_SCHEDULE: [u8; 48]` built from (core_start 6, core_end 18,
  passes 3); `layer_phys(il)`, `loop_visit(visit)→pass`; validation =
  upstream rules (fixed ≥2 passes, `0<start<end≤24`, block-aligned to 6,
  rank ≥1, n_slot 48). KV, short-conv state, controller scratch,
  prelude/bank buffers sized ×48 visits; weights stored ×24.
- Shapes/dtype: index math is u8/u32 host-side; state buffers follow the
  existing per-layer KV/conv layouts, no new dtype.
- Numerics: exact integer mapping; mismatch = load refusal.
- Phase-1 fallback: host-side table + per-visit buffer select in the
  forward loop. No kernel.

### 2. Controller fuse — NEW fuse, reused pieces
- Spec: per core layer ×4 (Q,K,V,FFN inputs): `g=RMSNorm(cur)‖RMSNorm(msg)`
  eps 1e-6 → `down [2·n_embd,64]`, `gain/forget [64,n_embd]` →
  `out=cur−tanh(F·g)·cur+tanh(G·g)·msg`. SWA controls all three attn
  inputs then per-input norm; FFN message applied at MoE input.
- Shapes: concat `[2·n_embd]`; down-out `[64]`; gain/forget-out `[n_embd]`;
  F32 compute (keep-listed, never requanted).
- Numerics: eps 1e-6 both norms; tanh in F32; bitwise vs oracle is the gate.
- Reuse: `rmsnorm_batched` (`norm.rs:420`); GEMM via existing dense/MoE
  GEMV paths. NEW is only the fused concat→norm→down→tanh-blend.
- Phase-1 fallback: three unfused calls (2× rmsnorm + 3× GEMV + host/blend
  elementwise). Fuse in Phase 4 behind the KLD pin.

### 3. `mix()` — NEW, host-composed in Phase 1
- Spec: `concat([blocks…,cur])` → RMSNorm 1e-6 → scores at `1/sqrt(n_embd)`
  with learned query → weighted sum. Bank = previous pass's
  `blocks[prelude_len+1..]` + its output; prelude restored each pass
  (skip-first-core-block detail).
- Shapes: `[n_bank+1, n_embd]` F32, one row out.
- Numerics: eps 1e-6; softmax in F32.
- Reuse: row-softmax shape reference only (`attention.rs:3582`,
  `:18680`). No direct reuse — bank/restore semantics are new.
- Phase-1 fallback: host loop over rows + existing softmax; fuse later.

### 4. Relative-bias gather — NEW
- Spec: R projected to `n_rel=16` dims/head (`attn_rel [n_embd,16·n_head]`);
  scores += `[16,1024]` table (`attn_rel_bias`) gathered by distance;
  **sink logit subtracted before gather**; then distance `get_rows`.
  Validate `loop_msg_query [n_embd,4]` packing `[q2i,kv2i,kv2i+1,q2i+1]`
  at ingest, fail closed.
- Shapes: per head `[16]`, table `[16,1024]` F32; distances int.
- Numerics: F32 gather-add; order (sink-subtract → gather) is load-bearing.
- Reuse: none structural; DS4 files are the sink-subtract *idiom*
  reference (`deepseek4_attn_swa.hip:5,87,124`). NEW kernel.
- Phase-1 fallback: host/naive gather (correctness first, fuse Phase 4).

### 5. Q/K + relative L2 clip — NEW elementwise
- Spec: optional per-channel scale → row L2 → factor `threshold/norm`
  clamped [0,1] → ×row. Q clips with `attn_q_clip` scale, K scaleless, R
  with `attn_rel_clip` scale; eps 1e-6.
- Shapes: Q/K `[n_heads·128]`, R `[n_heads·16]`; F32.
- Numerics: eps 1e-6; clamp bounds exact; NaN in = NaN out (fail closed).
- Reuse: pattern-match `logit_softcap_f32` (`norm.rs:8898`); fused
  precedent `fused_qk_l2_norm_scale_f32` (`gemm.rs:29761`). Fuse with §6
  norms in Phase 4.
- Phase-1 fallback: unfused elementwise kernel or host loop.

### 6. Value RMSNorm + full-layer Q/K input norms — REUSE
- Spec: value RMSNorm with no gamma (ones weight), eps 1e-6; full layers
  additionally norm Q/K inputs.
- Shapes: Q/K/V rows as in §5; F32.
- Numerics: eps 1e-6.
- Reuse: `rmsnorm_batched` (`norm.rs:420`) verbatim; scale-less via
  ones-weight exactly as glimmer (`muse-glimmer/src/forward.rs:13-14,263`).
- Phase-1: already GPU. No fallback needed.

### 7. Dynamic K/V conv + attn/FFN-out short convs — ADAPT
- Spec: one unified recurrent state width
  `max(2·n_embd_k_gqa + 2·n_embd)` holding K, V, attn-out, FFN-out windows
  at offsets `0, n_embd_k_gqa, 2·n_embd_k_gqa, 2·n_embd_k_gqa+n_embd`;
  SWA = static k/v convs; full layers = input-dependent taps
  (k=4 via `k/v_conv_proj [n_embd, n_conv·n_head_kv]` + `static_w`);
  post-conv scale `1/sqrt((dynamic?2:1)·n_conv+1)`; `out = cur + conv`.
- Shapes: kernel width 4 over the unified state; F32; per-visit state (§1).
- Numerics: F32; post-scale exact; static weights are
  `flip(0).T`-converted at ingest (plan §8).
- Reuse: `conv1d_silu_split_f32` (`norm.rs:5461`),
  `dynamic_causal_conv_f32` (`norm.rs:9211`), `dynamic_conv_f32`
  (`norm.rs:9470`), lane variant (`draft_lane_conv.rs:84`). Adapt =
  no-SiLU/static-or-dynamic-tap variant over the unified layout + per-visit
  indexing. gfx1201: qknorm-b128 path exists if later fused.
- Phase-1 fallback: naive per-tap loop calling the existing dynamic-conv
  kernel with Inkling weights; fuse in Phase 4.

### 8. Tau log-scale — REUSE (fold-in, no kernel)
- Spec: `tau=1+0.1·log(max(1,(pos+1)/8192))`, multiplies Q and R on full
  layers only.
- Phase-1: scalar multiply in the attention prologue. No kernel, no fallback.

### 9. Sliding-1024 + sink-softmax attention, NoPE — REUSE + adapt
- Spec: 5 sliding (window 1024) + 1 full per 6-block; NoPE everywhere;
  learned sink participates in softmax; learned output gate (§10).
- Shapes: 24Q/8KV hd128 F32; KV per-visit (§1), 48 deep.
- Numerics: F32 softmax with sink term; sliding window exact.
- Reuse: `attention_dflash_sliding_f32` (`attention.rs:16769`) for the
  windowed path; sink idiom from DS4 (`deepseek4_attn_swa.hip:87,124`);
  NoPE handling = cohere2moe full-layer precedent
  (`cohere2moe/src/forward.rs:174,188,589`). Adapt = sink + gate plumbing
  around the reused core.
- Phase-1 fallback: reused kernel as-is; full layers run the same kernel
  with window ≥ ctx (exact, slower) until a dedicated full path is fused.

### 10. Output-gate sigmoid multiply — REUSE
- Spec: elementwise `out *= sigmoid(gate)` after attention.
- Reuse: `hc_pre_post_sigmoid_scale_f32` (`attention.rs:20349`) or
  host-composed elementwise. Trivial either way.

### 11. MoE sigmoid top-8 routing, no renorm, ×8 — REUSE + scale detail
- Spec: `sigmoid(router + bias?)` → top-8 → use raw sigmoid scores (no
  renorm) → expert outputs combined × `route_scale` 8. Gate bias optional
  (`TENSOR_NOT_REQUIRED`); 128 experts; norm eps 1e-20 applies to expert
  layernorm, not the router.
- Shapes: router logits `[128]` → indices/weights `[8]`; F32.
- Numerics: sigmoid F32; top-8 exact; ×8.0 post-combine; eps 1e-20 in
  expert norm (do not "round" to 1e-6).
- Reuse: cohere2moe is the identical routing precedent (sigmoid, no
  renorm — `cohere2moe/src/forward.rs:18`, `config.rs:89`);
  `moe_topk_renorm_k8` (`gemv.rs:15243`) / batched (`:16572`) and
  `sigmoid_routed_prefill_supported` (`cohere2moe/src/forward.rs:400`)
  cover the GPU path; `route_scale`/`norm_topk_prob` are already
  dispatch concepts (`hipfire-dispatch/src/families/moe.rs:264,841`).
  NEW detail is only "no-renorm + ×8" as a config, not a kernel.
- Phase-1 fallback: existing sealed/prefill router path with
  `norm_topk_prob=false, route_scale=8`.

### 12. SiTU(4,25) experts — REUSE
- Spec: gate_up interleaved `[n_embd, 2·1152, 128]` (+optional bias),
  SiTU gate_beta 4 / value_beta 25, down proj; no shared experts.
- Reuse: `FfnActivation::Situ` + rows/grouped dispatch already in tree
  (`ffn_activation.rs:68-77,369-397,412,544`; scalar oracle `:1389-1416`).
  Maple clamped-SiLU is the nearest *pattern*, not the implementation —
  do not copy it.
- Phase-1: existing grouped-activation path with the small-expert
  geometry; prefill WMMA only with a served A/B (plan Phase 4).

### 13. Final softcap 30 — REUSE
- Spec: F32 head → softcap 30 → unpadded slice (tokenizer 200058, not
  201024; embeddings sized accordingly).
- Reuse: `logit_softcap_f32` (`norm.rs:8898`) verbatim (gemma4 precedent
  is the same function).
