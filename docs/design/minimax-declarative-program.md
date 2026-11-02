# MiniMax-M2 declarative decode program (#666 G7)

Single-lane MiniMax decode (`hipfire_arch_minimax::forward::decode_step_body`,
shared by eager decode and the opt-in hipGraph route) runs each layer as one
typed `hipfire_dispatch` step program. The architecture crate binds weights,
KV state and scratch (`crates/hipfire-arch-minimax/src/program.rs`); route
selection, fusion and executor bodies live in dispatch.

## Layer program

```
RmsnormAutomatic(attn_norm) → Gemv q, k, v (prerotated → fused Q8 QKV)
RmsnormAutomatic(q, rows=1, k=q_dim) → RmsnormAutomatic(k, rows=1, k=kv_dim)
Rope { kind: PrefixHalfSplit { rot_dim: rotary_dim } }
Attend (Q8 KV write + attention, KvTierPlan)
GemvResidual(o_proj)
Moe (sealed: SigmoidRoutedNoShared + selection bias)
```

then `RmsnormAutomatic(final_norm)` → `Gemv(lm_head)`.

| MiniMax semantic | Where it is enforced |
|---|---|
| QK-norm over the whole projection, before the head split | `Step::RmsnormAutomatic` in place, `rows = 1` (row-complete; `QkNorm` is head-local) |
| rotate-half RoPE on the first `rotary_dim` channels only | `RopeKind::PrefixHalfSplit`; refuses `HIPFIRE_ROPE_INTERLEAVED_LEGACY` (that probe silently switches the launcher to the interleaved pairing) |
| sigmoid router, select on `score + e_score_correction_bias`, weight on the unbiased score, normalize, no routed scale | `MoeParams::selection_bias` with `route_scale = 1.0`, `norm_topk_prob = true` |
| no shared expert | `MoeRecipe::SigmoidRoutedNoShared` |
| shared per-layer gate/up AWQ scale on the routed experts | the no-shared input basis rotates with the routed gate/up `awq_scale` (`rotate_x_mq_awq`); the opt-in down scale (`HIPFIRE_MINIMAX_ENABLE_DOWN_AWQ`) feeds `fused_silu_mul_rotate_mq_awq_batched` |
| packed expert blobs (one gate/up and one down allocation per layer) | `SealedExperts`: one byte view per expert, sealed and live-bound at load against the uploaded pointer tables |

The sealed MoE admits the biased router at top-8 for every indexed format
(MQ4/MQ6, MQ2-Lloyd, MQ3-Lloyd); below top-8 it stays MQ4/MQ6-only.

## Not cut over

- Batched prefill / verify (`forward_batch`) and the n-gram spec verify still
  issue kernels by hand.
- EP: `forward_ep` runs the Single attention step program per rank; its MoE is
  the EP-only hand block that accumulates into the zeroed partial. The sealed
  MoE has no EP sigmoid route yet (G14).
- Single decode needs sealed experts, so it refuses an EP-shard weight set.

## Parity

Every launch of the old lowered route is kept; the only change in order is the
routed-expert input rotation, which now runs after the router GEMV and sigmoid
(independent buffers). On `MiniMax-M2.7.mq2` token ids, every logit's bits,
the K/V cache and `n_tokens` match beta at 174 positions, eager and hipGraph.
The MQ2-/MQ3-Lloyd MoE down GEMV accumulates with atomics: on a tiny 512-wide
random-weight fixture beta itself differs run to run in the last logit bits,
so small-fixture parity on those tiers is token identity plus beta's own
spread. Evidence: #666 G7 hand-off.

## COR-007

`minimax_eight_cycle_unload_drift` (`crates/hipfire-generate/tests/dense_rollback_gpu.rs`,
ignored) runs eight load / generate / reset / unload cycles on
`MiniMax-M2.7.mq2` and fails if free VRAM after unload drifts more than
64 MiB from cycle 1 or falls on every cycle.
