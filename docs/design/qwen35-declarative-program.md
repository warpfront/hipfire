# Qwen3.5 declarative layer program

Status: in progress on `feat/qwen35-declarative` (cut from #774).

## Goal

Qwen3.5 declares each decoder layer as one typed `Step` list, as Qwen4 does
(`hipfire-arch-qwen4/src/gpu_forward.rs`). The dispatch crate owns every
executor body and every arch- or dtype-gated route choice. The same program
serves decode (`rows == 1`), batched prefill and speculative verify
(`rows > 1`). Each op picks GEMV or GEMM from `rows`.

The architecture crate keeps:

- weight and state ownership;
- the per-layer program shape (which ops, in which order, bound to which
  tensors);
- the surrounding lifecycle: hipGraph, Redline, DFlash capture and the chunk loop.

## Vocabulary

This port uses typed `Step` only. The #397 super-op path (`lower_variant` plus
`Qwen35Bindings`) is removed for single-GPU decode once the Step program is
bit-exact. The EP decode hooks (`ep_*`) keep their current binding until the EP
schedule consumes the same program.

New ops are model-neutral. They live in `hipfire-dispatch/src/pipeline/hybrid_ops.rs`:

| Step | Replaces (qwen35 today) | Route choice owned by dispatch |
|---|---|---|
| `GdnPrep` | `ATTEND_DN_PREP` | conv+qk-norm fusion, gfx1100 scalar prep, compact QK |
| `GdnRecurrence` | `RECUR_GDN` | FP32 / Q8 / Q8-compact / Q4 state |
| `GatedNorm` | `NORM_GATED` | fused gated-norm + MQ rotate |
| `GatedAttention` | `ATTEND_FULL` | fused FA prep, TriAttention tap, compaction offset, fused epilogue |
| `SwigluResidual` | `RESID_DOWN_SWIGLU` | GEMV family SwiGLU-residual variant |

Projections keep the existing `RmsnormAutomatic`, `Gemv` and `GemvResidual`
steps, which the fusion table already rewrites into qkv, qkvza or gate/up
kernels. MoE keeps `Step::Moe`.

## Slices

Each slice keeps greedy token ids and prefill logits byte-identical to the
pre-port build, on dense 4B and on the Ornith MoE fixture. Redline PM4 shadow
parity must hold. The tape hash may change only if the launch sequence changes.

1. **Decode.** The ops above with `rows == 1` executors; the arch crate builds
   the program in `qwen35/program.rs`; delete `lower_variant` and the
   single-GPU `Qwen35Bindings` bodies.
2. **Prefill.** Add `rows > 1` executors by moving the `batch_chunk_*` layer
   bodies. `forward_batch_chunk_impl` runs the slice-1 program.
3. **MTP head.** `mtp_head_forward_block_only_with_pos_buf` runs the MTP layer
   as `[GatedAttention, SwigluFfn | Moe]`, the trunk's decode steps. The MoE
   MTP experts load through the trunk MoE loader as a sealed
   `MoeFfnWeights`. The MTP head and the decode hand arms use a different
   gate (below).

## Gates per slice

- Greedy decode ids identical (serve battery and chain) on
  `qwen35-4b.mq4` and `ornith-1.5-35b-a3b.mq4r`.
- `redline_daemon_harness.py --pm4` shadow parity exact.
- EP route oracles unchanged (EP uses the shared MoE step).
- MTP head and decode hand arms: tau parity and coherent decoded text. The
  shared steps take the trunk's certified MQ4 fusions, which the hand paths
  never used. That moves draft logits in their last bits, and speculative
  decode on the A3B MoE is not AR-invariant, so the greedy text can differ
  where the verify window hits a near-tie.

## Progress

| Piece | State | Commit |
|---|---|---|
| Decode mixers + dense FFN as Steps (`program.rs`, `hybrid.rs`) | landed, bit-exact, tape hash unchanged | `ff892ed98` |
| Batched dense FFN (`rows > 1`) + `pipeline::batched` library | landed, bit-exact | `505277d01` |
| Batched full attention (`pipeline::batched_attention`) | landed, bit-exact | `9ec50733b` |
| Batched DeltaNet (`pipeline::batched_deltanet`) | landed, bit-exact | `5466e9e8c` |
| MoE layers (DN-MoE LA body, FA-MoE prep and output projection) | landed, bit-exact incl. PARO (`HIPFIRE_PARO_BATCHED=1`); separate executors because the MoE route differs in its input producers, PARO Givens arms and always-residual output projection, so merging with the dense executors would change numerics | `1f3265f3d` |
| Decode hand arms (DFlash hidden capture, VL mrope) folded into the step program; hand arms and `HIPFIRE_FORWARD_LOWERED=0` removed. `lower_variant` stays as the EP executor's super-op program | landed; 4B/Ornith AR + Ornith MTP byte-identical, Redline tape hashes unchanged; 9B DFlash battery byte-identical, chain 1 of 5 turns diverges late (tau 1.68→1.61); 27B VL (mrope, rope_delta=-272) byte-identical | `79f0122c8` |
| MTP head (`[GatedAttention, SwigluFfn \| Moe]`, sealed MoE experts) | landed; Ornith MTP battery tau per turn 1.46/1.36/1.21/0.89/1.13 vs 1.46/1.36/1.20/0.89/1.13 before; 1 of 5 turns diverges at token 3, all coherent; the sealed MoE step matches the old hand MoE decode byte for byte | `fb16c0344` |

Verification per landed slice (gfx1151): greedy serve battery/chain and a
1131-token prompt on `qwen35-4b.mq4`, greedy battery on
`ornith-1.5-35b-a3b.mq4r`, byte-identical to the pre-port daemon; emulated
EP2/EP4 route oracles exact; TP2 oracle unchanged (2.968e-1). Decode slices
also keep the Redline tape hash.

Extra fixtures after the port:

- `qwen3.6-35b-a3b.mq4r` + `qwen3.6-35b-a3b.mtp` (same architecture as Ornith).
  The AR battery on the final build is byte-identical to the pre-port daemon.
  The MTP battery is byte-identical to the build before the MTP slice, with
  the same per-turn tau (2.44/2.07/0.92/1.00/1.25).
- `qwen3.8-27b.mq4-xt` + `qwen3.8-27b.mtp` (dense MTP, `SwigluFfn` arm). The
  MTP battery (`--thinking-effort none`) is byte-identical to the build before
  the MTP slice, with tau 2.65/2.69/1.85/…. The daemon-protocol text equals AR.
