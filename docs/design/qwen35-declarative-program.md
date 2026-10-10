# Qwen3.5 declarative layer program

Status: landed (#795, carried by #813 on `beta`).

## Goal

Qwen3.5 declares each decoder layer as one typed `Step` list, as Qwen4 does
(`hipfire-arch-qwen4/src/gpu_forward.rs`). The dispatch crate owns every
executor body and every arch- or dtype-gated route choice. Decode runs the
program with `rows == 1`. Batched prefill and speculative verify run the
`pipeline::batched*` executors; of the decode steps only `SwigluFfn` has a
`rows > 1` arm so far.

The architecture crate keeps:

- weight and state ownership;
- the per-layer program shape (which ops, in which order, bound to which
  tensors);
- the surrounding lifecycle: hipGraph, Redline, DFlash capture and the chunk loop.

## Vocabulary

This port uses typed `Step` only. The #397 super-op path (`lower_variant` plus
`Qwen35Bindings`) no longer serves single-GPU decode; the EP decode hooks
(`ep_*`) keep it until the EP schedule consumes the same program.

New ops are model-neutral. They live in `hipfire-dispatch/src/pipeline/hybrid.rs`:

| Step | Stages (replace qwen35's super-op handlers) | Route choice owned by dispatch |
|---|---|---|
| `DeltaNetMixer` | `project` → `prepare` (`ATTEND_DN_PREP`) → `recur` (`RECUR_GDN`) → `gated_norm` (`NORM_GATED`) → `output` | conv+qk-norm fusion, gfx1100 scalar prep, compact QK; FP32 / Q8 / Q8-compact / Q4 state; fused gated-norm + MQ rotate |
| `GatedAttention` | `project` → `attend` (`ATTEND_FULL`) → `output` | fused FA prep, TriAttention tap, compaction offset, fused epilogue |
| `SwigluFfn` | `project` → `down` (`RESID_DOWN_SWIGLU`) | GEMV family SwiGLU-residual variant; `rows > 1` batched FFN |

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

## Rebased onto beta

Rebased onto `beta` `e268a0798` with #813. The code beta changed after the
port's merge base now lives in the dispatch executors: the packed-MQ4 FFN
routes and the gfx1151 A4 gate/up epilogue (`FfnGateOutput::Iu4A4`), the
widened `s4_residual_fast` gate, `fa2_gfx11_ctx_admitted` and the gfx1151
multirow assert in the attend step, the MQ6/HFQ6 fused qkv/qkvza keys, the
`memory.offload_exec=cpu` down projection, and `i_gpu_start` in the MTP MoE
config. `swiglu_down_residual` again runs the rotated formats it had dropped
(MFP4, MQ8, MQ4G128, ParoQ4G128, ...) as GEMV plus add; MFP4 is measured
below, the others are not.

Against the beta daemon (gfx1151, `serve_harness.py --thinking off --sampling
greedy --compare-transcript`), every row is `transcript_byte_identical=true`:
`qwen35-4b.mq4` battery, chain and a 2k-token prompt;
`qwen3.8-27b.mq4-xts` 2k-token prompt (MQ4V2 + AWQ, A4 epilogue route);
`ornith-1.5-35b-a3b.mq4r` AR and MTP; `qwen3.6-35b-a3b.mq4r` MTP;
`qwen3.5-9b.mq4` DFlash and a 40k-token prompt (wide FA2 prefill past 32K).
Greedy daemon runs: an MFP4 `Qwen3.5-4B` (MFP4G32 `w_down`) matches beta byte
for byte where the pre-rebase head failed with `unsupported
hybrid.swiglu_down_residual`; `qwen3.5-9b.mq4` with 20 resident layers and
`offload_exec=cpu` matches beta, with every host-mapped step on the CPU (0
left on the GPU). `qwen35-4b.mq4` Redline `--pm4` shadow exact on both (426
dispatches).

### Rebased onto beta `f4ea8109b`

Beta's exact wide verify chunks (`DenseBatchMath`, `HIPFIRE_CB_VERIFY_CHUNK128`)
edited the projection and FFN helpers this port moves. `DenseBatchMath` and
its MQ4G256V2 guard now live in `pipeline::batched`. `SingletonWmma` runs in
`dispatch_batched_gemm_epilogue`, the batched SwiGLU FFN
(`SwigluFfnBatch::math`), the attention projections and the DeltaNet
projections, with the same explicit `*VerifyExact` keys and the same refusals.
The qwen35 helpers re-export the type. The multi-request verify chunk passes
its per-request `ChainVerify` context (`with_verify_tile_attend`) to the moved
prepare and attend steps.

Against the beta daemon on gfx1151 (`serve_harness.py --thinking off
--sampling greedy --compare-transcript`), every row is
`transcript_byte_identical=true`: `qwen35-4b.mq4` battery, chain and 2k-token
prompt; `qwen3.5-9b.mq4` AR and `--dflash on` battery and chain (DFlash does
not engage on either arm); `qwen3.8-27b.mq4-xt` AR, 2k-token prompt, MTP and
DFlash (same tau per turn); `qwen3.6-35b-a3b.mq4r` and
`ornith-1.5-35b-a3b.mq4r` AR and MTP (same tau per turn). Daemon-protocol
greedy text is identical for `Qwen3.5-35B-A3B-PARO` with and without
`HIPFIRE_PARO_BATCHED=1`, and for one 27B vision request (mrope). Redline
`--kv-mode q8 --pm4` is shadow exact on both arms with the same decode tape:
`qwen35-4b.mq4` 426 launches (`a021b8f179688248`), `ornith-1.5-35b-a3b.mq4r`
843 launches (`ea6689ccdda27a05`). The `SingletonWmma` arms run only on
gfx1201 and were not exercised here.
