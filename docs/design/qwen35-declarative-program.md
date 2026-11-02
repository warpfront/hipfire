# Qwen3.5 declarative layer program (#666 G8)

Status: four slices are implemented on stacked branches and are in review,
not merged: slice 1 (single-GPU decode layers, HIPF-117, #915), slice 2
(batched prefill/verify bodies, HIPF-118, #917), slice 3 (MoE-layer batched
mixers, HIPF-119, #918) and slice 4 (MTP layer, DFlash capture, VL mrope,
HIPF-120, #919). They recut #795 from current `beta`. #795 is the design
reference only, and none of its bodies are cherry-picked: every slice moves
the current beta body, so beta's edits come along by construction.

## Goal

Qwen3.5 (and 3.6/3.8, same architecture) declares each decoder layer as
typed `Step`s, as Gemma 4 and Qwen4 do. `hipfire-dispatch` owns the executor
bodies and every arch-, shape- and dtype-gated route choice. The architecture
crate keeps:

- weight, recurrent-state and KV ownership;
- the per-layer program shape (`src/qwen35/program.rs`): which op, bound to
  which tensors;
- the lifecycle: hipGraph, Redline, the DFlash ring buffer and the chunk loop;
- the paths listed under [Still in the arch crate](#still-in-the-arch-crate).

## Slice 1: decode layers

Single-GPU decode (`forward_scratch_layers`) runs each layer as

    [DeltaNetMixer | GatedAttention, GatedMlp | Moe]

The mixers live in `hipfire-dispatch/src/pipeline/hybrid.rs`; the dense FFN
is the shared `GatedMlp` op on its `Product` ladder
(`hipfire-dispatch/src/pipeline/gated_mlp.rs`, HIPF-142, design note
`docs/design/gated-mlp-op.md` slice 4):

| Step | Stages | Route choices owned by dispatch |
|---|---|---|
| `DeltaNetMixer` | `project` → `prepare` → `recur` → `gated_norm` → `output` | QKVZA scalar prep, conv + QK norm, conv scalar prep, compact QK (2:1, 3:1), FP32/Q8/Q8-compact/Q4 state, fused gated norm + MQ rotate |
| `GatedAttention` | `project` → `attend` → `output` | fused FA prep, Q8 KV-write fold, TriAttention tap, compaction offset, (t, h, w) mrope, fused gate + MQ-rotate epilogue |
| `GatedMlp` (`Product`) | `project` → `down` | `Residual`: MQ SiLU·rotate + SwiGLU-residual GEMV, HFQ residual GEMV, plain GEMV + add for the rotated formats (MFP4, MQ8, MQ4G128, ParoQ4G128, …), `memory.offload_exec=cpu` split. `Write` (dense-TP partial): SiLU·mul + plain GEMV |

MoE layers call `moe_ffn_dispatch` (the sealed MoE program with its CPU
expert splice).

The same `Product` op serves the batched rows (prefill, DFlash verify, the
dense-TP prefill partial): `gated_mlp::execute_product_rows` takes it with a
`ProductRows` context (the chunk's `BatchScratch`, the WMMA flags,
`DflashFusionCtx`, `DenseBatchMath`) and runs slice 2's batched producer/GEMM
ladder, `Residual` as the residual epilogue and `Write(partial)` as the TP
`Partial` epilogue. It is the only entry to that ladder; the qwen35 binding is
`{DeltaNet,FullAttn}LayerWeights::ffn_rows`.

The executor bodies were moved from current beta's `Qwen35Bindings` super-op
handlers. Every gate exists once:

- `hybrid::{gdn_compact_qk_div, gated_norm_mq_rotate, fused_attention_prep,
  fused_attention_epilogue, qkvza_scalar_prep, conv_scalar_prep,
  conv_qknorm, gfx1151_radiowave, gfx1201_state_fusions}` read
  `HybridDims` (built by `program::hybrid_dims`).
- The qwen35 copies are deleted.
  `forward_slots/vmm/exact.rs` calls the hybrid predicates. The dense 27B
  shape predicate is `HybridDims::is_dense_27b_shape` (the widened prefill
  admission uses it too).
- `qkv/qkvza/gate_up_via_execute_steps` forward to `hybrid::normed_projection`.

The `Product` ladder's residual down has a runtime twin,
`hipfire_runtime::llama::weight_gemv_swiglu_residual`: same kernels, the
same calibration taps per token (one for MQ, two for HFQ, three for the
plain-GEMV fallback). In the fallback arm, `GemvFamily::run_auto` prepares
each rotation's own sign tables where the runtime's `weight_gemv` does it
explicitly. Its remaining callers are listed under [Still in the arch
crate](#still-in-the-arch-crate).

The decode error text changed: the step program wraps a failing launch's
error once, where the super-op executor wrapped it twice (`… (hipError=0)`
instead of `… (hipError=0) (hipError=0)`). No caller classifies errors by
that text.

## Slice 2: batched prefill and verify bodies

The dense batched (rows > 1) bodies of prefill and speculative verify moved
from `qwen35/prefill.rs` into three dispatch modules, as the rows > 1
executors of the slice-1 ops:

| Module | Executor (slice-1 op) | Stages and library |
|---|---|---|
| `pipeline::batched` | `execute_swiglu_ffn_batched` (now behind `GatedMlp` `Product` rows, `gated_mlp::execute_product_rows`) | gate/up and down stages; the IU4/A8/FP8-stream activation producers, `dispatch_batched_gemm_epilogue`, the key-selected GEMM runners (`run_*_key`), `DenseBatchMath`/`verify_dense_math` with the explicit `*VerifyExact` keys, packed-MQ4 and F1-lite routes |
| `pipeline::batched_attention` | `batch_chunk_full_attn_attn` (`GatedAttention`) | input projection, prepare (fused FA prep, TriAttention tap), attend (tile attend, ragged FP8Q packets, multirow Q8, gfx1100 split verify, A4 epilogue), output projection |
| `pipeline::batched_deltanet` | `batch_chunk_delta_net_attn` (`DeltaNetMixer`) | qkvza projection, pre-GDN (conv, gates, DFlash tape copies), sequential / chunk-scan / tree recurrence, gated norm and output projection |

The bodies moved verbatim from current beta, so every beta edit came along:
`DenseBatchMath` and the PM FP8 verify scope, `ChainVerifySplit` and the
gfx1100 split verify, IU4 verify, the FP8-stream family opt-outs, the gfx1201
lane-batched verify and the gfx1100/gfx1101 campaign gates. The two
byte-identical dense FFN bodies (DeltaNet and full-attention layers) are one
executor now; they differed only in their error text.

The arch crate binds field-compatible views in `program.rs`
(`PrefillBatchScratch::view`, `{DeltaNet,FullAttn}LayerWeights::{mixer_view,
ffn_view}`, `kv_view`, `flash_scratch`, `deltanet_state_view`,
`gdn_tape_view`) and keeps the chunk loop (`forward_batch_chunk_impl`), the
widened-prefill planner, embedding, final logits and the multi-request verify
lifecycle (`prefill_multi.rs`, which keeps its per-request contexts and calls
the moved stages directly).

Verify context: `DflashFusionCtx` (`Off`, `ChainVerify`, `ChainVerifySplit`)
and `TreeVerifyCtx`, `BatchSemantics`, `valid_lane_mask` and
`WIDENED_COMMIT_ROWS` are dispatch types now; the moved bodies take the enum,
never a bool. `StateQuant` is `hipfire_dispatch::ops::delta_net::StateQuant`.
The arch crate re-exports these at their old paths.

Every gate exists once: the moved fusion and route predicates
(`q8_multirow_attn_admitted`, `fa_pertoken_min_ctx`, `gfx12_fa_prep_admitted`,
`dflash_fa_batch_fusion_admitted`, `mq_f16_projection_fast_route`,
`s4_residual_fast`, the FP8-stream admissions, the gfx1100 split-verify
model) live in dispatch; the arch keeps one-line Qwen adapters for its public
`gfx1100_split_verify_{min_ctx,admitted}`. The fused qkv/qkvza key selectors
moved to `families::fused_qkv` beside the gate/up selector, and the runtime's
batched AWQ rotation helpers delegate to `pipeline::batched`. The beta
helpers shared by the moved dense bodies and the arch MoE bodies (the GDN
chunk-scan views/layer and the A3B PM KV codec / F5 Q-prep route) also live
once in dispatch. The TriAttention tap is one arch binding
(`program::batched_attention_tap(config).get()`, `Some` exactly when the tap
is enabled), passed to the attention executor and shared with the MoE
attention prep.

## Slice 3: MoE-layer batched mixers

The batched mixers of the MoE layers moved from `qwen35/prefill.rs` into
`pipeline::batched_moe`, verbatim from current beta:

| Executor | Body |
|---|---|
| `execute_deltanet_moe_layer_batched` | DeltaNet-MoE linear attention: norm (exact gfx1201 IU4 producer or FWHT), PARO / IU4-prepared / 6-bit / Q8 / low-bit / fused QKVZA projection, MoE chunk scan, fused GDN preamble or indexed Q/K or the legacy conv / QK-norm / repeat sequence, tree / sequential / chunked / lane-batched recurrence, gated norm and its gfx11/gfx12 quant producers, always-residual output projection; then the MoE FFN |
| `attention_moe_prep_batched` | FA-MoE norm, PARO / IU4-prepared / fused / plain / mixed-format QKV projection, then the A3B native FP8 FA2 Q prep (gfx1201), or Q/K norm, TriAttention tap and the A3B F5 PM Q-prep or RoPE |
| `attention_moe_finish_batched` | KV write + attend (or the F5 PM direct attend), sigmoid gate, always-residual output projection (incl. the exact gfx1201 rotate producer); then the MoE FFN |
| `execute_attention_moe_layer_batched` | FP8 Q-prep and F5 admission (FP8 wins), prep, finish, per-layer MoE trace span |

Current-beta integration carries #913's explicit 128-channel Paro rotation
arguments into the moved MoE executors. It also preserves #935's post-gate
QKV/alpha/beta tape copies: the DeltaNet-MoE binding forwards `tape_offset`
to dispatch so DFlash/MTP rollback reads the same rows.

The MoE-only route gates moved with them and exist once:
`HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` (exact IU4 producers),
`HIPFIRE_GDN_MOE_PRE_FUSED`, `HIPFIRE_GDN_QK_INDEXED`, the MoE GDN route
order and the chunk-scan output plane. The per-chunk MoE chunk-scan
admission stays in the chunk loop, beside the dense one, and reaches the
executor as a bool, as in slice 2.

They stay separate executors from the dense ones: the output projection
always accumulates into the residual, the PARO Givens arms rotate the
un-rotated norm in `x_norm_batch` (now a `BatchScratch` field) by each
projection's own tables, and the conv / QK-norm launch sequence differs, so
merging them would change numerics. MQ3G256 / MQ3-Lloyd projections are
refused inside the executors as well as by the arch's MoE MQ3 guard.

The MoE FFN stays an arch binding: the executors take it as a `moe_ffn`
callback, bound to `prefill_moe_ffn_body_batched_with_route_layered` (sealed
prefill MoE program, EP route, `routed_out`). The CPU expert splice for
`memory.offload_exec=cpu` runs inside that sealed program
(`MoeStage::CpuSplice`), so it comes along unchanged.

The F2 chunk pair runs the prep for both halves, its merged attend, then the
finish per half. The finish runs its own attend, so it rewrites the merged
attend's output, as on beta; the pair is admitted only for 24 query / 4 KV
heads at head_dim 256 (`fa_pair_env_admitted`), which no MoE fixture on
gfx1151 has.

## Slice 5: per-row FA fallback, F2 pair attend (HIPF-128)

- The batched prefill's full-attention fallback (FA layers the batched
  attention does not admit, e.g. a BF16 KV tier) runs the decode steps per
  row: `forward::forward_scratch_dense_layer` executes
  `[GatedAttention, GatedMlp(Product)]` from `DecodeBinding`, as
  `forward_scratch_layers` does. The hand body `run_fa_layer_body` and
  `kv_cache_attention_dispatch` are deleted.
- The F2 pair's staging and merged attend are
  `pipeline::batched_attention::{FaPairStage, FaPairHalf, FA_PAIR_ROWS,
  FA_PAIR_BATCH}`: `FaPairStage::alloc` / `attend` / `free_gpu`. The pair
  admission (`fa_pair_merge_admitted`, `fa_pair_env_admitted`) and the
  lockstep layer loop (`forward_prefill_chunk_pair`) stay with the chunk
  loop they mirror.
- Dense TP decode attention already runs through `MixerOutput::Write`
  (slice 4); there is no `dense_tp_attention_partial` left.

## Slice 4: MTP layer, DFlash capture, VL mrope

`forward_scratch_layers` is the only single-GPU decode loop. The hand arms
that served three cases are deleted, and each case now runs the step program:

- **DFlash hidden capture** (`forward_scratch_with_hidden`: the per-token
  prefill fallback and the DDTree empty-tree step). A layer's hidden row is
  written to the ring buffer once the layer is final, which on the deferred
  MoE combine (`moe_combine_next_rms_enabled`) is after the next layer's fused
  combine + norm.
- **VL mrope**: `GatedAttentionOp::mrope` (`MropeRope`: the per-token
  `(t, h, w)` phases in `pos_buf3` and the section split) rotates with
  `rope_mrope_halfsplit_f32`; the fused FA prep and the folded KV write are
  off for it.
- **CPU-spilled models** (`memory.offload_exec=cpu`): the projection and
  output GEMVs are `execute_steps` steps, so they take the CPU splice, and
  the `Product` residual down and sealed MoE keep their own CPU arms. The
  deferred MoE combine refuses a model that owns host CPU weights, because
  its prerotated QKV/QKVZA producers read the weights directly. One route
  change: where `gated_norm_mq_rotate` admits the shape (A3B), the DeltaNet
  `wo` CPU step now takes the GPU-rotated gated-norm output, where the hand
  arm rotated on the CPU. Greedy text was identical on gfx1151, but
  `offload_exec=cpu` output can move in the last bits.

`HIPFIRE_FORWARD_LOWERED` no longer affects Qwen3.5. The exact VMM slot route
drops its refusal for it, and `scripts/forward-lowered-parity.sh` drops its
Qwen3.5 rows. The variable keeps its `env-vars.md` row because llama, qwen2
and deepseek4 still read it.

The MTP head (`mtp_head_forward_block_only_with_pos_buf`, every per-token
draft and verify-tail caller) runs its decoder layer as the trunk's steps:

    [GatedAttention, GatedMlp(Product) | Moe]

- The attention reads the head's private KV cache. Its slot and launch shape
  come from `seq_len_hint - 1` and the RoPE / KV-write position from
  `pos_buf`, so captured proposal graphs replay at any position up to the
  hint. The tier takes the trunk's flash policy.
- MoE heads load their experts through the trunk MoE loader
  (`qwen35::load::load_moe_ffn` under `MTP_MOE_PREFIX`; the sidecar tensor
  names map in `qwen35_tensor_name_candidates`) as a sealed `MoeFfnWeights`,
  and run the trunk's sealed decode recipe (`moe_ffn_decode_rmsnorm`). The
  hand MoE loader, weight types and decode body are deleted. The qt53
  (MQ4G128V2) routed-down admission stays MTP-side (`validate_mtp_qt53_layout`
  before upload, `mtp_head_serial_route_check` per call, the batched-forward
  refusal); the sealed decode runs the qt53 down through its FWHT128
  fallback (`silu_mul_f32` + `rotate_x_mq_128_v2`), as the hand body did. No
  qt53 MTP head exists on gfx1151, so this route is not exercised on hardware.
- The shared steps take whatever trunk fusions their gates admit for the
  head's shape (fused QKV / gate-up, residual GEMV; on gfx1151 the gates
  admit the A3B fused FA prep and flash epilogue gate by code reading, not
  proven by a launch census). On gfx1151 the dense 27B-xt, A3B and Ornith MTP
  battery transcripts and per-turn tau stay byte-identical to the hand body.
  Where a fusion is not bit-exact to its unfused chain, draft logits move in
  their last bits and greedy speculative text can differ where a verify
  window hits a near-tie; the gate is tau parity per turn.

`lower_variant` and `Qwen35Bindings` stay, because two executors still
interpret the coarse super-op program: the EP decode executor (`forward_ep`
→ `run_layer_program_ep`, which needs explicit all-reduce points around the
MoE) and the gfx1201 exact VMM slot route (which batches each projection
across requests). Their handlers delegate to the hybrid stages.

## Still in the arch crate

- The MTP NextN prologue (embedding, `enorm`/`hnorm`, concat, `eh_proj`) and
  the batched MTP forward (`mtp_head_forward_block_batched`, with beta's fill
  changes `2053debf5`..`289dfb64f`: pitched D2D concat, KV-only fill, shared
  K/V rotation, the exact PM GEMM). Its callers are the prompt fill
  (`kv_only=true`) and `mtp_compose.rs` (`kv_only=false`).
- The PP decode loop (`ep_batch::forward_scratch_layers_multi`, multi-GPU pp
  migration #58). Dense TP decode mixer and FFN partials run through the
  dispatch `MixerOutput::Write` and `GatedMlp` `Write` arms.
- The gfx1201 exact-VMM row twins (`forward_slots/vmm/exact.rs`), which must
  change in lockstep with `DeltaNetMixerOp::prepare` and
  `GatedAttentionOp::attend`.
- The F2 pair's lockstep layer loop (`forward_prefill_chunk_pair`) and its
  admission, beside the single-chunk loop `forward_batch_chunk_impl`.
- `hipfire_runtime::llama::weight_gemv_swiglu_residual`, the twin of
  the `Product` ladder's residual down. Its only caller is the PP decode
  loop; it goes with it.

## Blast radius outside Qwen3.5

Code other architectures run now lives in dispatch:

- `hipfire_runtime::llama::{fused_rmsnorm_rotate_mq_batched_for,
  rotate_x_mq_batched_for, fused_rmsnorm_rotate_mq_f16_batched_for,
  fused_silu_mul_rotate_mq_batched_for,
  fused_silu_mul_rotate_mq_f16_batched_for}` forward to the same-named
  functions in `pipeline::batched`, and the runtime's MQ4G128V2 refusal
  (`reject_mq4g128v2`) to `pipeline::batched::reject_typed_only_format`.
  Their callers include LFM2, MiniMax, Muse-Glimmer and the runtime DFlash
  drafter.
- `GatedAttentionOp` gained the `mrope` operand; Qwen3.5 is its only user.

## Gates

Each slice byte-compares `mq4_prefill_state_oracle` dumps (prefill logits,
Q8 KV, GDN state + EF, conv state, every decode step's logits) against its
beta base, on `qwen3.5-9b.mq4`, `qwen3.8-27b.mq4-xt` and
`qwen3.6-35b-a3b.mq4r`, with at least two chunk boundaries plus an odd tail.
It also needs greedy serve A/B rows for the routes it touches, and Redline
`--kv-mode q8 --pm4` shadow parity with an unchanged decode tape hash.
Evidence: HIPF-117 (slice 1), HIPF-118 (slice 2; also the
`prefill_ragged_oracle` hashes. `cb_vmm_state_oracle` does not run on gfx1151:
its k=1 executor hits a nonfinite logit on beta too, so it needs a gfx1201
receipt). The gfx1100/gfx1201-only routes of slice 2 (gfx1100
`ChainVerifySplit`, gfx1201 lane-batched CB verify, PM FP8 verify scope, IU4
verify, FP8-stream producers, gfx12 FA prep, A3B PM/F5, A3B native FP8 FA2
prefill) are token-identical moves (Loki `fnaudit.py`: 57 token-identical fns,
76 differing only in bindings/renames; the beta `483e614b2`/`93b67256b` FP8
port is audited the same way in HIPF-118); their hardware receipts are
outstanding: a hiptrx runbook (`cb_vmm_state_oracle` k=1,2,3, a 27B DFlash
serve A/B, and an A3B native FP8 prefill A/B beta vs stack) and a hipx row
(27B DFlash with split verify). HIPF-119 (slice 3) adds the A3B/Ornith
oracle rows, the sealed MoE oracle, the emulated EP route oracles, A3B MTP
serve and PARO (Qwen3.5-35B-A3B-PARO stays per-token: its shared expert is
F32; shisa-Qwen3.6-35B-A3B-PARO reaches the moved PARO arms). Its gfx1201-only
routes (exact IU4 producers, fused GDN preamble, indexed Q/K, MoE chunk scan,
A3B F5, the A3B native FP8 FA2 Q prep of the MoE attention) and the Q8 / 6-bit /
low-bit / mixed-format attention arms have no gfx1151 fixture: the move is
checked token by token against beta (`move-audit.diff`), not on hardware; the
A3B native FP8 prefill A/B above covers the MoE attention too.

Slice 4 (HIPF-120) adds: `dump_qwen35_hidden_states` (every layer's captured
hidden row through `forward_scratch_with_hidden`) byte-compared against the
hand arms on 9B, 27B-xt and A3B, plus 16 4B formats in Loki's review; MTP
battery tau per turn on dense 27B-xt, A3B and Ornith (also the Q8 27B
sidecar, A3B KV q8/fwht2/fwht4/asym3 and the captured proposal graph); 9B and
27B-xt DFlash battery and chain; 27B VL image requests; and
`memory.offload_exec=cpu` greedy rows on 9B, A3B and 27B-xt. Not on hardware:
gfx1100/gfx1201 (including the gfx1100 27B fused FA prep / epilogue the MTP
head may now take, covered only by the parity tests) and qt53 MTP heads.

## Dense tensor parallelism on the generic engine (#927 item 4, HIPF-142)

`qwen35::mesh_tp` declares one step program for prefill and decode. Per
layer: the mixer (`SplitRule::HeadMixer`, declared input projections
`HeadSharded`, output projection `RowShard { axis: 1 }`) writing `mix_out`,
a residual `Accumulate`, `GatedMlp` (`SplitRule::GatedMlp`, `Product` ladder,
gate/up `ColumnShard { axis: 0 }`, down `RowShard { axis: 1 }`) writing
`ffn_out`, and a second `Accumulate`. On a pure TP mesh the splitter turns
each `Accumulate` into one rank-ordered `AllReduceAdd` into `x`, so a rank
runs `[embed, mixer] AllReduceAdd [ffn] AllReduceAdd [mixer] … [head]` —
the hand dense-TP route's two reductions per layer.

Each rank holds an even contiguous block of the query, KV, DeltaNet key and
value heads and of the FFN intermediate (`tp_rank_layouts`:
`dense_tp_rank_layouts` admitted only for even splits, loaded through
`load_weights_dense_tp_rank`). A decode epoch binds the decode scratch and
runs the mixers' `MixerOutput::Write` arm; a prefill epoch binds the rank's
batch scratch and runs the batched mixers and `Product` rows with the
`Partial` epilogue (last-row logits). `Qwen35TpBindings` is tied to the plan
of exactly those shards and refuses a prefill past the batch scratch,
sessions, checkpoints and prefix repair.

### Why one compound mixer rule

`SplitRule::HeadMixer` mirrors `GatedMlp`: whole input, head-split input
projections and state, row-split output projection, Partial output reduced
by its consumer. The alternative, `HeadLocal` plus a separate row-split
`Project` for the output projection, needs the mixer's attention output as a
graph resource between two ops. The hybrid mixers fuse that boundary
(fused gate + MQ-rotate attention epilogue feeding a prerotated GEMV,
gated norm + MQ rotate in DeltaNet), and `HeadLocal` has no partial output.
Qwen4's attention and GDN blocks are the intended second consumer; they are
still `Opaque` today.

### The `MixerOutput::Write` arm

`DeltaNetMixerOp` and `GatedAttentionOp` take `output: MixerOutput`.
`Residual` is the fused single-device body. `Write(out)` is the former hand
`dense_tp_{deltanet,attention}_partial` body, moved verbatim: the plain
chain (normed projection, unfused prep, plain recurrence or attention,
GEMV into `out`), with compacted KV and a prerotated input refused. The hand
dense-TP route now runs its decode mixers through this arm.

### What stays on the hand route

The product dense-TP route (`EpArch::Qwen35DenseTp` serve, bench, MTP and
DFlash dense TP) stays on `forward_{scratch,prefill}_dense_tp*` until the
generic engine has what it uses: per-segment hipGraph capture and replay of
decode, the uneven TP3/TP5 layouts (Q-group splits and replicated KV heads),
the peer-rooted reduction route, and the verify captures (hidden-state ring,
GDN tape, row capture and state rollback) that MTP and DFlash need. Each is
an executor capability, not a Qwen3.5 one.
