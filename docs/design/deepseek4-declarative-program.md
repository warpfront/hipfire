# DeepSeek4 declarative layer program (#666 G9)

Status: single-lane decode (`decode_step_body`, which the eager, hipGraph and
Redline capture routes share) and the existing EP handlers
(`Deepseek4Bindings::run_attend*` / `run_moe*`) run one-row step programs
(slice 1). Batched prefill and verify (direct and capture-safe), the MTP layer
block, the DSpark drafter stages, the TP3/TP4 gfx1201 prefill and the batched
heads run row-batched step programs (slice 2, below). Still sequenced in the
architecture crate: the one-row MTP draft (`mtp_forward`, `mtp_forward_ep`, on
the slice-1 binders), the DSpark head prologue / epilogue (main projection,
HC gate, markov head) and the hand-scheduled heterogeneous gfx1100+gfx1151
decode, which needs a dispatch stream fork/join step first.

## Program

`crates/hipfire-arch-deepseek4/src/program.rs` builds one step list per layer.
The architecture crate binds resident weights, per-layer caches and scratch;
`hipfire_dispatch::pipeline::mla` owns every kernel sequence and route choice.

| Block | Steps |
|---|---|
| Attention sublayer | `HcPre` → `QLoraPrepare` → [`PackedProjections`] → `QLoraProject` → `KvJoint` → `ApplyTailRope` → (ratio > 0) [`CompressorProject`] → `KvCompress` main → (ratio 4) [`PackedProjections`] → [`CompressorProject`] → `KvCompress` indexer → `SparseIndex` → `SwaRingWrite` → (ratio 128) `IdentityGather` → `SparseAttention` (ratio > 0; ratio 4 gathers the indexer's top-k first) or `SwaAttention` (ratio 0) → `ApplyTailRope` (inverse) → `OLora` → `HcPost` |
| FFN sublayer | `HcPre` → `ExpertFfn` (or `Clear` under `HIPFIRE_DEEPSEEK4_SKIP_FFN`) → `HcPost` |
| Head | `HcHead` → `PreparedProjection` (lm_head) |

The integrated StepGraph IDs for the shared operations are 32–49 (`HcPre`
through `CompressorProject`), and the rows-only operations append 50–54;
beta's existing IDs 28–31 remain unchanged. RowSlots uses beta's shared
12-slot attention-state stride, including global compressor commit slots,
and TP input uploads retain each rank's compressor cache placement.

`PackedProjections` appear only on the exact-gfx1201 MQ2R route, where the
projections that share the attention input run as one mixed-row launch and
their own steps skip the projection.

With HC ping-pong the attention mix writes `residual_streams_next` and the FFN
mix writes back, so a layer ends in the buffer it started in and needs no swap.
Without ping-pong each mix writes the `q` scratch and copies back
(`HcPostOp::writeback`). The per-sublayer EP handlers execute the same builders
on `state.residual_streams` and swap after a folded mix, as before.

Lazy allocation moved out of the operations: `program::ensure_layer` runs
every `ensure_*` binder of a layer (same sizes, dtypes and order as the old
first call), then `program::layer_program` binds immutable steps.
`state.pos_buf` is pointed at the layer's position slot during the ensure phase,
as `apply_tail_rope` did.

## Typed operations (`hipfire-dispatch`, `pipeline::mla`)

| Step | Operation | Notes |
|---|---|---|
| `HcPre` (`hc::HcPreOp`) | Manifold HC control (`c = α·(X·W) + base`), sigmoid / Sinkhorn finalize, input map | fused-control, fused-finalize, input-map and rsqrt-once routes as fields |
| `HcPost` (`hc::HcPostOp`) | `X' = comb·X + post·y` | branch: one output, or gfx1201 TP3/TP4 peer partials |
| `HcHead` (`hc::HcHeadOp`) | head HC combine, MTP capture of the full streams, output norm, FWHT | |
| `QLoraPrepare`, `QLoraProject` | attention RMSNorm (plain + rotated), `wq_a` → `q_norm` → `wq_b` → per-head norm | fused q_norm + rotate route |
| `KvJoint`, `ApplyTailRope` | joint latent KV + `kv_norm`; tail YaRN RoPE of Q and KV | YaRN parameters bound per layer |
| `CompressorProject` (`compressor::CompressorProjectOp`) | the compressor's `wkv` / `wgate` projections into the `kv` / `score` scratch (one step per flavor; skipped when a packed projection wrote them) | |
| `KvCompress` (`compressor::KvCompressOp`) | strided compressor over the projected row: APE, ring write, softmax pool, norm, RoPE, F32/F16 store, overlap shift | device-slot route (decode/capture) and host-gated route (batched prefill fallback) |
| `SparseIndex` (`attention::SparseIndexOp`) | ratio-4 indexer: query + RoPE, head weights, relu scores, top-k | two-stage, owner-parallel (gfx942) and portable top-k |
| `SwaRingWrite` (`attention::SwaRingWriteOp`) | the row's KV into both SWA rings (decode attention reads the rings, so the write comes first) | ring slot from device state |
| `IdentityGather` (`attention::IdentityGatherOp`) | ratio 128: the first active compressed rows, in order | F32 / F16 / block-cyclic sharded caches |
| `SwaAttention` (`attention::SwaAttentionOp`) | ratio 0: softmax over the window and the sink | developer `Pos0` route (`HIPFIRE_DEEPSEEK4_ATTN=pos0`) on every layer |
| `SparseAttention` (`attention::SparseAttentionOp`) | ratio > 0: one joint softmax over the window, the gathered compressed rows and the sink; ratio 4 gathers the indexer's top-k first | F32 / F16 / block-cyclic sharded caches |
| `ApplyTailRope` (`inverse`) | inverse tail RoPE of the attention output | same YaRN parameters as the forward rotation |
| `OLora` (`attention::OLoraOp`) | grouped O-LoRA: (rotate) → per-group `wo_a` → (rotate) → `wo_b` | grouped wo_a per arch (MQ4, E8 gfx1151 / gfx1201 / gfx1100, gfx942 backend, per group) |
| `ExpertFfn` (`expert_ffn::ExpertFfnOp`) | FFN norm, shared SwiGLU expert, `sqrt(softplus)` router, bias-aware or hash routed experts | serial, two-stream overlap and retained-split schedules; EP partial redirection |
| `PreparedProjection`, `PackedProjections` | one projection of an architecture-prepared (plain + rotated) input; several in one gfx1201 mixed-row launch | |

Model-owned exact-device kernels (the gfx942 grouped O-LoRA and parallel
indexer top-k) reach dispatch through `attention::LatentAttentionBackend`,
implemented by the loaded model's `Mq2rBackend`, as the bias-aware MoE already
does through `MoeBiasAwareMq2Backend`.

## Numerics contract

The move is bit-exact by construction: every operation launches the same
kernels, in the same order, with the same arguments and under the same branch
conditions as the pre-port hand functions. Developer captures
(`HIPFIRE_DS4_DENSE_ACT_DIR`, `HIPFIRE_DEEPSEEK4_STAGE_NORM`,
`HIPFIRE_DEEPSEEK4_DUMP_INDEXER`, `HIPFIRE_DS4_ROUTE_DUMP`) run the program one
step at a time and read their tensors after the step that produced them; each
captured tensor is unchanged by the rest of its step.

Removed: the hand layer loop and the #397 single-lane super-op path
(`decode_step_body_lowered`, `HIPFIRE_FORWARD_LOWERED` for DeepSeek4).

## Evidence

gfx1151 (halo), fixture `deepseek-v4-flash-0731.mq2r` (registry
`deepseek-v4-flash`, sha256 `cbf2bbcf…`, 82,191,359,851 bytes), against beta
`8d1801c77` (and earlier `09540d8c4`, one step per moved operation group):

- 2,300 committed positions (2,200 teacher-forced + 100 greedy): input and
  argmax tokens, full-logit hash and HC residual-stream hash at every
  position; every layer's SWA rings, main and indexer compressor caches, KV
  and score rings and `mtp_last_hidden` every 32 positions and at the end —
  all **bitwise identical**, eager and hipGraph. The run passes the
  128-token sliding window, the ratio-128 compressor commits and the 512-row
  indexer top-k saturation (position 2,048). Beta itself is deterministic on
  this harness (repeat eager runs and the hipGraph run hash identically).
- Serve (`serve_harness.py --thinking off --sampling greedy
  --compare-transcript`, prefill included): battery and chain transcripts
  byte-identical.
- Decode, `hipfire bench --matrix --tg 64`, A/B/A/B under the halo lock with
  no build running (power-limited at ~86 W): 26.45 / 26.33 → 26.35 / 26.35
  tok/s at ctx 128, 23.31 / 23.24 → 23.23 / 23.23 at ctx 2048.
- DSpark speculative decoding is not bitwise comparable: greedy DSpark is
  non-deterministic run to run on beta itself (four beta runs, four
  acceptance-τ sequences); the PR's runs reproduce τ sequences beta also
  produced.

Hardware-blocked on halo: the gfx1201 (packed projections, TP3/TP4 peer HC,
sharded caches), gfx942 and heterogeneous gfx1100+gfx1151 routes, and physical
EP. Fixture-missing on halo: the MTP sidecar.

## Row-batched program (slice 2)

`crates/hipfire-arch-deepseek4/src/rows_program.rs` binds one step program per
layer over a window of `rows` positions held in a `PrefillBatchScratch`
(`RowsWindow`: start position, row count, tokens, and whether every
position-dependent count comes from the device arrays uploaded at the window
boundary — the capture-safe verify body). Each operation is one `Step` variant
in `hipfire_dispatch::pipeline::mla`: one op struct with the fields both chains
share, `rows`, and, for the eighteen operations that also run at decode, a
route enum carrying only what differs: `Decode` (the single-lane decode
program) or `Batched` (this program). The builder picks the route, not the row
count, so a one-row verify window takes `Batched`; every route operation
refuses a decode route with other than one row (`attention::check_rows`). The
compressor projections (`CompressorProject`) and the attention
(`SwaRingWrite`, `IdentityGather`, `SwaAttention`, `SparseAttention`, the
inverse `ApplyTailRope`, `OLora`) run the same operations at decode and over
rows, so decode and prefill of one layer expose the same `OpKind`s apart from
`SwaStage`: decode attends the SWA rings in place, so it has no staging. The
five rows-only operations are `EmbedStreams`, `StreamMeanCapture`, `RingKv`,
`BlockStage` and `SwaStage`; they refuse an empty window. The DSpark head's
final norm runs as a row-complete
`Step::RmsnormAutomatic { rows: block }`, not an arch-crate `rmsnorm_batched`.

| Block | Steps |
|---|---|
| Window input | `EmbedStreams` (Q8 embedding lookup, HC stream broadcast) |
| Attention input | `HcPre` → `QLoraPrepare` (attention RMSNorm) → `PackedProjections` (gfx1151 three-row E8 pack) → `QLoraProject` (`wq_a` unless packed, `q_norm`, `wq_b`, per-head norm) → `KvJoint` → `ApplyTailRope` |
| Ratio 0 | `SwaStage` → `SwaAttention` |
| Ratio > 0 | `SwaStage` → `CompressorProject` (F16 WMMA or prepared-input GEMMs; skipped when packed) → `KvCompress` (whole window: APE, aligned batched pool / norm / RoPE / store, overlap ring update; or the no-event ring write) or one slice-1 `KvCompress` per row (capture route, unaligned window closing an event) → `SparseIndex` (ratio 4: indexer query, weights, ReLU scores, top-k, gather) or `IdentityGather` (ratio 128: in-order gather) → `SparseAttention` (gathered or direct top-k, F16 WMMA or F32) |
| Attention output | `ApplyTailRope` (inverse) → `OLora` (rotations, grouped `wo_a` by format, `wo_b`) → `SwaRingWrite` → `HcPost` |
| FFN | `HcPre` → `ExpertFfn` (FFN norm, shared SwiGLU expert, router, MoE family bias-aware / hash prefill) → `HcPost` |
| DSpark capture | `StreamMeanCapture` on the drafter's target layers |
| Heads | `HcHead` (`Batched` route: slice-1 `HcHead` per row, staged into one matrix) → `PreparedProjection` (`Batched` route, one batched lm_head) |

The batched projection router (`project_prepared_rows`) takes the route knobs
the model resolves once (`RowsProjection`: E8 token tiles, the gfx1151 batched
E8 GEMV ceiling, gfx1201 wide tiles, paired B3, HFQ4 WMMA).

Composed from the same builders (`push_attention_input`, `push_attention`,
`push_o_lora`, `push_mix`, `push_ffn`):

- the MTP layer block of `mtp_forward_batched`;
- the DSpark drafter stage (`drafter_stage_program`): the trunk attention
  input, `RingKv` (each context row's latent KV into the stage ring), the
  bidirectional `BlockStage`, `SwaAttention` over it, O-LoRA, both mixes and
  the score-routed FFN on the portable route;
- the gfx1201 TP3/TP4 prefill (`forward_ep_prefill_batch_chunked`): the
  attention at each rank's local head / group width, the rank-ordered rooted
  reductions, then the mixes and the FFN.

Developer captures (`HIPFIRE_DS4_DENSE_ACT_DIR`, `HIPFIRE_DEEPSEEK4_DUMP_STATE`,
`HIPFIRE_DEEPSEEK4_ATTN_TWIN` / `_ATTN_DEBUG_BISECT`) run the program one step
at a time, as in slice 1.

### Evidence (slice 2)

gfx1151 (halo), `deepseek-v4-flash-0731.mq2r` + its DSpark sidecar
(`deepseek-v4-flash-0731-dspark.mq2r`), against beta `fbcdb362b`, harness
`ds4_rows_parity` (evidence `~/mesh-validation/HIPF-79` on halo):

- Prefill in 512-row chunks over 2,300 positions: every row's full logits and
  argmax, the chunk's HC streams, and every layer's SWA rings, compressor and
  indexer caches, rings and score rings at each chunk boundary.
- Production `forward_prefill_batch_chunked` (1,024-row chunks), then 48
  greedy decode steps (logits, streams) and all caches.
- 40 verify windows of 1-8 rows (unaligned compressor windows included):
  every row's logits, streams, caches every 8 windows; also with
  `HIPFIRE_DEEPSEEK4_BATCH_HEAD=0`.
- DSpark: target-hidden capture over a 600-token prefill, 8 drafted blocks
  (`dspark_forward`: tokens, confidences), the body-only route
  (`dspark_run_body_and_hc_gate`: `x_head`) and every stage ring.
- Calibration capture (`HIPFIRE_DS4_DENSE_ACT_DIR`): all 554 dumped activation
  files.

All **bitwise identical**; beta is deterministic on the harness (two runs
hash identically). In that historical run the capture-safe verify body could
not be compared: both arms read slots 10 / 11 from a 10-slot batched stripe
and panicked. The current-beta integration preserves its repaired 12-slot
stripe; no local DS4 artifact is available to re-exercise that GPU route.

Speed, `deepseek4_prefill_bench` A/B/A/B under the halo lock (tok/s; the two
numbers are the two interleaved runs):

| Shape | beta `fbcdb362b` | this branch |
|---|---|---|
| prefill 4,096 tokens, 1,024-row chunks | 102.05 / 102.63 | 99.20 / 102.38 |
| verify window B=3 at depth 2,048 | 35.89 / 36.59 | 36.85 / 36.64 |
| verify window B=8 at depth 2,048 | 46.62 / 47.02 | 47.02 / 47.09 |
| verify window B=64 at depth 2,048 | 67.60 / 68.09 | 67.78 / 68.33 |
| AR decode at depth 2,048 | 25.09 / 24.22 | 24.50 / 24.55 |

Equal within halo's run-to-run spread (other agents build on the box).

Not covered on halo: the MTP sidecar (fixture missing; the MTP layer block
shares the trunk builders), gfx1201 / gfx942 and the TP3/TP4 prefill, the
heterogeneous route and physical EP (hardware).
