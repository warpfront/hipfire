# Muse Glimmer declarative layer program (#666 G11)

Status: every Glimmer forward route runs typed dispatch steps except the
continuous-batching decode (see below): single-lane decode (AR and the DFlash
hidden-capture decode, slice 1), batched prefill, batched DFlash verify, the
shared draft/verify LM head and the DFlash drafter (slice 2). Glimmer is
refused for every parallel axis until the G13 dense axes.

## Semantic checkpoint

Muse Glimmer (`muse_glimmer`, arch 14) is a dense decoder whose semantics
differ from the other families in ways that compute plausible but wrong text
when lost. The program is defined by these, and each is enforced where it can
be checked:

| Semantics | Where it is fixed |
|---|---|
| Embedding is followed by a weight-less RMS norm (no `sqrt(dim)` scale). | `forward::embed_lookup` (token staging, outside the program) |
| Two layer kinds. Sliding layers rotate (`layer_rope_theta != 0`), read the sliding cache and attend a `sliding_window` span; full layers are NoPE (`theta == 0`), read the full cache and attend causally. | `GlimmerConfig::from_metadata_json` refuses a layer whose `layer_types` entry and theta disagree (the window keys off the theta, the cache off the type). |
| Scale-less per-head QK RMS norm, then `Q *= qk_scale_factor` (3.87); the kernel's `1/sqrt(head_dim)` stays. | `SandwichAttentionOp` with ones `q_norm`/`k_norm` and `q_scale` |
| Gated attention: `attn_out *= sigmoid(self_attn.gate_proj(n))` before `o_proj`. | `SandwichAttentionOp::gate` |
| Sandwich norms with split eps: pre-norms use `rms_norm_eps` (1e-5), post-norms `post_norm_eps` (1e-8). | `SandwichStream::eps` / `post_eps` |
| SiLU SwiGLU, bias-free projections, untied LM head. | Config refuses `hidden_activation != "silu"` and `attention_bias`; the loader requires `lm_head`. |
| Logits: `output_multiplier` first, then `final_logit_softcapping`. | `program::logit_steps`: `Step::Scale` before `Step::Softcap` |

The refusals are pinned by `config::tests::layer_kinds_and_declared_semantics_fail_closed`.

## Program

`crates/hipfire-arch-muse-glimmer/src/program.rs` declares a layer as the two
shared sandwich sublayers of `hipfire_dispatch::pipeline::sandwich`, the same
operations Gemma 4 runs:

| Block | Step |
|---|---|
| Attention sublayer | `SandwichAttention` (Q/K/V/gate projections, QK norm, RoPE or NoPE, windowed Q8 attention, sigmoid gate, O projection, post-norm residual) |
| FFN sublayer | `SandwichMlp` (`Activation::Silu`, gate/up, the SwiGLU product, down, post-norm residual) |
| Head | `RmsnormAutomatic` + `Gemv` for the decode row; `RmsnormAutomatic { rows }` (in place over the rows) and `sandwich::gemm_rows` for the verify and draft picks; then `Scale` → `Softcap` |

One program serves the decode row and the rows of a prefill chunk or verify
block (`LayerIo::route`: `RowsRoute::Decode`, `Prefill` or `Verify`). Glimmer's differences from
Gemma are op fields, not separate operations:

- `SandwichStream::route` (the shared `pipeline::RowsRoute`: `Decode` for
  the decode row, `Prefill` / `Verify` for batched rows; the route, not the
  row count, picks the kernels) and `kernels = SandwichRoute::Steps(StepsRoute
  { shared_rotation, fused_silu_rotate })`: decode projections of one pre-norm
  run as generic `Gemv` steps, so the dispatch fusion table groups them as it
  did before; batches take their route's ladder at any row count.
  `HIPFIRE_GLIMMER_SHARED_ROT` and `HIPFIRE_GLIMMER_FUSED_SILU_ROTATE` are
  read in the arch crate and passed as these fields.
- `SandwichStream::post_eps` (post-norm eps) and `out` (the attention
  sublayer writes `x'` into `residual`, the FFN sublayer writes `x"` back
  into `x`, so neither copies the residual stream).
- `KvRoute::WindowedQ8 { tiled, rows_flash, wmma_flash }`: the
  absolute-position Q8 cache and its decode/prefill/verify attention ladders
  (`HIPFIRE_GLIMMER_WMMA_FULL` is read in the arch crate).
- `RopeKind::NoPe` for full layers, `AttnGate` for the attention gate.

`forward.rs` keeps the lifecycle around the program: the embedding and its
norm, staging of positions (device scalar, the tiled-attention copy and
partials; for batched rows the row positions, the rebased positions and flash
partials prefill reads past the window), the VMM KV pages, the per-layer
hidden capture for DFlash (host or device log), and logits download, argmax
or on-device sampling. Removed: the hand layer loops
(`glimmer_layer_decode`, the bodies of `prefill_chunk_batched` and
`verify_block_capture_impl`) and their projection helpers.

## Shared pieces (`hipfire-dispatch`)

Every Glimmer route runs on `pipeline::sandwich`; Glimmer adds no `Step`
variant.

- `sandwich::gemm_rows` is the one rows projection of sandwich models. Its
  `Ladder` is `Wide` (Gemma 4) or `Route(RowsRoute)` (Glimmer): Q8 takes the
  chunked GEMM on a route ladder and the gfx1151 4-warp WMMA tile on the wide
  one; on gfx12 a prefill chunk takes the residual (and Muse residual) WMMA
  GEMMs for MQ4/HFQ4, and four MQ4/HFQ4 projections of one rotated input run
  as one fused QKVZA WMMA GEMM.
- The SwiGLU product rotated for an MQ4/MQ6 down projection is
  `RotationFamily` `WithSwiGLU`, which gained the batched launchers.
- The drafter's bidirectional attention is `AttentionFamily`
  `run_full_attention_on` with `AttnFullF32` pinned to the scalar
  `DflashScalar` tile (the kernel the drafter ran before).
- `Step::RmsnormAutomatic` gained `rows`: more than one row takes the batched
  norm (and is a fusion barrier), so a hidden-width norm over a batch is the
  row-complete norm op, not `QkNorm` (head-local under the StepGraph
  contract). A one-row batch keeps the single-row kernel, which the batched
  kernel matches at one row. `rows` is an extent here, not a route: both run
  the same `rmsnorm_f32` kernel. Gemma 4's batched final norm runs on it too.
- One route convention (S10): an operation that serves decode and batched
  rows carries `rows` plus exactly one explicit route, and `rows` never picks
  kernels. The shared discriminator is `pipeline::RowsRoute { Decode,
  Prefill, Verify }` (lifted out of `sandwich`, `Decode` added); an
  operation whose routes need route-only operands keeps a per-op
  `XRoute::{Decode { .. }, Batched { .. }}` (DeepSeek4). `SandwichStream`
  carries one `route: RowsRoute`; `StepsRoute.batch` and the implicit "`Wide`
  batches when `rows > 1`" are gone, and `SandwichRoute::{Wide, Steps}` is
  only the ladder and fusion choice. Gemma 4 binds `Decode` exactly at one
  row, Glimmer what it passed before, so every kernel choice is unchanged.
- `SandwichAttentionOp` / `SandwichMlpOp` have `validate_for_gpu`, run by
  `validate_steps`: a decode route with more than one row and batched rows
  without positions are refused; windowed Q8 KV, draft KV and an output gate
  need the steps route (draft KV also batched rows); rows past a
  position-indexed cache (`position + rows <= physical_cap`, Tier and
  windowed Q8 KV) are refused; and the tiled / flash partials and positions,
  the rotation scratch and the BF16 staging the chosen kernels read, and the
  stream extents, are checked. Unit tests pin the decode-route, cache-bound
  and steps-route refusals (`sandwich::validation_tests`).
- StepGraph: `Step::SiluMul` takes the reserved hand-route ID `OpKind::SiluMul
  = 1001`; #877's `GatedShortConv` is 27. Beta folded the Gemma 4 sandwich MLP
  into `GatedMlp` (28) and retired `SandwichMlp` (20, never reused), so Glimmer's
  `Step::SandwichMlp` (its steps/rows ladders, not yet expressed as a
  `GatedMlp` ladder) takes the appended `OpKind::SandwichMlp = 31`.

## Numerics contract

FWHT-G256 projection formats read one shared rotated activation
(`GemvInput::Prerotated`), as the pre-port shared-rotation path did; every
other format (Q8 attention of the quality trunk, the Q8 LM head) reads the
plain norm (`GemvInput::Raw`) and takes its plain GEMV, so no Q8 projection
enters the fusion table. `HIPFIRE_GLIMMER_SHARED_ROT=0` and
`HIPFIRE_GLIMMER_FUSED_SILU_ROTATE=0` keep their meaning.

One launch-level change: prerotated MQ4 projections that share an input now
match the dispatch fusion table on every arch, so the FFN gate/up pair runs
`fused_gate_up_hfq4g256` and the `.mq4r` q/k/v/gate quartet
`fused_qkvza_hfq4g256`. The pre-port route fused them only on gfx1100 at the
exact Glimmer shape (where it called the same two kernels). On gfx1151 the
fused launches are bitwise identical to the separate GEMVs (evidence below);
gfx1201 is not measured.

Batched rows keep the kernels each route ran before, per route:

| Piece | Prefill chunk | Verify block |
|---|---|---|
| Pre-norm | batched RMS norm, or the fused batched norm + FWHT for FWHT-G256 formats | same |
| Projections | Q8 chunked batched GEMM; MQ4/HFQ4 prerotated batched-LM-head GEMM; MQ6 batched GEMM; F32 batched GEMM; BF16 staged MFMA GEMM; other formats one GEMV per row | same |
| gfx12, more than one row | Q/K/V/gate of one prerotated MQ4 input: one fused QKVZA WMMA GEMM; other MQ4/HFQ4 projections: the residual WMMA GEMM over a zeroed output (the Muse residual GEMM for M below 8192 at a multiple of 192 rows) | the batched-LM-head GEMMs |
| QK norm, scale, RoPE | batched per-head norm, scale, batched RoPE | same |
| Attention | per-row causal kernel until the chunk ends past the window (sliding) or past 2048 (full), then the query-tiled Q8 WMMA flash kernels; a one-row chunk, or `HIPFIRE_GLIMMER_WMMA_FULL=0`, takes the scalar key-tiled flash kernel on a tile-aligned rebased origin | the query-tiled WMMA flash kernels from two rows on, else the causal kernel |
| Post-norm residual | batched norm in place, then `out = residual + norm` | same |
| Head | batched final norm, GEMV of the last row only | batched final norm, then `glimmer_lm_head_picks` |

Deliberate changes on the batched routes:

- A verify block now maps the VMM KV pages its rows write, as prefill and
  decode do; before it relied on pages mapped by earlier work.
- BF16 projections of a verify block (calibration teachers, gfx942 only) take
  the staged MFMA GEMM prefill takes, not one GEMV per row. The BF16 LM-head
  picks no longer apply the output multiplier and the soft cap twice.
- gfx12: a q/k/v/gate (or gate/up) group of mixed formats takes the residual
  GEMM for its MQ4/HFQ4 members, like every other prefill projection; before,
  only uniform groups did. No shipped Glimmer artifact mixes them; gfx12 is
  not measured.

Removed developer switches: the bring-up ablations
(`HIPFIRE_GLIMMER_NO_{EMBED_NORM,QK_NORM,QK_SCALE,ATTN_GATE,OUTMUL,SOFTCAP}`,
`HIPFIRE_GLIMMER_ROPE_{ALL,INTERLEAVED}`, in every Glimmer path), the
decode-only fusion opt-outs (`HIPFIRE_GLIMMER_FUSED_POSTNORM`,
`HIPFIRE_GLIMMER_FUSED_QK_ROPE`) and the decode flash threshold override
(`HIPFIRE_GLIMMER_FLASH_DECODE`). Each changed the computed function or a
measured route choice, not a lifecycle. Slice 2 removes
`HIPFIRE_GLIMMER_NO_FLASH` (per-row sliding-window debug route),
`HIPFIRE_GLIMMER_FLASH_FULL` (`=0` brought back the batched kernel's LDS
failure past about 16K), and `HIPFIRE_GLIMMER_O_RESIDUAL` /
`HIPFIRE_GLIMMER_MUSE_GEMM` (gfx12 A/B switches between byte-identical
kernels), as well as the verify timing line of `HIPFIRE_GLIMMER_TIMING`.

## DFlash drafter

`drafter.rs::layer_steps` declares one drafter layer (arch 23, a two-norm
Llama block whose K/V span the context rows ahead of the block rows) as the
same two sandwich operations without post-norms (`post_norm: None` is the
plain pre-norm residual add), on the verify ladder. The attention's
`KvRoute::Draft` carries the context rows: the `delta` rows new since the
last window are projected from `fc`'s output, K-normed and cached, the cached
window is copied (a recorded `copy_f32_buffer` launch) ahead of the block's
K/V and rotated at its positions, and every block row attends all context and
block rows. The encoder `fc` projection runs `sandwich::gemm_rows`, and its
output norm and the final norm are `RmsnormAutomatic { rows }` steps in place (a row-complete norm, not the head-local `QkNorm`). The arch keeps the
context-row watermarks (which rows are new since the last window, rollback
detection), the host/device staging of the context rows, the noise embedding
and the positions.

Each drafter projection rotates its own input, with its own AWQ scale, and
the MQ4 `fc` input rotates into its own scratch sized to the `fc` K
(`num_extract * hidden` per row). Before slice 2 the projections of one input
shared one rotation, and the `fc` rotation view was `hidden` wide at the
context row's offset, so each rotation wrote five times its view, and past the
end of `kv_input_rot` once `fill_start + 5 * delta` exceeded its rows.

## Continuous-batching decode

`batch.rs` / `forward_batch.rs` (`GlimmerDecodeBatchState`,
`forward_decode_batch_glimmer`) keep their hand loop. Nothing outside the
crate calls them (the harness that drove them was removed in `ff778990a`), and
their layer omits the Q norm every other route applies, so beta's output is
not an oracle for a port. Whether to remove the core or port it onto the
layer program (gaining the Q norm) is an open decision.

## Single-only refusal

`admission::muse_glimmer_topology_refusal` refuses arch 14 with `pp > 1`,
`tp > 1` or `topology.ep > 1` at source admission, before teardown or
allocation, with one message. Before, `tp > 1` reached the generic EP
refusal, `pp > 1` the carrier default, and `topology.ep` was ignored for
Glimmer. Pinned by `admission::tests::muse_glimmer_refuses_every_parallel_axis`.

## Context cap

The KV caches hold `max_seq` absolute positions. The daemon
(`hipfire-generate` `generate_muse_glimmer`) refuses a prompt of `max_seq`
tokens or more and stops decode at the cap. A DFlash verify window writes
`block_size` rows from the current position, so once the next window would
cross the cap the spec loop finishes the request on AR decode, like a
request without a drafter (a request that fits `max_seq` completes; one past
it stops at the cap). Behind that, the sandwich attention's cache bound
(`position + rows <= physical_cap`) refuses any rows past the cache before
launch: the legacy KV backend has no mapping guard, and a caller that still
asks for such a window gets the request's error event (`… outside a cache
of N rows`), not a panic or a write past the cache. Pinned on the GPU by
`tests/kv_cap_gpu.rs` (`--ignored`; a verify block and a prefill chunk that
cross `max_seq` are refused, the window that fits still runs) and by
`hipfire-generate` `dense_rollback_gpu::glimmer_dflash_request_that_fits_max_seq_completes`
(`max_seq` 384, a greedy DFlash request of 312 tokens that fits completes on
the speculative route with no error; it failed before the AR tail).

## Evidence, slice 1 (gfx1151, halo)

Base: the route oracle ran against beta `09540d8c4`, serve, refusal and decode
against `f826e2197` (earlier rounds on `f826e2197` and `901e44fdd` gave the same
parity). Fixtures from the registry repo
`hipfire-models/muse-glimmer-30b`: `muse-glimmer-30b.mq4` (sha256
`87bc776a…0fc052`; Q8 attention and LM head, MQ4 FFN) and
`muse-glimmer-30b.mq4r` (sha256 `47ccfccd…ce7503`; MQ4 attention and FFN, Q8 LM
head).

Route oracle: a throwaway dump driver, built unchanged in both trees, runs
`decode_step` over 2,100 teacher-forced tokens plus 48 greedy tokens (legacy
KV) and records, at every committed position, the fed token, the argmax and
digests of the full logits and the residual stream, and every 64 positions a
digest of all KV bytes; the greedy tail keeps raw logits and the final KV
bytes. Positions past 1,280 take the tiled attention kernel and positions past
2,048 exercise the sliding window.

| Fixture | Positions | Tokens | Logits | Residual stream | KV |
|---|---|---|---|---|---|
| `.mq4` | 2,148 | identical | bitwise | bitwise | bitwise (35 checkpoints + final bytes) |
| `.mq4r` | 2,148 | identical | bitwise | bitwise | bitwise (35 checkpoints + final bytes) |

The DFlash host-capture decode (`decode_step_with_capture`, layers 1, 12, 25,
38, 51) is bitwise identical too, captured rows included (96 positions per
fixture).

Serve (`serve_harness.py --thinking off --sampling greedy --max-tokens 160
--compare-transcript`, beta daemon vs head daemon): battery and chain
`transcript_byte_identical=true` on both fixtures. Glimmer reasons regardless
of `--thinking off`, so at 160 tokens every turn is reasoning only on both
arms.

Refusal (`hipfire run`): beta refuses `--tp 2` with the generic EP message and
serves `HIPFIRE_TOPOLOGY_EP=2` on one device; head refuses both with the
single-device message before any kernel is built.

Decode, `hipfire bench --matrix --tg 64`, A/B/A/B under the halo lock, no
other cargo/rustc running (both arms equally power-limited):

| Fixture | ctx | beta | head | ratio |
|---|---|---|---|---|
| `.mq4` | 128 | 11.82 / 11.82 | 11.98 / 11.98 | 1.014 |
| `.mq4` | 2048 | 11.31 / 11.32 | 11.53 / 11.51 | 1.018 |
| `.mq4r` | 128 | 13.48 / 13.48 | 13.86 / 13.87 | 1.029 |
| `.mq4r` | 2048 | 12.84 / 12.85 | 13.24 / 13.26 | 1.031 |

Not covered: gfx1100 and gfx1201 (no hardware on halo; on gfx1201 the fused
MQ4 launches are new and unmeasured), the VMM KV backend in the route oracle
(serve ran legacy, the admitted backend on gfx1151), and DFlash speculative
decoding end to end (no `muse-glimmer:draft` fixture on halo).

## Evidence, slice 2 (gfx1151, halo)

Base beta `6ea3f5076`; head = this branch (#881 rebased onto that beta, plus
slice 2). Fixtures as above plus `muse-glimmer:draft`
(`muse-glimmer-30b-dflash.mq4`, sha256 `6fc0988e…e23d2d`, from the same
registry repo).

Route oracle: a throwaway driver, built unchanged in both trees, runs the
public API and digests every observable:

- a 2,251-token prompt in 192-token `prefill_with_capture` calls (through
  1,280 and the 2,048 window), with last-row logits, captured hidden rows and
  all KV bytes per call;
- 8 greedy decodes, then a 1-token and a 300-token turn (1-row and 44-row
  chunks past the window);
- verify blocks of 16/16/1/7 with full `logits_out`, partial accepts and
  rollback, then a tail decode;
- verify blocks of 16 at 300, 1,270 and 2,040;
- a device-capture prefill, verify, `commit_device_verify_capture` and
  decode, with the committed log rows;
- five drafter windows (context 300, +16, +384, a rollback to 650, and an
  empty context) with their draft picks.

All 130 digests per fixture are identical to beta on `.mq4` and `.mq4r`, on
legacy KV and, without the KV digests, on VMM KV. The rebased #881 alone and
the final commit give the same result.

Serve (`serve_harness.py --thinking off --sampling greedy --max-tokens 160
--compare-transcript`, beta daemon against head daemon): AR battery and chain
are byte-identical on both fixtures. With `--dflash on --draft` they are
byte-identical on `.mq4` (battery, chain) and on `.mq4r` battery (tau
4.8–10.4). The `.mq4r` chain differed in one turn. Glimmer's speculation profit
guard retires DFlash on wall-clock timing (that run's base retired at context
410, head at 434), and DFlash and AR greedy output differ on this model, so a
guarded DFlash transcript is not a reproducible oracle. With
`HIPFIRE_GLIMMER_SPEC_PROFIT_GUARD=0` both chains are byte-identical.

Prefill and decode (`hipfire bench --matrix --pp 512,2560 --ctx 128,2048
--tg 64 --runs 2 --warmups 1`, one halo lock per run, alternating arms, two
runs per arm; tok/s):

| Fixture | Row | beta | head | ratio |
|---|---|---|---|---|
| `.mq4` | pp512 | 30.75 / 30.70 | 30.45 / 30.30 | 0.989 |
| `.mq4` | pp2560 | 30.45 / 30.65 | 30.15 / 30.45 | 0.992 |
| `.mq4` | tg64 @ 128 | 11.81 / 11.81 | 11.98 / 11.99 | 1.015 |
| `.mq4` | tg64 @ 2048 | 11.31 | 11.54 / 11.54 | 1.020 |
| `.mq4r` | pp512 | 31.25 / 31.80 | 31.10 / 31.70 | 0.996 |
| `.mq4r` | pp2560 | 30.75 / 31.20 | 30.65 / 31.50 | 1.003 |
| `.mq4r` | tg64 @ 128 | 13.48 / 13.47 | 13.85 / 13.88 | 1.029 |
| `.mq4r` | tg64 @ 2048 | 12.86 / 12.84 | 13.26 / 13.27 | 1.032 |

Prefill is within the box's run-to-run spread. One head `.mq4` run started
under foreign CPU load (loadavg 27). The decode gain is slice 1's. Prefill
runs at about 30 tok/s on both arms; that is a separate gfx1151 kernel
question (HIPF-80).

Not covered: gfx1100, gfx1201 and gfx942 (no hardware on halo). The gfx12
prefill routes (fused QKVZA, residual and Muse GEMMs) and the BF16 routes
are moved code without a run. The continuous-batching decode is not ported.
