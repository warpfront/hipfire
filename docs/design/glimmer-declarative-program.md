# Muse Glimmer declarative layer program (#666 G11)

Status: single-lane decode cut over (AR `decode_step` and the DFlash
hidden-capture decode). Batched prefill, batched DFlash verify, the
continuous-batching decode (`forward_batch.rs`) and the drafter keep their
current code; they are the next G11 slices. Glimmer is refused for every
parallel axis until the G13 dense axes.

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
shared sandwich sublayers of `hipfire_dispatch::pipeline::sandwich`, the
operations Gemma 4 runs. The architecture crate binds resident weights, KV and
scratch; dispatch owns every executor body and route choice.

| Block | Step |
|---|---|
| Attention sublayer | `SandwichAttention`: Q/K/V/gate projections, QK norm, RoPE (NoPE on full layers), windowed Q8 attention, sigmoid gate, O projection, post-norm residual |
| FFN sublayer | `SandwichMlp` (`Activation::Silu`): gate/up, the SwiGLU product (FWHT-rotated for an MQ4/MQ6 down), down, post-norm residual |
| Head | `RmsnormAutomatic` → `Gemv` (lm_head) → `Scale` → `Softcap` |

Glimmer's differences from Gemma are op fields, not separate operations:

- `SandwichStream::route = RowsRoute::Decode` (the shared
  `pipeline::RowsRoute`: the route, not the row count, picks decode or batched
  kernels) and `kernels = SandwichRoute::Steps(StepsRoute { shared_rotation,
  fused_silu_rotate })`. Decode projections of one pre-norm run as generic
  `Gemv` steps, so the dispatch fusion table groups them;
  `HIPFIRE_GLIMMER_SHARED_ROT` and `HIPFIRE_GLIMMER_FUSED_SILU_ROTATE` are
  read in the arch crate and passed as fields.
- `SandwichStream::post_eps` and `out`: the attention sublayer writes its
  output stream into `state.residual` and the FFN sublayer writes back into
  `state.x`, which replaces the residual copy each sublayer made before.
- `KvRoute::WindowedQ8`: one decode row over an absolute-position Q8 cache
  (KV write, attention over `(pos - window, pos]`, `0` = all). Below
  `WINDOWED_Q8_TILED_MIN_SEQ` (1280) it takes the one-workgroup-per-head
  kernel, from there the key-tiled flash kernel; both are byte-identical, the
  threshold is the measured crossover (gfx1201).
- `RopeKind::NoPe` for full layers and `AttnGate` for the attention gate. The
  decode QK norm + RoPE is the fused kernel Gemma 4 runs.
- The SwiGLU product rotated for the down projection is `RotationFamily`
  `WithSwiGLU`.

`forward.rs` keeps the lifecycle around the program: the embedding and its
norm, staging of the position (device scalar, the tiled-attention copy and
partials, VMM KV pages), the per-layer hidden capture for DFlash, and logits
download or on-device sampling. Removed: the hand layer loop
(`glimmer_layer_decode`).

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

Removed developer switches: the bring-up ablations
(`HIPFIRE_GLIMMER_NO_{EMBED_NORM,QK_NORM,QK_SCALE,ATTN_GATE,OUTMUL,SOFTCAP}`,
`HIPFIRE_GLIMMER_ROPE_{ALL,INTERLEAVED}`, in every Glimmer path), the
decode-only fusion opt-outs (`HIPFIRE_GLIMMER_FUSED_POSTNORM`,
`HIPFIRE_GLIMMER_FUSED_QK_ROPE`) and the decode flash threshold override
(`HIPFIRE_GLIMMER_FLASH_DECODE`). Each changed the computed function or a
measured route choice, not a lifecycle.

## Single-only refusal

`admission::muse_glimmer_topology_refusal` refuses arch 14 with `pp > 1`,
`tp > 1` or `topology.ep > 1` at source admission, before teardown or
allocation, with one message. Before, `tp > 1` reached the generic EP
refusal, `pp > 1` the carrier default, and `topology.ep` was ignored for
Glimmer. Pinned by `admission::tests::muse_glimmer_refuses_every_parallel_axis`.

## Evidence (gfx1151, halo)

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
