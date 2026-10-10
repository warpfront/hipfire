# Gemma 4 declarative layer program (#666 G6)

Status: landed on `feat/gemma-declarative` (#813), rebased onto `beta` with
the #795 Qwen3.5 port it builds on.

## Goal

Gemma 4 declares each decoder layer as one typed `Step` list, as Qwen3.5
(#795) and Qwen4 do. `hipfire-dispatch` owns every executor body and every
arch-, dtype- or tier-gated route choice. One program serves decode
(`rows == 1`), batched prefill and EAGLE verify (`rows > 1`), and the arch-22
EAGLE/MTP drafter.

The architecture crate keeps:

- weight and state ownership (one config, one weight set, KV allocation)
  for every variant: 12B dense, E2B/E4B and the 26B-A4B MoE;
- the per-layer program shape (`src/program.rs`: which ops, in which order,
  bound to which tensors);
- lifecycle: hipGraph capture, the chunk loop, the session-cache owner
  (`src/bundle.rs`) and the EAGLE drafter/verify seams (`src/mtp.rs`). The
  spec-decode loop itself is the engine's generic `generate_spec`.

The #397 super-op facade (`lowered.rs` `lower_variant` / `Gemma4Bindings`)
and the hand-written decode arms in `forward.rs` are deleted, and so is the
whole `lowered.rs` stack. Gemma 4 has no EP/TP route, so nothing keeps the
super-op program alive for it.

## Before the port

Two parallel stacks, each with its own config, weights and loader (unified
since; see "Engine seams"):

| Stack | Fixtures | KV | Notes |
|---|---|---|---|
| eager (`config.rs`, `gemma4.rs`, `forward.rs`) | 12B dense (default), E2B/E4B, EAGLE target | Q8 both tiers, `physical_cap = max_seq` | fused qk-norm+RoPE and fused post-norm+residual; hipGraph decode; `forward_batch{,_spec}` B≤64 |
| lowered (`lowered.rs`) | 26B-A4B MoE (always); 12B with `HIPFIRE_{BATCHED,WMMA}_PREFILL=1` (removed) | sliding Q8 ring (`window`), full tier Q8 (beta default; asym3 hd512 under `legacy-asym3`) | super-op facade (removed); per-token serve prefill; `forward_prefill_batch` only for calibration tools |

## Vocabulary

New model-neutral ops live in `hipfire-dispatch/src/pipeline/sandwich.rs`
(not `deltanet`-gated). Each executor takes `rows`.

| Step | Semantics | Route choice owned by dispatch |
|---|---|---|
| `SandwichAttention` | `x = residual + post_norm(o_proj(attend(rope(norm_h(q)), rope(norm_h(k)), norm(v))))` on `input_norm(x)`. Optional V projection (`None` = K=V from pre-norm K); optional weightless V norm; Q prescale; **explicit `Rope { kind: RotateHalf \| PartialHalved { rot_pairs }, theta }`**; `kv_proj: Option<KvProjection>` (`None` = query-only attend over a shared slot, no cache write: E-series KV sharing and the drafter); `window` | fused norm+FWHT-rotate + prerotated GEMVs (MQ4), fused Q8 q+k, fused qk-norm+RoPE, fused post-norm+residual, KV tier via `KvTierPlan` |
| `SandwichMlp` | `x = residual + post_norm(down(act(gate(n)) * up(n)))`, `n = pre_norm(x)`, `act ∈ {GeluTanh}` | fused norm+rotate + fused gate/up (MQ4), fused post-norm+residual |
| `ParallelMoeMlp` | Gemma 4 MoE block: dense GeGLU MLP ∥ routed experts (router input `rms(x)·router_scale/√dim`, softmax top-k renorm, per-expert scale, GeGLU), `post_ffn_norm(norm1(mlp) + norm2(moe))`, residual add | expert dtype routes (MQ4 / HFQ4 / Q8 indexed) |
| `PerLayerInput` | E-series PLE branch: `x += post_norm(proj(gelu(gate(x)) * ple[layer]))` | batched strided PLE kernel |
| `Scale` | `x *= s` (layer scalar), replay-safe recorded scale | — |
| `Softcap` | `logits = tanh(logits / c) · c` | — |

Projections, the final norm and the LM head reuse `Gemv` and
`RmsnormAutomatic`.

## Slices

1. **Eager stack** (dense, E-series, EAGLE target). `SandwichAttention`,
   `SandwichMlp`, `PerLayerInput`, `Scale`, `Softcap` with `rows == 1` and
   `rows > 1` executors; `decode_step_body` and `forward_batch_spec` run the
   same program. The token-level PLE staging and embedding stay in the arch
   crate, outside the captured body.
2. **MoE.** `ParallelMoeMlp` (`rows == 1` and `rows > 1`); 26B-A4B runs the
   shared program on its sliding ring and full-tier KV.
3. **EAGLE/MTP drafter.** The drafter block is `[SandwichAttention(query-only,
   kv_proj: None, target KV), SandwichMlp, Scale]`; pre/post projection and
   head are `Gemv` steps.
4. **Deletion.** `lowered.rs` super-op facade and hand arms; calibration
   tools move to the shared batched program.

## Calibration taps

Calibration records each projection's unrotated input through
`Gpu::maybe_capture_activation`. The sandwich projection helpers (`gemv`,
`gemm_rows`) and the fused-QKV family fire it, so tools need no
per-architecture tap. While a collector is armed, the `rows == 1` fusions that
never materialize the unrotated input (fused norm+FWHT feeding prerotated
GEMVs) step aside; the other fusions tap their shared input. `gemm_rows`
stages BF16 teacher inputs for the MFMA GEMM and runs formats without a
batched kernel one GEMV per row.

## Gates per slice

- 12B dense, E2B, E4B AR: greedy token stream and the per-step top-16 logit
  trace (`HIPFIRE_GEMMA4_LOGIT_TRACE_DIR`) byte-identical to the pre-port
  daemon (`8a6c4710`).
- 26B-A4B MoE: the shared ops take the eager route's certified fusions
  (qk-norm+RoPE, post-norm+residual) that the lowered path never used, so its
  logits move in the last bits. Gate: coherent decoded text and recorded
  divergence point against the pre-port daemon, same relaxation #795 applied
  to its MTP and decode-arm slices.
- EAGLE drafter: tau parity and coherent text; greedy EAGLE output equals AR
  output where it did before.
- Redline: the 26B shadow oracle (`redline_shadow_gemma4`) stays exact.
  Residual and K=V copies are `copy_f32_buffer` launches, not D2D memcpys,
  so the retained recorder sees them; a memcpy there made PM4 replay inexact.
- Measured (gfx1151, pre-port daemon `8a6c4710`): every 12B/E2B/E4B row
  bitwise; 26B with the fused qk-norm+RoPE and post-norm routes off
  byte-identical (measured before those switches were removed), with them on
  2 of 5 prompts identical and 3 diverging late, coherent; EAGLE per-prompt tau
  3.368/3.459/2.667/3.447/3.514 before and after; 26B
  `redline_daemon_harness.py --pm4 --skip-prefill` shadow exact (1083
  launches, was 1263; tape hash `249f24e5f39d425d`, was `0909f3792962c37d`).
- `serve_harness.py --thinking off --sampling greedy --compare-transcript`
  against the pre-port daemon: 12B battery and chain, E4B battery
  `transcript_byte_identical=true`; 26B battery coherent (no runaway, empty or
  attractor turns), turn 2 diverges.
- Calibration tools on the program (gfx1151, `gemma4-12b.mq4`):
  `prefill_parity_gemma4` (725-token prompt) batched vs per-token same argmax
  and identical 24-token continuation on the 12B (8x faster prefill) and the
  26B-A4B q8-experts MoE (2.4x); `calib_sweep` coverage 328/328, Hessian
  consistency 0, q/k/(v) and gate/up identity PASS; `eval_hipfire` prefill vs
  per-token scoring within 0.3% KLD. After the port the serve gates above
  still hold: 12B/E2B/E4B tokens and top-16 logits bitwise, EAGLE tau
  unchanged, 26B text identical to the first port, Redline shadow exact with
  the same tape hash `249f24e5f39d425d`.
- KLD against a BF16 reference (`google/gemma-4-12B-it` `--format oracle`,
  pre-port `build_kld_ref_native_gemma4`, WT2 test, 8 x 512, top-256, ref md5
  `ea5e297378c7ff692013af63d8411323`), candidate `gemma4-12b.mq4`, Q8 KV:
  per-token scoring 0.7338 pre-port vs 0.7335 now; prefill scoring 0.8150 vs
  0.8242 (prefill wall 160 s vs 110 s). The pre-port `eval_hipfire` needed
  the Gemma-branch fix to score at all.
- Rebased onto `beta` `e268a0798` (gfx1151, beta daemon md5
  `94070f1ce9ce722fcbf4dcf7cc87da58` vs rebased daemon
  `ea7b9af9f33b242a81a400486f169f02`), five greedy prompts per row:
  12B AR, 12B prefill batch 64, 12B `HIPFIRE_GEMMA4_EAGLE=1` (strict), E2B,
  E4B, E2B/E4B prefill batch 64 have byte-identical token streams and top-16
  logit traces; EAGLE (now opt-in on beta via `HIPFIRE_GEMMA4_EAGLE=1`) is
  byte-identical with per-prompt tau 3.368/3.459/2.667/3.447/3.514 on both;
  26B-A4B q8-experts: 3 of 5 identical, 2 diverge late (chars 295 and 322),
  coherent. The local `gemma4-26b-a4b.mq4` produces garbage on beta for two
  reasons: its HFQ4-G128 expert `down_proj` (K = 704) was packed across rows
  by the quantizer before `b4846285e`, and its 4 MQ6G256 expert layers had no
  indexed gate/up route. Such files now refuse at load. A requantization with
  the current quantizer (MQ4G256V2/MQ6G256 gate/up, row-padded HFQ4-G128 down)
  loads and runs: coherent greedy text, serve battery with no runaway, empty or
  attractor turns, Redline `--pm4` shadow exact (1202 launches), hipGraph
  decode identical to direct decode, `prefill_parity_gemma4` same argmax and
  continuation. `serve_harness.py --thinking off
  --sampling greedy`: 12B battery and chain, E4B battery and 26B battery
  `transcript_byte_identical=true` against beta. 26B Redline `--pm4
  --skip-prefill` shadow exact on both (beta 1263 launches, hash
  `9f57d0ca48889bda`; here 1083, hash `f2675ac01238613c`; both with beta's
  default Q8 full tier), decode 38.1 vs 38.2 tok/s. `prefill_parity_gemma4`
  725 tokens: 12B and 26B batched vs per-token same argmax and identical
  24-token continuation (12B 4.2 s vs 33.8 s per-token, 26B 8.2 s vs 19.2 s).
  `calib_sweep` 12B coverage 328/328, consistency 0, identity PASS.
  `eval_hipfire` per-token 0.733489 and prefill 0.824203: both `.kldseq`
  files byte-identical to the pre-rebase branch.

## Engine seams

After the program port, Gemma 4 moved onto the seams Qwen4 introduced:

- **One stack.** `config.rs` carries the MoE fields, `gemma4.rs` loads every
  variant (pooled expert tensors and pointer tables for the 26B), and
  `Gemma4State` holds position-indexed sliding and full caches for all of
  them (the 26B's 1024-row sliding ring is gone). `lowered.rs` is deleted.
  `supports_batched_prefill` admits the 26B on Q8/Q8 loads. The decode
  hipGraph stays off when an expert `down_proj` is Q8_0 (its atomic sum is
  not replay-exact) and while a launch recorder is armed (`graph_exact`).
- **Session cache.** `Gemma4Bundle` (`bundle.rs`) implements
  `SessionState`: a snapshot is the Q8 K/V rows of every cache slot up to its
  position (`RowStream`s, no fixed state), scoped `gemma4/rows{N}` by prefill
  width because batched and per-token prefill write different bits. Boundaries
  fall every 1024 prompt tokens. The route-generic live/plan/begin/commit
  logic is `hipfire_runtime::session_cache::SessionDriver`, shared with
  Qwen4. Each request now plans against its own prompt. Before this, a Gemma 4
  request continued the previous request's KV in the same process.
- **Spec seams.** `mtp.rs`: `Gemma4Drafter: MtpDrafter` (driven by the
  generic `MtpSpeculator`) and `SpecTarget for Gemma4Bundle`. A window at
  position `p` holds the pending seed and the target's post-norm hidden of
  `p - 1`. It drafts `k` tokens at the constant query position `p` through
  the step-list draft head (`Embed`, `Scale`, `Gemv` pre-projection, the
  query-only blocks, norm, `DraftHead` whole-vocab argmax, `Gemv`
  post-projection), verifies `[seed, drafts]` in one batched forward and
  commits the greedy prefix plus bonus. KV is position-indexed, so
  `commit_prefix` only moves the cursor. `emit.rs` `GemmaSpecEmit` routes
  the thought channel, and `generate_gemma4_spec` wraps the engine's
  `generate_spec`. The inline EAGLE loop and `speculative.rs` are deleted.

Measured (gfx1151, PR-head daemon md5 `83c3ca2c175e0ac66b47ce1802e8f815`
vs `7cb5f3aa83272fd5283628332ad4e29d`, five greedy prompts, 128 tokens, a
daemon `reset` between prompts on both arms so neither carries the previous
request):

- 12B AR, 12B prefill batch 64, E2B, E4B, E4B prefill batch 64: token
  streams and top-16 logit traces byte-identical.
- 26B-A4B q8-experts and requantized MQ4: text identical on all five
  prompts. Decode 36.1 vs 36.4 tok/s (q8) and 43.2 vs 43.9 (MQ4), measured
  alone. Runs overlapping a cargo build read 10–15% lower on either binary.
- EAGLE (12B, `HIPFIRE_GEMMA4_EAGLE=1`, `spec` 2, 3 and 5): four of five
  prompts equal AR. The fifth (p1) leaves AR at step 91, where AR's top-1/top-2
  margin is 0.021 logits (26.2946 vs 26.2735), the smallest in that sequence.
  Batched verify logits differ from single-row decode by about 1e-2 from the
  first verify row on. Strict mode does not make them bitwise-equal on
  gfx1151, so greedy EAGLE equals AR only where the margin exceeds that
  noise. The PR-head loop matched AR on all five prompts. It re-verified the
  last prompt token, so its KV differs from this route's and the near-tie
  fell the other way. Per-prompt tau at `spec` 3: 3.342/3.175/2.592/3.528/3.024
  vs PR head 3.368/3.122/2.633/3.639/2.977. The first token now comes from
  prefill, and each window pairs its seed with the hidden that predicted it.

## Progress

| Piece | State | Commit |
|---|---|---|
| Eager decode + batched prefill/verify + E-series PLE as `[SandwichAttention, SandwichMlp, PerLayerInput?, Scale?]` (`program.rs`, dispatch `sandwich.rs`); hand arms deleted | landed; bitwise | `30b55f753` |
| MoE decode (`ParallelMoeMlp`), lowered decode on the shared program; super-op facade (`lower_variant`, `Gemma4Bindings`) and lowered hand arms deleted; unsupported expert formats refuse at load | landed; 26B byte-identical with the two non-bitwise fusions off, coherent with them on | `287e67d5b` |
| EAGLE draft head as one step list (`Gemv` pre-projection, query-only `[SandwichAttention, SandwichMlp, Scale?]` blocks over the target's last slot, norm, `lm_head`, post-projection) | landed; EAGLE text and per-prompt tau identical | `287e67d5b` |
| Calibration tools (`calib_sweep`, `eval_hipfire`, `prefill_parity_gemma4`) on the batched program; dispatch-owned calibration taps; batched `ParallelMoeMlp`; old batched prefill, `HIPFIRE_BATCHED_PREFILL`/`HIPFIRE_WMMA_PREFILL` deleted | landed; tool parity above | `e89080332` |
| One weight stack for every variant; `lowered.rs` deleted | landed; see "Engine seams" | this change |
| Engine session cache (`SessionState`, `SessionDriver` shared with Qwen4) | landed; see "Engine seams" | this change |
| EAGLE on `MtpDrafter`/`SpecTarget`/`SpecEmit` + `generate_spec`; `DraftHead` | landed; see "Engine seams" | this change |

## Remaining

- Batched verify is not bitwise-equal to single-row decode on gfx1151, so
  greedy EAGLE can leave the AR stream at near-tied logits. EAGLE stays
  opt-in until the verify and decode arithmetic match.
- In the session scenario a follow-up turn restored the newest snapshot
  instead of extending the previous turn's live state, so its rendered
  history is not a token-exact extension of that turn (assumed cause: the
  reply is re-rendered from text; Qwen keeps a verbatim assistant-turn cache).
