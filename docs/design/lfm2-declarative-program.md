# LFM2 declarative layer program (#666 G10)

Status: single-lane decode cut over (dense LFM2.5 and LFM2.5-8B-A1B MoE).
The continuous-batching decode (`forward_batch.rs`) is not cut over yet.

## Goal

LFM2 declares each decoder layer as typed `hipfire_dispatch` steps, as Qwen4,
Qwen3.5 (#795) and Gemma 4 (#813) do. The architecture crate binds resident
weights, KV and convolution state and scratch; dispatch owns every executor
body and route choice.

`crates/hipfire-arch-lfm2moe/src/program.rs` builds three step lists:

| Block | Steps |
|---|---|
| Conv mixer (LIV) | `RmsnormAutomatic` → `Gemv` (in_proj) → `GatedShortConv` → `GemvResidual` (out_proj) |
| Attention mixer | `RmsnormAutomatic` → `Gemv` ×3 → `QkNorm` ×2 → `Rope` → `Attend` → `GemvResidual` |
| Dense FFN | `RmsnormAutomatic` → `Gemv` (w1, w3) → `SiluMul` → `GemvResidual` (w2) |
| MoE FFN | `Moe` (sealed call: norm, router projection, biased sigmoid top-k, experts, combine) |
| Head | `RmsnormAutomatic` → `Gemv` (tied lm_head) |

`forward.rs` keeps only the lifecycle around the program: token/position
staging, hipGraph capture (`HIPFIRE_LFM2_GRAPH`), Redline retained replay
(dense only), and the oracle capture of per-layer hidden states.

Removed: the hand decode loop, the #397 super-op path (`Lfm2MoeBindings`,
`lfm2_lower_variant`, `HIPFIRE_FORWARD_LOWERED` for LFM) and the hand MoE
block.

## New shared pieces

- **`Step::GatedShortConv` (`GatedShortConvOp`)**: the double-gated causal
  short convolution with its rolling `[rows, channels, K-1]` state, over any
  number of independent lanes (`conv1d_gated_decode_f32`). Model-neutral: the
  batched decode can bind the same op with `rows = B`.
- **`Step::SiluMul`**: the one-row SwiGLU activation between the gate/up and
  down projections.
- **Biased sigmoid router in the sealed MoE (`MoeParams::selection_bias`,
  `MoeSelectionBias`)**: experts are selected on `sigmoid(logits) + bias`,
  weighted by the unbiased scores, normalized, and scaled by `route_scale`
  (`deepseek4_moe_topk_bias_aware_f32`). With a bias the sigmoid recipe takes
  any width `1..=8` and routes on the GPU; only the uniform FWHT `MQ4G256` and
  `MQ6G256` expert formats are admitted, because their indexed gate/up,
  activation, down and combine kernels read the width at run time. The
  unbiased sigmoid route (Cohere) stays top-8. DeepSeek4 (G9) uses the same
  selection.

LFM2 MoE layers seal their resident experts at load (`MoeFfn::seal`: one
`ExpertTable` row per expert, live pointer tables bound once). A layer with
mixed expert formats, or with the layer-wide expert AWQ sidecars the old hand
path read (no hipfire quantizer writes them), is refused at load instead of
running on a path that would drop them.

## Numerics contract

Every projection reads the plain RMS-normed input (`GemvInput::Raw`) and
rotates it for its own format, as the pre-port route did. This keeps every
kernel and reduction order of beta's route: the fusion table (fused QKV,
fused gate/up, fused rmsnorm+FWHT) is not admitted. An earlier revision that
fed prerotated inputs took the Q8 fused QKV and gate/up kernels; its logits
moved by up to 0.16 (350M Q8) with the same greedy tokens. Admitting those
fusions is a separate, measured change.

## Evidence (gfx1151, halo)

Base: beta `1be80ed54`; head: this branch rebased on it. Route oracle: a
throwaway dump driver (`~/mesh-validation/HIPF-14/lfm2_route_dump.rs`, built
unchanged in both trees) runs `decode_step` over a fixed 24-token prompt plus
48 greedy tokens and records at every committed position the fed token,
argmax, the full logits and digests of all convolution state and all KV
bytes, plus the final state bytes. Beta is deterministic run to run.

| Fixture (sha256) | Positions | Tokens | Logits | Conv state | KV |
|---|---|---|---|---|---|
| `lfm2.5-350m.q8` (`7cd38949…`) | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-350m.mq4` (`4885d1ce…`) | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-1.2b.mq4` (`a7aeba87…`) | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-8b-a1b.mq4` (`07da9942…`, MoE, sealed route) | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-vl-3b.mq4v2.hfq` (`ce009cd0…`, MQ4G256V2 text, VL tower) | 72 | identical | bitwise | bitwise | bitwise |

The VL fixture is `LiquidAI/LFM2.5-VL-3B` at revision `35a118d9`
(safetensors sha256 `8413ba08…`) quantized with
`hipfire-quantize --format mq4v2 --include-vision` (census 167 MQ4G256V2,
99 Q8F16, 441 vision/projector F16/F32). The hipGraph route
(`HIPFIRE_LFM2_GRAPH=1`) is bitwise on it too, beta vs head and vs the
default route.

LFM2.5-8B-A1B output is incoherent on beta until #878 (MoE down GEMV read
2 of 7 groups at K = 1792) lands. Every 8B row below that needs coherent text
therefore compares beta + #878 + #879 against head + #878 + #879; the route
oracle is bitwise there as well (and differs from the unfixed arms, so it
sees the kernel change).

Against the HF `transformers` CPU f32 reference on the bf16 checkpoints
(per-layer hidden states, chat-framed prompts; base and head dumps are
bitwise equal, so this is a quantization-noise check, not a port check):

| Model | Prompt | Layer 0 cos (last pos) | Last layer cos (last pos) | Greedy |
|---|---|---|---|---|
| 8B-A1B + #878 | France / hello / ocean | 0.983 / 0.983 / 0.983 | 0.925 / 0.881 / 0.936 | coherent think + answer, beta == head |
| VL-3B (text) | France / ocean | 0.988 / 0.988 | 0.987 / 0.974 | "The capital of France is Paris." equals HF greedy; beta == head |

Serve (`serve_harness.py --thinking off --sampling greedy
--compare-transcript`): `transcript_byte_identical=true` for battery and
chain on `lfm2.5-350m.q8`, `lfm2.5-1.2b.mq4`, `lfm2.5-vl-3b.mq4v2.hfq` (beta vs
head) and on `lfm2.5-8b-a1b.mq4` (fix arms, `--max-tokens 1024`: 5 / 5 turns
with a closed think block and a coherent answer, 0 empty, 0 attractor).

VL image route (`prefill_embed_step`, daemon protocol, greedy, 128 tokens):
`doge_napping.png`, `scene_2.jpg` and `barney_cigar.jpg`, two prompts each,
token streams identical beta vs head. The doge OCR turn reads all six meme
captions; scene_2 reads Yves Saint Laurent / OPIUM / Ermenegildo Zegna.

Lifecycle (`dense_rollback_gpu`, G4.10 seams: fault after prefill and after
the first decode, one error terminal, rollback, retry equals fresh,
unload/reload, reset-phase error visible): 3 / 3 on `lfm2.5-1.2b.mq4` on beta
and head. On 8B-A1B the after-first-decode fault never fires without #878
(beta and head alike); with #878 the head passes.

Redline (`redline_daemon_harness.py --pm4 --skip-prefill`): decode launch
sequence unchanged and PM4 shadow exact on both arms, `lfm2.5-350m.mq4`
(280 launches, hash `de9b61e3ebd1de33`) and `lfm2.5-350m.q8` (220 launches,
hash `0cc5dc6b514f31c9`).

Decode, A/B alternating under the halo lock, no other cargo/rustc running:

| Model | Measure | beta | head | ratio |
|---|---|---|---|---|
| `lfm2.5-1.2b.mq4` | `hipfire bench --matrix` tg128 @ ctx 128, 2 runs | 232.2 / 232.0 | 231.6 / 231.8 | 0.998 |
| `lfm2.5-1.2b.mq4` | same @ ctx 2048 | 125.78 / 125.76 | 125.69 / 125.68 | 0.999 |
| `lfm2.5-vl-3b.mq4v2.hfq` | same @ ctx 128 | 119.07 / 119.24 | 118.72 / 118.87 | 0.997 |
| `lfm2.5-vl-3b.mq4v2.hfq` | same @ ctx 2048 | 75.31 / 75.38 | 75.22 / 75.32 | 0.999 |
| `lfm2.5-8b-a1b.mq4` (+#878) | 256 greedy tokens via `decode_step`, 4 rounds | 154.5 / 154.3 / 153.9 / 153.4 | 154.3 / 153.9 / 154.6 / 154.4 | 1.002 |

The sealed MoE first ran the 8B-A1B gate/up through the top-8 launcher
(`gemv_hfq4g256_moe_gate_up_k8_indexed`): bitwise identical, but 1.2 % slower
at k = 4 on gfx1151. Below top-8 (only the biased route reaches those arms)
the executor now takes the batched HFQ4 and HFQ6 gate/up launchers the hand
path used.

Not covered: the `.mq6e` MQ6-expert artifact (not published, and the current
quantizer has no MQ6-expert recipe for LFM2; the sealed route calls the same
MQ6 kernels with the same arguments as the hand path, but it was not run) and
the gfx1201 retained-PM4 admission row in `docs/admissions.yml` (needs
gfx1201; the gfx1151 launch sequences are unchanged).
