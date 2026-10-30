# Multi-slot engine — remaining gap closure plan

**Status:** execution record. The closure pass on 2026-10-22 covers: post-image
text continuation, DFlash2 on VMM, swap-in prefix reuse, stop sequences, the
VMM host-snapshot prefix cache, permissive `response_format: json_object`,
slots `logprobs`, batched prefill for dense VMM slots, sampled MTP/DFlash,
multi-image requests, and `n`-choice fanout. The first seven are recorded in
[`2026-10-22-vmm-kv-multislot-engine.md`](2026-10-22-vmm-kv-multislot-engine.md)
§12; the rest are §2, §4-§6 here and §12.5 there. Remaining items are
validation cells, not unimplemented engine/API gaps.

**Baseline:** `feat/vmm-kv-multislot` at `2a3bc44b8` plus the working-tree
changes; gfx1100 (RX 7900 XTX), HIP 7.16, q8 KV, Qwen3.5-4B / Qwen3.8-27B
fixtures. `legacy` remains the default KV backend and the multi-slot route is
experimental.

---

## 1. VMM cross-session prefix cache (P0) — CLOSED

**Implementation:** `crates/hipfire-arch-qwen35/src/serve_vmm_prefix.rs` plus
capture/lookup hooks in `serve_engine.rs`. The selected v1 design is a bounded
**host-snapshot LRU**, not physical-page sharing:

- entry = `(token prefix, SlotSnapshot)`; the snapshot is the exact
  per-FA-layer KV prefix + DeltaNet state from `VmmSlotKv::capture_slot`;
- capture after a successful prefill chunk, only while a suffix remains and
  only past the per-slot `vmm_prefix_mark` watermark;
- cold hit restores KV and recurrent state into the slot's own owner,
  publishes `pool.set_seq_len(n)`, and prefills only the suffix; a failed
  restore resets the state and forgets the entry;
- LRU bounded by `serve.prefix_cache_max_bytes`; no owner churn, no retired
  VA, no page-table aliasing;
- `reused > 0` retires MTP/DFlash for the turn.

**Verified:** `scripts/check_slots_vmm_prefix.py` on the 4B fixture, 2 slots ×
2048, prefill chunk 512, 256 MiB ceiling. A 981-token shared prompt produced
STORE entries at 512 and 896 tokens (48/103 MiB); the second request reused
**896/981** tokens (`cached_tokens=896`), its greedy text was byte-identical
to a cache-disabled cold control, and a different-prefix request reported
`cached_tokens=0`. Host and runtime-container runs pass. Unit tests for
longest-prefix lookup, token/stamp misses, LRU eviction, and invalid-entry
rejection pass (`cargo test -p hipfire-arch-qwen35 --lib serve_vmm_prefix`).

**Not in v1:** physical granule sharing via `capture_vmm_prefixes` /
`fork_vmm_prefixes` / `map_shared_prefix`. That trades shared physical bytes
for VA churn and per-hit owner recreation; revisit only if measured host
memory pressure justifies it. The host-snapshot cache is the safe baseline.

## 2. Batched prefill on VMM (P1, perf) — CLOSED

**Implementation:** `forward_step_vmm` routes dense, non-spec prefilling
slots' prompt rows into the same descriptor-addressed `_vmm` batched step as
decode rows. MTP/DFlash prompt chunks stay on the singleton body because their
private hidden capture is not ported to the batched pass; **MoE/A3B prompt
chunks also stay singleton** because the `_vmm` batched prefill path hung
before the first sampling step on a 35B-A3B run (that path is not validated
for the MoE branches). The batch
therefore carries up to `n_slots × prefill_chunk` prompt rows; `positions[]`
stays the causal authority, and `final_logits_per_slot` produces the final-row
logits the engine samples. Prefix-cache capture and logprobs read the same
post-step state as before. A trace line reports the prompt rows batched per
step (`VMM batched prefill: N prompt rows in an M-row step`).

**Measured** (gfx1100, 4B MQ4, 4 slots, 953-token committed prompt, 4
concurrent streams, greedy, warm kernel cache): per-stream TTFT median
`1239 ms` on VMM vs `1351 ms` on legacy in one interleaved pair, and
`2072 ms` vs `2042 ms` in the other; the direction is not consistent and the
difference is within the run-to-run spread. The trace confirms four slots'
prompt rows (2048/1536/228) are in one step. The oracle ids are unchanged from
the singleton-prefill baseline, and the prefix/vision scripts pass. Slice 5 is
therefore closed with the batched path, not with a new kernel; decode is
unchanged.

## 3. Slots `logprobs` (P2, API) — CLOSED

**Implementation:** `SubmitRequest.logprobs_top_k: Option<usize>` (Some(0) =
chosen log-prob only). Requests with logprobs force AR (MTP/DFlash disabled and
jump-forward skipped) so each committed token has its own sampling row. After
GPU sampling, the engine downloads that slot's `logits_out` row once per
token, computes `saddle_core::logprobs::{logprob_of, top_k_logprobs}`, and
attaches `logprob`/`top_logprobs` to `Event::Token`. The daemon emitter routes
them to the existing OpenAI token envelope (candidate token text and bytes via
its tokenizer); the gateway already consumes those fields. The sequential-VL
path computes the same fields from the host logits it already samples.

**Verified:** `check_slots_vmm_vision.py --spec off|mtp` (4B): a greedy
`logprobs: true, top_logprobs: 5` request returned a token entry with a finite
log-prob and a 5-candidate list; `top_logprobs` without `logprobs` is a 400.
Host and runtime-container runs pass. The D2H copy is opt-in and documented;
a future sampler-kernel path can remove it. Multi-byte UTF-8 uses the
tokenizer's `bytes` field, matching the sequential route.

## 4. Sampled speculative decode (P2, correctness-sensitive) — CLOSED

**Implementation:** the slot engine no longer forces sampled requests to AR.
`spec_request_config` installs the request's temp/top-p/top-k/min-p and RNG
seed on whichever MTP/DFlash state exists, and re-applies it when prefill
lazily allocates that state. MTP's batched verify calls the same
`mtp_sampled_accept` used by the singleton (accept `u*p_d <= p_t`, residual
bonus on rejection, target-nucleus bonus on full accept). DFlash's slot path
now asks the selector for its sampled proposal (`selected_probabilities` +
top-K q rows), and the verify runs the singleton
`dflash_sampled_accept_rows` rejection loop over the trunk verify logits.
Greedy requests keep the old argmax path unchanged. The opt-out is the
existing `speculation.mtp_sampled` / `HIPFIRE_MTP_SAMPLED` switch: with it
off, sampled requests fall back to AR exactly as before (verified: temp=1.0 +
`HIPFIRE_MTP_SAMPLED=0` produced no MTP trace/window).

**Evidence:** 4B MTP, temp 1.0, seed 42: 16 sampled windows fired
(`[mtp-trace] result ...` with no greedy `argmax=` line), coherent
STOP-at-96 output. Qwen3.8-27B paired DFlash2, temp 1.0: `[dflash-trace]
sampled accept` lines with accepted 0..15, coherent paragraph. Greedy MTP and
greedy DFlash still take the argmax branch (separate runs).

**Not claimed:** byte equality with AR. The host acceptance/residual/bonus
code is the already-validated singleton implementation; this pass adds the
slot wiring and verifies live distribution-path execution, not a new 100k-draw
TV oracle. The existing `spec_sampling` and `dflash_sampled_accept_rows` unit
tests remain the statistical evidence for the acceptors themselves.

## 5. Multi-image requests (P2) — CLOSED

**Implementation:** `VisualData` carries an ordered `Vec<VisualImage>` (each
with its own patches/grid/post-merge count). The daemon accepts up to 8
base64 images (`images_base64`, request order), splices one
`<|vision_start|>…pads…<|vision_end|>` block per image, and builds one
M-RoPE table with a span per image. The engine encodes each image on its own
grid and concatenates the embedding matrix in the same order; the VMM
sequential-VL path and the arena batched path both handle the list. Single
image keeps the historical `patches`/`grid_*` fast path.

**Evidence:** 4B + `.vl`, `doge.jpeg` + `scene_1.jpg` in one request:
“second image?” → “street signpost with directions to various Paris…”;
“first image animal?” → “dog”. Both cold requests, greedy, no image-pad or
M-RoPE error. Unit tests cover two pad runs and mismatched span counts.

**Limitation:** images are all placed before the question text; arbitrary
interleaving of text/image parts inside one user message is not reconstructed.
Request order is stable and documented. Multi-image is a multi-slot route
capability; the standard route keeps its one-image wire and returns a typed
400 for a second image (verified).

## 6. Remaining refused API fields (P3) — CLOSED for n; echo/suffix N/A

**`n` implemented:** `/v1/chat/completions` accepts `n` 1..=8. The gateway
fans each choice into its own daemon attempt (sequential admission; distinct
sampler streams), reuses one response id, sets `choice.index`, and sums
completion usage. Streaming emits each choice as its own sequence of frames on
the same SSE body (choice index in every frame, one final `[DONE]`), with
distinct seeds. `n > 1` on the Anthropic/Responses adapters remains a typed
400 because those protocols do not define it. `n` outside 1..=8 and
non-integers are 400s.

**Evidence:** non-stream `n=3`, temp 1.0: three choices with indexes 0/1/2,
distinct reasoning text, one usage object (completion tokens summed). Stream
`n=2`: indexes {0,1}, `finish_reason=stop` for both, exactly one `[DONE]`,
contents `Blue` / `Red`.

**`echo` and `suffix`:** these are legacy `/v1/completions` fields, not
fields of `/v1/chat/completions`. The chat route continues to refuse them with
a typed 400 (the refusal list is an honest capability advertisement); adding
them to chat would be a non-OpenAI semantic. `best_of` and `logit_bias` remain
refused and are not part of this multislot storage/serving pass.

## 7. Validation matrix (P1/P2)

| Cell | Route | Status |
|---|---|---|
| prefix cache, host snapshot | 4B, shared 981-token prompt | **PASS** (`check_slots_vmm_prefix.py`: 896 reused, warm==cold, negative miss) |
| stop sequences | 4B, non-stream + stream | **PASS** (`finish=stop`, no stop bytes, next request served, malformed 400) |
| swap eviction/restore | 4B, 3 conversations / 2 slots | **PASS** (reused 31/61 after `swap-in`) |
| post-image text follow-up | 4B + `.vl`, continuation + MTP | **PASS** (reused 360/388, answer `Dog`) |
| DFlash2 VMM | paired Qwen3.8-27B + DFlash2 sidecar | **PASS** (52/52 arena/VMM windows; coherent serve one-liner) |
| batched VMM prefill | 4B, 953-token prompt, 4 concurrent streams | **PASS** (2048/1536/228 prompt rows batched; TTFT within run spread of legacy) |
| state oracle, 4 lanes | 27B, `test_slots_vmm_oracle`, graph off/on, DFlash on/off | **PASS** (first-token isolation, serial==singleton, burst==serial; DFlash windows fire on both routes) |
| state oracle, 4 lanes | 4B, graph off | **PASS** (same four checks; final `efa7277a…` daemon reproduces the earlier ids) |
| container route | `docker compose -f docker/compose.vmm-9001.yaml`, 4 slots, VMM, vision + MTP | **PASS** (single image → `Dog`; plain text → `Pong!`; two-image request → `Dog`; streaming `n=2` → indexes 0/1, `Blue`/`Red`, one `[DONE]`, aggregate usage; container log `slot KV tier Q8 on vmm`) |
| chunk boundary | 848-token prompt, chunk 512 | **PASS** (`ZEBRA-42` recall) |
| `json_object` | 4B | **PASS** (returns a JSON object) |
| `logprobs` | 4B, greedy + `--spec mtp` | **PASS** (finite chosen log-prob, 5 candidates; top_logprobs without logprobs → 400) |
| fp8 KV on VMM | gfx1201, `--kv-mode fp8 --kv-backend vmm` | blocked: host is gfx1100 |
| 9B dense | `qwen3.5:9b`, 2 slots × 2048, AR | **PASS** two concurrent coherent turns. The published 9B DFlash sidecar is a legacy (no selector) draft, which the slots route refuses by design: DFlash slots require DFlash2 |
| MoE/A3B | `qwen3.6-35b-a3b.mq4r`, 1 slot × 1024 | **PASS** short prompt (`pong`); prompt chunks stay on the singleton prefill path because the `_vmm` batched prefill route hung before sampling on this model |
| second arch | gfx1201 / gfx1151 | not run |
| long context | 4B, 4096-token slot cap, 3787-token prompt | **PASS** exact `NEEDLE-ALPHA-7731` recall at 3787+32 tokens; 6025-token prompt returned the typed over-cap 400 |
| budget-driven reclaim | 27B, 4 slots × 262144, budget 5700 MiB; request asking 200k positions needs 6.9 GB | **PASS** typed 503 `KV capacity` refusal; a subsequent `pong` request succeeded. The refusal's `retryable` bit was `false`; fixed to `true` for `RejectClass::Overload` |
| PM4/Redline parity | standard VMM route, 4B, `--pm4 --skip-prefill` | **PASS** (`pass=True`, stable decode hash, PM4 exact, GDN frame exact). The multi-slot executor does not use the singleton PM4 replay path; this verifies the VMM descriptor route it shares |
| sampled MTP / DFlash | 4B MTP and 27B DFlash2, temp 1.0 | **PASS** sampled windows fire (`mtp-trace` without greedy argmax; `dflash-trace sampled accept`), coherent output; greedy remains on the argmax path |
| multi-image | 4B + VL, doge + Paris scene | **PASS** first/second image questions answered correctly from one request on both the VMM sequential-VL and arena batched-VL paths; two pad runs and two MRoPE spans |
| `n` choices | 4B, non-stream n=3 and stream n=2 | **PASS** indexes 0/1/2, distinct choices, summed completion usage; one `[DONE]`, indexes 0/1 streamed |
| serve battery/chain | `scripts/serve_harness.py --kv-backend vmm`, 4B, 2 slots | **coherent, not terminal-pass** (5/5 turns coherent in each mode, no empty/attractor/retrieval miss; all turns were `finish=length` at 64 and the harness flags `!RUNAWAY`; chain showed `cached=107/230/323/424`) |

Final binaries for this pass: `hipfire` `658b40b77265da2246b4880a426524dc`,
`daemon` `98a1e7fc5e2c765bba24794219d74b1a`. Unit routes:
`hipfire-arch-qwen35` 337 passed / 24 ignored, `hipfire-runtime` serve 3,
`hipfire-daemon` 47, `hipfire-cli` 443, all with 0 failures on the final
source.

Each cell needs a fresh process, decoded text eyeballed, model/binary md5s,
and the route named in `docs/VALIDATION.md`. Container runs must mount the
host-built binaries for a source change until the image is rebuilt.

## 8. Ordering and exit criteria

1. Validation matrix — fp8 and a second arch remain blocked by hardware.
   Budget refusal is now exercised (typed retryable 503, engine survives).
2. A sampler-kernel logprobs path (remove the opt-in D2H fallback).
3. `logit_bias`/`best_of` remain intentionally refused chat fields; they are
   not part of this storage/serving pass.

Promotion of the VMM route out of experimental requires: §2 closed, every cell
in §7 green or blocked with a named reason,
`legacy` fallback unchanged, and a written default-backend decision. Until
then `kv_backend=vmm` stays opt-in.
