# Changelog

## Unreleased (0.4.2)

- **Batched exact DFlash draft across VMM continuous-batching lanes (`HIPFIRE_CB_DFLASH_DRAFT_BATCH=0` kills it, developer var, default on, byte-identical):** the draft of every planned full-block DFlash lane ran serially (8 lanes ≈ 61 ms/step on the 27B mq4-xts target + qwen3.8-27b-dflash.mq4v2 draft, gfx1201). Full-block lanes now share one draft-model forward (`hipfire_runtime::dflash::draft_forward_lanes`) and one target lm-head GEMM per chunk of ≤ 48 rows (three 16-row lanes), keeping each lane's own context K/V rings, positions, hidden context, attention and (DFlash2) causal convolutions, and the per-lane candidate selector; budget-tail lanes, lone lanes and non-exact machines/drafts (non-gfx12, non-MQ4v2 weights, MoE) keep the singleton `dflash_lane_draft` call. Eight lanes now draft in ≈ 36 ms/step. The CB state oracle (`cb_vmm_state_oracle --spec dflash --phase probe`) gains batched-draft cases for 1..4 lanes (tokens + full draft/ring state compared with the isolated singleton draft) plus the 8-lane serial-vs-batched timing.
- **gfx1201 MQ4V2 verify-tile GEMMs for 16 < N < 64 rows (`HIPFIRE_WMMA_BATCH_TILES`, default on, byte-identical):** the multi-request speculative-verify trunk (several requests' 4-/16-row blocks packed into one forward) ran the one-tile WMMA GEMMs, whose every block re-streams the whole activation panel for 16 weight rows, so qkvza/qkv/gate-up/residual/lm_head cost grew ~linearly with N. New `*_mq4g256v2_wmma_gfx12_vt{2,3,4}w{4,8}` kernels (`gemm_qkvza`, `gemm_qkv`, `gemm_gate_up`, `gemm_mq4g256v2_residual`, which is also the batched lm_head) run 4 or 8 waves per block over the whole batch panel, stage each 128-K slab in LDS once per block and feed up to four independent accumulators per wave. Each (weight row, batch row) output keeps the one-tile kernel's single WMMA chain, K order, headers, dequant and epilogue, so every row is byte-identical at any N (probe memcmp, N = 17..63). On an R9700, 27B kernel time at N = 63: gate/up 650 -> 304 us, lm_head 5.39 -> 2.06 ms, residual-down 275 -> 186 us; an 8-request x 4-row MTP verify trunk step went 62.3 -> 48.2 ms. N <= 16 and capture/replay keep the one-tile kernels; `HIPFIRE_WMMA_BATCH_TILES=0` restores them. Compiler-less installs need the four new `gemm_*_mq4g256v2_wmma_gfx12_vt` modules in the gfx1201 pack.
- **VMM continuous batching (experimental, default off, non-exact):** `serve.vmm_batch` / `HIPFIRE_SERVE_VMM_BATCH=1` together with `serve.batch_nonexact` / `HIPFIRE_SERVE_BATCH_NONEXACT=1` batches concurrent eligible TP1 Qwen3.5-family requests over the one resident weight set with per-request VMM KV/DeltaNet owners, planned by the runtime `BatchPlanner` (decode first, rotating prefill quantum within `serve.max_batch_tokens`) and executed through the arch VMM executor. The route is not singleton-exact (different prefill/decode kernels), hence the mandatory nonexact opt-in; the loaded ack reports `continuous_batch_route: "vmm"`, `continuous_batch_nonexact: true`, `continuous_batch_exact: false` and the actual owner receipt. Stage 1 serves greedy requests only and AR rows only; a request alone when it arrives keeps the unchanged singleton route (native MTP/DFlash included), and arrivals during a singleton generation wait for it. `serve.batch_spec` (`HIPFIRE_SERVE_BATCH_SPEC`) is a registered, not yet active, switch. Flags off, load/route/ack are unchanged. No throughput claim.
### Multi-slot engine (experimental, `serve.multi_slot`)
- **Multi-slot: quant-recipe admission is per SITE, not per layer.** A fused projection group (DeltaNet QKVZA, FullAttn QKV, FFN gate+up — dense AND MoE attention bodies) keeps its fused launch when every member shares one container, and any admitted mix inside the group falls back to per-weight plain GEMMs with each member reading the activation variant its own dtype wants (rotated consumers read the FWHT-rotated buffer, unrotated read the plain normed buffer; AWQ sidecars refuse mixed groups). Residual roles (wo, w_down) and the lm_head dispatch on their own dtype. This makes every `--tier` recipe load on slots — including `--tier pro` (MQ4G256V2 body + Q8_0 wo/embed/lm_head/conv1d), previously refused with "neither uniformly Q8_0 nor a uniformly batchable MQ-family dtype" — plus `--fixed-tier` mixes and `--kmap-dense`. Admitted projection containers are listed in QUANTIZE.md § "Multi-slot recipe admissibility"; Lloyd-V2/PARO/E8/G128 containers still refuse with named errors.
  - Review fixes folded in before GPU testing: the wo/w_down rotated residual path now resolves its kernel through an explicit per-container table (`slots_residual_gemm_key`) instead of the dispatch-level wildcard that would have silently decoded 200 B/group MQ6G256 and 104/112 B/group MQ3 headers through the 136 B/group HFQ4 v1 kernel; uniform-Q8 QKVZA/QKV groups degrade to the reference's plain-GEMM fork off WMMA archs and under `HIPFIRE_Q8_PREFILL_WMMA=0` (previously they silently took the WMMA-only fused kernel); `fused_gate_up_key_for` gained explicit Q8_0/HFQ6/MQ3/Lloyd arms (its HFQ4 catchall was a latent silent-misdispatch). New CPU unit tests lock all three invariants.
  - Verification (gfx1101): `saddle-lab/examples/test_forward_slots_golden.rs` reworked into a per-KV-mode matrix (legacy slab + paged pool) — q8/fwht2/fwht3/fwht4 at 0.000x tolerance (10/10 slot-steps each, n_slots 1..=4), paged fwht{2,3,4} at 0.000x, per-mode negative controls failing correctly at ~6000x; a fresh `--tier pro` MiMo-9B artifact passes 20/20 at 0.000x on q8 and fwht3. The negative control itself was vacuous before (with DECODE_STEPS=0 its "expected failure" was an index-out-of-bounds panic) — it now runs an explicit decode-step corruption and asserts the failure is a tolerance mismatch. f16 KV stays a slots-only tier (no sequential constructor exists to reference against); paged q8 skips the matrix (the WMMA attend family is slab-only, so paged q8 is numerically — not bit — comparable), both documented in the harness.
  - fp8 (E4M3) slot KV is implemented behind the same gfx1201-only gates as the sequential path (desc-aware wrappers around the shared q8 TUs, paged symbols, tier-plan geometry, fail-closed named refusal elsewhere). No gfx1201 host was available: the GPU path is marked untested-hw in code; behavior below gfx1201 is unchanged (refusal).
- **Multi-slot: bf16 KV attend now calls the same kernel as the sequential path** (`attention_bf16_kv_batched`, via a new desc-aware slots wrapper; the slots arm previously used the windowed flash tile with window=0). Kernel-level slots-vs-sequential parity for bf16 write and attend is BIT-EXACT on gfx1101 (`rdna-compute/examples/test_bf16_slots_parity.rs`: zero-base, non-zero-base, mixed two-slot, and write parity). Full-model logits still diverge ~6x a 1e-3-relative golden tolerance on one vocab element, unchanged by the arm switch — the sequential bf16 attend selection is itself a flash-mode/capture heuristic (`bf16_attend_key`), and the end-to-end root-cause is a tracked follow-up (docs/plans/slots-quant-recipe-admission.md §9); bf16 is therefore excluded from the golden matrix default until then.

- **ROCm 10.1 bring-up (HIP 7.16 / clang 24):** shared `ROCM_PATH` resolver, strict ROCm oracles (missing tools fail unless `HIPFIRE_TEST_REQUIRE_ROCM=0` explicitly marks a no-ROCm host), HIP code object v6 and a new JIT cache ABI. Remove `restrict` from both gfx1201 FP8 producer-pack helpers' cross-thread shared scratch, preserving the post-barrier row-scale broadcast without disabling load-PRE globally. Fresh-hipcc SiLU oracles select committed clang-23 or clang-24 goldens by compiler major and fail clearly for unknown toolchains; native builders retain their original DAGs. PeaceMaker's native writer identity now names the oracle-verified lld 24 layout while keeping its 0.4.1 ELF stamp and committed bundle bytes stable. Quality pins in `AGENTS.md` are re-measured on ROCm 10.1; HIP 7.15 values remain historical. Halo pins explicitly require a warm kernel cache: cold-JIT runs can differ in chunk 0, and this bring-up does not claim to fix that path.
- **Host-scaled kernel-pack builds:** compile independent modules with one compiler per worker, reserving one core per eight and bounding workers by available RAM (~1.5 GiB/job). `HIPFIRE_PACK_JOBS` overrides the budget; `scripts/build-kernel-pack.sh --jobs N` shares it across concurrent architectures without multiplying CPU usage. Kernel sources, compiler recipes and package formats are unchanged.
- **Kernels ready before the first dispatch on closed routes:** for Qwen3.6-27B MQ4G256V2 XTS on exact gfx1201 (single GPU, native fp8 K/V, Q8 DeltaNet state, default flags, AR or native MTP) the common loader compiles the route's missing kernels during the weight load with a bounded worker pool (`HIPFIRE_JIT_JOBS`; `1` is the concurrency kill switch) and loads every planned HIP and embedded PeaceMaker module before the first dispatch. `loaded` now waits for weights plus kernels, and no compile or module load happens after it on that route; an unplanned kernel fails instead. Every other route keeps lazy JIT. The daemon `profile` command now precompiles the arch's registry inventory instead of a separate hand-assembled Qwen3.5 recipe list. Cold-start/readiness only: no steady-state tok/s change and no numerical-fix claim.
- **PeaceMaker decode twins on gfx1201 by default (`kernel.pm_decode`, auto on exact gfx1201 only):** the five W1 decode projection modules (QKVZA, QKV, gate/up, multirow-r2 and residual MQ4v2) load embedded builder-emitted images instead of hipcc objects, at every load funnel and in the planned-route preload, so the route no longer compiles them. Each image is pinned by its bundle and gfx1201 ELF SHA-256 and by the incumbent HIP source digest the full-buffer oracle compared against; a mismatch is refused, never replaced with hipcc output. `HIPFIRE_PM_DECODE=0` restores the hipcc modules, which stay in the kernel packs. Other architectures ignore the key. Qwen3.6-27B MQ4 XTS on an R9700 (all A/B on one card): retained PM4 and HIP-graph greedy traces at ctx 512/8192/32768 (256 steps) have byte-identical logits, hidden, KV and DeltaNet dumps; native MTP gives identical greedy IDs and trunk KV/recurrent hashes at ~0.5K/9K/38K prompts; WT2/code24 `.kldseq` pins are unchanged (`ff88c499…`, `9b061631…`, two cards). Decode speed is unchanged: AR bench_decode median 37.17 vs 37.08 tok/s, MTP generate 71.4/55.1 vs 71.5/55.2 tok/s (3 fresh processes per arm). No speedup is claimed.
- **gfx11 IU4 opt-out packs:** package `gemm_mq4g256v2_residual_mmq` on gfx1100/gfx1151 with its exact runtime source and all Q8_1/X128 exports, so `HIPFIRE_IU4_PREFILL=0` does not need a first-run JIT for this module. The default IU4 objects, compiler flags and kernel math are unchanged.
- **Qwen4 and dense Qwen tool argument replay is on by default:**
  restores producer ordering for reordered echoes on Qwen4 and dense Qwen3.8,
  preserving typed values through one shared renderer.
  `HIPFIRE_QWEN4_TOOL_ARG_REPLAY=0` opts out; only exact extensions of the active committed
  token record are accepted. Byte-identical echoes and retired-MTP guards remain
  unchanged; uncertain candidates retain the safe miss. No measured speed claim.


## v0.4.1.1 — release draft

Public release/tag: **v0.4.1.1**; Cargo workspace/package version: **0.4.1+patch.1**. Cargo build metadata does not give this patch higher SemVer precedence than 0.4.1. Managed installs select the public Git tag with `hipfire update --tag v0.4.1.1`; update resolves Git revisions, not a SemVer latest-release ranking.

### Patch changes
- **Flash-Next MQ4 XTS naming, without changing model bytes:** the canonical filename is `qwen3.8-flash-next.mq4-xts`, with explicit tag `qwen3.8:flash-next-mq4-xts`. Upgraded clients normalize only the exact repository/size/SHA-bound entry, whether bundled, fetched or cached. Existing `flash-next`, `flash-next-mq4` and `flash-next-gptq3` wire entries retain `qwen3.8-flash-next-gptq3.mq4` through 0.4.x and at least 90 days, so old clients do not re-download. The legacy name remains valid; this is the MQ4 XTS recipe, not a new quantization.
  - The short alias `qwen3.8:flash` resolves to `qwen3.8:flash-next`; user-facing descriptions call it Qwen3.8-Flash-Next MQ4 XTS. The legacy `-gptq3` model tag still works; no incompatible hidden/deprecated wire fields are added.
  - Upgraded CLI clients automatically reuse SHA-verified local legacy bytes via a hardlink, symlink fallback, or verified old-path fallback, with no model transfer or duplicate payload copy. Legacy bytes remain available; pin mismatches and conflicting canonical files fail rather than silently downloading. Old runtimes do not gain this migration merely by refreshing their registry.
- **Tokenizer differential correctness:** HF-aligned byte-BPE whitespace boundaries and embedded Split patterns, MiniMax inverted/removed Split behavior, Unicode 9 NFC parity, North-Mini-Code right-aligned digit splitting, and LFM2.5-MoE whole-token precedence. The HF oracle stays dev-only. These fixes also affect shared Qwen/Flash-Next tokenizer paths and can change text-derived prompt IDs; fixed-ID KLD and kernel-object pins are not silently repinned or presented as tokenizer evidence. The v0.4.1 tokenizer entry below preserves the landed fixes' detailed historical evidence.
- **Nine existing DFlash helper kernels added to packs:** eight on gfx1201 and `attention_verify_gqa_gfx1100` on gfx1100, with exact runtime sources, exports, flags and profile identity; no HIP arithmetic changes. Cold-trace admission regressions cover the additions, while gfx1151/gfx906/gfx942 admission is unchanged. This closes the known dense-27B DFlash first-run JIT inventory gaps on gfx1201/gfx1100; final installed-pack zero-JIT evidence is pending below. No warmed-throughput improvement is claimed.
- **Correct `mtp_sampled` documentation:** default-on native Qwen4 sampled MTP supports repeat/presence/frequency penalties using the same AR history, penalty window and sampling policy for target verification and draft probabilities. Distribution preservation does not promise identical seeded sampled streams. `HIPFIRE_MTP_SAMPLED=0` keeps sampled requests on AR; unsupported DFlash/other-drafter penalties still fall back to AR. No sampled routing code changes in this patch.
- **Qwen4 PLE async upload defaults on:** `HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD=0` restores blocking deferred-PLE upload. Fencing, graph capture and device-token paths are unchanged. No speedup is claimed; final patch identity evidence is pending below. The v0.4.1 default-off entry remains as history.
- **`hipfire run --stats`:** one stderr footer reports authoritative committed token count, decode tok/s and TTFT, with optional τ. Unavailable metrics show `n/a`, never a wall-clock-rate substitute. Local and HTTP streaming/nonstreaming paths are supported; `--json` remains unchanged and suppresses the footer.

### Measured evidence — awaiting final candidate

This is a release draft, not a completed hardware-validation claim. Fill the table from the immutable candidate's receipts before publication; earlier warm-cache or RC4 results do not substitute for final cold-pack or PLE gates.

| Gate | Candidate / hardware / inputs | Measured result | Evidence |
|---|---|---|---|
| Tokenizer differential oracle and CPU integration | Awaiting final candidate | Not yet recorded | — |
| Flash-Next local migration and registry compatibility | Awaiting final candidate | Zero-transfer/conflict/race receipts not yet recorded | — |
| Cold installed-pack DFlash, graph and eager | gfx1201 / gfx1100; empty JIT cache | Zero-new-JIT result not yet recorded | — |
| Adjacent pack regression | gfx1151 | Not yet recorded | — |
| PLE blocking/async full-logit and session identity | gfx1151; three campaigns | Not yet recorded; no speed claim | — |
| Quality pins, serving battery/chain and stats | Final scoped card/model cells | Not yet recorded | — |
| Kernel packs and source/binary provenance | All five admitted pack architectures | Candidate SHA, asset hashes and manifests not yet recorded | — |

## v0.4.1 — 2026-10-07

### Release summary
- **Measured standing (release candidate `7dc9885eb0`, Strix Halo, release defaults, native MTP; board seconds as on Ciru's board: TC = sum of scenario durations, Hermes = sum of case wall seconds).**
  - TC70–84, three seeds (123 / 124 / 125): 23/30 at 169.87 s, 24/30 at 158.21 s, 21/30 at 162.86 s; **median 23/30 at 162.86 s**. Greedy "Original TC70-84" panel (earlier candidates): 22/30 at 127.04 s (RC3 `d1faafea0c`), 21/30 at 126.41 s (RC4c `486083f052`).
  - HermesAgent-20: pass 1 **95 at 729.79 s** (89 primary tool calls), pass 2 **96 at 782.03 s** (117 calls); smoke 100 (3/3).
  - Cold path and decode against RC3 and the shipped beta `90d90306d8` (three fresh processes per cell, release rule: median at least 2 % lower and disjoint ranges): pp8192 2,072.9 tok/s (beta 2,082.9), Ciru 32K prefill 1,776.5 (beta 1,788.2), greedy AR 33.8 (33.8), greedy MTP 58.7 (56.7), penalized sampled MTP 57.0 (RC3 55.9). No cell regresses against beta. One prompt (`humaneval_0_has_close_elements`, greedy MTP) read −5.0 % against RC3 in the first sweep; a re-run with beta, RC3 and RC5 interleaved did not reproduce it (RC5 57.5 [56.5–58.0], RC3 58.0 [57.7–59.9], beta 53.5 [53.4–53.5] tok/s). Ciru 8K/128K cold prefill were not re-run on RC5.
  - Penalties: ~free on AR (Halo 33.87 neutral vs 33.82 penalized tok/s) and at most ~1.5 % on sampled MTP (+0.74 ms per 52 ms window). The GPU prepass emits the same IDs as the host path (24/24 prompts on each arch) and is 0.76 ms/step faster than the host path on Halo AR.
  - R9700 (card B): serve gates 0 request errors across all 24 runs; cross-session (xsession) MTP on/off and chain MTP off pass with 68–78 % of prompt tokens served from cache; TC-shaped cross-session traffic runs error-free but gets no cache hits (0 of 13 turns); chain MTP on and interleave on/off fail only the schedule's cached-token expectation (a live continuation or an own-session turn reuses a chunk-boundary snapshot instead of more tokens). A 64K-token prompt at `max_seq` 262144 on the VMM backend decodes coherently.
  - CPU gates at the release candidate: nine `scripts/check-*.py` pass, the registry check passes, `cargo test --lib --workspace --locked` 4,563 passed / 0 failed / 73 ignored, and kernel idproof against beta shows 0 changed objects on gfx1100, gfx1151 and gfx1201 (one added: `logit_penalty_table` on gfx1151/gfx1201).
- **Defaults that are not bit-exact against v0.4.0** (each has an opt-out in `docs/env-vars.md`):
  - gfx1201 FN dense GDN scan (`HIPFIRE_FN_GDN_DENSE_SCAN=0`), symmetric IU4 MoE on gfx1151/gfx1201 (`HIPFIRE_QWEN4_MOE_SYM_IU4=0`), gathered F16 WMMA QSA (`HIPFIRE_QWEN4_QSA_WMMA_GATHER=0`), gfx1201 F16 WMMA HC/projection (`HIPFIRE_QWEN4_F16_WMMA_GFX1201=0`): KLD-gated.
  - Architecture-sized FN prefill chunks (`HIPFIRE_PREFILL_CHUNK_ROWS=1536` restores the old ceiling), sampled native MTP (`HIPFIRE_MTP_SAMPLED=0`), MTP + n-gram takeovers (`HIPFIRE_MTP_NGRAM=0`): output differs from the previous route.
  - Session cache and live continuation (`HIPFIRE_SESSION_CACHE_BYTES=0`): session-exact, not cold-exact.
  - Byte-level BPE whitespace tokenization: prompt token IDs change for inputs with whitespace runs before non-space text; no switch.
  - `qwen3.8:flash-next` resolves to the GPTQ3 requant; Gemma4 lowered full-attention KV uses Q8 for `auto` (`HIPFIRE_KV_MODE=legacy-asym3`).
- **Known issues:** the Flash-Next greedy chain's turn 4 can end inside its reasoning with an empty answer (model trajectory, deterministic, independent of the cache and of MTP); tool-call conversations do not reuse the session cache (re-rendered tool-call history is not a strict extension of the committed record; 0 of 13 TC-shaped cross-session turns hit on R9700); the `session_cache_hw` and `mtp_takeover_fill_hw` hardware tests run out of memory at weight load on a 34 GB R9700 (test-harness limit; both pass on Strix Halo); HermesAgent-20 pass 1 is 56.8 s slower than Halogen's published 673.02 s and 4 points lower (pass 2 is faster and higher); TC-75 and TC-76 score 0 on sampled Halo seeds.
- **Credit:** the session cache is built on @fivetide's #825 / #826.

### Tokenizer fidelity
- **Offline HF differential tokenizer tests:** a dev-only `tokenizers` 0.22.2 oracle with the pure-Rust regex backend compares exact IDs and lossless decode round-trips for 19 pinned shipped byte-BPE model entries (Qwen2/3/3.5/3.6/3.8/Flash-Next, DeepSeek4, MiniMax, LFM2/2.5, Llama, dots.ocr, Muse-Glimmer, North-Mini-Code, Maple, Ornith and VibeThinker). Every tokenizer runs all 945 corpus inputs, all 44 benchmark text prompts and 50,000 deterministic adversarial cases by default; `HIPFIRE_TOKENIZER_FUZZ_CASES` scales the run. All 19 families pass after the independent correctness fixes below. Full per-family mismatch and minimized-repro reports preserve every failure. HF `tokenizers` stays a dev-dependency, never a runtime dependency.
- **MiniMax HF Split fidelity:** `Removed` with `invert=true` now keeps regex matches and drops unmatched gaps rather than discarding the metadata regex and falling back to GPT-2 boundaries. The initial 50,989-case differential runs found 3,886 MiniMax-M2.5 and 3,768 M2.7 ID mismatches, including `n't` (HF `[2028]`, old runtime `[110,1141]`). All 106 minimized per-family mismatch-root repros are committed as fixed-ID regression inputs. The splitter adds no encode-time allocation; its mode participates in cache identity while existing isolated-split digests are preserved.
- **HF NFC Unicode-data parity:** Qwen3.5/3.8's initial 50,989-case differential runs each exposed one combining-mark ID divergence; ICU's newer Unicode tables reorder U+089B and U+1DF7 where HF `tokenizers` 0.22.2's Unicode 9 NFC tables do not. Runtime NFC now uses the HF implementation's Unicode data and a borrowed, allocation-free quick path for already-normalized input (including long combining runs); the NFC table version enters cache identity. Both minimized repros are pinned.
- **North-Mini-Code HF right-aligned digit Split:** recognize its exact `\d{1,3}(?=(?:\d{3})*\b)` stage and split Unicode digit runs from the right only at a Unicode word boundary, preserving the subsequent model-specific Split stage. The initial differential run found 18,365 mismatches in 50,989 cases, including `6789` (HF `[26,20024]`, old runtime `[23337,29]`); all 77 distinct minimized mismatch-root inputs are pinned. The specialized splitter keeps linear scanning with no new encode-time allocations or runtime regex dependency.
- **LFM2.5-MoE HF BPE whole-token precedence:** honor the model's `ignore_merges=true` by checking each full byte-level pre-token against the vocabulary before the ordinary BPE merge loop. The first 50,989-case run found 82 ID mismatches, including ` recursively` (HF `[91708]`, old runtime `[959,1274,1983]`); all 13 distinct minimized mismatch-root inputs are pinned. Vocab lookups use a stack buffer for short pieces and avoid work entirely when the flag is absent.
- **CPU encode throughput (Qwen3.8, single-thread release, median of seven post-warmup passes):** batching the 138 committed full-rendered Hermes requests takes 418 ms hipfire versus 467 ms HF (`encode`), 55 TC requests 28 versus 52 ms, and 14 TC captures 17 versus 27 ms. A 64K-token benchmark document takes 38 versus 49 ms; at 256K tokens hipfire instead takes 299 versus 211 ms. Already-normalized non-ASCII 64K input takes 9.9 ms hipfire versus 11.5 ms with eager NFC (quick path bypassed); normalization-heavy 64K synthetic text remains slow (~303 ms versus HF ~40 ms) despite exact IDs. These are workload-specific measurements, not a throughput improvement versus a previous hipfire binary. Separately, a sum of per-request medians estimates ~408 ms for one full encode of each of the 138 Hermes requests; scaling by the RC4c pass's 461,078 prompt tokens estimates ~284 ms for its 115 requests (0.03% of its 897 s wall), not a lever for the large pass-time gap.
- **Byte-level BPE whitespace boundaries now preserve the Hugging Face `\s+(?!\S)` alternative.** A whitespace run before a non-space leaves its last Unicode character for the following alternative; higher-priority newline alternatives and trailing whitespace keep their own boundaries. The runtime also reads the embedded HF Split regex rather than imposing three-digit grouping on every model. This is a CPU tokenizer correction, independent of GPU architecture or quantization.
  - The previous implementation selected one regex for every `is_gpt2_bpe` tokenizer: Qwen2 (arch 7), Qwen3 (1), Qwen3.5/3.6/3.8 dense including 27B (5), their MoE variants (6), Flash-Next/Qwen4 (16), and dots.ocr (8). Other byte-BPE families take this same runtime path; their exact pattern comes from model metadata, not their architecture ID. SentencePiece/Metaspace and diffusion's separate CLIP splitter are unchanged.
  - Reconstructed Halo Hermes prompts: the omitted whitespace alternative changes **120/138 token streams**, including **96 prompt counts** and 24 same-count divergences. The legacy counts match the HF reference on only 42/138. These are prompt-input changes, not a claim about new generation quality, tool-loop outcomes, or speed.
  - The committed offline HF corpus records input MD5s and exact IDs for 138 Hermes prompts, all 55 TC requests plus 14 Halo TC captures, all 44 `benchmarks/prompts/*.txt` files, and 694 synthetic whitespace/Unicode cases (945 total; 868,076 reference tokens). `scripts/gen_hf_tokenizer_corpus.py` regenerates it from local captures and the pinned HF tokenizer/template; `hf_tokenizer_corpus` checks every ID, not just counts.
  - The only existing assertion changed is dots.ocr's `build_prompt_ids_matches_hf_capture`: exact HF token IDs replace the decoded-text-only waiver for this bug; its capture is unchanged and the test remains model-gated. Existing runtime BPE, SentencePiece and cache-digest assertions are unchanged. New splitter/NFC regressions and the corpus test pin the corrected boundaries. The HF-declared NFC normalizer is honored for byte-BPE inputs (already-normalized inputs borrow their bytes), and the pre-tokenizer/normalizer now participate in persistent cache identity.
  - Fixed-ID KLD references, the teacher-forced Qwen4 oracle, and kernel-object `kernel_idproof` pins are intentionally unchanged: they do not re-encode prompt text. New text-derived calibration/reference files and greedy text-prompt identity baselines must use the corrected tokenizer on both sides; old and new text-prompt generation need not match. Frozen deprecated PFlash `.tok.jsonl` fixtures remain explicit fixed-ID inputs, not evidence of equivalence to newly encoded text. Its `--write-pretok` output and DeepSeek's exact-2048-token hardware prompt fixtures must be regenerated against their own model tokenizer if newly tokenized inputs are wanted; their historical pins are not silently redefined.

### Qwen3.8-27B verifier
- **gfx1201 packed-Q8 split-KV DFlash verification, opt-in and default off** (`HIPFIRE_GFX12_FA2_SPLIT_VERIFY=1`, `HIPFIRE_GFX12_FA2_SPLIT_COUNT=2|4|8`, default 8), by @HUSRCF (#760). Exact gfx1201 H24/KV4/D256, eager sequential multi-row verification only; AR, tree, independent-batch, graph and retained recording keep the established route. An undersized `flash_partials` buffer falls through to the rows route instead of failing verification.
  - R9700, canonical Qwen3.8-27B MQ4-XTS plus DFlash2 MQ4v2, Q8 VMM, B16/S8, `HIPFIRE_VERIFY_GRAPH=0`, 21,550 prompt tokens and 200 output tokens: **48.4 → 61.65 decode tok/s (+27.4 %)**, medians of six fresh processes per arm in three OFF/ON/ON/OFF blocks after warming. Every sample retained τ 2.21 and 62 cycles. This reproduces the speedup direction, not the PR's different-model absolute numbers.
  - Against beta `046b57fa46`, default-off committed token IDs and text are identical on that long prompt; on/off also match all 200 IDs there (not a universal bit-exactness claim). Greedy serve battery and chain pass on both arms; default-off text, token counts, τ and cycles equal beta. The B16 capacity fallback passes with `HIPFIRE_FLASH_PARTIALS_BATCH=1` and `max_seq=4096`.
  - No new persistent VRAM allocation: split records reuse `flash_partials`; observed load VRAM is unchanged. Default prefill and every FP8 shared-source code object retain their bytes. With the local JIT toolchain, partial/merge use 238/18 VGPRs and zero scratch or spills.
- **gfx1201 DFlash rollback without the DeltaNet restore copy is on by default (railgun D8, `HIPFIRE_DN_SNAPSHOT_FLIP`; `=0` restores the copy). Byte-identical.** It was opt-in in v0.4.0. It needs a Q8 DeltaNet state, the GDN tape and the two-launch replay, all defaults on exact gfx1201; the KV dtype and verify graph do not matter, so the default fp8-KV, graph-verify DFlash load takes it.
  - R9700 (card C), canonical Qwen3.8-27B MQ4-XTS plus DFlash2 MQ4v2, default config (KV `auto` → fp8, B16 verify graph), default vs `=0` in one binary: committed token IDs, text, τ and cycles are identical on a 763-token code prompt (τ 13.17, 18 cycles), a 65-token prose prompt (τ 1.09, 122 cycles) and a 21,550-token prompt (τ 2.07, 83 cycles), 256 output tokens each, and in every timed sample. Greedy serve battery and chain: text, token counts, τ, cycles and finish reasons equal.
  - The rollback's restore phase goes from 7,369 to 25 µs over the code prompt's 18 cycles, 50,508 to 162 µs over the prose prompt's 122 cycles (about 0.41 ms per cycle) and 34,022 to 104 µs at 21,550 tokens. End to end that is about 1 % of a prose decode, below this card's run-to-run spread: code 280.65 → 282.45 tok/s (+0.6 %, 4 fresh processes per arm, ABBA+BAAB), prose 47.0 → 46.7 tok/s (−0.6 %, 6 per arm, ABBA BAAB ABBA; both arms show the same 43/47 tok/s bands, which follow the clock, not the arm). So no end-to-end speedup is claimed, and none of the v0.4.0 entry's +2.36 % / +1.06 % was reproduced. Peak VRAM is unchanged (20,673.38 vs 20,673.34 MB).

### Highlights
- **gfx1100: MQ4V2 gate/up decode specialization, on by default and byte-identical,** for exactly `(gate_m, up_m, K) = (17408, 17408, 5120)`, by @HUSRCF (#816). The kernel is the generic fused gate/up kernel with its group count fixed at 20 at compile time; accumulation order, tail and reduction tree are unchanged. It does not change the weight representation or add a resident weight copy. Other shapes and architectures keep their existing routes. `kernel.mq4v2_gateup_k5120 = false` / `HIPFIRE_MQ4V2_GATEUP_K5120=0` restores the generic kernel. The 0.4.1 gfx1100 kernel pack carries `fused_gate_up_mq4g256v2_k5120_gfx1100`; compiler-less installs need that pack.
  - Default-on gate (hipx 7900 XTX, Qwen3.8-27B MQ4-XT, Q8 KV, default vs `=0` in one binary): the full 248,320-wide F32 logits are byte-identical at every one of 256 greedy steps (max abs diff 0), and the greedy tokens match. In the dispatch trace, the default arm launches the specialization 64 times and the generic kernel 0 times; `=0` is the reverse. AR decode, TG4096 on VMM with `max_seq=4223`: off 50.3 / 50.1 / 50.3 → default 50.8 / 50.9 / 50.8 tok/s (medians 50.3 → 50.8, **+1.0 %**, disjoint ranges), three fresh processes per arm in ABBAAB order after warm-up, with no build running during the timed runs. Greedy serve battery and chain (5 + 5 turns): text, token counts, finish reasons and request hashes identical.
  - Landing evidence (opt-in arms): same XTX workload, off 50.2 / 50.0 / 50.1 → on 50.8 / 50.7 / 50.9 decode tok/s (medians 50.1 → 50.8, +1.4 %), three fresh processes per arm in ABBAAB order after warm-up. This reproduces a small, workload-specific gain on XTX, not the author's W7900 +1.93 %. R9700 27B MQ4-XTS logits and 256 greedy token IDs are byte-identical to beta with the flag off, and full serve battery/chain outputs match. Kernel idproof adds only the gfx1100 specialization; every existing object on gfx1100/gfx1151/gfx1201 is unchanged.
- **Flash-Next prefill at the new defaults** (gathered F16 WMMA QSA attention on gfx1151/gfx1201, MQ6 X-LDS trunk projections), against the previous land head `20f981d7a`: pp8192 on an R9700 with `auto` expert placement 1011.6 → 1216.7 tok/s (+20.3 %); Strix Halo 1158.5 → 1446.2 tok/s (+24.8 %). Medians of 3 fresh processes per arm.
- **Flash-Next prefill chunk per arch** (8192 rows on gfx1151, 4096 on gfx1201), against `cd2b9d91a`: pp8192 on an R9700 with `auto` expert placement 746.2 → 1011.9 tok/s (+35.6 %); Strix Halo 1048.9 → 1164.8 tok/s (+11.0 %). Medians of 3 fresh processes per arm.
- **Flash-Next quality at the new defaults:** KLD against the BF16 source (WikiText-2, 32 × 512 tokens, Strix Halo) is 0.074318, with the same logits, byte for byte, as the earlier reference run. With only the gathered attention opted out, 16K logits are byte-identical to `20f981d7a` on both arches. At the defaults, the worst sampled 16K attention row is within 7.2e-4 relative error of the F32 reference on gfx1201 and 5.1e-4 on gfx1151, and the 128K needle answers correctly on both.
- **Flash-Next default flips (QSA PM, Halo hyper units, symmetric IU4 on gfx1151, long-prefill expert staging)**, against `fe77c0837`: on Strix Halo the shipped asymmetric `.mq4` goes from 1,447.6 to 1,544.9 tok/s at pp8192 (+6.7 %), and the 16K logits are byte-identical. The symmetric GPTQ3 requant takes the IU4 route by default and reaches pp8192 1,928.5 tok/s, at a BF16-source KLD of 0.1027 (WikiText-2, 32 × 512 tokens). On the F16 route the same artifact's KLD is 0.066. gfx1201 decode (R9700, `auto` placement, tg64) stays at 34.6 / 32.4 tok/s at 512 / 8K context. All figures are medians of 3 fresh processes.

### Flash-Next performance
- **Qwen4 n-gram takeover pricing:** compare expected emitted tokens per time using request-measured native window costs after calibration, with measured verify-row fits on gfx1151 (30.5 + 6.9 ms per additional row) and gfx1201 (30.3 + 14.8 ms per additional row) as fallback. Other architectures retain the previous chooser. The takeover yield prior, confirmation and exact verification are unchanged; CPU coverage rejects the R9700 copy regression and retains a winning Halo takeover. On Halo (TC s123, RC3, one seed per arm) the default 5/3/3 triple won the comparison: 22/30 at 162.4 s, against 22/30 at 171.7 s for 24/48/63 and 23/30 at 164.6 s for 16/8/32; turning n-gram off was score-neutral and slower on two of three seeds (median 163.2 s vs 159.6 s). No R9700 speed claim is made for the pricing.
  - **Fix: cost-aware n-gram takeovers keep the remaining-output draft cap (`n_max.min(k)`, `k = max_emit - 1`) before verify and head fill.** The pricing change dropped the per-call bound, so a confident long match could commit more rows than the remaining budget (3 drafts plus the bonus with 2 emits left). The host output was then clipped after the target and the MTP head had committed, leaving both device positions ahead of the host cursor, and the terminal pending-seed flush failed with `MTP position mismatch`. CPU tests pin every admitted proposal to the caller's budget and the committed rows to the emitted count. Halo confirmation (release candidate `7dc9885eb0`): sampled MTP (T 0.7, top-p 0.8, top-k 20) on the 8-prompt set, four repetitions in one daemon process each, plus `glimmer_prefill_1024` at three seeds sampled and greedy MTP and an AR arm, all seven arms finish with 0 errors and no `MTP position mismatch`; the unclamped build failed six of eight arms on the same prompt.
- **Qwen4 opt-in longer n-gram matches:** with a larger configured `n_max`, five observed takeover windows with at least 95% draft acceptance allow the chooser to extend beyond native depth only when expected emitted tokens/time improves, bounded by both the configured `n_max` and the per-call remaining-output draft budget. The default 5/3/3 is unchanged. Explicit 1–8-row cost buckets cover the measured few-row path; the distinct >=9-row expert-sharing route is conservatively disabled for extension until measured bucket costs are available (no linear extrapolation). CPU tests cover confident long, short and uncertain matches. No takeover window above four rows occurred in the Halo timing probe, so this path has no GPU measurement.
- **Qwen3.8-Flash-Next (Qwen4) native MTP prompt fill: a prefill chunk that does not end the prompt skips the final hyper, LM head, argmax and host readback (`HIPFIRE_QWEN4_MTP_SKIP_INTERMEDIATE_PICK`, default on; `=0` keeps them on every chunk).** `mtp_prefill` ran `spec_prefill_rows` on every capture-split or natural chunk and read back an argmax, although only the last chunk's value is used (the seed). A non-final chunk now runs the existing output-free trunk path (`Qwen4OutputPolicy::None`, the one `forward_chunk` already uses for non-final tiles) while still capturing the full wide hidden for the head, with the same tiles and trunk route. The final chunk, including one ending at the end-of-prompt capture boundary, keeps full logits, readback and the sampled draw; the seed, capture boundaries and the head's append are unchanged. No new kernel or threshold. This only applies to prompts filled in more than one chunk; a single-chunk fill, such as a short cache-hit suffix, is unchanged, and no speedup is claimed here.
- **Qwen3.8-Flash-Next (Qwen4) native MTP: a same-request AR floor retires a request to head-free target-only decoding when no measured speculative option beats ordinary AR on that card.** The first decode window is one timed ordinary AR token (price `ar_us`, sampled draw included) plus one head append; up to three speculative probe windows follow, ordered by the Halo cost table only. After them just this request's measured window costs, per actually drafted depth or per emitted token, are compared with `ar_us`; if none is strictly cheaper (a tie or invalid timing retires) the request stays on AR with no head work until the next request. Reported as `mtp_retired` / `ar_windows`. The bound is on speculation work (at most one calibration head append plus three probe windows before the first measured decision), not a guarantee on full-request time. `HIPFIRE_MTP_INCREMENTAL=0|1` remains a diagnostic override that bypasses the floor. No measurement is claimed here.
- **Qwen3.8-Flash-Next (Qwen4) prompt-fill levers at the 0.4.1 cut: `HIPFIRE_QWEN4_MTP_REUSE_PREFILL_IDS` and `HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD` default off (opt in with `=1`); `HIPFIRE_QWEN4_MTP_SKIP_INTERMEDIATE_PICK` stays on.** The ID-reuse path was not output-identical on the warm-cache gate (MTP after-window metadata) and saved about 0.14 s on a 26 s prefill; the async PLE upload's warm-cache and cache-quality identity gates had not completed. Neither carries a speed claim. The intermediate-pick skip is identical cold, warm and under cache-quality. The n-gram proposal is also clamped to the request's remaining emit budget, so sampled MTP no longer fails closed with a position mismatch at the budget edge.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: the native MTP prompt fill's batched head Append embeds from the token ids the trunk forward of the same chunk already uploaded, instead of uploading the same ids a second time (`HIPFIRE_QWEN4_MTP_REUSE_PREFILL_IDS=1` opts in; default off, the head keeps its own upload).** Every prefill chunk's target forward writes the chunk's ids to its device `token_ids` buffer, and the head's batched Append then copied the same ids host-to-device again per sub-chunk (one blocking copy each). The head now reads sub-chunk rows `off..off + n` of that buffer in place. The reuse is taken only when the forward uploaded the ids whole from the host, outside graph capture, and its host mirror equals the sub-chunk's tokens; a HIP single-row body, a device-argmax id or any mismatch keeps the upload. The kernel sequence and the other syncs are unchanged, and the MTP decode-window append keeps its own upload. **Default off:** on the RC4c identity gate the warm-cache path with reuse was not output-identical to the head's own upload (MTP after-window metadata), and the saving measured about 0.14 s on a 26 s prefill, so it stays an opt-in until that is resolved. No speed claim is made.
- **Qwen3.8-Flash-Next (Qwen4) prefill: the next chunk's PLE rows are fetched as a retained ticket that the next forward adopts, instead of a cache warm whose decoded rows were thrown away. Byte-identical; the 64–82 ms PLE wait that every Strix Halo chunk past ~57K tokens paid before its first PLE gather is gone.** After a forward commits, it enqueues the next chunk's row ids, hashed from the history it just committed. The next forward adopts the materialized ticket only when the epoch and the exact row-id sequence match; otherwise it drops the ticket and fetches as before. Rows are a pure function of their ids. Reset, restore, abort and the prefix-checkpoint quiesce all invalidate a retained ticket. `forward_chunk` also enqueues its first tile's rows ahead of the output-resource preflight.
  - rocprofv3, Strix Halo, Ciru 128,077-token prefill (fresh daemon): the idle time before each chunk's first PLE gather, for chunks 2–14, goes from 7.3–81.9 ms to 0.55–0.65 ms. The sum across the 16 chunks drops 1,044 → 817 ms. Chunk 0's first read has no predecessor to hide behind and is unchanged (5.8–5.9 s on both arms in this drive state). Chunk 1, whose lookahead overlaps only chunk 0's ~90 ms of GPU work, is not hidden (407 vs 762 ms, drive-state noise). The final partial chunk still waits ~47 ms: the prefix-checkpoint quiesce before it cancels its lookahead.
  - Speed (Strix Halo, Ciru prompts, `hipfire bench --prompt-file`, max_seq 262144, fresh processes, warm kernel cache, ABBA): 128K 1,632.4 / 1,625.9 → 1,636.8 / 1,641.4 prompt tok/s (+0.6 %, TTFT −0.48 s). 32K 1,760.3 / 1,654.6 / 1,758.0 / 1,683.1 → 1,290.9 / 1,760.8 / 1,752.3 / 1,757.8. The medians are 1,720.6 → 1,755.1, but both arms are noisy and the low cand sample is that arm's first 32K run.
  - Identity: `qwen4_qsa_ctx` 16K final-row logits md5 equals beta on Strix Halo (`7abadb46…`) and on an R9700 (`37609204…` at the dense GDN default, `6b5a824c…` before the rebase). The serve_harness battery on Strix Halo (greedy) matches beta's content on every turn, MTP `auto` equals MTP off, and `gen` and τ equal beta's. Kernel idproof: IDENTICAL on gfx1100, gfx1151 and gfx1201.
- **Qwen3.8-Flash-Next (Qwen4) native MTP on discrete GPUs: the draft head's MQ2 copy of the language head is requantized 2,048 rows at a time instead of through one vocab × hidden F32 scratch, so `auto` expert placement no longer reserves 2.4 GB for it. Byte-identical; on an R9700 one more expert layer stays in VRAM with MTP on, and MTP decode is +1.8 %.** The ranking copy was built by expanding the whole 248,320 × 2,560 head to F32 (2,425 MiB) and then reordering the packed copy into a second buffer. `auto_vram_reserve` charged that build scratch on top of the attached head, although it is freed at attach. `Gpu::requant_g256_rows` now expands, rotates and packs one 2,048-row chunk (20 MiB) at a time. It writes straight into the front/special/tail layout order, so the unordered copy is gone as well. Every requant kernel is row-local, so each packed row is unchanged; a GPU unit test checks the chunked, reordered output byte for byte against the one-shot requant. The rescore path reads the source head, as before.
  - R9700, `auto` placement, MTP on: the native MTP reserve goes 2,671 → 640 MiB at `max_seq` 16384 (2,745 → 714 MiB at the default 32768), and the layers in VRAM go `0..13` → `0..14`. AR loads charge no MTP and are unchanged (`0..15`).
  - Speed (R9700, `release-matrix` `decode.py`, 8 prompts × 128 tokens greedy, `max_seq` 16384, 3 ABBA fresh processes per arm, medians): MTP 39.97 → 40.68 tok/s (+1.8 %, τ 1.718 on both); AR 34.99 → 35.03 (unchanged). pp8192 (`hipfire bench --matrix`, spec off) 3,632.0 → 3,631.4 tok/s (flat).
  - Default config, Ciru 32,012-token prompt + 512 tokens (MTP, `max_seq` 32768): no OOM. Peak card VRAM goes 30,610 → 31,622 MiB of 32,624 MiB.
  - Identity: `qwen4_qsa_ctx` 16K final-row logits md5 equals beta on an R9700 (`6b5a824c…`) and on Strix Halo (`7abadb46…`). Greedy decode commits the same token IDs as beta, AR and MTP, on both arches, with the same per-request τ. serve_harness battery and chain on the R9700 (greedy): MTP on equals MTP off turn for turn, and the battery equals beta. Kernel idproof: IDENTICAL on gfx1100, gfx1151 and gfx1201.
- **Qwen3.8-Flash-Next (Qwen4) short-suffix prefill: the deferred PLE row upload is asynchronous behind a completion fence (`HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD=1` opts in; default off keeps the blocking `hipMemcpy`: cold output is identical, but the warm-cache and cache-quality identity gates had not completed at the RC5 cut). Byte-identical by construction; GPU timing not yet measured.** A prefill chunk of more than one token stages its PLE rows after layer 0 is enqueued. The blocking H→D copy there also waited for every earlier layer, which a short suffix prefill (one chunk, no device token, no later sync) pays as a host stall. The upload is now `hipMemcpyAsync` on the legacy stream, and the one shared host staging buffer is owned by a small `PleHostStage` (`ple_stage.rs`) that records a `hipEvent` right behind the copy. Every write to the buffer drains that fence first (`event_synchronize`), so a later forward cannot overwrite bytes a copy is still reading. This holds on exceptional exits too: the fence is recorded before any later error return, an enqueue or record failure leaves the state pending and the next write drains the whole device, `free_gpu` retires the fence before the device buffers go, and dropping the stage with a copy pending leaks the bytes instead of freeing them. Non-final silent chunks have no top-1 sync, so nothing here depends on a final-chunk readback. The single-token path, the device-token (decode) branch, graph capture and every numerical route are unchanged. CPU tests pin the staging state machine and the decision table; GPU confirmation is pending.
- **Qwen3.8-Flash-Next (Qwen4) prefill: the router and the shared expert's selector/gate/up GEMMs read one F16 conversion of the layer's normalized activation instead of converting it twice. Byte-identical; one fewer `convert_f32_to_f16` launch per layer.** Each layer converted the same F32 `x_norm_batch` twice, back to back: once for the router GEMM and once for the shared selector/gate/up. When the router and all three shared weights are BF16 on the F16 WMMA route with the same K, at more than 8 rows and outside graph capture and replay recording, the shared gate/up stage now issues the router's GEMM itself (`Gpu::gemm_bf16_xf32_f16_wmma_qwen4_groups`). Each group launches exactly the kernels its own call would, so the output bytes are unchanged. Other cases keep the old path.
  - rocprofv3 kernel trace, pp8192 prefill: large converts per prefill go 96 → 48 on Strix Halo (8192-row chunk, one per layer), 50.0 → 25.0 ms. On an R9700 they go 192 → 96 (two 4096-row chunks), 17.6 → 11.8 ms.
  - Speed (`hipfire bench --matrix --pp 8192`, 3 ABBA pairs of fresh processes after a warm-up): Strix Halo 2,043.8 / 2,047.0 / 2,046.6 → 2,056.6 / 2,057.0 / 2,058.7 tok/s (+0.51 % median). R9700, `auto` placement: 3,626.3 / 3,626.7 / 3,626.5 → 3,629.2 / 3,629.3 / 3,628.9 tok/s (+0.07 %).
  - Identity: `qwen4_qsa_ctx` 16K final-row logits md5 equals beta on Strix Halo (`7abadb46…`) and on an R9700 (`6b5a824c…`). Greedy decode of 8 prompts (128 tokens, AR and MTP) commits the same token IDs as beta on both arches. serve_harness battery on the R9700 (greedy, MTP on and off): every turn's content equals beta's, and MTP equals AR. Kernel idproof: IDENTICAL on gfx1100, gfx1151 and gfx1201.
- **Qwen3.8-Flash-Next (Qwen4): engine-owned session cache keeps many prompts warm (`memory.session_cache_bytes`, `HIPFIRE_SESSION_CACHE_BYTES`, default 8 GiB, `0` disables).** It replaces the single whole-chunk prefix checkpoint, which kept only one prompt: the next request with another prefix, such as another session or a subagent, overwrote it. `hipfire_runtime::session_cache` now owns keying, planning, LRU eviction, the memory guard and placement. Each prefill snapshots the state at every prefill-chunk boundary it crosses: the GDN, PLE and hyper state, the selections and, with MTP, the head's state and draft policy whole. Of the append-only QSA K/V, raw and pooled rows (and the head's) a snapshot holds only the rows above its parent, the deepest snapshot of the same prefix. A snapshot therefore costs one chunk of rows at any depth, and sessions that share a prefix (subagents with one system prompt) share its links. Restores walk the chain from the root. Snapshots never depend on live state, so a reset no longer drops them, and a snapshot with children is pinned, so eviction removes only leaves. A snapshot is published only after the client commits the turn. Qwen4 no longer reads `HIPFIRE_QWEN_PROMPT_CACHE`.
  - Strix Halo, canonical GPTQ3 artifact, serve, greedy, MTP: prompt A (21,550 tokens), then B (8,142 tokens), then A again. A's prefill-and-decode time went from 16.3 s cold to 4.9 s, with `cached_tokens` 16,384 and the same content. With `HIPFIRE_SESSION_CACHE_BYTES=0`, A reports `cached_tokens` 0 and the same content. Each snapshot is 513 MiB at any depth; full copies would grow by 481 MiB per chunk of depth (a 64K session: about 4 GiB for all eight boundaries instead of about 17 GiB).
  - Identity: `session_cache_hw` (F32 QSA, the serve default on gfx1151, Q8 GDN, VMM) prefills A (3 chunks + 300) and then B (A's first chunk + 8,392 other tokens) cold, then restores A through its three-link chain and B through the shared root, on the AR and on the native MTP route. Every state part (metadata, fixed parts, valid rows of every row stream, the MTP head and its draft policy included) is byte-equal to the cold prefill's, the AR final logits row is byte-equal with 16 equal greedy ids, the MTP seed is equal, and the cache holds exactly four one-chunk snapshots per route.
- **Qwen3.8-Flash-Next (Qwen4): a large PLE row request reads its SSD pages over up to 256 threads instead of 16 (`HIPFIRE_QWEN4_PLE_WIDE_READERS`, default on; `=0` keeps 16). Byte-identical; Ciru 8K cold prefill +5.6 %.** A cold prefill's first chunk fetches about 111K scattered 1,280-byte PLE pages at the layer-1 consumption point. No earlier chunk can warm them, so the GPU idles for the whole read (the 1.4 s gap at the first PLE gather in the Halo trace). Timestamps show the gap is the row-store read: about 40 ms of row ids, plan and cache scan, then the SSD reads, ~130 ms of copy and cache publish, and ~13 ms of staging and upload. Load, pinning and the lookahead are not involved. A request with more than 1,024 page reads now uses one reader per 64 reads, at most 256. Smaller requests (decode, MTP verify, short prompts) keep 16. Bytes, order and the row-store cache are unchanged.
  - Strix Halo, canonical GPTQ3 artifact, `hipfire bench` protocol, 8K chunk 0: host wait 1,555 → 1,178 ms. The SSD reads go 1,428 → 1,022 ms, the device's ceiling: ~100K IOPS at queue depth 128–256 against ~37–50K at 16. Publish stays at ~130–145 ms. rocprofv3, load followed directly by the prompt: the chunk-0 gap goes 1,254 → 1,105 ms at 8K and 1,451 → 1,077 ms at 32K, and chunks 1+ stay at 6–13 ms.
  - The read rate depends on the drive's state. After ≥ 30 s idle the same reads run at ~18K IOPS for any thread count, and chunk 0 then waits 2–7.5 s on both arms. That cost is device-bound and unchanged here. In that state the wider readers still finish the next chunk's lookahead warm within the current chunk: base stalled 3.0–3.2 s at chunk 1 in 2 of 5 32K runs, the candidate in 0 of 4. Load time is unchanged (both arms 47–71 s, the same bimodal spread).
  - Speed (Ciru prompts, `hipfire bench --prompt-file`, max_seq 262144, fresh processes back to back, `prefill_tok_s`, 3 ABBA pairs after a warm-up pair): 8K 1,447.0 / 1,451.6 / 1,456.1 → 1,533.4 / 1,534.7 / 1,516.9 (+5.6 % median). 32K 1,729.2 / 1,726.5 / 1,725.7 → 1,755.7 / 1,748.8 / 1,750.2 (+1.4 % median). With `=0` the 8K prefill measures 1,454.0.
  - Identity: `qwen4_qsa_ctx` 16K final-row logits md5 equals beta on Strix Halo (`7abadb46…`) and on an R9700 (`6b5a824c…`). serve_harness battery (greedy, MTP `auto` and off): every turn's content equals beta's, and MTP equals AR. Kernel idproof: IDENTICAL on gfx1100, gfx1151 and gfx1201.
- **QSA selector pair on PeaceMaker (gfx1151, default on).** The live pair's rows16 F32 score and select-from-scores kernels run from the certified builder module `kernels/qsa_select_pm_gfx1151.hxaco` on exact gfx1151 prefill calls with >= 512 rows, 4 index heads, dimension 128, <= 130816 pooled blocks, budget <= 512 and supported 32-bit buffer extents. Selection is byte-identical; recorder/graph capture, decode/verify, unsupported calls and other architectures keep hipcc. `HIPFIRE_QWEN4_QSA_SCORE_PM=0` and `HIPFIRE_QWEN4_QSA_SELECT_PM=0` independently keep their hipcc kernels.
  - Identity (Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`): `qwen4_qsa_ctx` final-row logits md5 equals base at ctx 16384 (`7abadb46…`) and 65536 (`dfd11299…`). Selector oracle (`qsa_select_pm_check --g3`): 83 cases (48 real 32K/64K first/last-chunk captures of all 12 QSA layers, 27 synthetic ragged/budget/residue/duplicate cases, all-tied rows at 16384, 32768, 65536 and 70000 blocks), 707 records, every score bit, the whole ordered selection, tails, `-1` fill and mirror byte-identical across the hipcc pair, PM score/select/both, PM under HIP graph capture and the live route; inputs unchanged, repeats identical. serve_harness battery and chain (greedy, MTP `auto` and off): every turn equals base and MTP equals AR. Certification: M7 lift byte-exact with no obligations for both symbols; the RIP twins, native writer, ROCm oracle and shipped bundle are identical. Kernel idproof vs base: IDENTICAL on gfx1100, gfx1151 and gfx1201.
  - Performance (Strix Halo, Ciru prompts, `hipfire bench --prompt-file`, fresh processes, `prefill_tok_s`): 32K, 3 ABBA pairs after a warm-up pair, median of 6 per arm, 1,660.7 → 1,712.8 tok/s (+3.1 %); 64K, one run per arm, 1,597.0 → 1,692.1 (+6.0 %).
- **QSA selector on PeaceMaker for the q8 QSA state (gfx1151, default on).** With BF16 index arenas (the opt-in `--kv-mode q8` state), the same eligible prefill calls now score with a certified builder rows16 BF16-pooled kernel (`kernels/qsa_select_bf16_pm_gfx1151.hxaco`: exact BF16 widening, per-head serial F32 FMA chains, hipcc's IEEE divide) and select with the PM select above, instead of the fused one-workgroup-per-row hipcc `indexed_attention_select_bf16_batched`. Selection is byte-identical; recorder/graph capture, decode/verify and unsupported calls keep the fused kernel, and `HIPFIRE_QWEN4_QSA_SCORE_PM=0` keeps it everywhere.
  - Identity (Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, q8 QSA state): `qwen4_qsa_ctx` final-row logits md5 equals beta at ctx 16384 (`f7b8d738…`) and 65536 (`d53aef64…`), and `HIPFIRE_QWEN4_QSA_SCORE_PM=0` gives the same 16384 md5. Selector oracle (`qsa_select_pm_check --g3-bf16`): 62 cases (48 real 32K/64K first/last-chunk q8 captures of all 12 QSA layers, all-tied rows at 16384/32768/65536 blocks, budgets 0/1/511, fewer visible blocks than budget, HIP graph capture taking the fused fallback), 0 failures; PM score bits equal the fused kernel's on every causal (row, block). serve_harness battery and chain under q8: MTP text equals AR. Certification: M7 lift byte-exact, no obligations; RIP twin, native writer, ROCm oracle and shipped bundle identical. Kernel idproof: no hipcc code object changes.
  - Performance (Strix Halo): selector time per chunk summed over the 12 QSA layers, medians of 3 fresh processes: 64K last chunk 810.7 → 213.6 ms, 32K last chunk 367.5 → 111.6 ms, first chunk 77.1 → 28.4 ms. Ciru 32K q8 cold prefill, one ABBA (`hipfire bench --kv-mode q8`, fresh processes, `prefill_tok_s`): 1,693.6 / 1,698.2 → 1,734.3 / 1,724.8 (+2.0 %).
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: the native MTP prompt fill runs the head's KV-only Append step over whole prefill chunks instead of one prompt row at a time (`HIPFIRE_QWEN4_MTP_BATCHED_FILL`; `=0` keeps the per-row loop). Byte-identical; with the PLE lookahead below, the Ciru 32K prefill goes 920.4 → 1,655.7 tok/s (+79.9 %) and 256K goes 735.0 → 1,149.0 (+56.3 %).** `mtp_prefill` used to run one ~20-launch single-row `forward_token(.., Append)` per prompt row after each chunked target forward. It now runs the same operator sequence once per sub-chunk of up to 1024 rows (about 17 launches): a batched Q8_0 embedding lookup, the grouped HC norms and the BF16 projections over all rows (the four HC branches of a row count as four projection rows), a broadcast add, the HC read (rows-grid K4 down projection with the 1/4 activation, up projection, branch mix), the three Q8_0 QSA projections as row-tiled staged GEMVs, one append prologue launch (index-query and key norm + RoPE, K/V cache append, index-key BF16 round trip and raw-key copy, with no query/gate branch), and one incremental pool of the blocks the rows complete. Each operator is the single-row kernel with a rows grid or a multi-row kernel with the single-row kernel's per-row reduction order, called directly rather than through the shared projection dispatch (which takes the F16 WMMA route from 512 rows). The (token p, hidden p) pairing, draft request state and the pending hidden row left for the first draft step are unchanged. gfx1151 only; other arches keep the per-row loop. The pass allocates about 260 MB of row-batched scratch at the first fill; the native MTP device-byte estimate charges it only on exact gfx1151 with the pass on, and `=0` allocates nothing.
  - Identity (Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, `qwen4_mtp_fill` digests of the head's QSA state, the target state families, the pending hidden row, the seed and the state after one draft window): `=0` and `=1` give the same output at 17 cold prompt lengths (1–9, 1023–1025, 2049, 8191–8193, 9217, 16385) and at 19 prefix-cache hits that restore the checkpoint and fill from `start_pos` 8192 (suffixes 1–9, 1023–1025, 2049, 8191–8193, 9217) or 16384 (suffixes 1, 1025, 8193). serve_harness battery and chain (greedy, MTP `auto` and off): content, finish reason, token count and τ of every turn equal beta `a193e4491`, and MTP text equals AR. Greedy generation on Ciru's 32K prompt: MTP and AR text equal base. Kernel idproof vs `a193e4491`: the new kernels are added to `add`, `qwen4_gemv_bf16_xf32` (and gfx1151's `gemv_bf16_xf32`) and `qwen4_gemv_q8_0`, and every pre-existing function's ISA is identical except `gemv_q8_0_k2560_staged_rows`, whose body moved into a device function shared with the tiled kernel. That body has the same FP op histogram on all three arches (80 `v_cvt_f32_i32`, 80 `v_fma_mix_f32`, one 80-step `v_fmac`/`v_fma` chain into one accumulator, 5 `v_add_f32` + `ds_swizzle`/`ds_bpermute` reduction tail); only load scheduling and waits differ. It is dispatched only on gfx1151 (2–8-row speculative verify), where the serve outputs above are unchanged.
  - Speed (Strix Halo, `hipfire bench --prompt-file` with Ciru's prompts, max_seq 262144, fresh processes; `prefill_tok_s`): 32K, 3 ABBA pairs after a warm-up pair: base 920.4 / 915.3 / 921.7 → 1,657.8 / 1,655.7 / 1,517.9 (the low run at gfxclk 2436 MHz, socket 124 W). MTP off prefills the same prompt at 1,730 tok/s. 256K, one run per arm: 735.0 → 1,149.0 (`thm_gfx` 17 % vs 69 %, so a thermal-limited lower bound). pp8192 (`hipfire bench --matrix --pp 8192 --max-seq 16384`, target forward only, 3 ABBA pairs after a warm-up pair, median of 3 samples per process): 2,007.5 / 2,010.4 / 2,007.3 → 2,014.4 / 2,009.0 / 2,011.5 tok/s (no regression).
- **Qwen3.8-Flash-Next (Qwen4): a chunked prefill warms the next chunk's SSD-resident PLE rows while the current chunk runs. Byte-identical; Ciru 32K prefill +10.1 %, 256K +9.7 %.** A chunked prefill used to wait 1.0–1.5 s per 8192-row chunk on the n-gram PLE rows (about 131K random rows per chunk) with the GPU idle. A forward that knows the following chunk's tokens now issues a best-effort `RowStore::warm` of their rows once it releases its own lease, within the store's `max_rows_per_prefetch`. The next forward's prefetch then reads the same pages, in the same order, from the row store's cache. `begin_prefix` clears any lookahead an aborted request left.
  - Identity (Strix Halo, canonical GPTQ3 artifact): `qwen4_qsa_ctx` final-row logits md5 equals base at ctx 8192 (`7a10260d…`) and 32768 (`53486795…`). Greedy generation on Ciru's 32K prompt, MTP `auto` and off: same text, τ 1.63 and cycle count as base. serve_harness battery and chain (greedy, MTP `auto` and off): content, finish reason, token count and τ of every turn equal base.
  - Speed (Ciru's prompts, same protocol as the batched-fill entry): 32K, 3 ABBA pairs after a warm-up pair: 920.6 / 920.3 / 921.6 → 1,014.1 / 1,008.0 / 1,013.5 tok/s. 256K, one run per arm: 735.0 → 806.4 tok/s. pp8192 (single chunk, so no lookahead): 2,002.2 / 2,007.8 / 2,005.2 → 2,002.2 / 2,008.6 / 2,006.2 tok/s (no regression).
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: four exact prefill fusions are on by default, and the MQ6 X-LDS tile table covers the N=8192/2048 trunk shapes, the a/b/z region fold and the HC-write launches. Byte-identical; pp8192 +5.1 %.** `HIPFIRE_QWEN4_HC_DOWN_TILE`, `HIPFIRE_QWEN4_HC_ROW_FOLD`, `HIPFIRE_QWEN4_ROUTER_FAST` and `HIPFIRE_QWEN4_PLE_FUSE` default on for exact gfx1151 (`=0` keeps each incumbent; other arches ignore them). The gfx1151 MQ6 region fold and HC-write launches route to measured BV8/RW8 twins at N=8192 (regions also at N=2048), keyed on the fold's total rows.
  - Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, IU4 route: pp8192 logits md5 (`qwen4_qsa_ctx`, ctx 8192) is `7a10260d…` on base and on every candidate (the fusions alone, the tile rows alone, both, and both under a rocprofv3 kernel trace). Greedy AR and MTP decode text matches base. The serve_harness battery (MTP auto, greedy) passes on coherence; a base battery for an identity diff did not run (GPU-lock collision). Kernel idproof: IDENTICAL on gfx1100, gfx1151 and gfx1201.
  - `hipfire bench --matrix --pp 8192 --max-seq 16384`, 3 ABBA pairs of fresh processes after a warm-up pair: 1922.8 [1921.9–1924.4] → 2021.0 [2020.6–2021.6] tok/s, **+5.11 %** (paired +5.05 / +5.14 / +5.11 %). The fusions alone give +2.87 %, the tile rows alone +2.13 %. Every run was fast-PPT limited (residency 50–68 %, both arms); the faster arm ran at a slightly lower gfxclk, so the gain is not a clock artifact. Absolute numbers apply to that power state.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: the gfx12 F16 WMMA route for the HC GEMMs and BF16 projections is on by default, with `HIPFIRE_QWEN4_HC_FUSE=1` and `HIPFIRE_QWEN4_HC_UP_TILE` (`=0` is the kill switch for each). pp8192 2,198 → 3,598 tok/s (+63.7 %).** `HIPFIRE_QWEN4_F16_WMMA_GFX1201` (opt-in in v0.4.0, "default off until a Flash-Next KLD check") now defaults on for exact gfx1201, and gfx1201 defaults to HC fusion level 1 plus the retiled HC up+mix on that read. Levels 2 and 3 measured slower on gfx1201 and stay opt-in there. The F16 route is not bit-exact against the multirow/SIMT arms; `HC_FUSE=1` and `HC_UP_TILE` are byte-identical on top of it. gfx1151 and every other arch are unchanged.
  - KLD against the BF16 teacher (R9700 card B, `auto` placement, canonical `qwen3.8-flash-next-gptq3.mq4`, `fn-bf16.{wt2,code24}.1.bin`; WikiText-2 24 × 2048, code 24 × 512), F16 route vs the multirow/SIMT arms: WikiText-2 0.0877 → 0.0845 (−0.0033, paired bootstrap 95 % CI [−0.0082, +0.0006]), code 0.1018 → 0.1065 (+0.0047 [−0.0030, +0.0131]). Both intervals include zero.
  - Identity (`qwen4_qsa_ctx` ctx 8192, chunk 4096, final-row logits): with no env the flipped build gives `5f800667…`, the md5 of the previous `F16_WMMA_GFX1201=1 HC_FUSE=1 HC_UP_TILE=1` opt-in; `HIPFIRE_QWEN4_F16_WMMA_GFX1201=0` gives `a78c15a6…`, the previous default (also beta `303f42249`'s md5). `HC_FUSE=1`/`HC_UP_TILE=1` were md5-identical on top of the F16 route.
  - Speed (`hipfire bench --matrix --pp 8192 --max-seq 16384`, fresh process per sample, ABBAAB order against beta `303f42249`): 2,219.5 / 2,197.8 / 2,185.1 → 3,597.5 / 3,597.6 / 3,594.7 tok/s (medians 2,197.8 → 3,597.5); perf level `auto`, sclk 2.51–2.57 GHz. Ciru's 32K prompt (32,012 tokens, `hipfire bench --prompt-file`, cold, one process per arm): 1,450.2 → 1,926.5 tok/s (+32.8 %).
  - serve_harness battery and chain (greedy, MTP `auto`, thinking off): every turn routes MTP (τ 1.04–1.85), finishes `stop` and stays on task. The battery answers 5/5 turns. Chain turn 4 (the four-sentence story) ends inside its reasoning with no answer, as beta `303f42249` does on turns 4 and 5 of the same chain; the flipped build answers turn 5.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: the MoE combine zero-init, the wave-per-token router and the fused PLE tail are on by default, and a prefill that stages host-mapped experts keeps every fusion across a MoE call. Byte-identical; pp8192 +1.3 %.** `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`, `HIPFIRE_QWEN4_ROUTER_FAST` and `HIPFIRE_QWEN4_PLE_FUSE` default on for exact gfx1201 as on gfx1151 (`=0` keeps each incumbent); none of them needs the F16 WMMA route. A G2-staged prefill (host-mapped experts, chunks of at least 2048 rows) used to run its layer program in pieces cut at every MoE call, so a fusion spanning a MoE call never applied there: the combine zero-init did not run at all, and with `HIPFIRE_QWEN4_HC_FUSE` only the attention blocks paired their HC read and write. The program now runs as one interpreter call; the stage wait and the stage free/refill run as hooks right before and after each MoE call (`execute_validated_steps_with_moe_hooks`).
  - R9700 (card B, `auto` placement), canonical `qwen3.8-flash-next-gptq3.mq4`, IU4 route: pp8192 logits md5 (`qwen4_qsa_ctx`, ctx 8192, chunk 4096) is `a78c15a6…` on base, with the flips alone and with the flips plus the one-program staging; on the opt-in F16 WMMA route it is `5f800667…` on base, with the flips, with the one-program staging, and with `HC_FUSE=1` or `3` plus `HC_UP_TILE=1`. The router and fused PLE tail unit tests (new: 37 tokens with ties, ±inf and NaN logits, both normalize and BF16-round settings) are byte-identical on gfx1201.
  - `hipfire bench --matrix --pp 8192 --max-seq 16384`, 3 ABBA pairs of fresh processes: flips alone 2,170.6 [2,167.7–2,171.1] → 2,187.1 [2,186.1–2,187.9] tok/s (+0.8 %); flips plus one-program staging 2,165.9 [2,164.0–2,171.0] → 2,194.6 [2,190.5–2,195.6] (+1.3 %, paired +1.22 / +1.33 / +1.13 %). Kernel trace: router 29.4 → 4.9 ms, PLE tail 9.0 → 3.9 ms, the 96 `moe_output` fills (3.3 ms) are gone and the combine drops from 51.3 to 44.2 ms. On the opt-in F16 WMMA route with `HC_FUSE=1 HC_UP_TILE=1`, every HC read now pairs its write (192 of 192, was 96): 3,532.5 [3,531.7–3,533.6] → 3,598.3 [3,597.1–3,598.5] tok/s against the route alone (+1.9 %).
- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: the GDN prefill recurrence runs a register-resident kernel by default (`HIPFIRE_QWEN4_GDN_PIPE`; `=0` keeps the persistent kernel). Byte-identical.** `gated_delta_step_pipe128_{f32,q8}` gives each value column one thread holding its 128-entry state in registers, uses one barrier per row, stages q/k through a 3-slot LDS ring and interleaves the output and state-update chains. It is gfx1201-only: idproof changes only the gfx1201 `tensor_ops` object.
  - R9700 (card B, `auto` placement), canonical `qwen3.8-flash-next-gptq3.mq4`: the kernel's output and state are byte-identical to the persistent kernel for rows 1–1000, F32 and Q8. pp8192 logits md5 is the same for base, the new kernel and `=0`. Greedy AR and MTP text matches base on 8 of 8 prompts. pp8192 on the F16 route, 3 fresh processes per arm in ABBA order: 1413.1 / 1417.5 / 1418.2 → 1455.2 / 1453.1 / 1453.4 tok/s (median +2.5 %, ranges disjoint). The bench power fields were not populated on this path.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: the MQ6 trunk GEMM runs from a certified builder (PeaceMaker) module by default (`HIPFIRE_QWEN4_MQ6_X4_PM`; `=0` keeps the hipcc `gemm_mq6g256v2_wmma_gfx12_bt8_x4`). Byte-identical; the kernel is 1.19–1.20× faster on the large trunk shapes, and pp8192 is +0.8 % at chunk 4096 and +3.5 % at chunk 8192.** `kernels/qwen4_mq6_x4_pm_gfx1201.hxaco` (`hipfire-isa emit --kernel qwen4_mq6_x4`, entries `_w4` for M ≤ 64 and `_w8`) is an exact twin of the F32 overwrite entry. It keeps the same 6-bit unpack, the same dequant (`v_cvt_f32_ubyte0`, `v_cvt_f16_f32`, one single-rounding `v_fma_f16`), the same gfx12 WMMA lane map, and the same ascending 16-K accumulation chain per output, with the `0 + acc` F32 epilogue. Only tiling and scheduling change: a wave owns 16 rows × 256 tokens, so one dequant feeds 16 WMMAs instead of 8. X is staged in double-buffered 32-K LDS chunks with one barrier per chunk. B fragments rotate through an 8-deep register ring, so there is no per-WMMA `s_wait_dscnt 0`. BF16-out, residual, regions and HC-write launches keep hipcc. M7 is obligation-free and the lift is byte-exact for both symbols.
  - Exactness (R9700, lab harness, memcmp against the hipcc object): there are zero differing bytes in every one of these checks.
    - The six pp8192 trace shapes (M 10240/2560/6144/12288/640/64), 3 processes each, both entries.
    - 216 further runs over 12 shapes, including ragged M/N/K-tail shapes, at random, rounding-boundary and inf/NaN-header inputs; 134 of these runs produce non-finite outputs.
  - Speed: for the kernel alone (`_w8`, 3 processes), 10240×2560×4096 goes from 2396 / 2430 / 2551 µs to 2009 / 1979 / 2133 µs (1.19 / 1.23 / 1.20×). Inside the model (kernel trace, `qwen4_qsa_ctx` ctx 8192), the trunk MQ6 GEMMs drop from 977.4 to 846.1 ms (−13.4 %).
  - Identity (card B, canonical `qwen3.8-flash-next-gptq3.mq4`): the `qwen4_qsa_ctx` ctx 16384 final-row logits md5 is `6b5a824c…` for beta `5d172b6639`, for the default and for `=0`.
  - pp8192 (`hipfire bench --matrix --pp 8192`, fresh processes in ABBA order against beta `5d172b6639`, warm sample, ranges disjoint):
    - chunk 4096: 3,595.0 / 3,601.0 / 3,596.4 → 3,626.0 / 3,633.3 / 3,625.9 tok/s. Staging exposure absorbs most of the kernel saving.
    - `HIPFIRE_PREFILL_CHUNK_ROWS=8192`: 3,858.5 / 3,852.6 / 3,853.9 → 3,987.4 / 3,986.5 / 3,988.8 tok/s.
  - Kernel idproof: identical on gfx1100, gfx1151 and gfx1201 (no `.hip` source changed).
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: the large MQ6 trunk GEMMs run from a certified builder (PeaceMaker) module by default (`HIPFIRE_QWEN4_MQ6_X4_PM`, the gfx1201 flag above; `=0` keeps the hipcc U3 / `gemm_mq6g256v2_wmma_gfx11_bt8_x4*` kernels).** The selective route takes only the plain F32 overwrite, the BF16-output overwrite and the a/b/z regions fold, on their 128-row 256-thread `_w8` entries of `kernels/qwen4_mq6_x4_pm_gfx1151.hxaco` (`hipfire-isa emit --kernel qwen4_mq6_x4_gfx11`; `qwen4_mq6_x4_pm_gfx1151_w8`, `_w8_bf16out`, `_w8_regions`; same arguments and output bytes as the hipcc entries, 256-token tiles), and only for launches of at least 2560 rows (regions: summed rows) and 2048 tokens, which are the large pp8192 trunk shapes. The 64-row `_w4` entries, the HC-write (`hcw`) launch, smaller launches and the residual route keep the U3 / hipcc kernels, as do gfx1100 and gfx1201 (gfx1201 routing is unchanged). New tests: a CPU-only threshold test and `mq6_x4_pm_gfx1151_matches_hipcc_bytes` (`--ignored`, exact gfx1151), which compares the module against the hipcc launches byte for byte on plain, BF16-output and regions shapes at and above the thresholds with ragged row and token tails.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: the HC-down GEMM runs on a 160×80 split-K tile with K staged 32 by default (`HIPFIRE_QWEN4_HC_DOWN_TILE`; `=0` keeps `qwen4_wmma_lds_64_128_32_64_k64_p_s4`). Byte-identical; the kernel is 1.16× faster at 4096 rows and 1.25× at 8192, and pp8192 is +0.1 % at chunk 4096 and +0.45 % at chunk 8192.** The new gfx1201-only entry `qwen4_wmma_lds_160_80_32_80_k32_p_s4` (five 32×80 waves, 178 VGPRs, no scratch) keeps the incumbent's four K splits, ascending 16-element WMMA substeps and fragment map, so the F32 partials and the reduced `low` are bytewise the incumbent's. It needs 15,360 B of LDS per workgroup instead of 24,576 (which capped a WGP at five workgroups). It was chosen from a sweep of about 60 exact hipcc tiles; direct-from-global loads and VGPR-capped variants were slower.
  - Exactness (R9700, lab harness, memcmp of the partials and the reduced output against the beta hipcc object): zero differing bytes in 425 checks on the shipped object. They cover M 320/319/161/100/4/1, K 256/10240/20480 and N 1–8192 including ragged tails, at random, wide-exponent/subnormal and ±inf/NaN inputs; 19 checks produce non-finite outputs. The ignored GPU test `hc_down_tile_gfx1201_matches_p_s4_bytes` passes.
  - Speed (kernel alone, untraced, 3 processes × 20 ABBA rounds): 320×10240×4096 goes from 451 / 455 / 458 µs to 388 / 392 / 393 µs (1.16 / 1.16 / 1.17×); 320×10240×8192 is 1.25 / 1.24 / 1.26×, 320×20480×4096 1.18 / 1.15 / 1.20× and 320×20480×8192 1.25 / 1.25 / 1.26×. Between 512 and 6000 rows it is 1.09–1.26×.
  - Inside the model (rocprofv3 kernel trace, daemon `bench_prefill` 8192 at chunk 4096, two steps, beta `547250af59` vs the branch): the 193 HC-down launches per step drop from 86.2 to 79.9 ms (1.08×); the split-K reduce is unchanged at 5.9 ms. Less than half of the isolated-kernel gain carries over at chunk 4096.
  - Identity (card B, canonical `qwen3.8-flash-next-gptq3.mq4`): the `qwen4_qsa_ctx` ctx 16384 final-row logits md5 is `6b5a824c…` for beta `7878272585` and `547250af59`, for the default and for `=0`. Greedy AR and MTP decode tokens match beta on 8 of 8 prompts, and the greedy serve battery is identical across MTP on, MTP off and beta (5 of 5 turns).
  - pp8192 (`hipfire bench --matrix --pp 8192`, fresh processes in ABBA order against beta `7878272585`, warm sample, 4 per arm):
    - chunk 4096: 3,629.3 / 3,626.3 / 3,626.5 / 3,627.1 → 3,630.5 / 3,630.1 / 3,629.4 / 3,629.1 tok/s (+0.08 %; the ranges touch). Staging exposure absorbs the kernel saving.
    - `HIPFIRE_PREFILL_CHUNK_ROWS=8192`: 3,990.3 / 3,988.6 / 3,985.2 / 3,986.4 → 4,006.9 / 4,006.5 / 4,004.0 / 4,002.8 tok/s (+0.45 %, disjoint).
  - Kernel idproof: identical on gfx1100, gfx1151 and gfx1201. The split-K module is JIT-only and not in the idproof pack; rebuilt by hand, its gfx1151 object is byte-identical and every existing gfx1201 entry disassembles identically.
- **Qwen4 symmetric IU4 MoE on gfx1151: experimental down NT4 row-repeat entries (`HIPFIRE_QWEN4_MOE_SYM_DOWN_RR=1|2|4`, read once, default 1).** A CTA reuses its expert-run prologue for two or four contiguous 64-row blocks; `M` must be divisible by `64*rr`. The existing entries and gfx1201 module are byte-identical. Both repeats match the captured real down launch under full-grid CPU emulation; the rr2 `one` route at 1536 tokens matches every Halo anchor, including rr4, over both whole output buffers. No performance claim: hardware drain stopped the remaining route, graph, sweep and timing gates.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: long prefill stages host-mapped experts into VRAM by default (`HIPFIRE_QWEN4_EXPERT_STAGE`, `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS` = 2048). Byte-identical; decode and short prompts keep their residency.** A prefill chunk of at least 2048 rows DMA-copies each host layer's expert bytes into VRAM ahead of its MoE. The two stages borrow the VRAM of the last two resident expert layers, and those layers are restored from pinned host copies (2,550 MiB of host RAM, taken at load) before the chunk returns. `auto` placement no longer reserves stage buffers. On an R9700 (card B, `auto` = 14 layers, single clean processes): pp2048 1057.1 → 1376.8 tok/s, pp4096 1215.1 → 1411.2, 16K prefill 1209.7 → 1395.5. The unhidden restore costs 11.5–18.6 ms per chunk (2 × 23.5 ms copies). tg64 at 512/8K is unchanged (34.07/31.95 vs 34.12/32.00), and 16K logits and greedy text are byte-identical to `HIPFIRE_QWEN4_EXPERT_STAGE=0`. Staging is off when the host RAM or GTT checks refuse the copies, and inside a stream capture. Other arches stay off unless `1` is set.
- **Qwen3.8-Flash-Next (Qwen4): default flips.** Each one has an opt-out.
  - **The gathered QSA attention runs from the certified builder (PeaceMaker) module by default on gfx1151 and gfx1201 (`HIPFIRE_QWEN4_QSA_PM`; `=0` keeps the hipcc kernels).** It has the hipcc kernels' ABI, grid, LDS layout and output bytes; a cache too large for its 32-bit buffer offsets keeps the hipcc kernels.
  - **The Halo hyper units are on by default on gfx1151 only: `HIPFIRE_QWEN4_HC_FUSE=3`, `HIPFIRE_QWEN4_HC_UP_TILE`, `HIPFIRE_QWEN4_GDN_CONV_QKNORM` and `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`.** All four are bytewise identical to the launches they replace. `=0` turns each one off. gfx1201 takes `HC_FUSE=1` and `HC_UP_TILE` by default with its F16 WMMA route (entry above) and `MOE_COMBINE_ZINIT` by default (entry above); `HIPFIRE_QWEN4_GDN_Q8_INLINE` (H6) stays opt-in.
  - **Sampled native MTP is on by default (`speculation.mtp_sampled` / `HIPFIRE_MTP_SAMPLED`; `=0` sends sampled requests to AR). It landed in this release as an experimental opt-in.** Before this, temperature > 0 requests always ran AR. Native MTP now verifies them by speculative rejection sampling (Leviathan et al. / Chen et al.). The draft token is drawn from `q`, the request's truncation applied to the draft head's 8 re-scored candidates. The re-score kernel now also writes those candidates and their exact logits; the argmax and margin words are unchanged. The draft is accepted with probability `min(1, p/q)`. A rejection emits a draw from `(p − q)+`, and a window whose drafts are all accepted draws its bonus from `p`. `p` is exactly the Qwen4 AR sampler's distribution (`llama::sample_top_k_p`): temperature, the request's top_k (absent = 20 candidates, 0 = the 64-wide pool), min_p and top_p, with the same pool, tie order and summation. `SpecRequestConfig.top_k` now carries the request value (`Option<u32>`) so absent and 0 stay distinct; chain verifiers still treat both as no cut. Greedy requests and non-neutral penalties route as before. The output matches AR in distribution, not byte for byte, and the same seed replays the same tokens. A think-budget force-close and an EOS inside an open think span go through the same emitter as greedy MTP. Evidence:
    - CPU, 200k trials per case, top_k ∈ {absent, 5, 20, 40, 64, 0} × min_p ∈ {0, 0.05, 0.1} at T1.0 / top_p 0.95, five other temperature/nucleus cases and a tied-logit row: `llama::sample_top_k_p`'s draws follow `p`, and verified draft-from-`q` emissions follow `p`, every case at α = 1e-6 with total variation < 0.01 (largest 0.008); acceptance equals Σ min(p, q). A two-position chained window with EOS also matches. Drawing the rejection replacement from `p` instead of the residual fails the same test in every control case (χ² ≥ 18,000).
    - Strix Halo, canonical GPTQ3 artifact, 2,000 cold trials per arm, T 1.0 / top_p 0.95 / top_k 40 / min_p 0.05: the first four generated tokens and the (t1, t2) joint match AR under two-sample χ². Batched route p = 0.29 / 0.66 / 0.83 / 0.90, joint 0.66; interleaved route p = 0.29 / 0.32 / 0.13 / 0.54, joint 0.32. On replay with the same seed, each route emitted the same tokens. Before the top_k/min_p change the same test passed at T 1.0 / top_p 0.95 and T 0.7 / top_p 0.8 with top_k absent, whose `p` is unchanged.
    - With the switch on, greedy serve battery and chain text is byte-identical with MTP on vs off (5/5 turns each).
    - Speed (Strix Halo, serve, ABBA fresh processes, T0.7 / top_p 0.8 / top_k 20, seed 7): `lru_cache_pep8_strict` 27.6 → 41.7 decode tok/s (1.51×, τ 1.41). Coherence at T0.7 / top_p 0.8 / top_k 40 / min_p 0.05, seeds 11 and 22, battery and chain: every MTP turn on-task and finished `stop`, no runaway or attractor; turn 2–5 decode 37.5–62.5 tok/s (τ 1.08–2.43) vs AR 32.0–33.7. In chain mode the model sometimes ends a turn inside its reasoning on both routes (AR 4 of 10 turns at seeds 11/22; MTP 3 of 20 at seeds 33–66); those turns end `stop` with reasoning only. A 64-token think budget (T1.0, thinking low) force-closes the reasoning on the sampled MTP route through the shared emitter, and every turn then answers.
    - Default flip (Strix Halo, canonical GPTQ3 artifact, serve_harness, MTP `auto`, thinking off, 768-token cap, no `HIPFIRE_*` overrides): greedy battery and chain text is byte-identical to beta `a193e4491` (5/5 turns each, same request md5s, every turn `drafter=mtp`). At T0.7 / top_p 0.8 / top_k 20 and at T1.0 / top_p 0.95, seeds 11 and 22, battery and chain (40 turns): every turn routes `drafter=mtp`, stays on task and finishes `stop`, with no runaway, attractor or empty turn (τ 0.94–2.83). A request that leaves `repeat_penalty` unset also takes MTP, with the same text as `repeat_penalty: 1.0`. `repeat_penalty: 1.1` and `HIPFIRE_MTP_SAMPLED=0` both run AR (no MTP route line or MTP timings, decode 32.3–32.7 tok/s).
    - **Lab alternative: SpecInfer-style naive sampled verify (`HIPFIRE_MTP_SAMPLED_MODE=naive`; unset or `leviathan` keeps the default above; any other value fails sampled requests)**, ported from #774 by @fivetide (`4bc6b9e2`). The draft head proposes its argmax; each verify row is drawn with the AR sampler and the request's config from the shared AR sampler RNG, and a draft is accepted iff it equals its row's draw. Every emitted token is one AR draw in AR's order, so a seeded request emits AR's exact ids wherever the verify logits equal AR's.
      - Identity (Strix Halo, canonical GPTQ3 artifact, 36 seeded cases: lru_cache and prose_river, 6 seeds each at T0.7/top_p 0.8/top_k 20, T1.0/top_p 0.95 and T0.8/top_p 0.95/top_k 40/min_p 0.05): naive ids equal AR's on 36/36 cases on the batched, interleaved and adaptive routes. The 23/24 default-route miss reported on #774 does not reproduce, also not with beta's few-row HC BF16 fix (`ec60f4377e`) undone. A CPU test (toy target, windows of depth 1–4, top_k 0/40 and min_p) checks the same property for 24 seeds. With the variable unset, the Leviathan route's ids equal `046b57fa46`'s for 100 of 100 seeded trials, and its 2,000-trial distribution test still matches AR.
      - Speed (Strix Halo, serve, one binary, 3 fresh processes per arm in ABC CBA ABC order, seed 7, 1,024-token cap; decode tok/s median, τ): `lru_cache_pep8_strict` T0.7/top_p 0.8/top_k 20: AR 32.1, Leviathan 44.7 (τ 1.23), naive 52.0 (τ 1.64); T1.0/top_p 0.95: AR 32.0, Leviathan 49.7 (τ 1.48), naive 37.8 (τ 1.10). `prose_river_short`: T0.7 AR 32.4, Leviathan 36.5, naive 36.4; T1.0 AR 32.5, Leviathan 37.5, naive 34.3. Each arm emits one seeded trajectory, so the arms decode different text. Leviathan stays the default.
  - **The symmetric IU4 MoE prefill route is requested by default on gfx1151 and gfx1201 (`HIPFIRE_QWEN4_MOE_SYM_IU4`; `=0` opts out).** Only layers whose routed-expert headers pass the load-time symmetric check take it, so a symmetric requant (GPTQ3) runs it and the shipped asymmetric `.mq4` stays on the F16 WMMA route automatically. The trade on the GPTQ3 artifact (Strix Halo, with this release's other defaults): pp8192 1,928.5 tok/s (median of 3) vs about 1,455 on the F16 route, BF16-reference KLD 0.1027 vs 0.0661.
    - gfx1201 (R9700 card B, `auto` placement, canonical GPTQ3 artifact sha256 `8b15b6fe…0972`, one binary built from `a193e4491` (daemon md5 `1061d192…`) with the route selected by env): pp8192 1,449.1 → 2,171.5 tok/s (+49.9 %; 1457.1 / 1446.1 / 1449.1 vs 2174.4 / 2171.5 / 2162.2, a fresh process per sample, pairs 1–2 in ABBA order, the third IU4 sample re-run later after a harness fault; sclk 2.44–2.52 GHz, perf level `auto`).
    - gfx1201 KLD against the BF16 teacher (`fn-bf16.{wt2,code24}.1.bin`, HFKLDR v1; WikiText-2 24 × 2048, code 24 × 512): WikiText-2 0.0572 → 0.0877 (+0.0306, paired bootstrap 95 % CI [+0.0276, +0.0337]), code 0.0733 → 0.1018 (+0.0285 [+0.0213, +0.0362]); top-1 agreement 0.918 → 0.897 and 0.929 → 0.912. The Strix Halo deltas on the same refs are +0.0295 and +0.0275. Agentic task scores on the R9700 tie within their seed spread (TC70-84 mean 81.0 IU4 vs 75.7 F16, HermesAgent-20 85.75 vs 88.5, benchlocal medium 66 vs 64 of 75).
    - With the variable unset, the gfx1201 pp8192 final-row logits (`qwen4_qsa_ctx` CTX 8192, chunk 4096) equal `=1` byte for byte (md5 `a78c15a6…`), and `=0` reproduces the previous default's `44df6365…`. Greedy decode text is unchanged for prompts under 512 tokens (decode never takes the route); a 1,024-token prompt's text changes with the prefill numerics.
  - **Native MTP stays on by default on gfx1201 with host-mapped experts (`--spec off` / `speculation.mtp = "off"` opts out).** This supersedes the v0.4.0 entry "MTP stays default-on only where the experts are device-resident" for exact gfx1201: an `auto` load on the R9700 now attaches the head. Other arches with host-mapped experts still keep AR under `auto` and print `qwen4 native MTP: off by default with host-mapped experts on <arch>`.
    - R9700 card B, `auto` placement, canonical GPTQ3 artifact, 8 prompts × 256 greedy tokens: MTP text equals AR on 8/8 prompts in each of 9 processes (3 AR-HIP, 3 MTP, 3 AR-PM4, ABC CBA ABC). Decode medians 34.2 → 45.2 (`glimmer_coding`, τ 2.4), 33.8 → 48.0 (`code_edit`, τ 2.81), 34.9 → 35.5 (`agentic_user_read`, the low end); median ratio 1.19×, MTP faster on 8/8. The v0.4.0 text mismatches on 4/8 prompts do not reproduce on this artifact.
    - Cost: the head takes two expert layers' VRAM (VRAM expert layers 0..15 → 0..13, pinned host 42.4 → 44.9 GiB) and prefill takes 5–12 % longer (1,024 tokens: 1,590 → 1,777 ms). The first MTP request against a cold kernel cache is slower while kernels compile.
  - **MTP-off Flash-Next decode on gfx1201 runs the retained Redline PM4 tape by default (`replay.backend = "hip"` / `HIPFIRE_REPLAY_BACKEND=hip` opts out).** `retained_redline_default` now admits `qwen4` on exact gfx1201 with PP=1, TP=1 and no drafter, the same gate as Qwen3.5 dense. A native MTP head is a drafter, so this applies only to loads with MTP off (`--spec off`); the default MTP load keeps its spec path. The load logs `[redline] enabling fail-closed retained default on gfx1201 (model_arch=qwen4, drafter=off, transport=pm4)`; a failed preparation or replay falls back closed to HIP.
    - `scripts/redline_daemon_harness.py --qwen4 --pm4 --skip-prefill --shadow-iterations 5` passes: capture stable at 973 launches (28 kernels), and HIP, PM4 and blob are bit-exact at positions 129–133 over 42.57 MB of state per position. Greedy text equals HIP on 8/8 prompts.
    - Serve (MTP off, greedy), one PM4 route-proof line per request: battery turn text equals the previous binary's HIP route and MTP `auto` on 5/5 turns; chain turn text equals MTP `auto` on 5/5 (the previous binary differs from the third turn on, where the prompt passes 512 tokens and the IU4 prefill above applies). Durability stress, 8 prompt lengths (511–8,193 tokens) solo ×2 plus queued concurrency 2 and 4, 64 requests per arm: 0 errors, and every PM4 and every HIP response equals the PM4 solo reference.
    - Decode, R9700 card B, 3 fresh processes per arm, 8 prompts: per-process means 34.20 / 34.15 / 34.20 (HIP) vs 35.01 / 35.01 / 35.08 tok/s (PM4); median per-prompt gain +2.3 % (range −0.3 % to +4.4 %). The tape is unpaced. railgun's lowering is not used: it refuses this program (`route=unmatched`, 15 segments) and falls back to HIP.
    - The Qwen4 weight sweep now registers its allocations as `Weights` in the `hip_bridge` allocator registry, as Qwen3.5 does, so `HIPFIRE_RAILGUN_CHECK=always` snapshots only live state instead of failing `hipMalloc` while copying the weights. On card B (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=6`), the PM4 lowering and its HIP twin are byte-equal on 23 of 23 decode steps (positions 244–266, 579 `State` surfaces, 5.03 GB per step), and the 1,108 `Weights` allocations (13.4 GB) are unchanged across the first step. The `flip_byte` negative control comes back unequal on exactly the flipped byte and poisons the route.
- **Qwen3.8-Flash-Next (Qwen4): live continuation inside a conversation, on top of the session cache above (AR and native MTP); the RC3 prefix and radix caches are removed.** A turn whose canonical tokens strictly extend the previous turn's committed consumed history (same route; for native MTP the head is at the same position) prefills only the suffix in place, with no copy; a hit always prefills a non-empty suffix. That continuation is session-exact (its state descends from decode, as in one continuous conversation), not cold-exact, so a live turn captures no snapshot and the cross-session cache keeps holding cold-exact state only; when the live state and the longest cached snapshot reach equally far, the live state is used. An unrepaired terminal (the consumed history is empty) leaves no live state, and the next turn plans from the cache's snapshots or prefills cold. Live continuation needs the session cache attached, so `HIPFIRE_SESSION_CACHE_BYTES=0` turns it off together with the snapshots and every turn prefills cold. The RC3 Qwen4 end-of-prompt checkpoint, the multi-entry radix store (exact-boundary checkpoints on VMM context storage with typed PM4 relocation), shared-turn anchors, periodic captures and the `[qwen4-radix]` / `[qwen4-prefix]` trace lines are gone; the session cache's chunk-multiple snapshots replace them. `HIPFIRE_QWEN_PROMPT_CACHE` and `HIPFIRE_QWEN_RADIX_CACHE` no longer affect Qwen4 (`HIPFIRE_QWEN_PROMPT_CACHE` still gates the Qwen3.5 and dense conversation cache; `HIPFIRE_QWEN_RADIX_CACHE` is read by nothing). `session_cache_hw` (`HIPFIRE_SESSION_CACHE_MODEL`) is the hardware test, and `qwen4_parity --mode cache-quality` is the reference-parity hit-vs-cold logits oracle of the session cache (`HIPFIRE_CQ_SESSION_BUDGET_BYTES` sets its cache budget; `HIPFIRE_QWEN4_CACHE_QUALITY_OUT` names the opt-in capture file).
- **Qwen3.8-Flash-Next (Qwen4): opt-in message-end session snapshots (`HIPFIRE_QWEN4_TURN_SNAPSHOTS=1`, default off; idea by fivetide).** With the switch on, every prefill also splits and snapshots right after each `<|im_end|>`, and live-continued turns capture too, so another session sharing a conversation prefix restores the deepest message end instead of re-prefilling below one prefill chunk. The snapshots are fivetide's self-contained copy snapshots (GDN/PLE/hyper, QSA rows as deltas, MTP head and draft policy), published at client commit, with no radix aliasing or relocation. They are session-exact, not cold-exact, and live in their own scope. The session-cache pool accepts these non-page-aligned boundaries (`CheckpointPool::insert_unaligned`). No hardware measurement yet; the default stays off until TC gates it.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: bytewise-identical fused PLE block, on by default (`HIPFIRE_QWEN4_PLE_FUSE`; `=0` keeps the incumbent launches; default flip above).** The once-per-prefill PLE chain (BF16-stream widen, `grouped_gate_bf16`, `grouped_norm_bf16`, depthwise conv, stream add) becomes three launches (`ple_gate_rows_bf16s`, `ple_norm_inv`, `ple_conv_add_bf16s_k4d3`) that keep every serial reduction order and F32 rounding point, and the PLE rows are gathered straight to F16 for the key/value GEMMs. Exact gfx1151 only, BF16 HC streams, eager (never under a recorder, retained tape or graph capture). Standalone kernel chain at 8192 rows: 23.35 -> 8.1 ms (gather + F32->F16: 1.18 -> 0.34 ms); streams and conv state are 0 differing bytes at 8192/8188/512/17/10/9/8/3 rows and Flash-Next 8192-context logits md5 matches with the flag off and on. pp8192 on Strix Halo over the arm-c flags (3 interleaved fresh processes per arm): 1874.6 -> 1886.5 tok/s median (-27.6 ms/step, within run-to-run noise of the ~16 ms kernel saving). Idea from Gufo upstream (gufo-org/gufo @1071b361, MIT).
- **Qwen4: experimental projection-region LDS WMMA launch (`HIPFIRE_QWEN4_PROJ_REGIONS=1`, default off).** The admitted F16 route combines router/shared gate/up/selector output rows without packing weights, sharing X staging across region boundaries. gfx11 HC-down can use the same ascending-K kernel alone. Native gfx1201 fragments are supported; its exact short selector and split-K=4 HC-down remain separate. This flag does not enable the F16 route, alter PLE/shared-down dispatch, or change HC-write hooks.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: two opt-in, bytewise-identical GDN prefill fusions (`HIPFIRE_QWEN4_GDN_CONV_QKNORM=1`, `HIPFIRE_QWEN4_GDN_Q8_INLINE=1`; both default off).** Both apply only to the chunked GDN prefill route (≥ 512 rows, eager, `HIPFIRE_QWEN4_F16_WMMA` on) on exact gfx1151; gfx1100, gfx1201 (persistent GDN route) and every other arch ignore them, and the existing gfx1201 F16 opt-in is not enabled by them. Unset or `0` keeps every launch of the incumbent route; each flag is its own kill switch.
  - `HIPFIRE_QWEN4_GDN_CONV_QKNORM`: the GDN convolution launch also stores the normalized Q/K (`gated_delta_conv_qknorm_bf16_f32_batched_k4`), and the separate `gated_delta_qk_norm_bf16_batched` launch is skipped. Needs 16 key heads of 128 in q|k|v order, `channels % 256 == 0`, a 3-row ring and a BF16 convolution output; any other shape, a row capture, a recorder or a graph capture keeps the two launches. The convolution output's Q/K columns are not written on this route (the recurrence reads only V from them).
  - `HIPFIRE_QWEN4_GDN_Q8_INLINE`: a Q8 GDN state is decoded and requantized inside the chunk recurrence (new module `gated_delta_chunk_q8_wmma`, inventoried for gfx1151 only) instead of by the `gdn_state_q8_to_f32` / `gdn_state_f32_to_q8` launches around it.
  - Evidence (Strix Halo, gfx1151, scoped `rdna-compute` tests): gated output, final state, convolution ring, gate/beta and the Q8 slot are byte-for-byte the incumbent arms' (530 rows with a ragged tile and a nonzero ring cursor; 512–530 rows at start positions 0 to 262143 for Q8). The existing Q8 chunk test's slot digest is the same with the flags on and off. Kernel packs vs `fddd7ada5`: gfx1100 and gfx1201 are identical (132 and 137 modules); gfx1151 is identical in 121 modules and adds `gated_delta_chunk_q8_wmma`. The new convolution kernel lives in the JIT-only `tensor_ops` module, which no pack carries.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151/gfx1201: three more opt-in, bytewise-identical prefill fusions (`HIPFIRE_QWEN4_HC_FUSE=1|2|3`, `HIPFIRE_QWEN4_HC_UP_TILE=1`, `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT=1`; all default off).** Exact gfx1151 and gfx1201 only (gfx1100, gfx1200 and every other arch ignore them); each needs the F16 WMMA prefill route it hooks into (gfx1201 also `HIPFIRE_QWEN4_F16_WMMA_GFX1201=1` for the HC read) and none runs under a recorder, a retained tape or a graph capture. Unset or `0` keeps every launch of the incumbent route; each flag is its own kill switch. A stage whose shapes or routes it does not admit runs the incumbent launches, having launched nothing of its own.
  - `HIPFIRE_QWEN4_HC_FUSE`: level 1 makes the HC read's norm launch also project its paired HC write's gates (`hyper_norm_gate_outputs`), so the write skips its own norm + gate launch; level 2 also has the MQ6G256V2 attention output projection (GDN `output`, QSA `o_proj`) apply the HC write in its GEMM epilogue (`gemm_mq6g256v2_wmma_gfx11_bt8_x4_hcw`, gfx1201 `gemm_mq6g256v2_wmma_gfx12_bt8_hcw` from 64 rows), so the attention output is never materialized; level 3 (gfx1151) also folds the BF16 scaled add and the HC write into the BF16 shared-expert down GEMM (`gemm_wmma_lds_128_256_32_64_k64_hcsd`). At level 3 the routed `moe_output` rows are not rewritten with the shared-down sum (nothing reads them after the write). The HC streams are bytewise those of the unfused sequence.
  - `HIPFIRE_QWEN4_HC_UP_TILE`: the HC read's up projection + branch mix runs on retiled operand-swapped entries (`hyper_read_up_wmma_bf16_swap` on gfx1151; `hyper_read_up_wmma_bf16_gfx1201_t128` on gfx1201 when `low_rank % 64 == 0`), bytewise the baseline entries' output.
  - `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`: the qt44 BF16-row grouped combine starts from +0.0 (`moe_down_combine_grouped_top10_bf16in_zinit`; the original symbol is unchanged) and the `moe_output` zero fill before the sealed MoE call is not launched. The fill stays whenever the call is not that route, the target is not exactly the cleared tensor, or anything is recorded or captured, and a fill taken over but not consumed by the combine fails the call instead of leaving the target uncleared.
  - Evidence (scoped tests, flags forced on in the test body, Strix Halo gfx1151 and a Radeon AI PRO R9700 gfx1201): HC read gates (`hyper_norm_gate_outputs`) vs `hyper_norm_gate` + `hyper_norm_f16` at 1, 37, 512 and 530 rows, F32 and BF16-bit streams, both arches byte-for-byte; HC up+mix retiled vs baseline at 131 ragged rows, both arches byte-for-byte; attention-epilogue GDN/QSA output projection + HC write vs projection + pregated write at 131 and 530 rows, F32 and BF16-bit streams, F32 and BF16 activations, both arches byte-for-byte; shared-down fold vs F16 GEMM + scaled add + `hyper_write` at 512, 530 and 1100 rows, F32 and BF16-bit streams, with and without the routed rewrite, gfx1151 byte-for-byte; zero-init combine into a dirty target vs zero fill + the unchanged combine at 1, 37 and 530 tokens, both arches byte-for-byte (kernel harness: 48 cases EXACT0 on gfx1201). R9700 per-call medians over three fresh processes, zero fill + combine → zero-init combine: 0.401 → 0.378 ms at 2048 tokens, 1.289 → 1.008 ms at 8192. Kernel packs vs `fddd7ada5`: gfx1100 and gfx1201 are identical (132 and 137 modules); gfx1151 is identical in 121 modules and adds only `gated_delta_chunk_q8_wmma`. Every other new entry lives in a JIT-only module (`tensor_ops`, `qwen4_gemm_mqv2_wmma_gfx11_bt`, `gemm_mq6g256v2_residual_wmma_gfx12_bt*`, `gemm_wmma_lds256`, `moe_down_combine_grouped_top10`) that no pack carries. A full-model forward with the flags on has not been run.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151: bytewise-identical MoE row fold, on by default (`HIPFIRE_QWEN4_HC_ROW_FOLD`; `=0` keeps the incumbent launches; default flip above; idea from Gufo upstream, gufo-org/gufo @1071b361, MIT).** With `HIPFIRE_QWEN4_HC_FUSE=3` and `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT=1`, when a sealed MoE call's HC write is directly followed by the next layer's HC read of the same streams (F16 WMMA read route, BF16 streams, eager), the shared-down GEMM stores BF16 rows (`gemm_wmma_lds_128_256_32_64_k64_bf16st`) and one per-token kernel (`hc_row_fold_norm_gate`) replaces the combine, the `hcsd` fold + HC write epilogue and the next read's `hyper_norm_gate_outputs`: it combines the ten expert rows in rank order, folds the shared row, writes the four streams and emits the next read's F16 normalized row and its paired write's gate logits; that read then skips its norm-and-gate launch. Unset or `0` keeps the incumbent launches.
  - Evidence (Strix Halo, gfx1151): standalone harness, 18 cases (N = 8192, 8188, 512, 17, 1; tied experts, -1 and out-of-range routes, +-0, NaN, inf, F16/BF16 extremes in every operand) memcmp of streams, F16 row and gate logits: 0 differing bytes; `qwen4_qsa_ctx` CTX=8192 logits md5 identical flag off and on. N = 8192 per MLP layer: combine 2.44 + hcsd 2.72 + norm-gate 1.75 = 6.91 ms -> bf16st 1.09 + row fold 4.91 = 6.00 ms (-0.91 ms).
- **Qwen3.8-Flash-Next (Qwen4): gathered F16 WMMA QSA prefill attention, default on for gfx1151 and gfx1201 (`HIPFIRE_QWEN4_QSA_WMMA_GATHER=0` opts out).** Prefill chunks of ≥ 512 rows that the full-window dense route does not take run on new modules: `indexed_attention_gathered_wmma.gfx1151.hip` (gfx1151, F32 state) and `indexed_attention_gathered_wmma.gfx1201.hip` (gfx1201, fp8 state). With `=0`, every launch is still `indexed_attention_attention_*_batched_hg4`, and every existing code object is byte-identical on gfx1100, gfx1151 and gfx1201.
  - Each call first converts the layer's cache rows `[0, end)` to F16 K plus a block-transposed V. One workgroup per (row, KV head) then walks the row's selection in 128-entry tiles.
  - The conversion uses the route's own scratch (2 KiB per context token, 512 MiB at 262K). It is reserved for the whole `max_seq` at load, before any capture or record, so it never grows under a captured graph or tape. `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=auto` charges it to its reserve; with `=0` the reserve is unchanged.
  - The route is not bit-exact. In-model QSA attention error against an f64 reference over the same fp8/F32 cache and selection is rel-L2 1.0–2.0e-4 per row (about 9× below a BF16-storage model of the same Q/K/V/P rounding), all of it F16 operand rounding. Whole-model logits are not bit-identical to flag-off and the network amplifies any numeric perturbation: per-row KL(off‖on) of 1e-2–1e-1 occurs at 2K–4K on gfx1201, and a near-tied argmax can flip. That is the same order as KL(off‖f64-attention), and against the f64 reference the gathered route is no farther than flag-off (gfx1201, Flash-Next 2K/4K, 385/897 rows: 99%-trimmed mean KL 4.2e-3 vs 8.6e-3 and 1.05e-2 vs 2.0e-2). Final-row KL(off‖on) ≤ 2.3e-4 holds only for the sampled 16K/64K/128K rows.
  - Acceptance bar: error against an f64 reference of the same quantized source no worse than BF16 storage of Q/K/V/P. On the worst-error 1536-row snapshot per arch the route measured max |err| 1.93e-3 (gfx1151) / 2.78e-3 (gfx1201) against 1.58e-2 / 1.68e-2 for BF16 storage, and p99 9.8e-5 / 1.24e-4 against 1.01e-3 / 1.23e-3. This compares aggregate max/p99 on two snapshots, not each element (93.4–93.5% of elements fall within their BF16-storage error), and is not a model-global bound.
  - `hipfire bench --matrix`, flag on vs off, 3 fresh processes each:
    - Strix Halo: pp8192 996.5–1001.9 → 1186.3–1190.9 tok/s; pp65536 892.4–896.6 → 1080.6–1084.9 tok/s.
    - R9700 (12 expert layers in VRAM): pp8192 710.5–718.2 → 769.6–783.2 tok/s; pp65536 707.2–716.1 → 774.0–790.2 tok/s.
- **Qwen3.8-Flash-Next (Qwen4): the MQ6G256V2 X-LDS trunk projections are default on inside the Qwen4 forward, byte-identical: U2 on gfx1201 (`HIPFIRE_QWEN4_MQ6_X4_GFX1201`), the U3 tile table and a/b/z region fold on gfx1151 (`HIPFIRE_QWEN4_MQ6_X4_TILE`, `HIPFIRE_QWEN4_MQ6_X4_REGIONS`).** Unset, U2 takes prefill chunks of ≥ 512 rows (decode keeps the GEMV), and the gfx1151 tile table (`docs/quant-formats/mq-v2-family.md`) and the fold apply only inside the Qwen4 forward; no other model changes route. An explicit value applies to every MQ6G256V2 GEMM as before, and `0` keeps the incumbent route. Every arm keeps each output's ascending-K WMMA chain.
- **Qwen3.8-Flash-Next (Qwen4): default-off, byte-exact QSA selector arm (`HIPFIRE_QWEN4_QSA_SELECT_EXACT`).** It is a config-owned `=1` opt-in read once (`0`/unset = every launch is the incumbent's, byte-identical to before); it changes no default, conversion cache, scratch reservation or accounting, or the dense-first / >= 512-row gather eligibility.
  - `HIPFIRE_QWEN4_QSA_SELECT_EXACT=1`: the batched selector runs `indexed_attention_select_{f32,bf16}_batched_exact` (new `indexed_attention_select_exact.hip`: 256-element tile sorts and a fixed-order top-512 merge instead of all-pairs ranks) for <= 2048 complete pooled blocks and <= 512 budget blocks outside a recorder or capture; the exact gate is identical selected and mirror bytes. Other launches keep the incumbent.
  - **No measured speedup is claimed here.** Planned gates, targets pending the B-card baseline numbers: Strix Halo gathered replacement + conversion <= 470 ms across its 60 pp8192 calls and all-QSA <= 640 ms; gfx1201 <= 180 ms / <= 260 ms; selector <= 40 ms Halo / <= 25 ms gfx1201. The arm is exact-gated by 0 differing bytes (max-abs = max-rel = ULP = 0) against the incumbent selector on the frozen pp8192 snapshots, not by KLD.
- **Qwen3.8-Flash-Next (Qwen4) on gfx1151 and gfx1201: the symmetric IU4 MoE prefill route is on by default (`HIPFIRE_QWEN4_MOE_SYM_IU4`; `=0` opts out). It requires a symmetric requant of the routed experts.** At load, every layer's routed QT44 gate/up and QT53 down headers are checked on the device for the symmetric grid (`zp == -8*sc`). A layer passes only if every header does; prefill chunks of 512+ rows on passing layers then use stable deterministic grouping, A4 activations and grouped IU4 GEMMs. The shipped asymmetric artifacts fail the check and keep the F16 WMMA route. `=0` restores the whole F16 route; decode, MTP, smaller prefill and every other arch never take it. The two grouped IU4 GEMMs are certified builder (PeaceMaker) code objects embedded per arch: expert-run tiles that read each expert's weights once per four 16-slot tiles, or per eight on gfx1201 layers whose experts are host-mapped. Every tile width produces the same bytes as the 16-slot tile and, on gfx1151, as the hipcc GEMMs (`HIPFIRE_QWEN4_MOE_SYM_PM=0` selects those as a same-binary control). The builder is written on the PM M0 typed core (`peacemaker-author`), which gains three scoped forms for it: `Wave::loop_until`/`break_if` (a loop left by a branch, back edge `s_branch`; the code after it continues from the join of its breaks' states, not the back edge's), `Wave::if_else` (each arm starts from the branch point's waits, LDS and hazard state, and the two are joined after it) and `Workgroup::handoff` (writer waves publish an LDS region at one barrier and branch to the kernel exit; reader waves meet them at their own barrier and run the rest of the kernel in wave scope, so no later barrier can wait on the departed writers). The grouping, producers and checkers are separately named JIT modules, and no existing module changes. Not bit-exact against the F16 WMMA route: it trades quality for speed, with the BF16-reference KLD and pp8192 figures recorded above (Strix Halo GPTQ3: 0.0661 → 0.1027 at 1,928.5 vs about 1,455 tok/s; gfx1201 WikiText-2 0.0572 → 0.0877).
- **Qwen4 symmetric IU4 MoE route on gfx1201: host-mapped expert layers now run the NT8 expert GEMMs.** The route chose NT8 by calling `Gpu::host_located` on the expert view. Expert views are borrowed byte views of the packed expert tensor, so that check was false on every layer and NT8 never ran. The projection view now records where its owning allocation lives, and `RoutedExpertWeights::host_mapped` passes that to the route. The borrowed view itself still does not claim host ownership. Measured on card E (R9700), `fn-sym-only.mq4`, route on, layers 0..16 in VRAM: in a 4K rocprofv3 trace, 192 of the 288 gate/up and 192 of the 288 down launches are now `_nt8` (all were `_nt4` before). Greedy output on 3 prompts and the 16K final-row logits (`ceb85a58…`) are byte-identical before and after. pp8192 over 3 fresh processes: 959.1 / 959.9 / 958.9 → 971.0 / 970.7 / 970.8 tok/s.
- **gfx1201: Flash-Next prefill no longer falls off two GEMM-selection cliffs above 1536-row chunks. Outputs are bit-identical.**
  - **MQ6 residual GEMM:** at a 2048-row chunk it dropped from its BT12 tile to BT8, because 2048 is not a multiple of 192. From 1024 tokens it now takes BT12 (ragged last tile included) whenever that still launches at least 256 waves. Tiny-M projections, such as the M = 48 GDN gates, keep BT8. This applies to every gfx12 MQ6 model: BT12 was the fastest tile on every measured shape it now takes.
  - **BF16 `r16w4`/`r16w4t` multirow route:** above 2048 rows it fell back to the four-row kernel, which is 2.0–2.6× slower per row. It now runs up to its 16-bit grid limit.
  - **Bit-identical:** every tile and route keeps each output's accumulation order. `qwen4_qsa_ctx` 16K final logits are byte-identical before and after at 2048-, 4096- and 8192-row chunks.
  - **Speed** (`exp/fn-chunk-clamp` developer chunk override; R9700, host-mapped experts, pp8192, 3 fresh processes per arm, medians):
    - 2048 rows, 12 VRAM expert layers: 825.0 → 866.5 tok/s (+5.0%).
    - 4096 rows, 12 layers: 835.5 → 1006.9 tok/s (+20.5%).
    - 8192 rows, 6 layers: 877.9 → 1082.6 tok/s (+23.3%).
  - **Unaffected:** the shipped 1536-row chunk; H2, which uses neither kernel; and Strix Halo's Flash-Next prefill, which takes the F16 WMMA route.
- **Qwen3.8-Flash-Next (Qwen4) prefill chunk per arch: 8192 rows on gfx1151 and 4096 on gfx1201 (was 1536 everywhere). pp8192 is +36.6% on an R9700 with auto expert placement (with the GEMM cliff fix above) and +10.7% on Strix Halo.** Every prefill chunk re-streams all routed experts, so fewer chunks mean fewer expert passes. The effective chunk had been 1,536 rows by accident: one chunk's PLE n-gram prefetch had to fit the row store's 8 MiB staging buffer (26,214 rows, i.e. 1,638 tokens of 16 heads), which sat below the 2,048-row cap.
  - The PLE row store now sizes its staging from the chunk (`RowStore::with_staging_rows`). `prefill.chunk_rows` / `HIPFIRE_PREFILL_CHUNK_ROWS` is the ceiling, rounded down to 256 rows. Other arches keep 1536.
  - At load the chunk steps down 4096 → 2048 → 1536 until its scratch fits free VRAM with 1 GiB to spare. `auto` expert placement charges the chunk's scratch against its reserve.
  - **gfx1201 rung.** A larger chunk costs VRAM expert layers under `auto`, so the default is the fastest pp8192 that costs at most 3% of the 2048-row rung's tg64. Measured on an R9700 at max_seq 66,560 (pp8192 / tg64 at ctx 512 / tg64 at ctx 8K):
    - 2048 rows: 16 layers; 877.7–879.4 / 35.0 / 32.8.
    - **4096 rows:** 14 layers; 1001.7–1004.1 / 34.0 / 31.9, i.e. −2.9% and −2.7% tg64.
    - 8192 rows: 11 layers; 1076.6–1077.9 / 32.6 / 30.7, i.e. −6.9% and −6.5% tg64, past the bound.
    - v0.4.0 (15 layers): tg64 34.5 / 32.4.
  - **Speculative logits are sized for verify blocks (64 rows), not the chunk.** Previously they held `vocab × 4` bytes per chunk row: 1.45 GiB at 1536 rows, 7.6 GiB at 8192. Prompt advances and the MTP prompt fill read only their final row.
    - MTP greedy serve on Strix Halo is byte-identical, with identical τ, to v0.4.0 and to the build before this change.
  - **Measured (3 fresh processes per arm, v0.4.0 `3c4ef29dd` vs this build, graph on, spec off).**
    - R9700, auto placement, max_seq 66,560: pp8192 733.5–738.4 → 1002.1–1002.3 tok/s; pp65536 730.4–736.1 → 963.0–963.9. Expert layers in VRAM go 15 → 14; free VRAM after load is 3,192 → 3,242 MiB.
    - Strix Halo, resident experts, max_seq 69,632: pp8192 997.9–1009.1 → 1108.9–1112.9; pp65536 897.2–903.3 → 997.8–1001.2. Free GPU pool after load is 21,658 → 16,218 MiB.
    - None of the ranges overlap. The pp operand is the bench's synthetic stream `token[i] = 10 + (i % 1000)`.
  - **Correctness.**
    - `qwen4_qsa_ctx` at 16K and 65K: every sampled QSA row is fma-exact against the CPU reference at the new chunk on both arches (144/144 and 288/288 rows; gfx1201 fp8, gfx1151 F32).
    - Results are not bit-exact across chunk sizes; the final-row argmax is unchanged.
    - The 128K needle answers correctly on both arches.
    - Stress shows 0 mismatches: the gfx1201 H2 co-tenant + Flash-Next MTP run, and Strix Halo retained record/replay.
    - H2 greedy is 33/33 byte-identical. Kernel objects are unchanged on gfx1100, gfx1151 and gfx1201.
- **Qwen3.8-Flash-Next (Qwen4) on gfx11/gfx12: new opt-in q8 QSA K/V state (`kv_cache = "q8"`, `--kv-mode q8`, `HIPFIRE_KV_MODE=q8`).** It uses Q8_0 blocks (32 int8 codes and one f16 scale, the dense q8 KV row) for the trunk QSA layers' K/V. Index keys keep the same BF16 values, so sparse selection doesn't change, and the native MTP layer keeps the exact F32 state. `auto` never picks q8, so the defaults are unchanged: `bf16` on gfx11 and `fp8` on gfx1201. Each trunk layer's context state drops from 4,736 to 1,408 bytes per token. For 12 trunk layers at 262,144 tokens that is 14,208 → 4,224 MiB.
  - Quality (Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, KLD against the HF BF16 teacher refs `kld/bf16-teacher/fn-bf16-{wt2-24x2048,code24-24x512}.kldref`): WikiText-2 24 × 2048 goes 0.085771 → 0.089721 (+0.0040, 95 % paired bootstrap [−0.0010, +0.0118]). Code 24 × 512 goes 0.100982 → 0.098447 (−0.0025, [−0.0075, +0.0022]). Without the opt-in, the build's KLD is identical to base.
  - Correctness: `qwen4_qsa_ctx` at 8K and 32K (chunk 4096) checks every sampled QSA row against the CPU F32 reference. At 32K that is 192/192 rows, 156 of them past the selection budget, with cosine ≥ 0.99999997 and relative L2 ≤ 2.2e-4. The final-row argmax matches the F32 state. Greedy MTP equals AR with q8 (`greedy_mtp_matches_ar_on_flash_next_{p1,seasons}`, adaptive and full-head). The serve_harness battery and chain (greedy, thinking off) give the same text with MTP `auto` and off.
  - Decode speed (Strix Halo, `hipfire bench --matrix --tg 128`, graph on, spec off, 3 ABBA processes per arm, medians): at 8K, 32.59 → 33.26 tok/s (+2.0 %); at 32K, 31.07 → 32.47 tok/s (+4.5 %). The per-arm spread is ≤ 0.05 tok/s.

- **Qwen3.8-Flash-Next (Qwen4) QSA long context: grouped sparse attention, a rows8/rows16 block-scoring route and a radix select from scores (folds the bit-exact QSA series of #774 by @fivetide). Byte-identical; on Strix Halo the Ciru 32K MTP decode goes 59.1 → 60.4 tok/s (+2.2 %), 32K prefill +0.7 % and 256K prefill +3.4 %.** `indexed_attention_attention_f32_batched_hg12` gives one workgroup to each KV group (12 query heads), so every selected K/V row is read once instead of three times; the few-row hg4 kernel keeps 16 value loads in flight per PV step (was 8). Block selection scores the pooled index once per read with `indexed_attention_select_scores_rows8_f32` (≤ 8 rows: decode and verify) or `_rows16_f32` (prefill), then `indexed_attention_select_from_scores` picks the top blocks with a radix select over per-wave LDS histograms.
  - Identity (canonical `qwen3.8-flash-next-gptq3.mq4`, `qwen4_qsa_ctx` final-row logits, chunk 4096): Strix Halo ctx 16384 `7abadb46…` and 65536 `dfd11299…`, R9700 (`auto` placement) ctx 16384 `6b5a824c…` and 65536 `709d5ecf…`, each equal to base. The select and grouped-attention unit tests (now with 37-row cases, three 16-row groups) pass on Strix Halo. serve_harness battery and chain (greedy, thinking off): MTP `auto` and off give the same content, finish and token counts on every turn, and both match base.
  - Speed (Strix Halo, `hipfire bench --prompt-file` with Ciru's prompts, fresh processes, ABBA after a warm-up pair): 32K decode (256 tokens, MTP, τ 1.93 on both) 59.1 / 59.1 / 59.1 → 60.5 / 60.3 / 60.4 tok/s. 32K prefill, the four pairs at full fast-PPT clocks: 1,628.0 / 1,657.5 / 1,652.4 / 1,657.8 → 1,665.8 / 1,669.0 / 1,668.6 / 1,662.3 tok/s (one pair, throttled to gfxclk 2,267 MHz, is left out). 256K, one run per arm: 1,157.8 → 1,197.4 tok/s.

- **Qwen3.8-Flash-Next (Qwen4) on gfx1201: prefill GDN runs the dense C64 chunk scan instead of the token-serial recurrence, on by default (`HIPFIRE_FN_GDN_DENSE_SCAN`; `=0` opts out). Not bit-exact; KLD-gated. pp8192 +5.5 %, Ciru 32K cold TTFT −9.1 %.** Admitted eager prefill steps take this route: exact gfx1201, FN geometry, a Q8 state, rows ≥ 64 in 512-row segments with a [64, 512) tail, and no replay recording or graph capture. They run the dense Qwen3.5 kernels `gdn_chunk_kkt_solve_batched` + `gdn_chunk_scan_bf16_mseg` in place of `gated_delta_step_pipe128_q8`. The new gfx1201 module `fn_gdn_dense` replaces the conv + gate-parameter launch with one producer (`fn_gdn_dense_producer`), which uses the same conv and parameter expressions and adds three steps: FN's serial BF16 Q/K l2 norm, F16 q/k/v packing into the conv output buffer, and the chunk-local log-decay cumsum. The gate consumer reads the scan's BF16 output directly (`fn_gdn_dense_gate_rotate_bf16in`). The Q8 slot serves as the scan's codes and scales in place. EF enters at zero, and the exit state is requantized round-to-nearest-even with no EF. FN keeps no EF, and pipe128 uses position-seeded stochastic rounding. Short tails, decode, MTP verify and every declined call keep their incumbent GDN kernels. `=0` runs the previous code byte for byte.
  - Quality: R9700, canonical `qwen3.8-flash-next-gptq3.mq4`, `qwen4_kld eval` against the FN BF16 teacher refs, 95 % paired bootstrap over chunks.
    - Code 24 × 512: 0.106456 → 0.097057 (−0.0094 [−0.0177, −0.0020], significantly better); PPL 4.7930 → 4.7497.
    - WikiText-2 24 × 2048: 0.084453 → 0.086579 (+0.0021 [−0.0007, +0.0053], not significant; 1.025× the point estimate); PPL 3.7164 → 3.7118.
    - With `=0` the logits sha256 equals beta's.
    - Teacher-forced decode after a dense prefill (`qwen4_kld eval --decode`) tests the exit requantization: a 1,024-row prefill followed by 1,023 `forward_token` steps per chunk. WikiText-2 goes 0.071530 → 0.071502 (−0.00003 [−0.0030, +0.0031]). Code goes 0.081155 → 0.078997 (−0.0022 [−0.0061, +0.0018]). The RNE/no-EF exit does not hurt continuing decode.
  - Speed (R9700 card B, `auto` expert placement, fresh processes, ABBA after a warm-up pair):
    - pp8192, chunk 4096: 3,626.3 / 3,624.4 / 3,633.8 → 3,815.3 / 3,827.4 / 3,828.4 tok/s (+5.5 % median).
    - pp8192 with `HIPFIRE_PREFILL_CHUNK_ROWS=8192`: 4,017.2 / 4,022.3 / 4,025.4 → 4,891.5 / 4,896.9 / 4,911.5 tok/s (+21.7 % median).
    - Ciru 32K cold prefill (`hipfire bench --prompt-file`, 32,012 tokens): TTFT 16,141.9 → 14,679.1 ms; prefill 1,983.2 → 2,180.8 tok/s.
    - rocprofv3, GDN chain per pp8192 step: 538.9 → 130.6 ms. The recurrence goes 432.0 → 42.2 ms. The fused producer costs 46.1 ms against the 44.3 ms plain conv.
    - Decode is unchanged: AR 35.04 → 34.99 tok/s, MTP 39.99 → 40.29 tok/s (τ 1.718 / 1.711), 8 prompts × 128 tokens.
    - Scratch: only the 64-row KKT blocks plus a 1.5 MiB EF plane (26.7 MB at chunk 4096). Peak VRAM rises by 30 MiB at chunk 4096 and 51 MiB at chunk 8192.
  - Behaviour: greedy MTP equals AR with the route on. On 8 decode prompts the committed IDs are identical. The serve_harness battery and chain match 5/5 turns each on content, token counts and finish. The 32K needle (10/50/90 % depth via `hipfire serve`) passes on both arms with identical text. Committed IDs differ from `=0` only on the prompt long enough to take the route (`glimmer_prefill_1024`).
  - Qwen3.8-27B is unchanged: the gfx1201 `eval_hipfire` pins reproduce (WT2 `483cfc58…`, code24 `6339dc9e…`). Kernel idproof against beta: gfx1100/gfx1151 identical, gfx1201 identical plus one added module (`fn_gdn_dense`).

### Dense performance
- **Strix Halo (gfx1151) dense IU4 prefill: three A4 activation producers run bit-identical faster twins, selected by exact gfx1151.** On Qwen3.8-27B (`qwen3.8-27b.mq4-xts`), pp8192, these three producers drop from about 364 to 323 ms per step in the kernel trace (−40 ms):
  - The gated-norm producer before the linear-attention `wo` runs `gated_norm_mq_rotate_{awq_,}i4_gfx1151_v2`: the gfx1100 `_v2` wave-group body plus the gfx1201 one-pass {5,7} `block_i4_128` emit, with the same grid as gfx1100. It goes from 2.38 to 2.01 ms per call.
  - The layer-input RMSNorm producer runs `fused_rmsnorm_mq_rotate_{awq_,}i4_b8_gfx1151`: the gfx1100 `_b8` batched Phase-1a plus the one-pass emit. It goes from 1.20 to 0.93 ms per call.
  - The fold RMSNorm producer (after the o_proj SET) runs `fused_rmsnorm_mq_rotate_{awq_,}i4_fold_b8`. This is a new batched Phase-1a over x/delta pairs (MAX8 + DRAIN), with each element still RN(x + delta) and the same fma chain. It goes from 2.67–2.72 to 2.59–2.64 ms per call.
  - A source opts into the one-pass emit on gfx1151 with `#define HIPFIRE_IU4_ONEPASS`. `HIPFIRE_RMSNORM_P1A_BATCHED=0` restores the incumbent RMSNorm and fold symbols.
  - gfx1100 and gfx1201 dispatch is unchanged. Every pre-existing source that includes the edited `block_i4_128_quant.hip` or `fused_rmsnorm_mq_rotate.hip` preprocesses identically on gfx1100, gfx1151 and gfx1201 (75 sources × 3 arches). The P0 digests of `fused_rmsnorm_mq_rotate{,_awq}` are re-pinned.
  - **Evidence (Strix Halo):**
    - The prompt is `benchmarks/prompts/deepseek4_mq2r_prose_ctx8192.txt`, cycled to 8192 tokens. Branch and beta `f7960b4e8` produce the same 64-bit hashes for all 176 producer outputs (64 RMSNorm, 64 fold including the in-place x, 48 gated-norm). The final logits are byte-identical (sha256 `e7dc5a20…`). This held over 2 processes per arm.
    - Daemon `bench_prefill 8192`, 6 fresh processes per arm, interleaved: ABAB then BABA. The drift-corrected effect is −47.6 ± 10.6 ms on the 1st prefill and −30.8 ± 23.9 ms on the 2nd. Heat made every run slower over the session (+12–20 ms per process).
- **Strix Halo (gfx1151) dense IU4 prefill: a fused A4 epilogue for the FFN gate/up GEMM, on by default (`HIPFIRE_V2B_A4_EPI`; `=0` opts out).** The new builder module `gemm_mq4g256v2_gate_up_silu_a4_iu4_pm_v2b_gfx1151` (`hipfire-isa` `iu4_v2b_a4`, natively emitted as its own code object; the certified V2B bundle is unchanged) retiles the V2B gate/up + SiLU GEMM to M512 × N128, so that one CTA holds whole 256-feature groups. Its epilogue then runs the down projection's input producer: the AWQ divide, signs1, the FWHT, ×0.0625·signs2 and the one-pass {5,7} `block_i4_128` emit. The 72-byte records go into a new dedicated down-sidecar scratch slot (`Int4MmqDownPrepared`), so `h` is never stored and `fused_silu_mul_mq_rotate_awq_i4_hin` does not run.
  - It is admitted on exact gfx1151 V2B shapes only (`M % 256 == 0`, `N % 128 == 0`, eager; no Redline recording or graph capture). `HIPFIRE_V2B_PM=0` and `HIPFIRE_F1LITE=0` also turn it off. Since the V2B tile needs a prefill chunk that is a multiple of 256 tokens, a prompt engages the fusion only on such chunks: `qwen38_issue693_longcode_20676.txt` runs it on its two full 8192-row chunks, and the 8K prose prompt `deepseek4_mq2r_prose_ctx8192.txt` does not run it at all (its single chunk takes the X5 tile).
  - `HIPFIRE_V2B_A4_EPI=retile` runs only the stage-1 M512 twin of the V2B SiLU entry, with the same `h` bits. `HIPFIRE_V2B_A4_PM_BUNDLE=<path>` loads the module from a file.
  - **Evidence (Strix Halo):**
    - The A4 records are md5-identical to the incumbent gate/up + hin at 17408 × 5120 × 8192, on random inputs at activation scales 1, 1e-3 and 1e3 and on the crafted edge corpus. Four other shapes were also checked. With the flag off and with `retile`, the `h` and A4 md5s match each other.
    - Per call, with the fused kernel against gate/up + hin over 3 fresh processes (ABBA, 20 event-timed launches per block): −1.231, −1.225 and −1.075 ms. The kernel trace inside the model gives −0.44 ms per call (41.31 against 38.76 + 2.99 ms).
    - On Qwen3.8-27B (`qwen3.8-27b.mq4-xts`), pp8192 with token ids 0..8191, the final logits are byte-identical with the flag on, with it off, and on beta `798d63b3a` (sha256 `c7089559…`).
    - Daemon `bench_prefill 8192`, 3 fresh processes per arm, ABBA: −48.6 to −65.8 ms on the 1st prefill and −18.1 to −101.4 ms on the 2nd.
    - **Default flip** (branch `perf/a4-epi-default-gfx1151` against `HIPFIRE_V2B_A4_EPI=0` on the same binary, Qwen3.8-27B `qwen3.8-27b.mq4-xts`):
      - pp8192 final logits (token ids 0..8191): md5 `e0e3ba1a…` in 5 fresh processes per arm, interleaved. 56 back-to-back pp8192 prefills at the default, in 6 processes of 8–10 prefills each, all give the same md5. pp32768 is identical in both arms (md5 `9612f2d0…`).
      - `scripts/serve_harness.py` greedy `battery`, `chain` and two long single-prompt rows (`deepseek4_mq2r_prose_ctx8192.txt` at 8,182 tokens and `qwen38_issue693_longcode_20676.txt` at 21,590 tokens): the committed token ids (`HIPFIRE_EMIT_TOKEN_IDS=1`) and the decoded reasoning and content are identical per case, and the text is coherent.
      - A daemon generate with `HIPFIRE_GRAPH=1` on `qwen38_issue693_longcode_20676.txt` under `rocprofv3 --kernel-trace`: the default arm launches the fused kernel 128 times in place of 128 V2B SiLU and 128 hin launches, all before the first decode step. The 29,152 decode-phase launches (HIP graph) and the 32 greedy tokens are identical in both arms.
      - `hipfire bench --matrix --pp 8192 --backend noslots --workload stateless --json --prompt-file benchmarks/prompts/deepseek4_mq2r_prose_ctx8192.txt` (md5 `b79d5378…`), 3 fresh processes per arm, interleaved ABBAAB against beta `798d63b3a`: the pp8192 median of the per-process medians goes from 1,178.6 to 1,185.5 tok/s (+0.6 %, inside the run-to-run spread of about 12 tok/s). The standard bench on the same prompt file (8,142 tokens, which never reaches the fusion) gives 780.3 tok/s on beta and 793.7 on the branch.
      - Kernel objects are identical to `860a454bc` on gfx1100 (132 modules) and gfx1201 (137 modules), checked with `scripts/kernel_idproof.py`.

### Fixes
- **Flash-Next (Qwen4) QSA: the select-from-scores kernel counts blocks tied at the selection threshold without overflowing at long context.** `indexed_attention_select_from_scores` scanned its per-thread "above" and "equal" counts as one packed `above + (equal << 16)` int. Past 32,767 tied blocks the equal field went negative. At up to 65,536 blocks (262K context at compress 4) the effect was out-of-bounds writes to negative LDS slots (undefined behavior), not a wrong selection. A wrong selection needs more than 65,535 ties: the field then wraps and later ties overwrite the first selection slots. The two counts now scan separately (32 more bytes of LDS); every input that did not overflow gives the same bytes. No other select kernel packs its counts.
  - New unit test `select_from_scores_survives_tens_of_thousands_of_ties` checks synthetic rows (all tied and partially tied, at 40,000, 65,536 and 70,000 blocks) against a CPU reference, with a guard canary after the selection. On the beta kernel the 70,000-block row fails at slot 0, on both Strix Halo and the R9700. With the fix, the new test and the existing QSA select tests pass on both. On Strix Halo with the canonical GPTQ3 artifact, `qwen4_qsa_ctx` final-row logits are unchanged at ctx 16384 (`7abadb46…`) and 65536 (`dfd11299…`). Ciru 32K cold prefill, 3 ABBA pairs: 1,660.3 / 1,653.9 / 1,664.6 → 1,661.3 / 1,372.1 / 1,662.8 tok/s. The low B2 run ran at gfxclk 2,260 MHz against about 2,580 MHz for the others. `kernel_idproof` changes only the `tensor_ops` object on gfx1100, gfx1151 and gfx1201.
- **gfx1151: the `gdn_chunk_scan_gfx1151` twin keeps its output WMMAs at full EXEC (#797 by @fivetide).** On fivetide's toolchain clang sank the final O WMMAs into the `tok < rows` store branch, so on a Qwen3.5-family prefill whose GDN chunk-scan segment ended mid 16-row block the valid rows of that block read undefined operands (off by ~5e3 or NaN; a 291-token prompt on `qwen3.8-27b.mq4-xt` decoded "MENTS"). An empty `asm volatile` pin on O keeps the WMMAs before the branch. New GPU test `crates/rdna-compute/tests/gdn_chunk_scan_tail.rs` (ignored) checks partial-chunk rows against the same rows of a full-chunk run.
  - On the release toolchain (HIP 7.15) the rebuilt `gdn_chunk_scan_gfx1151.hsaco` is byte-identical to beta's (`kernel_idproof`: only its source hash changes; every other gfx1100/gfx1151/gfx1201 object is identical), so the pin is a guard against the sinking, not a change to shipped code.
- **gfx11 Q8 FA2: the wide-envelope guard only demands pre-grown Q16 scratch for batches that only the wide envelope admits (#802 by @fivetide).** It rejected any Q16 scratch growth under capture or Redline recording, including batches the narrow envelope (512, 1024 on gfx1151) has always grown on demand, so a cold 27B Redline prefill capture failed with wide auto-on on gfx1151.
- **Correction: gfx1151 FA2 wide auto is now on, following the gfx1151 twin admission.** This supersedes the v0.4.0 gfx1151 default-off; the enablement is the deliberate commit `ae9c5cf9d12a5a63ae60630637d22d00a4fc964e`. Evidence is the existing v0.4.0 entry for the gfx1151 twin of the gfx11 Q8 FA2 kernel (recorded by `79b038956e`): Halo H2 pp8192 +2.13 % / +2.11 % (ABBA / BAAB), outputs bit-identical, and both Halo KLD `.kldseq` pins unchanged.
- **Qwen3.5/3.6/3.8 VL: the mrope forward entries grow the VMM KV mapped prefix like their non-mrope twins (#803 by @fivetide).** `forward_scratch_mrope` and `forward_scratch_embed_mrope` wrote KV without `ensure_mapped_capacity`, so a 27B image request (1900 vision tokens) decoded past the load-time mapped prefix and faulted with "Page not present".
- **Flash-Next (Qwen4) MTP windows stop at `<|im_end|>` instead of the config EOS (from #774 by @fivetide).** An accepted end-of-turn stays pending for the terminal flush instead of committing the draft tail past it. Committed tokens are unchanged.
- **Flash-Next (Qwen4): a `max_seq` above the model's `max_position_embeddings` is refused at load, before any allocation (from #774 by @fivetide).**
- **Flash-Next (Qwen4): an omitted `max_tokens` is fitted to the context left after the prompt on the AR route, and `hipfire run` without `-n` sends `max_tokens_fit` like serve (from #774 by @fivetide).** An explicit `-n` that does not fit is still refused.
- **Serve: a turn the model ends inside `<think>` finishes with `finish_reason: "stop"` and its reasoning, instead of an "open think span" stream error.** Flash-Next and Qwen3.6/3.8 sometimes emit end-of-turn while still reasoning. HermesAgent HA-13 (Flash-Next GPTQ3, xhigh, R9700) hit this on all 3 attempts at its first turn, ~1.5–2K reasoning tokens into a 16K budget, and scored 0. The daemon now ends such a turn the way vLLM does. The reasoning streamed so far is the `reasoning_content`, `content` is empty (or holds the answer prose that came before a generated `<think>`), no tool call is released, and there is no error event. The turn is not cached, and a multi-slot session is closed instead of kept for reuse, because the KV holds an unclosed think span that no rendered history reproduces. The state is not rolled back. Running out of `max_tokens` there is still `length`, and a grammar violation still fails closed. This covers Qwen AR (single GPU including Flash-Next, dense TP, MoE EP, continuous-batch lanes), the DFlash/MTP spec path, which also dropped the reasoning bytes held back at the end of the turn, and multi-slot.
  - CPU tests replay HA-13's reasoning tail and `<|im_end|>` through the AR producer and the spec emitter into the daemon wire (one `done` with `stop`, every reasoning byte, no `error`), and through the gateway streamed and non-streamed.
- **Serve: Flash-Next (Qwen4) now receives its history `reasoning_content`, as Qwen3.5/3.6 already did.** Serve forwarded history reasoning only for the arch labels in its table, and `qwen4` was missing from it. As a result, every multi-turn thinking conversation on Flash-Next (HermesAgent, `preserve_thinking: true`) ran without the model's own earlier reasoning: the Qwen3.8 template rendered each prior assistant turn with an empty `<think>\n\n</think>` block. Each new prompt also stopped extending the previous one, because the empty block's `\n\n` token (271) replaces the `\n` (198) that ended the previous prompt after the open `<think>`. A cache that resumes at the previous prompt's end therefore never hit. A CPU replay sends the captured HermesAgent requests through the serve projection and the Qwen4 render; it reproduces the served `prompt_tokens` on all 518 of them. Before the fix, 0 of 394 continuations extended the previous prompt; now all 394 do. Non-thinking turns (TC) render the same tokens as before. The arch table now lives in `saddle_core::caps::arch_replays_history_reasoning`, and a loader test requires every carrier with a history-replaying reasoning contract (Qwen Jinja, Muse Glimmer) to be in it. Maple (arch 15) is not covered: it reports the fallback `qwen3` label, which it shares with Qwen3 dense.
- **Flash-Next (Qwen4): an explicit `max_think_tokens` budget closes the reasoning and the model answers, instead of failing the request with "think token budget exceeded".** On the AR route (`qwen4_ar`, MTP off or temperature > 0) the budget failed the request closed. Both Qwen4 routes now splice the template's own close, `\n</think>\n\n` (Flash-Next tokens 198, 248069, 271), through the normal decode path once the budget is spent inside `<think>`, and the model answers with the rest of `max_tokens`. vLLM's `thinking_token_budget` and Qwen's budget recipe work the same way. AR and MTP share one budget rule: every generated token that leaves the stream inside a think span counts, including the first one; a newly opened span starts the count over; the spliced close does not count; and if the model re-opens `<think>` and spends the budget again, the turn ends as a reasoning-only `stop`. A close that `max_tokens` cuts short ends with `length`. `reasoning_effort` is still only a template sentence: no budget is derived from it, and without `max_think_tokens` the output is unchanged. The Qwen3.5 DFlash/MTP emitter now counts the first generated token too, so its budget closes one token earlier than before; it still splices `HIPFIRE_THINK_CONTINUATION` (default `</think>\n\n`). A turn the model ends inside reasoning now logs its last token id (`[open-think] … terminal token`), so captures show whether it was `<|im_end|>` or `<|endoftext|>`.
  - Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, greedy serve, MTP off and on, 6 requests each, with budgets 32 (non-streamed and streamed), 16, 2 (`reasoning_effort: low`) and 64, plus one request with no budget. No request errored and every one finished `stop`. Each budgeted request logged the splice `[198, 248069, 271]` on both routes (`qwen4_ar` and MTP, `drafter=mtp` on all 6 MTP requests), and the answer is coherent (for example `391` for 17 × 23, and a complete Fibonacci function). Reasoning, content, `finish_reason` and `completion_tokens` are identical with MTP on and off for all 6 requests.
  - CPU tests drive the AR budget step and producer and the Qwen4 MTP emitter over a scripted greedy model. They check the exact close tokens after the Nth think token followed by the answer, with both routes identical; a budget of 1; an unset or unreached budget giving byte-identical output; the re-open stop; and a close clipped by `max_tokens`.
- **Flash-Next (Qwen4) greedy MTP emits AR's exact tokens again with Q8 GDN state on the batched verify route.** The persistent GDN verify/rollback kernel dequantized the Q8 recurrent state once, carried F32 across every verify row and quantized once per window, while AR decode quantizes after every token; batched verify and rollback now apply the same per-token Q8 boundary at each row's absolute position (row-capture path only, gfx11+ SIMT; F32 state and unarmed/chunked prefill numerics are unchanged). The earlier claim that the few-row verify is bitwise the single-row route was false for this Q8 path. No performance claim: the extra per-row boundaries may cost verify throughput.
  - Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, prompt p1 (md5 `c91f8380…`, 1047 rendered tokens), 1200 greedy token IDs per arm, in process (`crates/hipfire-arch-qwen4/tests/greedy_mtp_identity_hw.rs`, each arm a fresh process): on beta `ac413be01`, forced batched and full-head batched first differed from AR at generated index 112 (4752 vs 3050) and adaptive at 130 (49869 vs 3309), while forced interleaved matched. With the repair, forced batched, forced interleaved, adaptive and full-head batched all match AR's 1200 IDs in 3 of 3 fresh runs; AR's IDs are unchanged. The `gdn_q8_verify_matches_single_row` seam test compares capture, final slot and every rollback keep-prefix (direct and graph replay) against sequential single-row Q8 decode.
- **Flash-Next (Qwen4) greedy MTP batched verify: the hyper-connection (HC) BF16 projections for 2–8 rows now sum in exactly the single-row decode order.** For one row, the HC write-gate projection and the HC read down projection run `gemv_bf16_xf32_k4`. For 2–8 rows they ran the multirow GEMM or the fused `hyper_norm_gate`, in a different order. The F32 difference of up to 1 ulp survives BF16 rounding only near a rounding boundary, so batched-verify MTP diverged from AR on some prompts. New `gemv_bf16_xf32_k4_rows_r{2..8}` kernels load the weights once and keep each row's order bitwise equal to the single-row kernel. They are routed on gfx1100, gfx1151 and gfx1201. Decode and >8-row prefill are unchanged.
  - Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`: the serve_harness battery (5 prompts) is byte-identical with MTP on and off in 3 of 3 fresh runs (before: the seasons prompt diverged at about token 55, already on `ac413be01`). In process, `greedy_mtp_identity_hw` (p1, 1200 IDs, plus the seasons prompt) matches AR on all four MTP arms in 3 of 3 runs. The rows kernels are bitwise equal to the single-row kernel on gfx1151, gfx1201 and gfx1100. Batched MTP decode on `lru_cache_pep8_strict.txt` is unchanged at a median of 66.3 tok/s, tau 2.5.
- **Qwen multi-turn: greedy MTP and AR now build byte-identical prompts from the conversation history.** Qwen4 AR re-encoded earlier assistant turns from text, while the MTP path spliced the emitted token IDs. AR now splices the emitted IDs too. Separately, a turn that ended with EOS inside an MTP window it could not repair cleared the whole assistant-turn cache, so later MTP requests re-encoded every earlier turn from text. The cache is keyed by emitted content, not KV state, so an unrepaired terminal now drops only the KV mirrors (conversation LCP and checkpoints). One shared `qwen_cached_turn_entry` now builds cache entries for the DFlash, AR and `generate()` stores.
  - Strix Halo, Flash-Next GPTQ3, serve_harness chain (5 turns, greedy): every turn is byte-identical with MTP on and off in 3 of 3 fresh runs, and identical from run to run. Before, turn 2 (and then turn 3) diverged because 36 of the prompt IDs in turn 0's code answer differed. A CPU test builds a 5-turn chat through both modes' real cache paths and checks that every earlier turn hits and the prompts are identical.
- **`hipfire bench --spec` now measures speculation instead of silently measuring AR.** Bench generates hard-coded `repeat_penalty 1.1`, and any non-neutral penalty routes the daemon to the plain AR decoder, so `--spec mtp|dflash|ngram|dspark|auto` reported AR numbers. Every bench generate now sends the neutral `1.0`, so `--spec off` and speculative runs measure the same workload. The JSON gains `spec_requested`, a per-run `spec_route` (`null` when AR ran) and `spec_tau`, and the bench warns when `--spec` is not `off` but no run speculated.
- **`Gpu::silu_f32` launches on the active stream instead of the legacy null stream.** The null-stream launch was unordered against the non-blocking active stream (silu could read its input before the producer finished) and escaped hipGraph capture entirely. On the HyperRead path (`layer_ops.rs` low-rank SiLU) it is reached only by arches without gfx11+ SIMT (gfx906/908/90a, CDNA3 gfx94x, RDNA1/2); gfx11xx/gfx12xx take the fused activation and never call it. Host-side only: no kernel object changes. The diffusion VAE/FLUX callers are also now stream-ordered.
- **Flash-Next (Qwen4) model switches are unload-first, like every other single-device load, which fixes the swap-loop refusal.** Previously the daemon kept the prior model resident until the new Qwen4 model was built. When the prior was a dense Qwen3.5/3.8 model with VMM KV arenas, the clean-VMM check before construction always refused: Flash-Next → 27B → Flash-Next failed until an explicit `unload`. Load requests are still admitted read-only first, so a refused admission keeps the prior model and its drafter. Once admitted, the prior model is retired and VMM must be clean before construction. A later failure therefore leaves no model loaded, and the load error now ends with `no model loaded (the prior model was unloaded before this load)`. Only tp>1 expert-parallel loads still build the new model before retiring the old one. The Qwen4 carrier this builds on traces to fivetide's #774.
  - Strix Halo, `hipfire serve`, canonical `qwen3.8-flash-next-gptq3.mq4` ↔ `qwen3.8-27b.mq4-xts` (Q8 VMM KV): Flash-Next → 27B ×3 then Flash-Next again. All 7 loads succeeded with no refusal. Greedy text was byte-identical for each model across cycles. VRAM in use returned to 75.0 GB with Flash-Next and 16.2 GB with 27B each cycle.
  - Failure boundaries on Halo. A rejected admission kept the prior model and its drafter: Flash-Next with MTP (identical text, tau 1.42 before and after) and 27B with DFlash (identical text, tau 3.20). After Flash-Next, an admitted load that failed (mtp=on, no head) returned the new error, the next generate answered `no model loaded`, and a following load recovered. Kernel idproof shows no object changes on gfx1100, gfx1151 or gfx1201.
- **Flash-Next (Qwen4) on gfx1151: QSA context state is demand-mapped VMM by default, so committed memory tracks the tokens a request touches instead of `max_seq`.** Load admission now resolves `kv_backend` for Qwen4: automatic selects `vmm` exactly on gfx1151 with a certified platform and HIP VMM runtime (elsewhere `legacy`, with the reason logged). Explicit `--kv-backend legacy` is honoured, and explicit `vmm` is refused only where unsupported. The target and native-MTP QSA arenas and the gathered-attention workspace reserve address space for `max_seq`; each forward, MTP step and prefix restore maps the rows it touches first, outside any capture or record. On unified memory the `auto` expert reserve now charges committed bytes, not the virtual extent (a discrete card keeps charging legacy storage, see the gfx1201 entry). It also charges the MTP row capture in the state's real GDN format (Q8 811,008 B per slot) instead of F32. New load and growth log lines: `qwen4 QSA context: …` and `qwen4 QSA context mapped for N tokens: …`. Replant slice S3; credit to fivetide for the PR774 Flash-Next ancestry.
  - Strix Halo, canonical `qwen3.8-flash-next-gptq3.mq4`, vmm vs `--kv-backend legacy` in separate processes: pp8192 final logits are byte-identical. 4200 greedy tokens after a 1000-token prompt (crossing 1024, 2048 and 4096) give identical ids. Prefix reuse at the 8192 checkpoint and rewind match cold prefill under both backends. Daemon greedy text, AR and native MTP, on 1000- and 3600-token prompts is byte-identical across backends and to the base `36ec162af` binary.
  - Memory: at `max_seq` 262144 the load commits 0 MiB of QSA context (15,392 MiB virtual, target plus MTP). A short prompt then commits 96 MiB, against 888 MiB committed at load by legacy storage at only 16384 tokens. `hipfire serve` Flash-Next ↔ dense 27B ×3 runs 7 loads with 0 refusals and byte-identical greedy text per model, so the VMM owner census reaches 0 at every switch. The serve_harness battery and chain (reasoning off) finish 5/5 turns each with no runaway, empty answer or attractor.
- **Flash-Next (Qwen4) on gfx1201: QSA context state and the gathered-attention workspace are demand-mapped VMM by default too.** Automatic `kv_backend` now selects `vmm` on exact gfx1201 as on gfx1151 (certified platform and HIP VMM runtime required), so an R9700 load no longer prints `WARNING HIPFIRE_KV_BACKEND=legacy: automatic VMM selection unavailable (Qwen4 VMM QSA state is certified only on gfx1151, not gfx1201)` and commits QSA pages as requests touch them instead of all of `max_seq` at load. `memory.kv_backend = "legacy"` / `--kv-backend legacy` remains the opt-out. Output is byte-identical to legacy storage. Expert residency does not change on a discrete card at any `max_seq`: the `auto` reserve there keeps charging legacy storage of all `max_seq` tokens (`Qwen4ContextCommit::for_expert_reserve`) whichever backend is selected, because VMM pages mapped later by longer prompts come out of the VRAM the placement left free. At 65,536 and 262,144 tokens the reserve, and so the placed expert layers, equal legacy storage's. Unified memory (gfx1151) still charges the committed pages. At the default `max_seq` the `auto` reserve already covers 32,768 legacy tokens.
  - R9700 (card B), canonical `qwen3.8-flash-next-gptq3.mq4`, one binary with VMM vs `memory.kv_backend = "legacy"`. `qwen4_qsa_ctx` final logits at 16K (`backend=vmm`) are `37609204…`, beta's gfx1201 baseline, and `6b5a824c…` with `HIPFIRE_FN_GDN_DENSE_SCAN=0`. At 64K with `HIPFIRE_FN_GDN_DENSE_SCAN=0`, VMM matches the legacy beta `0614d9e2a8` binary (`709d5ecf…`). 4,600 greedy tokens of AR decode on the retained Redline PM4 tape and of native MTP have identical ids under both backends, and VMM grows the mapping from 4,064 to 8,128 tokens mid-decode. Greedy MTP equals AR for 8 prompts × 128 tokens under both backends.
  - Memory at load (`max_seq` 32768): VRAM in use is 28,695 vs 29,459 MiB with native MTP (−764 MiB) and 29,796 vs 30,412 MiB without (−616 MiB). The load commits 0 MiB of QSA context, against 507 MiB target plus 148 MiB MTP for legacy storage. Expert layers are unchanged: 0..14 with MTP, 0..15 without. The Ciru 32,012-token prompt plus 512 tokens at the default `max_seq` with MTP peaks at 31,647 MiB without OOM. ABBA (same binary, legacy vs VMM): AR decode 34.95 → 35.10 tok/s, MTP decode 40.84 → 40.86, pp8192 3,822 → 3,826. All flat.
  - serve_harness battery and chain (greedy, reasoning off, VMM): 5/5 turns each with no runaway, empty answer or attractor; MTP on and off give the same content md5 and `gen` for every turn.
  - New GPU tests on gfx1201: `vmm_context_maps_every_served_qsa_format` covers fp8 (the gfx1201 default), F32 and q8 QSA arenas. `retained_pm4_tape_replays_over_grown_vmm_qsa_arena` replays a PM4 tape recorded over one mapped granule after in-place growth and matches HIP byte for byte. The VMM granularity is 2 MiB. Kernel idproof is identical on gfx1100, gfx1151 and gfx1201. The CPU test `discrete_reserve_matches_legacy_and_unified_memory_charges_committed` pins that a discrete VMM load's context commit, `auto` reserve (with the MTP and gather parts) and placed expert layers equal legacy's at `max_seq` 65,536 and 262,144 (fp8 and F32), and that unified memory still charges the first chunk's pages.
- **gfx1201: row- and token-scaled launches no longer put more than 65536 workgroups on `grid.y`/`grid.z`.** On an R9700, a module launch with 65537 or more workgroups on either axis runs every workgroup but wraps `blockIdx` modulo 65536, so rows past 65536 silently received another row's result; 65536 itself is addressed correctly. Launchers that could cross it now fold the row axis into `grid.x` or launch pieces of at most 65536 row tiles with their pointers advanced: Qwen3.5/Qwen4 batched gates, norms and rotations, the scratch quantizers, batched GEMV/MoE families, KV writes and transcodes, VAE im2col/transposes, and the 16-row F16 WMMA and 8-row HFQ4-G128 GEMMs. Any launch that still exceeds the limit on gfx1201 now fails with an error naming the axis instead of running.
  - R9700: the 65536-tile pieces match single small launches byte for byte at 1,048,616 rows (F16 WMMA) and 524,328 rows (HFQ4-G128). Nine byte oracles show the rewritten kernels match the previous ones at up to 131,075 rows, and the two MoE large-row GPU tests pass. Kernel idproof changes only objects whose source the fix edited (13 on gfx1100, 15 on gfx1151, 12 on gfx1201). H2 greedy is 33/33 byte-identical by default and with `HIPFIRE_GRAPH=1`.
- **gfx1100/gfx1151 A3B decode: the 16Q/2K FA prep twins now explicitly use the unfused RoPE chain's FMA operand order.** The second rotary output no longer contracts as `fma(x1, cos, x0 * sin)` instead of `fma(x0, sin, x1 * cos)`. The gfx1201 16Q/2K arithmetic is unchanged; the 24Q/4K twins retain their explicit reference contraction.
  - XTX and Strix Halo: the fused Q/gate/K planes match the unfused chain bit-for-bit at 10 rotary positions up to 2²⁰. With Ornith-1.5-35B-A3B MQ4, lowered and hand decode match all state/logit digests and 32 greedy steps at 13 prompt lengths (1–4097 tokens). Before the fix, every length differed on both arches.
  - Ornith A3B prefill, 24 × 512 tokens: before/after KLD sequences are byte-identical on both arches (ΔKLD = 0). This is against each base binary's own top-256 logits, not a BF16 quality reference; the pinned MI300X A3B artifact was unavailable.
  - JIT-cache object proof changes exactly `qwen35_fa_prep_gfx1100` and `qwen35_fa_prep_gfx1151` (68/66 other exercised objects identical). Installer idproof preserves every shipped code object on gfx1100/gfx1151/gfx1201; the shared 24Q/4K source builder changes only its gfx1100/gfx1201 source-hash/index receipts.
  - Both arches answer the two pinned A3B code prompts coherently with greedy, thinking-off, speculation-off serving: complete merge-sort and running-balance implementations, neither empty nor runaway.
  - H2 protected greedy gate remains 33/33 byte-identical (AR, MTP and DFlash, 11 responses each, gfx1201 card A).
- **Gemma4 lowered: full-attention KV follows `kv_cache`.** `auto`/`q8` use Q8 on every arch; `legacy-asym3` restores the previous Givens Asym3 K + Q8 V tier and is accepted by registry generation. Sliding attention stays Q8. Unsupported `fp8`/`bf16` fail loading; other unsupported names warn and fall back to Q8. The gfx1100 decode partials cover the smaller Q8 tile. This changes greedy text at the default; the explicit legacy mode retains the old cache.
- **Gemma4 lowered decode: hipGraph capture/replay is wired and is the default on exact gfx1201 only; every other arch is opt-in.** With both switches unset, bodies the capture admits capture after an eager warm-up and update the device position before each replay on gfx1201; gfx1100, gfx1151, gfx1200 and every other arch stay eager unless opted in. Precedence: `HIPFIRE_GEMMA4_GRAPH=0/1` wins, then `HIPFIRE_GRAPH=0/1`, then the arch default; `=0` is the kill switch. Prompt prefill stays eager. KV, scratch, weight and shape changes invalidate the capture; CPU expert fallback and atomic Q8 expert-down bodies stay eager. Graph-off, graph-on and two concurrent graph-on processes gave identical raw hashes, raw boundary bytes and 160 greedy tokens, and the H2 greedy gate with a fresh DFlash completed 33/33 across AR, MTP and DFlash. Gemma4-26B decode on gfx1201 measured 78.47 → 81.51 tok/s (+3.9 %), medians of 3 fresh processes per arm.
- **llama HFQ: `kv_cache=auto` no longer emits a false unsupported-mode warning.** It still resolves to Q8; MiniMax and LFM2-MoE use the same non-Qwen alias table.
- **Qwen3.8-Flash-Next (Qwen4): an unset `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` no longer runs out of VRAM on a discrete card.** Unset used to mean every routed expert (~64 GB) in VRAM, so `hipfire run`, `serve` and `bench` failed `hipMalloc` at layer 17 on a 32 GB card. Unset on a discrete GPU now keeps the experts resident only when all of them, the non-expert weights and the `auto` reserve fit in free VRAM, and is `auto` otherwise. The default changes only where the load used to fail: unified memory (Strix Halo) and any card that fits stay fully resident, and an explicit `N` or `auto` always wins. The load logs the choice on one `qwen4 expert placement` line.
  - **The host-memory switches (`HSA_USERPTR_FOR_PAGED_MEM=0`, `GPU_PINNED_MIN_XFER_SIZE=100000`) now apply unset too.** They used to follow only the variable being set, because the HIP runtime reads them once, before any model loads. So an unset load that resolved to `auto` got pageable userptr host memory: a no-env `hipfire run` of Flash-Next on a 32 GB gfx1201 under host-memory pressure spun 60 min in `hipMemcpy` (`upload_pooled_bytes`). The native CLI now names the model it starts the daemon for in the `configure` message (`hipfire run`, `bench`, and `serve`'s pre-warm and restart model), and a daemon started for a Qwen4 HFQ (arch 16) whose cards are all discrete-memory arches sets both switches before its HIP runtime loads. The rule rests on the arch and the cards, not on the placement, which is only resolved at load from free VRAM. A fully resident Qwen4 load on a card that fits therefore stages its uploads too. Other models, unified memory (Strix Halo) and an unrecognized arch keep ROCm's defaults, and a switch the operator set is left alone. `[hip-bridge] HSA_USERPTR_FOR_PAGED_MEM=0 GPU_PINNED_MIN_XFER_SIZE=100000 (…): host memory out of reclaim` logs the switches when hipfire sets them. A Qwen4 load in a daemon started for another model (serve switching models) cannot get them any more, and its log says so.
- **Redline: a retained tape that cannot reproduce its forward is no longer installed. Qwen4 with `HIPFIRE_REPLAY_BACKEND=redline` now matches eager.** On gfx1151 this is not a default route: `retained_redline_default` admits Qwen4 there only for `.mq4r` file names. On gfx1201 an MTP-off Flash-Next load now takes the retained PM4 default (see the default flips above); linear AQL is never a default. With the explicit backend and `HIPFIRE_REPLAY_TRANSPORT` unset (linear AQL), greedy decode degenerated from the first prompt on, with text like "The `ChatHistory is a persistent to the `ChatHistory…". This happened on gfx1201 and gfx1151 alike. There were two defects:
  - **Linear AQL submitted recorded kernargs verbatim and patched only the Qwen3.5 GDN frame.** Every position-derived field that a Qwen4 launch declares, such as the QSA block counts and starts and the GDN convolution cursor, therefore replayed at its capture-time value. `prepare_linear_aql_prefix` now refuses a tape with any declared or synthesized position binding, or a position-bound grid, and the route falls back to HIP. PM4 applies every binding and is unchanged.
  - **The Qwen4 capture census reported a memset inside the window as "effect-incomplete" and installed the tape anyway.** Every HIP memcpy and memset variant, synchronous or asynchronous, now counts in a per-thread `hip_bridge::memory_effects` tally that is never reset. `ReplayController` refuses to prepare an automatic capture whose window issued a device copy, readback or memset, so the Qwen3.5, Qwen4, DeepSeek4 and LFM2-MoE adapters fall back to HIP. Host→device input staging is still allowed. Qwen4's one memset is the one-time zeroing of the fused GDN rotation's head-pair counters, which each launch resets. It was harmless: base PM4 with it inside the window matched eager. It now runs before the window opens, so Qwen4's PM4 route still installs.
  - Evidence: a sequence of 705-, 16,344- and 32,745-token prompts, 48 greedy tokens each, on card B (gfx1201, N=12) and Strix Halo (gfx1151). With the fix, the window holds 0 memsets on both. The AQL arm falls back to HIP and matches eager on 3 of 3 prompts. The PM4 arm replays 47+48+48 tokens and also matches eager on 3 of 3. Kernel objects are unchanged on gfx1100, gfx1151 and gfx1201.
- **Qwen3.5 AR decode graph: a replay needs the buffers it was captured with.** `forward_scratch` replayed its captured hipGraph whenever one existed, and the graph bakes the device pointers of the capture call. So a caller that passed a new `DeltaNetState` (or KV cache, scratch or model) read and advanced the previous caller's state, and left its own untouched. The capture now records a pointer fingerprint of the model, every KV and DeltaNet tensor and every scratch buffer (`GraphState::ar_forward_binding`); a call with a different fingerprint re-captures, and every AR capture clears the fingerprint. The daemon keeps one state set per model and resets it in place, so its output is unchanged. This was found by the H2 prefill durability harness, which allocated a state per case. No kernel changes.
- **gfx1100 Qwen3.6/3.8-27B decode: the lowered decode (default) and the hand decode (`HIPFIRE_FORWARD_LOWERED=0`) produce the same bytes again.** On the XTX they differed at every prompt length, starting at the first full-attention layer. The fused 24Q/4K FA prep (`qwen36_27b_fa_prep_gfx1100`) let the compiler contract the second RoPE output as `fma(x1, cos, x0 * sin)`, while the unfused `rope_partial_halfsplit_f32` it replaces computes `fma(x0, sin, x1 * cos)`. The gfx1100 kernel now spells out the reference order with `__builtin_fmaf`, as its gfx1201 twin already did; both twins share one source builder. Only that gfx1100 code object changes; gfx1151 and gfx1201 objects are byte-identical. New GPU test `fa_prep_parity` compares the fusion with the unfused chain bit for bit.
- **gfx1100 and gfx1151 Qwen3.5-family decode: the fused DeltaNet conv + Q/K L2-norm (`conv1d_silu_split_qknorm*`) matches the unfused `conv1d_silu_split_f32` + `fused_qk_l2_norm_scale_f32` pair bit for bit.** It takes the explicit `__builtin_fmaf` / contract-off form gfx1201 already used for the conv tap sum and the first square-sum add. gfx1201 code objects are unchanged. `conv_qknorm_parity` now also runs on gfx1100 (including the scalar-prep variant) and gfx1151.
- **Scratch growth can no longer leave a captured hipGraph or PM4 tape pointing at freed memory through `qwen4_f16_x_scratch` or MMQ screening.** `Gpu::qwen4_f16_x_scratch` (the shared FP16 X scratch used by Qwen4 F16 WMMA prefill, the MQ F16 rotations and the QSA dense route) now invalidates captured state before it grows the buffer, as `ensure_fp16_x` does. `mmq_screen_weight` no longer sets `capture_mode = true` around its GEMMs. That toggle had suppressed the invalidation of the screen's own Q8_1/FP16 scratch growth, pushed stray kernarg blobs, and stayed stuck on if a GEMM returned an error. Inside a capture or recording, screening is skipped (the weight reports MMQ-unsafe, uncached). The Lloyd MMQ pad-to-128 path returns an error under capture/record instead of capturing pool temps it frees on return, and `unpad_prefill_ys` frees its temps when a copy fails. New GPU tests: `scratch_growth_capture`, `mmq_screen_capture_state`.
- **Kernels: every LDS-only workgroup barrier in the GDN kernels now waits for the wave's own LDS stores explicitly before it signals (`s_wait_dscnt 0` on gfx12, `s_waitcnt lgkmcnt(0)` with vmcnt/expcnt open on gfx11).** Covered: the H2 GDN chunk scan (gfx1201 `gdn_chunk_scan*`, gfx1100 `gdn_chunk_scan`, gfx1151 `gdn_chunk_scan_gfx1151`) and `gdn_lds_barrier` in `tensor_ops.hip`, which the Flash-Next persistent GDN step (every MTP/verify step and few-row batch on gfx1100/gfx1151/gfx1201) uses at 7 cross-wave handoffs, one carried across rows. The release fence with the "local" MMRA only adds a soft wait, which the compiler may drop as redundant without seeing stores carried around a loop (llvm/llvm-project#220357, the gfx1151 VerifyAttn race). Today's objects happened to keep a wait at every barrier; the explicit wait makes that hold by construction. The wave-private LDS handoffs in `gdn_chunk_kkt_solve.gfx11`, `gdn_chunk_scan_kkt_solve.gfx1201`, the gfx1201 Q8 FA2 prefill (`attention_q8_0_fa2_gqa.gfx1201`) and `gated_delta_chunk_wmma.gfx1151` get a wavefront release/acquire fence around their `wave_barrier`, which by itself has no memory semantics. Arithmetic is unchanged.
- **Dispatch: legacy 6-bit MQ6G256/HFQ6G256 (qt=15/qt=8) weights select the HFQ6 kernels in the container key selectors. Before, the batched MoE prefill ran the HFQ4 kernel on them, and the default `ornith-1.5:35b-a3b` (`.mq4`) emitted `!!!!` for every prompt on gfx1201.** `fused_gate_up_key_for`, `residual_gemm_key_for`, `fused_qkvza_key_for` and `fused_qkv_key_for` had arms only for the V2/MQ4C containers, so MQ6G256 fell through to the 136 B/group HFQ4 keys and read the 200 B/group 6-bit container as noise. The batched MoE prefill's shared-expert gate/up stage reaches `fused_gate_up_key_for` with MQ6G256 on checkpoints whose K-map Promote6 tier is v1 MQ6: `ornith-1.5:35b-a3b` `.mq4` (43 tensors: layers 0, 1, 38, 39 and 16 shared-expert down projections) and `qwen3.6-35b-a3b.mq4-awq-mi300x`. With the fix, both give coherent greedy text on gfx1201, with ordinary serve and on the multi-slot engine. Every other caller already took an explicit 6-bit arm first, so their routing is unchanged; H2 greedy-gate2 stays 33/33. `ornith-1.5:35b-a3b-mq4r` and the shipped `qwen3.6:35b-a3b` (`.mq4p`) carry no MQ6G256 shared-expert or attention tensors and were not affected. `HIPFIRE_MOE_MQ6_ADMIT=0` was the workaround before this fix.
- **Qwen3.5/3.6: `HIPFIRE_KV_V` is resolved once, at admission, so automatic VMM context sizing uses the V format the model actually loads.** Admission used to size the KV cache for V=q8 while the carrier loaded the env V (e.g. `lloyd3`), and a K/V combination the env made invalid was first refused mid-load. A typed `kv_v` still wins; the env still applies only to Qwen3.5-family single-rank loads.
- **Serve: Qwen XML tool-call arguments reach the client with the bytes the model wrote.** The parser `trim()`med every `<parameter>` value, so no tool call could write a file ending in a newline (HermesAgent HA-12 wrote 87 of 88 bytes). It now strips at most one leading and one trailing `\n`, the newline the chat template puts around each value, as vLLM's Qwen3-Coder parser does (`trim_one_wrapping_newline`). It also turned JSON-looking text into objects that the gateway then re-serialized compactly: a `string` argument written as `{"sum": 10}` arrived as `{"sum":10}`, and an HA-17 sub-agent retried that write 14 times. Values now stay strings until the gateway types them by the request's declared schema, in declaration order, following vLLM's schema conversion. A `string` parameter keeps the exact text. `integer`, `number`, `boolean`, `object` and `array` parameters are converted when the text parses. `null`/`None` becomes JSON null unless the parameter is a plain `string`. Undeclared parameters stay raw strings. Hermes-JSON tool calls are unchanged, except that a typed value is now turned into a string only when the slot accepts nothing but `string`.
- **The kernel pack now carries every module Flash-Next (Qwen4) and Qwen3.5-MoE load on gfx1100, gfx1151 and gfx1201, so a host without a device compiler loads them instead of failing.** The pack held none of the Qwen4 modules and only part of the Qwen3.5-MoE route. With no hipcc, the first unpackaged module stopped the load (`<module>: no usable cached kernel image exists, but hipcc is unavailable`). The registry rows come from kernel-load traces: an empty exe-relative `kernels/compiled/<arch>` makes the runtime write `<module>.index.json` for every module it compiles. The traces covered load, greedy AR on ~20- to ~3K-token prompts and native MTP. Each source is the expression its callsite passes to `ensure_kernel`. gfx1100 Flash-Next rows come from the callsites' gfx1100 arch checks, because no trace host can hold Flash-Next on an XTX. The two dense Qwen3.8-27B gaps are also packaged: `attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100` and `attention_verify_wmma_gfx1151`. Inventory: gfx1100 132 → 218 modules, gfx1151 122 → 200, gfx1201 137 → 205. Packs grow from 0.53/0.59/0.62 MB to 2.72/2.52/1.34 MB gzipped (gfx1100/gfx1151/gfx1201). `tests/fixtures/kernel-trace-qwen4-flash-next.tsv` and `kernel-trace-qwen35.tsv` pin the traced sets.
  - Strix Halo, with `HIPFIRE_NO_DEVICE_COMPILER=1`, an empty kernel cache and only the pack installed: `qwen3.8-flash-next-gptq3.mq4` produced 32 coherent greedy tokens with both AR and native MTP (tau 1.58), and the two runs gave the same text. The pack built from the base commit fails this same load at `qwen4_moe_iu4_sym_gfx1151` with the error above. On an R9700, `ornith-1.5-35b-a3b.mq4` also produced 32 coherent greedy tokens with AR and MTP, pack-only, with the same text from both. Kernel idproof against the base commit shows only added modules.
  - Known gap, not fixed here: Gemma4 and LFM2.5 still load modules the pack lacks on gfx1201. Gemma4-12B lacks 8: `attention_flash_q8_0_tile_batched`, `fused_gate_up_q8_0`, `fused_gemma4_qk_norm_rope`, `gelu_tanh_f32`, `gemm_hfq4g256_residual_wmma_gfx12_bt4`, `logit_softcap_f32`, `rmsnorm_residual_add` and `rope_partial_halved_batched`. Gemma4-26B-A4B lacks 8: `gelu_tanh_f32`, `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed`, `gemv_mq4g256_moe_gate_up_k8_indexed_k2816`, `kv_cache_write_q8_0_ring`, `logit_softcap_f32`, `moe_softmax_topk_k8`, `rope` and `rope_partial_halved`. LFM2.5-1.2B lacks 2: `conv1d_gated_decode` and `rope`. These models still need a device compiler. DeepSeek4 was not traced.
- **The kernel pack now also carries the Flash-Next (Qwen4) modules that long prefill, native MTP and serve load, so a cold kernel cache with the pack installed JIT-compiles nothing.** The earlier traces stopped at ~3K-token prompts.
  - What was missing on gfx1151: with the pack installed and an empty kernel cache, a 128K Flash-Next request on Strix Halo still compiled three modules. `gemm_bf16_xf32_multirow_pto2` is the HC projection in native MTP's batched fill; the pack had that kernel only as a symbol of another module key. `hc_row_fold` and `moe_router_softmax_top10_f32_fast` are both default on.
  - What was missing on gfx1201: `qwen4_silu_mul`, the shared-expert activation, which was packaged only for gfx1100/gfx1151. Also these default prefill modules, which the fixed pack now serves:
    - `gemm_wmma_lds_splitk`: the split-K WMMA projection module, covering the HC input_mix_down 160 × 80 tile and its reducer, the 128 × 128 router/shared tile and the 128 × 256 wide projections;
    - `hyper_read_up_wmma_gfx1201`;
    - the symmetric-IU4 MoE load check `qwen4_moe_sym_check_gfx1201`;
    - the symmetric-IU4 MoE prefill modules `mq_rotate_x_i4`, `qwen4_moe_rotate128_i4` and `qwen4_moe_scatter_stable_top10`;
    - `moe_router_softmax_top10_f32_fast`.
  - `gemm_bf16_xf32_multirow_pto2` is also registered on gfx1100 and gfx1201, since the callsite selects it on every gfx11+ SIMT arch.
  - The gfx1201 `tensor_ops` index gains four symbols the BF16 QSA state path reaches: `indexed_attention_attention_f32_batched_hg12`, `indexed_attention_select_scores_rows16_f32`, `indexed_attention_select_scores_rows8_f32` and `indexed_attention_select_from_scores`. A packaged index that lacks a requested symbol is rejected, and the module is recompiled.
  - The Flash-Next trace fixture now has these rows, so the registry test fails if any of them leaves the pack.
  - Packs change by additions only. Idproof against `928bed93c7`: every existing module is identical except the gfx1201 `tensor_ops` index, which only gains the four symbols; its object and hash are unchanged. gfx1100 adds 1 module, gfx1151 adds 3 and gfx1201 adds 9.
  - Strix Halo, `qwen3.8-flash-next-gptq3.mq4`, pack installed, empty kernel cache:
    - Before the fix, the first 128K Ciru request compiled the three gfx1151 modules above. Its TTFT was 78.4 s, against 81.3 s warm.
    - After the fix, this sequence wrote **0** objects into the kernel cache: Ciru 128K cold, 128K warm, 8K and 32K, then greedy AR, greedy MTP, sampled MTP and the serve battery. TTFT was 78.36 s cold and 78.55 s warm.
    - Token IDs in all three decode modes match the pre-fix run on 8 of 8 prompts, and the serve battery passes.
  - R9700 (card B), same model, same sequence and setup:
    - The intermediate pack, built without `qwen4_silu_mul` on gfx1201, still compiled that one module at 128K cold (TTFT 71.63 s, against 71.35 s warm).
    - The final pack wrote **0** objects into the kernel cache. TTFT was 70.66 s cold and 70.65 s warm.
    - Token IDs in all three decode modes match between the two runs on 8 of 8 prompts, and the serve battery passes.
  - The 129 s TTFT of the release matrix's tiny 128K run came from a lane with no kernel pack installed. That lane compiled every module, which took about 78 s. It was not a gap in the pack.
- **Serve: Flash-Next (Qwen4) accepts `tools` with native MTP attached.** A greedy request with `tools` used to fail with HTTP 400 (`tools are not supported on producer route qwen4_spec`) whenever the MTP head was loaded, and worked only with MTP off. `qwen4_spec` is now tool-capable: its spec emitter (`Qwen35Emit`) runs the same `<tool_call>` router over committed tokens as the AR producer. The tool-call grammar is forced off on Qwen4 MTP, as on Qwen4 AR, so a post-acceptance reject can never end an MTP turn that AR would continue. Qwen4 AR now enables its tool router only when the request carries `tools` (the MTP emitter's rule), so tool markup without `tools` is plain content on both routes.
  - Strix Halo, `qwen3.8-flash-next-gptq3.mq4`, `hipfire serve`, greedy, thinking off: a 4-turn tool round trip (call → tool result → answer → new question → call → result → answer), plus the streamed first turn, gives byte-identical content, tool calls (`get_weather {"city":"Paris"}` / `{"city":"Tokyo"}`), `finish_reason` and token counts with MTP on and off. With MTP on, all 19 greedy requests in the run logged `drafter=mtp`. The MTP-off turns match the base `3fb9dec05` binary byte for byte; base with MTP on returns the 400 above. The serve_harness battery (5 prompts) is byte-identical with MTP on and off in 2 of 2 runs per arm.
  - CPU parity tests drive the AR producer and the MTP emitter over the same token streams (tool calls, reasoning, stops) and require identical client output.
- **Serve: OpenAI `stop` works on Flash-Next (Qwen4) AR and native MTP.** Both routes answered `stop` with HTTP 400. They now take a string or up to 4 strings, match only the answer (not reasoning) across token boundaries, hold back a possible partial match while streaming, never return the stop text or anything after it, and finish with `stop`. A stop that cuts a tool call drops the call. Qwen4 AR previously checked a suffix only after the text had been streamed; it now uses the same holdback matcher as the Qwen3.5 routes.
  - Strix Halo, Flash-Next, greedy, streamed and not: `stop: ", 12"` on a 1–30 count returns `1, …, 11`; `"9, 2"` (split across tokens) cuts after `18, 1`; the earliest of four stops wins; `["Paris"]` returns `The capital of France is `; an unmatched stop returns the full answer; five stops is a 400. Content, `finish_reason` and token counts are identical with MTP on and off, and no stream chunk carries stop text. The same stop requests on Qwen3.8-27B (`qwen3.8-27b.mq4-xt`) with DFlash (every generation logged `drafter=dflash`) match its AR output.
- **Sampling: the CPU sampler honours request `top_k` and `min_p`.** `sample_cpu` (Flash-Next AR, the grammar-constrained Qwen3.5 AR and pipeline-parallel branches, the multi-slot VL first token) hard-coded a top-20 nucleus and ignored both. The new `llama::sample_top_k_p` follows the GPU `sample_top_p` kernel: a 20-wide gather when `top_k` is absent or ≤ 20, else 64-wide (`0` or > 64 = 64); `min_p` drops candidates below `min_p` × the top probability; then the nucleus. When nothing is cut (fields absent, `top_k: 20`, `min_p: 0`) the arithmetic and RNG draws are the legacy ones, so default output is unchanged.
  - Strix Halo, Flash-Next AR, `temperature 0.7, top_p 0.8, seed 123`: with no `top_k`/`min_p`, and with `top_k: 20, min_p: 0`, the text is byte-identical to the base binary. `top_k: 1` and `min_p: 1` now equal the greedy text (the base ignored both and returned the default sample); `top_k: 40` and `min_p: 0.2` at temperature 1.5 now change the sample. A CPU test pins the base sampler's token and RNG stream, and 200,000 fuzzed rows (ties, `-inf`, greedy) match the base function exactly.
  - Still not honoured (not changed here): Gemma4 eager, LFM2, Cohere2, MiniMax, DeepSeek V4 AR, LLaMA AR and single-slot Qwen3.5-VL image turns ignore `top_k`; those plus Muse Glimmer, DeepSeek V4 speculative and n-gram speculative decoding ignore `min_p`. `response_format` is still accepted and not enforced on the standard serve route (only multi-slot enforces `json_schema`); `docs/SERVE.md` now says so.
- **Legacy 6-bit dense Qwen3.5-family models (`HFQ6G256`/`MQ6G256`, e.g. `qwen3.5:9b-mq6`) decode again, and the gfx1201 retained PM4 tape records their residual GEMV (#810 by @aldrouil).** The FFN down-projection pre-rotates its input for `MQ6G256` and takes the fused SwiGLU-residual route, but `dispatch_swiglu_residual` had no `MQ6G256` arm, so every mode failed with `unsupported gemv.swiglu_residual` before the first token. It now launches `gemv_hfq6g256_residual`, the kernel `dispatch_gemv_residual` already used for `MQ6G256`. That kernel (wave32 and wave64 arms) also launched raw instead of through the recorder-aware `launch_maybe_blob`, so a retained tape ran it during capture but never replayed it (`wo` and `w_down` of every block); it now records like its HFQ3/HFQ4 siblings. `hipfire-xdna`'s `Bo::as_mut_slice` allows `clippy::mut_from_ref`, so the advisory clippy job passes.
- **Serve: unregistered Flash-Next (Qwen4) arch16 trunks are listed in `/v1/models`.** The serve-trunk allowlist that classifies files the registry does not name omitted arch id 16, so a locally quantized Flash-Next trunk was hidden from discovery even though it answers text completions. Follow-up to #787 by @aldrouil.

### Serve & CLI
- **Multi-slot serve: thinking-on turns follow the rendered template, and `reasoning_effort` is accepted instead of refused (#811 by @ghazni101).** The slot route reads the assistant framing back from the Jinja render, classifies the rendered assistant turn by its think markers instead of a suffix match, and starts the stream parser's think state from the prompt rather than from `enable_thinking`. `reasoning_effort` is no longer refused with "reasoning_effort is not supported": it goes to the template with the sequential route's suppression rules (`none`/`off`/`chat`, thinking off, or a one-token think budget leave it unset), and it is gone from the multi-slot refused-fields advert. On Qwen3.8-27B (R9700, 2 slots) the greedy serve battery and chain text is byte-identical to beta, and decode is unchanged (battery and chain mean over 3 fresh processes per arm, ABBA: beta 25.92 / 26.71 / 26.77, #811 25.47 / 26.72 / 26.82 tok/s).
- **Serve: `timings.decode_tok_s` is only a daemon-measured decode rate (#807 by @ghazni101).** `completion_timings` fell back to the wall-clock `tok_s` (prefill included) when the daemon sent no `decode_tok_s`, which only happens on the multi-slot route, so multi-slot responses reported prefill time as decode speed. Multi-slot responses now leave `timings.decode_tok_s` unset; the wall rate is still in `hipfire.tok_s`. The standard route already sends `decode_tok_s` and is unchanged: on Qwen3.8-27B (R9700) the serve battery and chain text is byte-identical to beta.
- **Serve: an opt-in chat UI at `/ui` (#808 by @ghazni101; `--ui`, `serve.ui` or `HIPFIRE_SERVE_UI`, default off).** `hipfire serve` can serve a self-contained browser chat client from embedded assets: `/` redirects to `/ui`, conversations are kept in the browser, and with multi-slot serve each conversation generates in parallel. The reasoning-effort picker appears only when the loaded template's render changes with `reasoning_effort` (`reasoning_effort_native`). With the UI off, `/` and `/ui/*` stay 404 as before. On Qwen3.8-27B (R9700, 2 slots) with the UI on, the greedy serve battery and chain text is byte-identical to beta, the UI assets are served and `/ui/../etc/passwd` and unknown asset names are 404. With the UI off, decode is unchanged (battery and chain mean over 3 fresh processes per arm, ABBA: beta 25.92 / 26.71 / 26.77, this stack 27.27 / 27.27 / 27.57 tok/s).
- **Dense tensor-parallel Qwen3.5/3.6 serves MTP speculative decode, opt-in with `speculation = "mtp"` (#769 by @alpineQ, ported onto the SCS MTP split).** At `tp>1` on a dense trunk, an explicit `speculation.mtp = on` (`--spec mtp`) replicates the MTP head on rank 0, from the bundled trailer, the CLI-resolved `.mtp` sidecar, or the trunk's sibling `.mtp`, and the request takes the spec route. A missing head is a load error, as on one GPU. `auto` (the default) and `off` keep the AR mesh loop even with a head beside the trunk, so existing `tp>1` serves are unchanged.
  - Rank 0 drafts with the same `mtp_draft_phase_inner` the single-slot and multi-slot paths use (no proposal graph on the mesh). The verify runs across ranks through the dense-TP persistent-PBS capture forward. The accept rule and the `prev_hidden` capture are the shared `mtp_verify_accept`, and every rank restores and replays its own DeltaNet state. The prompt fill uses land's `(tok_p, h_{p-1})` pairing.
  - Single-GPU and multi-slot MTP run the same operations in the same order as before: `mtp_verify_accept` is the former head of `mtp_accept_and_rollback`, moved verbatim.
  - Not taken from #769: the `MtpTrunkBackend` trait and the 2..=3-row Q8 multi-row attend on dense TP, so the dense-TP prefill and verify attention are land's. Block-verify drafters (n-gram, DFlash, DSpark) stay refused on dense TP. The n-gram-mod composition inside MTP drafts natively there.
- **Serve: multi-slot serve (`serve.multi_slot = true`) runs 35B-A3B checkpoints whose MoE attention is legacy MQ6G256 (`ornith-1.5:35b-a3b` `.mq4`, `qwen3.6-35b-a3b.mq4-awq-mi300x`). Before, every request failed with "DeltaNetMoe layer attention weights must be uniformly Q8_0 or uniformly MQ4G256".** These checkpoints keep layers 0, 1, 38 and 39 at uniformly MQ6G256 attention. The single-slot batched prefill already ran them through its 6-bit arms, but the multi-slot MoE gate (`require_batchable_{deltanet,fullattn}_moe_layer`) refused that dtype. The gate now admits uniform MQ6G256, and the slot bodies take the same FWHT rotate and container-selected HFQ6 kernels as the reference. HFQ6G256, PARO, Lloyd and mixed-dtype MoE attention are still refused with the same error. The shipped `qwen3.6:35b-a3b` (`.mq4p`, all-Q8_0 attention) and `ornith-1.5:35b-a3b-mq4r` were never refused. On these MoE checkpoints, concurrent greedy text depends on batch composition even at 2 slots: it is byte-identical across repeats of the same 2- or 4-slot submission, but 1–4 of 4 prompts diverge from 1-slot serial. That divergence predates this change; the base binary shows it on `.mq4p`.
- **Config: `speculation.mtp_ngram` (`off` | `on` | `auto`, default `off`) turns on MTP + ngram-mod. `HIPFIRE_MTP_NGRAM` overrides it.** Before, the composition could only be enabled by that env var. `on` arms it for greedy, thinking-off MTP requests. `auto` resolves to off, so the default is unchanged. On gfx1201 / H2 it is +13 % to +387 % decode on copy-heavy prompts and within about 1 % on the rest of the serve battery. It is not the default because its greedy text differs from MTP alone on some prompts (tool_edit, edit-session turn 2, the coding session), and because the n-gram pool persists across requests, the same prompt can decode differently later in the same process.
- **Serve: `n != 1` is refused with a 400 `invalid_request_error` on the standard chat route too, not only under multi-slot.** The standard route used to accept `n: 2` and return one choice. One validator serves both routes; `n: 1` and `n: null` pass.
- **Multi-slot: a request without sampling fields gets the same defaults as the sequential route.** The slot backend defaulted to `temperature=0.0`, `top_p=1.0`, `max_tokens=512` and ignored `max_tokens_fit`. It now uses the `.hfq` author recommendation, then the arch ladder (Qwen3.5: 0.3 / 0.8), a 4096 `max_tokens` default, and clamps an omitted `max_tokens` to the room left after the prompt. The sequential, continuous-batch, and slot routes share `resolve_temp_top_p` and `DEFAULT_GENERATE_MAX_TOKENS`.
- **Serve: every client mistake the gateway refuses is a 400, by error type, not by wording.** `request_error_status` used to grep the message for `invalid`/`required`/`max_tokens`/…, so refusals without one of those words were 500s: `n != 1`, `best_of`/`logit_bias`, out-of-range `temperature`/`top_p`/`top_k`/penalties, unknown message roles, `unsupported tool_choice value`, `top_logprobs requires logprobs`, `logprobs must be a boolean`, remote or second `image_url`, non-boolean `enable_thinking`, non-string `thinking_budget`, and the multi-slot refusals. Gateway validation now raises a typed `InvalidRequest` (400) and a missing model a typed `ModelNotFound` (404); the message ladder is gone. Untyped errors are 500, so a model that ignores `tool_choice: "required"` or emits a malformed canonical tool call is now a 500 (it was a 400). Multi-slot refusals no longer carry the `[request validation] ` prefix.
- **Daemon: a bad `seed`, an oversize `image_base64`, or an image sent to a model with no vision encoder is a typed `validation`/`unsupported` error (HTTP 400), not `internal` (500).**
- **Serve: vision (Qwen3.5-VL, dots.ocr image/text/n-gram, LFM2-VL) and pipeline-parallel generations report `finish_reason: "length"` when `max_tokens` runs out.** Their `done` carried no `finish_reason`, which the gateway reads as `stop`. dots.ocr n-gram also reports `length` when the context runs out. Continuous-batch lanes pass the producer's reason through instead of folding everything else into `stop`.
- **Daemon: a KV-eviction (TriAttention/CASK) HIP failure during Qwen AR prefill, decode, think close, budget alert or trailer fails the request closed with a `gpu` error and an attested rollback, instead of panicking the daemon.** A multi-slot terminal with no bound lane ticket is an `internal` error instead of a panic.
- **Windows and WSL2: the daemon starts again; WSL2 is the recommended Windows path.** 0.4.0 refused to start `daemon.exe` on every non-Unix host ("per-GPU locking requires Unix flock"), and under WSL2/ROCDXG the daemon exited because it looked for the KFD topology. Native Windows now takes the same per-card lock files with `LockFileEx` (exclusive, non-blocking, holder PID reported; default directory `%ProgramData%\hipfire\locks`). Hosts without KFD (WSL2: `/dev/dxg` and no `/dev/kfd`; native Windows) identify cards from HIP's UUID and PCI address and refuse `hardware.devices`. Under WSL2 the retained Redline PM4 default falls back to the HIP graph with one log line, an explicit `replay.backend=redline|shadow` is refused, and automatic KV selects legacy (explicit `vmm` refused); `HIPFIRE_UNSAFE_WSL_REDLINE=1` and `HIPFIRE_UNSAFE_WSL_VMM_KV=1` lift these and are unsafe until certified. The MSVC-only `/STACK:` link flag no longer breaks `x86_64-pc-windows-gnu` links (gnu gets `--stack`). GETTING_STARTED documents AMD's ROCm 7.2.1 ROCDXG install and the WSL GPU list, and native Windows as best-effort (no PM4, legacy KV, no TP). Native Linux is unchanged.
- **`hipfire bench --json` records the GPU power state and memory configuration of every measured run (standard, `--ttft` and `--matrix` paths), from world-readable sysfs only.** A sampler reads the card's `gpu_metrics`, its amdgpu hwmon and k10temp `Tctl` every 0.5 s during each run. Per run, a `power` entry reports gfxclk avg/min/max, the lowest SMU gfx clock cap, socket and gfx power, the maximum gfx temperature and Tctl, mean fclk, PPT/STAPM/THM/PROCHOT throttle residency as a percent of the run, the perf level, the `OD_SCLK` range, every labelled hwmon input, and VRAM carve-out/GTT total, used and free plus host `MemTotal`/`MemAvailable`. `power_device` names the card, which is resolved the same way as the daemon's logical device 0. `gpu_metrics` is decoded by its header revision; only format 3.0 (Strix Halo and other `gpu_metrics_v3_0` APUs) is decoded. Any other revision, or a missing file, keeps the hwmon and sysfs fields and marks the rest `"unavailable"`, and sampling never fails a bench. A run whose limiter residency exceeds 10 % adds one `warning:` line. On hipx (Strix Halo, dense 27B, `--matrix --pp 8192`, 3 runs at 1277.1 / 1269.2 / 1260.8 tok/s) the runs read socket power 145.1 / 156.0 / 156.4 W, fast PPT residency 84.3 / 87.3 / 88.6 %, gfx temperature max 86.4 / 91.0 / 92.9 °C and VRAM carve-out 98,304 MiB, and the bench printed the throttle warning. [`docs/CLI.md`](docs/CLI.md#bench-power-and-memory-state) documents the fields and adds a Strix Halo host note: stock prefill is fast-PPT- and then thermal-limited, and a ~130 W cap on all four limits costs ~5 % of burst prefill. [`perf-benchmarking.md`](docs/methodology/perf-benchmarking.md#gpu-power-and-memory-state) now requires these fields with every kept claim.
- **`hipfire bench --max-seq N`, and `--matrix`/`--redline` loads sized to the run.** A matrix load used to take the configured or automatic context whatever the shape: 32768 for Qwen3.8-Flash-Next (its admission default), or min(trained context, card capacity) for growing Qwen VMM KV. Without the flag it now gets what the run needs: the largest `--pp`, or `--ctx` plus `--tg`, plus the 32 positions of probe headroom. The value is at least 512 and is capped at the model's declared `max_position_embeddings`. A shape past that declared context fails before the daemon starts, naming the need and the limit. So does a `--max-seq` below the need. The standard, `--ttft`, `--exp` and `--concurrency` paths have no fixed shape, so they keep the configured or automatic context unless `--max-seq` is given. Every JSON report records `max_seq` (`effective`, `requested`, `source`, `run_need`, `model_limit`). On hipx (Strix Halo, 3 runs per process):
  - Flash-Next GPTQ3 `--matrix --pp 8192 --tg 32` loads 20064 instead of 32768, with pp8192 at 1927.3 / 1921.8 / 1918.3 tok/s.
  - A dense Qwen3.8-27B matrix (`--pp 512,8192 --ctx 128,8192 --tg 32`) loads 8256 instead of 262144. Decode is the same in all four processes of a base/new/new/base run (15.09–15.10 tok/s at ctx 128, 14.68–14.69 at 8192). Prefill followed GPU temperature, not the context: pp8192 medians were 1258.5 for the cold base process, 1193.2 and 1197.3 for the new ones, and 1174.2 for the hot base process.
  - The dense standard path loads the same automatic 262144 as before.

### Quantizer
- **Quantize: V2 MQ4 K-map Promote6 uses MQ6G256V2 (qt47), not legacy MQ6G256 (qt15), on the audited LLaMA/Qwen3, Qwen3.5-family and Qwen2/dots.ocr loader paths.** This covers promoted attention, routed experts, shared experts and shared-down; legacy MQ4 and compact MQ4C keep v1. Encoding is unchanged for arch IDs 10, 11, 12, 13, 14, 22, 23 and other unaudited IDs; V2 Lloyd stacked experts retain legacy qt13, while ordinary MQ4/MQ4V2 and nonexpert V2 Lloyd promotions retain qt15. Existing artifacts and runtime kernels are unchanged. This is not layout-only: V2 uses two fp16 affine grids instead of one f32 grid, at the same 200-byte group size. CPU quantization of real Qwen3.5-35B-A3B tensors across 14 Promote6 roles lowered mean-absolute and relative-L2 reconstruction error in every sampled role; the largest sampled max-absolute error was 5.498e-3 (v1) versus 5.440e-3 (v2). Decoded tokens are not claimed byte-identical.
- **Quantize: a flag the chosen path never reads is an error (exit 2), not a silent no-op.** `hipfire-quantize` dispatches to one of seven paths (`--flux-pipe`, `--qwen4-flash-next`, final-code import, the four special `--format`s, GGUF weight input, `--reap-overlay`, and the safetensors pipeline), and each path reads only some flags. For example, `--format maple --mq4v2-symmetric` used to write an asymmetric model. A table in `cli.rs` maps each flag to the paths that read it. A test fails when a new flag has no entry. `--threads` now applies to every path. On the safetensors path, `--awq` with a format, arch or tier that has no AWQ code is an error. So is `--imatrix` when neither `--awq` nor an imatrix-consuming format is given, `--reap-overlay`/`--reap-bake` with `--mq*v2-symmetric`, and an unknown `--kmap-mode`, which used to fall back to `alternating`.
- **Quantize: an unknown `--format` is an error.** A typo such as `mq4v3` used to write a legacy Q4F16G64 (qt=0) model. GGUF-only formats (`tq2`, `bq1`) are refused on safetensors input. `q4f16` is still accepted.
- **Quantize: Cohere2-MoE `--format mq4` writes MQ4G256 experts.** It used to write Q8 experts. Every MQ4 spelling (`mq4`, `mq4v2`, `mq4v1`) writes MQ4G256 (qt=13), the only MQ4 dtype the cohere2moe loader reads. A format with no Cohere2 tier is an error.
- **Quantize: a Gemma-4 unified build with `--include-vision` no longer sets `has_vision: true`.** The Gemma-4 text-only path drops every vision tensor, but `has_vision` was set before that skip.
- **Quantize: spill and write failures stop the run with exit 2 and leave no partial output.** A failed spill write used to turn tensors into zero-length entries, and an unwritable spill dir silently held the whole model in RAM. `write_hfq` now writes `<output>.tmp.<pid>`, fsyncs it, and renames it into place, so a failed write leaves an existing output untouched.
- **Quantize: a HF model ID resolves to the `refs/main` snapshot.** It used to take whichever cached snapshot `read_dir` listed first. Without `refs/main`, the newest snapshot is used and a warning is printed.
- **Quantize: `--source-url` and `--license` are recorded in a `hipfire_provenance` metadata object.** Builds without either flag are byte-identical to before. The no-op flags `--allow-lowbit-ptq`, `--allow-degenerate-ternary` and `--awq-imatrix` are removed. The `--format mq4-mq6exp` warning now says it writes MQ4G256 v1 dense tensors, not `mq4` (V2). The crate no longer has blanket `#![allow(dead_code, unused_imports, …)]` attributes, and the dead code they hid is deleted. The hand-rolled SHA-256 is replaced by the `sha2` crate.
- **Qwen3.8-Flash-Next: the default `qwen3.8:flash-next` pull is now the symmetric GPTQ3 requant.** The routed experts are symmetric MQ4G256V2/MQ4G128V2 solved with Hessian GPTQ on a 262,144-token calibration corpus disjoint from WikiText-2 (sha256 `e00111ee…`, no AWQ fold; 553 of 25,088 experts keep RTN). The other 1227 tensors are byte-identical to the previous file. On the F16 MoE route (`HIPFIRE_QWEN4_MOE_SYM_IU4=0` on gfx1151 or gfx1201), against the BF16 source (32 × 512 WikiText-2, ref sha256 `0314efab…`): mean KLD 0.0661 vs 0.0743 for the previous RTN-asymmetric file, p99 1.095 vs 1.263, top-1 0.8995 vs 0.8936, PPL 5.481 vs 5.537, at the same pp8192 on gfx1151 (1459–1461 tok/s for both). On gfx1151 and gfx1201 the default is now the symmetric IU4 MoE route for symmetric artifacts such as this one (`=0` opts out), which trades quality for speed. On Strix Halo at the 0.4.1 defaults, pp8192 is 1928.5 tok/s (median of 3 fresh processes) at BF16 KLD 0.102650 (logits sha256 `ead06843…`). Set `HIPFIRE_QWEN4_MOE_SYM_IU4=0` for quality. The new file is `qwen3.8-flash-next-gptq3.mq4` (125,288,544,792 B, sha256 `8b15b6fe…`), also tagged `qwen3.8:flash-next-gptq3`. The previous RTN-asymmetric file keeps its name `qwen3.8-flash-next.mq4`, so v0.4.0 registries that pin its sha256 still pull it, and is tagged `qwen3.8:flash-next-rtn-asym`.

### PM / kernel toolchain
- **PM native emission carries its own version stamp (`hipfire peacemaker native-emit <crate version>`) in `.comment`, not a ROCm linker identity.** Oracle identity compares every other ELF section and header field exactly, normalizing only the comment contents and their derived layout; committed PM bundles and checksum pins are refreshed without changing instructions, kernel descriptors or metadata. Native build parse-back runs when `llvm-objdump` is available and logs a skip otherwise; the toolchain oracle still requires it.
- **PM builds its code objects natively: no ROCm assembler, linker or bundler in the build path.** The new `hipfire_isa::native` lowers each builder instruction to `peacemaker-ir` with `peacemaker_lift::text::parse_line` and the gfx11/gfx12 codecs, resolves branch labels, and writes the code object v6 that `llvm-mc` + `ld.lld -shared` produced: `.text` with 256-byte `s_nop 0` padding between module kernels, the 64-byte `<sym>.kd` descriptors in `.rodata`, the MessagePack `NT_AMDGPU_METADATA` note, `.dynsym`/`.gnu.hash`/`.hash`/`.dynstr`/`.dynamic`, the relro padding and the symbol tables. It also writes the `__CLANG_OFFLOAD_BUNDLE__` wrapper (`-bundle-align=4096`). `peacemaker custom build`/`profile` and the certification tests use it by default; `hipfire-isa emit --co FILE --bundle FILE [--host-target TRIPLE]` needs no ROCm at all. The ROCm path survives only as the test oracle `toolchain::oracle_assemble_link_bundle` (feature `toolchain`), and the `lift` feature is gone (`peacemaker-ir`/`peacemaker-lift` are plain dependencies).
  - `peacemaker native --arch=… file.s [--co=…] [--bundle=…] [--host-target=…]` is the native build alone, without certification; it is the command a certification manifest records for the build (tool role `native`).
  - Identity: `tests/native_identity.rs` builds every PM-emitted symbol on gfx1100/gfx1151/gfx1201 alone, in every product module, and as a `peacemaker profile` build per arch, through both paths: 55 symbols, 55 single-symbol objects and 12 modules. Every section other than `.comment` is byte-identical to `llvm-mc` + `ld.lld` and `clang-offload-bundler`; only the stamp and its derived layout differ. A strict whole-file comparison helper remains available explicitly.
  - `parse_line` now reads gfx1100/gfx1151 (`s_waitcnt` counters, `glc`/`slc`/`dlc`, true16 halves, VOPD, per-target `hwreg` names) and the remaining gfx1201 builder spellings (`null` SOFFSET, GLOBAL SADDR and `th:`, `swizzle(BROADCAST,…)`, packed `op_sel`, `v_fmamk_f32`, `row_xmask`, numeric branch offsets). New table rows `s_getreg_b32` (gfx1100) and `v_wmma_i32_16x16x16_iu8` (gfx1201), each pinned to its llvm-mc sample; the gfx12 decoder takes SRC1/SRC2 true16 halves from their own OPSEL bits. Certification records the native writer as the new radiowave tool role `native`.
- **PM LDS address analysis follows the CFG instead of stopping at block boundaries.** Unsigned interval tracking resolves constants, lane counts, hardware wave-ID extracts, guarded loop induction, scalar multiplies and vector multiply/shift-add chains; overflow, memory-derived values and unproved partial-EXEC writes remain conservative. On the roadmap's distinct JIT corpora, unknown addresses fall from 7,226/7,246 (99.72%) to 6,596/7,246 (91.03%) on gfx1100, from 12,680/12,705 (99.80%) to 12,009/12,705 (94.52%) on gfx1151, and from 7,419/7,606 (97.54%) to 6,870/7,606 (90.32%) on the current gfx1201 cache snapshot. Wait replay now uses the metadata-selected scalar mask width and distinguishes pending d16 low/high halves, clearing all 16 historical packaged RAW sites per gfx11 arch (eight had already been cleared by the GLOBAL-address-width fix). Real missing-wait and VerifyAttn race negatives still fire. The pinned `s_cmp_le_u32` row completes GDN chunk-scan lifting on gfx1151; both gfx1151/gfx1201 compiled modules audit as checked with zero unwaited LDS barriers.
- **Peacemaker `lint OBJECT.co` reports CPU-only steady-loop schedule costs from lifted gfx11/gfx12 ISA.** It counts WMMA, accumulator chains, VALU and VMEM packets/bytes per WMMA, and outstanding VMEM units before WMMA via CFG wait replay. A declared K128 tile rule and captured expert rows add per-epoch WMMA and executed-padding fractions. The initial 26-symbol calibration distinguishes all six NT4/NT8 wait-deletion regressions (full drain before WMMA) from 20 unchanged symbols; `crates/hipfire-isa/tombstone.toml` records seven rejected kernel/design families and their measured kill evidence.
- **PM GEMM micro-kernel library:** composable fragment loaders, independent seeded IU4/IU8 chains, typed prefetch distance, and SET/ADD/imported-SiLU/BF16-RNE epilogues. Dense `pm_v2b` uses the shared parts with byte-identical assembly, builder proofs and bundles for all three symbols. The host grouped scheduler orders experts heaviest-first and executes only pad16 tails; captured L4/L24/L44 routes at 1536 and 8192 tokens reach their exact pad16 floor. No dispatch change or timing claim.
- **Standalone RIP front end (`hipfire-rip`, R6 first step):** `.rip` kernels are a Rust-subset dialect evaluated without rustc, so editing one needs no cargo rebuild; dedicated kernel syntax is to follow. LDS regions, barriers and waits go through a new runtime-checked driver in `peacemaker-author` (`runtime`), which makes the same backend calls as the typed core and refuses illegal transitions. The QSA convert and attention ports (`kernels/qsa_gather.rip`, gfx1151 and gfx1201) produce the same assembly, `BuilderProof`/`ModuleProof` and committed `.hxaco` bytes as `qsa_gather.rip.rs`, and offline Peacemaker certification has zero obligations. `.rip` twins of both historical LDS races are rejected, and so are a raw-escape barrier and a loop-carried undrained store. The Rust DSL is unchanged. `scripts/no-gpu-ci.sh` now runs `peacemaker-author`'s tests, including its trybuild compile-fail race tests.
- **PM CPU snapshot replay (`peacemaker_ir::emu`, `pm-emu`).** Lifted gfx1151/gfx1201 wave32 kernels execute at captured device addresses, including LDS/barriers, raw-buffer bounds and WMMA fragment layouts. A separate opcode-semantics table and bit-level numerical models replace host-libm approximations; missing semantics and unsupported numerical cases are hard errors with wave/register bit diagnostics. Decoded SOPP barrier IDs retain their 16-bit meaning. Measured NaN, divide and conversion edges preserve payloads and destination masks; mixed half-result FMA and gfx1201 scaled DIV_FMAS round once without intermediate-F32 loss. The lossless hardware corpus records explicit qualification limits. Full-grid replay is byte-identical across all buffers for 24 three-poison small fixtures and eight real-route fixtures covering QSA convert/attention and symmetric NT4 gate-up/SiLU/down on both architectures. `PM_R2_SNAP` exercises captured replay; selected-workgroup runs report their coverage limit. No inference dispatch change.
- **Default-off PM MoE schedule compositions:** NT4 independent seeded chains retain each output's original K/fold order; compact expert/M-panel lists are heaviest-first, and whole-dead tiles issue no VMEM/DS traffic. The CPU screen enumerates resource-bounded chain/prefetch variants, requires byte-exact lifting, and strictly certifies the selected gfx1151/gfx1201 schedules. Shipped NT4 bundles and runtime dispatch are unchanged; no timing gain is claimed.
- **PM emission accounts for every emitted SGPR operand and VCC on gfx12.** Imported epilogue regions are included in the scalar high-water scan, with explicit register-range widths. Kernels using VCC reserve it and include its two registers in metadata, matching hipcc; obsolete per-kernel VCC reservations are removed. The embedded gfx1201 bundles are regenerated with only their `.note` metadata changed and `.text` byte-identical: `_b1` 875fb564 → 8edd7565 and `_b1s` 4ffe3580 → cc0288f8 (`.sgpr_count` 88 → 90), F2 c4e5c52d → 6a2dbe59 (104 → 106), and the sym module c49d3f02 → 7179393f (82/70 → 80/68, since its manual VCC reservation is gone). The gfx1151 sym module is byte-identical. Instructions are unchanged by this metadata correction: gfx10–12 allocate 128 SGPRs, but HIP occupancy queries consume `.sgpr_count`, so correct accounting also matters to dispatch occupancy decisions.
- **PM operand widths follow addressing mode and wave metadata.** GLOBAL instructions with an SGPR base read one 32-bit VGPR offset on gfx11/gfx12; `saddr=off` still reads a 64-bit vector address. Explicit scalar lane masks contribute one SGPR in wave32 and two in wave64 to resource accounting and definedness, without narrowing real scalar data pairs.
- **PM wait/definedness replay preserves immutable scalar guard correlation and loop issue positions.** Comparisons correlate only when each scalar input has one dominating definition outside CFG cycles; mutable or unknown guards keep both edges. In-order waits use the minimum younger-unit suffix across paths rather than first-issue IDs or union size; out-of-order counters retain conservative retirement. This clears GDN scan's original 35 modelling findings. The authoring-only GDN scan phase 2 now drains each step's P loads unconditionally, before the DOP=0 skip (down to the six younger Q/K/P prefetch loads, and to zero on the final step), so the DOP=0 and DOP≠0 paths join with the same load state. No shipped bundle or runtime route changes.
  - Immutable-guard eligibility is prefiltered before graph construction, and dominance/cycle proofs use iterative reachability only for repeated candidates. Long instruction streams no longer require recursive instruction-level SCC walks.
  - Wait convergence retains states at block entries and captures each mid-block branch exit at its issuing position. Straight-line transfer avoids per-instruction fixpoint copies without losing predicate-selected reachability or ordered retirement bounds.
  - Out-of-order counters do not accumulate unused suffix ages during convergence: their pending-event union is unchanged and only zero waits retire them. Waits and definedness first attempt an unpartitioned conservative proof, refining immutable guards only when a hazard or undefined read remains; already-proven kernels avoid Cartesian path replays.
- **PM independent SGPR hazard certification follows the CFG.** Decoded guards retire pending writes only on reaching paths, including loop backedges; another branch arm or a guard before a newer write cannot satisfy the obligation. Explicit vector lane masks use the kernel's metadata-selected wave width, while genuine 64-bit scalar operands remain pairs. The pinned gfx1201 codec now covers GDN scan's unsigned minimum, e64 f16 conversion and halfword LDS accesses, with byte-exact round-trip checks. This does not regenerate shipped bundles or change any runtime route.
- **PM kernel language M0: the new `peacemaker-author` crate puts a typed core in front of the `hipfire-isa` builder, and the three shipped IU4 builder kernels are ported onto it with byte-identical output.** Targets are types (`Gfx1100`, `Gfx1151`, `Gfx1201`, with `MmaIu4` and `SplitBarrier` capabilities); LDS regions carry their ownership state (`LdsRegion<R, Free | Writing | Published>`, two-buffer `Ring`); a store returns a `Pending` token that must become `Drained` before the barrier that publishes it; `loop_carried` requires the same state type on entry and back edge; a branch body under a wave-uniform condition gets no barrier. The Halo VerifyAttn `lgkmcnt` race and the gfx11 FA2 mailbox race are now rustc errors (`trybuild` tests, with fixed controls that compile and run). `iu4_v2c` (gfx1100), `iu4_v2b` (gfx1151) and `iu4_gemm` (gfx1201) declare LDS, store, load, drain, barrier, loop and skip through the typed API. All 39 builder emissions re-emit byte-identical `.s` and proofs, the five embedded `.hxaco` `.text` sections are unchanged, and the HIP kernel packs are identical on gfx1100, gfx1151 and gfx1201. A builder wrapped by the typed core refuses untyped LDS, barrier and loop calls. The lowering seam is sealed: every state-changing `Backend` entry point takes an `Auth` only the core mints (one session per `Workgroup::new`), so raw access through `Wave::isa` cannot call them (a `trybuild` error), and a second `Workgroup::new` over the same backend, which a wave-scope body could use to reach a barrier, is refused. Kernel exits are typed (`Workgroup::exit`/`exit_if`/`end`: the exit label followed by `s_endpgm`, nothing after it). Where control paths meet (a skip target), the backend joins every path's wait ledger, LDS slot states and gfx11/gfx12 hazard trackers, so a wait or guard either path needs is emitted; the ported kernels need none beyond what they emitted, and stay byte-identical. Building `hipfire-isa` plus the new crate takes 1.07× (build) and 1.09× (check) as long as `hipfire-isa` alone did. No runtime crate depends on it.
  - The lowering seam is closed to raw control flow. `Builder::push` takes one plain statement: a branch, `s_endpgm`, trap, PC write, barrier, label, directive or embedded line break is refused, so wave-scope code can no longer re-enter earlier code through `isa()`. The builder's program, ledger and LDS slots are private (read-only `program()`/`ledger()`). Untyped kernels branch through `Builder::control`, which a sealed builder refuses; typed kernels branch only through the core: `Wave::forward` blocks of forward-only, wave-uniform branches (`Forward::branch_if`/`goto`/`place`), `Workgroup::exit_unless`, and `Workgroup::end_with` for an exit tail. `iu4_gemm`'s fused epilogue and GDN dispatch, and the `iu4_v2c`/`iu4_v2b` exits, use them.
  - Every branch target joins every path into it, and a join after `s_branch` takes the branch's state instead of the unreachable fall-through. Ledger joins reconcile per id: equal-shape in-order loads merge slot by slot and keep both paths' ids (waits stay exact); stores, gfx11 LGKM with SMEM, gfx12 KM and unequal shapes keep the `maybe` union, which drains the counter.
  - Loop heads join the entry with the back edge: the body is re-emitted from the joined hazard trackers and LDS slots until the back edge adds nothing. This corrects the gfx1201 GDN loops' `s_wait_alu depctr_*` guards: the back edge needs more in `gdn_scan` (+4), the fp8 QKVZA+GDN symbol (+4) and the iu4 `qkvzagdn` symbol (+5), and the path-exact joins drop 12 guards the iu4 `qkvzagdn` symbol inherited from code that never reaches them (one more inherited guard stays, now owed by the back edge). The missing guards were a latent hazard; no failure was observed. Every other builder emission is byte-identical. The embedded gfx1201 bundles are regenerated with the same recipe (it reproduces the old bundles from the old emission): `_b1` c6e714e7 → 875fb564, `_b1s` d5190d97 → 4ffe3580, F2 73168098 → c4e5c52d. Only the fused QKVZA(+GDN) symbols change, and only in `s_wait_alu` guards.
  - The Qwen4 symmetric IU4 MoE emitter (`qwen4_moe_sym`) branches and ends only through the typed control: SCC0 forward blocks, skips and exits keep each branch's polarity, a barrier after a wave exit is refused, and handoff reader bodies receive the `End`. The `run_walk` loop-head join adds two `s_wait_alu depctr_sa_sdst(0)` guards to each gfx1201 NT4/NT8 gate/up and down symbol; `qwen4_moe_iu4_sym_pm_gfx1201` is regenerated (49f96d9e → c49d3f02). The gfx1151 module is byte-identical.
- **peacemaker audit: new CPU-only finding `lds_store_unwaited_at_barrier`.** It flags a workgroup barrier that some CFG path, loop back-edges included, reaches while a DS store or atomic still holds LGKMcnt (gfx11) or DScnt (gfx12). That is the bug class of the gfx1151 VerifyAttn co-residency race fixed in `1ba84942a`. The check lifts the code object byte-exactly and reports peacemaker-ir's `barrier-ds-pending` facts (`passes::waits` replay plus `passes::barriers`), so there is no second wait analysis. `peacemaker audit --arch …` reports it per kernel. `peacemaker audit --lds-barrier PATH...` sweeps every `*.hsaco`/`*.co` under the given paths and fails on any finding or any module the lifter rejects. A fixture test runs in `scripts/no-gpu-ci.sh`. It covers the pre-fix VerifyAttn object (flags exactly the `.LBB0_36` loop-head barrier), the fixed object, and gfx1201 KT48, which must both be clean. The gfx1100 table gains `buffer_load_u16`, `v_cvt_f16_i16_e32` and `v_mul_f16_e32`.
  - Limit: a module the gfx11/gfx12 tables cannot fully decode is reported as unchecked, not clean. At origin/beta `1176d7887`, the lifter covers 34/45, 33/41 and 30/45 barrier-bearing compiled modules on gfx1100, gfx1151 and gfx1201, with 0 findings.
  - The gfx12 and gfx11 codec tables gain the 63 + 5 opcode/form rows the current JIT caches use (FLAT on gfx12, 64-bit and f16 compares, DPP min/add/or, `v_min3`/`v_minimum3`/`v_maxmin`, f64 converts, DS add/or/`load_b96`/`load_2addr_b64`, scalar f16 converts, aperture sources `src_shared_base`…), each pinned to its llvm-mc sample with a decode/re-encode test. `peacemaker audit --lds-barrier` over the gfx1201 JIT caches at `fe77c0837` (integrate041h/i and iu4-roofline, 412 modules, 62 distinct objects) goes from 176 unchecked modules (34 distinct objects) to 0, every module re-emitted byte-exact, 0 findings.
- **The offline railgun crates (`railgun`, `railgun-cert`, `railgun-corpus`, `railgun-jitcorpus`) and the peacemaker kernarg pass are now on beta.** This is the offline half of `railgun/m1p`: the program model and the G7 shadow diff, the kernarg certifier and the recording-predicate inventory, the receipt reader, and the JIT receipt corpus (`railgun-jit-corpus build|check`).
  - `rdna-compute` gains offline-only entry points that the corpus needs: `kernel_registry::{default_route_entries, railgun_entries, corpus_entries}` and `KernelCompiler::{publish_package, toolchain_id, arch, jit_cache_key, jit_argv, rocm_env_root, compile_to_paths}`. `kernel_registry::entries` (the installer packs) is unchanged, and `railgun_copy.hip` is not in it.
  - The recording inventory was re-audited against beta. The 27 new recording-dependent sites now have rows, and six new decisions are recorded. One of them, the gfx1201 DFlash multi-layer GDN replay, checks only `!is_recording()`; it is `NotEquivalent`, so DFlash on gfx1201 is refused as a railgun default.
  - On beta, the modules `gemv_mq4g256` and `mq_rotate_x` compile to the same object. The corpus reader now accepts duplicate rows for one object when they carry the same receipt.
- **railgun runtime half, all developer-only and off by default (`railgun/m1p` rebased onto 0.4.1).** `rdna-compute`, and so the daemon, now links `railgun` and `railgun-corpus`; with no `HIPFIRE_RAILGUN_*` variable set every railgun hook is an empty `Option` and the replay route, kernels and kernarg bytes are unchanged.
  - M1 shadow (`HIPFIRE_RAILGUN_SHADOW`): railgun authors its program next to every Redline recording and diffs its gfx12 lowering, with per-arch pacing, against the prepared tape; nothing is submitted.
  - M2 (`HIPFIRE_RAILGUN_CHECK=sample(N)|always`, `HIPFIRE_RAILGUN_BACKEND=railgun`, `HIPFIRE_RAILGUN_DIGEST=1`): a check mode that runs a HIP twin of the prepared Qwen3.5 PM4 step over the `hip_bridge::registry` allocation registry, and a backend that executes railgun's lowering and fails closed to HIP. The registry records allocations only while check mode or the digest is configured.
  - G0 recording invariance: `bench_decode` accepts `"g0": "observe"|"record"`, and the launch funnels hand the eager arm's launches to the G0 observation. Also `railgun_g0_dflash_cycle`, `redline_greedy_trace` with `probe_refusal`, and `scripts/redline_daemon_harness.py --g0`.
  - `mc_g2` (MC G2 copies-only gate) builds with `--features mc-g2`, using the new `HipRuntime::memcpy_dtod_raw`.
- **gfx11 PM4 diagnostics, default off.** `HIPFIRE_REDLINE_GAP_TIMING=1` prints a per-token host breakdown of the plain-AR decode step. `redline_dispatch_profile` now runs on gfx11, with a timestamp after every dispatch and every compute-idle boundary. `HIPFIRE_REDLINE_IB_POOL=vmem` also applies on gfx11. `HIPFIRE_GFX1100_PM4_EXPERIMENTS=1` lets the gfx1151 initiator, interleave and resource-limit knobs apply to gfx1100. `railgun --features boundary-bench --example boundary_bench` is a gfx11 PM4 boundary-cost microbenchmark.
- **PM text front end and LDS bound check for the fused A4 epilogue:** `parse_line` reads `ds_swizzle_b32 … offset:swizzle(SWAP,n)` (`0x1F | n << 10`, pinned against `llvm-mc`), and `pm_check::lds_bounds` evaluates `v_xor_b32_e32` in the entry block.

### Internal & CI
- CI (from #774 by @fivetide): `hipfire-quantize`'s `anchored_index_range_and_no_affine_alias` drops a 2-bit `<= 3` assert that clippy 1.98 denies (`bad_bit_mask`) and that could never fail, and checks the exact output size instead; `peacemaker-ir`'s `global_atomic_return_form_is_the_glc_bit` resolves `llvm-mc` through `ROCM_PATH` and runs its encoding cross-check only where the toolchain exists, so it no longer panics on the no-GPU runner.
- **XDNA2 / AIE2P NPU stack folded in from the standalone NPU workspace (`c1bf57a`), experimental and dark.** Three pieces: `pm-npu` (AIE2P emitter: llvm-aie-derived VLIW encoder/scheduler, CDO/PDI and DPU-transaction writers, GEMM/IU4/IEF15/ring/experts kernel designs, plus the pure-CPU AIE2P simulator as `pm_npu::sim` with its suites `golden_bytes`, `lean`, `ring`, `experts`, `g80`, `iu4_ief15` as integration tests); `railgun::npu` behind the new, default-off `railgun` feature `npu` (direct amdxdna DRM-ioctl runtime: device, heap, hardware context, BOs, submit/chain, persistent ring, tape, FCLK guard, dlopen'd HIP interop); `npu-tools` (every NPU binary: `npu-tools`, `npu-gemm`, `npu-hello`, `npu-experts`, `npu-ring`, `npu-railgun`, `npu-bw`, `npu-coop`, and the `coop27` GPU+NPU co-op prototype behind `--features lab`). Nothing in the daemon, engine or loader depends on them; default `railgun` builds are unchanged. CI builds `railgun --features npu` and `npu-tools --features lab`. Docs: `docs/npu/` (silicon evidence in `docs/npu/report.md`); third-party derivations (llvm-aie, mlir-aie, aie-rt, amdxdna UAPI) recorded in NOTICE.
- **XDNA2 NPU co-op ring: the CPU-free GPU → NPU → GPU persistent ring runs 271 → 128 µs per run on gemm 512×1280×2560 (−53 %) and 164 → 113 µs on the 512×2560×640 down shape (#818 by @fivetide). Experimental and dark: no inference route changes.** Every file it touches is in `pm-npu`, `railgun::npu` (feature `npu`, default off), `npu-tools` (`npu-coop`, `tools/npu/npu-coop-bench.sh`) and their docs and crate maps. Nothing in `hipfire-daemon`, `hipfire-cli`, the engine, `rdna-compute`, any kernel, the registry or any `Cargo.toml`/`Cargo.lock` changes, and no default-path crate depends on `pm-npu` or `npu-tools`. What it adds: producer and consumer GPU streams, so the next sequence number is published while the NPU computes (`HipRuntime::stream`, `launch_on`, `record_on`); a lean-first ring re-arm (`ring::persistent_with`, `Bodies::LeanFirst`); an A-repeat V9 design (`ArrayDesign::with_a_repeat`); a G80 ring with a resident shared B (`append_lean_run_body_b_resident`); and `npu-coop` rounds queued one ahead. It also fixes two Halo bring-up problems: `NPU_IGPU_PCI` selects the iGPU PCI bus id (default `0000:bf:00.0`), and `Device::userptr_bo` falls back to the caller's mapping when Linux 7.2's amdxdna refuses to mmap a userptr BO (EINVAL). Measured on gfx1151 with the fabric clock pinned, 16 rounds, median of 3 processes, every run byte-exact against the eager design and the CPU reference; the simulator suite `ring` pins that lean, lean-first and B-resident ring state equals the full-body ring. `tools/npu/npu-coop-bench.sh` reproduces the numbers.
- Registry: the parked `qwen3.8:27b-mq4l*` tags are dropped (never published); `registry/pending/` is removed.
- The experimental MW16 GEMM route (`kernel.mw16` / `HIPFIRE_MW16`, default off) is removed together with its call-site predicates and tests; the live F16 mw16 kernels stay. Kernel packs are unchanged.
- **Perf: Qwen AR (and the generic AR and secondary AR loops) and the DFlash/MTP spec emitter append each token's bytes to one buffer instead of re-decoding the whole generated history every token.** The think-budget scans read that buffer too. `Tokenizer::decode_token_bytes_into` is the per-token decode; `decode_bytes` is built on it.
- **Perf: `process_value`/`developer_var` answer from a table rendered once from the active policy, not a scan of the whole config schema plus three allocations per call.** The DSpark drafter's Q8/HFQ4 GEMM knobs (`HIPFIRE_DSPARK_Q8_WMMA`, `_Q8_4W`, `_HFQ4_WMMA`) are read once per process instead of on every GEMM.
- Usage accounting is unchanged and now pinned by a test: Qwen AR/spec/batch count the EOS token in `completion_tokens`; DeepSeek V4 and LFM do not.
- Cleanup: one `qwen35_eos_filter_config` (was duplicated in AR and the spec emitter); one `hipfire_config::parse_tri_state` for the `dspark`/`dflash`/`mtp` `on`/`off`/`auto` modes; the multi-slot preflight uses `arch_label`; dead `release_held_finish_tool_calls` removed; the gateway builds `conversation_messages` only on the multi-slot route, returns the completion it already delivered at `commit_ready` instead of rebuilding it, drops the provisional request id, and writes SSE frames without an intermediate `String`; the slot `commit_ready` and spec windows avoid a clone each.
- **Tooling: `scripts/serve_harness.py` names the KV mode it actually certified.** Without `--kv` it already resolves the product default (`auto`, which is native fp8 on gfx1201; a registry `q8` is left to `auto`). It now prints `kv_mode_resolved=<mode> (observed in …)` in the report header after the serve warms and again at the end of the run, read from the loaded ACK or the loader's `KV cache: requested mode=…, effective KV=…` log line, and says `UNOBSERVED` when neither was seen (for example `--no-spawn`).
- **rdna-compute: the 70 raw `hip.launch_kernel` sites in `gemm.rs` and the 5 `dequantize_*_to_f16` launches in `dispatch.rs` now go through `launch_maybe_blob`.** Under hipGraph capture these launches handed HIP pointers to stack locals, so a captured graph kept dangling kernargs; under PM4 recording they were never recorded. They cover the HFQ3/HFQ4/HFQ6 qkv/qkvza/gate_up/residual batched fallbacks (gfx10, gfx906, gfx942 and shapes the WMMA/MMQ gates reject), the gfx10 MMQ tiles, the gfx942 MFMA residuals and the F16/F32/Q8 utility GEMMs. A new `launch_params_blob!` builds the blob from the same locals as the params array; debug builds assert the two agree byte for byte at every launch. No kernel selection changes.

## v0.4.0 — 2026-09-30
- **Qwen3.8-Flash-Next (Qwen4, arch 16): `auto` KV is fp8 QSA K/V on exact gfx1201; the GDN recurrent state is Q8 by default; `max_seq` goes up to the native 262,144.** The QSA K/V arenas store E4M3 codes with one f16 scale per head and token, and the indexer's raw and pooled keys are stored as BF16, which holds the same values as the F32 arenas. `auto` keeps the exact `bf16` state (F32 arenas) on gfx1100, gfx1151 (Halo) and every other arch. An explicit `fp8` is refused off gfx1201, and every other `kv_cache` value is refused. The GatedDeltaNet state uses Qwen3.5's Q8 DeltaNet format on every arch; the `state_quant: "fp32"` load parameter opts out. Past 15,360 pooled blocks, the QSA selector keeps its score rows in global memory.
  - **Validation (gfx1201, host-mapped experts N=12, tp=1).** There is no Flash-Next KLD reference yet (the MI300X requant is 0.4.1), so fp8 was validated by parity and greedy agreement, **not by KLD**. At 2K, 8K, 16K, 32K and 64K, every sampled QSA row matches the CPU F32-FMA reference in all three arms (fp8+Q8, bf16+Q8, bf16+fp32). The final-row argmax equals bf16 at every context, KL(bf16‖fp8) ≤ 5.1e-3, and top-5 overlap is 4 or 5 out of 5. Greedy text on the three flash prompts is coherent but not byte-identical to bf16 (first divergence after 40–442 characters). The serve battery loads with `auto` and serves 5 of 5 turns with 0 runaway and 0 empty turns.
- **Serve: multi-slot serve (`serve.multi_slot = true`) exits when it cannot load its model. It no longer reports ready and answers every request 500.** An H2 multi-slot serve started beside a second H2 serve on the same card ran out of VRAM building its slot engine (`SlotEngine spawn: pbs: HipError(2): hipMalloc: out of memory`). Serve logged "pre-warm failed … serving lazily" and `/health` stayed 200 `ok`, while every request reloaded the weights, hit the same OOM, and returned 500. The slot engine is multi-slot serve's only backend. A failed pre-warm, or a failed reload after a daemon respawn, now logs the load error and sets `/health` to 503 `unhealthy`. Serve then stops the daemon and exits 1, as it already does when the daemon cannot be respawned. Single-slot serve still serves lazily after a failed pre-warm.
- **Serve: a tool call cut off by `max_tokens` on DFlash or MTP ends with `finish_reason: "length"`, as it does on AR, instead of a non-retryable "malformed tool protocol" error.** The Qwen spec emitter reported an output that ended inside `<tool_call>` as `malformed_protocol`, and the spec terminal checked that before the length cap. It now reports `truncated_tool_call`, which yields to a pure length exit exactly like an open think span: the answer streamed so far, no tool call, no cache store and no state rollback. The model ending its turn inside the call still fails closed, and so does a call that goes malformed mid-stream (AR errors on the spot there). The multi-slot route keeps failing closed at any exit, as before.
- **Serve: experimental multi-slot engine (SCS stack #779–#792, by @ghazni101; opt-in `serve.multi_slot = true`, default off).** Several requests decode concurrently on one GPU.
  - Paged KV slot descriptors and a page pool (#779), and a runtime for prefix index, fairness, admission, host swap, the session table and a strict JSON-schema grammar matcher (#780).
  - A Qwen3.5 multi-slot engine with a scheduler, checkpointing, per-slot MTP/DFlash speculation and the vision ladder (#781).
  - The serve surface, with daemon slots, `hipfire serve` wiring, client streaming and loader admission (#782).
  - An evidence suite (#783), flat 16-bit KV tiers (bf16, and a new f16) on the slot engine (#791), and the generic scheduler moved into `hipfire-runtime` (#792).
  - `/health` gains `capabilities`. The engine's overload class maps to 503 + Retry-After, and request validation to 400.
  - On the multi-slot route only: `serve.stream_stall_timeout_ms`, `serve.max_queue_bytes`, strict `max_tokens` and `response_format` json_schema.
  - **The default route is unchanged.** `kv_slot_desc.h` is land's 24-byte descriptor. Every descriptor launch compiles a separately named `*_paged` module (`kv_slot_desc_paged.h`), which the installer registry packages. The `.text` of the 37 default modules whose source bytes changed is identical on gfx1100, gfx1151 and gfx1201. The land `http.rs` contract (CORS, `/health` token, 503 + Retry-After) is kept.
  - Evidence:
    - Single-slot MTP and AR greedy text equal land on all 8 workload genres, and code-edit τ is 2.81. H2 KLD pins are unchanged on gfx1201 (A4/fp8), XTX and Strix Halo.
    - gfx1201 decode at 512/8K/32K and pp8192 are within 0.1 % of land.
    - Multi-slot on gfx1201 returned 200 to every request at 2 and 4 slots, 1.41× and 1.45× faster than the same prompts run one at a time.
- **VMM KV: a released arena's virtual range is retired, never freed, so no later reservation or `hipMalloc` in the process can land on a VA the GPU has translated before.** On ROCm 10.0 (HIP 7.15), a VA can keep translating to its first backing after `hipMemUnmap` + `hipMemMap` of another handle, even after `hipDeviceSynchronize`. In the 80 × 2 MiB repro that is every page on gfx1201 and 84% on gfx1100; gfx1151 is unaffected. The exact trigger is not isolated. `VmmArena::release` used to call `hipMemAddressFree`, and the next hint-less `hipMemAddressReserve` returns exactly the freed range, so every unload → load in one daemon (model swap, serve idle eviction, `max_seq` change, failed-load retry, DeepSeek4 compressor caches, Glimmer and dense-TP KV) re-mapped VAs the previous model had used.
  - `release` still unmaps every segment and frees every physical handle; only the VA stays reserved. `hip_bridge::retired_va_bytes()` reports the total. Qwen3.8-27B at `max_seq` 4096 retires 192 MiB per load, about 8 GiB at 256K. Reservations are refused with a restart hint once 64 TiB is retired (half of the 128 TiB user VA space).
  - `map_next` poisons the arena after its access-reset cleanup, so a retry cannot map a new handle at the address it just unmapped.
  - The unused `HipRuntime::mem_address_free` binding is removed.
  - On the shipped 0.4.0 sequence the reuse did not show up as stale reads. A HIP replay of hip-bridge's reserve/grow/release (kernel writes and reads, 20 unload → load cycles per case, same size, different size, `hipFree` → VMM, fresh process) got the same VA back and read 0 stale pages. In one daemon, H2 greedy output stayed byte-identical to a fresh process across same-model, `max_seq` 8192 and Qwen3.5-0.8B → H2 reloads, and across two serve idle evictions, on gfx1201, gfx1100 and gfx1151. The fix removes the exposure rather than a reproduced failure. `freed_vmm_tensor_va_is_never_reserved_again` (GPU test) fails on the old code with an exact VA match and passes now.
- **gfx1201 Q8-KV prefill above 32K stays on the FA2 kernel (R9700 H2 prefill at 49K: 134.9 → 34.6 s; at 65K: 299.3 → 56.8 s; byte-identical at or below 32,768 context).** On exact gfx1201 with Q8 KV, every 512-row prefill segment whose context passed 32,768 lost the default flash route. Dispatch's gfx12 default envelope (`gfx12_query16_default_eligible`) ends at 32K, and the Q8 FA2 arm and launcher (`attention_q8_0_fa2_gqa_gfx1201`) refused anything longer. Those segments ran the tiled partials+reduce kernel in 16-row launches, each scanning the whole context: at 49K it took 113 s of the 135 s prefill.
  - The FA2 body never depended on the cap. It walks 64-key tiles up to `max(positions) + 1`, with 64-bit K/V row offsets and fixed 64 KiB LDS, so its admission (`Gpu::gfx12_q8_fa2_prefill_admitted`, shared by dispatch and the launcher) now runs to the model's 262,144 positions.
  - Above 32K the FA2 arm also takes batches of 64–512 rows that are not a multiple of 16, such as a chunk's tail. At or below 32K those still go to query16, and nothing else changes there.
  - Unchanged: fp8 KV (the gfx1201 `auto` default for single-GPU Qwen 27B), which was already admitted to 262,144; speculative verify and Redline recording; fwht3-K; and gfx11, whose FA2 ingresses are raised separately (see the gfx11 FA2 past-32K entry).
  - **hipGraph capture takes the FA2 arm at every context once warmed, so a captured Q8 prefill replays the eager call's bytes.** The arm used to refuse capture outright. That eager-only gate came from the research opt-in and is older than the launcher's replay-idempotent Q pre-convert and owned kernarg blobs. A captured call ran query16 at or below 32K and the tiled kernel above it, so the graph differed from the same eager call at every context. A captured call now takes FA2 when the launcher cannot allocate mid-capture (module loaded, f16 Q scratch already holds the batch); a cold capture keeps the incumbent. Serve prefill does not capture, so serve output does not change.
    - Capture stress through `attention_q8_0_flash_prefill_wmma` (H24/KV4/D256), 18 shapes: 8,192 and 16,384; ±1 at 32,768, 49,152 and 65,536; 32,832, 32,833, 33,000, 48,974 and 65,456; batches 64, 65, 333, 334, 511 and 512. Each shape gets one real hipGraph, replayed into poisoned output with guards. Eager goldens equal the previous build's. 0 mismatches in 1, 2 and 4 concurrent processes × 10,000 launches. The previous build's capture differs from eager on all 17 FA2 shapes.
  - Greedy text at 8K and 32K is byte-identical to the previous build. Above 32K the f16 FA2 body rounds differently from the f32 tiled kernel, so greedy text diverges early; on one of five long prompts it differs from the first token. Both versions are coherent, answer the prompt, and quote a 64K prompt's first file correctly. The gfx1201 A4 KLD pins are unchanged (WT2 `483cfc58…` 0.069571, code24 `6339dc9e…` 0.052119).
- **gfx1201 speculative-verify attention: VerifyAttn shares one K/V scan across the GQA group and the verify rows (byte-identical; opt-out `HIPFIRE_VERIFY_ATTN=0` / `kernel.verify_attn=false`).** DFlash, MTP and n-gram verify blocks of 1–32 rows used `attention_flash_{fp8_e4m3,q8_0}_tile_batched`. Under the default verify HipGraph that route runs at every context, and in eager mode above 4,096. It launched one wave per (q head, 128-key tile, row), so every K/V byte was read once per q head and row (96× at B = 16), and the fp8 tile decoded E4M3 in software per element.
  - `attention_verify_gqa_{fp8,q8}_gfx1201` runs one 256-thread workgroup per (kv head, 8 rows) and split. Each K/V slice is read once for 6 q heads × 8 rows and decoded with the hardware fp8 converter.
  - The grid is `[⌈rows/8⌉ · kv_heads, splits]`, and the tile range comes from `positions[]` on the device. The grid never depends on the live context, and under capture it no longer scales with `physical_cap`.
  - `attention_verify_reduce_gfx1201` is the parallel twin of the batched reduce.
  - **Identity.** Partials and output are byte-identical to the tile_batched + reduce pair, so the committed stream, τ and KLD do not change. Checked with a poisoned-guard oracle: 126 cases, fp8 and Q8, B 1–32, context 1–64K, eager, capture and sub-batched. Stress ran 200× serial, under a one-CU mask and as two concurrent processes.
  - gfx1151 and every other attention route are unchanged (gfx1100 has its own twins, next entry). Eager Q8 verify above 4K keeps the multi-row R4/R8 kernel on gfx1201.
- **gfx1100 speculative-verify attention: VerifyAttn twins of both gfx1100 verify routes (byte-identical; opt-out `HIPFIRE_VERIFY_ATTN=0` / `kernel.verify_attn=false`).** On the 7900 XTX, DFlash (B = 16) and MTP (B = 4) verify is eager: above 4,096 tokens it runs the multi-row R4/R8 tile (`attention_flash_q8_0_rows{8,4}_d8`), below it the batched `attention_flash_q8_0_tile_batched`.
  - `attention_verify_gqa_q8_rows_gfx1100` reproduces R4/R8's online exp2 softmax exactly (the running max as a lane scan of the associative "leftmost max", the sum and output recurrences serial in key order, R4/R8's empty records included); `attention_verify_gqa_q8_gfx1100` is the Stage-0 twin of the batched tile. Both read each K/V slice once for 6 q heads per row, one wave per query row, and dequantize Q8 exactly in 2.25 VALU per element (`v_perm` f16 codes + `v_fma_mix`). The split grid is fixed (4,608 active waves) and the tile range comes from `positions[]` on the device.
  - **Identity.** Partials and output are byte-identical to the R4/R8 and tile_batched + reduce pairs: poisoned-guard oracle 115/115 on the XTX (B 1–32, context 1–64K, eager, capture-sized and sub-batched), 60 s soak, 200× stress serial, under a one-CU mask and as two concurrent processes; XTX and Halo KLD pins unchanged; DFlash and MTP greedy text identical to the opt-out on every request.
  - **Per layer (H2, XTX).** R4/R8 route, B = 16: 0.6345 → 0.3285 ms at 8K, 0.9031 → 0.6020 ms at 16K, 1.6713 → 1.2747 ms at 32K; B = 4: 1.82×, 1.79×, 1.84×. tile_batched route: 2.4–3.9×.
  - **Serve (XTX, Q8 KV, greedy, 256 tokens, two fresh processes per arm).** DFlash decode 54.9 → 60.6 tok/s at 32K (+10.4 %), 61.9 → 66.2 at 16K, 85.9 → 89.2 at 8K, and 47.6 → 53.7 / 44.5 → 51.2 at 48K / 64K (one process); MTP 55.6 → 61.3 at 32K (+10.2 %), 64.9 → 68.2 at 16K. τ unchanged. Neither DFlash nor MTP crosses below AR (45.8 tok/s at 32K) within 2K–32K, before or after; above 32K, AR extrapolated, DFlash stays about 1.23× AR through 64K with the fix against about 1.08× without it.
  - gfx1201 and gfx1151 are unchanged.
- **gfx1151 speculative-verify attention: VerifyAttn splits the single-slot WMMA flash prefill into a context-parallel S launch and a per-dim-chunk softmax + P·V walk (byte-identical; opt-out `HIPFIRE_VERIFY_ATTN=0` / `kernel.verify_attn=false`).** On the Strix Halo every DFlash, MTP and n-gram verify (1–32 rows, H2 head_dim 256 / GQA 6, Q8 KV) and every short prefill tail runs `attention_q8_0_flash_prefill_wmma`. That kernel is one wave per (q head, 16 rows) walking the whole context, so a 32K verify ran 24 waves (48 above 16 rows), each through 2,048 serial 16-key steps.
  - `attention_verify_wmma_qk_gfx1151` computes S = Q·Kᵀ context-parallel. Each workgroup dequantizes a K tile once into LDS for all fragments of its kv head, runs the reference's 16-WMMA chain per fragment, and stores the reference's f16 S tile (scaled and causal-masked) in the flash partials.
  - `attention_verify_wmma_pv{1s_d4,2_d2}_gfx1151` then replay the reference's per-row online softmax and O = α·O + P·V, one 16-dim chunk (or two) per workgroup, with V dequantized once into LDS. (head, row) pairs are packed densely into ⌈6·rows/16⌉ WMMA fragments.
  - Both grids depend on the context only through `max_ctx_len`, and the live key range comes from `positions[]` on the device.
  - Every LDS barrier waits for the wave's own LDS stores (`s_waitcnt lgkmcnt(0)`). An LDS-only fence emits no such wait, and without it four concurrent processes lost K-tile stores (8 of 40,000 stressed launches).
  - **Identity.** Output is byte-identical to `attention_q8_0_flash_prefill_wmma`, so the committed stream, τ and KLD do not change. Poisoned-guard oracle 139/139 on the Halo (all walkers, both layouts, eager and capture, cap 262,144), 60 s soak, stress serial, under a one-CU mask and as 2 and 4 concurrent processes (0 of 40,000 launches at 4 processes, ragged live lengths included), 2- and 4-process soaks, and 2 stress processes alongside a live DFlash serve; Halo KLD pins unchanged; AR greedy text identical to the previous build at 2K, 8K and 32K; DFlash and MTP greedy text identical to the opt-out on every request.
  - **Per layer (H2, Halo, Q8 KV).** B = 16 (DFlash): 0.872 → 0.071 ms at 2K, 3.34 → 0.252 ms at 8K, 18.01 → 0.969 ms at 32K, 35.98 → 1.95 ms at 64K (12.3–18.6×). B = 4 (MTP): 0.815 → 0.049 ms at 2K, 17.58 → 0.648 ms at 32K (16.7–27.1× over 2K–64K). B = 1: 15.5–27.8×. Captured launches are within 4 % of eager.
  - **Serve (Halo, Q8 KV, greedy, 256 tokens, three fresh processes per arm; two for the MTP opt-out).** DFlash decode 9.6 → 27.0 tok/s at 32K (2.81×) and 27.4 → 31.0 at 2K (+13.1 %); MTP 6.6 → 20.4 at 32K (3.09×) and 21.5 → 24.7 at 2K (+14.9 %). τ unchanged (DFlash 3.25 / 3.11, MTP 1.97 / 1.68 at 2K / 32K). AR decodes 15.0 tok/s at 2K and 13.8 at 32K (128 tokens): without the fix DFlash falls below AR near 24K and MTP near 16K (linear between 2K and 32K); with it neither does within 2K–32K, and DFlash holds 1.96× AR at 32K.
  - gfx1201, gfx1100 and the Qwen3.8 Flash-Next attention (QSA) are unchanged.
- **Kernel-PR fold (#788, #756, #794), exact-arch scoped.**
  - #788 (HUSRCF): on exact gfx1100 through 8K (Q8 decode tile32, batched tile128) the shared flash-partials buffer holds max(one decode row at tile32, 32 batched rows at tile128), 50.7 MB instead of 101.4 MB for Qwen3.8-27B. Every other arch keeps the legacy sizing byte for byte. Batched launchers sub-batch by capacity, so only launch counts change past 32 rows.
  - #756 (HUSRCF): exact gfx1151 may take the Q8 R4/R8 multi-row attend, opt-in only: set `HIPFIRE_FA_PERTOKEN_MIN_CTX` (unset leaves gfx1151 on the batched route; gfx1100/gfx1201 keep the 4096 default). Unreachable under the default verify graph (capture).
  - #794 (HUSRCF): opt-in gfx1100 packed MQ4 FFN prefill (`HIPFIRE_GFX1100_PACKED_MQ4_PREFILL=1`) and widened native MQ4 prefill (`HIPFIRE_GFX1100_MQ4_WIDE_PREFILL=1`), uniform MQ4G256 (v1) models only. Default off. MQ4V2 (qt44, including `qwen3.8:27b-mq4-xts`) keeps the landed builder V2C / IU4 routes even with the env set: packed was not shown faster there. Binary evidence archives are not carried.
- **0.4.0 deprecation pass: these surfaces are deprecated and will be removed in 0.5.0.** They still work in 0.4.0. Each one prints one warning line when used, except where noted. Loads that use none of them behave byte-for-byte as before (H2 greedy is byte-identical to `86f3a6c76`). Each entry point carries `// lifecycle: deprecated since 0.4.0, removal 0.5.0 — <reason>`.
  - **PFlash** (`crates/hipfire-pflash`, `speculation.prefill.*`, `diagnostic.pflash.score_layer`, `HIPFIRE_PFLASH_*`; prefix caching supersedes it). A load with `prefill_compression` other than `off` prints `[hipfire-daemon] warning: PFlash is deprecated and will be removed in 0.5.0; …`. `dflash_spec_demo --pflash`, `scripts/pflash-gate.sh` and `scripts/pflash-niah-bench.sh` warn too. The config help and TUI knobs say "Deprecated PFlash (removal in 0.5.0)".
  - **Givens asym KV** (`legacy-asym2|3|4`) and the **`asymN` / `turbo*` KV aliases** (fwht3 supersedes them; spell `fwhtN` or `q8`). A load whose `kv_mode` (or `HIPFIRE_KV_MODE`) or `kv_k` names one prints `[hipfire-daemon] warning: KV name '<name>' is deprecated and will be removed in 0.5.0 (…)`. The resolved KV mode is unchanged.
  - **Legacy contiguous DFlash knobs** `HIPFIRE_DFLASH_WINDOW=0` and `HIPFIRE_DFLASH_CTX_CAP`: setting either one warns at DFlash load. The implicit fallback for drafts that declare no window is unchanged.
  - **Retired coherence-gate batteries** (`scripts/coherence-gate-*.sh`, `scripts/_coherence_runner.py`, `scripts/awq_coherence_check.sh`): each one warns on start.
  - **GGUF weight input to `hipfire-quantize` / `hipfire quantize`**: GGUF→mqN is lossy double quantization; use llama.cpp for GGUF. It warns once. `--imatrix <file.gguf>` is unaffected.
  - **MQ4R / DS4 MQ2R route selection by file extension**: marked, documented and allowlisted, with **no runtime warning**, because the extension is still the only way to select the route. 0.5.0 selects by HFQ metadata.
  - New `scripts/check-lifecycle.py` (run by `scripts/no-gpu-ci.sh`). It checks two things:
    - (a) Default configs never reach deprecated code: deprecated gate keys keep inert defaults, `auto` KV never resolves to Givens asym, and registry cards, `docs/configs/*.toml` and CI entry points never opt in. Only the extension-based route selection is allowlisted.
    - (b) Every config key (`docs/CONFIG.md` § Lifecycle status) and every env var (the `docs/env-vars.md` generated inventory, now with a Lifecycle column) carries a lifecycle status (`stable` / `experimental` / `developer` / `harness` / `deprecated`).
    - `--write` regenerates both tables.
- **CASK / TriAttention KV eviction is deprecated and will be removed in 0.5.0.** It still loads in 0.4.0, but it is not supported, recommended or an acceptance route. Loads that do not use it behave as before; the only change they can see is the wording of one DFlash context error (below).
  - A load that sets `cask_sidecar` (`memory.cask.sidecar`, `HIPFIRE_CASK_SIDECAR`, or `cask_auto_attach` finding a sibling sidecar) or `cask=true` prints one line: `[hipfire-daemon] warning: CASK is deprecated and will be removed in 0.5.0; not supported (…)`, then behaves as before. `hipfire sidecar-gen` and the `dflash_spec_demo --cask*` flags print the same warning.
  - The `memory.cask.*` config keys are marked experimental, and their help text, the `hipfire sidecar-gen` help, and the TUI `cask` knob say "deprecated, removal in 0.5.0". Each CASK entry point carries `// lifecycle: deprecated since 0.4.0, removal 0.5.0`. GLOSSARY lists CASK and TriAttention as `legacy`, deprecated since 0.4.0.
  - README, GETTING_STARTED, QUANTIZE, CONFIG, CLI, MODELS and env-vars no longer present CASK as a long-context or post-quantize route. AGENTS.md drops the CASK pitfall row and the (nonexistent) `hipfire bench --cask-*` flag rows for one deprecation note.
  - A failed sidecar load no longer suggests `HIPFIRE_CASK_OFF=1`, which nothing reads; it says to clear `memory.cask.sidecar`. The `HIPFIRE_FORCE_A3B_EVICTION` row, also unread, is gone from env-vars. The DFlash `prompt+max_tokens exceeds ctx_capacity` error now says "raise max_seq or shorten the request" instead of "enable cask_sidecar for long decode".
  - `hipfire pull qwen3.6:27b` still fetches that tag's 2.4 MB `triattn` sidecar, and nothing attaches it by default.
- **gfx1201 Qwen3.5-family decode: the lowered decode is byte-identical to the hand decode again (H2 `qwen3.8:27b-mq4-xts`, fp8 and Q8 KV).** Default decode output on gfx1201 moves by ULPs; it now equals `HIPFIRE_FORWARD_LOWERED=0`, which is unchanged.
  - **Cause.** The lowered decode runs `conv1d_silu_split_qknorm`, a decode-only fusion of `conv1d_silu_split_f32` and `fused_qk_l2_norm_scale_f32`; the hand decode runs that unfused pair. Both leave FMA contraction to the compiler, and on gfx1201 it contracted the same sums differently in the fused kernel:
    - The conv tap sum rounded `w3·x` and fused `w2·s0`; the reference does the opposite.
    - The Q/K square sum fused the lane's first element onto its second; the reference does the opposite.
    - Every lowered step therefore differed from the hand step from layer 0 on: logits, DeltaNet state, and the KV of later layers.
    - This is also the source of the DFlash exception above, where the target prompt ends in a one-token chunk.
  - **Fix.** On gfx1201 only (`__gfx1201__`), the fused kernel spells out the reference kernels' order with `__builtin_fmaf` and contraction off. VALU count and register use are unchanged.
    - gfx1100 and gfx1151 code objects are byte-identical to before. Their fused kernel has the same contraction mismatch, so their lowered decode is expected to differ from the hand decode too. This is inferred from the disassembly and has not been run.
  - **Identity** (R9700, H2, 32 greedy decode steps after each prompt). Lowered equals hand in every step's logits, the DeltaNet state and the KV at 19 prompt lengths, with fp8 and with Q8 KV. The lengths cover every final-chunk class of the 8,192-row prefill plan: 1, 2, 3, 17, 255–257, 511–513, 1,000, 2,049, 4,097, 8,191–8,194, 8,705 and 16,385.
    - Before the fix, all 19 lengths differed. At 16,385 tokens the greedy text diverged at decode token 11.
    - The gfx1201 KLD pins are unchanged, because prefill does not use this kernel.
  - `crates/rdna-compute/tests/conv_qknorm_parity.rs` (GPU, `--ignored`) compares the fused kernel with the unfused pair bit for bit, with both direct launches and hipGraph replay. It covers the 27B and A3B head counts and one head either side of each.
- **Prebuilt kernel packs: a release-tag install on an admitted GPU installs the registry kernels without compiling them.** Each tag can carry `hipfire-kernels-<tag>-<arch>.tar.gz` (+ `.sha256`, `.manifest.json`) for gfx1201, gfx1100, gfx1151, gfx906 and gfx942: the `kernels/compiled/<arch>` tree of that commit's exact registry plus a manifest with the commit, ROCm and HIP version, compiler identity, code-object version and admitted ROCm range. On gfx1201 an installed pack is byte-identical to the local `daemon --precompile` output (all 396 files).
  - **Coverage limit.** A pack holds only the registry. Kernels outside it still JIT on first use, so they still need hipcc. The registry carries every module the traced Qwen3.8 H2 runs (`hipfire run`/`serve`) JIT-compiled, each with sources byte-identical to the runtime's: on gfx1201 the 6 decode modules plus 30 prefill, MTP and DFlash modules (GDN chunk-scan prefill, qresident FA2, i4 slab producers, MTP/DFlash sampling and replay; 132 modules, up from 96), on gfx1100 64 more (126) and on gfx1151 56 more (115). On gfx1201 a pack-only install without a device compiler (`HIPFIRE_NO_DEVICE_COMPILER=1`, runtime-only ROCm root) runs H2 greedy AR, MTP and DFlash `hipfire run --json` (short and ~7 KB prompts, up to 128 tokens, reasoning off) with zero JIT compiles and text identical to a `--compile-kernels` install. gfx1100/gfx1151 coverage comes from static H2 serve traces and has not been run pack-only; routes outside the traces may still JIT.
  - **Build.** `scripts/build-kernel-pack.sh --tag T [arch …]` compiles `git archive` of the tag's commit with a fresh `HOME` and no `HIPFIRE_*` feature overrides (hipcc, no GPU), re-verifies every index through the new `hipfire-kernel-manifest`, and writes reproducible tarballs. The tag-triggered `.github/workflows/release.yml` runs it in `rocm/dev-ubuntu-26.04:10.0.0-full` (`HIPFIRE_ROCM_IMAGE` overrides), which carries the same ROCm 10.0.0 packages (HIP 7.15.26333) as the project's GPU hosts, so its packs admit ROCm [10.0, 11.0) and install there; a workflow-step run of that image produced all 15 assets byte-identical to a host build. `hipfire-kernel-manifest` falls back to the resolver's ROCm version when TheRock's `/opt/rocm` has no `.info/version`. The script plus `gh release upload` is the local alternative.
  - **Install.** `install.sh --tag` / a tag-named `--ref` (so also `hipfire update --tag`) and `install.ps1 -Tag` download the pack for the detected arch and install it only if the SHA-256 matches, the manifest names the checkout's commit, arch and cache ABI, the local ROCm is inside the pack's range, a local hipcc (if any) is the compiler that built it, the config selects the same kernel flags, and every `.index.json` matches the checked-out sources and object bytes with no stray files. Anything else prints the reason and compiles locally, as does `--compile-kernels` / `-CompileKernels`. `--kernel-pack-url` / `-KernelPackUrl` read the assets from another URL, `file://` URL or directory. Without a device compiler, `hipfire setup` accepts a runtime-only ROCm root and fails only if the pack is unusable. `hipfire kernel-pack install` exposes the same path. `install.json` records the installed pack.
  - **Runtime.** Unchanged behaviour: a pack index that does not match this build (any identity field, symbol, source, flags, ABI, object bytes, or a toolchain other than the local hipcc) is rejected, and recompiled when hipcc exists. New tests pin the foreign-toolchain rebuild and the remaining index fields; the installer's whole-pack check and the runtime lookup share one per-object check.
- **Qwen3.8-Flash-Next (qwen4): greedy serve requests now actually run native MTP.** The route selector sent a greedy Qwen4 request to AR whenever `top_p`, `top_k` or `min_p` was on the daemon wire, and `hipfire serve` forwards `top_p`/`top_k` whenever the client or the registry sets them (`qwen3.8:flash-next` sets both). The loader still printed `qwen4 native MTP speculator enabled`, but every such request decoded on AR: no `tau`/`mtp_windows`, AR's decode tok/s, AR's text. Greedy argmax ignores those fields, so now only non-neutral repeat/presence/frequency penalties, adaptive KV and the force-AR switches keep a greedy request on AR.
  - **MTP stays default-on only where the experts are device-resident.** On the Strix Halo, greedy MTP text equals AR on 8/8 prompts at a 1.41× median decode. With host-mapped routed experts (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, the discrete-card path), gfx1201 at N=12 gained only 1.08× (slower on low-τ prompts), and greedy MTP text differed from AR on 4/8 prompts. Those differences are deterministic near-tie alternatives, present before this fix. So an `auto` load with host-mapped experts keeps AR and prints `qwen4 native MTP: off by default with host-mapped experts`. `--spec mtp` (`speculation.mtp = "on"`) still attaches the head there.
- **Qwen4 on gfx1201: opt-in WMMA route for the HC GEMMs and the BF16 projections (`HIPFIRE_QWEN4_F16_WMMA_GFX1201=1`, default off).** `qwen4_f16_wmma_applies` required the RDNA3-only `has_wmma_w32`, so every Qwen4 BF16 projection at ≥512 rows (HC `input_mix_down`, PLE `key_proj`/`value_proj`, Flash-Next router/shared expert) and the HC read tail runs SIMT kernels on the R9700. With the opt-in, gfx1201 runs them on gfx12 WMMA: a split-K LDS GEMM (`gemm_f16_x_f16_wmma_lds_splitk.hip`, deterministic K split) over the existing F16 weight shadow, and a gfx12 port of the BF16 WMMA HC read (`hyper_read_up_wmma.gfx1201.hip`). Kernel times on the R9700 (event-timed, multirow/SIMT in brackets): HC down 320×20480×1536 0.340 ms (4.78), 320×10240×1900 0.217 ms (1.15); HC read 5120×1536 0.552 ms (2.35); PLE 20480×2560×1536 2.09 ms (64.1). Not bit-exact: F16 operands and WMMA summation order (projection error vs an F64 reference 3.7e-6 of max |y| at K 20480, multirow 1.6e-7; the HC read differs from the SIMT read by at most one BF16 step). Projections shorter than 16 rows keep the multirow kernel. gfx1100 and gfx1151 compile and dispatch exactly as before (on the Strix Halo no split-K tile beat the existing gfx11 64×64 tile at the HC down shapes: 320×10240×1536 0.469 ms vs 0.54–0.96× for the candidates). Captured and recorded forwards never take the route (`qwen4_f16_wmma_applies` refuses under hipGraph capture / Redline recording).
  - **Default off until a Flash-Next KLD check (planned for 0.4.1).** The route changes Flash-Next logits: teacher-forced against the default route over 3 real prompts × 256 greedy tokens, top-1 agreement is 98.8 %, 99.2 % and 98.8 % (mean |Δlogit| 0.167 / 0.043 / 0.058), and greedy text diverges at tokens 14, 128 and 68. Flash-Next has no KLD reference yet, so the default stays on the multirow/SIMT arms. With the knob unset, Flash-Next greedy decode on gfx1201 (3 prompts × 256 tokens, every step's logits) is bitwise equal to the pre-flip build's `=0` route.
  - **Opt-in speed on an R9700** (same binary, fresh process per pass, knob `0` vs `1`): Flash-Next prefill of a 1,870-token prompt 724.6 → 801.8 tok/s (+10.7 %, median of 3 processes per arm) and of a 6,794-token prompt 769.1 → 885.2 tok/s (+15.1 %; 3 processes vs 2, the third opt-in process timed out during a cold model load); decode unchanged (27.1 tok/s at 6,794). Synthetic dense Qwen4-27B pp8192: 631.3 → 1,043.0 tok/s (+65.2 %, median of 3 processes per arm, ABBAAB order); tg1 at 8,192 unchanged (22.9–23.0 tok/s). Durability with the opt-in: 2- and 4-process stress over ragged shapes (tile edges ±1, K split edges), with a hipGraph-capture leg, 40 iterations: 0 outputs differ bitwise between iterations, 0 outside the multirow reference tolerance, 0 captured-fallback mismatches.
- **Qwen4 with host-mapped experts: host-memory pressure no longer stalls the GPU (ROCm/rocm-systems#12528); a load that would pass TTM's GTT cap is refused up front.** Under ROCm's defaults the 46 GB of host-mapped routed experts (`hipHostMalloc`) and the source of every large pageable `hipMemcpy` (weight uploads from the mapped model file, the per-forward PLE rows) are KFD userptr BOs. When the kernel reclaims or migrates their pages, KFD evicts every queue of the process until it can restore them. The GPU sits idle while HIP busy-waits on the copy signal, with one core at 100 % and no error. Live on an R9700 at N=12 during a 262K load: `hipMemcpy` ← `memcpy_htod` ← `weight_store::upload_pooled_bytes`, GPU 3 % busy, swap full.
  - **Fix.** In a process that sets `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, `HipRuntime::load` sets two switches before the runtime loads, unless the operator already set them. `HSA_USERPTR_FOR_PAGED_MEM=0` makes `hipHostMalloc` memory GTT, which is outside reclaim. `GPU_PINNED_MIN_XFER_SIZE=100000` sends every pageable copy through clr's staging buffer instead of pinning its source. The opt-out is to set either switch explicitly; `HSA_USERPTR_FOR_PAGED_MEM=1` restores pageable host experts.
  - **Scope.** Other processes keep ROCm's defaults. Set globally, the two switches slowed H2's weight sweep on gfx1201 from 1.00–1.01 s to 1.20–1.22 s (3 fresh processes per arm against land). Its pp8192 (5,126–5,153 tok/s), tg128 (40.32–40.34 tok/s) and greedy text did not change. Scoped, H2 loads with ROCm's defaults: sweep 1.14–1.16 s against land's 1.14–1.18 s in the same batch, and greedy AR/MTP/DFlash text 33/33 identical to the protected baseline.
  - **GTT cap.** TTM's `pages_limit` (half of RAM by default; 61.4 GiB on a 128 GB host) caps GTT. The expert placement now refuses before allocating when the host-mapped experts exceed that cap minus the GTT every amdgpu device already holds. The error names `/sys/module/ttm/parameters/pages_limit` and the ways out: keep more layers in VRAM, raise `ttm.pages_limit`, or opt out. For example, with one N=12 serve running, the check read 15.2 GiB of GTT left from live sysfs and refused a second N=12 placement (46.1 GiB). The host-mapped allocation error also names the cap. An N=12 load fits a 128 GB host. It does not fit a 64 GB host at the default limit.
  - **Load under a memcg squeeze** (R9700, N=12, ctx 2048). Default switches at `MemoryHigh` 54 G: KFD evicted the queues for 14.9 s before the first forward and 17.7 s by the end, and the load took 43 s. With both switches at `MemoryHigh` 7 G: 0 ms evicted, a 29 s load, and a 2,048-token prefill in 3.5 s instead of 6.3 s. The rows and logits were identical.
  - **Long runs** (R9700, N=12, both switches, a build that admits 262,144 tokens). A 30-minute parity ladder to 262,080 tokens in one process had 0 ms of KFD eviction (358 samples at 5 s), with a 20 s load. A `hipfire serve` needle at 260,131 prompt tokens was answered correctly.
- **Qwen4 with host-mapped experts: a reload after an earlier GTT-mode Flash-Next process exits is no longer refused because of amdgpu's TTM page pool.** The exited process's GTT pages stay in TTM's page pool (up to `page_pool_size`, half of RAM), which `MemAvailable` does not count, although the next GTT allocation takes its pages from the pool first and memory pressure shrinks it. The host-RAM check now adds an estimate of the pool: host memory outside every `/proc/meminfo` counter, minus the GTT amdgpu devices hold and 4 GiB for other drivers (2.76 GiB measured with the pool empty), capped at `page_pool_size`. The pool itself is readable only by root (`/sys/kernel/debug/ttm/page_pool`). The GTT-cap check is unchanged. On an R9700 after an N=4 load left 14,720,592 pool pages, the base build refused an N=12 load ("MemAvailable is 45.9 GiB"); the fix admitted it (MemAvailable 46.1 GiB + 55.3 GiB estimated), drained the pool to 3,129,272 pages while loading, and produced the same greedy text. With a 66.5 GiB anonymous hog resident, it still refused (MemAvailable 17.4 GiB + 16.1 GiB estimated). `echo 2 | sudo tee /proc/sys/vm/drop_caches` empties the pool at once (docs/env-vars.md).
- **Qwen3.5/3.6/3.8 MTP: the prompt fill takes AR's prefill route and pairs each token with the previous position's hidden; MTP + ngram-mod keeps native MTP after an n-gram hit.** Greedy MTP text can change (details below); AR is byte-identical to beta.
  - **Prompt fill route (opt-out `HIPFIRE_MTP_OWN_PREFILL=1`).** The fill used to prefill the trunk in `prefill_max_batch`-row chunks (512 on gfx1201 FP8) through a hidden-capturing forward that ran as a speculative verify. That gave it the sequential GDN recurrence instead of the chunk scan, the `SpeculativeVerify` dispatch workload (no gfx1201 query16 flash prefill on Q8 KV; on gfx1100 the batched attention variant and no wide Q8 FA2), and no widened chunk. The trunk now follows AR's serve plan: `ordinary_prefill_chunk_limit` + `ordinary_serve_prefill_chunk_len`, with the same widened admission, 512-row commit stride and singleton rule. Its hidden rows reach the head through the new `forward_prefill_batch_capture_hidden`, whose capture selects the same kernels as a plain AR prefill. After a cold prompt prefill on H2 (gfx1201), the trunk's DeltaNet state, KV prefix and last-token logits are byte-identical to AR's at 40 to 9,000 prompt tokens, on Q8 and FP8 KV. Every other hidden capture (DFlash, MTP verify, MTP probe, independent batch decode) keeps its kernels. Prefix-cache DeltaNet checkpoints during the fill now land at trunk chunk ends, as AR's do. The forced advance and the terminal-repair replay use the same route.
  - **Head pairing.** The fill wrote head slot p from (token p, hidden p). Decode, and the pairing the head was trained on (vLLM's MTP proposer shifts tokens left by one against the target hidden), use (token p, hidden p-1). Row 0 of a fill takes the committed predecessor hidden when `prev_hidden` holds it; otherwise it gets a zero hidden. The head's top-1 on the next prompt token rises from 0.544 to 0.615 (codeedit prompt).
  - **MTP + ngram-mod (`HIPFIRE_MTP_NGRAM=1`, greedy, thinking off).** A takeover window now writes the head KV for exactly the rows the trunk kept (the seed plus the accepted drafts), with the same (token p, hidden p-1) pairing, in one batched KV-only head pass. On gfx1201 / H2 its rows are byte-identical to the per-row decode forward (`tests/mtp_takeover_fill.rs`). Before, the first accepted n-gram window retired native MTP for the rest of the request, and every later pool miss ran as a one-token (k=0) verify. The composition also arms through `hipfire serve` again: it gates on resolved thinking-off, not on the legacy `max_think_tokens == 1` sentinel that serve never sends. The wire fields `mtp_retired` and `ar_windows` remain, now always `false` and `0`.
  - **Serve on H2 + the qwen3.8 head** (R9700, fp8 KV, greedy; four fresh processes per arm, ABBA + BAAB, against beta MTP). MTP decode rises on 10 of 11 genres: codeedit 39.3 → 68.1 tok/s (τ 1.21 → 2.81, same text), tool_call 54.0 → 82.4, tool_answer 38.8 → 48.2, summarize 44.6 → 50.0, longctx 43.9 → 48.1, reason_think 67.5 → 71.2, codegen 64.0 → 66.2. MTP TTFT falls on all 6 serve genres with prompts of 246 tokens or more: summarize (1,791 tokens) 0.736 → 0.599 s, and longctx (6,414 tokens) 2.335 → 1.841 s. The 8-turn session goes from 43.2 to 45.9 tok/s.
    - The exception is the 65-token prose prompt: 47.7 → 44.2 tok/s (τ 1.22 → 1.07), with a different greedy text; the 4-turn chat chain moves 47.8 → 46.5. Where the text is the same in both builds, land is neutral to faster: the `serve_harness` greedy battery is ≥ beta on 5/5 prompts, and 16 short prose/chat prompts run 50.15 vs 49.65 tok/s pooled.
  - **Greedy MTP now matches AR byte for byte on 5 of 8 serve genres, up from 2** (codegen, codeedit, tool_call, tool_answer, reason_think). Prose, summarize and longctx still differ from AR, at tokens 332, 174 and 137, where the batched K+1 verify and its lm_head round differently from single-token decode.
- **Qwen3.5 DFlash: the target's prompt prefill runs on AR's route (necessary and proper; opt-out `HIPFIRE_DFLASH_LEGACY_PREFILL=1`; R9700 serve TTFT −7 to −9 % from 2K to 18K prompt tokens).** DFlash seeded the target in fixed 256-row chunks. The hidden-state ring kept every chunk off the widened chunk and the GDN chunk scan, and each chunk allocated a fresh prefill scratch.
  - The seed now plans its chunks exactly as AR's ordinary prefill does (`ordinary_prefill_chunk_limit`, then `ordinary_serve_prefill_chunk_len`). A ring-only forward (no per-token hidden, tape or tree) takes AR's route when eager.
  - Chunks wider than the 256-row ring staging write their hidden rows straight to the ring at its head. Verify, graph-captured and Redline-recorded forwards keep the staged ring unchanged.
  - An outer chunk wider than the hidden ring runs as ring-sized pieces cut on 512-row commits. The registry DFlash2 ring is its 2,048-row window.
  - Chunks that do not reuse AR's retained widened scratch share one seed-owned prefill scratch.
  - **Identity.** After the prompt, the target's last-token logits, KV and DeltaNet state are byte-identical to AR's prefill of the same prompt, with the production 2,048-row ring:
    - R9700 (gfx1201) at 512/2,048/6,400/16,384 tokens, Q8 and fp8 KV, cold and prompt-cache suffix;
    - 7900 XTX (gfx1100) and Strix Halo (gfx1151) at 2,048/6,400 tokens, Q8 KV.
    - The hidden rows handed to the draft are the same whether written directly, staged, wrapped or in one context-sized ring.
    - The exception is a prompt whose AR plan ends in a one-token chunk (for example 8,193 tokens at the 8,192 ceiling, or a 1-token prompt). The seed runs that token through the per-token hidden-extracting forward, as before. AR's decode forward takes the lowered pipeline, which on gfx1201 is not byte-identical to that path.
  - **Behaviour change.** Prompts and prompt-cache suffixes longer than 256 tokens now prefill differently from the previous DFlash, so DFlash's committed text can differ from it, never from AR's prefill. Up to 256 tokens the two routes are byte-identical. On the E1 DFlash fixture (27- and 61-token prompts) the committed tokens and τ are unchanged.
  - **Opt-out.** It keeps the 256-row staged route and reuses the shared scratch. Its target state and hidden rows are byte-identical to the previous DFlash.
  - **Measured on an R9700** (4 fresh processes per arm, ABBA then BAAB, W = 2,048):
    - `hipfire serve --dflash on` prefill at 2,553 / 7,861 / 18,142 prompt tokens: 1,040.9 → 967.5 ms, 3,469.8 → 3,151.0 ms, 9,740.1 → 8,844.5 ms.
    - The seed alone: 731.5 → 652.4 ms at 2,048 tokens and 8,135.1 → 7,323.8 ms at 16,384 tokens.
  - The gfx1100 and gfx1151 KLD pins are unchanged.
- **Qwen3.8-Flash-Next (`qwen3.8:flash-next`, arch 16 / `qwen4`) is supported, and sealed MoE runs through shared dispatch steps** (folds #774 by @fivetide up to `071ec2452`, which contains all of #755).
  - Flash-Next is a 48-layer hybrid model: 3:1 GatedDeltaNet / QSA sparse full attention, 4-stream hyper-connections, a 54 GB per-layer n-gram embedding table streamed from the mapped file, a 512-expert top-10 MoE with a shared expert, and one native MTP layer. `qwen3.8:flash-next` pulls `qwen3.8-flash-next.mq4` (125.3 GB, sha256 `8aa01cf4…f650`).
  - Loads are single-device, text-only, legacy KV, `max_seq` up to 61,440. By default the routed experts (~64 GB) stay in VRAM, which needs a ≥96 GiB device (Strix Halo gfx1151), and native greedy MTP attaches. On 24/32 GB discrete cards, `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N|auto` keeps the experts of layers 0..N in VRAM and reads the rest from pinned host RAM over PCIe. A gfx1201 load at N=12 pins 46.1 GiB of host RAM. MTP is opt-in there (see the MTP serve entry). The prefill/decode kernels are tuned on gfx1151; other GPUs take portable kernels.
  - **`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=auto` with `--spec mtp` no longer runs out of VRAM at MTP attach.** The placement reserved 6.5 GiB, sized without MTP, and chose N=16. The head's lm_head requant then failed `hipMalloc` with 1,658 MB free: it needs a 2.54 GB F32 scratch, which the GPU pool kept afterwards. That scratch now goes back to the device when the requant returns. When the head stays attached, `auto` also reserves its bytes: what it keeps, plus the larger of that scratch and the first speculative request's verify rows and `K+1`-row GDN rings. On a gfx1201 R9700 with host-mapped experts at `max_seq` 32,768, `auto` places 16 layers by default (MTP off) and 14 with `--spec mtp`. That reserve is re-measured at 32,768 tokens: a full 32,700-token prefill plus decode at N=16 peaked 457 MiB under it. The trunk QSA arenas' growth is added only past 32,768 tokens. When the free VRAM cannot hold the non-expert weights plus the reserve even at N=0 (for example beside another process's model), the load is refused up front instead of failing at the first request.
  - Qwen3.5/3.6 and Cohere2 MoE sequencing now lowers to sealed dispatch step programs, with one root-routed EP schedule for decode, prefill and batch (#755).
  - Not folded: the Qwen3.5 declarative-program port (`ff892ed98..b6dff816c`). Flash-Next does not use it, it changes no speed, and it re-homes the landed per-arch Qwen3.5 prefill/decode routes.
  - Shipped kernels stay as they were: #774's new kernels live in their own `qwen4_*` modules (and `gemv_mq4g256v2_xbatch`), so every kernel source that existed before compiles to the previous release's text off gfx1151. The packaged code objects, with their `.hash`/`.index.json`, are byte-identical on gfx1100, gfx1151 and gfx1201, and so are the JIT sources and cache keys of `bf16_round_trip`, `gemm_f16_x_f16_wmma_lds256`, `gemm_mq4g256v2_moe_grouped_wmma_k2`, `gemm_mqv2_wmma_gfx11_bt`, `gemv_bf16_xf32`, `gemv_mq4g256v2_moe_gate_up_k8_indexed_batched`, `gemv_mq6g256v2`, `hc_streams_init_from_embed_batched` and `moe_scatter_fused_k8` on gfx1100/gfx1201. On exact gfx1151 those modules compile #774's copy (`qwen4_<name>.hip`): the LDS-only WLDS barrier, the paired-tile grouped MoE gate/up, the split-decode MQ{2,3,5,6}V2 residual BT, the unrolled `gemv_bf16_xf32`, the MQ4V2 MoE gate/up K=2560 fast path and the block-scan `moe_scatter_fused_k8`. `gemv_q8_0` keeps its original body everywhere.
  - The multi-turn prompt cache stays on for Qwen3.5/3.8 MTP (the PR had turned it off for every speculator named `mtp`; only native Qwen4 MTP replays cold). The PARO A3B sidecar-shape load failure is fixed (`f82b5eb76`).
- **Redline: the recorded-HIP shadow oracle applies declared kernarg bindings.** `Gpu::replay_recorded_hip_prefix_at` substituted position only through the synthesized bindings and the GDN frame. Every field a Flash-Next launch declares (QSA block counts and start positions, the GDN convolution ring cursor, the index-key row offset) therefore replayed at its capture-time value, and past the capture position the oracle Redline checks PM4 against was stale. The PM4 plan and the oracle now take their bindings from one builder (`replay::retained_kernarg_bindings`). The Qwen4 shadow compares the oracle at every window, not only the capture position, and `scripts/redline_daemon_harness.py --qwen4` gates it per window. Qwen3.5 tapes declare no bindings, so their oracle is unchanged.
- **`qwen3.8:27b-mq4-xts` is the H2 checkpoint and the canonical dense fixture.** The tag now pins `qwen3.8-27b.mq4-xts` to H2: symmetric MQ4V2 XT with per-group AWQ alpha and three GPTQ scale-refit rounds (14,987,185,152 B, sha256 `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`). It replaces the 2026-09-23 test upload (QAT r7s200, `de8ee825…`), so a copy pulled before this release fails the new checksum, and `hipfire pull qwen3.8:27b-mq4-xts` fetches it again. The DFlash draft (`qwen38-27b-dflash-mq4.hfq`) and vision sidecars are the ones `qwen3.8:27b-mq4-xt` uses. The bare `qwen3.8:27b` tag and the `:fast` aliases are unchanged.
  - `AGENTS.md` §5 now pins `qwen3.8-27b.mq4-xts`, with the H2 `.kldseq` quality pins for gfx1201 (A4 and native fp8), gfx1100 and gfx1151. The asymmetric `qwen3.8-27b.mq4-xt` (`80e7c624…`) moves to the historical note.
  - `scripts/check_fixture.sh` checks the new pin and now always verifies the sha256 (the `--sha` flag is gone). `qwen3.8-27b.mq4-xt` has exactly the fixture's size, so the size alone cannot tell the two apart.
  - The bundled `registry/v1.json` is stamped `2026-09-29T23:15:36Z`, so a build from this tree uses it rather than the fetched master `v1.json`, which is older and does not declare `qwen3.8:27b-mq4-xts`.
- **Retained PM4 kernargs live in VRAM (host-writable GPU-agent pool) instead of host memory (every PM4 route; byte-identical; H2 decode vs the host-pool kernargs, same binary: R9700 +3.34 / +3.17 / +3.03 %, 7900 XTX +7.19 / +7.18 / +7.51 %, Strix Halo +0.23 / +0.20 / +0.19 % at ctx 512 / 8,192 / 32,768).** Kernels whose prologue chains dependent kernarg `s_load`s paid host-memory latency on every retained replay. `ReplayController::prepare_pm4_prefix_inner`, the single kernarg allocation point for H2, MQ4R, DS4, LFM and DSpark, now takes the segments from the GPU-agent pool, and the allocation grants access to both the GPU and CPU agents. The per-token position and GDN-frame patches stay plain 4-byte host stores through the BAR mapping; before the doorbell, one SeqCst fence and a one-byte readback of the last device-local segment publish them. The indirect buffer, timestamps, completion and fence words stay in the host pool. The daemon log records the choice as `[redline] retained PM4 kernargs: pool=vram` or `pool=host (<reason>)`.
  - `HIPFIRE_PM4_KERNARG_POOL=host` (`diagnostic.replay.pm4_kernarg_pool = "host"`) returns the host pool. The host pool is also kept automatically on small-BAR systems (CPU access never allowed), when `hsa_amd_agent_memory_pool_get_info` is missing, and when a gfx1151 non-system entry acquire is set. No default route selection changed.
  - Byte-identical, 256 greedy steps at ctx 512 and 8,192 on all three cards: VRAM = host = HIP graph in every comparison (poisons 0x00 and 0xFF with the primed state loaded, each pool's own prefill, and WikiText-2), and on gfx1201 both equal beta `70eef6a14`; the negative control diverges at step 0. Fixed-position replay stress (200 replays, and 24 with `ROC_GLOBAL_CU_MASK=0x1`) reproduces the graph bytes on every card. A 33,000-step XTX sweep (positions 500 → 33,499) is byte-identical to the HIP graph (median step 21,254 → 19,669 µs). KLD pins unchanged: gfx1201 A4 and FP8 (`483cfc58…`, `6339dc9e…`, `19227d5f…`, `5e7ac9dc…`), XTX `ccf95389…` / `a7eac2b4…`, Halo `c1056943…` / `04f06883…`. No kernel faults in any process.
  - Throughput, `hipfire bench --matrix --pp 8192 --tg 128`, four fresh processes per arm in ABBA+BAAB order over HIP graph / PM4-host / PM4-VRAM, tok/s medians at ctx 512 / 8,192 / 32,768:
    - R9700, H2 default route: host 39.930 / 39.063 / 36.776 → VRAM 41.266 / 40.301 / 37.891 (beta 39.956 / 39.068 / 36.800); every VRAM process is ahead of every host and beta process at every context; pp8192 5,142.4 → 5,135.7, ranges overlap. `noslots` stateless 26.170 → 26.870 tok/s (+2.67 %).
    - 7900 XTX, H2 `--redline`: host 50.697 / 48.780 / 45.172 → VRAM 54.344 / 52.281 / 48.565 (HIP graph 51.106 / 49.267 / 45.937, +6.33 / +6.12 / +5.72 %); every VRAM process is ahead of every other at every context; pp8192 ranges overlap.
    - Strix Halo, H2 `--redline`: host 15.663 / 15.274 / 14.282 → VRAM 15.700 / 15.305 / 14.308 (HIP graph 15.079 / 14.699 / 13.781). At 32,768 every VRAM process is ahead of every host process; at 512 and 8,192 one noisy VRAM process (per-run 14.10–15.42 tok/s) overlaps the host range. pp8192 is within the Halo's process-to-process spread (1,134–1,237 tok/s across all arms).
- **Kernel objects are byte-reproducible: runtime JIT and packaged compiles add `-fuse-cuid=none` (kernel cache ABI 4 → 5).** hipcc named a `__hip_cuid_<md5>` symbol after the input path and the full argv. The JIT compiles every source under the cache root to a unique temp name, so two compiles of the same source and recipe never gave the same bytes, although the code was the same. The JIT argv and the packaging recipe now share one core flag list, `--genco --offload-arch=<arch> -O3 --no-offload-compress -fuse-cuid=none`. All 217 registry modules (gfx1201 96, gfx1100 62, gfx1151 59), compiled twice through `hipfire-kernel-pack` under different cache roots, are now byte-identical; before, 0 of 217 were.
  - The code matches the previous objects in all 217 modules: the same instructions and metadata, and the same kernel descriptors apart from their entry offsets. The marker is now `__hip_cuid_` with no suffix. That shortens the string tables and moves `.rodata` 64 bytes lower in 44 modules. There, the descriptors' entry offsets change, and so does one PC-relative literal in `kv_cache_write_asym_k_givens3_batched`. Every one of them still resolves to the same target.
  - The ABI bump re-keys every hot JIT entry and packaging key. Pack indexes from older installs fail their flags and ABI check, so the first run after upgrading recompiles its kernels. Compiler-free installs (no hipcc, or `HIPFIRE_NO_DEVICE_COMPILER=1`) must re-run the installer.
  - Checks on gfx1201, from empty kernel caches:
    - both A4 KLD pins are unchanged (WT2 `483cfc58…` 0.069571, code24 `6339dc9e…` 0.052119);
    - 256-step greedy decode at ctx 512 is byte-identical to beta on the default (retained PM4) route and the graph route;
    - decode tg128 at ctx 512 (39.845 → 39.884 tok/s) and pp8192 (5,125.2 → 5,122.6 tok/s) are unchanged against beta: separate binaries on an R9700, four fresh processes per arm, ABBA then BAAB, and the two arms' ranges overlap.
- **gfx1201 DFlash rollback without the DeltaNet restore copy (railgun D8, opt-in `HIPFIRE_DN_SNAPSHOT_FLIP=1`, default off; byte-identical; Qwen3.8-27B MQ4-xt + DFlash2 B=16 code fixture 312.50 → 319.89 tok/s, +2.36 %, prose 55.43 → 56.02 tok/s, +1.06 %).** Exact gfx1201, tape path with the two-launch GDN replay. A full accept (EF residual on, HIP/HipGraph verify) keeps the verify-advanced DeltaNet state; any shorter accept replays from the snapshot into the live buffers (`dflash_gdn_replay_pre_ml_from` + `gated_delta_net_q8_fast_ml_from`, the E0 kernels with the state loads redirected). The snapshot stays the pre-window state, so terminal repair is unchanged; no DeltaNet pointer moves, so no graph or table is re-bound. Code cycle 45.778 → 44.579 ms (restore 0.421 → 0.001 ms, code replay median 1.048 → 0 ms). Same binary, four fresh processes per arm, ABBA +7.57 / BAAB +7.41 tok/s, every on process ahead; committed tokens and τ identical over 16,336 tokens; `test_dflash_replay_from_snapshot` byte-identical to restore + replay.
- **gfx1201 DFlash cycle (railgun E0): bulk DeltaNet snapshot, two-launch GDN replay and re-gridded selectors (all byte-identical; Qwen3.8-27B MQ4-xt + DFlash2 B=16 code fixture, cycle 48.712 → 45.557 ms, decode 292.64 → 313.19 tok/s, +7.0 %).** Measured on one R9700 with `HIPFIRE_SPEC_PHASES=1`, four fresh processes per arm in ABBA+BAAB order against beta `8d302c342`; committed tokens and τ identical over 16,336 tokens. The three levers, each on exact gfx1201 only:
  - DeltaNet snapshot save/restore use the descriptor-driven bulk byte copy that gfx1100 already used (one launch each instead of 192 `hipMemcpy`): ngram 0.890 → 0.594 ms, restore 0.928 → 0.417 ms. `HIPFIRE_DN_SNAPSHOT_BULK_OFF=1` opts out.
  - The GDN tape replay runs all 48 LinearAttention layers in two launches over device pointer tables (`dflash_gdn_replay_pre_ml`, `gated_delta_net_q8_fast_ml`) instead of four launches per layer: replay 1.960 → 1.049 ms. `HIPFIRE_GDN_REPLAY_ML_OFF=1` opts out.
  - `topk_values_batched_f32` (DFlash2 selector) and `argmax_f32_batched` run 1024-thread rows with wave32 reductions; top-K rows whose leading K+1 values tie rerun the shipping body: draft 8.969 → 7.562 ms. `HIPFIRE_SELECT_REGRID_OFF=1` opts out.
- **Decode round 2 (H2 plain-AR decode; R9700 gfx1201, 7900 XTX gfx1100, Strix Halo gfx1151): the gfx1201 Qwen3.6-27B decode fusions and PM4 pacing, the gfx1100 GQA-shared Q8_0 decode attention and decode norm grids, and the gfx1151 decode norm grids and GQA-shared Q8_0 decode attention land together (all byte-identical; H2 decode R9700 38.80 → 39.96 / 37.93 → 39.08 / 35.81 → 36.80 tok/s, XTX 48.90 → 51.20 / 47.19 → 49.34 / 40.77 → 46.01 tok/s, Strix Halo 14.853 → 15.080 / 14.346 → 14.699 / 13.014 → 13.778 tok/s at ctx 512/8,192/32,768).** Each change is selected by exact arch, so every card keeps dispatching the other two cards' kernels with the same arguments as before. The R9700 figures are against beta `8d302c342` with separate binaries; the XTX and Halo figures are against both opt-outs in the same binary. Opt-outs:
  - gfx1201: `HIPFIRE_GDN_COMPACT3=0`, `HIPFIRE_GATED_NORM_MQ_ROTATE=0` and `HIPFIRE_QWEN35_FA_PREP_FUSE=0` turn the fusions off on gfx1201, as on gfx1100.
  - gfx1201: `HIPFIRE_GFX1201_PM4_PACING=off` (or `replay.gfx1201_pm4_pacing = "off"`) returns the unpaced tape.
  - gfx1100: `HIPFIRE_GFX1100_DECODE_ATTN_GQA=0` turns off DecAttn.
  - gfx1100: `HIPFIRE_GFX1100_DEC_NORM=0` (or `kernel.gfx1100_dec_norm = false`) turns off DecNorm.
  - gfx1151: `HIPFIRE_G12_DEC_NORM=0` (or `kernel.g12_dec_norm = false`) turns off DecNorm on gfx1151 as on gfx1201.
  - gfx1151: `HIPFIRE_GFX1151_Q8_DECODE_ATTN_GQA=0` turns off the gfx1151 decode attention twin.
  - **gfx1201 decode round 2 (H2): the Qwen3.6-27B decode fusions on exact gfx1201 and PM4 pacing of the retained decode tape (both byte-identical; R9700 H2 decode 38.80 → 39.96 / 37.93 → 39.08 / 35.81 → 36.80 tok/s at ctx 512/8,192/32,768, +3.00 %/+3.03 %/+2.75 %).** gfx1201 decode goes from 25.77/26.36/27.92 to 25.02/25.59/27.18 ms per token against beta `8d302c342`. Measured with separate md5-pinned binaries, four fresh processes per arm, ABBA −0.761/−0.783/−0.764 ms then BAAB −0.756/−0.754/−0.741 ms, `hipfire bench --pp 128 --ctx 512,8192,32768 --tg 128`, default route (retained Redline PM4), fp8 KV, clocks auto. Every decode/r2 process was faster than every beta process. pp8192 is unchanged (5,144.85 → 5,144.6 tok/s). All four `.kldseq` pins are unchanged (A4 WT2 `483cfc58…`, A4 code24 `6339dc9e…`, fp8 WT2 `19227d5f…`, fp8 code24 `5e7ac9dc…`).
    - **gfx1201 Qwen3.6-27B decode fusions (byte-identical; 899 → 755 launches per H2 token; −0.271/−0.326/−0.305 ms on/off, same binary).** `gdn_compact3_enabled`, `gated_norm_mq_rotate_enabled` and `qwen35_fa_prep_enabled` now also admit exact gfx1201 for the exact Qwen3.6-27B dense shape (dim 5120, 48 v-heads, 24Q/4K): the compact-3 GDN, the 48-head gated-norm/MQ-rotate and the 24Q/4K FA prep that gfx1100 already ships. They launch as gfx1201-named twins (`gated_norm_mq_rotate_{,awq_}k6144_gfx1201`, `qwen36_27b_fa_prep_gfx1201`), which are in the Redline name-keyed tables with a padded-replay contract test. The FA-prep twin pins both RoPE outputs with explicit `__builtin_fmaf` in `rope_partial_halfsplit_f32_headgrid`'s order; the shared source contracted the second output as `fma(x1, cos, x0*sin)`, which is not bit-exact. Each fused kernel is byte-identical to the unfused chain on real captured H2 decode activations (ctx 512/8,192/32,768, every layer, 200× serial and 200× on one WGP, 3 poison patterns, guard bytes clean, negative control caught). The retained tape is equal to the HIP graph at every poison, and the default route is equal to beta, over 256 greedy steps at ctx 512/8,192/32,768. One continuous 33,000-step greedy decode (positions 500 → 33,499) is byte-identical to beta at every step.
    - **gfx1201 PM4 pacing (byte-identical; −0.445/−0.383/−0.453 ms on/off, same binary).** On the retained gfx1201 Qwen3.5-dense decode tape, one `NOP` with 64 body dwords now follows every `DISPATCH_DIRECT` (`Gfx12DispatchPacing::PostDispatchNop`; H2 tape 25,796 → 74,871 dwords). A NOP writes no register and no memory. The gain comes from when the command processor reaches the next boundary's `CS_PARTIAL_FLUSH`.
      - Sweep at auto clocks, 4 processes per arm, ms at 512/8k: 32 dwords −0.375/−0.378, 64 dwords −0.424/−0.405, 128 dwords −0.451/−0.408.
      - Aligning every dispatch to a 32- or 64-dword IB boundary instead is 0.04–0.09 ms slower than the unpaced tape.
      - At pinned `profile_peak` (2.19 GHz) and `profile_standard` (1.61 GHz) the plateau is the same 64–128 dwords. At `profile_min_sclk` (495 MHz) pacing costs 0.03–0.24 ms, so the length is fixed, not clock-calibrated.
      - The daemon sets it only when the Redline retained default admits exact gfx1201 + `qwen3_5` and the load is not an MQ4R default. MQ4R on gfx1100/gfx1151/gfx1201, DS4 on gfx1151 and every gfx11 tape are unchanged; gfx11 tapes have no pacing path at all.
      - Checks: paced PM4 = unpaced PM4 = HIP graph at every poison and ctx 512/8,192/32,768; the default route = beta over 256 greedy steps; the 200-replay stress passes; the 33,000-step length sweep with both levers on is byte-identical to beta.
  - **gfx1100 (7900 XTX) decode (H2): GQA-shared Q8_0 decode attention with a load-ahead gated reduce, and the decode norms as multi-workgroup grids (both byte-identical; XTX H2 decode 48.90 → 51.20 / 47.19 → 49.34 / 40.77 → 46.01 tok/s at ctx 512/8,192/32,768, +4.71 %/+4.55 %/+12.85 %).** Both levers are selected on exact gfx1100 only (two gfx1100-only `.hip` files; every Rust change is behind `is_gfx1100()` / `arch == "gfx1100"`). Measured against both opt-outs in the same binary: four fresh processes per arm, ABBA then BAAB, `hipfire bench --matrix --pp 128 --ctx 512,8192,32768 --tg 128 --backend noslots --workload stateless`, graph on, clocks auto, KV Q8/Q8. Decode goes from 20.45/21.19/24.53 to 19.53/20.27/21.74 ms per token (ABBA +4.73/+4.62/+12.85 %, BAAB +4.31/+4.26/+12.79 %). Every lever-on process beat every opted-out process at every context. pp8192 is unchanged: 3,006.45 → 3,010.75 tok/s (ABBA +0.06 %, BAAB −0.16 %). Greedy decode produces logits byte-identical to beta at every step, with each lever alone and with both, in two tests: 256 steps at ctx 512/8,192/32,768 (graph and eager), and a continuous 32,800-step sweep over positions 500–33,299. Both XTX `.kldseq` pins are unchanged with both levers on, both off and each alone (WT2 `ccf95389…`, KLD 0.068687; code24 `a7eac2b4…`, KLD 0.052666). Report: /home/kaden/qcal/perf/decode-gfx1100/report.md.
    - **gfx1100 Q8_0 decode attention: GQA-shared flash tile + load-ahead reduce/gate/rotate (byte-identical).** On exact gfx1100, with head_dim 256, GQA group 6, tile 128, full causal attention and the gated AWQ MQ-rotating epilogue (H2 decode), `attention_flash_q8_0_tile_gqa_gfx1100` replaces the one-wave-per-(q head, tile) `attention_flash_q8_0_tile`. One 256-thread workgroup per (kv head, tile) serves the six q heads of its kv head, so each K/V tile is read once, and every load of the tile is issued at entry. As in the gfx1201 GQA tile, the grid is capped at 512 tiles with a tile loop. `attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100` replaces `attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100`. It runs one 1024-thread workgroup per head, issues every tile's max/sum and 64 tiles of one dim per thread up front, then runs the reference's increasing-tile chains through an LDS carry. Per-head arithmetic reproduces the reference ISA: dequant product, lane leaf FMA chains, the xor-16/8/4/2/1 tree, softmax source, increasing-key V FMA, the 2-pass combine, gate, and `/awq`·signs·FWHT-256·signs. Admission caps max_tiles at 7,680 (the reduce keeps 2·max_tiles f32 of dynamic LDS). Partials and output are whole-buffer identical to beta's objects on 18 real H2 decode captures (layers 0/7/15 at seq 513/516/8,193/8,196/32,769/32,772, plus every prefix down to 1). That is 1,008 cases covering 3 poisons, 256 B guards, and graph and eager grids. The negative control was detected in 108/108 cases. The same captures passed 200× serial and 200× under `ROC_GLOBAL_CU_MASK=0x1` (the mask was confirmed in effect). Synthetic seqs up to 131,075 are also identical. A 60 s unprofiled soak at seq 131,075 kept the card on the bus: vendor id sampled every 0.5 s, no kernel messages. The candidate binary's in-model captures are byte-identical to beta's, and its runtime JIT objects are code-identical to the gated ones. Tile + reduce per layer (event timing): 74.0 → 28.4 µs at seq 517, 107.4 → 54.2 at 8,197, 300.0 → 153.5 at 32,773. H2 decode, same binary with the opt-out off/on (four fresh processes per arm, ABBA then BAAB): ctx 512 50.03 → 51.25 tok/s (+2.31 %/+2.27 %, −0.47 ms/token), ctx 8,192 48.27 → 49.39 (+2.27 %/+2.17 %, −0.47 ms), ctx 32,768 41.61 → 46.04 (+10.59 %/+10.67 %, −2.31 ms). Every lever-on process beat every opt-out process. `HIPFIRE_GFX1100_DECODE_ATTN_GQA=0` restores the reference pair.
    - **gfx1100 decode norms as multi-workgroup grids (bit-exact).** On exact gfx1100 (`kernel.gfx1100_dec_norm`, default on) the gfx1201 DecNorm twins run on the XTX. The f32 AWQ RMSNorm+FWHT before every MQ4 input projection (128 launches per H2 token) runs `fused_rmsnorm_mq_rotate_awq_g12dec` on grid K/256 instead of 1. The out-of-place single-row `rmsnorm_f32` with n > 256 (the final norm) runs `rmsnorm_f32_rowsplit` on n/256 workgroups. RoPE is untouched, because gfx1100 decode fuses it into its FA prep. Standalone on H2 shapes the two kernels go from 6.41 to 3.63 µs and from 7.28 to 3.05 µs per call. Output is byte-identical to beta's objects on 2,000 captured AWQ-norm inputs and 79 final-norm H2 decode inputs, with 3 poisons and guards. Negative controls were detected, and the inputs passed 200× serial and 200× under `ROC_GLOBAL_CU_MASK=0x1` (mask confirmed in effect). H2 decode, same binary with the opt-out off/on: ctx 512 50.22 → 51.30 tok/s (+2.22 %/+2.00 %, −0.42 ms/token), ctx 8,192 48.47 → 49.44 (+2.06 %/+1.92 %, −0.40 ms), ctx 32,768 45.24 → 46.10 (+1.92 %/+1.90 %, −0.42 ms). Every lever-on process beat every opt-out process. `HIPFIRE_GFX1100_DEC_NORM=0` restores the single-workgroup launches.
  - **Strix Halo (gfx1151) decode (H2): the decode norm grids widen to exact gfx1151 and a gfx1151 Q8_0 GQA-shared decode attention lands (both byte-identical; Strix Halo H2 decode 14.853 → 15.080 / 14.346 → 14.699 / 13.014 → 13.778 tok/s at ctx 512/8,192/32,768, +1.53 %/+2.46 %/+5.87 %).** Both are selected on exact gfx1151 only; gfx1100 and gfx1201 dispatch the same kernels with the same arguments as before (the gfx1201 fp8 GQA pair now goes through the shared `attention_flash_decode_gqa_pair` launcher, same symbols, grid and kernargs).
    - **gfx1151 decode norms as multi-workgroup grids.** `kernel.g12_dec_norm` now defaults on for exact gfx1151 too: the f32 AWQ RMSNorm+FWHT producer before every MQ4 input projection runs `fused_rmsnorm_mq_rotate_awq_g12dec` on grid K/256, the out-of-place single-row `rmsnorm_f32` (n > 256) runs `rmsnorm_f32_rowsplit` on n/256 workgroups, and the half-split partial RoPE runs `rope_partial_halfsplit_f32_headgrid` with one workgroup per head. Same sources as gfx1201; the outputs are byte-identical on real H2 decode activations at ctx 512/8,192/32,768 (whole buffer + guards, 3 poisons, negative controls detected, 200× serial and 200× under a one-CU mask, RoPE positions 0…262,144). Standalone on the Halo: AWQ K = 5,120 5.96 → 3.16 µs, `rmsnorm_f32` n = 5,120 6.77 → 2.49 µs, RoPE 5.08 → 2.20 µs. H2 tg128, same binary, four fresh processes per arm, ABBA then BAAB, `hipfire bench --pp 8192 --ctx 512,8192,32768 --tg 128 --backend noslots --workload stateless`, graph on, Q8 KV: off → on 14.958 → 15.079 / 14.583 → 14.700 / 13.678 → 13.779 tok/s at ctx 512/8,192/32,768 (+0.81 %/+0.80 %/+0.74 %; ABBA +0.116/+0.113/+0.100, BAAB +0.121/+0.115/+0.100 tok/s), every on process ahead of every off process; pp8192 1,162.45 → 1,157.55 tok/s, within noise (halves +36.65 / −42.70).
    - **gfx1151 Q8_0 decode attention: GQA-shared flash tile + head-dim-split reduce.** On exact gfx1151, Q8_0 KV, head_dim 256, GQA group 6, tile 128, full causal and no output gate (H2 decode), `attention_flash_q8_0_tile_gqa_gfx1151` replaces `attention_flash_q8_0_tile`: one 256-thread workgroup per (kv head, tile) serves the six q heads, so each Q8_0 K/V tile is read once, and the whole tile's Q, V rows and K codes/scales are issued at entry; each 16-lane row reproduces the reference lanes' dequant (`scale * code`, one rounding), fma leaf order and xor-16…1 add tree, softmax and in-order V accumulation. `attention_flash_reduce_dsplit_gfx1151` (the gfx1201 head-dim-split reduce, renamed) replaces `attention_flash_q8_0_reduce`. The tile grid is capped at 512 tiles with a tile loop. Redline's replay tables type both kernels with their reference twins' 13/7-argument ABIs and pointer effects. Standalone on the Halo, the tile + reduce pair at seq 517 / 8,197 / 32,773: 45.77 → 22.56 / 200.62 → 120.59 / 663.24 → 401.71 µs. The outputs are byte-identical to beta's objects on real H2 decode activations (seq C+1…C+4 for C = 512/8,192/32,768, full-attention layers 0/7/15: 36/36 captures, 3 poisons, graph and eager grids, whole buffer + guards, a one-K-code-byte negative control detected in all 36, 200× serial and 200× under a one-CU mask, 60 s soak). H2 tg128, same binary, four fresh processes per arm, ABBA then BAAB: off → on 14.974 → 15.079 / 14.459 → 14.698 / 13.108 → 13.777 tok/s (+0.70 %/+1.65 %/+5.10 %; ABBA +0.106/+0.242/+0.671, BAAB +0.099/+0.233/+0.663 tok/s), every on process ahead of every off process; pp8192 1,154.90 → 1,154.25 tok/s, within noise (halves +38.35 / −8.30).
    - **Both levers together** (same binary, both opt-outs vs defaults, four fresh processes per arm, ABBA then BAAB): 14.853 → 15.080 / 14.346 → 14.699 / 13.014 → 13.778 tok/s at ctx 512/8,192/32,768 (+1.53 %/+2.46 %/+5.87 %; ABBA +0.225/+0.352/+0.761, BAAB +0.230/+0.355/+0.768 tok/s), every on process ahead of every off process; pp8192 1,152.2 → 1,156.6 tok/s, within noise (halves +7.55 / −0.95). Checks: 256-step greedy logits equal beta at ctx 512/8,192/32,768 (synthetic and WikiText-2 primes, every step's whole logits + hidden, final KV arena + DeltaNet), with the opt-out arm (both levers off) also equal to beta; one continuous 32,788-step greedy decode (positions 512 … 33,299) is byte-identical to beta at every step; the Halo KLD pins `c1056943…` (WT2) / `04f06883…` (code24) are unchanged.

- Qwen4 raw-I64 PLE metadata records are HFQM qt=54 (qt=52 is MQ4G256V2L).
  The published `qwen3.8:flash-next` and `qwen3.8:flash-next-mq6q8-pleq8`
  files were re-tagged in place (payloads unchanged) and re-pinned; the loader
  refuses the old qt=52 spelling, so re-pull them.

- Qwen4 native MTP is now on by default: `speculation.mtp = auto` attaches the
  MTP head, where before only an explicit `on` did. Greedy requests then run
  MTP (1.77x AR decode on `qwen3.8-flash-next.mq4`, tokens equal to AR's).
  Sampled requests stay on the AR route, and attaching the head showed no
  measurable prefill cost. `speculation.mtp = off` keeps AR.
  A `.mq4r` load is the exception: there the retained Redline default stays
  in force. Its tape is the single-row AR forward, which no MTP window runs,
  so under `auto` the load stays AR, and an explicit `on` still refuses.

- Qwen4 MTP draft head runs as one engine `Step` list: the prologue
  (`Embed`, `HyperNorm`, `Project`, `BroadcastAdd`), HC read/write, indexed
  attention (new `IndexedAttentionMode` append-only and selection-reuse
  modes), MoE and final HC read execute through `hipfire_dispatch`, and the
  draft ranking moved to the shared `DraftHead`. Greedy MTP tokens still
  equal AR's; 9-prompt MTP geomean 59.37 → 59.04 tok/s (best of 3, within
  noise), mean tau unchanged at 2.10.

- Qwen4 decode 32.4 → 33.4 tok/s on gfx1151 (1131-token prompt, 5 runs,
  same binary), prefill unchanged at ~1255 tok/s:
  - Single-token forwards also read the shared expert as load-time Q8_0
    copies (+0.25 GB, like the HC read projections); 32.4 → 32.8 tok/s on
    the published `qwen3.8-flash-next.mq6q8-pleq8`.
  - Recipe r2, `qwen3.8-flash-next.mq4` (125288540696 B,
    quantized from the BF16 checkpoint): the language head ships MQ6G256V2
    instead of Q8F16 (−0.18 GB read per token); every other tensor is
    unchanged. 32.8 → 33.4 tok/s. KLD against the BF16 source, 32 chunks:
    decode route 0.07539 → 0.07602, prefill route 0.07347 → 0.07432.
    Published on HF `hipfire-models/qwen3.8-flash-next`; `qwen3.8:flash-next`
    now pulls it (tier tag `qwen3.8:flash-next-mq4`), and the previous file
    stays available as `qwen3.8:flash-next-mq6q8-pleq8`.
  - Measured and not taken: Q8 in the file for the shared expert, the PLE
    key/value and the HC read projections. Decode already reads Q8 copies
    of the matrices it streams, while prefill then lost its BF16 source
    (prefill-route KLD +0.0035) and short prompts had no fast exact Q8
    multi-row GEMM (+46% prefill time at ~100 tokens).

- Qwen4 native MTP decode on gfx1151: 28.7 → 54.1 tok/s (geomean of the
  committed code and prose prompts, greedy, `qwen3.8-flash-next.mq6q8-pleq8`;
  AR ~33 on the same build), with greedy MTP tokens equal to AR's. The 2..8-row
  verify forward is now bitwise the single-row decode route and ~2x cheaper
  than before (HC Q8 copies, staged Q8 LM head and MQ6 kernels over all rows,
  fused HC write+norm, per-slot MoE down, and ~35% fewer launches: fused
  rotations, QSA prologue, GDN conv+params, grouped-MoE round trips, router,
  unscatter and shared activation); the verify keeps only its last GDN state
  and a rejected suffix re-runs the kept rows' recurrence for every GDN layer
  in one launch, and state snapshots copy in one launch; each window picks
  the draft depth (or the interleaved route) from per-depth draft agreement
  and stops early once the drafts' exact logit margins make the prefix
  unlikely to be accepted;
  drafts come from an MQ2 copy of the LM head re-scored exactly on its top 8
  (`HIPFIRE_MTP_DRAFT_HEAD`), ranking only token ids below 100 000 plus the
  control tokens until an input token outside them appears, and the MTP
  head's attention, selection reuse and chain-final K/V-only steps no longer
  stall or waste work. SSD-resident PLE rows are read concurrently and warmed
  while drafting, which also helps AR decode on cold rows.
  `HIPFIRE_MTP_INCREMENTAL` unset now means adaptive (`0` batched, `1`
  interleaved).
  The re-score reads a Q8_0 or MQ6G256V2 head. On `qwen3.8-flash-next.mq4`
  (MQ6G256V2 head) it lifts greedy MTP from 55.4 to 59.5 tok/s (geomean over
  9 committed prompts, 3 interleaved rounds; mean tau 1.89 → 2.10), level
  with the same artifact carrying a Q8_0 head (59.2), whose AR is 1.6% slower
  (33.1 vs 33.7).
  [The checkpoint](docs/perf-checkpoints/2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md)
  and amendments [1](docs/perf-checkpoints/2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151-amendment-1.md),
  [2](docs/perf-checkpoints/2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151-amendment-2.md).

- Qwen4 decode on gfx1151: 19.2 → 32.3 tok/s (1131-token prompt, greedy,
  `qwen3.8-flash-next.mq6q8-pleq8`). Per-token dispatches drop by fusing the
  small kernels between the streaming GEMVs (HC write + next HC read norm and
  the next write's gate, QSA decode prologue, GDN step + gated norm + output
  rotation, rotations written by their producers) and latency-bound kernels
  issue their loads up front. Pure-greedy sampling takes the argmax on the GPU
  (`argmax_f32` now matches `llama::argmax`: first finite maximum), and the
  greedy token feeds the next forward on the device, read back only at the
  PLE layer, so the token boundary no longer idles the GPU. One change
  (four-wave K split of the decode HC GEMVs) moves numerics and lowers decode
  KLD (32 chunks 0.075160 → 0.074299); every other change is bit-exact.
  Then 30.0 → 32.3 tok/s: single-token forwards on gfx1151 read the HC read
  projections as Q8_0 requantized from BF16 at load (+0.67 GB; prefill keeps
  BF16), decode KLD 0.074299 → 0.075390 over 32 chunks (paired t 1.59, inside
  the noise band).
  [The checkpoint](docs/perf-checkpoints/2026-09-27-qwen4-decode-autoresearch-gfx1151.md).

- Qwen4 prefill on gfx1151: 185 → 1301 tok/s on a 1131-token prompt
  (TTFT 6.10 → 0.87 s; decode unchanged at ~20 tok/s) for
  `qwen3.8-flash-next.mq6q8-pleq8`. Routed MoE gate/up (the WMMA kernel from
  PR #775, @nwoolmer) and down, the BF16 dense projections (through a
  model-lifetime F16 weight shadow), the HC
  read, full-window QSA attention and the chunked GDN recurrence run on F16
  WMMA from 512 tokens; KLD against the BF16 source is 0.073471 against
  0.074745 bit-exact, not separated. `HIPFIRE_QWEN4_F16_WMMA=0` keeps the
  bit-exact arms. Every other change is bit-exact. Method, per-run ledger and
  reusable levers:
  [the checkpoint](docs/perf-checkpoints/2026-09-26-qwen4-prefill-autoresearch-gfx1151.md).

- The Qwen4 routes tuned on gfx1151 now dispatch by ISA capability, on by
  default wherever their kernels exist: the SIMT/LDS kernels (BF16 multirow,
  MQ4G128 multirow, MoE O4xR8/O8xR16, GDN/HC/QSA fusions) on gfx11 and gfx12
  (`has_gfx11_plus_simt`), the gfx11 WMMA kernels (F16 prefill, MQ6 X-LDS,
  QT53 down, GDN chunk, QSA dense, HC WMMA read) on every gfx11 GPU
  (`has_wmma_w32`); the grouped gate/up F16 WMMA F32 arm also uses its gfx12
  sibling. `HIPFIRE_QWEN4_GFX11` is removed. Measured only on gfx1151; gfx12
  runs the portable kernels where no gfx12 WMMA port exists.

- Review follow-up: Qwen4's grouped QT44/QT53 route now carries an
  architecture-declared geometry/format contract and remains limited to its
  proven gfx1151 kernel shape. Shared tensor ops use portable HIP fallbacks
  off the Qwen4-tuned GPUs; pipeline steps are immutable, with QSA bookkeeping owned by
  the caller and granular MoE stages sharing one step variant. Qwen4 admission
  is carrier-owned, its HIP source lives under `kernels/src/`, and reference
  MTP/parity code requires the `reference-parity` feature outside tests.

- Qwen4 Flash-Next MTP now preserves the trained head's final HC mixer and
  QSA cadence, and keeps routed MoE output separate from shared-expert scratch.
  On gfx1151 a shared-weight F32 MQ6 verifier matches scalar target rows
  exactly; recent acceptance selects it for high-agreement windows.
  Teacher-forced MTP steps omit unused logits, and target prefill computes only
  its final logit row. The canonical greedy fixtures emit AR-identical tokens;
  three fresh-process runs per mode reached decode parity on code and prose,
  not the 80% speculative speedup seen elsewhere.

- Qwen4 decode now runs on the retained PM4 route end to end on gfx1151. The
  specialized sealed-MoE route is admitted on evidence rather than argument: the
  capture census reconciles against an independent launch count (2845 recorded plus
  3 named external input-boundary launches), is stable across positions, and the
  window contains no device copy, memset or readback — the per-layer `Clear` and the
  QSA/depthwise device copies are now recorded launches (`zero_f32`,
  `copy_f32_buffer`). At launch the retained body also proves the expert pointer
  tables name the live expert tensors and that each table's pointer mapping is the
  one the tape latched. The eligible-forward boundary moved into the Qwen4 forward
  so the tape covers only the body, and replay and HIP derive the same per-layer
  QSA bookkeeping before executing it. Measured: a 116-token
  greedy prompt decoded through 115 retained PM4 replays produces the byte-identical
  stream. REDLINE §7 certification (shadow parity, route-proof ledger, serve,
  long-context, reset) is not claimed by this change; see
  [the plan record](docs/design/qwen4-program-retained-pm4.md).
- Retained capture now has a diagnostic census for the Qwen4 declarative program
  (`HIPFIRE_REPLAY_DIAGNOSTIC_SPECIALIZED_MOE_CAPTURE=1`, measurement only: it never
  installs a plan and never routes). It reconciles the retained tape against an
  independent `hip_bridge` launch count, audits non-launch effects inside the
  capture window, and names any kernel that escaped the recorder. First use found
  one such launch (`Gpu::add_f32` was raw while a funnel-based twin existed for the
  same kernel — now a single funnel path with the twin deleted and its callers
  migrated), proved the decode body's launch set position-stable across prompt
  lengths, and quantified the remaining admission blocker: per decode forward the
  window still contains 48 layer memsets, 13 device-to-device state copies and 2
  host uploads, none of which a launch tape replays. Admission therefore stays
  refused. See [the plan record](docs/design/qwen4-program-retained-pm4.md).
- Retained replay no longer refuses the *whole model* when a family's MoE route has no
  admitted pointer contract: the specialized sealed-MoE guard is now scoped to the
  retained body (`is_recording()` or a routed plan), so prefill, ineligible forwards,
  and an already-poisoned route keep running on HIP. Qwen4 gained the corresponding
  engine-level retained-body discipline in `hipfire-generate` (prefill ineligible;
  single-token decode poisons and logs when the route cannot be retained; a body
  failure inside a capture window poisons instead of failing every later forward; a
  routed-but-unprepared state fails closed). With the Redline default armed, the
  `.mq4r` Qwen4 artifact now loads, prefills, and generates on HIP with the refusal
  reason logged, byte-identical to the explicit HIP baseline; capture stays
  fail-closed, and `hipfire-arch-qwen4` still carries no replay code. See
  [the plan record](docs/design/qwen4-program-retained-pm4.md).
- Qwen4 QSA launches now declare position-independent shapes and position-derived
  fields, so a retained tape cannot bake a capture position into them. The pool
  grid, the select LDS/symbol, and the attention LDS/symbol come from declared
  capacities while the active lengths stay scalars; `position_start` and the
  pooling count are declared at the launch that computes them (the pooling-count
  declaration is verified against the launched value); and the index-key write
  passes its row offset as a scalar (`ReplayKernargBinding::PositionMulU32`)
  instead of a position-shifted device pointer, which required
  `copy_rows_strided_f32` to bound a row-absolute column offset by the destination
  extent rather than by one row pitch. Measured on gfx1151: pinned shapes are
  bit-identical to the position-derived ones (including the batched select symbol
  at an active count of zero), a 116-token greedy stream is byte-identical across
  5 interleaved fresh-process pairs with equal tok/s medians, and no route is
  admitted. See [the plan record](docs/design/qwen4-program-retained-pm4.md).
- Retained-replay funnel coverage for the Qwen4 declarative program: the shared
  tensor-op owner (`rdna-compute::tensor_ops`) launched every Qwen4 GDN/QSA/HC
  op through the raw `launch_kernel_blob` entry, so those launches never reached
  `ReplayController` and a capture would have been a silently truncated tape.
  A new `Gpu::launch_blob_recorded` entry joins the same recorder, exact-bytes
  capture, and HipGraph capture-blob accounting as the params-shaped funnel, and
  all 31 raw sites migrated to it. Dynamic kernarg fields can now be *declared*
  at the lowering that computes them (`ReplayKernargBinding::PositionDivU32` /
  `PositionModU32`, carried on the recorded launch and merged at prepare with
  one owner per slot); the GDN convolution ring cursor declares the first such
  binding instead of being discovered by recording differencing. No route is
  admitted, no PM4 preparation is claimed, and the sealed-MoE pointer-contract
  refusal for specialized routes still fails closed. See [the plan record](docs/design/qwen4-program-retained-pm4.md).
- Rename Qwen4 CPU/reference modules to `reference_forward` and `reference_mtp`; production `gpu_forward`/`mtp_gpu` paths remain unchanged, the old public module paths are removed, and the reference modules are test/feature-gated.
- Sealed MoE calls lower to granular computation programs; Qwen root-routed EP decode and batched prefill share a checked collective schedule. Compact EP gathers expert outputs in global top-k slot layout and runs the ordinary single-device slot-order combine once on root before byte-broadcasting the finished partial, avoiding rank-grouped floating-point reassociation. Existing kernels, ownership, other-family reduction order, and diagnostic policies are retained. This does not admit new parallel axes or product replay routes; see [the design and validation boundary](docs/design/sealed-granular-moe.md).
- Add experimental `qwen3.8:flash-next` registry availability for the uploaded 178 GB HFQ artifact with a conservative 128 GB tested-hardware gate; the runtime minimum is unmeasured, and this does not change product or replay admission.
- Qwen4's production layer path now binds architecture-owned typed
  descriptors to neutral shared HyperRead/Write, GDN, QSA, grouped-depthwise,
  Clear, and sealed-MoE contracts. Composite steps are preflighted before
  token/embedding/PLE effects, exact row views preserve reusable arena
  capacity semantics, and QSA scalar metadata publishes only after success.
  Gfx1151 two-token and full natural `[128,128,35]` scalar/natural smokes are
  bit-exact; the full oracle also matches the immutable original, frozen HC,
  and frozen N8 references. Corrected AR and native MTP serve smokes both
  return `Paris` with `finish=stop`, `saw_done=true`, nonempty output, no
  runaway, and no stream error; MTP reports `tau=1.0`, one cycle, and
  `mtp=true`. This is an ownership/correctness seam only: no replay/PM4,
  throughput, or quality promotion claim. See [the current ownership
  record](docs/design/qwen4-shared-token-batched-prefill-progress-20260918.md).
- Qwen4's centralized bounded token-batched prefill now carries conservative HC row/grid-Y batching with scalar rows=1 unchanged and exact natural 128/128/35 oracle parity. Explicit marker evidence records frozen-N8 natural HC calls 112,326 versus candidate 1,158; fresh product ABBAAB remains below the requested 500 prefill / 25 decode tok/s thresholds (both open/blocked), with no product throughput or decode-win claim. A deterministic Paris fixture independently passes ordinary AR and native MTP correctness. See [the implementation record](docs/design/qwen4-shared-token-batched-prefill-progress-20260918.md) and [the historical measurement](docs/perf-checkpoints/2026-09-19-qwen4-hc-rows-gridy-final-measurement.md).
- The Qwen4 `gfx1151` prefill tuning lineage is documented across six staged
  levers: QSA selection-only, QSA selection-plus-attention, the selection
  sentinel, GDN shared exact-128 BF16 QK norm, the N8 measured-prefill
  multirow allowlist, and HC rows/grid-Y batching. These remain fixture-bound
  engineering records rather than six product promotions; see the
  [six-lever lineage](docs/design/qwen4-shared-token-batched-prefill-progress-20260918.md)
  and [final HC measurement](docs/perf-checkpoints/2026-09-19-qwen4-hc-rows-gridy-final-measurement.md).
- Add developer-only `HIPFIRE_EMULATE_GPUS` logical-rank aliasing and a single-gfx1151 Qwen EP4 batch diagnostic exception. Physical-device admission remains unchanged; logical-rank results do not prove physical EP transport, performance, or G5 acceptance.
- Fix grouped expert-parallel MoE outputs: rank-local down projections are unscattered to canonical token/top-k slots before gathering across ranks, so root combines in single-device slot order.
- **gfx1201 decode + Strix Halo round 4 (H2): retained Redline PM4 becomes the default gfx1201 decode backend for single-GPU Qwen3.5 dense, and lands with the gfx1201 decode norm grids, the gfx1201 GQA fp8 decode attention and the gfx1151 prefill round 4 (all byte-identical; R9700 H2 decode 36.44 → 38.89 / 35.28 → 38.08 / 31.44 → 35.93 tok/s at ctx 512/8,192/32,768, +6.71 %/+7.91 %/+14.30 %; Halo pp8192 1,128.1 → 1,140.9 tok/s).** gfx1201 decode goes from 27.44/28.34/31.81 to 25.72/26.26/27.83 ms per token against beta `1621a0090`. That was measured with separate md5-pinned binaries, four fresh processes per arm, ABBA +6.46/+7.77/+14.02 % then BAAB +6.64/+7.87/+14.29 %, `hipfire bench --pp 128 --ctx 512,8192,32768 --tg 128`, graph on, fp8 KV, 300 W. Every land process beat every beta process at every context. Each lever is priced by switching it off in the land binary, as tok/s gained with it on: Redline PM4 +2.16/+2.31/+2.25 % (−0.56/−0.61/−0.63 ms); DecNorm +1.41/+1.54/+1.52 % (−0.36/−0.41/−0.42 ms); DecAttn +2.16/+3.31/+9.99 % (−0.55/−0.87/−2.78 ms). In each of those comparisons, every default process beat every opted-out process. With Redline switched off (`replay.backend = "hip"`), the land binary still gains +4.45/+5.48/+11.78 % over beta. gfx1201 pp8192 is unchanged: 5,134.1 → 5,138.4 tok/s (+0.08 %) on the default route and 5,142.8 (+0.17 %) with `replay.backend = "hip"`. All four gfx1201 WT2/code24 `.kldseq` pins are unchanged (A4 `483cfc58…`/`6339dc9e…`, fp8 `19227d5f…`/`5e7ac9dc…`). On the default route, greedy decode (256 steps at ctx 512/8,192/32,768, synthetic and WikiText-2 primes) is byte-identical to beta: every step's logits and hidden state, plus the final KV arena and DeltaNet state. The Halo keeps both `.kldseq` pins (`c1056943…`/`04f06883…`) and its greedy decode text (`bd30cd27…`). Against the same beta, Halo pp8192 went +0.94 % (ABBA) and +1.20 % (BAAB), every land process ahead of every beta process, with tg1 unchanged. gfx1100 objects are unchanged. Opt-outs:
  - `HIPFIRE_G12_DEC_NORM=0` (or `kernel.g12_dec_norm = false`) turns off DecNorm.
  - `HIPFIRE_FP8_DECODE_ATTN_GQA=0` turns off DecAttn.
  - `replay.backend = "hip"` (or `HIPFIRE_REPLAY_BACKEND=hip`, or the wizard's `hip` profile) returns decode to the HIP AR graph.
  - On the Halo, `HIPFIRE_V2B_PM_BUNDLE=<path>` loads another V2B bundle (e.g. the round-2 `24a8d9eb…`), `HIPFIRE_V2B_PM=0` returns to hipcc V2B, and `HIPFIRE_GFX1151_GDN_SCAN=0` returns to `gdn_chunk_scan`.
  - **gfx1201 Redline default for Qwen3.5 dense plain-AR decode (byte-identical).** `retained_redline_default` now also admits: exact gfx1201, `model_arch` `qwen3_5` (dense, any weight format), pp = tp = 1 and no drafter. On those loads the daemon requests Auto/PM4 (`[redline] enabling fail-closed retained default on gfx1201 …`). The first eligible forward records as ordinary HIP; each later token is one retained PM4 IB. On H2 that IB is 899 launches of 22 kernels, all typed, with the same 80 elided waits as beta's tape. An explicit backend always wins. A failed tape preparation or replay falls back closed to the HIP graph. The GDN chunk-scan prefill is now withheld only while a forward is being recorded (`is_recording()`), not for the whole life of a Redline process. Before that fix, `--redline` lost 23.7 % of pp8192; now a Redline process prefills byte-identically to the HIP-graph default. The Radiowave fallback tables in `replay.rs` now list the GQA fp8 decode tile and dsplit reduce next to their twins (kernarg sizes and pointer effects). PM4 matches the HIP graph with every lever on, for every step's logits, hidden state, DeltaNet state and KV rows. This holds at ctx 512/8,192/32,768, under four poisons, in the `redline_daemon_harness.py` oracle and in a 200-replay stress. Documented in `docs/REDLINE.md` §3, including its known limits: a new gfx1201 decode kernel must be covered by Radiowave or the `replay.rs` tables, and `ROC_GLOBAL_CU_MASK` does not reach the PM4 queue.
  - **gfx1201 decode: the single-workgroup norms run as multi-workgroup grids (bit-exact; `HIPFIRE_G12_DEC_NORM=0` opts out).** Plain AR decode launched three kernels as one workgroup (or one wave) on the 64-CU R9700. On exact gfx1201 (`kernel.g12_dec_norm`, default on): (1) the f32 AWQ RMSNorm+FWHT producer before every MQ4 input projection (128 launches per H2 token) runs `fused_rmsnorm_mq_rotate_awq_g12dec` on grid K/256 instead of 1. Every workgroup redoes the row's sum of squares, with loads batched as in the `_v2` IU4 producers (same elements, same sequential fma chain), and the unchanged reduction tree, so rms is bit-identical; wave 0 then rotates only group `blockIdx.x`. (2) The out-of-place single-row `rmsnorm_f32` with n > 256 (the final norm before the LM head) runs `rmsnorm_f32_rowsplit`: n/256 workgroups, each recomputing the row's rms with the same fma chain and LDS halving tree and writing its own 256 outputs. (3) The half-split partial RoPE runs `rope_partial_halfsplit_f32_headgrid`: one workgroup per head instead of one wave looping over 28 heads. Each workgroup recomputes the same frequency, cos and sin. Traced per call on H2: 6.56 → 2.36 µs, 10.76 → 2.44 µs and 6.68 → 1.88 µs. Other arches, batched rows, in-place `rmsnorm_f32` and the legacy interleaved RoPE keep their launches. Redline replay descriptors and acquire policies list the new symbols next to their twins. Byte-identical to beta on captured H2 decode inputs at ctx 512/8192/32768 (whole buffers, guards, 3 poisons, negative controls detected, 200× serial and `ROC_GLOBAL_CU_MASK=0x1` stress, RoPE at every position 0…262,143); greedy and fixed-token decode logits of every step equal beta at ctx 512/8192 (and 32,768 fixed); H2 WT2/code24 `.kldseq` pins unchanged on both routes (fp8 `19227d5f…`/`5e7ac9dc…`, A4 `483cfc58…`/`6339dc9e…`). Same binary, four fresh processes per arm, ABBA then BAAB, card B 300 W: plain-AR decode 27.431 → 26.862 / 27.419 → 26.825 ms at ctx 512, 28.314 → 27.756 / 28.323 → 27.721 ms at 8,192, 31.741 → 31.203 / 31.763 → 31.198 ms at 32,768 (−0.54 to −0.60 ms/token; every on process beat every off process); pp8192 unchanged (+0.03 % / −0.08 %). Report: /home/kaden/qcal/perf/decode-1201/dec-norm/report.md.
  - **gfx1201 native-fp8 decode attention: GQA-shared flash tile + head-dim-split reduce (byte-identical).** On exact gfx1201, fp8 KV, head_dim 256, GQA group 6, tile 128 and no output gate (H2 decode), `attention_flash_fp8_e4m3_tile_gqa_gfx1201` replaces the one-wave-per-(q head, tile) `attention_flash_fp8_e4m3_tile`. It runs one 256-thread workgroup per (kv head, tile), so each K/V tile is read once for its six q heads, and it issues the whole tile's loads at entry. Its grid is capped at 512 tiles with a tile loop, so the graph-captured 2,048-tile grid dispatches few idle workgroups. `attention_flash_reduce_dsplit_gfx1201` replaces `attention_flash_q8_0_reduce`: one workgroup per (head, 32-dim chunk), with operands fetched in parallel instead of the reference's per-tile dependent scalar loads. Per-head arithmetic reproduces the reference ISA exactly: leaf FMA chains, the contracted xor-16 step, the same reduction tree, the reference softmax source and increasing-key/tile FMA chains. Partials and output are whole-buffer identical to beta on 18 real H2 decode captures (layers 0/7/15 at ctx 512/8,192/32,768; 3 poisons, 256 B guards, graph and eager grids, 200× serial stress, negative control detected), and the candidate binary's in-model captures are byte-identical to beta's. Greedy decode (256 tokens at ctx 512 and 8,192) produces lm_head logits byte-identical to beta at every step, in graph and eager mode; all four WT2/code24 `.kldseq` pins are unchanged; a kernel trace shows only the new pair with the lever on and only the reference pair with it off; and Redline PM4 decode text equals the HIP-graph route's. Per-layer tile+reduce on card A (rocprofv3): 37.2 → 9.7 µs at ctx 512, 87.6 → 45.2 µs at 8k, 293.6 → 144.6 µs at 32k. H2 decode, same binary with the opt-out off/on (four fresh processes per arm, ABBA then BAAB, graph on, clocks auto, card A): ctx 512 36.48 → 37.20 tok/s (+1.94%/+1.98%, −0.53 ms/token), ctx 8,192 35.31 → **36.40** (+3.05%/+3.11%, −0.85 ms/token), ctx 32,768 31.45 → 34.40 (+9.38%/+9.42%, −2.73 ms/token). Every lever-on process beat every opt-out process at all three contexts, and pp8192 is unchanged (5,123.7 → 5,124.4 tok/s, +0.07%/+0.01%). Report: /home/kaden/qcal/perf/decode-1201/dec-attn/report.md. `HIPFIRE_FP8_DECODE_ATTN_GQA=0` restores the reference pair.
  - **gfx1151 (Strix Halo) prefill round 4: ds_swizzle scale broadcast in the builder V2B and a gfx1151 GDN chunk-scan twin (both bit-exact; H2 pp8192 1,124.65 → 1,138.1 tok/s, +1.08 %/+1.26 %).** Both are selected on exact gfx1151 only. gfx1100 and gfx1201 are code-identical: the gfx1100 V2C and gfx1201 fp8/`_b1`/`_b1s` bundles re-emit byte-for-byte, and the shared `gdn_chunk_scan.gfx1201.hip` is unchanged. Both Halo `.kldseq` pins are unchanged with both levers on and with both off (WT2 `c1056943…`, KLD 0.068955; code24 `04f06883…`, KLD 0.051358). Greedy decode text is also unchanged, on == off (md5 `bd30cd27…`). Same binary, four fresh processes per arm, ABBA then BAAB, `hipfire bench --pp 512,4096,8192 --backend noslots --workload stateless`, graph on. Both levers against both opted out: pp8192 1,127.65 → 1,139.8 (+1.08 %) and 1,123.9 → 1,138.1 (+1.26 %); pp4096 +1.06 %/+1.35 %; pp512 +1.00 %/+1.42 %; tg1 unchanged (14.25–14.26).
    - **Builder V2B `ds_swizzle` scale broadcast (gfx1151).** The embedded `kernels/gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151.hxaco` is now `485d21ba…` (was `24a8d9eb…`). It is re-certified per symbol with `peacemaker custom build --arch gfx1151`: 256 VGPR, 0 spills, SGPR 54, LDS bound 65,536, M7 lift byte-exact, obligations {}, 0 ambiguous delays. The 32 DPP `row_share` scale moves per K128 epoch and wave become `ds_swizzle_b32` broadcasts on the LDS crossbar. Pass 0's broadcast and its f16→f32 convert issue after WMMA step 2 in SET and gate/up, and after step 6 in the FFN-down ADD; each later pass's broadcast issues right after the previous fold. The contracts now pin 0 `v_mov_b32_dpp` and 128/256/128 `ds_swizzle_b32` (SET/ADD/SiLU). Fold ops, order and DAG are unchanged. Outputs are byte-identical to hipcc V2B and to `24a8d9eb` on real H2 tensors (SET M6400 K5120, ADD M5120 K17408 GSHIFT 2, gate/up M17408 K5120) at N 8192/1024/1023/513/512/256/255: whole buffer plus 256 B guards, 200× serial and 200× single-CU co-resident stress over 3 poisons, and a negative control detected at 21/21 cases. Kernel time on real operands: SET −1.13 %, gate/up −1.10 %, ADD −0.74 %. H2 pp8192 with this lever alone (opt-out `HIPFIRE_V2B_PM_BUNDLE=<24a8d9eb bundle>`): 1,136.55 → 1,140.55 (+0.35 %) and 1,130.0 → 1,138.6 (+0.76 %).
    - **GDN chunk scan twin (gfx1151).** `gdn_chunk_scan_gfx1151` (`kernels/src/gdn_chunk_scan.gfx1151.hip`, `-ffp-contract=off`) runs the gfx1201 pipelined chunk loop with gfx11 operand and accumulator layouts. K is staged transposed in the dead z tile, so the state update reads rows instead of 16 `ds_load_u16` per fragment. Each 256-thread workgroup owns one BV=64 value half of a head: grid [2, 48], 243 VGPR, 35,328 B LDS. Every LDS value, WMMA operand, per-accumulator order and rounding is the shipped kernel's. It is byte-identical to `gdn_chunk_scan`, checked on two captured H2 layers: `out` plus guards, and the q8/scale/EF state. Layer 2 was checked at N 8192/1024/1023/513/512/256/255, and layer 50, a later chunk with non-zero initial state, at N 4291/1024/1023/513/512/256/255. Both layers passed 3 poisons, 200× serial and 200× single-CU stress, and a negative control detected at 14/14 N; the twin also replays the in-model captures byte-for-byte. KKT + scan per layer: −14.7 %. `HIPFIRE_GFX1151_GDN_SCAN=0` restores `gdn_chunk_scan`. H2 pp8192 with this lever alone: 1,135.4 → 1,139.8 (+0.39 %) and 1,131.1 → 1,139.85 (+0.77 %).
- **gfx11 round 3 (H2 pp8192, 7900 XTX gfx1100 + Strix Halo gfx1151): the exact-gfx1100 FA2 variant and the gfx1151 FA2 twin land together (both bit-exact).** Each card runs its own FA2 module, selected by exact arch in the gfx11 Q8 FA2 launcher: `attention_q8_0_fa2_gqa_gfx1100` on gfx1100 only (`HIPFIRE_GFX1100_FA2_R3=0` opts out), and `attention_q8_0_fa2_gqa_gfx1151`, launched whole-chunk, on gfx1151 only (`HIPFIRE_GFX1151_FA2_TWIN=0` / `HIPFIRE_GFX11_Q8_FA2_WIDE=0` opt out). Every gfx11 FA2 object has the same `.text`/`.rodata` as its own branch (R3 `e3f852fa…`, twin `643bff93…`, shared gfx11 fill `3b564d7c…`/`684fdd63…`). gfx1201 is code-identical to beta: every gfx1201-buildable JIT source is unchanged, and all builder bundles (gfx1100 `cad59b86…`, gfx1151 `24a8d9eb…`, fp8 `73168098…`, `_b1` `c6e714e7…`, `_b1s` `d5190d97…`) re-emit byte-for-byte. All four gfx11 `.kldseq` pins are unchanged (XTX `ccf95389…`/`a7eac2b4…`, Halo `c1056943…`/`04f06883…`), and greedy decode text is unchanged on both cards. A new in-model check pushed a real 12,483-token prompt through the daemon on both cards; its first chunk ran as one 8,192-row launch per layer. The FA2 outputs of all 16 attention layers were SHA-256-identical, in 512-row blocks, to the opt-out and to beta (on the Halo, against the sixteen 512-row gfx11 commits). Against beta `7cf333271` (separate binaries, four fresh processes per arm, ABBA then BAAB, `--backend noslots --workload stateless`, clocks auto), pp8192 went: XTX 2,928.20 → 3,001.30 tok/s (+2.50%) in ABBA and 2,925.40 → 2,998.60 (+2.50%) in BAAB, median of process medians 2,925.6 → **2,999.8**; Halo 1,113.25 → 1,138.70 (+2.29%) in ABBA and 1,107.80 → 1,131.70 (+2.16%) in BAAB, median 1,108.95 → **1,133.4**. On both cards every land process beat every base process at pp8192 and pp4096. Report: /home/kaden/qcal/perf/gfx11-land-r3/report.md.
  - **gfx1100 (7900 XTX) FA2 prefill attention: exact-gfx1100 variant of the Q8 FA2 fill body, bit-exact (kernel −24.7 % at N4096@4096, −28.5 % at N4096@0 on a real H2 layer; H2 pp8192 on the XTX 2,927.95 → 3,003.0 tok/s, +2.40 %/+2.52 %).** The fill helper waves' K/V plane stores were bank-conflicted (V 8-way, K 4-way) and stalled the compute waves sharing their SIMDs. The variant makes those stores conflict-free (a helper wave's lanes cover 16 key pairs or keys, and half the lanes store the neighbouring bank group), builds in CU mode (3 compute + 1 helper wave per SIMD), skips the O rescale when every lane's alpha is exactly 1.0f, starts sacc from opaque zeros with clamped Q rows (188 → 181 VGPR), and dispatches the heaviest q tiles first. Symbols `attention_q8_0_fa2_gqa_gfx1100` / `attention_fa2_q_preconvert_gfx1100`; `HIPFIRE_GFX1100_FA2_R3=0` restores the shared gfx11 body. The gfx1151 object and every other build of `attention_q8_0_fa2_gqa.gfx11.hip` are code-identical. XTX WT2/code24 `.kldseq` are unchanged (`ccf95389…`/`a7eac2b4…`) with the variant on and off. A kernel trace of code24 shows 384 + 384 launches of the gfx1100 pair with the variant on and of the gfx11 pair with it off; every other kernel family's count is equal. Greedy decode text is unchanged (on == off). Same binary with `HIPFIRE_GFX1100_FA2_R3` off/on (4 fresh processes per arm, ABBA then BAAB, `--backend noslots --workload stateless`, q8 KV, graph on): pp8192 2,933.85 → 3,004.35 tok/s (+2.40 %) and 2,926.45 → 3,000.05 (+2.52 %), median of the process medians 2,927.95 → 3,003.0; pp4096 +1.50 %/+1.56 %; pp512 +0.32 %/+0.24 %; tg1 −0.21 %/−0.60 %.
  - **gfx1151 (Strix Halo) prefill attention: a gfx1151 twin of the gfx11 Q8 FA2 kernel, launched whole-chunk (bit-exact).** `attention_q8_0_fa2_gqa_gfx1151` (`kernels/src/attention_q8_0_fa2_gqa.gfx1151.hip`) is the gfx11 fill build with the same arithmetic, compiled `-mcumode` (each SIMD holds three compute waves and one helper; WGP mode put 2,1,2,1 compute waves per SIMD and a `buffer_gl0_inv` after every barrier), dispatching the heaviest q-tiles first, with opaque zero-init of the QK accumulators (188 → 180 VGPR), clamped unconditional Q loads, helper V stores that cover 32 LDS banks per store, and the O rescale skipped when alpha is 1.0 on every lane. `kernel.gfx11_q8_fa2_wide` auto now enables gfx1151 as well, so one launch per layer replaces 16 per-commit launches (whose 128 workgroups filled only ~63 of 80 slots). On a captured real H2 layer (sustained 130 W): 42.75 → 30.23 ms per layer (ABBA −27.5 %, BAAB −29.7 %), gfx energy −35.8 %; byte-identical at N 8192/1024/1023/513/512/256/255 and the 16-commit layer, 200× serial and co-resident stress. `HIPFIRE_GFX1151_FA2_TWIN=0` restores the gfx11 module; `HIPFIRE_GFX11_Q8_FA2_WIDE=0` the per-commit launches. gfx1100/gfx1201 objects are unchanged (`.text`/`.rodata` of the gfx11 FA2 module on gfx1100 and of the gfx1201 FA2 module identical to `7cf333271`). Both Halo `.kldseq` pins are unchanged with the twin and with both opt-outs (WT2 `c1056943…`, code24 `04f06883…`), and greedy decode text is unchanged. A kernel trace of the bench route shows 64 whole-chunk `attention_q8_0_fa2_gqa_gfx1151` launches (512 × 4 workgroups) where the opt-out ran 1,024 per-commit gfx11 launches. H2 pp8192 on the Halo, same binary, defaults against both opt-outs, four fresh processes per arm: 1,112.65 → 1,136.30 tok/s (+2.13%) in ABBA and 1,110.55 → 1,134.00 (+2.11%) in BAAB; the median of process medians is 1,134.85 tok/s (opt-out 1,110.55). pp4096 +0.88%/+1.40%; pp512 and tg1 are unchanged.
- **gfx11 round 2 (H2 pp8192, 7900 XTX gfx1100 + Strix Halo gfx1151): the gfx1100 builder V2C round 2 and the two gfx1151 round-2 levers land together.** Each is selected by exact arch at dispatch: the gfx1100 bundle `cad59b86…` on gfx1100, and the gfx1151 V2B bundle `24a8d9eb…` and `qwen35_fa_prep_batched_gfx1151` on gfx1151 only. gfx1201 is code-identical to beta: its fp8, `_b1` and `_b1s` bundles re-emit byte-for-byte, and every JIT object has identical `.text`/`.rodata`. All four gfx11 `.kldseq` pins are unchanged (XTX `ccf95389…`/`a7eac2b4…`, Halo `c1056943…`/`04f06883…`), and greedy decode text is unchanged on both cards. Against beta `0ff072458` (separate binaries, four fresh processes per arm, ABBA then BAAB, `--backend noslots --workload stateless`, clocks auto), pp8192 went: XTX 2,851.20 → 2,925.15 tok/s (+2.59%) in ABBA and 2,849.00 → 2,923.35 (+2.61%) in BAAB, median of process medians 2,849.9 → **2,924.05**; Halo 1,103.60 → 1,117.60 (+1.27%) in ABBA and 1,098.30 → 1,113.10 (+1.35%) in BAAB, median 1,099.55 → **1,115.3**. On both cards every land process beat every beta process at pp512/4096/8192. Report: /home/kaden/qcal/perf/gfx11-land-r2/report.md.
  - **gfx1100 builder V2C round 2 (XTX): ds_swizzle scale broadcast, bank-aware fold pairing, gate/up wave priority.** The embedded `gemm_mq4g256v2_residual_iu4_pm_gfx1100.hxaco` is now `cad59b86…` (was `00c4747c…`), re-certified per symbol (M7 obligations {}). The 16 DPP `row_share` scale moves per K128 epoch run as `ds_swizzle_b32` on the LDS crossbar, issued ahead of each fold pass; SET moves fold products out of the accumulator's VGPR bank; ADD and gate/up pair each fmac with a later fragment's magic add over four distinct banks (sums re-laid to `acc + (j^2)`); gate/up runs its K loop at `s_setprio 1`. Same ops in the same per-element order: outputs byte-identical to hipcc V2C on 14 real H2 sites × 7 N × 3 poisons, 200× serial and 200× single-CU co-resident stress, negative controls detected; H2 WT2/code24 `.kldseq` equal the gfx1100 pins (`ccf95389…`/`a7eac2b4…`) with the new bundle and with the old one; greedy ~7k-token decode text is unchanged. Kernel time vs `00c4747c…` at N8192: gate/up −4.6 %, FFN down −3.2 %, K6144 ADDs −2.5 %, QKV SET −1.0 %. H2 e2e on the 7900 XTX (same binary, `HIPFIRE_GFX1100_PM_BUNDLE=<00c4747c…>` as the off arm, 4 fresh processes per arm, ABBA+BAAB, `--backend noslots --workload stateless`, graph on, clocks auto): pp8192 2,850.95→2,924.45 tok/s (+2.58 %) and 2,849.60→2,923.10 (+2.58 %), median 2,850.6→2,924.15; pp4096 +2.74/+2.78 % (3,071.1 tok/s), pp512 +3.08/+3.41 %; tg1 unchanged. `HIPFIRE_GFX1100_PM_BUNDLE=<path>` loads another bundle for same-binary A/B. Receipts: `/home/kaden/qcal/perf/gfx11-3/xtx-r2/report.md`.
  - **gfx1151 (Strix Halo) prefill round 2: two bit-exact levers.** (1) The certified builder V2B bundle (`kernels/gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151.hxaco`, SHA-256 `24a8d9eb…`) places each fold product at `t + (j^2)`, so no `v_dual_fmac_f32` half reads one VGPR bank three times (a per-packet ablation priced fmac packets at 2.3–2.9 cycles against ~1.1 for mul/add), and the SET stores each pass's final sums during its last epoch (one b128 per WMMA step, per-fragment SGPR Y bases) instead of bursting them after the K loop. Same op DAG and output bytes: whole-buffer identity against hipcc V2B on real H2 tensors at N 8192/1024/1023/513/512/256/255 (3 poisons, guards, negative control detected at every N and site); `peacemaker custom build --arch gfx1151` passes all three contracts (0 M7 obligations). Kernel time vs the previous bundle (10 s arms, ABBA+BAAB, per-arm SMU clock): SET −1.97%/−2.29%, FFN-down ADD −1.17%/−1.93%, gate/up SiLU −1.28%/−1.48%. `HIPFIRE_V2B_PM_BUNDLE=<path>` loads another builder bundle (e.g. the previous one) as a same-binary opt-out. (2) Ordinary gfx1151 prefill runs `qwen35_fa_prep_batched_gfx1151` (the gfx1100 batched FA prep source under its own symbol) instead of deinterleave+Q-norm, K rmsnorm and half-split RoPE: byte-identical to the chain on a captured real H2 layer call at all seven N (and the replayed chain equals the in-model outputs), 6.40 → 4.29 ms per N8192 call. `HIPFIRE_GFX1151_FA_PREP=0` restores the chain. gfx1100 and gfx1201 JIT objects and builder bundles are code-identical. `pmprof` gains a `v2b` mode with gfx1151 SET/SiLU profile points. H2 WT2/code24 `.kldseq` equal the gfx1151 pins (`c1056943…`/`04f06883…`) with the levers on and off, greedy decode text is unchanged, and 200× serial and single-CU co-resident stress passes for every GEMM site and the FA prep at all seven N. H2 pp8192 on the Halo (`hipfire bench --backend noslots --workload stateless`, fresh process per run, ABBA+BAAB), each lever against its own opt-out: bundle 1,106.6→1,114.95 / 1,102.3→1,114.5 tok/s (+0.76%/+1.11%), FA prep 1,111.55→1,113.6 / 1,104.3→1,109.8 (+0.18%/+0.50%), both 1,095.8→1,108.95 / 1,094.5→1,108.65 (+1.20%/+1.29%); decode tok/s unchanged (see `qcal/perf/gfx11-3/halo-r2/report.md`).
- **gfx1201 native-fp8 prefill: the F2 Row GEMMs batch and pipeline their fold ratio loads, and the fused QKVZA+GDN epilogue pads its LDS ring rows (bit-exact).** Three builder changes in the embedded F2 bundle `gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco` (SHA-256 `73168098…`; the five K128 symbols are code-identical). (1) Each Row fold loads the eight ratios of a row group with two `ds_load_b128` instead of eight `ds_load_b32` pairs: 4 LDS waits per K128 instead of 16. (2) Those loads run one row group ahead, alternating between v168..v175 and v[183:186]/v[188:191], which hold nothing in a Row K region; one exposed wait per fold. (3) `gemm_mq4g256v2_fp8_qkvzagdn_row_b1` spaces its 19 GDN ring rows 1,040 bytes apart instead of 1,024 (a 16-byte bank skew), and declares the 304 bytes beyond the 19,456-byte launch allocation as fixed LDS in its kernel descriptor; the host launch is unchanged. Same ratios and the same multiplies in the same order: outputs are byte-identical to the previous bundle on real H2 activations at all ten routed sites and N 8192/1024/1023/513/512/256/255 (whole buffers and guards, three poisons), and under 200× serial and single-CU co-resident stress; fp8 and A4 WT2/code24 `.kldseq` pins unchanged. Same-binary card-C H2 A/B, graph on, four fresh processes per arm (ABBA+BAAB), embedded bundle vs the previous one through the override: native-fp8 (`HIPFIRE_IU4_PREFILL=0`) pp8192 3,856.25→3,935.35 tok/s (+2.05%; 2,124.3→2,081.6 ms per forward), +1.94% in ABBA and +2.02% in BAAB, every new-bundle process faster than every old-bundle process; pp4096 +2.38%, pp512 +2.33%; standalone kernels −1.2% to −2.8% wall (gate/up −1.6%/−2.5%, fused QKVZA+GDN −2.8%). Report: /home/kaden/qcal/perf/fp8-rb2g/report.md. Opt-out: `HIPFIRE_G12_FP8_F2_BUNDLE=<path>` (developer override) loads the previous bundle from a file. `peacemaker custom build` contracts may now declare `group_segment_fixed_bytes` (the kernel descriptor's fixed LDS, default 0).
- **gfx1201 a4-push3 (H2 A4 pp8192 ≥ 5,000 tok/s): three bit-exact A4 prefill levers land together.** The builder's iu4 `_b1s`/`_b1` GEMMs take the fold's magic as a VOPD literal, keep each fmac's product in another VGPR bank and run their K-loops at wave priority 1; the RMSNorm→A4 slab producer divides by AWQ through load-time reciprocal planes; and the GDN chunk scan runs one launch per layer across all commit segments. Each is detailed below. The combined binary keeps all four WT2/code24 `.kldseq` pins byte-identical (A4 launches only the new `_b1s` bundle `d5190d97…`, the `_fdiv` producer and `gdn_chunk_scan_bf16_mseg`), and greedy decode text is unchanged on both the A4 and fp8 routes. Against beta `da629b75d` (separate md5-pinned binaries, four fresh processes per arm, ABBA then BAAB, graph on, 300 W, card A), A4 pp8192 went 4,870.20 → 5,113.70 tok/s (+5.00%) in ABBA and 4,870.00 → 5,108.55 (+4.90%) in BAAB; the median of process medians went 4,870.0 → **5,110.3 tok/s** (1,682.1 → 1,603.1 ms per forward), and every land process beat every beta process (pp4096 +5.08%/+5.11%, pp512 +5.08%/+4.95%). The fp8 route runs the same GPU code (every JIT object `.text`/`.rodata`-identical to beta, no iu4 bundle loaded; the reciprocal planes add 5 MB at load): fp8 pp8192 on card B went 3,875.30 → 3,872.75 tok/s (−0.07%) in ABBA and 3,881.50 → 3,879.40 (−0.05%) in BAAB, and a second run gave −0.18%/−0.01% (median of medians 3,878.2 → 3,881.8). The same protocol with the beta binary in both arms gave −0.06%/−0.08%, so the fp8 difference is inside the A/A spread.
  - **gfx1201 A4 (iu4) `_b1s`/`_b1` GEMMs: cheaper fold issue and epilogue wave priority, bit-exact (H2 A4 pp8192 4,863.65 → 5,081.00 tok/s, +4.15%/+4.45%).** Three scheduling/operand changes in the builder's iu4 K-loop and epilogues (`crates/hipfire-isa`, all five symbols of both bundles: SET, ADD, gate/up SiLU, packed-bf16 SiLU and the fused `qkvzagdn`): (1) **literal magic** — the fold's stage-1 `v_dual_subrev_f32` subtracts the magic 12582912.0f as the packet's shared VOPD literal `0x4b400000` instead of reading the loop-invariant `v128..v135` copies (which now only seed the WMMAs); (2) **wave priority** — every symbol runs its K-loop at `s_setprio 1` and drops to 0 at its epilogue label (the fused projection already did this for its GDN epilogue; its stores stay at 1), so co-resident K-loop waves win issue arbitration over a finishing tile; (3) **bank-aware fold packets** — each fmac half used to read its product, C and acc from the same VGPR bank (`j % 4`); the product of element `j` now lives at `t + (j ^ 2)`, and row group 1's stage-1 subrev rides in row group 0's fmac packets as the literal-subrev half (`fmac :: subrev`), so the fold keeps its 96 packets per K128 block but no packet reads one VGPR bank more than twice (fmac packets read it three times before). Every element keeps its ops and their order (`float(C) - magic`, `RN(sc*d)`, `fma`), so outputs are byte-identical: whole-buffer oracle on captured real H2 launches (SET/ADD/SiLU/SiLU-bf16 at N 8192/1024/1023/513/512/256/255, both layouts, guards, 3 poisons, N=8192 replays equal to the in-model outputs; fused projection on real and synthetic activations), a negative control (literal off by one ulp) mismatching every case, 200× serial and co-resident (`ROC_GLOBAL_CU_MASK=0x1`) stress of the slab SET/ADD/SiLU/SiLU-bf16 symbols, and all four WT2/code24 `.kldseq` pins unchanged. Kernel time on real captured H2 launches (card A, 300 W, 10 s/arm, ABBA/BAAB): gate/up SiLU −3.65 %/−4.06 %, FFN down −2.60 %/−2.62 %, K6144 ADD −3.86 %/−4.13 %, attention Q+gate SET −3.39 %/−3.73 %, K/V SET −4.75 %/−4.70 %, fused GDN input projection −4.04 %/−4.31 %. The operand pairing alone (without the bank move) was killed at −0.13 %/−0.24 % on gate/up. H2 A4 pp8192, same binary, embedded bundles vs the opt-out (card A, 300 W, graph on, four fresh processes per arm, ABBA then BAAB): 4,881.45 → 5,083.95 tok/s (+4.15%) in ABBA and 4,863.20 → 5,079.70 (+4.45%) in BAAB; median of process medians 4,863.65 → **5,081.00 tok/s** (1,684.33 → 1,612.28 ms per forward, -72.1 ms); pp4096 +4.18%/+4.53%, pp512 +4.62%/+4.75%. Every opt-in process beat every opt-out process at all three sizes. Opt-out: the developer bundle override `HIPFIRE_G12_IU4_B1S_BUNDLE`/`HIPFIRE_G12_IU4_B1_BUNDLE` pointed at the previous bundles (SHA-256 `3afa298e…`/`b884f5f7…`, e.g. `git show da629b75d:kernels/<name>.hxaco`). New bundles: `_b1s` `d5190d97…` (83,504 B), `_b1` `c6e714e7…` (83,224 B); peacemaker-certified (187/192 VGPR, 0 spills), shape contracts updated for the new packet census (per symbol 64 subrev-first and +64 fmac-first packets).
  - Exact gfx1201 A4 (iu4) prefill no longer runs an IEEE divide per element in the RMSNorm→A4 slab producer. The loader appends two planes, R = RN(1/a) and Rlo = RN(fma(−a, R, 1)·R), to each AWQ scale that an RMSNorm producer feeds (DeltaNet `in_proj_qkv`, attention `q_proj`, `mlp.gate_proj`; the scale tensor becomes `[k, 3]`, +5.24 MB on H2). They are computed at every load from the checkpoint, never cached. The new `fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2_slab_fdiv` twin forms each (x·w·rms)/a as q0 = fma(y, R, y·Rlo), q = fma(fma(−a, q0, y), R, q0) (one Markstein correction). Its wave falls back to the IEEE divide unless every |q| lies in [2^-80, 2^80]. An exhaustive gfx1201 check (all 2^32 f32 y against every f16 scale value, every distinct H2 scale, and the range extremes; 3.06e14 inputs, 0 mismatches) shows it equals the divide, and the slab bytes, the four `.kldseq` pins and the decode text are unchanged. H2 in-model RMSNorm 414.6→377.0 µs/call (−4.8 ms per pp8192 forward); pp8192 on card B, same binary, 4 fresh processes per arm: 4,904.65→4,905.85 tok/s (+0.02%) ABBA, 4,905.50→4,916.65 (+0.23%) BAAB. `HIPFIRE_A4_RMS_FDIV=0` opts out.
  - Exact gfx1201 A4 (iu4) GDN chunk-scan prefill now runs one `gdn_chunk_scan_bf16_mseg` launch per layer instead of one `gdn_chunk_scan_bf16` launch per 512-row commit segment, when the batched KKT solve has written A for every row. The launch walks the segments in order. Between segments the state takes the per-launch q8/scale/EF round trip in registers (same row max, `rint`/clamp, f16 residual, contraction off), and only the last segment stores it. That removes 15 launch gaps and 15 state stores and reloads per H2 layer; the scan `out` plane and the final state are byte-identical (real-activation oracle at 7 N, co-resident stress, four `.kldseq` pins). Standalone, a layer drops from 1.089 to 0.973 ms (−10.7%). pp8192 on card B, same binary, 4 fresh processes per arm: 4,915.20→4,928.60 tok/s (+0.27%) ABBA, 4,911.60→4,927.60 (+0.33%) BAAB. `HIPFIRE_GDN_SCAN_MSEG=0` keeps the per-segment launches.
- **gfx11 land (H2 pp8192, 7900 XTX gfx1100 + Strix Halo gfx1151): the gfx1100 builder V2C, the gfx1151 builder V2B with the V2B down raster and Z|β|α scatter, and three gfx1100 producer levers land together.** Every kernel is selected by exact arch at dispatch (V2C tile and levers on gfx1100 only, V2B tile on gfx1151 only), and all are bit-exact: H2 WT2/code24 `.kldseq` equal beta's per-arch pins on both cards (gfx1100 `ccf95389…`/`a7eac2b4…`, gfx1151 `c1056943…`/`04f06883…`) and greedy decode text equals beta's. Each arch's JIT `.text`/`.rodata` equals its own branch's, and the embedded `gemm_mq4g256v2_residual_iu4_pm_gfx1100.hxaco` (`00c4747c…`) and `…_pm_v2b_gfx1151.hxaco` (`7a884d2a…`) re-certify byte-identically from the combined builder; gfx1201 builder bundles are byte-identical to beta, its JIT objects code-identical, and its A4 WT2/code24 pins unchanged. Against a separate beta `da629b75d` build (four fresh processes per arm, ABBA then BAAB, `--backend noslots --workload stateless`, graph on, clocks auto): XTX pp8192 2,675.40→2,845.45 tok/s (+6.36%) and 2,671.40→2,839.05 (+6.28%), median 2,672.45→2,841.5 (pp4096 +6.6%, pp512 +7.4%); Halo pp8192 1,050.90→1,092.60 (+3.97%) and 1,043.75→1,088.50 (+4.29%), median 1,045.65→1,090.6 (pp4096 +4.0/+4.4%, pp512 +3.5/+4.0%); tg1 unchanged on both. Opt-outs: `HIPFIRE_GFX1100_PM_GEMM=0`, `HIPFIRE_GFX1100_GATED_NORM_V2=0`, `HIPFIRE_GFX1100_FA_PREP=0`, `HIPFIRE_RMSNORM_P1A_BATCHED=0` (gfx1100); `HIPFIRE_V2B_PM=0`, `HIPFIRE_V2B_DOWN_SWZ=0`, `HIPFIRE_V2B_ZBA_SCATTER=0` (gfx1151). Receipts: `/home/kaden/qcal/perf/gfx11-land/report.md`.
  - Exact gfx1100 symmetric MQ4V2 IU4 prefill now runs every V2C GEMM (SET, ADD and the F1-lite gate/up SiLU) from an embedded, certified builder code object, `gemm_mq4g256v2_residual_iu4_pm_gfx1100.hxaco` (`hipfire-isa emit --kernel iu4_v2c --epi all`; one contract-checked `peacemaker custom build --arch gfx1100` per symbol, M7 obligation-free). It is the V2C algorithm with the same grid, ABI and fold order, launched as one 256-thread block, with every fold op VOPD-paired: 96 packets + 25 other VALU per K128 epoch where hipcc V2C issues 89 + 90 (SET) and 65 + 139 (gate/up). The gate/up epilogue instantiates hipcc's gfx1100 `SILU_MUL` DAG as an imported golden region (`crates/hipfire-isa/kernels/iu4_v2c.gfx1100.silu.region.s`, re-sliced from a fresh hipcc object by a test). ADD touches the residual tile over epochs E-16..E-13 (hipcc `_add_touch`'s window, EXEC-gated inside the loop), and each epoch publishes its staged LDS packet before fold pass 1. Outputs are byte-identical to hipcc V2C on real H2 operands at every site and N (200× serial and single-CU stress), and WT2/code24 `.kldseq` equal beta's gfx1100 pins. Standalone at N4096 the kernels take 0.916× (gate/up), 0.931× (FFN down), 0.908× (GDN/attn out) and 0.930–0.955× (SET sites) of hipcc V2C's time. H2 on the 7900 XTX, same binary, 4 fresh processes per arm: pp8192 2,678.55→2,825.15 tok/s (+5.47%) in ABBA and 2,680.00→2,828.95 (+5.56%) in BAAB, pp4096 +5.7/+5.8%, pp512 +6.2/+6.6%, tg1 unchanged; greedy decode text identical. `HIPFIRE_GFX1100_PM_GEMM=0` restores the hipcc V2C entries; gfx1151 and gfx1201 objects and routes are unchanged. Receipts: `/home/kaden/qcal/perf/gfx11-2/xtx-builder/report.md`.
  - Exact gfx1151 (Strix Halo) V2B prefill: two gfx1151-only twins in `gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip` (under `#if defined(__gfx1151__)`, so gfx1100 objects keep beta's `.text`/`.rodata`). FFN down runs `_add_touch_swz`, `_add_touch` with a grouped CTA raster (4 row tiles fastest per group, `HIPFIRE_V2B_DOWN_SWZ=<rows>`, `0` restores the token-fastest order), so each dispatch round re-reads its 80 MB A4 activation from the 32 MiB MALL instead of DRAM: 20.04→19.02 ms per N8192 call. The GDN Z|β|α SET runs `_set_zba`, whose epilogue writes Z, β and α directly, so `split_mq4v2_z_betaalpha` no longer runs on the V2B route (`HIPFIRE_V2B_ZBA_SCATTER=0` restores SET + split): 9.18→7.34 ms per N8192 call. Both are byte-identical: real-activation whole-buffer poisoned parity passed 200 serial and 200 single-CU repeats at N 8192/1024/1023/513/512/256/255, and H2 WT2/code24 `.kldseq` on gfx1151 equal beta's (`c1056943…`/`04f06883…`). Same-binary graph-on H2 pp8192 on gfx1151, four fresh processes per arm, both levers against both off: 1,041.80→1,056.75 tok/s (+1.435%) in ABBA and 1,040.50→1,057.60 (+1.643%) in BAAB (pp4096 +1.185%/+1.708%, pp512 +0.817%/+0.786%); each lever alone was faster in both orders (down raster pp8192 +1.120%/+0.730%, Z|β|α +0.976%/+1.179%). Evidence: `/home/kaden/qcal/perf/gfx11-1/halo-levers/report.md`.
  - Exact gfx1151 (Strix Halo) V2B prefill now runs its SET, FFN-down ADD and fused gate/up SiLU from a certified builder code object (`kernels/gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151.hxaco`, `hipfire-isa emit --kernel iu4_v2b --epi all --arch gfx1151`) instead of the hipcc V2B `_set`, `_add_touch(_swz)` and `gate_up_silu` entries. Same M256×N256 16-wave tile and output bytes (whole-buffer identical on real H2 tensors at N 8192/1024/1023/513/512/256/255, 200× serial and single-CU stress); every fold op is VOPD-paired (234 VALU slots per K128 epoch per wave against hipcc's 313–355), the staged packet is published before the last pass and the last pass's fold runs behind the next epoch's loads. The ADD entry keeps the residual touch and takes the down raster as a group shift. Z|β|α (`_set_zba`) and the plain `_add` (`HIPFIRE_V2B_ADDEPI=0`) stay on hipcc. `HIPFIRE_V2B_PM=0` restores the hipcc entries. Kernel time on real H2 N8192 shapes (≥10 s arms, ABBA+BAAB, faster in both orders): SET −3.9%, FFN down ADD −3.8%, gate/up SiLU −3.8%; WT2/code24 `.kldseq` equal beta's gfx1151 pins in both arms and greedy decode is identical. H2 e2e on Halo (same binary, 8 fresh processes, 4 per arm, ABBA+BAAB, `--backend noslots --workload stateless`, builder vs `HIPFIRE_V2B_PM=0`): pp8192 1,097.1 vs 1,071.85 tok/s (+2.36%) in ABBA and 1,088.7 vs 1,066.5 (+2.08%) in BAAB, median 1,090.4 vs 1,068.85 (+2.02%); pp4096 +2.33%, pp512 +2.90%, tg1 +0.02%. gfx1100 objects are unchanged. Receipts: `/home/kaden/qcal/perf/gfx11-2/halo-builder/report.md`.
  - Exact gfx1100 (7900 XTX) prefill gains three bit-exact producer levers, each a gfx1100-only twin selected at dispatch (gfx1151 and gfx1201 dispatch unchanged symbols; JIT `.text`/`.rodata` identical for all 138 (const, arch) pairs). (1) The gated-norm A4 producer before the linear-attention `wo` runs one wave per 256-group (`gated_norm_mq_rotate_{awq_,}i4_gfx1100_v2`, the gfx1201 wave-group body, grid [ceil(K/512), N]): 490 → 329 µs per N4096 call; `HIPFIRE_GFX1100_GATED_NORM_V2=0` opts out. (2) Ordinary (non-DFlash) full-attention prefill uses the batched `qwen35_fa_prep_batched_gfx1100` prep instead of deinterleave + K RMSNorm + RoPE: 884 → 558 µs per N4096 call; `HIPFIRE_GFX1100_FA_PREP=0` opts out. (3) The IU4 RMSNorm producers run `fused_rmsnorm_mq_rotate_{awq_,}i4_b8` (Phase-1a 8 loads in flight + drain; the gfx11 drain is `s_waitcnt vmcnt(0)`): 226 → 220 µs per call; `HIPFIRE_RMSNORM_P1A_BATCHED=0` opts out. Whole-buffer byte identity at N 8192/1024/1023/513/512/256/255 × 3 poisons, 200× serial and co-resident stress, negative controls detected; WT2 and code24 24-chunk `.kldseq` equal beta's gfx1100 pins (`ccf95389…` / `a7eac2b4…`) with levers on and all off; greedy ~7k-token decode is identical. H2 e2e on the XTX (same binary, 8 fresh processes, 4 per arm, ABBA+BAAB, `--backend noslots --workload stateless`, vs all levers off): all on pp8192 2,683.15 → 2,697.3 (+0.53%) and 2,675.6 → 2,691.0 (+0.58%), pp4096 +0.81/+0.91%, pp512 +0.78/+1.04%; alone, gated norm +0.34/+0.25%, FA prep +0.24/+0.24%, RMSNorm +0.06/+0.05% at pp8192. The prefill-chunk receipt now re-prints whenever the admitted rows change, exposing that gfx1100 pp8192 requests after the first fall back to 2×4096 chunks (the released widened PBS stays in the process pool, invisible to admission); extending the gfx1201 grow-only PBS route to gfx1100 fixed that but measured −0.21%/−0.16% at pp8192, so gfx1100 keeps the fallback. Receipts: `/home/kaden/qcal/perf/gfx11-1/xtx-levers/report.md`.
  - `hipfire-isa` gains a gfx11 (gfx1100) builder target: gfx11 `s_waitcnt` VM/LGKM (combined `vmcnt(n) lgkmcnt(m)`) and `s_waitcnt_vscnt` waits, an LDS barrier that drains LGKMcnt stores before `s_barrier` (it previously skipped the drain on gfx11), M7's gfx11 hazard facts (gfx1100-only TRANS-use; no VALU-SGPR->VMEM NOP rule under `NoDataDepHazard`), gfx11 VOPD cross-half dependency rejection, no `s_delay_alu` on gfx11 WMMA chains, and a gfx11 text wait replay. `peacemaker custom build --arch gfx1100` certifies gfx11 builder objects with an all-lane LDS bound and M7's lift (byte-exact) plus whole-program wait/hazard/barrier analysis; M7's definedness seed now places gfx11 workgroup ids in the system SGPRs after the user SGPRs. `peacemaker profile --arch gfx1100|gfx1151` instruments gfx11 kernels (20-bit `SHADER_CYCLES` points, 100 MHz realtime at entry/exit). First kernel: `hipfire-isa emit --kernel iu4_v2c --arch gfx1100` (`gemm_mq4g256v2_residual_iu4_pm_set_gfx1100`), the hipcc V2C SET algorithm, bit-exact with V2C on real H2 inputs and 5.8% faster standalone at M17408 K5120 N8192. gfx1201 builder output is byte-identical.
- **Fix: gfx1201 native-fp8 prefill output no longer depends on what else runs on the GPU.** The fp8-stream RMSNorm producers (`fused_rmsnorm_mq_rotate_*mq4v2_fp8*`, including the default `…_awq_mq4v2_fp8_inreg_short_gfx12`) passed their `reduce` LDS as the E4M3 pack's scratch. The pack's first `smem8[wave] = amax` store comes before any barrier, so wave 0 could overwrite `reduce[0]` (the broadcast rms) before a lagging wave had loaded it. That wave then normalised with wave 0's amax, producing a wrong-scale row. On an idle card the waves stay in step and the pins reproduce. Next to another process's ALU-bound HIP waves it fired often enough to raise H2 code24 KLD from 0.029867 to 0.38–0.50; a co-resident torch loop gave small run-to-run changes. The pack now uses `reduce[8..15]`. Only seven DS immediate offsets change and the code is otherwise identical. The WT2/code24 `.kldseq` pins are unchanged on an idle card and now also reproduce under co-resident load. Evidence: `/home/kaden/qcal/perf/fp8-coload/report.md`.
- **gfx1201 a4-push2 (H2 A4 pp8192): four bit-exact A4 prefill levers land together.** The slab HIN producer runs on a token-fastest grid, the GDN KKT solve runs once per layer chunk, the full-attention out-projection's A4 slab is formed in the attention epilogue, and the fused GDN input projection's epilogue runs under a priority-1 K-loop with 528-byte LDS ring rows; each is detailed below. The combined binary keeps all four WT2/code24 `.kldseq` pins byte-identical. Against beta `4e61e5f6f` (separate pinned binaries, four fresh processes per arm, ABBA then BAAB, graph on, 300 W, card B), A4 pp8192 went 4,769.05 → 4,905.80 tok/s (+2.87%) in ABBA and 4,777.20 → 4,902.25 (+2.62%) in BAAB. The median of process medians went 4,775.5 → 4,903.85 tok/s, which is 1,715.4 → 1,670.5 ms per forward (−44.9 ms). pp4096 improved +1.36%/+1.41% and pp512 +0.29%/+0.54%. Native-fp8 (`HIPFIRE_IU4_PREFILL=0`, card E) pp8192 moved +0.17%/+0.20%; that is not slower, but it is inside the +0.36%/+0.34% spread of a beta-vs-beta control. Only the batched KKT reaches that route. Decode text is byte-identical to beta on both routes.
  - Exact gfx1201 A4 (iu4) prefill now launches the slab-layout HIN producer token-fastest: `fused_silu_mul_mq_rotate_awq_i4_hin_gfx12_slab_tokfast` runs on grid [N, K/256] instead of the `_slab` twin's [K/256, N], so neighbouring workgroups fill each 128-byte line of the slab `d`/`s`/`qs` planes (the group-fastest order filled a `d` line from 32 tokens dispatched 68 workgroups apart). Same per-workgroup math and stores, so the sidecar is byte-identical (whole-buffer oracle on captured H2 activations at N 8192/1024/1023/513/512/256/255, 200-repeat serial and single-CU stress; all four WT2/code24 `.kldseq` pins unchanged). Standalone N8192 on real activations: 1,731.8 → 1,089.7 µs/call (−37.1%) in ABBA and BAAB. H2 A4 pp8192, same binary, four fresh processes per arm: 4,739.0 → 4,812.95 tok/s (+1.56%) in ABBA and 4,740.45 → 4,808.05 (+1.43%) in BAAB (about −26 ms per forward); pp4096 +0.28%/+0.22%. `HIPFIRE_A4_HIN_TOKFAST=0` restores the group-fastest `_slab` twin. The other slab producers keep their grids: token-fastest measured +0.4% for the default gated-norm `_xbf16` twin and +6.5%/+49.5% for the sigmoid-mul twins, and the RMSNorm twin is already one workgroup per token.
  - Exact gfx1201 GDN chunk-scan prefill now solves the KKT (A) blocks of a whole layer chunk in one `gdn_chunk_kkt_solve_batched` launch before the commit-segment scans, instead of one `gdn_chunk_kkt_solve` launch per ≤512-row segment; each scan reads its A blocks at its row offset. The batched twin is the same source with the 512-row cap lifted and a key-head-fastest grid [16, ⌈N/64⌉] (the chunk-fastest grid was 46% slower than the per-segment launches), so every A value is unchanged: the per-segment object is code-identical to before, and the whole-buffer oracle on captured H2 activations (4 layers, N 8192/1024/1023/513/512/256/255, guards, 3 poison patterns, 200-repeat serial and single-CU stress) found 0 differing bytes; all four WT2/code24 `.kldseq` pins are unchanged on both the A4 and native-fp8 routes. Admitted when segments are 64-row aligned and the chunk spans more than one segment; `HIPFIRE_GDN_KKT_BATCHED=0` restores the per-segment launches. Standalone N8192 on real activations: 0.274/0.261 → 0.158/0.159 ms per layer (−42.1%/−39.3%, ABBA/BAAB); in-model KKT time per pp8192 forward 12.93 → 8.65 ms (768 → 48 launches). H2 pp8192, same binary, four fresh processes per arm, graph on, 300 W: A4 4,809.95 → 4,818.85 tok/s (+0.185%) in ABBA and 4,805.50 → 4,822.00 (+0.343%) in BAAB (pp4096 +0.03%/+0.33%); native-fp8 (`HIPFIRE_IU4_PREFILL=0`) 3,819.75 → 3,830.05 (+0.270%) and 3,822.30 → 3,829.70 (+0.194%). Evidence: `/home/kaden/qcal/perf/a4-5k2/kkt-batched/report.md`.
  - Exact gfx1201 A4 (iu4) prefill now forms the full-attention out-projection's A4 slab in the attention epilogue. On the fp8q Q-resident route, where the sigmoid gate stays in place for the IU4 AWQ producer, `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a4epi_gfx1201` (a `HIPFIRE_FA2_A4_EPILOGUE` twin of the q8 body in the same source file) applies the sigmoid gate, AWQ divide, FWHT-256 and `block_i4_128` quantisation to its register-resident output. It writes the slab `sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12_slab` formed from the f32 output, so that producer launch and the per-layer f32 round trip (201 MB at N8192) are gone; the out-projection reads the slab directly. The arithmetic follows the producer element by element. The sigmoid is a lean expf/reciprocal sequence that returns the producer's bits for all 2^32 inputs. The q8 attention and the other kernels in the file are code-identical. Default on; `HIPFIRE_A4_ATTN_EPI=0` restores attention + producer. Byte-identical: the H2 in-model slab equals beta's producer slab at FA layers 0/5/10/15; real-activation whole-buffer poisoned parity passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255; the A4 and fp8 WT2/code24 `.kldseq` pins are unchanged; and greedy decode is identical to beta. Card B, 300 W, real N8192 activations, 10 s/arm: attention + producer 5,996.3/6,011.6 → 5,693.9/5,707.0 µs/call (−5.04%/−5.07%, ABBA/BAAB). Same-binary graph-on H2 A4 pp8192 over four fresh processes per arm improved 4,758.25→4,776.05 tok/s (+0.374%) in ABBA and 4,761.15→4,771.60 (+0.219%) in BAAB (−5.4 ms per forward); pp4096 improved +0.212%/+0.163%. Evidence: `/home/kaden/qcal/perf/a4-5k2/attn-epi/report.md`.
  - **gfx1201 A4 fused GDN input projection: cheaper GDN epilogue, bit-exact.** Two scheduling-only changes in the builder's `Epi::QkvzaGdn` symbols (`gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s` and its token twin `_b1`): (1) the kernel runs its K-loop at wave priority 1 and drops the QKV tiles' GDN epilogue to priority 0, so co-resident K-loop waves win issue arbitration; (2) the epilogue's LDS ring rows are padded from 512 to 528 bytes, so the 16 token rows of each accumulator drain (W) store land four dword banks apart instead of all on the same four banks. Same arithmetic; whole-buffer bytes are identical to the previous bundles (200 serial + 200 co-resident repeats at N 8192/1024/1023/513/512/256/255, synthetic and real-prefill activations) and all four `.kldseq` pins are unchanged. The other eight iu4 symbols are code-identical. Card E, 300 W, real H2 layer-0 activations at N8192 (10 s/arm): `_b1s` 4,992.9→4,793.0 µs/call (−199.9) in ABBA and 5,039.6→4,856.6 (−183.0) in BAAB; `_b1` −198.3/−204.9 µs/call. Same-binary H2 A4 (default route) pp8192, four fresh processes per arm, graph on, opt-out as the off arm: 4,754.10→4,780.75 tok/s (+0.561%) in ABBA and 4,747.45→4,773.10 (+0.540%) in BAAB (1,724.4→1,714.6 ms/forward); pp4096 +0.523%/+0.610%, pp512 +0.239%/+0.549%. The native-fp8 route is code-identical: the F2 bundle is unchanged and its JIT objects have the same device code. Bundles: `_b1s` SHA-256 `3afa298e…` (was `c8f957fc…`), `_b1` `b884f5f7…` (was `89c7c87a…`). Developer opt-out / A/B with one binary: `HIPFIRE_G12_IU4_B1S_BUNDLE=<path>` and `HIPFIRE_G12_IU4_B1_BUNDLE=<path>` load a `_b1s` / `_b1` bundle from a file instead of the embedded one. Evidence: `/home/kaden/qcal/perf/a4-5k2/gdn-epi-trim/report.md`.
- Peacemaker's diagnostic ISA table, codec, CFG, and code-object lifter now recognize gfx1100 and gfx1151. Two-dword VMEM layout and label lowering restore byte-identical lift/emit on all **62/62 gfx1100** and **59/59 gfx1151** packaged objects (84/81 analyzed kernels) and the separate runtime JIT corpus: **233/233** and **229/229** compiled variants (331/425 kernels), including **76/76** and **119/119** distinct JIT-only objects. Six JIT censuses report zero rejected modules or unknown encodings; gfx11 row-share DPP, WMMA, wave64 MoE, DS, FLAT and scratch forms have target-specific codec support. Gfx11 full-barrier/LGKM/DS analyses and gfx1100 TRANS-use, both targets' VCMPX→PERMLANE and WMMA chaining hazards follow pinned ROCm LLVM rules; a separate non-runtime LLVM-MC probe exercises VCMPX→PERMLANE positive and safe cases. The gfx1201 control remains 95/95 byte-exact with 150 kernels analyzed and unchanged obligations. This is offline compiler/encoder diagnosis, not strict certification or a GPU benchmark; packaging/admission remains off. Evidence: `/home/kaden/qcal/peacemaker/m7/gfx11/report.md`.
- **gfx1201 a4-push (H2 A4 pp8192): three A4 prefill levers land together.** The FA sigmoid gate is read in place (no gate copy), the A4 activation sidecar is stored slab-major and read by the new builder `_b1s` bundle, and the GDN input projection runs as one fused QKVZ+beta/alpha launch with `gdn_chunk_prep` in its epilogue; each is detailed below. The fused projection now has a slab-layout twin, `gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s`, in the `_b1s` bundle (source differs from the `_b1` symbol only in the activation staging offsets, the same delta as every other `_b1s` symbol), so both levers apply at the GDN input site; the launcher selects `_b1s` exactly when a slab producer wrote the activations and `_b1` otherwise, never a token-layout reader on slab data. Whole-buffer poisoned parity of `_b1s` vs `_b1` (x, z, beta, alpha, conv ring, q/k/v) on the same logical inputs passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255; `qkvzagdn_b1s` is peacemaker-certified (192 VGPR, SGPR 104, 0 spills, GDN LDS ring certificate, max LDS end 18,432 of 20,480), the four existing `_b1s` symbols are code-identical and the `_b1s` bundle (`c8f957fc…`, 5 kernels) round-trips through `peacemaker-lift` byte for byte. The new `sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12_slab` producer twin (gate in place + slab store) matches its token twin byte for byte over the same seven N, 200 serial and 200 co-resident repeats. Combined against beta `14b87e308` (separate pinned binaries, four fresh processes per arm, graph on, 300 W): default A4 pp8192 4,632.60→4,755.35 tok/s (+2.650%) in ABBA and 4,635.65→4,759.40 (+2.670%) in BAAB on card A (1,767.5→1,721.5 ms per forward), pp4096 +4.158%/+4.150% and pp512 +5.875%/+6.186%. Native-fp8 (`HIPFIRE_IU4_PREFILL=0`) pp8192 on card C is flat within the protocol's order noise over two runs: ABBA −0.145%/−0.074%, BAAB +0.003%/+0.083%; the beta binary against itself in the same protocol measured ABBA −0.133%, BAAB +0.141%, and the fp8 route runs the same GPU code (every JIT kernel is code-identical to beta's and its builder bundles are unchanged). A4 and fp8 WT2/code24 `.kldseq` stay byte-identical to their pins (`483cfc58…`/`6339dc9e…`, `19227d5f…`/`5e7ac9dc…`), with `qkvzagdn_b1s` running ×1,152 in each A4 run. Evidence: `/home/kaden/qcal/perf/a4-5k/land/report.md`.
  - Exact gfx1201 A4 (iu4) prefill no longer copies the full-attention sigmoid gate. When the IU4 AWQ sigmoid producer is the out-projection's producer, the fp8q attention prep runs its `qwen35_fa_prep_fp8q_nogate_batched_gfx1201` twin (no gate read or write; 402.7 of the gated prep's 722.2 MB per N8192 call) and the producer's new `sigmoid_mul_rotate_x_mq_awq_i4_gil_gfx12` twin reads the gate in place from the Q/gate projection rows, as the fp8 route already does. Same f32 gate values, so byte-identical output. Default on; `HIPFIRE_A4_FA_GATE_IL=0` restores the gate copy. `Fp8SigmoidGate` is renamed `SigmoidGate`. Card B, 300 W, N8192: prep 1,229.5→531.7 µs/call and producer 746.9→745.9 µs/call in both ABBA and BAAB (12 s/arm), the prep→KV write→attention→producer chain −616/−610 µs per layer; same-binary graph-on H2 A4 pp8192 over four fresh processes per arm improved 4,657.65→4,669.45 tok/s (+0.253%) in ABBA and 4,636.25→4,657.70 (+0.463%) in BAAB. Poisoned whole-buffer parity of Q codes/scales, K, the KV caches, the attention output and the IU4 blocks against the beta objects passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255; fp8 and A4 WT2/code24 `.kldseq` stay byte-identical to their pins, and all other symbols built from `mq_rotate_x_i4.hip` are code-identical. Evidence: `/home/kaden/qcal/perf/a4-5k/attn-nogate/report.md`.
  - Exact gfx1201 symmetric MQ4V2 A4 prefill now stores the A4 activation sidecar slab-major: each K128 block keeps its N×72 bytes, laid out as the planes `[d 4N][s 4N][qs 0..31: 32N][qs 32..63: 32N]`, so a wave's 8-token staging load reads 256 contiguous bytes and its 16-token `d` load 64 bytes. The five default gfx1201 A4 producers (RMSNorm AWQ `_v2`, HIN, gated norm AWQ `_v2` and `_v2_xbf16`, sigmoid-mul) have `_slab` twins that store the same values at the moved addresses, and the new builder `_b1s` bundle (`hipfire-isa --alayout slab`: the `_b1` K-loop and epilogues with slab staging offsets) is their only reader; the host tags each slab producer's scratch generation and the consumers route to `_b1s`, failing closed otherwise. Bit-identical: the GEMM oracle (`_b1s` on slab Xq vs `_b1` on token Xq) matched 44 cases / 996,767,808 words plus 200 serial and 200 co-resident whole-buffer repeats at N 8192/1024/1023/513/512/256/255, the producer twins matched their token twins over the same N, and the A4/fp8 WT2/code24 `.kldseq` pins are unchanged. Standalone at 300 W the slab layout is faster at every site in both orders (N8192 gate/up +4.30%, down ADD +4.58%, GDN QKV SET +6.19%; same cycles per op at a 57–95 MHz higher clock). Same-binary graph-on H2 A4 pp8192 over four fresh processes per arm improved 4,615.85→4,666.95 tok/s (+1.15% ABBA, +1.04% BAAB). Default on; `HIPFIRE_A4_SLAB=0` restores the token layout and `_b1`. Evidence: `/home/kaden/qcal/perf/a4-5k/b1-slab/report.md`.
  - Exact gfx1201 symmetric MQ4V2 A4 prefill on the GDN chunk-scan route now runs the DeltaNet input projection as one builder `_b1` launch (`gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1`) instead of four SETs plus `gdn_chunk_prep`. The launch covers QKV and a load-time Z fold (Z, then beta and alpha sharing one 128-row tile; 129 row tiles instead of 80+48+1+1). Its QKV tiles (one 128-channel q/k/v head each) run the prep in the epilogue with F2's imported hipcc regions and lean SiLU: conv1d+SiLU, q/k head norm, q scale and FP16 q/k/v, never storing the f32 q/k/v plane. The unchanged `gdn_chunk_prep_fixup` completes the gates, tile-head tokens and conv ring. The K-loop and fold order are the SET symbol's; the other four `_b1` symbols are code-identical and the new symbol is peacemaker-certified (192 VGPR, 0 spills, GDN LDS ring certificate). Outputs are byte-identical to four SETs + prep: whole-buffer poisoned parity passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255; A4 WT2/code24 `.kldseq` stay on their pins (`483cfc58…`/`6339dc9e…`) with the fused symbol running in both, and the fp8 pins (`19227d5f…`/`5e7ac9dc…`) are unchanged. N8192 layer-0 kernel time falls 5,710.6→5,179.0 µs (ABBA) and 5,777.9→5,239.8 µs (BAAB), −9.3% each. Same-binary graph-on H2 pp8192 on card E improved 4,659.05→4,720.70 tok/s (+1.323%) in ABBA and 4,650.00→4,722.50 (+1.559%) in BAAB, four fresh processes per arm (−25.1 ms per forward); the replaced Z buffers are returned to HIP at load, so peak VRAM rises only 96 MiB. `HIPFIRE_GDN_PREP_FUSED=0` restores the four SETs + prep (and skips the Z fold at load); `HIPFIRE_G12_IU4_GDN_COVERAGE=1` prints each fused launch.
- **gfx1201 fp8-push (H2 pp8192): three prefill levers land together.** The lean F2 QKVZA+GDN epilogue, the batched fp8 SHORT RMSNorm phase-1a and the A4 bf16 GDN chunk-scan plane are detailed below. Combined against beta `204ce521c` (separate pinned binaries, four fresh processes per arm, graph on, 300 W): native-fp8 (`HIPFIRE_IU4_PREFILL=0`) pp8192 3,824.35→3,852.50 tok/s (+0.736%) in ABBA and 3,826.05→3,854.30 (+0.738%) in BAAB on card A; default A4 pp8192 4,635.90→4,641.20 (+0.114%) in ABBA and 4,636.50→4,646.25 (+0.210%) in BAAB on card C. fp8 WT2/code24 `.kldseq` stay byte-identical to their pins (`19227d5f…`/`5e7ac9dc…`); A4 takes the scan-bf16 pins (`483cfc58…`/`6339dc9e…`), and `HIPFIRE_GDN_SCAN_OUT=f32` reproduces the previous A4 pins (`4abf44e8…`/`aad7ce38…`). Evidence: `/home/kaden/qcal/perf/fp8-4k5/land/report.md`.
  - The gfx1201 F2 QKVZA+GDN epilogue (`gemm_mq4g256v2_fp8_qkvzagdn_row_b1`) issues about 37% fewer VALU instructions per conv1d+SiLU channel pair and loads one LDS ring row per token instead of four. The SiLU keeps the imported conv MACs but replaces the exp range selects with a `v_med3_num_f32` clamp and the IEEE division with a prescaled `v_rcp`, one residual correction and `v_div_fixup_f32`; `tools/gdn/silu_exhaust.py` proves it returns the imported hipcc sequence's bits for all 2^32 f32 inputs on gfx1201. Outputs are byte-identical to the previous bundle: H2 whole-buffer poisoned parity passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255, and the fp8 and A4 WT2/code24 `.kldseq` pins are unchanged. The other 11 F2 symbols are code-identical. On H2 the kernel is −1.68%/−1.96% faster (ABBA/BAAB), and graph-on native-fp8 pp8192 improved by +9.7/+14.6 tok/s (+0.25%/+0.38%) in four fresh processes per arm. New developer variable `HIPFIRE_G12_FP8_F2_BUNDLE=<path>` loads the F2 bundle from a file, so builder bundles can be A/B-compared with one binary. Evidence: `/home/kaden/qcal/perf/fp8-4k5/qkvza-epi/report.md`.
  - Exact gfx1201 fp8 SHORT RMSNorm producers (the H2 K=5120 input/post-attn path) now keep up to 8 row loads in flight per wave in the phase-1a sum-of-squares pass instead of one: same elements, same strictly sequential `fma(v, v, acc)` chain from +0, same reduction tree (bit-identical rms). Default on; `HIPFIRE_RMSNORM_P1A_BATCHED=0` selects the single-outstanding `_SEQ` twin (same entry symbol, distinct module). Standalone N8192 at 300 W improved AWQ 444.7→355.5 µs/call and plain 440.9→397.2 µs/call, each faster in ABBA and BAAB at ≥10 s/arm; whole-buffer poisoned parity passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255 for both symbols. Same-binary graph-on pp8192 improved 3,834.20→3,855.60 tok/s (+0.558%) in ABBA and 3,848.15→3,861.80 (+0.355%) in BAAB; H2 fp8 WT2/code24 and A4 WT2/code24 `.kldseq` stayed byte-identical to the quality pins. 20-wide batches were measured to corrupt single rows intermittently under sustained pressure and are capped at 8; the IU4 `_v2` twin keeps its 20-wide shape. Evidence: `/home/kaden/qcal/perf/fp8-4k5/rms-floor/report.md`.
  - Exact gfx1201 A4 (iu4) prefill now stores the GDN chunk-scan output as a round-to-nearest-even bf16 plane (`gdn_chunk_scan_bf16`) instead of f32, and the A4 gated-norm producers read it through `_xbf16` twins that widen each element exactly before the unchanged norm/FWHT/quantization arithmetic. This halves the plane's store and re-read bytes. `HIPFIRE_GDN_SCAN_OUT=f32|bf16` overrides the route default. The fp8-route producer has a certified `_xbf16` twin as well, but it stays opt-in (default f32) because graph-on H2 pp8192 measured −0.158%/−0.059% (ABBA/BAAB). A4 improved 4,642.85→4,646.20 tok/s (+0.072%) ABBA and 4,619.80→4,633.65 (+0.300%) BAAB over four fresh processes per arm. Quality is not bit-exact. With H2 A4, WT2 KLD goes 0.069250→0.069571 and code24 0.054174→0.052119; the new pins are WT2 `483cfc58b4cc71ea3a9a71c657833e34aa847f513aceae50c1c0f0dfe65ae898` and code24 `6339dc9e54bbd498774667cfe25c8405049d20c2c61e930ab7e783b662c5f13a`. They are byte-identical to the f32-store quality emulation (`HIPFIRE_GDN_SCAN_OUT_EMU=bf16`, a diagnostic). The fp8 pins are unchanged. Poisoned whole-buffer parity against that emulation passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512/256/255. Evidence: `/home/kaden/qcal/perf/fp8-4k5/scan-bf16/report.md`.
- Exact gfx1201 native-fp8 prefill now runs `gdn_chunk_prep` (conv1d+SiLU, q/k head norm, q scale, FP16 q/k/v) inside the F2 QKVZA epilogue (`gemm_mq4g256v2_fp8_qkvzagdn_row_b1`) on the GDN chunk-scan route, so the f32 q/k/v plane is no longer written and re-read; a small `gdn_chunk_prep_fixup` pass computes the gates, the three tile-head tokens of each 128-token tile and the conv ring with prep's own expressions. The epilogue regions are peacemaker-lifted from the production prep object, and all existing F2 symbols are byte-identical. Default on; `HIPFIRE_GDN_PREP_FUSED=0` restores the f32 projection + `gdn_chunk_prep`. H2 whole-buffer poisoned parity passed 200 serial and 200 single-CU co-resident repeats at N 8192/1024/1023/513/512. H2 fp8 WT2/code24 and A4 WT2/code24 `.kldseq` are byte-identical to their pins, and greedy decode matches off/on. Same-binary graph-on fp8 pp8192, four fresh processes per arm, improved 3,806.80→3,825.05 tok/s (+0.479%) in ABBA and 3,805.55→3,829.20 (+0.621%) in BAAB (≈13 ms/forward). Evidence: `/home/kaden/qcal/perf/fp8-4k5/gdnprep-fuse/report.md`.
- Exact gfx1201 fused fp8 F2 gate/up SiLU now stores packed bf16 h, and its HIN producer widens the half-width input before the unchanged FWHT/quantization arithmetic. `HIPFIRE_BF16_H_FP8=0` restores the f32 handoff. On H2, graph-on native-fp8 pp8192 in four fresh processes per arm improved 3,789.65→3,812.80 tok/s (+0.611%) in ABBA and 3,783.30→3,811.40 (+0.743%) in BAAB; WT2/code24 `.kldseq` remained byte-identical to quality emulation (KLD 0.041859/0.029867). The separately certified A4 bf16 route remains opt-in (`HIPFIRE_BF16_H_A4=1`), not a production default: graph-on pp8192 regressed −1.274%/−1.119%. Both paths reuse the existing f32-sized h allocation; no VRAM reduction is claimed. Evidence: `/home/kaden/qcal/perf/iu4-6k/bf16-h/report.md`.
- **Breaking:** `hardware.devices` / `HIPFIRE_DEVICES` (singular alias `HIPFIRE_DEVICE`) now names physical cards, not ROCr agent ordinals. Entries, in logical order: an index into the PCI-address-sorted GPUs (the `rocm-smi` order), `gfxNNNN` (first free card of that arch; `gfx1201,gfx1201` gives two), `GPU-<uuid>`, or a PCI address `[DDDD:]BB:DD.F` (the only exact form for cards whose UUID reads `GPU-XX`). Numeric lists that relied on ROCr order must be rewritten; the daemon prints the full device table on any resolution error. The daemon resolves the list from `/sys/class/kfd/kfd/topology` before HIP starts, lowers each card to its UUID (or ROCr ordinal when it has none) in `ROCR_VISIBLE_DEVICES` with HIP `0..N-1`, and after HIP loads — at startup and in multi-GPU `Gpus` construction — aborts unless each logical device's arch and PCI bus ID match. Unknown selectors, misses, duplicates, busy exact cards and exhausted arches fail closed. The client no longer lowers the list itself. Raw `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` remain the expert override when the key is unset.
- Daemon GPU reservations are machine-wide, non-blocking flock files keyed by the resolved card identity: `gpu-GPU-<uuid>.lock`, or `gpu-pci-<dddd:bb:dd.f>.lock` for cards without a UUID (previously every no-UUID card shared one HIP-derived name). A `gfxNNNN` entry takes the lock while resolving, so concurrent daemons asking for the same arch get different cards; without `hardware.devices` the HIP-visible cards are matched to KFD by PCI address and locked sorted, all-or-nothing. The lock directory is `HIPFIRE_LOCK_DIR` when set, otherwise `/run/lock/hipfire` if writable, else `/tmp/hipfire-locks`; `scripts/gpu-lock.sh` accepts a UUID or PCI address and uses the same keys and directory policy. Distinct GPUs may run concurrently under the same HOME; conflicts report the card, PCI BDF and holder PID. Per-GPU-set HOME PID files (`daemon-<identity>[_...].pid`) remain advisory for uninstall.
- **Breaking for compiler-free installs:** unindexed `.hsaco` files (including older `.hash` pairs) are rejected; re-run the installer or rebuild the Nix/container package to get exact-source registry objects with matching `.index.json`. Native installers and `daemon --precompile` now package the registry instead of a model-format hand list; Nix and container images bundle indexed objects beside the daemon. With hipcc available, a missing or rejected index logs `packaged object rejected (...); falling back to JIT` and recompiles that module; valid indexed modules keep loading from the package.
- Installed objects use a portable packaging key and a versioned per-module index binding module, symbols, arch, source SHA-256, flags/profile, cache ABI, hipcc identity and object SHA-256. Hot JIT entries retain toolchain-specific keys. `hipfire-kernel-pack` publishes exact-source indexed objects; `daemon --precompile --module rmsnorm_f32` remains a numerical production-route diagnostic.
- Architectures without an admitted registry remain hipcc JIT-only; installers do not install unindexed objects for them.


### v0.4.0 folded PRs — partial GPU offload, serve discovery, speculative UTF-8 streaming

- **Partial GPU offload for single-GPU dense Qwen3.5 (#793, Avery Drouillard; opt-in, default off).** `memory.gpu_layer_budget` (env `HIPFIRE_GPU_LAYER_BUDGET`) keeps that many layers in VRAM and places the weights of the layers before them in pinned host RAM (`hipHostMalloc` mapped), so a model larger than the card still loads; the KV cache stays in VRAM. `memory.offload_exec` (env `HIPFIRE_OFFLOAD_EXEC`) chooses who multiplies a spilled layer: `pcie` (default) runs the GPU kernels against the host-mapped weights over the link; `cpu` runs those layers' decode GEMVs on the host (new `hipfire-cpu` crate, AVX2 row dots for every dense quant format with a scalar fallback). The TUI Settings easy list gains `GPU layers` and `Offload exec` rows.
  - Unset (the default) places every layer in VRAM through the same upload calls as before, and no kernel source changes.
  - A spill is refused at load admission, before the resident model is unloaded, when it cannot be honoured: tp>1 (the dense-TP and EP rank loaders keep every layer resident), pp>1 (the first band would spill into host RAM mapped by device 0, which has never been run) and MoE models (only attention and DeltaNet weights would spill; the expert stacks stay in VRAM). The Qwen3.5 weight source refuses the same cases for every other load path. The `partial offload: N resident / M offloaded` line now prints when the loader applies the placement, not at every config parse.
  - `memory.offload_exec=cpu` with a spill is not bit-identical to a resident load and runs without a HIP graph. On gfx1201 such a process no longer gets the retained Redline PM4 default, whose tape would skip the host steps; an explicit `replay.backend` other than `hip` is refused at load, as before.
  - The gfx1201 A4 `Z|beta|alpha` fold of a spilled DeltaNet layer keeps the folded rows in host RAM instead of moving them back to VRAM.
  - Also from the PR: `Gpu::gemv_mq4g256` no longer passes the FWHT sign tables as kernargs (no product caller used the wrapper), `vmm_tensor_smoke` no longer asserts that a non-granular map size is rejected, and the `dispatch` GPU tests serialize on a module lock.
- **`/v1/models` lists only ids that can serve, and publishes the resident model's load facts (#787, Avery Drouillard).** Registry tiers whose suffix is not in `MODEL_SUFFIXES` (`-xt`, `-pro`, `.bq1`, `.tq2`) are listed; registry sidecars (vision tower, DFlash drafts, `mtp`/`dspark`/`triattn`/FLUX components) are not, since selecting one failed only after a full trunk load. A file the registry does not name is listed only when its 32-byte HFQ header (`HfqFile::probe_arch_id`) carries a text-serving arch id (0, 1, 5–15). `hipfire list` still shows every file. The resident model comes first, with the OpenAI `created` (file mtime), `meta.n_ctx` (the `max_seq` it was loaded with), `meta.n_embd`, `meta.n_vocab` and `architecture.input_modalities` from the load ack; other entries carry `created` only. `/health` gains `n_ctx` (`null` with nothing resident) and still reads no state behind the load lock. The daemon's load-ack `vl` is `LoadedModel::has_vision_encoder()`, so an LFM2-VL tower now counts. A failed switch resets these facts along with the residency (the residency part landed earlier in this release).
- **DeepSeek V4 and Cohere2-MoE speculative decode stream multi-token characters whole (rest of #595, Flaviu-Gheorghe Grosan).** `Deepseek4Emit` (MTP/DSpark) and `Cohere2MoeEmit` decoded each committed token on its own, so an emoji or byte-fallback CJK character reached the client as U+FFFD; they now stream through `TokenTextStream` like the AR routes, and a character still split when generation stops is flushed at finish. Cohere's structural-marker guard checks the token's own bytes. The rest of #595 (the AR and vision loops) landed earlier in this release.

### v0.4.0 fixes — serve gateway, generation, Qwen3.8 MTP head, and CI gates

- **Qwen3.8-27B ships an MTP head, and MTP turns on by default when it is present.** Every Qwen3.8-27B trunk tier (`qwen3.8:27b` and the `-mq3*`, `-mq4*`, `-mq5*` and `-mq6*` tags) now declares one shared `mtp` sidecar, `qwen3.8-27b.mtp` (225,716,224 B, sha256 `f0d46d07ded75abc095ebfdbccc167c68418a6515b187bf810395faa460bd32e`). `hipfire pull` fetches it with the trunk. The head is the native `mtp.*` layer of the public `Qwen/Qwen3.8-27B` checkpoint, packed by the in-tree `mtp_extract --quant mq4`; re-extracting it reproduces the file byte for byte. It holds no trunk weights and reuses the trunk's embeddings and `lm_head`, so every tier shares it. The MQ4L tiers and the DFlash drafts declare no head.
  - Because `speculation = "auto"` and `mtp_mode = "auto"` are the defaults, a pulled head runs MTP on every request (`drafter=mtp`); `mtp_mode = "off"` (or `speculation = "off"`, `run --spec off`) turns it off. On gfx1201 with H2 (`qwen3.8:27b-mq4-xts`), decode against AR measured 1.09–1.65× on six of eight task genres, 0.96× on a tool answer and 0.95× on code editing. Greedy MTP output can differ from greedy AR output.
  - A head beside a symlinked trunk is now found. The CLI canonicalized the model path, so the loader looked only beside the symlink's target. The CLI now resolves the sidecar into the load (`params.mtp` → `LoadCtx.mtp_path`) from the models directory, then beside the path as typed, then beside the canonical trunk; direct daemon clients keep the `<trunk>.mtp` lookup.
  - The loader refuses a head whose `n_embd` or `vocab_size` differs from the trunk's hidden size and vocab, before any GPU upload; before, only `arch_id == 21` was checked. `auto` logs `MTP head not loaded: … does not match this trunk …` and runs AR, and `on` fails the load with both sets of dimensions. A wrong-arch or unreadable file is refused the same way; before, it panicked.
  - `hipfire rm` keeps the shared head while another installed tier declares it; before, removing any tier deleted it.
  - The bundled `registry/v1.json` is re-stamped `2026-09-30T01:25:00Z`. With the head's HF identity simulated, `scripts/registry_gen.py` reproduces it exactly.
- **`hipfire serve` binds `127.0.0.1` by default (was `0.0.0.0`).** The API has no authentication and no TLS, so the old default exposed it to every reachable host. `serve.host` (config, `HIPFIRE_HOST`, or `hipfire serve 0.0.0.0:11435`) still listens on all interfaces when you want that; the container docs already pass `0.0.0.0` inside the container. The `default` config profile and `docs/configs/user.toml` carry the new value. Taken from #780 (host default only).
- **A chat request can no longer download models or load arbitrary files.** A `model` naming a registry model that is not installed used to start the download inside the request (82 GB for `deepseek-v4-flash-preview`), and a `model` naming any readable file (`/etc/hostname`) was sent to the daemon, which unloaded the resident model first. Both now return 404 unless enabled: `serve.allow_request_pull` (`HIPFIRE_SERVE_ALLOW_REQUEST_PULL`) and `serve.allow_request_paths` (`HIPFIRE_SERVE_ALLOW_REQUEST_PATHS`), both default `false`. Requests may still name installed models: files under the models directory (symlinks included), catalog paths, and the pre-warm model. The pre-warm model itself may still be pulled at startup.
- **A second serve client no longer gets 503 after 30 s while the first generation runs: `serve.queue_timeout_ms` defaults to 600000 (10 min).** Serve is single-stream by default (continuous batching covers only thinking-off, single-turn Qwen), so a queued request waits for the running generation. The old 30 s wait covered 4096 tokens only above ~137 tok/s; the 27B decodes at ~40 tok/s on an R9700, so 4096 tokens take ~100 s and a long thinking turn several minutes. 10 minutes covers one full generation of ~24K tokens at that rate [inferred from the decode rate, not measured end to end]. The 503 now says "server busy", names `serve.queue_timeout_ms`, and caps `Retry-After` at 30 s instead of echoing the whole wait. `0` still waits forever.
- **An omitted `max_tokens` no longer reloads the model and then fails.** Qwen3.5/3.6/3.8 registry tags default `max_tokens` to 81920 (DeepSeek V4 Flash to 393216); a request without `max_tokens` needed `max_tokens + 1024` of context, so serve reloaded the model (which cannot grow `max_seq`) and the daemon then refused the request, on every request. Now serve never reloads for `max_tokens`. When the client omits it, the generate carries `max_tokens_fit` and the daemon fits the default to `context − prompt − 64` at the context check of each route (Qwen AR, DFlash/MTP entry, Qwen TP/EP and PP, DeepSeek V4 single-GPU and EP, MiniMax EP, LFM2-MoE, arch 15, Qwen-VL). An explicit `max_tokens` at or above the loaded context is a 400 from serve; a smaller one that does not fit the actual prompt is refused by the daemon (`context_length`) as before. MiniMax single-GPU and Qwen2 keep their reset-then-refuse guard.
- **Serve respawns a daemon that exits, and `/health` stops saying `ok` while it is gone.** Serve spawned the daemon once; after a crash, panic or sticky GPU fault every request failed until a manual restart, while `/health` stayed `200 ok`. Now a supervisor checks the daemon every second (and every request checks it before use); a dead daemon is reaped, respawned (5 attempts, 1/2/4/8 s apart) and the resident model is reloaded. `/health` answers 503 with `status: "restarting"` during that, and `"unhealthy"` if respawning fails, after which serve exits 1 so a service manager can restart it. The daemon now exits 75 right after answering the request that latched a sticky GPU fault (HipError 700/719) instead of refusing every later request with "process restart required".
- **A failed model switch no longer leaves serve pointing at an empty daemon.** At tp≤1 the daemon unloads the resident model before loading the next; when that load (or the MQ4R decode pre-warm) failed, serve kept the old model as resident, generated against nothing, and `/health` still named it. Serve now clears its resident state (`/health.model` becomes `null`) and reloads on the next request. At tp>1 the daemon keeps the old model on a failed load, and so does serve. The residency part of #787, ported without its `/v1/models` changes.
- **Serve: error statuses come from the daemon's typed error class, and streams fail with an error event instead of a torn 200.** A typed daemon error now maps on its `class`: `validation`, `context_length` and `unsupported` → 400, `transient` → 503 with `Retry-After: 1`, anything else → 500. Before, the status came from matching words in the message, so validation errors such as an open `<think>` at the length cap or a bad `seed` returned 500 and SDKs retried them. Gateway errors, which carry no class, keep the message matching ("model not found" → 404, `max_tokens` → 400).
  - A streaming request no longer sends `200` and the role chunk before the request is checked. The stream commits when generation sends its first frame, so a failure in request validation, model load or the daemon's own checks gets its real status as an ordinary JSON error. If generation is still silent 15 s after admission (a cold load or a long prefill), the stream commits anyway so client timeouts do not fire; a failure after that point is reported in the stream.
  - A stream that fails after it committed ends with `data: {"error":{"message":…,"type":…}}` and `data: [DONE]` in a cleanly terminated body. Before, the body was cut off with no error and no `[DONE]`. A failure after the acknowledged `[DONE]` (the daemon's commit) still cuts the body off, because nothing may follow `[DONE]`.
- **Serve: SSE streams send a `: keepalive` comment after 15 s without a frame (the SSE prefill keepalive of #784, without its adaptive-B change).** A long prefill wrote nothing between the role chunk and the first token, so clients with an idle-read timeout dropped long prompts mid-prefill: undici (Node `fetch`) aborts at its 300 s `bodyTimeout`, which #784 measured at about 108k prompt tokens on `qwen3.8:27b-mq4-xt`, and nginx's default `proxy_read_timeout` is 60 s. SSE parsers drop comment lines, so the content of the stream is unchanged. The keepalive stops at the `[DONE]` frame: no byte follows it.
- **Serve survives `accept` failures, bounds its connections, and drains requests on SIGTERM.** An `accept` error (for example `EMFILE`: 56 idle connections under `ulimit -n 64` did it) ended the whole server with exit 1 and left `serve.pid` behind. It is now logged and retried with a backoff of 10 ms doubling to 1 s.
  - At most 512 client connections are served at once; further connects wait in the listen backlog.
  - A client must send a complete request head within 30 s of connecting or of its previous response. This also closes idle keep-alive connections, which before stayed open forever.
  - SIGTERM or Ctrl-C now stops accepting and frees the port at once, lets requests in flight finish for up to 30 s, and then exits. Before, it exited at once and cut off every stream in flight. A second signal exits at once. `hipfire stop` still reports "stopped" after its 5 s wait, so it can return while a long request is still finishing.
- **Serve `/metrics`: `hipfire_requests_failed_total` and `hipfire_admission_rejected_total` now count.** Both counters existed but nothing incremented them, so after three failed requests `/metrics` still read `failed_total 0`. A chat or image request answered with an error, or whose stream ended in an error event, now counts as failed; a request refused by admission (queue full, or queue wait timed out) counts as rejected and not as failed. Oversized or unparseable bodies count in neither.
- **Serve: a queued `/v1/images/*` request no longer blocks a server thread.** It waited for admission with the blocking `acquire`, which parks a tokio worker for up to the queue timeout (30 s by default), so a few queued image requests could stall every other endpoint, `/health` included. It now waits asynchronously like chat, and a refusal is a 503 with `Retry-After` counted in `hipfire_admission_rejected_total`, as for chat, instead of a 500.
- **Fix: multi-slot serve (opt-in `serve.multi_slot`) no longer refuses new conversations after a few chats (#776).** A session evicted from its slot (swapped to host, or dropped cold) kept its VRAM admission until it was closed, and finished sessions are kept for prefix reuse. So every distinct conversation held a charge for good: on a 32 GiB gfx1201 with qwen3.8:27b MQ4, 2 slots × 131,072 ctx, the 5th conversation was refused with `WouldExceedBudget` while a slot sat free. Admission now follows GPU residency. Eviction returns the charge; a restore takes it back first, and if the budget refuses, the snapshot is dropped and the next turn re-prefills. Closing a session releases only a charge it still holds, so it cannot release another session's equal-sized grant a second time.
- **Running out of `max_tokens` inside `<think>` finishes with `finish_reason: "length"`, not an error.** Qwen3.6/3.8 think by default, so any client cap below the reasoning length used to return HTTP 500 (non-stream) or a torn stream with no `[DONE]`. The reasoning streamed so far is returned as `reasoning_content`, `content` is empty, and the turn is not cached, as for any other length stop. An end-of-turn token inside `<think>` still fails closed. Covers Qwen AR (single GPU, dense TP, MoE EP, batch lanes) and the DFlash/MTP spec path, where the length stop also keeps the state instead of rolling it back.
- **`stop` accepts the OpenAI string form and never returns the stop text.** `"stop": "\n"` was ignored; `["END"]` leaked `END` into the output on Qwen. Stops are matched on the answer only (never inside reasoning), a possible stop prefix is held back before emission, and a turn ended by a stop is not cached. More than 4 sequences, a sequence over 64 characters, or a non-string value is a `validation` error instead of silent truncation. Honoured on the Qwen3.5-family routes (`qwen_ar`, `qwen_dflash` incl. MTP); every other route (DeepSeek V4, Gemma4, Glimmer, Cohere, LFM, Qwen2, Maple, MiniMax, LLaMA, dots, pipeline parallel) now refuses `stop` with an `unsupported` error rather than ignoring or leaking it.
- **Multi-byte characters no longer stream as U+FFFD on non-Qwen3.5 routes (#595, re-homed).** Per-token `decode(&[tok])` split emoji and byte-fallback CJK into replacement characters on DeepSeek V4 (AR, heterogeneous, EP), Gemma4 (eager, EAGLE, lowered), Muse Glimmer, LFM2 and LFM2-VL, MiniMax (AR, EP), Cohere2 and Qwen2. `TokenTextStream` holds back only an incomplete trailing code point. LFM2.5-1.2B, emoji + CJK prompt: 8 U+FFFD before, 0 after.
- **DeepSeek V4 tool calls reach the client.** The legacy (non-contract-v2) serve fold read tool calls only from separate `tool_calls` events, but DeepSeek V4 stages them on the terminal, so clients got `finish_reason: "tool_calls"` with no `tool_calls`. Calls staged on the terminal now replace the buffer; a malformed staged call fails the request. Ported from #593.
- **`--tp` > 1: tool calls are parsed, and unsupported sampling and images are refused.** The Qwen TP and MoE-EP decode loops built the output router with tools disabled, so tool calls came back as `<tool_call>` text with `finish_reason: "stop"`. They now route tools like single-GPU Qwen AR and stage the calls on the terminal. The tensor-parallel decode has no penalty sampler and the EP load has no vision tower: a request with a non-neutral `repeat_penalty`, `presence_penalty` or `frequency_penalty`, or with an image, now gets an `unsupported` error (400) instead of silently losing it, and serve no longer forwards configured penalty defaults at tp>1.
- **Fix: native MTP no longer aborts when a scalar Q8 batched GEMM gets more than 64 rows (#767, fixes #732).** The scalar `gemm_q8_0_batched` kernel takes at most 64 rows and asserts above that. Five MTP call sites called it directly with widths such as 70, 76 and 256, which an ordinary system + user request or a long prefill produces. The MTP head's Q8 projections and the DFlash+MTP Q8 lm_head always take the scalar kernel. The MTP head's lm_head and the MTP trunk-verify lm_head take it on architectures without WMMA, or when WMMA is turned off (`HIPFIRE_MTP_HEAD_LMHEAD_WMMA=0`, `HIPFIRE_MTP_Q8_VERIFY_WMMA=0`). These call sites now use `gemm_q8_0_batched_f32_chunked`, which launches the same kernel in 64-row pieces. At 64 rows or fewer it is the same single launch, so the output is byte-identical. Checked on gfx1201 at 1 to 256 rows (including 65, 70, 76, 127–129, 200 and 256) over three shapes: every row of the chunked output equals a single-row launch byte for byte and matches a CPU emulation of the kernel's arithmetic bit for bit, and nothing is written past the output.
- **A malformed MQ4-Lloyd (qt=52) codebook sidecar fails the load instead of panicking the daemon.** `lloyd_lut_c16_from_levels` ended in an `assert!`, so a `lloyd_levels` sidecar with the right dtype and shape but a level outside [0,15] or NaN aborted the process. It now returns an error, and the load reports `sidecar ... invalid: C16 code ... out of [-120,120]`. Valid sidecars build the same LUTs as before.
- **The serve gateway tests are in the required CI job.** `hipfire-cli` is bin-only, so the `unit tests` job's `cargo test --lib --workspace` never ran its fake-daemon HTTP tests. The job now also runs `cargo test --locked -p hipfire-cli`. The hw-gate selector (`scripts/hw-gate/select.py`) now puts `crates/hipfire-cli/src/serve/**` and `crates/hipfire-client/**` in the `serve` bucket; before, gateway-only changes selected no hardware route.
- **The no-GPU gates are green again (#778, refreshed onto the E1/D8 land).** `leanup-ratchets`, `check-layering`, `check-crate-maps` and `check-env-docs` pass. The four crates beta added are listed in `scripts/layering.txt`, and `HIPFIRE_LOCK_DIR` is an allowed pre-config read. The ceilings are raised to beta's actual values: `daemon_lines` 4863 → 5362, `ungated_examples` 48 → 61 and `bypass_total` 262 → 304. The gfx1201 P0 source fixture is re-pinned for `fused_rmsnorm_mq_rotate{,_awq}`, `fused_silu_mul_mq_rotate_awq` and `gated_delta_net_q8_fast`. On the capture machine, the trace comparison now names those re-pins and the later `-fuse-cuid=none` flag instead of failing on them. The unbuildable `kx3_foldfree_probe` example is removed.
- **Build temps and scratch trees are out of the source tree.** The 18 hipcc `-save-temps` files at the repo root (`x*-hip-amdgcn-amd-amdhsa-*.tmp.bc`, one `.s.tmp`) and the 24 `scratch-*` directories (1,094 files, including `.hsaco` objects and host binaries) are removed, and `.gitignore` now ignores both patterns at the root. They stay reachable in git history.
- **docs/QUANTIZATION.md names the MQ4-Lloyd `--format` the quantizer accepts.** The qt=52 row said `mq4v2lloyd`, which no parser matches. It now says `mq4v2-lloyd`; `mq4l`, `mq4v2l`, `mq4g256v2l` and `mq4g256v2-lloyd` are also accepted.
- **DFlash adaptive verify-block (#784, by Samuel Ishida): the trailing-τ controller ships opt-in, default off.** `dflash_adaptive_b` was accepted but ignored; it now reaches the loader (`SpecLoadCfg.dflash_adaptive_b`) and, when set to `true`, the DFlash chain speculator shrinks its proposal width to `clamp(ceil(τ̂) + 2, 2, B)` over the trailing 8 verify cycles once the context passes 2,048 tokens (`hipfire_runtime::dflash_adaptive_block`). The default is `false` in the config, the daemon and the loader, so every load keeps the fixed full block: with the setting off, the proposal width and `block_size()` equal the configured block exactly as before. It stays off by default because it is not output-identical (window boundaries move; per-position sampling stays target-lossless) and has no long-context gfx1201/gfx1151 measurement yet. `HIPFIRE_DFLASH_ADAPTIVE_B=0` forces the fixed block when the setting is on, and a load that admits the retained-PM4 verify route keeps the fixed B=16 shape. The generate loop's overflow guard reads the live width each cycle. Also from #784: the `hipfire-autoheal` playbook 11 (loader-wedge GPU reset). #784's SSE prefill keepalive landed earlier in this section.
- **DFlash on gfx1201: the hidden-ring commit/scatter fusion and the exact-F16 residual producers (S2/S4 of #759, by HUSRCF) now run on the R9700 too; +1.8 % DFlash decode, byte-identical.** Both were exact-gfx1100 only. `ArchCaps::supports_dflash_hidden_scatter_fusion` and `supports_dflash_f16_residual_fusions` now admit exact gfx1100 and exact gfx1201; gfx1151 and every other arch keep the per-row copies and the F32 residual path.
  - On gfx1201 the direct-F16 residual consumer launches the same module, symbol, grid and block that `gemm_hfq4g256_residual_wmma_gfx12_mq4v2` picks for a verify batch of at most 16 rows (the ldsstage kernel when enabled and K % 512 == 0, else the gfx12 base), fed the producer's F16 sidecar instead of a separate F32→F16 convert.
  - The S2 and S4 parity oracles (`test_dflash_hidden_scatter_gfx1100`, `test_mq_f16_residual_producers_gfx1100`) pass bit-exact on gfx1201.
  - E1 DFlash fixture on an R9700, 4 fresh processes per arm, ABBA then BAAB: 56/56 rows keep the same committed tokens and τ; code rows 314.2 → 319.9 tok/s (+1.83 %), the 1,100-token prose row 55.3 → 56.1 tok/s (+1.37 %), every fused process faster than every baseline process. On a 7900 XTX (gfx1100) the fixture's committed tokens and τ are unchanged and both oracles pass. The gfx1201 A4/fp8 and gfx1100 KLD pins are unchanged.
  - Kill switches as before: `HIPFIRE_HIDDEN_SCATTER_FUSE_OFF=1`, `HIPFIRE_MQ_F16_RESIDUAL_OFF=1`.
- **The kernel-pack release workflow (`.github/workflows/release.yml`) uploads with `gh`; its first tag push failed to start (`startup_failure`, no jobs).** The repo's Actions allowlist admits only GitHub-owned actions and a few named ones, so `softprops/action-gh-release@v2` was blocked. The last step now runs the runner's `gh`: it creates the tag's release as a draft if there is none, then `gh release upload <tag> dist/* --clobber`. An existing release keeps its draft, prerelease and latest state.
- **Registry: the three `qwen3.8:27b-mq4l*` tags move from `registry/models.json` to `registry/pending/mq4l.json`, unpublished, so `scripts/registry_gen.py` and the daily `registry.yml` refresh run clean again.** `quant_for()` knows no `mq4l` quant, so the generator had failed closed on those entries since they were curated, and the published `registry/v1.json` never listed them. Regenerated `v1.json` holds the same models, aliases and digests as before; only the key order and `§` escaping of the `qwen3.8:27b-mq4-xts` entry take the generator's form. The entries go back into `models.json` once the generator supports mq4l (target 0.4.1).


### v0.4.0 — KV backend, long context, and gfx11 prefill

**Breaking migration** (not yet tagged or released).
- Compiler-free kernel loading now requires a nonempty `.hsaco` and a matching `.hash` sidecar for cold and legacy objects. Missing, stale or empty pairs fail with the module, architecture, expected key and artifact path instead of loading an unverified image; reinstall the kernels or install hipcc to rebuild. Content-keyed hot hits and compiler-available recompilation remain unchanged.

- Exact gfx1201 symmetric MQ4V2 A4 prefill now embeds the certified builder `_b1` code object for SET, ADD and fused gate/up-SiLU. `HIPFIRE_G12_IU4_ISA=0` disables it; unsupported K/row-quad/output alignment falls through to hipcc `_v3`. H2 and mq4-xts WT2/code24 `.kldseq` pairs and one greedy decode were byte-identical; same-binary card-A fresh-process ABBA×2 improved pp8192 4,138.05 → 4,429.2 tok/s (+7.04%) with tg1 +0.197%. Default-on battery and chain completed five coherent turns each. Evidence: `/home/kaden/qcal/perf/iu4-6k/s3/report.md`.
- Exact gfx1201 symmetric MQ4V2 A4 builder `_b1` now prefetches the next K128 block's slab-1 activations/weights into its retired payload ring during the current block's fold; next-block scale/header loads also start before the B1 barrier. SET, residual ADD and fused gate/up SiLU retain their WMMA/fold order and epilogues. Card-A H2 N8192 paired ABBA+BAAB kernel throughput improves +0.31% gate/up, +2.09% down and +0.76% SET, each positive in both orders; all three N512 sites also improve. Original-vs-new whole-buffer H2 poisoned parity passed 200 serial and 200 same-CU repeats at N8192/1024/1023/513/512/256/255 for all three families. WT2 and code24 24-chunk prefill `.kldseq` files remain byte-identical to their H2 A4 pins. Eight fresh-process daemon ABBA+BAAB runs improved pp8192 4,584.85 → 4,620.30 tok/s (+0.773%, both orders positive); measured forward time fell 13.709 ms. Evidence: `/home/kaden/qcal/perf/iu4-6k/b1-prefetch/report.md`.
- On exact gfx1201, Qwen H24/KV4/D256 full-attention prefill fuses Q deinterleave, Q/K RMSNorm and RoPE into one wave-per-row kernel and preconverts Q into the exact E4M3 codes and F32 row scales consumed by the Q-resident v2 attention twin. `kernel.gfx12_fa_prep_fused=false` / `HIPFIRE_GFX12_FA_PREP_FUSED=0` restores the old three-stage prep; `kernel.gfx12_fa_prep_fp8q=false` retains F32 Q with fused prep. Card-A pp8192 rocprof measured 2.145 → 1.306 ms/layer prep and 5.791 → 5.814 ms/layer attention, saving 0.816 ms/layer (about 13 ms/pass); fresh-process ABBA improved pp8192 3,713.4 → 3,732.7 tok/s (+0.518%). WT2 c24 stayed byte-identical (KLD 0.083278, kldseq md5 `9d0e860f41db992820ebdc9483c0a041`); gfx11 remains unchanged. Evidence: `scratch-g12attn/REPORT.md`.
- gfx1201 native-FP8 prefill attention now runs a bit-exact v2 schedule of the register-resident-Q kernel on exact gfx1201 (same geometry, WMMA order and rounding points as v1): prefetched K/V tiles with pre-transposed V loads, branch-free masking, an exact FMA-corrected x/448, bracketed fp8 code conversion in place of per-element IEEE division, and heaviest-first workgroups. `kernel.attn_qresident_v2=false` or `HIPFIRE_ATTN_QRESIDENT_V2=0` restores v1. On card-A the pp8192 attention kernel fell from 10.39 to 5.77 ms per call (78.9 → 143 causal TFLOPS), fresh-process pp8192 rose 3,607.6 → 3,717.1 tok/s (+3.0%), and WT2 c24 stayed byte-identical (0.083278, kldseq md5 `9d0e860f41db992820ebdc9483c0a041`). Evidence: `scratch-g12attn/REPORT.md`.
- gfx11 symmetric MQ4V2 full-tile IU4 prefill now stages X5 activation packets ahead of each WMMA half on exact gfx1100/gfx1151 for eager M128×N128, K-multiple-of-256 SET/ADD; partial tiles, other architectures, and non-eager paths retain the incumbent. `HIPFIRE_IU4_X5=0` opts out. On the prior X5 branch, Q8/VMM two-cycle fresh-process ABBA improved pp8192 +9.13% on Halo and +4.74% on gfx1100, with tg128 within 1% and byte-identical q8/q8 WT2 c24 outputs. The composite `mq4-lloyd` gate and absolute throughput are recorded in `scratch-land042/REPORT.md`.
- VMM is the default KV backend. The former `contiguous` backend spelling is renamed `legacy`; the old name is rejected with a migration message (`use --kv-backend legacy` / `memory.kv_backend = "legacy"`).
- Legacy loads emit one stable warning carrying the `HIPFIRE_KV_BACKEND=legacy` token (automatic fallback or explicit operator choice). Loaded JSON, bench `--json`, and harness output expose `kv_backend*` fields (`kv_backend`, `kv_backend_request`, `kv_backend_reason`, `kv_backend_legacy`, `kv_backend_warning`).
- New public `--kv-k` / `--kv-v` flags and typed `memory.kv_k` / `memory.kv_v` (Qwen family) override K and V independently of the whole-cache `--kv-mode` / `memory.kv_cache` preset.
- **Breaking:** on Qwen, `asymN` / `turboN` (and bare `turbo`) now mean FWHT-N (e.g. `asym3` → FWHT3). The previous Givens/rotated asym K requires the explicit `legacy-asymN` spelling.
- Qwen-family default KV is q8/q8 everywhere except eligible exact gfx1201 (native fp8). Other families are unchanged (Maple BF16, DeepSeek compressor F32, Gemma lowered policy, etc.). Emergency kill switch: `HIPFIRE_QWEN_KV_DEFAULT_Q8=0` restores the prior implicit Qwen per-site default without overriding authored CLI/config or the gfx1201 FP8 path.
- For eligible growing Qwen VMM KV, the omitted `max_seq` default is `min(trained model context, measured post-weight card capacity)`; legacy and other owners retain their prior bounds. Registry tags no longer pin `max_seq`; an explicit CLI/config override still wins.
- On exact gfx1201, native-FP8 prefill attention accepts contexts through 262,144 tokens (formerly 32,768) across the Qwen packet admission, dispatch, and compute ceilings. Growing VMM `max_seq` still reserves the **minimum 512-row full PBS** rather than the optional lean or widened scratch; larger rungs are admitted against free VRAM per request.
- Native Qwen fp8/bf16 now resolve as indivisible K+V layouts rather than exposing a phantom Q8 V axis. The measured `max_seq` card bound uses the resolved cache stride (gfx1201 H24/KV4/D256: fp8 1,032 B per K/V row, 33,024 B per token across 16 KV layers; q8 34,816 B per token). Loaded/diag/bench JSON and the serve harness report effective `kv_mode=fp8` for native fp8.

- Exact gfx1100 now defaults to whole-chunk Q8/Q8 FA2 prefill (`kernel.gfx11_q8_fa2_wide=auto`); gfx1151 remains default-off, with explicit true available for experiments and explicit false opting gfx1100 out. On Qwen3.8-27B MQ4-XT with q8 VMM and lean PBS, gfx1100 matched ABBA pp8192 gained +1.423%, pp512 changed −0.375%, decode −0.229%; real-layer attention and graph replay were bit-exact, TTFT-511 regressed 0.082%, WT2 q8/q8 c24 outputs were byte-identical at KLD 0.076879, and the serve battery passed 5/5. Halo failed the throughput gate (pp8192 −0.366% after a third pair), so its default stays off.
- Exact gfx1100 symmetric MQ4V2 full-tile IU4 prefill (eager, M%128 == N%128 == 0) now routes SET/ADD to GEMM v2 V2C with the unchanged MQ4V2 weights and `block_i4_128` activations. On the XTX, pp8192 improved 2,334.7 → 2,564.9 tok/s (+9.86%) and pp512 +12.96%; WT2 c24 q8/q8 KLD sequence was byte-identical. `HIPFIRE_IU4_V2C=0` restores X5; gfx1151 is unchanged.
- Exact gfx1151 symmetric MQ4V2 full-tile IU4 prefill (eager, M%256 == N%256 == 0, at least two rounds of the grid: (M/256)·(N/256) ≥ CU count) now routes SET/ADD to GEMM v2 V2B (M256×N256, 16 waves, 1 CTA/WGP) with the unchanged MQ4V2 weights and `block_i4_128` activations, bit-identical to X5; smaller grids keep X5. On Halo the pp8192 SET+ADD GEMM time fell 7,338 → 5,696 ms (−22.4%), and fresh-process Q8/VMM ABBA ×2 against mq4-lloyd improved pp8192 835.5 → 991.8 tok/s (+18.71%) and pp512 +17.78% with tg128 −0.02%; WT2 c24 q8/q8 KLD sequence byte-identical. `HIPFIRE_IU4_V2B=0` restores X5; gfx1100 is unchanged.
- F1-lite on exact gfx1201: with symmetric MQ4V2 weights and the gfx12 AWQ IU4 SwiGLU producer (eager, `Residual` epilogue, gate/up M%128), one launch of the production gfx12 IU4 SET tile folds gate and up rows (16-row groups interleaved by address; artifact unchanged) and stores `h = silu(g)*u` in FP32; `fused_silu_mul_mq_rotate_awq_i4_hin_gfx12` reads h instead of gate + up. A two-group raster keeps each group's weights cache-resident like one SET's. Exact: WT2 c24 fp8 KLD sequence byte-identical (0.083278); existing SET/ADD entries are instruction-identical. pp8192 trace GPU −47.2 ms; card-A fresh-process ABBA pp8192 3,606 → 3,692 tok/s (+2.38%), tg128 flat. `HIPFIRE_F1LITE=0` restores the SET pair and producer.
- Exact gfx1201 IU4 prefill (SET, ADD and the F1-lite gate/up SiLU entry) now builds `_g12r` modules by default (`HIPFIRE_G12_RASTER=0` restores the incumbent modules). CTAs walk bands of 8 row tiles, rows fastest and then tokens, so the ~96 resident CTAs share an L2-sized W/X working set; this replaces the SiLU entry's two-group raster. Full tiles also store through one LDS tile as whole b128 token columns: 512 B for SET/ADD, 256 B of h for SiLU. SET/ADD use 8 barriers instead of 16. The K order, fold DAG and SET/ADD/SiLU arithmetic are unchanged, and outputs are bit-identical: an all-bit oracle covers the full pp8192 shapes plus M/N tails. In pp8192 rocprof on card-E (same binary, off→on), IU4 GEMM time fell 1,640.3 → 1,585.4 ms (−3.35%, every shape faster), total GPU time fell 55.5 ms (−2.62%), and traced prefill rose 3,891 → 3,996 tok/s. The WT2 c24 fp8 KLD sequence is byte-identical. See `scratch-g12raster2/REPORT.md`.
- Qwen3.5 four-MQ4V2 linear-attention beta/alpha prefill rows now fold into the load-time Z MQ4V2 matrix on exact gfx1100/gfx1151, reusing the existing IU4 SET selector and splitting its F32 result; decode and fallback weights remain unchanged. `HIPFIRE_IU4_BAFOLD=0` restores the separate MW4 beta/alpha tail and F32→F16 conversion. In isolated pp8192 traces, targeted kernel time fell 11.3 ms on gfx1100 and 114.3 ms on gfx1151 (with V2B); c24 q8/q8 WT2 KLD changed +0.000392 and −0.000928 respectively, within the +0.0005 limit but not byte-identical. Integrated ABBA throughput remains gated separately.
- F1-lite: on exact gfx1100 (V2C) and gfx1151 (V2B), when an FFN's gate/up pair would run the GEMM v2 tile and `down_proj` takes the AWQ IU4 SwiGLU producer (eager, uniform MQ4G256V2, `Residual` epilogue), one GEMM v2 launch folds the gate and up rows, address-interleaved at 16-row granularity (weights and artifact unchanged). Its epilogue stores `h = silu(g)*u` in FP32. The AWQ IU4 producer then reads h instead of gate + up; FWHT-256 needs the whole row, so the rotation and quantisation stay in that pass. Exact: the WT2 c24 q8/q8 KLD sequence is byte-identical on both cards (0.076879 / 0.076901). In pp8192 traces the targeted FFN kernels save 38.0 ms on XTX and 167.3 ms on Halo, where the request wall fell 1.76%. `HIPFIRE_F1LITE=0` restores the SET pair and gate/up producer.
- gfx11 Qwen3.8 GDN chunk prefill now swaps the prep grid to row-major on exact gfx1100/gfx1151, preserving the original FP32 prefix and Q/K/V arithmetic; gfx1100 also solves each KKT value head independently. The exact C64 scan and Halo KKT remain unchanged. `HIPFIRE_GDN_PREP_GFX11=0` and `HIPFIRE_GDN_KKT_GFX1100=0` restore the incumbents. Isolated pp8192 ROCprof prep fell 72.060→69.728 ms on XTX and 307.420→267.312 ms on Halo; XTX KKT fell 48.487→31.162 ms. Integrated stack WT2 c24 q8/q8 sequences are byte-identical with both levers off/on (KLD 0.077271 XTX / 0.075973 Halo). Halo C32 was rejected after a real-layer state-parity and single-chunk quality regression; see `scratch-gdn/REPORT.md`.
- Exact gfx1100 GEMM v2 V2C residual ADD now runs `gemm_mq4g256v2_residual_iu4_v2c_add_touch_gfx11`: starting 16 K128 epochs before the end of the K loop, each thread loads one dword of one 64-byte residual line per epoch ahead of the staging packet, so the ADD epilogue reads the residual from cache. The output bytes are unchanged, and SET, gate/up and the other architectures are untouched. `HIPFIRE_V2C_ADDEPI=0` restores the plain `_add`. In pp8192 rocprof on the XTX (same binaries, off→on), V2C ADD time fell 755.9 → 744.1 ms and total GPU time fell 3,100.3 → 3,091.4 ms (−0.29%). The WT2 c24 q8/q8 KLD sequence was byte-identical off and on. The alternative, folding the out-projection residual add into the IU4 RMSNorm, was exact but slower than the touch on gfx1100, so it was not taken.
- On exact gfx1100/gfx1151, the gfx11 Q8 FA2 prefill (Q16 tile) now uses a warp-specialized K/V fill. The two helper waves dequantize V(t) while the six compute waves run QK(t), then dequantize K(t+1) while the compute waves run softmax/PV(t). The raw bytes are register-pipelined only in the helpers, so the kernel still uses 256 threads and 32 KiB LDS, and VGPRs fall from 218 to 188. The output is bit-exact. `HIPFIRE_FA2_FILL=0` restores the all-wave per-tile fill. In pp8192 rocprof (same binaries, off→on), FA2 time fell 347.4 → 268.7 ms on XTX (−22.6%) and 830.6 → 617.0 ms on Halo (−25.7%). Total GPU time fell 63.0 ms (−2.05%) on XTX and 187.6 ms (−2.46%) on Halo. The WT2 c24 q8/q8 sequences are byte-identical (KLD 0.077271 XTX / 0.075973 Halo). See `scratch-fa2fill4/REPORT.md`.
- gfx11 Q8 FA2 16-query tiling: exact gfx1100/gfx1151 use Q16 (grid ceil(batch/16)x4, block 256, KT32, 32,768-B dynamic LDS, 2 resident blocks/CU, 0 spills); other gfx11 targets and fwht3-K retain Q8. Bit-exact vs the incumbent over the full 16-layer H24/KV4/D256 8192-query causal shape (805,306,368 f32 outputs, zero unequal words). Fresh-process Q8/VMM ABBA on Qwen3.8-27B MQ4-XT: gfx1100 pp8192 +1.992% (ship pass), no row >1% below landed; Halo pp8192 +0.576% / pp512 -0.698% (stackable-only exception, default-route FA2 trace -19.78%). WT2 q8/q8 c24 KLD gfx1100 0.076879/0.076879, gfx1151 0.076901/0.076901; serve battery 5/5 on both.
- Batched RoPE now vectorizes across heads with shared angles: one 256-thread block per position computes the angle table once into dynamic LDS (2*half f32) and consecutive threads rotate adjacent dimensions across Q/K heads, retaining the incumbent angle expressions and operation order for bit-identical output. The interleaved per-pair kernel remains behind `HIPFIRE_ROPE_INTERLEAVED_LEGACY=1`; other shapes keep their existing routes.
- Q8 flash attention now bounds the reducer LDS below 64 KiB/workgroup: `q8_flash_tile_size` raises any preferred tile (including an explicit `HIPFIRE_Q8_FLASH_TILE` override) just enough that the maximum graph/replay-stable tile count plus the gated reducer's head-dimension scratch fits in 32 KiB LDS. At H8/D256 with a 262,144-token KV reservation the policy selects tile 64 (4,096 correction floats) instead of faulting with `HSA_STATUS_ERROR_INVALID_ALLOCATION` (65,544 B requested) in `attention_flash_q8_0_reduce`; short-context tile choices are unchanged. A focused CPU test pins the 2K, 262K, and 1M reservation boundaries.
- gfx11 lean ordinary-prefill PBS now defaults on for the admitted fused MQ4V2 route: it frees ~1.9 GB of unused scratch and removes pp8192 admission slicing on 24 GB cards. The VMM q8 gfx1100 gate improved pp8192 by +1.22%; WT2 was bit-identical. Non-fused, verify, and other-architecture routes retain full scratch; `kernel.gfx11_lean_pbs=false` opts out.
- gfx11 C prefill baseline defaults on: partial-N IU4 gridspec, gfx1100 LF16 shape, symmetric-only gfx1100/gfx1151 IU4 symfold, and the two-point {5,7} fused-producer A4 search; gfx1201 and asymmetric artifacts retain their prior routes. Qwen3.8-27B XT q8 matrix: gfx1100 pp512 2,125.2 / pp8192 2,070.9 tok/s; gfx1151 796.0 / 737.3 tok/s. WT2 q8/q8 c24 KLD: gfx1100 0.076879 / gfx1151 0.076901; serve battery 5/5 on both. Source: `exp/gfx11-stack` at `f62eee968`.
- gfx11 IU4 partial-N GEMMs now default to an unchecked full-tile interior plus one guarded tail on exact gfx1100/gfx1151; `kernel.gfx11_iu4_gridspec=false` or `HIPFIRE_GFX11_IU4_GRIDSPEC=0` restores the checked whole-grid route, and gfx1201 behavior is unchanged. The route is bit-exact across 64 fresh-process comparisons, corrected nonzero-residual ADD smokes (0/30,254,080 mismatches on each card), and byte-identical OFF/ON c24 outputs. Exact-5,909 graph-on TTFT prefill improved 1,616.6→1,651.4 tok/s (+2.2%) on gfx1100 and 519.8→651.1 tok/s (+25.2%) on gfx1151. This gain is invisible to pp8192 by construction: 8192 is an exact multiple of the 128-column tile and already dispatches the unchecked route, while real prompt lengths are not; Halo's 519.8 tok/s baseline TTFT prefill against its approximately 690 tok/s pp8192 baseline shows that the partial-N penalty was real and simply unmeasured.
- gfx1201 native-FP8 prefill attention now defaults to the register-resident-Q H24/KV4/D256 route; `kernel.attn_qresident=false` or `HIPFIRE_ATTN_QRESIDENT=0` restores the packet body, and every other architecture/shape keeps its existing route. On Qwen3.8-27B MQ4-XT, paired card-A gates improved TTFT by 1.865% and 3.153% and pp8192 by 1.727% and 1.860%, with pp512 never regressing, tg128 within 0.1%, both quality routes passing, and a 5/5 no-think serve battery. The clean flag-free headline is 1,635.666 ms / 3,612.60 prompt tok/s for the canonical 5,909-token prompt.
- Symmetric MQ4V2 artifacts on gfx1201 now select centered FP8-v2 GEMM twins for gate/up, residual, qkvza, and qkv prefill: packed nibbles expand directly through an E4M3 `(q-8)` table, scale-only metadata replaces scale+zero staging, and each K128 partial uses one scale FMA instead of scale+zero correction. `HIPFIRE_FP8_SYMFOLD=0` restores the asymmetric twins for developer comparison; non-symmetric artifacts are unchanged. On Qwen3.8-27B XT, paired pp8192 improved by 6.1–7.4%, the four profiled v2 GEMM families fell 8.4% in aggregate, fp8/q8 WT2 c24 KLD was 0.049063, and the no-think serve battery passed 5/5.
- `hipfire-quantize --awq-a4-aware` keeps MQ4V2's wire/header and FWHT seeds unchanged while selecting each shared activation group's AWQ alpha from `{0.35,0.45,0.55,0.65,0.75}` by sampled W4A4 block-output error. Activation fake quantization mirrors the gfx1201 four-candidate `block_i4_128` MSE recipe bit-for-bit, and fused qkvza/gate-up siblings reuse the anchor projection's alpha.
- Drop obsolete `HIPFIRE_LLOYD_GFX12` opt-in: MQ3/MQ4 Lloyd WMMA batched prefill is always-on for gfx1200/gfx1201 (validated since PR #195).
- W4A4 iu4-direct MQ4V2 prefill is now default-on through `kernel.iu4_prefill` on its exact gfx1100/gfx1151/gfx1201 kernel routes; unsupported architectures keep their incumbent path. `hipfire config set kernel.iu4_prefill false` opts out globally, and the same key supports per-model opt-out; `HIPFIRE_IU4_PREFILL` remains a developer override.
- gfx1201 MQ4-XT prefill defaults (opt-out): `kernel.gfx12_mq4v2_fp8_v2` default ON on exact gfx1201 (staged-tile v2 at N>=256, geometry default 128x128 with 2 slabs — `HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` still selects 64x256/128x64/256x64, `=0` restores s2bt8/BT); `kernel.gfx12_gdn_pre_fused` default ON (byte-exact 3→1 preamble); `prefill.chunk_rows` replaces the widened perf ceiling (8192 on exact gfx1100/gfx1151/gfx1201, 512 elsewhere, `HIPFIRE_PREFILL_CHUNK_ROWS`, explicit `HIPFIRE_PREFILL_MAX_BATCH` wins, VRAM rung admission unchanged — daemon logs the requested/admitted rows); unset/auto `memory.kv_cache` means native fp8 on exact gfx1201 single-GPU Qwen (explicit modes untouched, pp excluded). `HIPFIRE_GFX12_FA2_FP8` / `kernel.gfx12_fa2_fp8` deleted: fp8 KV on FA2-eligible shapes always takes the stage-b route-N body (route-Q fail-louds removed with it; the caller-less route-Q launcher is retained for B4). iu4 is default-on through `kernel.iu4_prefill`. Evidence on Qwen3.8-27B XT (GPU 0): the scan-on c24 WT2 KLD is 0.078397 against the 0.078514 acceptance ceiling.
- The portable GDN chunk scan is default-on for the exact Qwen3.8-27B dense-prefill shape on gfx1100/gfx1151/gfx1201 (`kernel.gfx12_gdn_chunk_scan`; `HIPFIRE_GFX12_GDN_CHUNK_SCAN=0` opts out). The scan evaluates causal decay as bounded `exp(G_i-G_j)` / `exp(G_last-G_j)` factors, folds the Q8 error-feedback residual into the initial state before decay, accepts full 512-row segments plus guarded 64..511 tails, and retains the incumbent path for every failed predicate. Final pinned-daemon medians (OFF→ON): pp512 2419.5→2656.0, pp2048 2503.2→2785.5, pp8192 2433.0→2709.8, pp32768 2086.6→2284.1, decode 36.4886→36.4846 tok/s. gfx1100 q8 WT2 c2 is 0.055645; gfx1201 fp8/q8 c24 is 0.078397 (accepted under the 0.078514 ceiling).
- Redline generalized tape slice 1: the retained PM4 tape builds kernarg segments from `ReplayBindings` + `KernargAbi` (pointer slots per launch, everything else recorded bytes) instead of the raw snapshot, with a byte-equality fail-closed gate (`HIPFIRE_REPLAY_BINDINGS_VERIFY=1` for release) and a per-tape typed/untyped census at prepare. Scratch growth no longer drops the retained route: the revision bumps and segments re-encode in place (no re-lowering); a resource that actually moved still fails closed to HIP and re-capture. Zero behaviour change expected; gfx1100/1151/1201 harness exact + pm4_ib gates pending in the slice receipts.
- MQ4-Lloyd (qt=52 `MQ4G256V2L` / `DType::MQ4G256V2Lloyd`): same 136 B V2 group container as qt=44 plus per-tensor F32[16] `lloyd_levels` sidecar; product tiers `mq4l-xt` / `mq4l` / `mq4l-pro`. WT2 KLD vs AWQ'd uniform tiers (24-chunk, gfx1201, prefill scoring, q8 KV): xt 0.0482→0.0408, base 0.0405→0.0346, pro 0.0335→0.0284.
- mq4-pro batched-prefill fix: invalidate the F16 x-cache after `gated_norm_f32_batched` / `sigmoid_mul_f32` so Q8 `out_proj` does not read stale activations under the default gfx1201 FP8 prefill path (1-chunk WT2 KLD 6.42→0.019).
- AWQ rmsnorm-rotate rewrite: `fused_rmsnorm_mq_rotate_awq` is derived from `fused_rmsnorm_mq_rotate.hip` via `-DHIPFIRE_RMSNORM_AWQ` (retired the LDS-staged fork and the gfx1100-only `_direct` variant / `HIPFIRE_GFX1100_AWQ_NORM_DIRECT`); +2.5% decode on AWQ'd MQ4V2 artifacts on gfx1201.

- gfx11 W4A4 iu4-direct MMQ prefill (opt-in `HIPFIRE_IU4_PREFILL=1`, exact gfx1100/gfx1151, `kernel.iu4_prefill`; `=0`/unset keeps the X128 path byte-for-byte). Weight nibbles feed `wmma_i32_16x16x16_iu4` directly (halved A-side LDS, 31744 B) with int4 activations from the `quantize_int4_mmq_ds128` prelude (per-128 MSE-clip grid, f32 single-rounding FMA). Measured on Qwen3.8-27B XT (`qwen3.8-27b.mq4-xt`): XTX pp512 1203.6->1588.8 tok/s (+32%), pp2048 1185.4->1557.0 (+31%), pp8192 1093.9->1405.1 (+28%); Halo pp512 449.9->581.3 (+29%), pp2048 442.0->568.8 (+29%), pp8192 412.5->519.2 (+26%); decode tok/s unchanged (XTX 48.82->48.92, Halo 14.22->14.18). In-model pp512 profile: `full_set_occ3` 272x 581 us/call, `full_add_occ3` 128x 650 us/call (XTX; Halo 1721/1683 us), prelude `quantize_int4_mmq_ds128` 256x 75 us/call (was 370 us f64; Halo 153 us). Gates: int4 pre-pass bit-oracle 4/4 arms on both GPUs, GEMM parity relL2~5e-8, 0 spills on both archs, greedy HumanEval `below_zero` byte-identical off/on (md5 b84c1c19), WT2 KLD 0.057307->0.076078 (+0.0188, ship rule on<=off+0.02), serve battery 5/5 turns finish=stop with 0 runaway/empty/attractor.
- GQA-fused FA2 prefill attention is now the default (opt-out): `HIPFIRE_GFX12_FA2_PREFILL` default ON on exact gfx1201 (`kernel.gfx12_fa2_prefill`; `=0` restores the incumbent) and `HIPFIRE_GFX11_FA2_PREFILL` default ON on gfx1100/gfx1151 (`kernel.gfx11_fa2_prefill`; `=0` restores the incumbent). Both cover the Qwen NH24/NKV4/HD256 prefill envelope (Q8-K and fwht3-K arms, every batch 64..512 including guarded partial query tiles, ctx 64..32768, eager only); launchers keep their exact arch/shape/eager predicates.

- gfx11 GQA-fused FA2 prefill attention (opt-in `HIPFIRE_GFX11_FA2_PREFILL=1`, gfx1100/gfx1151, Qwen NH24/NKV4/HD256): 7-shape oracle PASS on XTX+Halo, 0 spills, KT32 pinned. qwen3.8-27b.mq4-xt matrix pp tok/s off->on: XTX 1177.1->1189.8 (+1.1%) @512, 1081.7->1172.9 (+8.4%) @2048, 786.0->1085.9 (+38.2%) @8192; Halo 443.2->446.2 (+0.7%), 411.4->436.6 (+6.1%), 301.7->405.0 (+34.2%). pp8192 attention 1616.5->810.4 ms (XTX), 3332.9->1870.4 ms (Halo). Greedy short prompt bit-identical (1200 tok); WT2 KLD 0.057961->0.057307 (ship gate on<=off+0.0005).
- gfx1201 FP8-WMMA MQ4v2 prefill is now the default (opt-out): `HIPFIRE_GFX12_MQ4V2_FP8_{GATEUP,RESID,QKVZA,QKV}` default ON on exact gfx1201 (`=0` on any one restores the F16 route), the two-slab S2BT8 symbols are the default (`HIPFIRE_GFX12_MQ4V2_FP8_SLABS=1` selects single-slab), and the admitted path sizes the prefill chunk at 512 instead of 384. Measured on Qwen3.8-27B XT (`qwen3.8-27b.mq4-xt`, GPU 0): pp512 1387 vs 841 tok/s opt-out, pp2048 1218 vs 812 tok/s, tg1@128 unchanged (~36.5 tok/s). Quality delta is nil on the evidence run: greedy (`-t 0 -n 1200`, `benchmarks/prompts/humaneval_3_below_zero.txt`) is byte-identical between the default FP8 path and the all-flags-`=0` F16 path, and the serve battery reports 5/5 coherent turns (runaway=0 empty=0 attractor=0).
- gfx11 MMQ per-128 (X128) activation path is now the default on gfx1100/gfx1151 (`HIPFIRE_GFX11_MMQ_X128=0` restores the per-32 route; other arches stay per-32). The X128 prelude quantizes activations per 128-K half and the consumer accumulates each half in one i32 WMMA tile with a single float correction. WT2 KLD gate passed on XTX: 0.058189 -> 0.058375 (+0.000186 <= +0.0005); greedy HumanEval `below_zero` output is bit-identical off/on on both GPUs. Measured on Qwen3.8-27B XT: XTX pp512 990 -> 1141 tok/s (+15%), pp2048 927 -> 1059 (+14%), full_set 1017 -> 864 us/call, full_add 1151 -> 956 us/call; Halo pp512 377 -> 438 (+16%), pp2048 354 -> 407 (+15%); decode tok/s unchanged. Serve battery on XTX with defaults: 5/5 turns finish=stop, 0 runaway/empty/attractor.
- gfx11 MQ4V2 prefill: route the GDN beta/alpha tails (M=48, K=5120) in
  `gemm_qkvza_mq4g256v2_wmma` off the MMQ base kernel to the unified MW4 f16
  kernel via a zeroed-Y SET wrapper (`gemm_mq4g256v2_small_tail_set`).
  Throwaway-measured at (48, 5120, N in 128..512), ABBA x3: MW4 wins every
  cell (Halo 38..50 us vs MMQ 284; XTX 61..90 us vs MMQ 373..458) at relL2
  ~3e-4 vs CPU f64. In-model pp512: XTX 993.5 -> 1018.6 tok/s (+2.5%), Halo
  377.2 -> 383.4 (+1.6%); a/b rows 96x~283 us -> 96x~36 us; greedy md5
  bit-identical before/after on both GPUs.
- gfx1201 FP8-WMMA MQ4v2 prefill is now the default (opt-out): `HIPFIRE_GFX12_MQ4V2_FP8_{GATEUP,RESID,QKVZA,QKV}` default ON on exact gfx1201 (`=0` on any one restores the F16 route), the two-slab S2BT8 symbols are the default (`HIPFIRE_GFX12_MQ4V2_FP8_SLABS=1` selects single-slab), and the admitted path sizes the prefill chunk at 512 instead of 384. Measured on Qwen3.8-27B XT (`qwen3.8-27b.mq4-xt`, GPU 0): pp512 1387 vs 841 tok/s opt-out, pp2048 1218 vs 812 tok/s, tg1@128 unchanged (~36.5 tok/s). Quality delta is nil on the evidence run: greedy (`-t 0 -n 1200`, `benchmarks/prompts/humaneval_3_below_zero.txt`) is byte-identical between the default FP8 path and the all-flags-`=0` F16 path, and the serve battery reports 5/5 coherent turns (runaway=0 empty=0 attractor=0).

## v0.3.1 — DFlash cache repair, admission hardening, image gen

- Sealed MoE execution contracts (#755, fivetide; G5 constituent): manifest-derived expert plans bound transactionally to Qwen3.6-A3B / Ornith and Cohere MoE decode and prefill; raw MoE escape hatches removed; rank-local sealed EP with root-authoritative routes and owned reduction leases, verified on 4× R9700. EP cross-route logit equivalence is diagnostic, not an acceptance gate — G5 acceptance and PM4 admission remain open on #666.
- gfx1201 admits the AlpineQ R4/R8 Q8 multi-row verifier for long-context DFlash (#748, HUSRCF). Measured on Qwen3.8-27B XT at 21,550 tokens: AR 32.6 → DFlash 39.3 tok/s (+20.6% over AR); the previous batched route was a 19% loss vs AR at that context. The route is excluded under HIP graph capture and retained/PM4 recording.
- Daemon slots restore tool turns (#753, alpineQ): daemon-owned canonical tool-call ids survive parsing and OpenAI lowering, pending tool results are brokered per session/call and published only after Commit, Jinja tool-history projection. Slot grammar honours the documented `HIPFIRE_QWEN35_GRAMMAR` switch; rich history keeps `reasoning_content` for the `qwen3_5`/`qwen3_5_moe` arch spellings the daemon emits.
- MTP prompt cache: strict-prefix terminal repair from the last window snapshot (#749, alpineQ) and a bounded DeltaNet checkpoint ring for divergent renders (#751, alpineQ), both default on; policy (`HIPFIRE_SPEC_WINDOW_ROLLBACK`, `HIPFIRE_DFLASH_CKPT_RESUME`, `HIPFIRE_CACHE_CKPT_INTERVAL`/`_MAX`) is resolved in `hipfire-config` (`mtp_cache_policy`). Evidence disclosure: on gfx1201 / Ornith-1.5-35B-A3B the repaired trunk KV is byte-identical to normally-stepped KV and the prompt-prefix rows are prefill-exact; the target DeltaNet state was not compared at an identical position and the MTP-head KV differs from step-written KV (draft-side only; verify-gated). Neither MTP stepping nor repair is byte-identical to a cold prefill of the same tokens — the first decode-path row already differs. The oracle lives in `crates/hipfire-arch-qwen35/tests/{mtp_cache_byte_identity,mtp_step_oracle}.rs` (`#[ignore]`, `HIPFIRE_MTP_BYTE_IDENTITY_MODEL`). Opt out with `HIPFIRE_SPEC_WINDOW_ROLLBACK=0` / `HIPFIRE_DFLASH_CKPT_RESUME=0`.
- Source-aware admission and refusal-before-teardown (#682, #687).
- Registry-declared DFlash draft sidecars: `pull` fetches them, `auto`/`on` semantics, shared-sidecar-aware `rm` (#686).
- DFlash prompt-cache repair on terminal overshoot (`RepairForTerminal`) (#695).
- Template-aware primer splice (#692).
- Qwen AR/DFlash: rich assistant reasoning history preserves the verbatim generated token span across template framing (whole-envelope store + Jinja splice); edited history falls back safely to plain retokenize.
- Transactional DFlash constructors with emitter rollback (#691).
- Qwen35 prefill/decode scratch and per-layer weight construction retain actual owners until publication, reclaiming every staged allocation on failure so immediate retries reuse the pool.
- MQ-V2 prefill admit rule (#690).
- gfx1100 DFlash launch fusion and split-K residual tiers (#702 body, S1–S8).
- Dense-TP prefill chunking equals arch batch × tp (#725).
- MTP head inherits trunk flash policy; tile-sized partials (#726).
- XML tool calls parsed with grammar off (#729).
- `HIPFIRE_RCCL_LIB` for non-standard ROCm layouts (#728).
- `memory.oom_guard` (default `auto`; env `HIPFIRE_OOM_GUARD`) (#697).
- `ornith-1.5:fast` alias → `ornith-1.5:35b-a3b-mq4r` (#680).
- `moe_topk_renorm_k8` barrier (partial #670, nwoolmer).
- Image generation: FLUX.1 schnell / FLUX.2 Klein via `hipfire img`, `POST /v1/images/generations` + `/edits`, `hipfire-quantize --flux-pipe` (philhug; first release; RDNA3/3.5 measured).
- Qwen3.8-27B vision as a shared sidecar: `qwen3.8-27b-vision.hfq` (F16 tower, mmproj-style) pairs with every text quant tier — `hipfire pull` fetches it, `run`/`serve` accept `--vision`, `hipfire-quantize --vision-only` packs it. No trunk requantization. Loading is gated by `vision_mode` (`vision.mode`, env `HIPFIRE_VISION_MODE`; default **`off`** = text-only, no tower VRAM): `hipfire config set vision_mode auto` loads the tower when present, `on` fails the load closed without it. Validated on the committed 6-image desc/OCR battery; tower parity vs HF is decoder-bounded (zune-jpeg vs libjpeg chroma), see `benchmarks/vision/`.
- Gate overhaul: `change_gate` / agentic-review retired (#700); hw-gate pins Qwen3.8 MQ4-XT and drops qwen3.6 as current fixture.
- S1+S2 dependency hygiene and panic-free config CLI (#701).
- All production `HIPFIRE_*` reads are config-owned.
- Opt-in VCN JPEG preprocessing for existing VL serving: `image.decode` stays `cpu` by default; `vcn`/`auto` attempt shared VCN decode with guarded JPEG dimensions and validated VA plane layout/ownership, falling back to CPU on unsupported inputs, unavailable platforms, or recoverable decode failure. A failed terminal GPU completion fails closed (quarantine + request error + nonzero daemon exit; restart required) instead of unsafe same-device CPU fallback. This is a JPEG prepass only — not a replacement vision tokenizer or learned tower.
- Manifest-route weight uploads go through the GPU buffer pool (`weight_store` pooled fulfillment + pool-return rollback) instead of raw `hip.malloc` paired with pooled frees, so repeated load/unload cycles on one context hold post-warmup free VRAM flat instead of retaining ~one model's weights per cycle. Single plain-manifest transactional load only — the legacy loader path is unchanged. Provenance: per-cycle upload journal count, decode parity, and pool-hit counters in the pinned-fixture cycle test; pooled manifest vs legacy forwards bit-identical. AWQ numerics now rest on a post-`output_norm` quantized-lm_head oracle (uniform 2.0-vs-4.0 sidecars forward at an exact 2:1 logit ratio, alternating sidecar proves per-channel application); the pre-norm o_proj pair only records sidecar attachment since RMSNorm erases global scales.
- DFlash weight and scratch constructors now roll back late failures for immediate retry, including pool-aware F32 leaf uploads and AWQ sidecar attachment.
- Preserve all eleven weight groups in K=2816 HFQ4/MQ4 MoE gate/up kernels while retaining the K=2048 path (#734 prerequisite; it did not itself enable Gemma serving — the lowered route arrives via #667 below).
- Maple head overlays and BF16 KV tier (#670, nwoolmer): `hipfire run maple-preview --head q4k|bf16` loads single-tensor head overlays that validate and attach during source admission before teardown (non-Maple, EP, and REAP combinations are refused there); truncated payloads refuse at open and short reads refuse instead of zero-filling. Flat BF16 KV tier with windowed attention kernels, selectable via `--kv-mode`; the batch-router GEMM selects a gfx12 WMMA sister kernel on RDNA4. No quality or performance claims are made here.
- Gemma 4 26B-A4B lowered route (#667): admission serves MoE/batched lowered loads end to end (`generate_gemma4_lowered`) instead of refusing them; `max_seq` is the logical authority (scratch flash partials and the full asym3 cache are sized from it) while the sliding Q8 ring's physical allocation is capped at `min(sliding_window, max_seq)`. The separate batched-prefill API shares single-token indexed MoE semantics (`moe_token_indexed`) with Q8 projections on an explicit F32 batched path (no automatic F16-staged WMMA); HD512 Q preload/lane alignment plus four-term dot association in the batched Q8 attention tile. Exact numerical parity across four prompt sizes (15/124/370/1108 tokens: byte-identical 262144 logits + 24 continuation tokens) on gfx1201 — that parity rig used a full-Q8 test cache, while serving/replay exercise the production mixed sliding-Q8/full-asym3 tiers; ordinary tokenwise-prefill chat serving: battery and chain each 5/5 coherent (no empty/attractor/runaway); stable 2-token prefill and 124-token-context decode captures with PM4 vs HIP/blob exact logits/KV/state across 3 successive positions. No broad quality or performance claims. Fixture: `gemma-4-26b-a4b-it.hfq4g128-maintainer.hf4` (15,343,188,028 bytes, sha256 `11cf46cba97f5e279d351f9d31cf4bdd78cb1fbc7c16da2433e6141ae7e07d53`), a maintainer-generated fixture distinct from the missing author artifact.
- Gemma lowered loads reject `max_seq < 128` before replacing the resident model. Partial weight, scratch and KV construction now reclaims owned GPU allocations for retry, including AWQ sidecars and position buffers; lowered weight uploads reuse the unload pool. The diagnostic oracle follows the production mixed-Q8/asym3 cache geometry for short contexts and sliding-ring rollover.

### Validation

Fixture: `qwen3.8-27b.mq4-xt` (sha256 `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`) with paired draft sha256 `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`. Routes run: battery / chain / session AR+DFlash, ornith `.mq4r` PM4 route proof, tp=2, MTP@8192.

## v0.3.0 — MQ V2 wire schema, Bonsai, Redline across RDNA

### Quant wire schema (Bonsai + Magnum V2)

This release locks the HFQM quant-type register through qt=50 and lands the
Magnum V2 header family for dense Qwen3.8 bodies. Authoritative IDs live in
[`docs/quant-formats/qt-register.txt`](docs/quant-formats/qt-register.txt);
layout and product boundary in
[`docs/quant-formats/mq4-v2.md`](docs/quant-formats/mq4-v2.md),
[`docs/quant-formats/mq-v2-family.md`](docs/quant-formats/mq-v2-family.md), and
[`docs/quant-formats/ladder.md`](docs/quant-formats/ladder.md).

- **Bonsai (passthrough):** qt=40 `TQ2G128` (ternary), qt=41 `BQ1G128` (1-bit).
- **Magnum GL (passthrough):** qt=38 `MQ2G256GL`, qt=39 `MQ3G256GL`.
- **Magnum V2 (passthrough):** dual fp16 scale+zero per 128 weights, with
  payload packing unchanged from matching v1 widths — qt=44 `MQ4G256V2`
  (136 B), qt=47 `MQ6G256V2` (200 B), qt=48 `MQ5G256V2` (168 B), qt=49
  `MQ3G256V2` (104 B), and qt=50 `MQ2G256V2` (72 B). Bits 2/3/4/5/6 group
  bytes are 72/104/136/168/200.
- **MQ4C (passthrough):** qt=45 `MQ4CG256` is a separate 136 B padded-compat
  layout: one fp16 scale+zero header per 256 weights at `[0..4)`, zero padding
  at `[4..8)`, and the nibble payload at `+8`.
- **CLI alias:** `--format mq4` resolves **qt=44 MQ4G256V2**. Legacy v1 MQ4
  remains `mq4v1` / `mq4g256` / `magnum` (qt=13). Artifacts self-describe by
  qt; old files are not reinterpreted.
- **Product ladder:** `mqN` / `mqN-xt` / `mqN pro` name measured model-bpw
  class and role lifts, not codec generation. Dense Qwen3.8 has measured
  evidence for MQ3–MQ6 V2 cells. **MQ2V2 is wire- and runtime-supported but
  quality-rejected** (catastrophic KLD on the dated ladder); do not ship or
  advertise it as a product body.
- **Cross-arch runtime (measured policy, not undated tok/s claims):** gfx1151
  weight-reuse defaults across V2 ops bits 2–6; gfx1201 QKV BT8 for bits
  2/4/5/6 (MQ3 base); gfx1100 MQ4 adaptive reuse and base V2 WMMA.
  Capture/replay retain fixed contracts; unsupported surfaces fail closed or
  fall back rather than inventing routes.

Operator and design docs:
[`docs/QUANTIZE.md`](docs/QUANTIZE.md),
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

### Redline across RDNA

Redline is now an in-tree dispatch and retained-replay substrate for the RDNA
family. It records hipfire's real decode graph, derives resource dependencies,
retains invariant command state, and lowers validated routes through public
ROCr queue interfaces. Unsupported graphs, failed shadow checks, ABI drift,
queue faults, and model changes fail closed to ordinary HIP dispatch.

Qwen 3.6 35B-A3B MQ4R is the release's performance SKU. Ordinary AR with Q8
KV reaches 253.3 tok/s on gfx1100, 115.1 tok/s on gfx1151, and 203.9 tok/s on
gfx1201 at TG128. Clean eight-turn serving runs average 191.0, 92.2, and 169.5
tok/s respectively; final turns remain at 160.3 tok/s at 18.2K, 82.5 tok/s at
21.3K, and 146.7 tok/s at 22.2K. The gfx1201 campaign raised the path from
approximately 110 to 203.9 tok/s without speculative decoding or manual clock
pinning.

Two model families join the registry. Qwen 3.8 27B is a dense arch-5 text
tower with native 262K context and an xhigh thinking default, shipping as a
quality trunk (`qwen3.8:27b`, MQ4) plus a cheaper MQ4R speed SKU
(`qwen3.8:27b-fast`). Muse Glimmer 30B is a new architecture (arch 14): a
dense text tower with a perception encoder and a 3:1 sliding/full attention
layout where the full layers are NoPE, shipping as `muse-glimmer` (MQ4 body,
Q8 attention and lm_head), `muse-glimmer:fast` (MQ4R), and a block-diffusion
DFlash drafter (`muse-glimmer:draft`, arch 23) that reuses the target's embed
and lm_head and pairs with either SKU. Glimmer's `.mq4r` SKU is not lowered to
Redline PM4 yet, so automatic Redline admission is withheld for it.

Single-GPU Qwen A3B `.mq4r` models now request retained PM4 by default on
gfx1100, gfx1151, and gfx1201. gfx1200 and other architectures remain opt-in;
the built-in `hip` configuration profile disables the automatic Redline route.

gfx1151 QKVZA uses the hybrid header-load kernel for the K=2048/total_m=1281
production shape and the all-buffer kernel for its production shape. The Silver
baseline repairs the hybrid path's coefficient-domain numerics and certifies
coherent retained-PM4 output. Its 114.209 tok/s high-water is 0.812 tok/s
(0.71%, approximately 1 tok/s) below the 115.021 tok/s Golden floor. ROCm 7.14
is only a hypothesis for that gap; neither causality nor absence of a Redline
route regression is proven.

Contributor deltas staged for this release:

- #465: LLaMA Site A attention dispatch and expanded HFQ KV policy.
- #466: Qwen2 instruct chat-template application.
- #468: dots.ocr text-only daemon generation.
- #473: VibeThinker-3B MQ4 and MQ6 registry entries.
- #476: DeepSeek V4 gfx1151 i8-WMMA prefill.
- #479: MiniMax gfx1151 grouped/dense prefill kernels and fail-closed guards.
- #480: MiniMax projection routing through `execute_steps`.
- #482: fused QKV bias across per-row dtypes, isolated from Redline's ABI.
- #487: architecture-generic MMQ screening.
- #496: default-off RDNA3 QKVZA split-tail experiment.
- #497: multi-stage runtime and GPU gate-runner Containerfile.
- #501: DSpark request telemetry and non-fatal pre-warm recovery.
- #513: native Qwen XML tool calls across CLI, daemon, and cached history.
- #528: DeepSeek V4 DSpark sidecar registration and re-pull discovery.
- #529: quickstart refresh and historical benchmark labeling.

The user-facing control plane is now Rust-only. `hipfire-cli`,
`hipfire-config`, `hipfire-registry`, `hipfire-client`, and `hipfire-tui` own
install, config, model lifecycle, chat, serving, and OpenAI-compatible HTTP;
the installed binary and runtime images carry no Bun, Node.js, or TypeScript
payload. Sparse typed TOML is the persistent policy surface, while environment
variables remain for bootstrap, one-shot compatibility, and explicit developer
experiments.

Named `default`, `dev`, `hip`, and `redline` configuration profiles now have a
clean `hipfire config profile set|create` control surface. The `hip` profile is
an explicit automatic-Redline opt-out. Bare `hipfire config profile` opens a
profile wizard that can select or snapshot profiles and search the complete
typed variable catalog by key, compatibility name, environment variable,
scope, category, or help text.

Registry `recommended_settings` now lower through the typed config resolver,
including temperature, top-p, top-k, min-p, presence penalty, repeat penalty,
and fallback system prompts. The Rust client also restores chat-history
framing, stable OpenAI response metadata, base64 vision passthrough,
cached-token accounting, and DeepSeek cache-fingerprint normalization. PFlash,
TriAttention sidecar auto-attach, and CASK m-folding remain disabled by default.

The release also refreshes the Rust dependency surface, adds standard
`clap`/`safetensors`/`half`/`tracing` support, introduces Redline property
coverage, enables `cargo-deny` in CI, removes the stale modular-rebase helper,
and removes the dead AWQ router exclusion.

## v0.2.1 — Dispatch unification (#397)

The centralized kernel-dispatch program lands: every GEMV/GEMM/attention/
MoE/rotation/fused projection across qwen35, llama, qwen2, dots-ocr,
deepseek4, minimax and lfm2moe now resolves through typed kernel families
with per-(arch x dtype) tables — a new dense quant is a table entry plus a
kernel file, no model code. The lowered forward-as-pipeline decode is
default-on for all seven arches (qwen2/dots-ocr byte-parity validated on
gfx1100 + gfx1201). KvTierPlan unifies KV-write/flash-attend resolution
with adaptive-KV exemptions; GPU-free coverage gates assert dispatch plans
for the entire fleet including RDNA4 rows.

Also in this release: gfx12 F16 GEMV fix (broken gemm_f16_tiled fallback
rewrote ds4 DSA compressor garbage on RDNA4 — restored byte-parity with
RDNA3.5 and ds4 EP tool-calling); gemm_f16_tiled rewritten (correct + up
to 13.8x faster); jinja chat templates default-on; DSML render/parse on
the EP serve path; q8 error-feedback DeltaNet state default; master line
fully merged back (hunt-3 fixes, FP32/Q4 DeltaNet spec-decode, MQ6 DFlash,
F32 oracle passthrough quantizer mode).


## v0.2.0 — DeepSeek V4 Flash + tokenizer diagnostics

DeepSeek V4 Flash is now a first-class hipfire architecture (`arch_id=9`).
The new `hipfire-arch-deepseek4` crate wires the production
`deepseek-v4-flash.mq2lloyd` path through the canonical daemon and CLI
surface: `hipfire-quantize` → `hipfire serve` → `hipfire run`, with no Python
in the hot path.

### DeepSeek V4 Flash

- New architecture crate: config parsing, weight loading, runtime state,
  prefill/decode, and MTP speculative decode live under
  `crates/hipfire-arch-deepseek4/`.
- New DeepSeek V4 kernel surface: sliding-window attention, compressed-KV
  indexer, Hyper-Connections, compressor/indexer projections, MQ2-Lloyd MoE
  GEMVs, tail-only YaRN RoPE, and glue kernels.
- New quantizer formats for DeepSeek V4 source / Q8 / Q8+MTP packaging, plus
  `hfq_split` for moving `mtp.0.*` tensors into an optional sidecar so normal
  decode does not upload the extra MTP layer.
- Daemon support for `arch_id=9` includes batched prefill, deterministic
  routing defaults, plain decode, and MTP speculative decode controlled by
  `HIPFIRE_DEEPSEEK4_SPEC_DECODE`, `HIPFIRE_DEEPSEEK4_SPEC_K`, and
  `HIPFIRE_DEEPSEEK4_MTP_ADDON`.
- Tool-call output works on both plain decode and the DeepSeek MTP path: the
  daemon emits `tool_calls` events with `finish_reason: "tool_calls"` instead
  of leaking raw DSML/tool-call text.

Validated on gfx1151 / Radeon 8060S with `HIP_VISIBLE_DEVICES=1`:

- `cargo test -p hipfire-arch-deepseek4 --lib`
- `cargo check -p hipfire-arch-deepseek4 --examples`
- `cargo check --workspace --examples`
- `./scripts/coherence-gate.sh --full`
- `./scripts/coherence-gate-deepseek4-mtp.sh --full`

### Tokenizer interned symbols + loud OOV at construction

`Tokenizer::from_*` constructors now return `Result<Self, TokenizerError>`
instead of `Option<Self>`. Inconsistent vocab/merges pairs (e.g. truncated
quantizer output, vocab missing a byte char, merges referencing absent
symbols) are now rejected loudly at load time with a specific error variant
instead of silently producing a `Tokenizer` whose `encode_gpt2_bpe` would
emit id 0 for OOV symbols downstream (#203).

The internal merge representation is now token-id-keyed end-to-end. The GPT-2
BPE encoder operates on `Vec<u32>` (was `Vec<String>`); merge-pair lookups
use `HashMap<(u32, u32), u32>` (was `HashMap<(String, String), usize>`).
Heap-loop String clones are eliminated. For a Qwen3-class vocab this saves
roughly 13 MB per loaded tokenizer.

### Public API change

```
Old                                          → New
Tokenizer::from_gguf(...) -> Option<Self>    → Result<Self, TokenizerError>
Tokenizer::from_hf_json(...) -> Option<Self>  → Result<Self, TokenizerError>
Tokenizer::from_hfq_metadata(...)            → Result<Self, TokenizerError>
Tokenizer::from_gguf_meta_json(...)          → Result<Self, TokenizerError>
Speculative::load_tokenizer(&self) -> Option<Tokenizer>
                                             → Result<Tokenizer, TokenizerError>
```

New public types in `hipfire_runtime::tokenizer`:

- `TokenizerError` — variants: `MetadataMissing { field }`, `MalformedJson`,
  `MissingByteSymbol { byte, char }`, `MissingMergeOperand { rank, left,
  right, missing_side }`, `MissingMergeResult { rank, expected }`. Implements
  `Display`, `std::error::Error`, `From<serde_json::Error>`.
- `Side` — `Left | Right`, used by `MissingMergeOperand`.

### Migration guide for contributors

```
Old caller pattern                           → New caller pattern
.expect("...")                                (unchanged — works on Result)
.unwrap()                                     (unchanged — works on Result)
.ok_or(_)?  / .ok_or_else(|| _)?              .map_err(|e| ...: {e})?
                                              or .map_err(|_| _)?
if let Some(t) = ...from_*(...)               if let Ok(t) = ...from_*(...)
.unwrap_or_else(|| ...)                       .unwrap_or_else(|_| ...)
                                              (closure now receives error)
```

### Why

Pre-existing `Option`-returning constructors made every failure mode look
identical from the outside. A user whose model failed to load got `None`
with no information about why. With many possible failure modes (corrupt
JSON, missing metadata, vocab/merges drift after re-quantization), this
made remote debugging painful. The new `Result` variants describe exactly
what's wrong (which byte, which merge rank, which symbol), so a single log
line is enough to diagnose.

The OOV consistency check at construction is the structural fix for #203.
The encoder no longer needs to silently fall back to id 0 in its final walk
because the constructor guarantees no OOV symbol can survive merges.

### Caveats

- The SentencePiece encoder's single-char fallback at `best_len == 0`
  still silently drops missing chars. That's a separate failure mode that
  needs an `encode_strict` variant returning `Result<Vec<u32>, EncodeError>`;
  deferred to a follow-up PR.
- The `max_token_chars` cap on the SentencePiece greedy scan (bounding the
  pre-existing O(N²) tail on unmatchable inputs) is also deferred.

## v0.1.20 — engine modularization

The `crates/engine/` monolith is split into a runtime crate plus
per-arch crates. The new layout is contributor-facing: each arch is
its own crate that implements the `Architecture` trait declared in
`hipfire-runtime`, so adding a new model family is a localized change
instead of a 5K-line edit to `engine/src/`. Behavior is unchanged for
end users — the daemon, CLI, and kernel surface are byte-identical.

### Migration guide for contributors

Grep'ing for old paths? Map:

```
Old path                                  → New path
crates/engine/src/lib.rs                  → crates/hipfire-runtime/src/lib.rs
crates/engine/src/qwen35.rs               → crates/hipfire-arch-qwen35/src/qwen35.rs
crates/engine/src/qwen35_vl.rs            → crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs
crates/engine/src/image.rs                → crates/hipfire-arch-qwen35-vl/src/image.rs
crates/engine/src/llama.rs                → crates/hipfire-runtime/src/llama.rs (facade-stage; PR 14 physically splits)
crates/engine/src/speculative.rs          → crates/hipfire-arch-qwen35/src/speculative.rs
crates/engine/src/pflash.rs               → crates/hipfire-arch-qwen35/src/pflash.rs
crates/engine/src/loop_guard.rs           → crates/hipfire-runtime/src/loop_guard.rs (NEW in PR 1)
crates/engine/src/sampler.rs              → crates/hipfire-runtime/src/sampler.rs (NEW in PR 3)
crates/engine/src/prompt_frame.rs         → crates/hipfire-runtime/src/prompt_frame.rs (NEW in PR 2)
crates/engine/src/eos_filter.rs           → crates/hipfire-runtime/src/eos_filter.rs (NEW in PR 4)
```

`engine` itself is gone; downstream consumers (`use engine::...`)
update to `use hipfire_runtime::...` (runtime symbols) or
`use hipfire_arch_qwen35::...` (arch-specific symbols).

### What this enables

- Compile-time isolation per arch: a change in `hipfire-arch-qwen35`
  doesn't trigger a recompile of the LLaMA forward path.
- Clean trait-based bring-up for new arches: implement
  `Architecture` in your own crate, register the `arch_id`, done.
  See `crates/hipfire-arch-toy/` for a copy-paste template.
- Forward-port path for the gemma branch: gemma4 lands as
  `crates/hipfire-arch-gemma4/` without re-touching the qwen35 code.
- Selective build for downstream library consumers: feature flags
  `arch-qwen35` / `arch-qwen35-vl` / `arch-llama` on
  `hipfire-runtime` (PR 12) trim the resolved dep graph to only the
  arches the consumer actually needs.
- Per-arch policy overrides without a daemon `match arch_id` ladder:
  `LoopGuardOverrides`, `SamplerOverrides`, `PromptFrameOverrides`,
  `EosFilterOverrides` are returned by the arch's trait impl.

### Known limitations

- `hipfire-arch-llama` is currently a facade. The LLaMA-family
  forward body (`forward_scratch*`, `forward_prefill_batch*`, etc.)
  still lives in `crates/hipfire-runtime/src/llama.rs` because the
  qwen35 hybrid path's pflash drafter reaches into shared transformer
  primitives (`KvCache`, `WeightTensor`, GEMV dispatch helpers,
  dequantizers) that haven't been extracted into a dedicated
  `runtime::transformer` sub-module yet. PR 14 (planned) will physically
  split the LLaMA-arch-only functions out once that extraction lands.
- In-tree binary feature-gating is limited by a cargo cycle.
  `hipfire-runtime`'s examples (`daemon`, `infer_qwen35`, etc.) consume
  the arch crates via `[dev-dependencies]`. Cargo's resolver follows
  dev-dep edges in both directions, so building an example with
  `--no-default-features` from inside the workspace re-activates the
  arch features unconditionally. Downstream library consumers (who
  don't enter the dev-dep cycle) respect `--no-default-features`
  faithfully. PR 12 documents the tradeoff in
  `crates/hipfire-runtime/Cargo.toml`.

### How to add a new arch

Copy `crates/hipfire-arch-toy/` as a starting template. The toy crate
is a minimum-viable `Architecture` trait impl with hardcoded stub
values and heavy explanatory comments — no real model logic. See
`CONTRIBUTING.md` "Crate topology" for the full decision tree.

## v0.1.9-alpha.1 (2026-05-02)

Patch release. Closes #111 — MQ4 single-token attractor on `<tool_call>`
that left agentic harnesses unable to dispatch tool calls on
`qwen3.6:27b.mq4` (and any other MQ4-quant Qwen3+ model with structured-
output drift). Two complementary defenses ship together:

- **Engine (daemon)** — new GPU-side `apply_unclosed_attractor_block`
  scans the recent decode window for unclosed `<tool_call>` openers
  (`opens − closes`); when depth ≥ 2, writes a single 4-byte `-INF`
  to the logits buffer at the opener's token offset before the next
  `gpu.sample_top_p`. Same gate for `<think>` to head off the
  thinking-mode boundary corruption the same reporter saw. Cost is
  zero when not tripped, ~5 µs when tripped — no D2H, no kernel
  change. The `(opens − closes)` invariant means legitimate multi-
  tool turns never trip: a complete `<tool_call>...</tool_call>`
  decrements depth before the next opener arrives.

- **CLI (parser)** — `parseToolCalls` now strips any leading
  `<tool_call>\s*` repeats from a captured block before JSON parse.
  Defense-in-depth in case a nested opener does slip through (the
  engine block fires before the third opener, but the second still
  ships in the visible stream).

Verified on hardware (gfx1100 / 7900 XTX / ROCm 7.2 / qwen3.6:27b.mq4)
against the two prompts the reporter posted as still-broken after the
v0.1.9-alpha defensive parser ship:

```
Prompt 1: "what files are in this directory?"
→ tool_calls: [{ name: "bash", arguments: { command: "ls -la" } }]

Prompt 2: "write a file here named test.md with the text test inside"
→ tool_calls: [{ name: "write", arguments: { path: "test.md", content: "test" } }]
```

Both `finish_reason: "tool_calls"`. No attractor loop, no nested
openers, no parser repair signal on stderr. The model emits clean spec
JSON now that the gate is in place.

Tests: 10 Rust unit tests (`llama::tests`) covering threshold edges,
window scope, complete-pair-allow, depth-saturate-at-zero. 14 Bun tests
(`cli/parse_tool_calls.test.ts`) including 4 new for nested-opener
strip. Coherence-gate green on 6/6 prompts including the existing
tool-call coverage test.

Still a stopgap on the symptom. The underlying root cause is MQ4
calibration drift on structured-output token positions; Path C
calibration retrain (#39) is the proper fix.

## v0.1.9-alpha (2026-05-02)

Headline: **MQ3 is production-ready.** The sub-4-bit Magnum Quant from
v0.1.8-alpha is now a full first-class citizen alongside MQ4 — K4-unrolled
decode GEMV, WMMA prefill family, DFlash cross-quant matrix, gfx12 port.
27B MQ3 fits 128K context in 24 GB where MQ4 OOMs at ~115K. Plus six
contributor PRs land in the same cycle, two arch bring-ups, and a sweep
of cache-lifecycle and parser hardening in response to user reports.

### Highlights — MQ3 production push

- **K4-unrolled MQ3 decode GEMV + fused residual** (gfx1100). 9B MQ3
  decode 114 → 141 tok/s (+24%); 4B and 0.8B see proportional wins. Same
  pattern as the v0.1.8 K4 unroll on HFQ4: 4 weight reads + 4 X reads
  hoisted, 4 dequant + accumulate pairs in the body. Kernel matches MQ4
  decode within 2% on every size despite the 104 vs 136 B/group.
- **WMMA prefill family for HFQ3** — `gemm_qkvza_hfq3g256_wmma`,
  `gemm_qkv_hfq3g256_wmma`, `gemm_gate_up_hfq3g256_wmma`,
  `gemm_hfq3g256_residual_wmma`. Closes the 17× prefill gap that gated
  ship: 9B MQ3 pp32 962 tok/s, pp128 1527 tok/s. Arch-gated to gfx11
  wave32 WMMA (`gfx1100/1101/1102/1150/1151`); gfx12 K4 variant landed in
  this same cycle.
- **gfx12 (RDNA4) MQ3 WMMA port** — full 4-kernel family ported to
  `_w32_gfx12` builtin with K4 unroll + half8_t lane-split matching the
  v0.1.8 HFQ4 work. gfx1201 baseline + speed-baselines committed.
- **DFlash + MQ3 cross-quant matrix.** MQ3-target ↔ MQ3-draft, MQ3-target
  ↔ MQ4-draft, MQ4-target ↔ MQ3-draft all validated end-to-end on
  gfx1100. Refusal logic narrowed: MoE/A3B + MQ3 still refused (no MoE
  branched WMMA path); dense MQ3 ships. CLI auto-discovery prefers
  `dirname(target)` first then mq3↔mq4 cross-quant fallback dirs.
- **27B MQ3 context-fit data** — fits 128K context in 24 GB on gfx1100
  with `asym3` KV (10.7 GiB weights vs MQ4's 13.4 GiB). The 2.7 GiB
  saved on weights is the difference between fitting 128K and OOMing
  at 112K.

### Highlights — contributor PRs

- **PR #118 — Per-weight MMQ auto-dispatch** (@fivetide). HFQ4 prefill
  routes to the MMQ i8-WMMA path automatically when `batch_size ≥ 256`
  and arch supports it (`gfx1100/1101/1102/1103/1150/1151/1152`). 9B
  pp512 +27% (1672 → 2122 tok/s). Default-on; opt out with
  `HIPFIRE_MMQ=off`. Tri-state config (`off`/`on`/`auto`) exposed in
  hipfire-tui.
- **PR #117 — Windows `serve -d` parity** (@fivetide). `hipfire serve`
  daemon mode wired through PowerShell on Windows; matches the Linux
  `--detach` UX. Includes `compile-kernels.ps1` (PowerShell port of
  `compile-kernels.sh`), `install.ps1` parity with daemon precompile,
  and `hipcc.exe`-first preference for paths with spaces.
- **PR #103 — Raw-filename → registry-tag** (@fivetide). `hipfire run
  qwen3.5-9b.mq4 "..."` now resolves the per-model overrides
  (`max_think_tokens`, `temp`, etc.) the same way as the registry-tag
  form. Prior behavior silently fell back to global defaults.
- **PR #91 — gfx12 WMMA K4 K-tile unroll** (@RobinVanCauter). 7
  `.gfx12.hip` kernels rewritten with `kt += 4` inner loop;
  `tests/speed-baselines/gfx1201.txt` refreshed; closes #65. Includes
  `bench-cold.sh` for cold-process N-run distribution capture.
- **PR #93 — gfx906 / Vega 20 / MI50 bring-up** (@myaple). Family-141
  rev-detect splits gfx906 from gfx900 in `redline::device`; wave64
  dispatch list extended to gfx906; `compile-kernels.sh` skips
  WMMA/dot8 on gfx906; HSA_OVERRIDE + rocminfo fallbacks added. 196/196
  kernel compiles + 16/16 channel-tests on local MI50.
- **PR #109 — MQ2 refuse + MQ3 advisory + sweep harness** (mine, Codex
  follow-ups). MQ2 refused-by-default (severe quality cliff confirmed);
  MQ3 emits an advisory on sub-9B; sweep harness reproducibility per
  CLAUDE.md (committed prompts as files with md5 manifest, scratch off
  /tmp).

### Engine

- **Cache-invalidation lifecycle** (Codex stop-time follow-ups). Three
  cross-cutting issues fixed:
  - `Gpu::invalidate_weight_caches()` clears `mmq_screen_cache` and
    drains `fp16_shadow_cache` on `unload_model`. Previous behavior left
    pointer-keyed cache hits on freed buffers — silent corruption on
    next model load if HIP reused the address.
  - `Gpu::invalidate_graph_state()` calls `graph_destroy +
    verify_graph_destroy_all + replay_graph_destroy_all` on unload.
    Captured hipGraphs over freed weight tensors would replay against
    garbage on the next `forward_scratch_warmed_up` call.
  - `graph_destroy()` resets `ar_forward_warmed_up = false`. Without
    this, the next `forward_scratch` would skip the warmup path and try
    to replay a destroyed graph.
- **Defensive `parseToolCalls`** (#111 stopgap). Three known
  malformations now repaired before the OpenAI shape returns: spec form,
  flat form, and XML-tag corruption. Token-attractor root cause
  (calibration retrain) deferred to a follow-up release.
- **Daemon UX hardening**. `Gpu::init()` failures convert from
  `expect()` panic to a friendly platform-specific checklist via
  `report_gpu_init_failure()` + `exit(1)`. Bun stack on the CLI side
  also caught: `Engine.recv()` cleanly `process.exit(code)`s when the
  daemon early-exits, instead of throwing through a stack trace.
- **gfx1152 / Strix Halo APU arch gating**. Added to all RDNA 3.5
  dispatch lists. Does not yet fix #50 (segfault on `--precompile`);
  awaiting reporter backtrace.

### Tooling

- **`scripts/speed-gate.sh` DPM warmup** (this release). `bench_run`
  now sets `HIPFIRE_DPM_WARMUP_SECS=3` so `pp32` measurements are
  reproducible regardless of GPU thermal state. Cold-DPM penalty was
  ~16% on the 32-token prefill probe; baseline 1240 was implicitly
  warm-captured and unreproducible across fresh-process runs without
  this fix.

### Known caveats

- **MQ3 collapses on sub-9B models**. 0.8B and 4B in MQ3 are advisory
  only — they parse and dispatch but quality drops below MQ4 by a wide
  margin on real prompts. Matches QuIP# / sub-4-bit literature.
- **MQ2 is refused by default**. The quantizer requires
  `--format mq2 --i-know-this-is-broken` to opt in. Lloyd-Max MQ2
  (qt=19) and Lloyd-Max MQ3 (qt=20) are the path forward; spike
  shipped this cycle, full PRs to follow.
- **MQ3 + MoE / A3B is unsupported**. The MQ3 batched path lacks an
  MoE-branched WMMA kernel; daemon refuses MQ3 weights inside
  DeltaNetMoe / FullAttnMoe layers at load time.
- **#111 token-attractor unresolved**. The parser stopgap masks
  symptoms; calibration retrain is the real fix and lands in a
  follow-up release.
- **#50 (gfx1152 segfault)** still open pending reporter data.
- **#119 (ROCm 7.2 / clang 22 regression on Strix Halo)** filed this
  cycle; no engine-side fix yet.

### Upgrade

```bash
hipfire update                      # if installed via curl-bash
# or
git pull && cargo install --path crates/engine
```

Windows: re-run `install.ps1` — daemon.exe + kernel blobs refresh
automatically.

## v0.1.8-alpha.2 (2026-04-27)

Released the same day as alpha.1 — a second cycle's worth of contributor PRs and
infrastructure hardening landed too quickly to bundle. Five PRs merged
(#71, #72, #73, #74, #75), three Codex-flagged config-path hardening passes on
the just-merged DDTree wire-up, plus a stale-path bugfix that was blocking 27B
DFlash measurement in the speed-gate.

### Highlights

- **gfx12 WMMA dispatch is now feature-complete** for HFQ4 prefill on RDNA4.
  PR #62 wired the qkv / qkvza / gate_up scaffolds; PR #71 (@RobinVanCauter) closes
  the last un-ported GEMM hot path: `gemm_hfq4g256_residual_wmma_gfx12`. The
  residual kernel was ~42% of 9B prefill GEMM time on the dot2 fallback. R9700
  numbers vs master: 9B prefill +29.7%/+31.3% (pp32/pp128), 27B prefill
  +27.8%/+42.4%, 4B prefill +26.0%/+25.5%. Decode unaffected. New
  `tests/speed-baselines/gfx1201.txt` floor committed.
- **DDTree wire-up + Path C PRD** (PR #72, @flamme-demon). DDTree was implemented
  in `speculative.rs` but never reachable from production. Now opt-in via
  `HIPFIRE_DDTREE_BUDGET=<n>`; default decode path bit-exact preserved. The PR
  also ships `select_main_path()` + 6 unit tests in `ddtree.rs` (the first brick
  of Path C) and a 382-line PRD documenting the main-path-first orchestrator
  pattern as a follow-up to Path A revert (`ecbc49d`) and Path B dead-end
  (`39aa358`). Local validation on 7900 XTX with Qwen3.6-27B + DFlash MQ4 draft:
  12/12 attractor-clean across the `path-c-smoke.sh --full` battery, on the
  exact target/draft pair where Paths A/B1 single-token-attractor failed. Path C
  avoids the linearization-slot RoPE phase skew by construction (heap-pop
  ordering invariant). See `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
- **gfx908 / MI100 CDNA1 bring-up** (PR #67, @linus-amg). 2× MI100 hardware
  validation. Wave64 dispatch added to 4 fused projection sites + a new
  `fused_gate_up_hfq4g256_wave64` kernel (HFQ4 gate+up FFN GEMV fused; same
  shape as `fused_qkv_hfq4g256_wave64`, no MFMA). Cross-process verified
  across 5 fresh-process runs per metric: 9B decode +9.3% (64.4 → 70.3 tok/s),
  4B decode +4.9%, A3B MoE decode +11.0% (86.6 → 96.1). Also extracted
  `has_wave64_native(arch)` predicate replacing 7 inline `matches!()` —
  gfx90a (CDNA2) is now a one-line addition once it has hardware to validate.
  New `tests/speed-baselines/gfx908.txt`. Two negative results documented in
  the PR body so the next CDNA1 contributor doesn't re-burn the hours
  (`v_dot2_f32_f16` regresses prefill on gfx908 despite the instruction
  existing; wave64 batched GEMM loses to fp16-packed at small batch).
- **Opt-in HFQ4 MMQ prefill path** (PR #73, @KotDath). Q8_1 activation
  pre-quantize + i8 WMMA over 128×128 output/batch tiles, similar in shape to
  llama.cpp's AMD MMQ prompt-processing path. Gated behind `HIPFIRE_MMQ=1`,
  architecture-gated to gfx1100/1101/1102/1103/1150/1151. Targets the
  Strix Halo prefill gap vs llama.cpp (#60) where the author measured the
  largest wins; on gfx1100, +19.8% on 4B pp256 once the per-batch quantize
  amortizes (small batch is dominated by quantize overhead and is not a
  target workload). Default behavior unchanged — gate verified bit-exact on
  master baseline.

### API

- **`/v1/chat/completions` thinking-mode fix** (#74, fae2867). Per-model
  `max_think_tokens` was silently dropped on the OpenAI-compatible API path,
  so models with `thinking: "on"` could consume the entire `max_tokens`
  budget inside a single `<think>...</think>` block; the downstream strip
  then left `message.content` empty while `completion_tokens` reported the
  full burn. Reproducer in #74. Also fixed `prompt_tokens: 0` hardcode —
  `total_tokens` now correctly reports `prompt + completion`.

### CLI

- **`hipfire list` shows `.mq6` models** (PR #75, @Nereuxofficial, 8c352d7).
  `listLocal()` was missing `.mq6` from its discovery filter. One-line fix.

### Speculative decode hardening

- **HFQ6 WMMA graph-capture safety** (83358c6). All 6 HFQ6 WMMA wrappers
  (3 gfx11 + 3 gfx12) used raw `unsafe self.hip.launch_kernel` instead of
  `launch_maybe_blob` — same bug class as the hipGraph dangling-kernarg
  story (#19, `project_hipgraph_moe_investigation`). Pre-fix: latent on gfx11
  (HIPFIRE_GRAPH=1 + MQ6 + prefill is a niche combination; speed-gate uses
  `.mq4` so the bug was dormant). PR #62 routed gfx12 HFQ6 to the same
  broken wrappers; Codex stop-time review caught it before any user hit it.
  Migrated all 6 to `launch_maybe_blob` with proper `KernargBlob` builders.
- **DDTree daemon config hardening** (0931afb, ce36dc8, c8ba1c1). Three
  Codex-flagged crashable env-var paths in the just-merged DDTree wire-up:
  - `HIPFIRE_DDTREE_BUDGET` had no upper bound. `DdtreeScratch::attn_bias`
    is `max_n²`; budget=10000 silently allocated 400 MB, budget=100000
    OOMed. Capped at 256 (paper Algorithm 1 typically uses ≤22).
  - `HIPFIRE_DDTREE_TOPK` was clamped to vocab_size (152064 on Qwen3.6).
    The active kernel `run_dflash_draft_for_topk_gpu` asserts `k <= 8`
    (speculative.rs:3302); my first cap of 32 still let values 9-32 panic
    the kernel. Re-capped at `min(8, vocab_size)`. Default scales to
    tiny-vocab models too: `min(4, vocab_size.max(1))`.
  - `HIPFIRE_DDTREE_PATH_C` was re-read inside the per-spec-cycle decode
    loop (microseconds of waste on the hot path) AND silently accepted
    invalid values like `"phase3"`. Hoisted out, eagerly validated, warns
    once on bad input.
  All three replace silent OOM / silent fallback with clear stderr lines.

### Tooling

- **`scripts/speed-gate.sh` 27B DFlash draft path fix** (#61, a3b8f11).
  Reported by @m0n5t3r as MISSING_DRAFT despite the file being downloaded.
  Root cause: gate hardcoded `qwen35-27b-dflash.mq4` (legacy basename +
  extension); registry standardized on `qwen35-27b-dflash-mq4.hfq` when the
  `<base>-<quant>.hfq` convention landed. Gate now accepts both names.
  9B path was already correct so it's untouched. Fixes 27B DFlash anchor
  rows in any future `--update-baselines` capture across all arches.

### Issues filed for follow-up

- **#65** — gfx12 WMMA: tune 9B prefill (multi-row, K-tile, s_prefetch,
  launch_bounds). RDNA4 follow-up to PR #71. Each lever is a discrete
  experiment. Hardware: R9700 / 9070 XT.
- **#70** — gfx908 / MI100 CDNA1: port MFMA prefill kernels (4 kernels +
  channel-tests). Closes ~35× prefill gap vs gfx1100. Toolchain validated;
  per-PR-#56 channel-test discipline applies.
- **#41** — DDTree on gfx1100 RoPE phase-skew. Superseded by PR #72's Path C
  orchestrator (different mechanism, attractor-clean). Closing in 7 days
  unless reopened.

### Upgrade

```bash
hipfire update                      # if installed via curl-bash
# or
git pull && cargo install --path crates/engine
```

Windows: re-run `install.ps1` — the dynamic-release-query (#69) will pull the
fresh `daemon.exe` automatically; asset-id cache stamp prevents a stale binary
from being preserved.

### Known issues

- **#60 reporter (@h2252) hit a `--gen 0` panic** on `bench_qwen35_mq4` pre-alpha.2.
  Cannot reproduce on master; many of today's commits could have addressed it
  incidentally. If you hit this on alpha.2, please retry with
  `RUST_BACKTRACE=1` and post the trace on #60.
- **#68 (Windows + qwen3.6:27b VL trace)** — root-caused to the
  v0.1.0-alpha-pinned daemon.exe; alpha.1's fresh binary should resolve.
  Awaiting reporter confirmation.
- **#50 (gfx1152 / Strix Halo APU segfault)** — different SKU than the
  gfx1151 work that landed today. Awaiting bt + dmesg + cache-clean repro
  from reporter.

## v0.1.8-alpha.1 (2026-04-27)

Point release rolling up the post-v0.1.8-alpha work. Two contributor PRs land in
this cycle (gfx1201 RDNA4 WMMA port from @RobinVanCauter, gfx1151 Strix Halo
autodetect from @KotDath), plus a feature on the input side (GGUF →
HFQ4/MQ4 conversion) and a docs nuke + rewrite that swaps a 39-file legacy
tree for 10 canonical pages.

### Highlights

- **RDNA4 / 9070 XT unblock end-to-end** (#54). gfx1201 WMMA codegen crash
  resolved via dispatch fallback (`6e100c2`); first canonical gfx12 WMMA
  scaffold (`6924f2a`) with C-output mapping hypothesis derived from the
  CK trait swap; full validated 5-kernel + 6-channel-test contributor port
  (PR #56) hardware-tested on R9700 silicon. C-mapping
  `acc[j] = C[8*(tid>>4) + j][tid & 15]` validated, propagated across the
  family. Public dispatch still routes gfx12 through dot2 fallback pending
  perf measurement (#57); the WMMA methods on `Gpu` are exposed for
  channel-tests now and ready to flip when numbers land.
- **gfx1151 / Strix Halo autodetect fix** (PR #59, @KotDath). KFD
  `gfx_target_version 110501` was decoding to `gfx11051` instead of
  `gfx1151`; fixed via explicit known-version table in `cli/index.ts`
  + `scripts/install.sh`. Same refactor incidentally fixes a latent
  same-class bug for any arch with non-zero step bytes
  (`100302 → gfx1003` was equally wrong before this PR). Hardware-validated
  on Ryzen AI Max+ 395 / Radeon 8060S; speed-baseline contribution welcome
  at #61. Issue #50 (gfx1152, separate Strix Halo SKU) untouched —
  detection now correctly resolves the arch, but the engine-side segfault
  reported there still needs reproduction info.
- **GGUF → HFQ4 / MQ4 import**. New `hipfire quantize <file.gguf>` mode
  accepts any GGUF the engine can load (`Q4_K_M` / `Q8_0` / `Q4_0` / `Q6_K`
  / `F16` / `BF16` / `F32` source quantizations) and re-quantizes to
  hipfire's native HFQ4-G256 (default for dense Llama / Mistral / older
  Qwen) or MQ4-G256 (FWHT-rotated, opt-in for Qwen 3.5+ family). Tensor
  names are translated GGUF → safetensors style at write time so the
  engine's existing `load_weights_hfq` consumes the output unchanged;
  the GGUF tokenizer is preserved verbatim under `meta.gguf_meta` and
  `Tokenizer::from_hfq_metadata` reads it directly (no GGUF-on-disk
  fallback). End-to-end UX:
  ```bash
  hipfire quantize ./tinyllama.Q4_K_M.gguf --install --register tinyllama:1b-gguf
  hipfire run tinyllama:1b-gguf "..."
  ```
  Quality is lower than quantizing from full-precision safetensors (it's a
  double-quant roundtrip — raise to `--format hf6` or `--format mq6` if
  you have the disk space). Format defaults are dense-aware: HF4 for GGUF
  input, MQ4 for safetensors directories.
- **`dflash_mode` default flipped to `off`** (was `auto`). DFlash is now
  opt-in: `hipfire config set dflash_mode auto` re-enables the genre-
  conditional auto-routing (DFlash on for dense Qwen 3.5+ targets, off
  for A3B without a TriAttention sidecar). Bare `hipfire run <target>`
  without that config flip stays pure AR even when a paired draft is on
  disk; the daemon logs `[hipfire] DFlash disabled (dflash_mode=off)` so
  it's not a silent footgun. Background: per-genre measurements show
  DFlash a clear win on code, modest on instruct, and a net loss on
  long-form prose. Default-on overpromised; default-off + opt-in matches
  the actual win surface.
- **Docs rewrite + LICENSE**. The 39-file `docs/` tree (mix of canonical
  user docs and operational artifacts: agent prompts, daily standups,
  port plans, perf checkpoints) consolidated to 10 canonical pages —
  `GETTING_STARTED` / `CLI` / `MODELS` / `QUANTIZE` / `CONFIG` / `SERVE` /
  `BENCHMARKS` / `ARCHITECTURE` / `QUANTIZATION` /
  `methodology/perf-benchmarking`. README cut 371 → 89 lines; first-time
  visitors see the pitch + headline benchmark + install in 10 seconds
  rather than scrolling through the model catalog. New top-level `LICENSE`
  file (was missing despite the README and `Cargo.toml` declaring MIT).
- **New `hipfire-kernel-tuning` agent skill**. Sibling to the
  `hipfire-arch-port` skill from earlier this cycle. Codifies the
  empirical kernel-perf methodology from this repo's git log: 6-step
  workflow (measure → root-cause → pick lever → cross-arch verify →
  three gates → cross-process measure), levers catalog (multi-row,
  K-tile depth, wave64 port, `s_prefetch_data`, WMMA / MFMA, fused
  projections, ISA flags, rocBLAS fallback), cross-arch dispatch
  routing rules, and five worked case studies — wave64 CDNA3 port
  (+2× MI300X decode, `4105035`), nontemporal-load fake-win revert
  (-13% caught only by clean-baseline bisect, `34eb024`), k2x32 null
  result kept for posterity (`f670e16`), gfx11 WMMA C-mapping silent
  corruption (~6 weeks before catch, `b7ac66a`), and 27B DFlash perf
  recovery root-caused to a single newline character in a bench
  prompt (`9a2c667`).
- **Vision correctness** (#23 / PR #35). `load_and_preprocess` writes
  pixel bytes in R,B,G order so the upstream HuggingFace
  `patch_embed`-export channel transposition cancels at inference; full
  details below in "Fixed".
- **27B DFlash perf restored** (`9a2c667`, ~40% recovery). PR #32
  cleanup-dead-wmma-kernels removed `gemm_hfq4g256_residual_wmma{,2,_k4}.hip`
  thinking they were dead — they were on the K4 / WMMA dispatch path for
  27B verify-shape GEMMs. Per-cycle cost on 64-layer × B=16 verify forward
  was 57 → 100+ ms. Fix landed via revert + cherry-pick of the 8 master
  commits that did NOT introduce the regression. Empirical anchor: 27B-3.5
  LRU code DFlash @ max=120 = 199 tok/s τ=10.36 (was: 95 tok/s in
  pre-revert state).

### Added

- **`hipfire quantize <file.gguf>`** — see GGUF import in highlights.
  New `crates/hipfire-quantize/src/gguf_input.rs` (self-contained reader +
  dequant for Q4_0 / Q8_0 / Q4_K / Q6_K / F16 / BF16 / F32). New
  `Tokenizer::from_gguf_meta_json` engine-side path so converted files
  carry their own tokenizer metadata.
- **gfx12 (RDNA4) WMMA kernels**: 6 new `kernels/src/gemm_*_wmma.gfx12.hip`
  kernels with channel tests in `crates/engine/examples/test_wmma_*_gfx12.rs`.
  Compile-tested green on gfx1200 + gfx1201 via the family-tag override
  in `scripts/compile-kernels.sh`.
- **`scripts/_detect-gpu.sh`** — shared `rocminfo` + `amdgpu-arch` GPU
  detection helper. Three previously hardcoded "RX 5700 XT" bench banners
  now derive from `hipfire_gpu_banner`.
- **`hipfire-kernel-tuning` skill** + extension to the `hipfire-arch-port`
  skill (canonical kernel referenced, validated C-mapping documented).
- **`gfx_target_version` known-version table** (PR #59). Explicit
  Record<number, string> for 100100 / 100300 / 100302 / 110000 / 110001 /
  110501 / 120000 / 120001, with algorithmic fallback for unknown versions.
- **CONTRIBUTING.md rewrite**. 271 → 216 lines, tester path now genuinely
  uses installer-provided binaries (`hipfire diag` + `hipfire bench`), all
  four agent skills indexed.
- **Top-level `LICENSE` file** (MIT, copyright 2026 Kaden Schutt).

### Changed

- **`prompt_normalize` default ON** (was opt-in since v0.1.8-alpha).
  Engine collapses `\n{3,}` → `\n\n` at engine entry, lifting 27B-3.5 LRU
  DFlash by +24% (159 → 199 tok/s). Opt out via `HIPFIRE_NORMALIZE_PROMPT=0`
  or `prompt_normalize=false` config when raw `\n{3,}` whitespace is
  semantically load-bearing (rare). Zero correctness cost on Qwen3.5/3.6
  vocab — `\n\n\n` was a rare BPE token (rank 1102) getting in the way of
  the much hotter `\n\n` (rank 271).
- **`dflash_mode` default OFF** (was `auto`). See highlights.
- **GGUF input format default**: `--format hf4` (was implicitly `mq4`).
  MQ4's FWHT rotation is calibrated for Qwen 3.5+ training; on Llama-style
  dense models it adds runtime overhead with no quality benefit. Override
  with `--format mq4` for Qwen 3.5+ family GGUFs.
- **CLI test-kernels.sh + megabench-q35.sh + bench-matrix.sh** auto-detect
  arch + GPU name via `_detect-gpu.sh`; the previous hardcoded "RX 5700 XT
  / gfx1010" defaults that bled into bench reports are gone.
- **AGENTS.md** — DFlash default-off surfaced in §3.6 pull-flow recipe and
  added to the §6 pitfalls table; flag table corrected
  (`HIPFIRE_DFLASH_DRAFT` description now says "filename auto-match" not
  "auto-discover", with empty-string opt-out path documented).

### Fixed

- **gfx1201 WMMA codegen crash on first dispatch** (#54). Routes to dot2
  fallback until per-arch WMMA kernels land. The dispatch predicate
  `has_wmma_f16` now matches gfx11 only; the gfx12 WMMA path is exposed
  on `Gpu` for channel-tests and will flip via #57 when perf is measured
  on R9700.
- **gfx1151 (Strix Halo) autodetect** (PR #59). KFD `gfx_target_version
  110501` decoded to `gfx11051` instead of `gfx1151` due to a
  `padStart(2, '0')` + `replace(/^(gfx\d{4})0$/, '$1')` interaction in
  the version decoder. Same class of bug also affected `100302 →
  gfx1003` (should be `gfx1030`). Fixed via explicit known-version table
  + algorithmic fallback consolidation in `cli/index.ts` and
  `scripts/install.sh`.
- **27B DFlash perf** — see highlights.
- **#23 — VL model misidentifies green and blue objects.** Pure-color
  probing (red/green/blue PNGs, temp=0 greedy decoding) showed the
  vision encoder reading green pixels as blue and blue pixels as green
  while red came through correctly — a classic G↔B transposition. Root
  cause is most likely a channel permutation in the HuggingFace
  `patch_embed` weight export (input conv channels 1 and 2 appear
  transposed); the repair lives in preprocessing — `load_and_preprocess`
  now writes pixel bytes in R,B,G order so the two transpositions
  cancel. Regression test pins the contract at
  `crates/engine/tests/channel_order.rs`.
- **Vision weight upload shape encoding.** `qwen35_vl::load_f16_gpu`
  passed byte-length as the tensor shape on both the F16-direct and
  HFQ4-dequant paths, so downstream shape-aware dispatch saw a tensor
  shaped `[byte_count]` instead of `[element_count]`. Corrected to use
  the element count.
- **Quant-format visibility for vision weights.** The loader now logs
  the detected quant format (F16 / HFQ4-G256 / HFQ4-G128) for
  `model.visual.patch_embed.proj.weight` at load time so HFQ4 models
  can be distinguished from F16 models at a glance during debugging.
- **Dead kernel file cleanup.** Removed `gemm_f16_wmma_tiled.hip` and
  `vit_attention_flash.hip` — neither was referenced via `ensure_kernel`
  dispatch and both were stale copies superseded by the active
  `vit_attention` and `gemm` kernels.
- **CLI `quantize` `--format hf6` symmetry** — the safetensors path's
  `use_hfq6` flag now accepts the `hf6` short alias to match the GGUF
  path's `GgufFormat::from_flag`, eliminating a silent fall-through where
  `hipfire quantize <safetensors-dir> --format hf6` would silently
  downgrade to the q4k default.
- **Output extension on GGUF conversion** — was `.hfq4` / `.hfq6` (which
  the CLI's `resolveModelTag` / `list` / fuzzy lookup don't recognize),
  now `.hf4` / `.hf6` so converted files surface in `hipfire list` and
  resolve via tag aliases.
- **CONTRIBUTING tester path** — was claiming "no Rust required" but
  pointed at scripts that run `cargo build`. Tester path now uses
  installer-provided `hipfire diag` + `hipfire bench`.

### Documentation

- 10 canonical `docs/` pages replace the prior 39-file tree (archive at
  `~/hipfire-docs-archive-2026-04-27/`, history preserved by git).
- `.skills/hipfire-kernel-tuning/` — new agent skill (5 markdown files,
  893 lines total).
- `README.md` cut 371 → 89 lines.
- `CONTRIBUTING.md` rewritten end-to-end.
- `AGENTS.md` updated for default-off DFlash + flag table corrections.

### Issues filed for follow-up

- **#57** — gfx12 WMMA dispatch wiring + perf vs dot2 (R9700 / 9070 XT
  hardware-gated; PR #56 landed kernels but didn't flip dispatch).
- **#58** — multi-GPU support roadmap. Pipeline-parallel first cut design
  open for discussion.
- **#60** — prefill scaling regression vs llama.cpp at pp≥512 on 9B+.
  Diagnostic phase needs no kernel writing, anyone with a 7900 XTX can
  contribute the per-kernel `HIPFIRE_PROFILE=1` breakdown.
- **#61** — gfx1151 (Strix Halo) speed-baseline + perf bench. One-command
  bootstrap for any Strix Halo owner.

### Upgrade

```
hipfire update
```

No config migration. `~/.hipfire/config.json` from v0.1.8-alpha remains
compatible. If you were relying on default-on DFlash, re-enable
explicitly:

```
hipfire config set dflash_mode auto
```

---

## v0.1.7-alpha.2 (2026-04-18)

Hotfix release for three user-visible regressions in v0.1.7-alpha. No
behavior changes beyond the fixes listed — intended as a drop-in
replacement for anyone running v0.1.7-alpha.

### Fixes

- **`hipfire config` TUI crash** (`TypeError: undefined is not an object
  (evaluating 'meta[k].label')`). The v0.1.7-alpha release added 8 new
  config keys (`experimental_budget_alert`, `dflash_adaptive_b`,
  `cask_sidecar`, `cask`, `cask_budget`, `cask_beta`, `cask_core_frac`,
  `cask_fold_m`) to `CONFIG_DEFAULTS` without matching entries in the
  TUI's `meta` field descriptor table, so every interactive `hipfire
  config` invocation on a real TTY threw on first render. Non-interactive
  `hipfire config list|get|set` flows were unaffected. Added full meta
  entries + boolean option round-tripping in `cycleOption` / `commitEdit`.
- **A3B DFlash default-on perf regression** (2-5× slower than plain AR on
  code/prose). A3B drafts reject most tokens (τ≈1.0-1.5 outside math),
  and the spec cycle overhead dominates the AR win. New `dflash_mode`
  per-model config key: `on | off | auto`. `auto` keeps dense targets
  running DFlash as before and flips A3B off unless a `cask_sidecar` is
  configured (A3B long-context on 24 GB consumer cards needs eviction to
  fit). Daemon-side belt-and-suspenders: `dflash_mode=off` skips draft
  load outright even when a draft path is supplied.
- **`hipfire config set dflash_mode <value>` → "Unknown key"**. The
  dflash_mode key was not in the released alpha's validKeys list. Ships
  as part of the same commit as the default-off gate above.

### Upgrade path

```
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/install.sh | bash
# or: hipfire update
```

No config migration needed — `~/.hipfire/config.json` written by
v0.1.7-alpha remains compatible. If you want to explicitly disable
DFlash on A3B (defaults to auto-off now anyway), either edit config.json
or run:

```
hipfire config set dflash_mode off
hipfire config qwen3.5:35b-a3b set dflash_mode off   # per-model override
```

Full v0.1.7 stable release (rocBLAS MFMA on MI300X, hipGraph+MoE fix,
full Hermes agent validation) tracking on `dflash` branch.

## v0.1.7-alpha (2026-04-18)

Pre-release tag cutting the dflash branch against master. Gated to full
v0.1.7 on the outcome of the Hermes-agent + hipfire stack validation
currently running on MI300X.

### Highlights

- **FlashTriAttn long-context wins shipped.** DFlash speculative decode +
  TriAttention KV eviction composes cleanly. Measured on 7900 XTX, 9B MQ4,
  ~1500-token prompt, 200-token decode, `--cask-budget 512 --cask-beta 128`:
  baseline 150 tok/s τ=5.31 → **FlashTriAttn 214 tok/s τ=5.36 (+42% speedup,
  τ unchanged)**. With 1M-token wikitext sidecars, τ no longer drops — earlier
  builds lost ~27% τ because the sidecar was under-calibrated.
- **CASK core-aware m-folding** merges non-core KV instead of dropping.
  Composes with FlashTriAttn. Still has a ~3% τ drop from merge smoothing —
  the GPU merge kernel (task #82) eliminates the CPU hop; full tok/s win
  lands in 0.1.7 stable.
- **Qwen3.5-35B-A3B and Qwen3.6-35B-A3B MoE** end-to-end in DFlash. Batched
  MoE prefill, fused sigmoid+residual GEMV, indexed expert dispatch. On
  7900 XTX A3B decodes at ~115 tok/s (single turn) / 96 tok/s (multi-turn).
- **MI300X (gfx942) wave64 port.** 10 hot HFQ4 kernels re-written for
  block=[64,1,1] 2-rows-per-block pattern. A3B decode 48.6 → **96 tok/s**
  on MI300X (matches 7900 XTX baseline despite the 4× memory bandwidth gap
  between consumer and datacenter silicon).
- **DFlash tape-replay rollback** lets multi-turn state recover from an
  incorrect verify without a full target re-run.
- **Batched-prefill TriAttention tap** (4.5–5× faster sidecar cals) — what
  made it possible to calibrate 1M-token sidecars across 5 targets on one
  MI300X overnight.

### Bench snapshot (7900 XTX, MQ4, branch @ `a306013`)

DFlash τ + tok/s per prompt class (ctx=4K, no CASK):

| model | short | code | math |
|-------|-------|------|------|
| 4B    | 53 tok/s τ=1.27 | 92 tok/s τ=2.49 | 148 tok/s τ=6.0 |
| 9B    | 112 tok/s τ=1.52 | **461 tok/s τ=9.95** | 288 tok/s τ=5.77 |
| 27B   | 20 tok/s τ=2.21 | 41 tok/s τ=5.66 | 42 tok/s τ=6.14 |

Sidecar reconstruction r̄ (1M wikitext tokens, default validation prompt):

| model | mean r̄ | % heads > 0.95 R |
|-------|---------|-----------------|
| 4B    | 0.564   | 5.7% |
| 9B    | 0.629   | 5.8% |
| 27B   | 0.542   | — |
| 3.5-A3B | 0.552 | — |
| 3.6-A3B | 0.552 | — |

Paper Figure 3 target is r̄ ≈ 0.5; we're above it on every model.

### CLI + daemon config (0.1.7-alpha knobs)

Per-model config (via `hipfire config` or `~/.hipfire/per_model_config.json`):

```
dflash_adaptive_b   boolean   default true     # τ-window trip-wire block shrink
dflash_mode         enum      default auto     # on | off | auto (A3B-aware)
cask_sidecar        string    default ""       # path to a .triattn.bin
cask                boolean   default false    # enable m-folding (on top of sidecar)
cask_budget         int       default 512
cask_beta           int       default 128
cask_core_frac      float     default 0.5
cask_fold_m         int       default 2
```

The daemon protocol accepts all of these in the `load` message's `params` object.
`cask_sidecar` is accepted and logged today; the generate-loop integration
lands in 0.1.7 stable (current serve users run DFlash without eviction —
use `dflash_spec_demo` directly for the `--cask-sidecar` path).

### Post-alpha fixes (land in v0.1.7 stable)

- **`dflash_mode` gate** — A3B DFlash silently routed every temp=0 request
  through DFlash in the alpha; a 7900 XTX sweep showed it's 2-5× slower than
  plain AR on code/prose (A3B draft rejects most drafted tokens — τ≈1.0-1.5
  — and the cycle overhead dwarfs the AR win). New per-model config key
  `dflash_mode: on | off | auto`. `auto` keeps dense-on, flips A3B off
  unless a `cask_sidecar` is configured (long-ctx A3B on 24 GB consumer
  cards needs eviction for correctness, and that combo wins on τ too).
  Daemon-side belt-and-suspenders: `dflash_mode=off` skips draft load even
  when a draft path is supplied. Also fixes the draft-discovery regex so
  A3B targets pick up `qwen3{N}-35b-a3b-dflash-*.hfq` under `on`/`auto+sidecar`.

### Pending for v0.1.7 stable

- Wire `cask_sidecar` + adaptive-B through the daemon's generate loop so
  `hipfire serve` honors it automatically.
- Hermes agent + hipfire stack validation on MI300X (task #125) — gates the
  stable release.
- GPU-side CASK merge kernel (task #82) to flip FlashCASK net-positive.
- DDTree integration into the CLI/daemon (currently τ-positive but not yet
  tok/s-positive without hipGraph coverage).

## v0.1.6 "deltacut" (2026-04-14)

Focus: **Qwen3.5-35B-A3B (MoE) support** end-to-end — quantizer, loader,
forward path, daemon wiring, and a stack of fused MoE kernels that take the
first-working-dense-compute path from 28 tok/s to 115 tok/s of production
decode throughput on gfx1100. Plus serve/install/bench polish.

### Qwen3.5-35B-A3B — first MoE model

35B total params / 3B activated per token. 256 experts, top-8 routing, plus
one always-on shared expert. Hybrid attention (30 DeltaNet + 10 FullAttn)
like the dense 9B, with A3B-specific shape differences: head_dim=256, 16 Q
heads / 2 KV heads, `partial_rotary_factor=0.25`, `attn_output_gate=true`.

- **Quantizer** (`hipfire-quantize`): recognizes `qwen3_5_moe` (arch id 6),
  splits the 3D-stacked `mlp.experts.{gate_up,down}_proj` tensors per-expert
  into 256 MQ4G256 blobs apiece. Rayon-parallelized across experts (80% of
  cores by default; override with `--threads N` or `HIPFIRE_QUANT_THREADS`).
  67 GB safetensors → 18.7 GB MQ4 in ~30 s.
- **Engine**: new `DeltaNetMoe` / `FullAttnMoe` `LayerWeights` variants,
  separate `SharedExpertWeights { gate, up, down }` struct (the loader was
  previously stashing `gate_proj` into the routed-expert fused slot and
  silently skipping `up_proj`), and a `moe_ffn_decode` hot path that routes
  through four new kernels (below).
- **Daemon / CLI**: `arch_id=6` dispatches through the same `qwen35` path
  as dense 5, with the loaded response reporting `arch: "qwen3_5_moe"`.
  Registry entry `qwen3.5:35b-a3b` is marked local-only (`repo: ""`) until
  the HF upload lands; `hipfire pull` short-circuits with a clear message
  instead of 404'ing.

### MoE fused-kernel stack (four new kernels)

Built up across nine incremental optimizations (each commit verified byte-
identical or byte-equivalent against the previous stage through the A3B
smoke test). Final routed-expert compute is **3 kernel launches per layer**,
down from 24 in the dense-compute reference.

- **`moe_softmax_topk_renorm_k8`** — single-workgroup GPU softmax + top-8
  selection + (optional) renormalization. Writes `[k]` indices and `[k]`
  weights to device buffers, eliminating the per-layer D2H sync the
  CPU-side top-K path needed.
- **`gemv_hfq4g256_moe_gate_up_k8_indexed`** — eight top-K experts' fused
  `gate_up` HFQ4-G256 GEMV in one launch. Reads expert IDs from a
  device-side `topk_indices` buffer; weight bases come from a per-layer
  `expert_gate_up_ptrs` pointer table built once at load. Output is split
  `[k × mi]` gate + `[k × mi]` up so the existing batched
  `fused_silu_mul_rotate_mq` consumes it unchanged.
- **`gemv_hfq4g256_moe_down_residual_scaled_k8_indexed`** — same pattern
  for the down projection. Reads scales from `topk_weights`, atomicAdds
  the weighted contribution into `x_residual`.
- **`scaled_add_inplace`** (CPU-scalar + GPU-scalar variants) — fuses the
  old (`scale_f32` + `add_inplace_f32`) pair used by the per-expert
  accumulator. The GPU-scalar variant reads the scale from a 1-element
  device buffer, keeping the shared-expert sigmoid gate on-device.
- **`gemv_hfq4g256_residual_scaled`** (CPU + GPU scalar) — one-kernel
  replacement for the `weight_gemv_residual` + explicit scale pair on the
  MQ4 SwiGLU down tail.

### MoE decode speed progression (gfx1100, A3B MQ4, greedy chat)

Each stage is a separate commit and a separate incremental win:

| Stage | tok/s | vs P1 |
|-------|-------|-------|
| Phase 1 dense-compute reference | 28 | 1.00× |
| Phase 2a (GPU sigmoid + fused scaled-add) | 77 | 2.75× |
| Phase 2a-ii (fused MQ4 `gemv_residual_scaled`) | 88 | 3.15× |
| Phase 2a-iii (pre-rotate x\_norm once per layer) | 102 | 3.65× |
| Phase 2c step 1 (fused 8-expert gate\_up) | 111 | 3.98× |
| Phase 2c step 2 (batched silu\_mul\_rotate) | 125 | 4.48× |
| Phase 2c step 3 (fused 8-expert down + atomicAdd) | 140 | 5.01× |
| Phase 2b+2c (GPU top-K + indexed kernels) | 153 | 5.46× |
| + hipGraph (single-turn smoke test only — see Known Issues) | 162 | 5.80× |

Production daemon path: **~115 tok/s** at `HIPFIRE_KV_MODE=asym3` (default).
Prefill is still per-token-fallback for MoE (`forward_prefill_batch`
eligibility check requires a dense DeltaNet layer), so pp ≈ decode at
~143 tok/s on 641 tokens — batched MoE prefill is v0.1.7 material.

### Daemon / serve / install polish

- **Daemon flock mutex** (`~/.hipfire/daemon.pid`). A second daemon process
  exits with `FATAL: hipfire daemon already running (PID N)` before
  touching the GPU instead of silently double-consuming VRAM. Fd released
  automatically on kill, so stale PID content is harmless.
- **Install precompiles MQ4 + asym3 defaults** for the detected arch at
  install time, so the first `hipfire run` doesn't eat a multi-minute JIT
  stall. `hipfire update` syncs the CLI before the cargo rebuild so the
  registry change propagates in the same invocation.
- **Serve**: frees weights on idle eviction (was leaking across eviction
  cycles), respects the per-model `max_tokens` config (default was a
  hardcoded 512 even after you set one), bumps the detach readiness
  timeout from 30 s to 5 min for cold kernel JIT, and enforces the KV
  budget end-to-end so oversized requests return a clean error rather
  than writing past the cache.
- **`hipfire run`** surfaces KV-budget errors instead of exiting 0 with no
  output. Spawns cargo/git via absolute paths detected via `autodetect`
  so `HIPFIRE_UPDATE` behaves the same whether invoked via a shell shim
  or directly.
- **`hipfire bench`** gained pp128/pp512/pp1024 prefill-scaling numbers,
  explicit prefill + decode split, and TTFT. Fixed a GPU-sync bug that
  was reporting prefill tok/s 5–10× too optimistic.

### Experimental

- **Gated `think-budget` alert injection.** When the model has burned
  more than `experimental_budget_alert_tokens` inside an open `<think>`
  block, the daemon splices a configurable nudge string into the stream
  — tokens are emitted to stdout AND forward-fed through the KV cache so
  the next sample sees the model having "said" them. Hard-gated behind
  config; off by default. See `experimental_budget_alert_tokens` /
  `experimental_budget_alert_text`.

### Known issues

- **hipGraph + MoE multi-turn corruption** ([#19](https://github.com/warpfront/hipfire/issues/19)).
  Single-shot short decodes with `HIPFIRE_GRAPH=1` on A3B look healthy
  (162 tok/s, byte-coherent at 30 tokens), but state diverges from the
  direct path after ~40 decoded tokens — the model starts skipping a
  number in a count, loops on a single token, etc. Root cause unclear
  after a full kernel audit (all individually graph-safe). `forward_scratch`
  gates `use_graph` on `config.num_experts == 0`; dense Qwen3.5 still
  takes the graph fast path. Cost: ~30% of the potential A3B decode
  ceiling. Tracking for v0.1.7.

## v0.1.5 "redline" (2026-04-13)

First full (non-alpha) release. Focus: **RotorQuant asymmetric KV cache** for
multi-turn recall, plus a full UX overhaul that makes hipfire feel like
Ollama — background daemon, idle eviction, interactive TUI config, per-model
overrides, and `hipfire run` auto-connecting to a running serve.

### Asymmetric KV cache (asym{4,3,2}) — replaces givens

K is rotated-quantized at 2/3/4-bit with Lloyd-Max centroids; V stays Q8_0
in normal space. Value-side reuses the existing Q8_0 flash reduce path so
only K needs the rotation machinery. Always flash, always batched prefill.

- **asym3 is the new default** on every RDNA3/RDNA4 card (5.5× compression
  vs fp32, verbatim rare-token recall on Qwen 3.5 9B multi-turn).
- **asym4** — 5.1× compression for headroom-to-spare workflows.
- **asym2** — 6.0× compression for 8 GB cards (still recall-safe for
  common tokens).
- **Legacy aliases:** `turbo`/`turbo3` → asym3, `turbo4` → asym4,
  `turbo2` → asym2.

The givens2/givens4 rotation family has been fully removed from kernels,
dispatch, and the daemon. `KvCache::new_gpu_givens{4,2}` /
`new_gpu_givens4_deferred` are gone. 11 kernel files deleted.

### Multi-turn recall — fixed

Multi-turn prompts like "My name is Kaden. … What is my name?" were
returning "Kendall" / "Kade" on 9B MQ4 + givens4 KV. Root-caused to
**two bugs** landing together:

1. **K kernel head_dim=256 half-coverage.** All rotated-K kernels had
   `tid×4 × 32threads = 128` only — second half of Qwen 3.5's 256-dim head
   was silently uninitialized. Fixed via explicit 2-pass loop
   (`half=0,1`). Invisible to md5, perf benchmarks, or single-turn tests.
2. **KV precision for rare tokens.** 4-bit K collapses the outlier
   components that carry rare-token identity ("aden" subtoken). asym3's
   3-bit quantization is precise enough — asymmetric because V reuses Q8_0.

Verified: MQ4 + asym3 KV recalls "Kaden" correctly on 0.8B/4B/9B/27B.

### Flash attention — configurable per codepath

- `flash_mode` config key, tri-state `auto|always|never`.
- Only affects the Q8 path (asym modes are flash-only — no non-flash
  kernel exists). TUI surfaces `(ignored — asym is flash-only)` when a
  user has asym KV selected.
- `HIPFIRE_ATTN_FLASH` env var accepts any of `auto|always|never|0|1|2|off|on|force`.
- Dispatch: `use_flash = capture_mode || mode==2 || (mode==1 && ctx≥2048) || ctx>15000`.

### Daemon UX — Ollama-style

- **`hipfire serve -d`** / `--detach` — forks via setsid+nohup, writes PID
  to `~/.hipfire/serve.pid`, logs to `~/.hipfire/serve.log`. Polls
  `/health` up to 30s to confirm up.
- **`hipfire stop`** — SIGTERM + 5s grace + SIGKILL fallback.
- **`hipfire ps`** — lists daemons, quantize jobs, HF uploads with ETIME
  + RSS + serve-port status.
- **`hipfire run` HTTP fallback** — if a serve is running on `cfg.port`,
  run streams through its `/v1/chat/completions` instead of spawning its
  own cold-start daemon. Skips the 2-5s load cost per invocation.
- **Idle eviction** — `idle_timeout` config (default 300s). Serve unloads
  the model when no request has arrived within the window; next request
  reloads. 0 = never unload.

### Interactive config TUI

`hipfire config` launches a keyboard-driven settings editor. No more
hunt-and-peck `config set X Y`.

- ↑↓ nav, ←→/space cycle enum values, -/+ tweak numbers, Enter edits
  free-text, `r` resets/removes-override, `s` saves, `q` save+quit,
  Ctrl+C aborts.
- Long enum lists collapse to `←→ cycle (N/M)` to avoid line-wrap.
- Values color-coded by source: green if user-set, dim if default.
- Scripting still works: `hipfire config set <key> <value>`,
  `hipfire config get <key>`, `hipfire config reset [key]`.

### Per-model config overlays

- `hipfire config <model:tag>` launches the same TUI scoped to that model.
  Rows show `(inherited)` vs `(overridden)` with cyan highlighting; `r`
  removes the override instead of resetting.
- Stored as sparse JSON at `~/.hipfire/per_model_config.json` — only
  overridden keys are persisted.
- Resolution order: `--flag > per-model > global > registry default > engine fallback`.
- Overridable keys: kv_cache, flash_mode, temperature, top_p,
  repeat_penalty, max_tokens, max_seq, thinking, max_think_tokens.
  Global-only: port, idle_timeout, default_model.
- Global TUI has a "[per-model configs]" nav row at the bottom; Enter
  opens a model picker sub-TUI that lists all registered tags with
  override count + drill-down.

### New config keys

- **`max_seq`** (default 32768) — KV cache capacity allocated at model
  load. Wired through to daemon via `params.max_seq` — fixes the pre-
  existing panic when `max_tokens > 4096` with the old hardcoded default.
- **`flash_mode`** (default auto) — see above.
- **`thinking`** (default on) — `on` = model uses `<think>...</think>`
  (stripped from display); `off` = prepends a no-think directive to the
  system prompt. Advisory (instruction-tuned models comply).
- **`max_think_tokens`** (default 0 = unlimited) — reasoning budget per
  turn. Stored + passed to daemon today; hard enforcement (forced
  `</think>` emission) is a follow-up.
- **`idle_timeout`** (default 300s) — serve auto-eviction window.

### Quantize CLI — one-shot download→quantize→upload

`hipfire quantize <hf-id|local-dir>` now supports:
- `--both` (shorthand for `--format mq4 --format mq6`)
- `--stem <name>` overrides the output basename
- `--output-dir <dir>` for multi-format outputs
- `--upload <owner/repo>` — pushes to HuggingFace after quantize
- `--create-repo` — invokes `hf repos create --exist-ok` first
- `--install` — copies to `~/.hipfire/models/` so `hipfire run` finds it
- `--register <tag>` — writes a user alias to `~/.hipfire/models.json`
  so the custom tag resolves alongside the built-in registry

Example: `hipfire quantize Jackrong/Qwopus3.5-4B-v3 --both --upload schuttdev/hipfire-qwopus-4b --create-repo --install --register qwopus:4b`

### HuggingFace uploads this cycle

- `schuttdev/hipfire-qwen3.5-{0.8b,4b,9b,27b}` — MQ6 added alongside MQ4
- `schuttdev/hipfire-qwopus-{4b,9b,27b}` — MQ4 + MQ6 (Jackrong Qwopus 3.5 v3)
- `schuttdev/hipfire-carnice-{9b,27b}` — MQ4 + MQ6 (kai-os Carnice)

### Misc

- **First-run banner** on bare `hipfire` when `~/.hipfire/config.json`
  and `~/.hipfire/models/` are both absent — walks new users through
  `diag → pull → run → config`.
- **User aliases** — `findModel` consults `~/.hipfire/models.json` before
  the built-in REGISTRY, so custom fine-tunes addressed by their
  registered tag always resolve.
- **Sampler greedy fast-path** for `temperature ≤ 1e-6` — avoids the
  `1/0 → NaN` path that surfaced at temp=0.
- **`speed-gate.sh`** switched from the retired `HIPFIRE_KV_MODE=givens4`
  to `asym3`.

## v0.1.5-alpha "ichigo" (2026-04-11)

The ichigo release focuses on one thing: **MagnumQuant**, a new 4-bit weight
format that delivers Q8-grade output quality at Q4 memory bandwidth, protected
by a mandatory byte-exact quality gate. The supporting work — cross-architecture
fused projection kernels, a silent-corruption fix in the 4-accumulator GEMV
inner loop, and arch-aware quality baselines — lands in the same cycle because
MQ4 wouldn't be trustworthy without them.

### MagnumQuant (MQ4) — new quantization format

FWHT-rotated 4-bit weights in 256-element groups. Matches Q8 output quality
at Q4 bandwidth on every model we've measured.

- **Qwen3.5 MQ4 family on Hugging Face** — `schuttdev/hipfire-qwen3.5-{0.8b,4b,9b,27b}` with model cards
- **`.mq4` file extension** — recognized by CLI, daemon, and weight loader
- **CLI tags** — `hipfire pull qwen3.5:{size}-mq4` pulls the quality-gated MQ4 variant
- **HF4 remains the default** (still the fastest path) — MQ4 is explicit opt-in for quality-sensitive workloads
- **`magnum` research crate** — butterfly rotation + adaptive-mode quantizer, used for the encoder

### Mandatory byte-exact quality gate

Every change to kernels, quant formats, dispatch, fusion, rotation, rmsnorm,
or the forward pass must pass `scripts/quality-gate.sh --fast` before being
committed. Enforced automatically via `.githooks/pre-commit`.

- **Deterministic greedy decoding** (temp=0, no sampling, no repeat penalty)
- **9-test matrix** — 3 models (0.8B / 4B / 9B MQ4) × 3 prompts (compiler, math, federalist)
- **Per-GPU baselines** — `tests/quality-baselines/{gfx1010,gfx1100}/` with auto-detection via `amdgpu-arch` / `offload-arch`, honors `HSA_OVERRIDE_GFX_VERSION`
- **Byte-exact token-ID comparison** — stricter than prose coherence or md5 checks

### Silent MQ4 corruption fix — 4-accumulator interleave

A tail-group accumulator bug in the gfx1100 4x-unroll HFQ4 GEMV was dumping
all tail groups into `acc0` instead of distributing them across `acc[g%4]`.
Output was visually coherent and benchmarks passed, but token IDs diverged
from reference on any hidden_dim where `hidden_dim % (4*64) != 0`. The bug
hid for weeks because 9B/27B happened to have no tail.

- **Fixed in `5302926`** (gfx1100 4x-unroll variant)
- Same 4-accumulator interleave pattern ported to `gemv_hfq4g256` (default),
  `gemv_hfq4g256_wide`, `fused_gate_up_hfq4g256`, and `gemv_q8_0_wide`
- **The quality gate above was designed around catching this class of bug.**
  Every quality difference is now a signal until proven otherwise with
  byte-exact evidence.

### Cross-architecture fused projection kernels

The three fused GEMV projections that originated as gfx1100-tuned single-arch
kernels now compile and run on any RDNA arch from one source family, consolidated
via the 4-accumulator interleave pattern.

- **4-way LA projection** — `wqkv + wz + w_beta + w_alpha` in one launch
- **3-way FA projection** — `wq + wk + wv` in one launch
- **FFN gate+up** — `gate + up` MQ4/HF4 GEMV in one launch
- Active on gfx1010 / gfx1013 / gfx1030 / gfx1100 via dtype gate (no per-arch fork)
- Consolidation landed in `9d05c9f` (net −187 lines)

### Qwen3.5 forward-pass fusions (gfx1100)

Every layer boundary in the DeltaNet hybrid got at least one kernel fusion
this cycle.

- **conv1d + SiLU + Q/K/V split** → single kernel
- **l2_norm(Q) + l2_norm(K) + scale(Q)** → single kernel
- **sigmoid(dn_beta) + alpha_gate(dn_alpha)** → single kernel
- **sigmoid(fa_gate) + mul(fa_attn_out, fa_gate)** → single kernel
- **rmsnorm + FWHT rotation** → single kernel (Phase 3.6)
- **residual add + wo / w_down GEMV** → single kernel (Phase 3.7)
- **SwiGLU + MQ4 w_down rotation** → single kernel (Phase 3.8)
- **Per-head Q/K memcpy loop** → fused deinterleave kernel (+52%–76%)

### Multi-row HFQ4 GEMV on non-RDNA3

`R=2` multi-row HFQ4 GEMV is the new default on gfx1010 / gfx1013 / gfx1030
(RDNA1/RDNA2). Single-row was already at the bandwidth ceiling on gfx1100,
so it keeps `R=1`.

- **+2.75% measured on BC-250** (gfx1013)
- Configurable via `HIPFIRE_GEMV_ROWS` env var
- Kept opt-in on gfx1100 since the multi-row sweep showed monotonic regression

### Performance (RX 7900 XTX, gfx1100, forward-only MQ4)

| Model          | tok/s   |
|----------------|---------|
| Qwen3.5-0.8B   | **447** |
| Qwen3.5-4B     | **187** |
| Qwen3.5-9B     | **135** |
| Qwen3.5-27B    | **46**  |

End-to-end steady-state with the default CPU sampler is ~82% of forward-only;
the gap is a fixed sampling pipeline cost, not throughput-bound.

### Performance (Radeon Pro V620, gfx1030)

Baseline from an external tester on V620 (32 GB, ROCm 7.2.0) measured at
`dcd928e` — i.e. **before** the cross-arch fused-projection consolidation.
Post-consolidation V620 numbers pending hardware access; expect an uplift
on top of these.

| Model            | tok/s    | vs master |
|------------------|----------|-----------|
| Qwen3.5-9B HF4   | **61.8** | +118%     |
| Qwen3.5-9B MQ4   | **62.4** | —         |
| Qwen3.5-27B HF4  | **21.0** | —         |
| Qwen3.5-27B MQ4  | **20.9** | —         |

**27B MQ4 matches 27B HF4 throughput within 0.5%** — the 0.7 GB FWHT metadata
overhead is bandwidth-free on the RDNA2 L2 cache.

### Experimental: GPU-assisted top-K sampling

Off by default. Enable with `HIPFIRE_GPU_TOPK=1`. Net-neutral on gfx1100
(top-K extraction cost ≈ saved CPU sampling time) but lays the hardware
groundwork for a fully on-device sampler. Debug harness via
`HIPFIRE_SAMPLE_COMPARE=1` cross-checks CPU vs GPU paths byte-exact.

### Experimental: hipGraph / kernarg blob

Kernarg blob path in `hip-bridge` makes kernel launches hipGraph-capture-safe
for gfx1100. Real-kernel POC on gfx1013 produced a **negative result** (capture
hangs on RDNA1), documented in `6da45fd`. hipGraph integration is parked until
the gfx1013 regression is understood.

### Experimental: Redline / HSA bridge

Thin Rust FFI to `libhsa-runtime64.so` via the new `hsa-bridge` crate, part
of the Phase 1/2 redline audit for a direct-KMD dispatch path that bypasses
the full ROCm userspace stack.

### Experimental: speculative decoding (infrastructure)

Dual model slot + autoregressive verify-and-accept loop + DFlash hidden-state
extraction land in-tree but are not wired to the main inference path yet.
Expect activation in a later release.

### CLI / Serve

- `hipfire pull qwen3.5:{size}-mq4` — MQ4 family tags wired into the registry
- `.mq4` extension recognized across CLI, daemon, and model loader
- **`listLocal()` bug fix** — stale dangling symlinks no longer abort the local-model scan and drop every file after the bad entry
- Fuzzy model search requires explicit tag for `.mq4` (won't silently substitute for HF4)

### Diagnostics & profiling

- **Per-kernel bandwidth profiler** for the gfx1100 forward pass — each kernel's effective GB/s vs theoretical ceiling
- **Per-arch bench + profile + top-5 logit dump** examples
- Kernel efficiency profiler with hardware caps + occupancy analysis

### Known limitations

- **Non-RDNA3 byte-exact re-verification pending.** The cross-arch consolidation
  (`9d05c9f`) passes the gfx1100 byte-exact quality gate (9/9 on 2026-04-11),
  but post-consolidation byte-exact verification on gfx1010 / gfx1013 / gfx1030
  is deferred pending hardware access. The V620 baseline above is functionally
  validated at `dcd928e` (prose coherence + factual accuracy + bandwidth).
  Tracked in #64.
- **llama.cpp Q4_K_M comparison on non-RDNA3** — deferred; tracked in #65.
- **MQ6 family** — not included in 0.1.5; tracked in #67.
- **HF4/HF6 daemon HTTP response trailing-bytes bug** reported on an external
  V620 setup; investigated on k9lin (7900 XTX / Bun 1.3.5 / current tree) and
  **not reproducible**. If you hit it, please file with `bun --version` and
  `curl -v -o body.bin` output.

## v0.1.4-alpha (2026-04-08)

### Sampling
- **Frequency-scaled repeat penalty** — replaces the flat penalty with a
  count-based score weighted by recency decay. Tokens seen once far back get
  barely penalized (~1.01x); tokens repeated 3x recently get hit hard (~p³).
  Fixes long-generation word salad on all architectures. Default penalty
  dropped 1.3 → 1.15 (effective range now 1.0–1.5x).

### Kernels
- **`ds_swizzle_b32` FWHT butterfly passes** — replaces `__shfl_xor`
  (`ds_bpermute`) in all FWHT butterfly passes. 40 instructions upgraded,
  -3 VGPRs in turbo attention kernels (31→28 on gfx1010). Verified on
  gfx1010 / gfx1030 / gfx1100 / gfx1200 / gfx1201.

### gfx1100 DeltaNet correctness
- RDNA3-specific DeltaNet code path fix (details in commit `2abf27a`).

## v0.1.3-alpha (2026-04-05)

### DeltaNet Quality Fix
- **Stochastic rounding** in Q8/Q4 state requantization — fixes coherence degradation after ~500 tokens
- Gate activation verified correct (matches flash-linear-attention reference)
- Coherent output at 5000+ tokens on 4B/9B models

### 3x Speed Improvement
- **Deinterleave kernel** replaces per-head memcpy loop in full-attention layers
- 576 individual HIP memcpy calls → 9 single kernel dispatches per token
- 9B Q4: 15 → 43 tok/s

### Multi-Turn Conversation
- Cumulative KV cache + DeltaNet state across turns
- System prompt support via ChatML (`<|im_start|>system`)
- KV capacity guard with auto-reset + DeltaNet state zeroing
- Correct ChatML boundary handling (newline token run through forward)

### Interactive REPL
- `hipfire run` — ollama-style interactive chat
- `--system`, `--turbo`, `--asym`, `--hf4`, `--boundary`, `--temp`, `--max-seq` flags
- `/reset`, `/stats`, `/quit`, `/help` commands
- Thinking blocks shown dimmed, speed stats per response

### Asymmetric KV Cache (TurboQuant+)
- Q8 keys + turbo4 values — 5.1x compression vs FP32
- Attention kernel rewritten for warp-cooperative structure
- Boundary layer protection (LA-V7): first/last N KV layers at Q8
- Polynomial centroid dequant: pure ALU, zero constant memory traffic
- 9B fits at 8K+ context on 8GB VRAM (was OOM at >2K)

### Redline Engine (experimental)
- Direct-KMD GPU compute via bare libdrm_amdgpu — no HIP/ROCm needed
- 30.5µs FastDispatch, 0.5ms startup, 2.8MB RSS
- RELEASE_MEM + WAIT_REG_MEM compute barriers on gfx1010
- Dispatch API: load module, kernel, command buffer, chain dispatch
- Benchmarks: redline vs HIP numbers in benchmarks/redline_vs_hip.md

### Universal GPU Support
- JIT kernel compilation via hipcc for any detected GPU arch
- Removed pre-compiled kernel blobs (9MB, stale cache source)
- Dynamic arch detection from gfx_target_version (no whitelist)
- Targets: RDNA1-4, APUs (Strix Halo), datacenter (BC-250)

### Windows Fix
- .exe extension for daemon/infer/run binary lookup

### HF4-V Experiment
- Hipfire-native 4-bit V format (no FWHT, 32 VGPRs)
- Benchmarked: FWHT rotation confirmed as memory access optimization on RDNA1
- Turbo4+poly remains optimal compressed V path

## v0.1.2-alpha (2026-03-29)

- Initial Qwen3.5 DeltaNet support
- TurboQuant KV cache (turbo2/3/4)
- HFQ4/HFQ6 weight formats
- CLI: pull, run, serve, update, diag
