# Flash-Next KL-to-bf16 quality method

## Decision and reporting contract

Use the pinned **source checkpoint, including unquantized routed experts**, as teacher. Extend the existing layer-streamed CPU source oracle; do not call a BF16-trunk/MQ4-expert engine artifact a bf16 reference. The primary metric is full-vocabulary, teacher-to-candidate KL, in **nats per scored next-token position**:

`D_t = sum_v p_T(v|x_<t) * (log p_T(v|x_<t) - log p_C(v|x_<t))`.

Use temperature 1, no top-k/top-p sampling, no repetition penalties, and identical token IDs, prefixes, masks, resets and chat template. Teacher forcing means every implementation consumes the frozen corpus, not its own generated continuation. Teacher and candidate remain on their own hidden-state trajectories. Compute log-normalizers and KL reductions in float64 from the actual forward logits; do not round probabilities or clip away the tail. Persist the per-position values.

Publish, separately for text and assistant positions and for their equal-weight mixture:

1. Candidate versus source bf16, ordinary routing.
2. Candidate versus the same teacher, teacher expert IDs forced at every layer and input position.
3. Bf16 versus bf16 numerical floor, ordinary routing; additionally the same floor with teacher IDs forced.

Also publish mean, document-bootstrap 95% CI, p50/p95/p99, teacher and candidate NLL/PPL, top-1 agreement, routing agreement and context-length strata. Report paired ordinary-minus-forced differences, not an additive attribution of all damage to routing. Do **not** subtract the bf16 floor from KL: KL is not additive.

External published KL claims are treated as comparable **in kind** only. This is not a reproduction of an undisclosed corpus, arithmetic, KL support or forcing semantics. Label our direction, support and forcing semantics explicitly; a different corpus cannot establish a like-for-like numerical win.

All repository citations below are relative to `/home/kaden/ClaudeCode/warpfront/hipfire-beta`. This document is a design, not a measured quality result. No GPU runs or competitor software were used. Numerical estimates marked `[INFERENCE]` are planning envelopes, not benchmark observations.

## (a) What exists, and what it cannot establish

### Engine KLD example

`crates/hipfire-arch-qwen4/examples/qwen4_kld.rs:8-17` explicitly says that the production engine requires MQ4 routed experts and that its BF16-trunk teacher measures trunk/head/PLE damage only. Thus it **cannot expose the expert quantization error shared by teacher and candidate**. “Share one route” in this comment means the production forward path, not teacher-ID replay: different hidden states can still select different experts.

The example already offers:

- Engine `ref`/`eval`, top-256 by default, chunk-reset all-row logits, and a `--decode` path (`qwen4_kld.rs:19-27`).
- Text tokenization into independent windows, with `WindowSpec::legacy_half`, rather than assistant-role masks (`qwen4_kld.rs:424-448`).
- Import of external F32 `[chunk][scored position][vocab]` logits using a template reference (`qwen4_kld.rs:479-536`). This importer is useful plumbing, but turns logits into top-k reference blocks; it does not preserve full-vocabulary teacher probabilities for exact KL.

The named `reference_oracle/kld_teacher.py` exporter is referenced at `qwen4_kld.rs:14-16,479-481` but was not present in the inspected oracle directory. Do not treat that comment as a runnable exporter. Keep the engine reference as a **trunk-relative diagnostic**, not the headline bf16 metric.

### True source oracle

`crates/hipfire-arch-qwen4/reference_oracle/quality.py:4-9` already describes the desired memory strategy: needed embedding/PLE rows, one decoder layer at a time, selected expert slices and bounded LM-head rows. BF16 source words are decoded by viewing uint16 storage as `torch.bfloat16` (`quality.py:129-147`), and the source adapters use the hidden tensor's dtype (`reference_oracle/upstream.py:1756-1761`). The MoE adapter loads source router/shared weights and source expert gate/up/down slices, not MQ4 codes (`upstream.py:1854-1903`). This is the correct starting point for a true source-weight teacher.

Current limits:

- Exactly **17 pinned input tokens**, hence 16 next-token targets, enforced by count and SHA, not merely a default (`quality.py:42-50,62-90,191-196`). A `--tokens` argument does not make this a general corpus evaluator.
- **Top-32** logits/IDs and a full-vocabulary normalizer, not all teacher logits (`quality.py:315-355`). LM head is only 25 rows per read, about 9,933 chunks across the 248,320-token vocabulary (`quality.py:50,320-327`).
- `compare()` constructs uniform-tail surrogates on the union of two top-k supports (`quality.py:409-458`). This is not exact full-vocabulary KL; its tail accounting must not become the primary estimator. Its displayed CI endpoints equal the mean, rather than expressing sampling uncertainty (`quality.py:462-463`).
- Source identity is recorded, but determinism is explicitly **not verified**, and MTP is unavailable (`quality.py:368-381`). The reported `peak_layer_cache_bytes` is an accumulated allocation counter (`quality.py:119-125`), not measured live RSS.

Reuse these source operators and provenance checks, but extend the corpus/export/reduction layer and measure actual memory. The pinned checkpoint and Transformers identities are in `upstream.py:99-103`.

## (b) Feasible true-bf16 teacher and capacity

### Weight accounting: 4 cards do not fit

Pinned geometry is H=2,560, L=48, E=512, top-k=10, expert intermediate I=640, vocab=248,320; 36 GDN and 12 full-attention layers (`src/config.rs:1011-1048`, including full-attention interval 4). Source dtype is BF16 and recurrent state is F32 (`config.rs:1002,1032-1035`).

Calculated from these dimensions (decimal GB unless marked GiB):

| Component | Calculation | BF16 bytes |
|---|---|---:|
| One routed expert, gate/up/down | `3*2560*640 = 4,915,200` parameters | 9.8304 MB = 9.375 MiB |
| Routed experts in one layer | `512*4,915,200` parameters | 5.0331648 GB = 4.6875 GiB |
| All routed experts | `48*512*4,915,200 = 120,795,955,200` parameters | **241.5919104 GB = 225 GiB** |
| Nominal 125B core | `125e9*2` | **250 GB = 232.831 GiB** |
| Untied embedding or LM head, each | `248320*2560*2` | 1.2713984 GB |
| External PLE table, if fully BF16-resident | `320001536*160*2` | **102.40049152 GB = 95.368 GiB** |
| Nominal core plus full BF16 PLE | `250 + 102.40049152` | **352.40049152 GB = 328.199 GiB** |

PLE has 16 heads, valid rows 320,001,446 and padded rows 320,001,536 at width 160 (`src/ple.rs:14-26`). `[INFERENCE]` The numerical match of the routed-expert count to 125B after adding the trunk suggests the advertised 125B excludes this external table; do not silently assume the table is included. The exact manifest sum, including MTP and all small tensors, is an implementation gate. The source reader verifies actual tensor dtypes: do not infer a specific source tensor's dtype from an artifact recipe.

Even giving each of the four cards **32 GiB**, the generous aggregate is 128 GiB = 137.439 GB. **Experts alone exceed it by 97 GiB**; the nominal core exceeds it by 104.831 GiB before activations and workspaces. If “32 GB” is decimal, the deficit is larger. A fully resident bf16 teacher cannot run across these four R9700s. The 122 GB host RAM also cannot hold the core. Adding host RAM and VRAM is not a sound residency plan: it leaves no practical headroom even under generous unit interpretations, and excludes the full PLE table.

### Recommended implementation: CPU layer stream, expert chunks

Use the pinned source forward under inference/no-grad mode with BF16 source weights/activations and the source's specified F32 norm/router/recurrent accumulation where applicable. This is a **source-bf16 forward**, not an all-F32 forward merely initialized from BF16 weights. Do not change casts to accelerate it without treating that change as a separate implementation/floor variant.

1. Stage immutable source ranges in a local, hash-checked tensor store. Do not repeatedly issue tiny remote requests during scoring. Existing range access is appropriate for provenance/bootstrap but needs a coalesced local binary reader for production corpus sizes; `_StreamWeights.full` currently concatenates many pieces (`quality.py:155-168`).
2. Process about 8,192 input tokens per layer pass, grouping independent sequences without allowing attention, convolution, recurrent state or PLE history to cross document boundaries. Apply each layer to all sequences in that pass before discarding its weights.
3. Group positions by the teacher's selected experts. Read each used source expert once for that layer/pass; apply it to its selected rows and release it. Expert streaming adapters already exist (`upstream.py:1547-1573,1877-1895`). A bounded expert cache or one complete BF16 expert layer is optional, not a full-model cache.
4. Fetch only the PLE rows needed by the corpus (`quality.py:215-224`). Worst-case full physical row fetches are `16*positions*160*2` bytes before deduplication. The proposed 327,680 input positions need at most **1.678 GB**, not a 102.4 GB resident table. Do not truncate row contents or replace source rows with quantized PLE rows.
5. Stream the source LM head in locally efficient chunks, preserving each forward's BF16 matmul semantics, and export full F32 logits or replayable source final hidden states plus an immutable head identity. The latter requires only **320 MiB** for 65,536 scored BF16 hidden rows; full F32 teacher logits occupy **65.096 GB**. The exact scorer must reconstruct *all* vocabulary logits, not top-32 values.

`[INFERENCE]` Expected CPU peak RSS: roughly **16–40 GiB**, depending on expert caching and QSA temporaries; set a **64 GiB hard admission budget** and measure it on the pilot. One full expert layer is 4.6875 GiB BF16 or 9.375 GiB after F32 expansion; two prefetched BF16 layers are 9.375 GiB. An 8,192-row, four-way H=2,560 hyper-connection activation is 160 MiB BF16/320 MiB F32 per buffer. A naive full-attention score tensor alone is `24*8192^2*4 = 6.442 GB` in F32, so context caps and temporary liveness matter. Do not batch long sequences into one giant attention matrix. Current source operators must pass pilot RSS and causal-mask checks before scaling.

### Time and storage estimates

`[INFERENCE]` For the corpus below: 327,680 forwarded positions, 65,536 scored positions, approximately 40 passes of 8,192 positions. With sufficiently diverse top-10 routing, nearly all 512 experts are used in a pass; conservatively budget **250 GB of core reads per pass**, or **10 TB per teacher traversal**. At assumed sustained local storage throughput 0.5–3 GB/s, that is **5.56–0.93 hours of I/O**. These bandwidths are assumptions, not measured disk performance.

`[INFERENCE]` Budget 12–20 GFLOP per forwarded token for active expert/shared/trunk/head and attention work, i.e. **3.93–6.55 PFLOP per traversal**. At an assumed effective CPU rate of 0.2–2 TFLOP/s, matmul arithmetic alone is **9.10–0.55 hours**. Those rates are not measured, and the current Python/recurrent/operator overhead can dominate. Reserve **10–48 hours per optimized CPU teacher traversal**, with two bf16 floor implementations costing roughly twice that; the unchanged tiny-range/Python implementation could take substantially longer. A timed CPU pilot replaces this estimate before authorizing the full corpus. Reuse a frozen teacher output across all candidate runs.

A CPU-only snapshot in this session found a 32-core/64-thread Threadripper 9970X (`lscpu`) and 849G free on the output filesystem (`df -h /home/kaden/qcal/release-0.4.2`). These are command observations, not source citations or reserved capacity. Budget roughly **550–650 GB local disk** for a full source checkpoint including BF16 PLE, two 65.1 GB bf16 logit exports, one candidate export and temporary/index space. Selected-PLE staging and final-hidden replay reduce that substantially. Recheck capacity at execution time; no large files were created here.

### Other feasible placements

- **Four-GPU BF16 layer streaming** is feasible in memory, not fully resident: a source expert layer is only 5.03 GB before sharding, and the trunk/sequence workspaces can be bounded. It requires an offload scheduler, exact BF16 expert dispatch and source-equivalent QSA/GDN/PLE, not today's MQ4-only engine teacher. `[INFERENCE]` With 10 TB of cumulative transfers at an assumed effective 10–25 GB/s, host-to-device traffic alone is 16.7–6.7 minutes per traversal; cold source storage remains the 0.93–5.56 hour bound, plus compute and synchronization. No GPU timing or feasibility run was performed.
- **Per-expert CPU/GPU chunking** keeps each expert's source storage to 9.375 MiB (18.75 MiB expanded F32). Group selected tokens before loading it; serial token-by-token reload would multiply I/O catastrophically. It is a memory optimization of the same real source forward, not a different teacher.

Start with CPU: source arithmetic and streaming already exist, and no multigpu engine implementation is required to produce the teacher.

## (c) Teacher routing replay

### What hooks exist

There is an **observation hook**, not a ready Qwen4 forced-routing API: `HIPFIRE_QWEN4_ROUTE_TRACE` copies layer top-k expert IDs after MoE and writes one line per single-token forward (`src/gpu_forward.rs:2058-2068,2270-2282,3836-3865`). It is limited to `n == 1`; it does not replace routing or cover general chunk rows. Token/chunk entry points are at `gpu_forward.rs:2549,2631`.

The source MoE constructs `router` and installs it as `sparse.gate` (`upstream.py:1863-1864,1897-1903`). Wrap this gate's forward to capture **logits, scores and ordered indices** without executing a second router; the tuple and expert-call conventions are shown at `upstream.py:2813,2915-2916`. This capture is new work, not an existing persisted teacher route tape.

Internal dispatch has `MoeRouterInput::PrecomputedSoftmaxTopK` (`crates/hipfire-dispatch/src/pipeline/sealed_moe.rs:256-261`) and an actual routing skip in `pipeline/moe_program.rs:1424-1429`. But these are **EP non-root route-receipt mechanisms**, not arbitrary Qwen4 replay: root precomputed routes are rejected (`sealed_moe.rs:1928-1931,2463-2467`), and grouped prefill requires a route receipt (`sealed_moe.rs:2003-2006`). The apparent legacy `skip_routing` in `pipeline/mod.rs` is literally false (`mod.rs:2899,3143`); do not rely on its nearby pre-routed comment as a usable hook. Qwen4 selects ordinary `SoftmaxTopK` (`gpu_forward.rs:407-425`); the top-10 production sites include `pipeline/mod.rs:3172-3179` and `pipeline/sealed_moe.rs:2085-2089`.

### Defined variant and new hook

Primary **ID-forced KL**: at every layer, compute the candidate's own router logits on its own hidden state, replace its selected IDs with the teacher's ordered top-10 IDs, and gather/normalize the candidate probabilities **on those IDs**, with the source's normalization and dtype-rounding rules. Do not retain candidate weights belonging to the candidate's discarded IDs. Do not transplant teacher hidden states or expert outputs. Shared expert and shared gate remain candidate computations.

Also support a separately named **IDs-and-weights-forced diagnostic** that replays the teacher's normalized scores as well as IDs. This removes additional router-score differences and must not be conflated with ID-only forcing. External forcing semantics are not always disclosed; show the distinction rather than guessing.

`[INFERENCE / proposed implementation]` Add an opt-in, evaluator-only route-tape input at Qwen4 bundle/forward lowering and a typed **quality-replay route authority** in the MoE dispatch. At the router-to-expert-dispatch boundary, inject tape IDs and either candidate gathered/renormalized scores or teacher scores, before expert grouping or indexed GEMV/GEMM. Existing EP precomputed machinery may share mechanics, but do not forge an EP receipt, overwrite buffers behind the seal's assumptions, or weaken production proof gates. Disable router/shared fused variants that cannot honor the replay contract. Thread replay through chunk, decode and future EP root routing; non-root ranks must consume the root-authoritative forced route.

Tape key: checkpoint/source/operator hashes, corpus/tokenizer/template hashes, sequence ID, zero-based input position, layer, ordered IDs and F32 scores. Capture **all context positions**, not just scored rows: earlier MoE choices affect later hidden states. Preserve expert order/ties and sequence resets; reject missing rows, duplicate/out-of-range IDs, wrong top-k, nonfinite scores and incompatible identities. For 327,680 inputs, IDs alone need **600 MiB**; adding 10 F32 scores per row doubles that to **1,200 MiB**. The 65,536 scored-row tape alone is 120 MiB and is insufficient for replay.

## (d) Bf16-vs-bf16 floor

Produce source-bf16 A and source-bf16 B with identical checkpoint/token/PLE inputs and no quantized weights. Preferred eventual floor is source CPU versus an independently implemented streamed BF16 forward. The feasible initial floor is the **same pinned CPU source forward with two legal FP32 reduction orders**, retaining identical BF16 storage and output-rounding boundaries:

- A: pinned/native reduction and expert accumulation order.
- B: a blocked/pairwise FP32 GEMM reduction and/or different expert contribution summation order, rounding to BF16 at the same operator boundaries. Normalization, GDN state precision, causal masks, top-k rules and nonlinearities remain unchanged.

This measures a numerical-order floor, not independent-implementation coverage. Instrument the actual reduction path; changing a thread count or LM-head chunk size alone does not demonstrate a different reduction order. An all-F32 forward is a useful accuracy check, **not** a bf16/bf16 floor.

Score A→B with ordinary routing and record their routing disagreements; then repeat B on A's IDs for a same-routing floor. A→A identical-logit comparison is the zero/estimator sanity check, not the requested floor. A legitimate floor can be zero on some rows; never inject perturbations or tune it to reproduce 0.0077. Publish the observed floor and CI without treating the competitor's value as our required constant.

## (e) Frozen corpus and precision of the estimate

Start with **65,536 scored positions**, split equally:

- **Text:** 128 independent held-out documents, 256 positions per document; balance general prose, technical material and multilingual text with recorded sources/licenses.
- **Assistant:** 128 independent held-out conversation/task instances, 256 assistant-content targets per instance; balance explanation, coding and reasoning answers. Use existing fixed human/reference answers, not completions regenerated separately by each candidate. Apply the pinned chat template; prompts and role delimiters are context, but only assistant-content next-token targets enter this stratum.

For each stratum assign 96 sequences of 512 tokens, 24 of 2,048 and 8 of 8,192, each containing 256 eligible score positions plus context/next-token targets. Total forward work is **327,680 tokens**. Documents may be selected for sufficient length, but selection and position masks must be frozen before inspecting candidate logits. For assistant rows, include prompt and earlier assistant context; every scored position needs an actual next-token target. Exclude padding and the final row without a target. Keep independent-document resets, n-gram boundary/history semantics and recurrent state boundaries identical across implementations. Record language/task and context-length strata; the eight long-context units per stratum are diagnostic, not sufficient for a precise stand-alone long-context claim.

Use deterministic position sampling with a published seed, not “interesting” or high-KL positions. Sampling 256 positions across each eligible region is acceptable if its weighting is frozen and reported. Weight each of the two primary strata equally; their unit sizes are equal. Compute CIs by resampling **documents/conversations**, not individual correlated tokens, with at least 2,000 fixed-seed bootstrap replicates. Resample paired runs together when comparing ordinary/forced/floor variants.

`[INFERENCE]` 65K scores are a practical starting size, not a guarantee of stability. Require a 95% CI half-width ≤0.002 nats in each primary stratum and mixture (and ≤5% of the mean where mean ≥0.02), plus agreement of the two predeclared corpus halves within their uncertainty. If this fails, append independent units in deterministic doubling rounds without changing existing units or dropping outliers. Publish final sample size and the stopping rule; exceptional rows stay in the mean and p99. Keep corpus IDs/hashes so all candidates and both floor variants use the same final corpus.

## (f) Ordered implementation and acceptance gates

1. **Pin identities and verify capacity.** Freeze checkpoint revision/index/tensor hashes, operator AST/source hashes, tokenizer, chat template and full dtype/shape manifest. Sum BF16 bytes including external PLE and MTP separately. Reserve local disk and a 64 GiB RSS limit. **Gate:** every admitted teacher weight is true source BF16/F32/I64 as appropriate, no MQ4/Q8 substitute; exact role/shape/dtype match; sufficient disk. Main next-token KL excludes speculative MTP; do not represent it as MTP quality.
2. **Generalize source corpus handling without losing the canonical test.** Replace the hard-coded scoring-only restriction with a manifest carrying independent sequences, role masks, score masks and resets. Preserve the original 17-token canonical fixture as a regression case. Add local coalesced source reads and explicit inference mode. **Gate:** canonical source outputs/identity match, boundary/target-shift/mask checks pass, and BF16 expert/PLE bytes match source tensors; no cross-sequence state leak.
3. **Build exact full-vocabulary export/scoring.** Stream all logits or source final-hidden/head replay; add chunked F64 logsumexp and exact KL. Keep HFKLDR top-k output only as a clearly labeled secondary diagnostic. **Gate:** independently computed toy-distribution KL agrees within 1e-10; identical logits yield |KL|≤1e-10; streamed/materialized reduction agreement within 1e-8 nats; extreme-logit cases remain finite. Fail materially negative KL rather than silently clipping it. Teacher target NLL equals independently recomputed NLL from the exported logits. Role masks, last-target exclusion and counts agree.
4. **Capture teacher routing and implement sealed replay.** Add source gate capture, complete all-position tapes and the evaluator-only typed replay path above. **Gate:** replayed IDs match the tape exactly at every layer/input position; gathered probabilities normalize correctly; replay of an implementation's own tape preserves its output within declared numerical tolerance; ordinary mode is unchanged when replay is disabled. Test chunk/decode and reset alignment, malformed tape rejection, and future EP authoritative-route propagation separately.
5. **Implement and validate floor A/B.** Use true BF16 experts in both variants; record the actual arithmetic-order difference and all cast boundaries. **Gate:** same weight/input identities, all source operators exercised, A→A zero sanity check, reproducible A/B logits and routing with fixed runtime settings, ordinary and same-routing floor artifacts produced. Unexpectedly large localized differences must be explained before calling this a numerical floor; do not normalize them away.
6. **Timed CPU pilot.** Run the original canonical sequence and a frozen small pilot including text/assistant and long-context cases, then an 8,192-input-token layer-stream pass. This is future execution, not performed here. **Gate:** peak measured RSS <64 GiB, no full-model residency/swap thrash, correct next-token count and all-vocab output, actual bytes read and seconds/layer recorded, reproducibility checked, and extrapolated disk/time fits the execution allocation. Replace the planning envelope with those measured estimates.
7. **Freeze and run the quality matrix.** Generate A once and reuse it; export B and candidate ordinary/ID-forced (optional IDs+weights diagnostic) on exactly the same corpus. Run candidate chunk and decode routes separately rather than silently mixing them. Candidate GPU execution is a later main-agent gate, not authorized by this CPU-only design assignment. **Gate:** all provenance/mask/tape identities match, per-position output is complete/finite, bootstrap stability rule passes or corpus expands as specified, and both text/assistant results plus floor are present. Report candidate weight recipe, topology and route; do not reuse a single-device score as multigpu evidence.
8. **Publish one reproducible result bundle.** Include corpus manifest, full teacher or replay identity, route tape, floor variant identity, exact estimator code/version, per-position records, grouped CIs and command logs. **Method acceptance:** it can observe expert quantization damage, separates free/forced routing, and measures a real bf16/bf16 floor. **Quality targets:** compare the measured ordinary/forced means and uncertainty to the supplied 0.060/0.041 reference claims, without declaring an apples-to-apples pass unless scoring semantics and corpus are shared. No quality gate is passed by this design document alone.

The first revision of this document was a read-only design deliverable. The implementation and execution placement below supersede its planning-stage status; the source-inspection citations above refer to the original `hipfire-beta` baseline, not the new worktree's shifted line numbers.

## Implementation and rental execution cutover

Main subsequently requested implementation, then Kaden cancelled the local timed pilot and selected a rented **single MI300X (192 GB, large host RAM)**. No Flash-Next BF16 weight payload or real teacher forward was run on hiptrx. The 22 MiB attempted tokenizer cache was removed; offline validation is recorded separately in `offline-validation.json` and `offline-validation.log`.

Implementation is in `/home/kaden/ClaudeCode/warpfront/hipfire-kl-method`, branch `work/fn-kl-method`, from `origin/beta` at `d7ae367`, committed without pushing:

- `reference_oracle/quality.py`: generic independent-sequence corpus/assistant masks; layer-major shared source-weight passes; CPU or `cuda`/ROCm device selection; exact all-vocabulary export/KL; all-input-position route files; bootstrap and split-half stability evidence; paired native/reverse-order BF16 floor and self sanity; timing/RSS/device-memory/provenance records.
- `reference_oracle/upstream.py`: validated local/streamed source manifest and raw BF16 reads; actual router-call capture; source-provenance-preserving reverse-expert accumulation order. The default pinned source hashes are not weakened.
- `reference_oracle/quality_corpus.py`: deterministic local-tokenizer corpus freezing, actual rendered assistant-content masks, source/template hashes and explicit code-only HumanEval pilot-proxy labels.
- `reference_oracle/rent_setup.py`: fresh rental venv, pinned ROCm torch/device wheels and HF source download/hash checks. This setup is not executed on hiptrx.

The old 17-token/top-32 mode remains a canonical regression path; its limitations do not apply to the new `--exact-export` / `--exact-floor` path. Source BF16 LM-head outputs may be stored **losslessly** as BF16 words after exact round-trip verification, reducing a 65,536-score reference to **32.548 GB**; arbitrary F32 candidate logits are not rounded. UInt16 expert IDs are exact for 512 experts, reducing a complete 327,680-input route tape to **300 MiB IDs + 600 MiB F32 scores** per implementation.

This code captures the teacher route tape. Production-engine candidate-ID replay was not included in Main's narrowed implementation assignment and is not claimed to exist; its design and IDs-only versus IDs+scores semantics remain as specified above. MTP quality also remains out of scope.

Use `RENT-RUNBOOK.md` for exact setup, teacher/floor commands and hashed return files. Official AMD wheels for **ROCm 10.1.0 / torch 2.14.0+rocm10.1.0 / gfx942** were verified to exist; the runbook only permits an explicitly recorded ROCm-7.14 fallback if a matching 10.1 wheel is genuinely unavailable, never merely on a network failure. Target-rental throughput, floor and estimator stability are **unmeasured** until its approved pilot runs. The earlier CPU time envelope is not a measured rental result.
