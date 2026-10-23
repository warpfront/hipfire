# Native gfx1151 RW4 output-projection HC gate sharing — 2026-10-21

**Lifecycle:** historical. Fixture-bound measured evidence; no product floor/default/admission follows from this ledger. Parent `0689861a512a5c8ab9d0b5bc02a836f37ff936ed`, branch `experiment/halo-native-pd-20261021`, dirty one-file source at build. Local commit only, no push; full native GPU–NPU Prefill+Decode goal remains active.

## Change and mechanism

In `qwen4_gemm_mqv2_wmma_gfx11_bt.hip`, add a default-off final template bool, enabled only by the incumbent RW4 HCW wrapper and compile-gated to gfx1151. The eight lanes owning one token's eight channel-items share its four HC gates through ds_bpermute. Gate evaluations32→4 per token/WG, preserving the same hcw_gate instructions/rounding and raw gate bits. All lanes shuffle before ragged token/channel exits. Existing K16 WMMA order and HC BF16/add/mul/NaN operand order are unchanged. No new pass, allocation, copy, host/device handoff, workgroup barrier or public knob. RW8, DQ and other architectures retain incumbent behavior.

Three exploratory fresh-process C++ screens, seven alternating-order paired samples each,10s warmup per shape,20iterations/sample and null mirror (production-style). The5198 tail is1.85/2.16/2.37% faster;2048 also slightly faster. No power telemetry in that microprobe; use it as kernel evidence only. Both RW4 code objects declare256VGPR/36SGPR/18432LDS/zero scratch; HIP active blocks per MP stays6. Full native API uses compiled60-byte kernarg with existing64-byte host padding. Static entry instruction rows10727→10421 are not retired counters; all85 other entry instruction streams compare unchanged. gfx1100/1101/1102/1150 compile-only checks pass, with sharing disabled there; no other-arch hardware/performance claim.

## Correctness and actual runtime proof

54 frozen-parent/candidate/independent-RW8 oracle records cover96/127/128/129/2048/5198 rows and both BF16/F32 HC streams (including native-shaped timing checks). Mirror and every stream byte match; explicit finite, NaN-poison, prefix/suffix guards and actual written-state checks pass. Six changed-gate-input bare-kernel graph replays pass. CPU4096 provider mappings pass, separately from GPU proof.

Actual `qwen4_mq6_rows` API: scopeoff, two HC stream formats, six row counts, offsets0/1/17/256,48 cases, zero mismatches/errors/skips. Its8192 RW8 reference remains unchanged. Scopeoff matters because scopeon can pad route rows to8192. Initial API build omitted required lab and failed before execution; corrected documented feature build passes, old log retained.

Fresh baseline/candidate scoped Redline pass BF16/AR with splitK0 only there. Real q8/VMM AR/MTP service keeps production defaults: four processes/16 turns, full assistant/reasoning/tool-call/request transcripts byte-identical; decoded code summary, primes, merge sort and LRU inspected. Inputs21607/24/38/243, outputs97/43/156/48, allstop. MTP effective in every MTP response. Harness KV marker parser remains UNOBSERVED; Qwen4 logs independently show q8/VMM. Production HCW disallows recording/capture, so Redline is infrastructure/fallback evidence, not a claim that the modified eager HCW was captured.

Runtime cache source and loaded ELF snapshots are retained for both binaries. Both target instruction streams exactly equal the corresponding frozen standalone shader. Native trace confirms48 tail calls per arm,96 unchanged RW8 calls, no scratch.

## Native measurements

Same committed21582-token prompt MD5 b4d0b63cddcac872648ddf3cdd92cac2; model `/home/husrcf/.hipfire/models/qwen3.8-flash-next.mq4-xts`,125288544792B, MD5 be007fc3219e9f6cdb1d4dfa8380625f, model/PLE/overlay pins retained. q8/VMM,NPUauto,maxseq32768,PLE8GiB,cachesoff,DPM10s,outputbudget256. AR BAABBA and MTP ABBAAB:3 separate native daemon processes/arm/mode, one same warmup+one measurement per process. MTP actual256 observed, AR actual count remains CLI UNOBSERVED. Raw power/PPT/clock/thermal/memory objects retained. No remote CPU build/audit or profiler overlaps unprofiled timings.

| mode | baseline TTFT ms raw | candidate TTFT ms raw | TTFT median delta | baseline decode tok/s raw | candidate decode tok/s raw |
|---|---|---|---:|---|---|
| off | [9624.2, 9632.8, 9678.9] | [9549.6, 9630.1, 9624.8] | -0.08% | [33.7, 33.7, 33.7] | [33.7, 33.7, 33.7] |
| mtp | [9825.8, 9864.0, 9810.5] | [9852.2, 9815.0, 9897.6] | +0.27% | [53.0, 53.5, 53.5] | [53.4, 51.9, 51.0] |

No stable whole-model gain is claimed from overlapping samples. The original MTP median is3.0% lower for the candidate; it was investigated before admission rather than discarded as noise. Power/thermal/order variation remains a limitation; adoption of repeated-work removal is separate from proving a throughput gain.

### MTP loss investigation and separately declared controls

The primary six default MTP processes remain separate. Two subsequent diagnostic tee processes use the exact CLI request, each with one warmup+two measured requests. All six256-token piece streams and decoded texts are identical, text MD5 9e53b2f6f9cb985acddbc735d020aceb. Content was read: a coherent summary of the supplied code, ending at the expected token budget. Diagnostic baseline tau1.97/1.80/1.93 versus candidate2.11/1.66/1.87 shows intra-process schedule variation with identical greedy output. Tee timings are excluded from all default-path performance summaries.

MTP source explicitly measures AR/window walltime to choose draft budget and routes; the greedy acceptance comparison itself uses token equality. This establishes a timing-dependent scheduling mechanism, not a complete causal proof for the original loss. The source review and available trace limitations are retained; no fixed-schedule override was used to rescue headline timings.

Four further default-path fresh daemon processes were declared BAAB before running: candidate/baseline/baseline/candidate, same warmup1+measurement1. No proxy, profiler, extra trace or remote CPU build/audit. No performance/tau samples were filtered. Original6, control4 and all10 MTP samples follow separately.

| MTP sample set | baseline TTFT ms raw | candidate TTFT ms raw | baseline decode tok/s raw | candidate decode tok/s raw | decode median delta |
|---|---|---|---|---|---:|
| original6 | [9825.8, 9864.0, 9810.5] | [9852.2, 9815.0, 9897.6] | [53.0, 53.5, 53.5] | [53.4, 51.9, 51.0] | -2.99% |
| controls4 | [9817.4, 9852.0] | [9812.5, 9808.3] | [53.5, 53.5] | [53.5, 53.4] | -0.09% |
| all10 | [9825.8, 9864.0, 9810.5, 9817.4, 9852.0] | [9852.2, 9815.0, 9897.6, 9812.5, 9808.3] | [53.0, 53.5, 53.5, 53.5, 53.5] | [53.4, 51.9, 51.0, 53.5, 53.4] | -0.19% |

The adopted disposition follows inspection of the declared controls, raw spreads, tau and power; it does not establish statistical equivalence or a stable MTP speedup. Keep the primary negative median visible. Per-run power/clock/thermal/PPT telemetry is preserved in the receipt.

Separate instrumented cap1 native profile: tail48 calls325.096→313.973ms, median6.802→6.544ms; unchanged RW8 median10.082→10.035ms with a25.7ms cold/outlier baseline maximum retained. Do not attribute its sum shift to this source change. Both96-sample NPU traces have zero nonstartup visible NPU wait; initial visible waits103.583/117.319ms remain startup costs. These are diagnostic attribution and overlap checks, not unprofiled TTFT numbers.

## Prior rejected screen and continuation

Before this lever, isolate QSA LDS compaction18432→13312 bytes, same160VGPR and same4 active blocks/MP. OriginalPM/HIP/candidate bytes and27 changed-input graph replays pass, but three fresh native-shaped kernel screens reproduce~1% slower5198 tails while8K cells stay neutral. Retain it isolated; fewer reserved bytes did not produce more resident workgroups. Build/probe/capture-harness failures and the successful corrected synchronization are retained in the prior QSA report; no tolerance relaxed, no QSA production source/image changed.

Continue full PD work. Decode source/profile review finds shared FFN mathematically independent of routed experts until combine, but its Q8_0 GU/down GPU budget is only~35–41us/layer; AR's upper bound includes router/selector that must remain. Existing NPU prefill programs pad to at least256 rows and do not implement this few-row shared FFN. Exact decode ready/submit/full-FFN latency remains unknown. Prefill's~48ms chains and13–19us instrumented wait spans cannot substitute for those unknowns. Source review is retained, not an offload implementation or a speed claim.

The first independent Decode candidate is MTP routed QT44 GU R4 inverse tiling, preserving the current inverse map and four-chain numeric DAG; the existing routed GU profile is~5.3x the combined shared GU/down budget. More accumulators/LDS and disjoint routing may erase reuse, so test an isolated kernel before dispatch changes. Prefill RW16 HCW X-staging reuse remains another source-only hypothesis: VECS2 assertion,64KiB LDS and possible24→16 resident waves must be tested before implementation claims. Existing output semantics and NPU overlap remain required.

Full real GPU-free pre-commit CI passed on the final source. The attached CI log is retained in the receipt; hook fixture skips are separately qualified in post-commit evidence.

Research/raw evidence: `.codeinsight+research/halo-native-pd-20261021/rw4-hcw-cycle-v1`; adjacent compressed receipt embeds source/tool/report snapshots and artifact identities. No retired coherence gate used.
