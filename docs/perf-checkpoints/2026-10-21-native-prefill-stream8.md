# Native gfx1151 streamed NT8 prefill gate/up — 2026-10-21

Parent `6fb608dc9754c8c859dcb9daf01c78a7a56fc8bb`, branch `experiment/halo-native-pd-20261021`; source dirty at build. Local commit only; full PD goal remains active.

## Decision and scope

Adopt only gate/up for gfx1151, VRAM MQ4 experts, M=1280/K=2560/top10/E=512 and source rows 7168..8192. New separate native image processes eight pad16 tiles per weight stream using two rotating operand buffers. VGPR stays 208; fixed LDS 8 -> 16 KiB. Existing 64-byte ABI, block128 and grid, quantization, ascending K128 fold, BF16/RNE and SwiGLU DAG are retained. No extra pass, allocation, copy or synchronization. Smaller banks, host weights, all other shapes/arches and down retain their incumbent entries. Existing PM=0 kill switch applies.

Adoption follows measured work removal, exactness and absence of sustained native loss. No stable whole-model speedup is claimed from overlapping timing ranges.

Retained research: `/home/husrcf/Code/hipfire-native-pd-20261021/.codeinsight+research/halo-native-pd-20261021/prefill-chains-cycle-v1`. `PLAN.md` records the original chain screen; `NATIVE-ADMISSION-PLAN.md` extends it before native timing. Adjacent compressed receipt embeds source/tool snapshots and raw reports and pins validation artifacts.

## Exact identities

Common command source metadata describes the candidate workspace at job creation. Baseline is the unchanged frozen prior binary equivalent to parent 6fb608dc9754; see BINARY-IDENTITY-SUPPLEMENT.md and the prior native-qt53-k640 receipt for its actual build provenance.

- baseline: `/home/husrcf/Code/hipfire-native-pd-20261021/.codeinsight+research/halo-native-pd-20261021/mtp-down-cycle-v1/k640-candidate-v1/pd-daemon`, MD5 `4c67acc7037ef7490c986062e7856854`.
- candidate: `/home/husrcf/Code/hipfire-native-pd-20261021/.codeinsight+research/halo-native-pd-20261021/prefill-chains-cycle-v1/stream8-candidate-v1/pd-daemon`, MD5 `a294a8c9c87ce2ef9311f6fadf425bb5`.
- oracle: `/home/husrcf/Code/hipfire-native-pd-20261021/.codeinsight+research/halo-native-pd-20261021/prefill-chains-cycle-v1/stream8-candidate-v1/qwen4_moe_sym`, MD5 `12594dfc165240821d9bc303c08f5afd`.

Model `/home/husrcf/.hipfire/models/qwen3.8-flash-next.mq4-xts`, 125288544792 bytes, MD5 `be007fc3219e9f6cdb1d4dfa8380625f`; model/PLE/overlay pins retained. Device gfx1151 Strix Halo; same ROCm HIP7.16 fixture as parent. Prompt `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`, MD5 `b4d0b63cddcac872648ddf3cdd92cac2`, bench observes 21582 input tokens.

## Rejected and retained screens

- Independent NT4 chains 2/4: exact but no stable envelope gain; four chains especially hurt down.
- Keeping original prefetch removes experimental dead-tile guards/drains but still gives no stable gain. The clone initially failed to build due to missing shared `rocm.rs`; fixed in retained v2.
- First stream8 emission fails loop-ledger back-edge identity. Prime only the required first-weight registers at loop entry; preserve operand prefetch. Do not weaken verifier or drain all loads.
- Four operand buffers use 240 VGPR and lose to two buffers.
- NT8 down regresses skewed 5198/8192 shapes; do not adopt.
- Smaller skewed GU banks 4096/4672 regress slightly; 5198 is neutral. Restrict default selector to the measured larger range.
- Finalizer v1 mixed the job-count scalar with binary dictionaries while rendering the report; partial receipt quarantined, fixed in v2. No hardware result was rerun or changed.
- First formal oracle build omitted its required `lab` feature, failed before tests, then passed with the documented feature.
- Initial exploratory command metadata recorded intended env overrides that were not applied explicitly; identity supplement records the discrepancy. Later named runners apply and record effective env.

| Two-buffer GU screen | original ms | stream8 ms | delta |
|---|---:|---:|---:|
| 7168 all | 10.3323 | 9.5214 | -7.85% |
| 7168 skew | 9.8825 | 9.3696 | -5.19% |
| 8192 all | 11.0804 | 10.1838 | -8.09% |
| 8192 skew | 10.8948 | 10.2321 | -6.08% |
| 8191 skew | 10.7764 | 10.2212 | -5.15% |
| 5198 skew | 7.7227 | 7.6941 | -0.37% |
| 4672 skew | 7.2982 | 7.3631 | +0.89% |
| 4096 skew | 6.8884 | 6.9385 | +0.73% |

Synthetic screens retain five raw palindromic samples and explicit 10s GPU warmup. They are kernel evidence, not native TTFT admission. Effective HIP reference flag is `-DIU4_A4_CANDIDATES=2`.

## Correctness and runtime validation

- Native build/parse-back, schema5 shape/resource contract, all-lane LDS bounds and obligation-free M7 lift/reemit pass; no spill/private memory.
- Twelve emitter tests pass, including unchanged incumbent module bytes and exact fresh native bundle equality. Selector boundary and image 64-byte ABI tests pass.
- Original canonical numerical oracle: skew7168, all8192 and graph/skew8192 pass. All full-buffer PM NT1/NT4 and independent HIP NT1/NT4 anchors are byte-identical, CPU K128 folds sample 26 tiles per case, poison/padding/guards clean, header negative controls refuse. Graph covers five replays and changed-input rounds.
- Fresh baseline and candidate Redline capture/replay pass on retained BF16/AR equal-math route, `HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0` only there. The parent documents identical default split-K arithmetic differences. This gate is decode infrastructure coverage, not evidence that the new long-prefill kernel is captured. Real q8/MTP tests keep production defaults.
- Four native serve processes, 16 turns, both AR/MTP byte-identical; actual inputs21607/24/38/243, outputs97/43/156/48, all stop. Decoded text inspected: codebase summary, prime list, merge sort and LRU code are coherent. Qwen4 logs and IPC independently show q8/VMM; serve harness KV marker parser remains UNOBSERVED, not promoted into a pass.
- Full pre-commit-ci with real cargo-deny passes. No retired coherence/quality gate used.

## Matched native GPU-NPU measurements

Separate processes, MTP ABBAAB (3/arm), AR ABBA (2/arm). One identical warmup + one measurement each, fixed q8/VMM/NPUauto, max_seq32768, PLE8GiB, session caches off, DPM10s. Output budget256; actual MTP256 observed. AR actual output count is not exposed by this CLI and stays UNOBSERVED. Every raw power/clock/throttle/memory object is retained. No profiler or remote CPU build/audit overlaps timing.

| Mode | baseline TTFT ms (raw) | candidate TTFT ms (raw) | median delta | baseline decode tok/s | candidate decode tok/s |
|---|---|---|---:|---|---|
| mtp | [9899.4, 9979.7, 10029.0] | [9919.4, 9948.7, 9968.3] | -0.31% | [51.9, 54.2, 52.5] | [51.9, 53.7, 54.8] |
| off | [9720.3, 9717.5] | [9674.4, 9687.5] | -0.39% | [33.5, 33.7] | [33.7, 33.6] |

Power/thermal qualification: all arms are PPT-limited in this session. Raw fast/slow-PPT residency, clocks, temperatures and identical memory carve-out are retained; compare only these matched conditions. Thermal/ordering variation further limits any small speedup claim.

## Actual native kernel attribution

Separate cap1 normal-IPC profiler runs are diagnostic only. Same prompt bytes and configuration across arms. Normal commit/EOF shutdown; no lifetime shim or signal shutdown.

| arm | old GU calls / total ms | streamed GU calls / total ms | original down calls / total ms |
|---|---:|---:|---:|
| baseline | 144 / 1335.033 | 0 / 0.000 | 144 / 896.106 |
| candidate | 48 / 349.248 | 96 / 941.956 | 144 / 887.158 |

The new GU runs on eligible large banks and incumbent GU remains for short tails; down is unchanged. NPU plan/pair samples are retained in the receipt. Profiler durations include instrumentation and are not substituted for unprofiled TTFT.

## Continue from here

Keep native prefill/decode collaboration active. Investigate the remaining GPU MoE/down and output projections using actual native profiles and overlap headroom. Larger GPU improvements can expose NPU wait; check the full pipeline rather than summing isolated timings. No return to the full NPU drafter direction.
