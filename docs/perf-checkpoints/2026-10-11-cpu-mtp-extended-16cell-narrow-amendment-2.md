# Amendment 2 — CPU versus PCIe MoE numerical divergence — 2026-10-11

**Lifecycle:** `historical`  
**Authority:** Fixture-bound measured evidence, not a current default, admission decision or shipped performance claim.  
**Disposition:** No demonstrated CPU arithmetic correctness defect in the measured graded expert paths; no numerical fix made.

Amends [amendment 1](2026-10-11-cpu-mtp-extended-16cell-narrow-amendment-1.md) and supplements the [original checkpoint](2026-10-11-cpu-mtp-extended-16cell-narrow.md). Both remain unchanged.

## Scope and interpretation

This investigation compares **CPU AR versus PCIe AR**, with speculation disabled. It does not investigate MTP drafting, verification or acceptance. It neither resolves nor retracts amendment 1's separate **same-backend PCIe AR versus PCIe MTP** math-prompt divergence.

The measured local arithmetic differences satisfy the existing expert-operation numerical tolerance. Small floating-point differences are amplified through quantized recurrent state and subsequent routing changes into meaningfully different generated text. Both observed prose responses were coherent; that does **not** establish factual correctness of every generated statement or correctness across all models, prompts and formats.

No arithmetic fix is justified by the measurements below. No candidate mitigation was implemented or benchmarked, so this record does **not** establish that reducing divergence would necessarily sacrifice significant performance. Guaranteeing CPU/GPU token identity would be a stronger contract than the documented local numerical tolerance.

## Fixture and method

- GPU: RX 9070 XT, gfx1201; HIP 7.2. CPU: Ryzen 7800X3D.
- Model: `/home/avery/.hipfire/models/qwen3.6-35b-a3b.mq4p`, MD5 `f9b5b13eb24ecbe4b39f3a3f4a297645`.
- Graded expert formats: qt13 MQ4G256, qt15 MQ6G256, qt20 MQ3G256-Lloyd.
- Expert budget 24: first 16 layers' routed expert banks host-mapped, 96 owner tensors, 6.41 GiB. No whole-layer spill. Same weights and placement in both arms.
- Greedy GPU argmax, speculation off, graph/replay graph disabled, `max_seq=4096`, Q8 recurrent state with default error feedback, Q8 KV.
- Numerical probe uses the existing Qwen35 forward APIs, raw prompt tokenization and legacy KV allocation. Native performance measurements use `hipfire bench`, its prompt handling and VMM KV. These are separate instruments; do not compare their timing or assume their prompt token counts are interchangeable.
- Prose prompt: `Explain why cities develop near rivers, in three short paragraphs.` in a closed-thinking ChatML envelope, 25 prompt tokens; exact bytes retained in `.codeinsight+research/offload-numerics/prose.txt`, MD5 `fcb445a5723a7fb6949e9c4343fde391`.
- Performance prompt: committed `benchmarks/prompts/merge_sort_thinking_off.txt`, MD5 `253c7ac50857fe6d0e10fb0d2c5e35c0`.
- Frozen native CLI MD5 `c86e6a5123ee17db8072d304584af17a`; daemon MD5 `fd635fbfe94ed68fb02815d6ace11a25`. These identities, not a commit alone, identify the measured binaries.
- Final numerical probe binary MD5 `7e8c795bd900dbb8e7afdfdd5ade3e54`. Diagnostic readbacks synchronize; no probe timing is a performance result.

Compare matching generated prefixes only, before the first differing token. Reconstruct CPU intermediates from the recorded PCIe input, selected experts and their exact weight bytes to distinguish local arithmetic error from already-different model state. GPU scratch intermediates on the CPU-deferred arm are not treated as CPU results. Independently accumulate decoded quantized codes, scale and zero terms in f64; oracle outputs are rounded to f32 for array comparison. Additionally isolate DN using the identical recorded PCIe rotated activation and SiLU using identical GPU GU values.

Bounded state snapshots and earlier captures produced bit-identical logits within each backend at the captured divergent position. Temporary lab hooks, example, dependency and lockfile changes were removed after measurement; source and additive hook patch are retained locally for reproduction.

## Earliest numerical divergence

Prefill results and all 30 recurrent-state snapshots are equal between arms. During the first CPU-executed decode, the first measured difference is **layer 0 routed GU**, after consuming the first generated token and predicting the second. Its input, residual before the expert operation, router scores, selected expert IDs and routing weights are bit-identical.

Selected IDs: `[238, 43, 120, 147, 112, 148, 254, 105]`.

| Stage, first CPU-executed decode / layer 0 | Maximum absolute error | Peak-relative error | RMS error |
| --- | ---: | ---: | ---: |
| Gate projection | 7.152557e-7 | 1.563940e-7 | 6.587982e-8 |
| Up projection | 2.682209e-7 | 2.206324e-7 | 4.410518e-8 |
| SiLU, including upstream GU differences | 1.192093e-7 | 1.784125e-7 | 8.554922e-9 |
| FWHT-rotated activation | 4.516914e-8 | 3.088878e-7 | 9.150127e-9 |
| Down projection, chained | 1.769513e-8 | 4.035399e-7 | 2.758819e-9 |
| Weighted expert aggregate | 3.958121e-9 | 3.020350e-7 | 1.206188e-9 |
| Final layer-0 residual | 1.490116e-8 | 2.550917e-8 | 1.537732e-9 |

Peak-relative error is `max(abs(cpu-pcie)) / max(abs(pcie))`; RMS is `sqrt(mean((cpu-pcie)^2))`. Aggregate comparison re-sums the captured CPU DN outputs and compares against an independently executed GPU combine over its captured DN outputs.

With identical GU inputs, isolated SiLU maximum error is `1.117587e-8`, RMS `4.436005e-10`. With identical GPU rotated inputs, isolated CPU/GPU DN maximum error is `1.396984e-8`, peak-relative `3.185842e-7`, RMS `1.707460e-9`.

Across the first decode's selected host experts, maximum GU CPU-versus-f64 peak-relative errors were MQ4 `4.882909e-7`, MQ6 `3.642235e-7`, MQ3-Lloyd `1.847255e-7`. No format-specific scale, codebook or rotation inconsistency was demonstrated.

## First differing output token and amplification

The first differing generated token is **103, one-based** (index 102). The preceding 102 token IDs are equal. PCIe chooses ID 888, ` over`; CPU chooses ID 11, `,`.

PCIe continues `moving heavy goods and people over long distances`; CPU continues `moving heavy goods and people, connecting inland regions to coastal ports and global markets`. This is substantive continuation divergence, not whitespace or a post-terminator artifact.

| Candidate at output index 102 | PCIe logit | CPU logit |
| --- | ---: | ---: |
| ` over`, ID 888 | 23.383314133 | 23.255716324 |
| `,`, ID 11 | 23.227600098 | 23.258789063 |

Top-1 versus top-2 margin: PCIe `0.155714035`, CPU `0.003072739`. Full-vocabulary maximum absolute error is `0.193050861`, peak-relative `0.008255924`, RMS `0.033977256`, relative RMS `0.010996479`.

The full-logit error is not merely last-bit noise by this position. It must not be judged as if both models still had identical intermediate state:

- After the first CPU decode, DeltaNet layer 1 has one differing Q8 code; decoded-state maximum error `7.911570e-6`. Layer 0 recurrent state remains equal.
- At output index 102, DeltaNet layer 1 has 12,837 differing Q8 codes; decoded-state maximum error `0.059025656`.
- At that position, layer 0 input and routing are still equal; later routing differs. For example, layer 12 replaces expert 71 with expert 233. PCIe's router-score gap between those candidates is `0.008271217`, and the perturbed CPU scores reverse their order.
- Final layer-39 residual maximum error is `0.014074653`, RMS `0.004384762`.
- Re-running CPU arithmetic on each host layer's **exact PCIe inputs** at this same position still gives worst chained peak-relative error `6.799252e-7`, at DN in layer 7. Isolated DN worst peak-relative error is `5.408496e-7`.

These measurements distinguish small local arithmetic differences from cumulative recurrent/routing divergence. Differing state-code counts alone are not evidence of an encoding defect; the identical-input expert references are the arithmetic check.

## Numerical cause and intended contract

Relevant production implementations are `crates/hipfire-cpu/src/simd/x86.rs`, `crates/hipfire-dispatch/src/cpu_exec.rs`, and the mixed GPU kernels `gemv_mixed_moe_gate_up_k8_indexed_batched.hip` and `gemv_mixed_moe_down_k8_indexed_batched_expanded.hip`.

- For affine formats, CPU SIMD factors the dot into `scale * sum(code*x) + zero * sum(x)`. GPU mixed kernels evaluate distributed `(scale*code + zero)*x` terms. Algebraically equivalent expressions round differently in f32.
- SIMD FMA chains and horizontal reductions differ from GPU lane accumulation and shuffle reductions. MQ3-Lloyd shares the codebook semantics; reduction order still differs.
- CPU and GPU SiLU use the same expression, but host exponential and device `expf` can differ slightly. The isolated measurement bounds that contribution here.
- CPU successively adds weighted experts into the residual. GPU computes the weighted sum first, then adds it once to the residual. This changes association, not the mathematical operation.

The design document `docs/plans/partial-gpu-offload-design.md` explicitly does not require CPU/GPU bit identity. `crates/hipfire-arch-qwen35/tests/gpu_moe_cpu_parity.rs` sets `REL_TOL=5e-3`, normalized by maximum reference magnitude. Measured local errors are substantially below that bound. This is **not** a validated bound on full-model logit drift, generated-token identity or universal answer quality.

**Conclusion:** expected f32 differences, amplified through Q8 recurrent state and later discrete routing decisions, explain the measured cross-backend divergence. No CPU arithmetic correctness defect was demonstrated. No quantization, sampling, token selection, model weights, SIMD or MTP behavior was changed.

## Correctness validation

- `cargo test --release -p hipfire-cpu --lib`: **37 passed**.
- `cargo test --release -p hipfire-dispatch --lib moe_splice_tests -- --nocapture`: **4 passed**.
- After removing temporary hooks: `cargo test --release -p hipfire-arch-qwen35 --features lab --locked --test gpu_moe_cpu_parity graded -- --ignored --test-threads=1 --nocapture`, with the graded fixture above and GPU lock: **3 passed, 0 failed, 1 filtered**.
- Graded tests cover real qt13/15/20 bytes in device-resident and host-mapped placements; chained four-row k8 execution; CPU three-row k4 residual parity; and stream reuse with rows `3 -> 2 -> 3 -> 1`. Chained four-row peak-relative errors: gate `3.467e-7`, SiLU+rotation `6.020e-7`, DN `7.399e-7`. CPU multirow residual reference error is zero.
- An initial all-four-test invocation had three passes and a separate qt44 fixture-lookup failure: the supplied uniform fixture had no qt44 GU tensor. That test is not counted as passed; its known fixture failure was not rerun in the final graded invocation.
- Initial state-snapshot instrumentation used F32 tensor metadata to infer a Q8 byte extent. Existing transfer bounds rejected it before copying; those two jobs exited 101. The probe was corrected to use exact Q8/Q4 buffer extents and rerun successfully. Failed snapshots are not evidence.
- No fix was made, and no MTP drafting/verification/acceptance investigation was performed. Multirow CPU baseline was exercised; no new MTP compatibility claim is made.

## Timing-disabled native performance measurements

Delegated to `hipfire-benchmark-runner`, using the existing `hipfire bench` infrastructure, not a replacement benchmark harness. Each cell uses a fresh native process and the frozen binaries above:

`--spec off --backend noslots --workload stateless --runs 1 --warmups 3 --max-tokens 256 --max-seq 4096 --kv-mode q8 --kv-v q8 --kv-backend vmm --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt --json`

Timing/diagnostic traces are unset. GPU jobs share `/tmp/hipfire-gpu.lock`. The clean repeat follows completion of parent builds; all six repeat cells completed successfully with the same placement and speculation off.

| Backend | Original decode samples, tok/s | Clean-repeat decode samples, tok/s | Clean-repeat median, tok/s |
| --- | --- | --- | ---: |
| CPU | 59.7, 8.2, 56.2 | 60.2, 58.4, 60.1 | 60.1 |
| PCIe | 60.5, 60.5, 60.7 | 60.5, 60.2, 60.6 | 60.5 |

The original CPU 8.2 tok/s outlier is retained, not discarded. Low clocks, temperature and host contention were observed or plausible context; no isolated causal attribution is established. Shared GPU locking rules out overlap between these locked GPU jobs. The clean-repeat median difference is within observed CPU spread. **No before/after regression or performance-win claim:** no numerical fix or production behavior change was made.

## Evidence and durability limits

Local discovery pointers, not durable published artifacts:

- `.codeinsight+research/offload-numerics/`: `fixture.json`, `measurement-method.json`, `first-numerical-divergence.json`, `early-format-errors.json`, `divergent-token-metrics.json`, `isolated-stage-and-state-metrics.json`, `conclusion.json`; backend reports and intermediate arrays; bounded recurrent-state snapshots; `diagnostic-source/` with source and additive `hooks.patch`.
- `.codeinsight+research/offload-numerics-perf-20261011/`: original full native bench JSON, stdout and logs; `clean-repeat/` contains all six repeat reports, manifest and runner protocol.

Numbers in this checkpoint are historical evidence only under the pinned fixture and methods. Reproduction requires the exact prompt bytes, model and binary identities and local receipts above; local paths do not become durable evidence merely by being cited here.
