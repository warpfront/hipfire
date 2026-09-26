# Qwen4 HC BF16 roundtrip chunking — rejected candidate measurement — 2026-09-19

**Lifecycle:** `historical`

**Disposition:** rejected. This is fixture-bound candidate evidence, not a
product performance claim, promotion decision, retained-replay admission, or
current baseline. The source change was restored after the valid primary A/B
failed the predeclared clear-win bar.

## Fixture and products

The host was AMD Strix Halo `gfx1151`, one visible device, HIP/ROCm `7.2.3`.
The model was `/home/bjoern/.hipfire/models/qwen3.8-flash-next.hfq`, SHA256
`7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`, MD5
`fda74d3760dc803e778e9b30a2fe0ebd`. The primary prompt was
`/home/bjoern/hipfire/benchmarks/prompts/glimmer_prefill_256.txt`, 291 tokens,
SHA256 `184b6e9ab08b3d07f9cebf2ce9506cfe34d8d2ffc03c910b105e13bbd6ac6a8e`,
MD5 `973900074bfd15d4adeeecdff3359082`.

Primary A was the frozen baseline `hipfire` SHA256
`ced9e89879d4cf8e8d910bf18737e5a0c46f44612cc226ec5e5641f93887f9f1` and
`daemon` SHA256 `41a493a4b80998b50d782872a70ee816868b40ebe544c7eea027481aa9f47bc5`.
Primary B was the candidate `hipfire` SHA256
`972c1349419101fbeeb0ab4ec67f22ba4d448107ec4846d3be51562ad62a77f6` and
`daemon` SHA256 `42c02389678c348b25c8715959909f62d16802765dbbd08c557b77a4a3be86af`.
The complete predeclared identity and protocol are in
[`manifest.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/perf-v3-abbaab-20260919/manifest.json).

## Candidate source and profile evidence

The candidate touched only shared
`crates/hipfire-dispatch/src/pipeline/layer_ops.rs`: its three row-wise
`bf16_roundtrip_f32` calls in `execute_hyper_read` were replaced with
contiguous chunks bounded by the existing caller-owned BF16 scratch tensor.
No kernel, scratch allocation, precision/reduction, architecture gate, GDN,
replay, PM4, or QT53 policy changed. Candidate source and the exact one-file
diff remain in
[`candidate-source-snapshot.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/candidate-source-snapshot.json)
and
[`candidate-git-diff.patch`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/candidate-git-diff.patch).

The candidate ROCTX profile established a real natural-path launch reduction:
`observed_target_natural_launches=4656`, while scalar `grid512` remained
`84681`. A host-only dirty-scratch/tail/row-crossing simulation was exact for
rows `1`, `35`, and `128`:
[`chunk-tail-row-crossing-probe.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/chunk-tail-row-crossing-probe.json).
The candidate GPU oracle was exact (`exact_bits=true`, `state_exact=true`,
`ids_equal=true`), and the AR/native-MTP Paris smoke passed; both are
correctness/integration evidence only:
[`oracle/result.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/oracle/result.json)
and
[`serve-smoke-20260919-paris/summary.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/serve-smoke-20260919-paris/summary.json).

For the actual natural ROCTX marker, raw `trace_kernel_trace.csv` coverage
was 1,143,870 data rows overall and 107,195 dispatches across 39 symbols
inside the marker. All-symbol device-duration sum was 4,715.890965 ms versus
7,142.418151 ms marker wall time. The top five were:

| kernel | calls | device duration |
|---|---:|---:|
| `gemm_mq4g256v2_moe_grouped_top10_simt` | 144 | 1,109.758875 ms |
| `gemm_mq4g128v2_moe_grouped_top10_multirow_gfx1151` | 144 | 836.797373 ms |
| `gemm_bf16_xf32_multirow` | 2,218 | 740.026696 ms |
| `gemm_bf16_xf32_multirow_n8_gfx1151` | 100 | 718.992121 ms |
| `gated_delta_step_shared_norm128_gfx1151` | 10,476 | 493.483439 ms |

The ranking uses every traced kernel symbol inside the actual ROCTX bounds,
not the target-only hot-symbol filter. Durable machine-readable evidence is
[`natural-marker-all-symbol-top5.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/metadata/natural-marker-all-symbol-top5.json).
This is profile-only attribution; it is not native throughput attribution.

## Native primary A/B

Each arm used a private `HOME`/config with `max_seq=2048`,
`HIPFIRE_DPM_WARMUP_SECS=10`, the cooperative
`flock -w 3600 /tmp/hipfire-gpu.lock`, q8 contiguous KV, speculation off,
noslots, stateless workload, and the same native `--runs 11` command.  The
standard non-matrix native bench ignores `--warmups`; its one internal
`Hello` warmup is not a matching 291-token warmup.  The declared protocol
discarded zero-based samples `0..9` and retained only sample `10`; raw sample
arrays are preserved at length 11 and no CLI aggregate median was used.  The
cooperative lock serialized the arms, but unrelated GPU clients were
preserved, so this is not an uncontended measurement.  The arm order was
manually executed ABBAAB: `A1, B1, B2, A2, A3, B3`.

The durable summary is
[`primary-summary.json`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/perf-v3-abbaab-20260919/primary-summary.json),
with raw arm directories beside it. Retained sample-10 group medians were:

| measure | A median | B median | B minus A |
|---|---:|---:|---:|
| prefill | 51.4 tok/s | 53.4 tok/s | +2.0 tok/s (+3.9%) |
| decode | 13.2 tok/s | 13.2 tok/s | 0.0 tok/s |
| TTFT | 5,661.8 ms | 5,451.2 ms | -210.6 ms |

The primary prefill delta was smaller than the maximum observed arm spread
(`2.2 tok/s`), so it is **no clear win**. Decode was equal, not a demonstrated
slowdown or speedup. The user target of `>500 tok/s` prefill and `>=25 tok/s`
decode was not met. The invalid `v1` configuration run and cold exploratory
`v2` timings are preserved but excluded from this decision.

The separately run existing France fixture was 24 tokens and emitted the
bench warning that its `prefill_tok_s` measures launch overhead rather than
prefill throughput. Its A/B records are retained under
[`second-prompt-20260919`](../../.codeinsight+research/qwen4/current-pp-profile-20260919-v1/candidate/perf-v3-abbaab-20260919/second-prompt-20260919/)
as exploratory warmup/launch evidence only; it cannot repair the primary bar
or serve as transfer evidence.

## Final source and host boundary

The candidate source was rejected and surgically restored to the exact
pre-experiment SHA256
`30984594a8bd6055e8deba1e487422a43a86292f514b0cbcc8e697043d06b496`.
Candidate binaries, traces, raw sample arrays, oracle/smoke records, invalid
harness artifacts, and the corrected warmup protocol remain preserved for
archaeology. After restore, `cargo test -p hipfire-dispatch --lib` passed
285 tests (one ignored), scoped dispatch clippy completed with existing
repository warnings, and scoped rustfmt left the restored source hash
unchanged.

This checkpoint records launch reduction, numerical parity, and warmup/harness
discovery only. It makes no native performance, 500-token/s, decode-target,
product, ISA, retained replay, or cross-architecture claim.
