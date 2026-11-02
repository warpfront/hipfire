# Qwen4 HC rows/grid-Y batching — final scoped measurement — 2026-09-19

**Lifecycle:** `historical`

**Disposition:** final fixture-bound evidence for the conservative HC rows/grid-Y
change on `gfx1151`. This record is not a product promotion, a throughput
claim, or an admission decision. It supersedes neither the 2026-09-18
measurement record nor the immutable N8 freeze; it records the fresh candidate
and its boundaries.

## Fixture identity

The frozen-N8 A/B and current HC candidate were measured on AMD Strix Halo,
`gfx1151`, with one visible device and ROCm `7.2.3` (HIP
`7.2.53211-9999`). The exact model artifact was
`/home/bjoern/.hipfire/models/qwen3.8-flash-next.hfq`, SHA256
`7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`, MD5
`fda74d3760dc803e778e9b30a2fe0ebd`. The benchmark prompt was
`/home/bjoern/hipfire/benchmarks/prompts/glimmer_prefill_256.txt` (291
tokens), SHA256
`184b6e9ab08b3d07f9cebf2ce9506cfe34d8d2ffc03c910b105e13bbd6ac6a8e`, MD5
`973900074bfd15d4adeeecdff3359082`.

The model and prompt identities are independently recorded in the
[verified fixture manifest](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/bench-summary.json).

The frozen-N8 A product binaries were `hipfire` SHA256
`a24d158b204fa5694f76f4a43616a5270990e8916a08728d5e09b18003f2fec1`, MD5
`a3890ae901c9ebadead01ddbe970d7a5`, and `daemon` SHA256
`dc57a84a0540bb19e271ee8c558f2aa665198afc43a1360aced84c90d30d1a12`, MD5
`262a0ad772decbedd0dccc9a1f992008`. The current HC B binaries were
`target/release/hipfire` SHA256
`d8cdc49e26103ff9efcf8261c46d911e2d7a81c09aa3fb767cbc46de5d3af580`, MD5
`14265eeae83f6c0b9b4d6eb2cf5b851e`, and `target/release/daemon` SHA256
`9ef6cb8a9dea3dea766b443fd17adb0ec395ad9fa2809e61422d4d7736089c91`, MD5
`e450bf489878e637b4ce40e529c6c2e9`.

The binary MD5 values above were computed from the product paths whose SHA256
values match the frozen and rebuilt manifests.

The separate deterministic Paris correctness prompt was
`candidate-hc-batch/smoke_prompt_capital_of_france.txt`, MD5
`98555b6c6b26fb027cd51af29992b3e8`; its AR/MTP result is linked below.

These identities are also captured in the [final freeze
manifest](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/hc-rows-gridy-20260919-v3/manifest.json).

## Change and numerical boundary

The candidate changes only the HC row plumbing and the shared Qwen4 pipeline:

- `kernels/src/qwen4_ops.hip` — SHA256
  `c8a3d64afe941b3e410ef4e84b38a52473b551a5b6282dffdcb8926db877b30e`.
- `crates/rdna-compute/src/qwen4.rs` — SHA256
  `f8619106b0849ecb3c941731789947c3a273f06ffbc9d70e5a99c8af67184b98`.
- `crates/hipfire-dispatch/src/pipeline/qwen4_program.rs` — SHA256
  `49a0ba8a6581ad9d1efe77dbf3d8868f3a2dce1e42714f8c9b8611823da4bc7d`.

Rows=1 keeps the prior launch shape and arithmetic. Natural final chunks use
one row-batched HC launch with grid-Y 128, 128, and 35. Reduction order,
per-row recurrence, BF16 boundaries, and scratch ownership were not redesigned.

The focused primitive probe exercised rows `[1, 2, 35, 128]`; norm,
read-projected, and write comparisons had zero mismatches for every case:
[`hc-batch-probe.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/candidate-hc-batch/hc-batch-probe.json).

## Explicit marker evidence

The same ROCTX marker template was used for both the frozen N8 profile and the
candidate. Heuristic phase segmentation is withdrawn and is not used here.
The exact natural-marker HC counts are:

| route | HC norm | HC read-projected | HC write | total HC calls |
|---|---:|---:|---:|---:|
| frozen N8 natural marker | 56,163 | 28,227 | 27,936 | **112,326** |
| current HC natural marker | 579 | 291 | 288 | **1,158** |

The frozen row is from
`candidate-n8-gfx1151/phase-marker-evidence.json`; the candidate row and
actual grid-Y distributions are in
[`hc-marker-evidence.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/candidate-hc-batch/hc-marker-evidence.json).
The candidate trace shows:

- scalar prefill and scalar continuation: HC grid-Y is `1` for every HC
  dispatch;
- batched continuation: HC grid-Y is `1` for every HC dispatch;
- natural chunks: norm is Y=128 for 386 dispatches and Y=35 for 193;
  read-projected is Y=128 for 194 and Y=35 for 97; write is Y=128 for 192
  and Y=35 for 96.

No `109024` count and no earlier heuristic phase split is valid evidence.

## Correctness evidence

The fresh greedy-16 scalar-vs-natural comparison passed exact state, logits,
and token-ID checks. The cross-version comparison across the immutable original and
frozen N8 also passed with zero step mismatches:
[`oracle-parity-evidence-20260919.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/candidate-hc-batch/oracle-parity-evidence-20260919.json).

A separate short deterministic correctness fixture used the same model and
q8-contiguous settings, explicit `max_think_tokens=1`, and a recorded
capital-of-France prompt. Both ordinary AR and native MTP returned `Paris`,
finished normally, emitted content, saw `done`, and had no stream error or
runaway. MTP was active with `tau=1.0` and one cycle. The corrected semantic
record is [`corrected-validation.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/candidate-hc-batch/serve-smoke-20260919-032000-paris/corrected-validation.json);
the original raw summary is preserved and records the throwaway aggregator's
pre-fix `smoke-failed` status.

This correctness fixture is not a performance measurement.

## Fresh product A/B boundary

The rebuilt product manifest is
[`manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/candidate-hc-batch/product-rebuilt-20260919/manifest.json).
The fresh candidate product hashes are:

- `target/release/hipfire`:
  `d8cdc49e26103ff9efcf8261c46d911e2d7a81c09aa3fb767cbc46de5d3af580`;
- `target/release/daemon`:
  `9ef6cb8a9dea3dea766b443fd17adb0ec395ad9fa2809e61422d4d7736089c91`.

Both differ from the frozen N8 product hashes. The preflight-gated ABBAAB order
was `A1, B1, B2, A2, A3, B3`, with the byte-identical 291-token prompt,
`max_tokens=16`, ten warmups, q8 contiguous KV, speculation off, noslots,
and stateless workload. Raw evidence is in
[`metrics-summary.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/candidate-hc-batch/abbaab-20260919-024719-fresh/metrics-summary.json).

| route | prefill samples (tok/s) | prefill median | decode samples (tok/s) | decode median | TTFT median (ms) |
|---|---|---:|---|---:|---:|
| frozen N8 A | 26.4, 26.0, 24.3 | 26.0 | 8.9, 8.8, 5.5 | 8.8 | 11,183.3 |
| current HC B | 28.8, 27.5, 30.5 | 28.8 | 8.8, 8.9, 9.5 | 8.9 | 10,114.2 |

Median B/A ratios are prefill `1.107692`, decode `1.011364`, and TTFT
`0.904402`. The approximately 1.1% decode difference is not a meaningful
decode-win claim. User thresholds `prefill > 500 tok/s` and `decode >= 25
tok/s` are both **not met**. Both threshold acceptances remain open/blocked;
further uncontended profiling is a prerequisite. This record does not assign
the gap to contention and does not promise that an idle GPU would solve it.

The earlier ABBAAB run is retained but explicitly invalid because its candidate
binary matched the frozen N8 binary; it must not be cited.

## Host gates

The fresh release build completed under
`flock -w 3600 /tmp/hipfire-build.lock` with
`cargo build --release --workspace --all-targets --locked` and produced the
hashes above. The final cached build log is
[`final-hc-build-release-workspace-all-targets-locked-20260919.log`](../../.codeinsight+research/qwen4/shared-token-prefill/final-verification/final-hc-build-release-workspace-all-targets-locked-20260919.log).
Final host checks then completed under the same build lock:

- `RUST_TEST_THREADS=1 cargo test --lib --workspace --locked` — exit 0;
  log: [`final-hc-test-lib-workspace-locked-20260919.log`](../../.codeinsight+research/qwen4/shared-token-prefill/final-verification/final-hc-test-lib-workspace-locked-20260919.log).
- `cargo clippy --workspace --all-targets --locked` — exit 0 with the
  repository's existing warnings; no warning suppression — log:
  [`final-hc-clippy-workspace-all-targets-locked-20260919.log`](../../.codeinsight+research/qwen4/shared-token-prefill/final-verification/final-hc-clippy-workspace-all-targets-locked-20260919.log).
- `bash scripts/ci-rustfmt-changed.sh` plus explicit checks of the two
  untracked Qwen4 pipeline files — exit 0; log:
  [`final-hc-scoped-rustfmt-20260919.log`](../../.codeinsight+research/qwen4/shared-token-prefill/final-verification/final-hc-scoped-rustfmt-20260919.log).

The immutable final freeze is
[`hc-rows-gridy-20260919-v3/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/hc-rows-gridy-20260919-v3/manifest.json).

## Explicit non-claims and blockers

This checkpoint does not claim either target threshold, an end-to-end product
throughput win, a memory reduction, an ISA promotion, retained replay/PM4
admission, physical EP2/EP4 validation, or native MTP speedup. The HC marker
count reduction and exact correctness results are evidence for this scoped
candidate only. Further uncontended profiling and any product promotion review
remain required.
