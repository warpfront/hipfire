# Qwen4 shared token-batched prefill — final scoped measurement — 2026-09-18

**Lifecycle:** `historical`

**Disposition:** six fresh product-CLI observations and one serve smoke after the
HC source correction. This is fixture-bound evidence only: not a product
baseline, speedup claim, admission decision, or `docs/BENCHMARKS.md` claim.
The durable machine-readable summary is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/bench-summary.json`.

## Scope and identity

- Host GPU: AMD Strix Halo, `gfx1151`, HIP `7.2`; one visible device
  (`HIP_VISIBLE_DEVICES=0`). GPU contention is **unknown/contended**: the
  cooperative `/tmp/hipfire-gpu.lock` was used, but unrelated GPU clients were
  observed on the host. The lock is not hardware isolation.
- Model: `/home/bjoern/.hipfire/models/qwen3.8-flash-next.hfq`; SHA256
  `7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`;
  MD5 `fda74d3760dc803e778e9b30a2fe0ebd`.
- Product binaries: `target/release/daemon` SHA256
  `bb346923ed1d974282d2b0068a6df16bcac6230499800f89ea9ed931671de53f`;
  `target/release/hipfire` SHA256
  `4e9211dc6f28fb6a671e045034a645c2619f493d2521223ed76cc946ebb633de`.
- Source lineage: semantic source SHA256
  `a553fe861610e79776c9b9856fcd92329e5200a887cb798c99d76a8a178847b8`;
  final formatter-only source SHA256
  `f1f37bfd5f19aea3301f5de7ff455e710097b66197e6436b67f3cf4cf13377b3`.
- Kernel cache: shared gfx1151 cache at
  `.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/kernel-cache/gfx1151`;
  342 `.hash` files, listing SHA256
  `a8f78492780c2fdf960b96451f0142b16b21318128fb1cfe042a863f489da20b`,
  content SHA256
  `a03ddd17a2b356569a8a9caac58c1ed18dd1e1eeed2b3a45ed7fceefab58e863`.
- **Memory boundary: STATIC only.** This record preserves static fixture,
  binary, source, cache, and raw-output identity. It does not measure or claim
  allocator behavior, dynamic memory footprint, residency, bandwidth, or a
  memory optimization.

The six product samples used fresh private `HOME`/`HIPFIRE_HOME` directories,
`HIPFIRE_DPM_WARMUP_SECS=10`, q8 contiguous KV, speculation off, noslots,
stateless workload, `--max-tokens 16`, ten warmups, and one measured run:

```text
target/release/hipfire bench <model> --prompt-file <prompt> --runs 1 \
  --max-tokens 16 --warmups 10 --kv-mode q8 --kv-backend contiguous \
  --spec off --backend noslots --workload stateless --json
```

The compiler was enabled (`HIPFIRE_NO_DEVICE_COMPILER` unset). The exact
per-sample environment and command are retained in each driver JSON.

## Fresh product observations

Prompt `benchmarks/prompts/bare_factual.txt` is 42 tokens, MD5
`1d32df5f12c414d3e34c7b35b6611e6c`, SHA256
`25c0b9185242eb3aaff3e8be6b471fe4e54a06bd11e7800f5b30bcc2a0bc64ba`.
At 42 tokens the CLI explicitly warns that `prefill_tok_s` measures launch
overhead, not prefill throughput.

| sample | prefill tok/s | decode tok/s | TTFT ms | wall tok/s | raw JSON (SHA256) |
|---|---:|---:|---:|---:|---|
| short-1 | 17.8 | 8.4 | 2359.8 | 3.8 | [`short-1.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/short-1.json) (`3aec9623bc95f8c7c1a08ebfc5ddadb68b769883fd749abf1dda0d6476e59af3`)|
| short-2 | 18.5 | 8.4 | 2273.3 | 3.8 | [`short-2.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/short-2.json) (`99120075df70da37e517a48341d222f371833210c62a5afe0cc8f7f0a74b8664`)|
| short-3 | 18.9 | 8.7 | 2221.3 | 3.9 | [`short-3.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/short-3.json) (`9223166351036170967eeb1399091215edc4d28bfdcfa590e5f0b26b125de7d5`)|
| **median (42 tokens)** | **18.5** | **8.4** | **2273.3** | **3.8** | launch-overhead observation only |

Prompt `benchmarks/prompts/glimmer_prefill_256.txt` is 291 tokens, MD5
`973900074bfd15d4adeeecdff3359082`, SHA256
`184b6e9ab08b3d07f9cebf2ce9506cfe34d8d2ffc03c910b105e13bbd6ac6a8e`.
This is the only current product fixture in this record suitable for a
prefill-throughput observation.

| sample | prefill tok/s | decode tok/s | TTFT ms | wall tok/s | raw JSON (SHA256) |
|---|---:|---:|---:|---:|---|
| ctx291-1 | 5.5 | 1.8 | 52562.2 | 0.3 | [`ctx291-1.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/ctx291-1.json) (`dad4dedd5b88c48994cf5517f484bb6197828303cf940ef97def2b8060736b86`)|
| ctx291-2 | 5.5 | 1.6 | 53160.3 | 0.3 | [`ctx291-2.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/ctx291-2.json) (`711675f32c496bb7ae39c1b76509370f9f9ab759575125a740b6e64d1105e30c`)|
| ctx291-3 | 5.2 | 1.8 | 56166.4 | 0.2 | [`ctx291-3.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/ctx291-3.json) (`1b8c188fbfe6b14da93fd634a35e597d795670e8f9c3091bfcb88da54a23f302`)|
| **median (291 tokens)** | **5.5** | **1.8** | **53160.3** | **0.3** | measured fixture only |

All six commands returned `0`. The six rows are observations under unknown
external contention, not an A/B, a grouped-benefit estimate, or a launch-count
measurement.

### Serve smoke

A separate `serve_harness.py --mode battery` smoke used the 291-token prompt,
q8 contiguous KV, speculation off, greedy sampling, thinking off, max sequence
2048, and max tokens 16. Driver:
[`serve-ctx291-1.driver.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/serve-ctx291-1.driver.json)
(SHA256 `797d85e8481bdac278f97cfbc7e72692c6e7112973c9cd4f61e629f29b30f2c5`).
The result JSON is
[`serve-ctx291-1.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/serve-ctx291-1.json)
(SHA256 `5ab52e2e940f2d1a70c151a6525170ddf7bdad3304247fb8e57317522c990545`).
The process returned `0`, but its sole row was `ctx=0`, `gen=0`,
`finish=null`, `empty=true`, `saw_done=false` (`ttft=98.125 s`,
`wall=110.322 s`). It emitted no completed visible generation, so this smoke
is not folded into the six-sample medians and is not a product continuation
claim.

## Profile disposition

The requested rocprof route is **blocked** at runtime startup/compiler probe.
There are no current kernel-count, launch-count, grouped-benefit, ISA, or
attribution claims.

- The first daemon-wrapper attempt failed because the wrapped daemon could not
  load `libatomic.so.1`; its raw driver is
  `.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/profile-ctx291/driver.json`.
- The retry added the GCC `libatomic` directory and ran four serve retries for
  732.38 s, but produced no warmed result or CSV. Its raw driver is
  [`profile-ctx291-retry/driver.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/profile-ctx291-retry/driver.json)
  (SHA256 `1825c45facb82477f192562d1b22714e1c63b78032c38563e0ae6c87de7eea01`).
- A corrected merged-toolkit direct oracle profile used the exact qwen4
  greedy16 oracle, the 291-token manifest, `--max-chunk 128`, and
  `--greedy-continuation 16`; it stalled for 1491.35 s and was stopped after
  no result or CSV appeared. Driver:
  [`profile-direct-oracle/driver.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/profile-direct-oracle/driver.json)
  (SHA256 `04b85d888dd2d71c59b76b399167da0f5d0d4f08c2194045eb63c8ee08037f22`).
- The final tiny rocprof smoke wrapped `target/release/bf16-multirow-probe
  --help` with the merged toolkit and an evidence compiler wrapper. It timed
  out at 60 s during rocprof startup/compiler probing and emitted no CSV:
  [`profile-smoke-wrapper/driver.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/profile-smoke-wrapper/driver.json)
  (SHA256 `50b262344c78cf198654c801aca77d7a8ec1dd215b797a7b81b335a049b525de`).
- The internal fallback (`HIPFIRE_PROFILE=1`,
  `HIPFIRE_PROFILE_DECODE=1`) completed a normal short bench, but emitted no
  runtime profile markers. Driver:
  [`internal-profile-short/driver.json`](../../.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/phase6-final-clean/internal-profile-short/driver.json)
  (SHA256 `660ea4bdcec2641f58964c0530b17763df15d3c28291f77a08967a658d831684`).
  It is retained as a fallback-route observation, not a replacement for
  rocprof attribution.

The working merged-toolkit environment used for the corrected attempts was:
`HIPFIRE_ROCM_PATH=ROCM_PATH=/home/bjoern/.hipfire/rocm-merged`,
`HIP_PATH=/nix/store/lqklrnx2bc9k765jyxc0d8q6h15wlybb-clr-7.2.3`,
`HSA_PATH=/nix/store/gcd42p421bwgrdzraczkg8x6wvvpwbsk-rocm-runtime-7.2.3`,
`HIPFIRE_HIPCC=/home/bjoern/.hipfire/rocm-merged/bin/hipcc`, and
`LD_LIBRARY_PATH=/home/bjoern/.hipfire/rocm-merged/lib:/nix/store/lqklrnx2bc9k765jyxc0d8q6h15wlybb-clr-7.2.3/lib:/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib`,
plus `HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=/nix/store/bb0c3xjc5cn4f8hgbbgwcp03wa9d4gxw-rocm-device-libs-22.0.0-rocm/amdgcn/bitcode`.
No more GPU/profile runs were made after the tiny probe.

## Historical comparison and explicit limits

The old source-matched product record
`.codeinsight+research/qwen4/shared-token-prefill/baseline-source-matched/runs/result.json`
reported the same nominal 291-token / 16-token shape at approximately
`4.1 tok/s` prefill and `1.8 tok/s` decode (`71393.4 ms` prefill). It is
**not an output-equivalent baseline**: its old max-128 `forward_chunk` path
was guarded at layer 1 by `ple.lease.is_none()`, so row 0 acquired/applied
PLE while later rows skipped it; the first divergence was row 2 and state
divergence was `gdn[1].recurrent`. Therefore these current observations must
not be converted into a speedup claim against that old PLE-invalid result.

This record does not claim: a grouped-benefit percentage; launch counts;
rocprof kernel/ISA attribution; an ISA-fit conclusion; a dynamic-memory
result; uncontended performance; dense retained replay or PM4 admission;
physical EP2/EP4 validation; cross-GPU/quant transfer; product defaults; or
promotion.  Semantic correctness evidence is recorded in the
[`implementation progress checkpoint`](../design/qwen4-shared-token-batched-prefill-progress-20260918.md),
including the direct scalar-vs-natural cap-128 greedy-16 proof.  This file is
the final measured-benchmark/profile boundary only.
