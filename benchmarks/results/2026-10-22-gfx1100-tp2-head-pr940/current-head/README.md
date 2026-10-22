# Current-head TP2 validation for PR 940

On 2026-10-22, PR 940 was merged normally with beta `88e9f0d6f9671841e6cc73e93990b3e7adb2c5c9`, producing `cb149177464fa08ee19575b97dd0ff957109ea25`. Both parent histories are preserved. The tested code tree is `7d9d217e7edd8c4f3c630f6241bdcec522503abc`, with no tracked source changes when the binaries were frozen. Later evidence-only commits do not change that tested implementation.

## GPU results

| Check | Head disabled | Narrow head default |
|---|---|---|
| Independent TP1 versus TP2, 32 positions | FAIL: maximum normalized error 1.029e-2 | FAIL: maximum normalized error 1.029e-2 |
| Unchanged numerical bound | <3e-3 | <3e-3 |
| Argmax agreement with TP1 | 32/32 | 32/32 |
| Repeated prefill oracle | 3/3 bit-exact | 3/3 bit-exact |
| Product serve battery | 5/5 stop, no empty/attractor/runaway/stream error | 5/5 stop, no empty/attractor/runaway/stream error |

Head off/default complete logits are byte-identical: 32 by 248320 FP32 values, 31,784,960 bytes, MD5 `532b0a5a419e5c7567dc0f9850d0bc2e`, SHA256 `a93e5301c834a8dcbf8c6f12d43e0415d5b25a5cd91775a05a5e941a6ab494ac`. The default arm also checks the sharded head against an unsliced reference on the actual TP state. These checks establish no head-induced difference on this fixture; they do not make the independent TP1/TP2 gate pass.

The five battery assistant texts and request MD5s are equal across arms. The decoded code, arithmetic, seasons explanation, story and coding guidance were read and are on-task. Both arms explicitly use Q8 VMM, TP2, speculation off, greedy, thinking off, max_seq 16384 and max_tokens 2048. Numerical examples instead explicitly construct legacy Q8 KV and FP32 recurrent state, and exercise whole-token graph replay; this is not a product VMM numerical admission.

## Maintainer decision requested

The independent numerical gate remains **failed and unresolved**. The earlier clean-beta `66db8cb5` diagnostic reproduced the failure without either proposed feature; disabling the existing residual LDS-staged tier reduced the old 32-position worst error to 1.636e-3. That intervention was diagnostic only: this PR does not change the residual kernel, its default, or the 3e-3 gate. Current head off/default both retain the same normal-path failure. A clean current-beta oracle was built separately, but its GPU execution is still pending, so the older clean-beta result is not relabelled as a fresh clean-88e9 result.

Please decide whether the independent TP1/TP2 discrepancy must be resolved before this feature can advance, or should be tracked separately on the strength of the exact candidate-off/on fixture. The author does not make that disposition; this PR remains Draft and the independent numerical gate is not checked as passed.

## Throughput status

The planned current-head fresh-process ABBAAB throughput matrix did not produce a measured sample. Two attempts were refused before model load or timing because GPU1 was reserved by an independent task. Both raw failures are preserved in `lock-refusal-decode-1-base/` and `decode-1-base/`; neither is treated as a numerical failure or a throughput sample. No lock file was removed and no foreign process was terminated. The historical 52.8 to 53.6 tok/s result remains a measurement of the original beta `66db8cb5` experiment, not of this merge.

## CPU and build checks

Product CLI/daemon and the numerical example were rebuilt on the merged source and verified before freezing. `scripts/pre-commit-ci.sh origin/beta` passed, including workspace clippy, formatting, maps, ratchets, supply-chain, env documentation and lifecycle checks. Head selector/layout/topology tests passed 3/3.

`railgun-cert` recording tests passed 2/3 and failed the full-tree inventory scan. The exact same six missing MQ6 predicate rows reproduce on clean beta `88e9f0d6f`: `gemm_mq6g256v2_fp8_applies`, `gemm_mq6g256v2_fp8_prepared`, `mq6_fp8_prepare_x`, `gemm_mq6g256v2_i8_applies`, `gemm_mq6g256v2_i8_prequant`, and `mq6_i8_prepare_x`. The two PR-owned TP head guards remain registered. This is not a full recording-test pass, and no unrelated MQ6 classification is invented here. Full current-head workspace tests/no-GPU CI are not claimed complete.

## Identity and raw files

Hardware: W7900 and W7900 Dual Slot, both gfx1100, PCI `0000:53:00.0` and `0000:9e:00.0`; HIP `7.15.26333-0000000`. Target: Qwen3.8-27B MQ4XTS, 14,987,185,152 bytes, MD5 `6702fe8a56c0c8ec85aca1804b8df29e`, SHA256 `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae` (SHA256 rechecked on 2026-10-22).

CLI MD5 `227a064cec4eb925dfc7d04e59c08d66`; daemon MD5 `1bc8845b120a84d40e005f805e393d01`; oracle MD5 `e9e76da6a27bf2b2f8f5fe8639305c9c`. Full SHA256 identities and source tree are in `binary-source.json`. The separately built clean-beta oracle identity is in `clean-beta-oracle-source.json`.

`summary.json` preserves per-arm failures and battery outcomes. Each run directory contains its original identity, full native JSON where available, compressed stdout/stderr/serve logs, and original 100-ms VRAM samples. The oracle's inline prompt is preserved in its committed source; command lines and graph selectors are retained in the run identities. `cpu/` contains gzip-compressed build and test logs, including the clean-beta inventory failure. `manifest.json` covers all published payloads except itself by byte count and SHA256. Weights, executable binaries and raw F32 tensors are not committed.
