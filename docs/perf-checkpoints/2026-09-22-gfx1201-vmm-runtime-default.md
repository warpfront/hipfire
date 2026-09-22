# gfx1201 VMM runtime default — 2026-09-22

**Disposition:** measured route/default restoration; not a kernel speedup, admission-policy promotion, or cross-architecture certification. Source: branch `vmm-default`, based on `kaden/mq4-lloyd` (`34ccae74b`), measured before the evidence commit with source changes in this branch. No HIP kernels changed.

## Identity and protocol

- Card-B only: AMD Radeon AI PRO R9700, exact `gfx1201`, UUID `GPU-e475645fe0200397`; HIP/ROCm `7.15.26333-0000000`, kernel/driver `7.0.0-31-generic`. No gfx1100/gfx1151 certification follows from this measurement.
- Exact model `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`, 14,987,185,152 bytes, MD5 `2cfe88923b3671ca16a8de6ec1122fde`, SHA-256 `de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5`; Qwen3.8 27B MQ4v2 QAT. Vision sidecar skipped (`vision_mode=off`). No draft sidecar or speculative decoding.
- CLI `/home/kaden/ClaudeCode/warpfront/wt-vmmdefault/target/release/hipfire` MD5 `125231ec864f219a9a7758016b83430c`; **final matched matrix** daemon `target/release/daemon` MD5 `bc16355db9dd8370cc64d75acccdbc36`. Earlier HTTP battery and local tag probe used daemon MD5 `dc56b8bae3bed030fdbc937db25bfc5c`; the subsequent source change added conservative non-gfx1201/Windows certification gating, then both matrix arms were remeasured on the final binary.
- Matrix prompt: 8,192 little-endian `u32` token IDs `10 + (i % 1000)` for `i=0..8191`, byte MD5 `c596ea04a31b9977adeca9eda88f622c`. Matrix `--pp 512,8192 --ctx 128 --tg 128 --spec off --kv-mode fp8 --json`; each process `--runs 3 --warmups 1`. An unrecorded fresh-process `--runs 1` warmup preceded each arm. One process is one sample; the three within-process measurements are summarized by their median.
- `HOME=/home/kaden/.hipfire-homes/vmmdefault`, `ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397`, `HIP_VISIBLE_DEVICES=GPU-e475645fe0200397`, `HIPFIRE_KERNEL_CACHE=$HOME/.hipfire_kernels`, `HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models`, `HIPFIRE_GRAPH=1`, `HIPFIRE_DAEMON_BIN=/home/kaden/ClaudeCode/warpfront/wt-vmmdefault/target/release/daemon`. Stale daemon PID removed. No explicit backend option in automatic arm; comparator added **only** `--kv-backend contiguous`.

## Fresh-process matrix

| Arm, process | pp8192 resident samples (tok/s) | Process median | tg128@128 median (tok/s) |
|---|---|---:|---:|
| automatic VMM 1 | 3649.8, 3648.2, 3644.1 | 3648.2 | 36.50 |
| automatic VMM 2 | 3643.5, 3642.9, 3642.0 | 3642.9 | 36.51 |
| automatic VMM 3 | 3640.2, 3642.5, 3643.1 | 3642.5 | 36.52 |
| explicit contiguous 1 | 3576.3, 3578.4, 3577.8 | 3577.8 | 36.54 |
| explicit contiguous 2 | 3572.5, 3573.4, 3574.2 | 3573.4 | 36.54 |
| explicit contiguous 3 | 3573.6, 3574.5, 3575.0 | 3574.5 | 36.52 |

All automatic process medians exceed the 3,620 tok/s guard and all automatic decode medians are within 1% of 36.5 tok/s. The median-of-process-medians pp8192 observation is 3,642.9 versus 3,574.5 tok/s (about 1.9%); order was three VMM then three contiguous, **not** ABBA. This is the observed cost of accidental contiguous selection for this fixture, not a promoted A/B gain over already-VMM loading. Raw commands/JSON/daemon logs: workstation-local `scratch-vmmdefault/guard-certified/` and `scratch-vmmdefault/guard-contiguous-certified/` (uncommitted). The loaded diagnostic reports `kv_backend_request=automatic, kv_backend=vmm` in the first arm, and explicit contiguous in the second. Logs prove actual storage: `KV cache: Fp8 vmm (16/64 layers carry KV; K/V 1032B/row; mapped_prefix=2032 / physical_cap=262144 / max_seq=262144)` versus `KV backend: explicit contiguous override` and `KV cache: fp8-e4m3 (`. Reported free VRAM after VMM load was 16,838 MiB of 32,624 MiB; reservation capacity was 262,144 positions, with only the prefix mapped at load. No loaded-VRAM advantage claim is made from this one diagnostic.

An isolated local catalog alias `vmmdefault:27b` pointed to the **same model bytes** and a fresh tag invocation without `--kv-backend` reported `kv_backend_request=automatic, kv_backend=vmm`, `KV cache: Fp8 vmm`, pp8192 3,674.6 tok/s and tg128@128 36.56 tok/s (single exploratory process, not included in the three-process floor). Its independent local-alias default context was 32,768; the path policy used 262,144. This verifies backend selection parity, not context-policy parity.

## Correctness and scope limits

`hip-bridge` `vmm_arena_smoke` passed boundary growth, over-reserve failure, cleanup after failure, owner teardown, and unload/reload on card-B. Native HTTP serve with the exact model and fp8 showed 5/5 coherent, nonempty code, arithmetic, factual, prose, and instruction answers, no runaway/attractor/retrieval failures. Committed prompt fixture `scripts/fixtures/vmmdefault_serve_prompts.json` byte MD5 `04808fefe996cba11891eb65eea1ef1c`. Separate automatic-VMM and explicit-contiguous services used identical greedy requests with seed 42; all five full decoded transcripts (answer and reasoning) and request MD5s were byte-identical, transcript MD5 `98cd8d89d3213491a06c7c88489a14a7`. Workstation-local JSON/logs: `scratch-vmmdefault/serve/battery-seeded.json`, `scratch-vmmdefault/serve-contiguous/battery-seeded.json`, respective `serve-seeded.log` and `serve.log`. Both emitted the corresponding actual backend allocation marker. The initial stochastic/unseeded attempt diverged as expected, so it is not parity evidence.

No exact logit/KV tensor oracle was run; this is output-level parity only. `HIPFIRE_GRAPH=1` was set and prefill/decode paths ran, but no separate graph-capture/replay trace or native fp8/bf16 adaptive oracle was recorded. No DFlash/MTP/Redline replay was enabled in this fixture. Other architectures, OSes, topologies and modes remain gated by read-only admission with a reasoned contiguous fallback or explicit VMM refusal, not extrapolated from card-B.
